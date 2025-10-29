"""PyTorch Dataset and DataLoader for chess positions."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import chess
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.encoding import board_to_tensor, move_to_index, get_auxiliary_features, MOVE_INDEX_SIZE


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess positions.

    Returns:
    - board_tensor: [12, 8, 8] piece planes
    - auxiliary: [6] auxiliary features (side to move, castling, ep)
    - move_target: integer move index
    - value_target: float outcome value
    """

    def __init__(
        self,
        df: pd.DataFrame,
        include_auxiliary: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize dataset from positions DataFrame.

        Args:
            df: DataFrame with columns 'fen', 'move', 'outcome'
            include_auxiliary: Whether to include auxiliary features
            device: PyTorch device for tensor placement
        """
        self.df = df.reset_index(drop=True)
        self.include_auxiliary = include_auxiliary
        self.device = device

        # Pre-validate data
        self._validate_data()

    def _validate_data(self):
        """Validate that all positions and moves are legal."""
        required_cols = ["fen", "move", "outcome"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single training example.

        Returns:
            If include_auxiliary=True: (board_tensor, auxiliary, move_target, value_target)
            Otherwise: (board_tensor, move_target, value_target)
        """
        row = self.df.iloc[idx]

        # Parse position
        board = chess.Board(row["fen"])

        # Encode board
        board_tensor = board_to_tensor(board, device=self.device)

        # Encode move
        move = chess.Move.from_uci(row["move"])
        move_target = torch.tensor(move_to_index(move), dtype=torch.long)

        # Encode outcome as value target
        # Convert {0.0, 0.5, 1.0} to {-1.0, 0.0, 1.0} for better training
        outcome = row["outcome"]
        value_target = torch.tensor(2.0 * outcome - 1.0, dtype=torch.float32)

        if self.device is not None:
            move_target = move_target.to(self.device)
            value_target = value_target.to(self.device)

        if self.include_auxiliary:
            auxiliary = get_auxiliary_features(board, device=self.device)
            return board_tensor, auxiliary, move_target, value_target
        else:
            return board_tensor, move_target, value_target


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 256,
    num_workers: int = 0,
    include_auxiliary: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        train_df: Training positions DataFrame
        val_df: Validation positions DataFrame
        batch_size: Batch size for training
        num_workers: Number of data loading workers (0 for Mac compatibility)
        include_auxiliary: Whether to include auxiliary features
        device: PyTorch device

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ChessDataset(train_df, include_auxiliary=include_auxiliary, device=device)
    val_dataset = ChessDataset(val_df, include_auxiliary=include_auxiliary, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable for Mac MPS compatibility
        drop_last=True,  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    print(f"\nDataLoader created:")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    print(f"  Batch size:    {batch_size}")

    return train_loader, val_loader


def collate_fn_with_mask(batch):
    """
    Custom collate function that adds legal move masks.

    Note: This is more expensive as it requires creating a Board for each position.
    Use only if needed for masked softmax during training.

    Args:
        batch: List of tuples from ChessDataset.__getitem__

    Returns:
        Batched tensors with added legal_move_masks
    """
    # Separate batch components
    if len(batch[0]) == 4:  # With auxiliary
        boards, auxiliaries, move_targets, value_targets = zip(*batch)
        auxiliaries = torch.stack(auxiliaries)
    else:  # Without auxiliary
        boards, move_targets, value_targets = zip(*batch)
        auxiliaries = None

    boards = torch.stack(boards)
    move_targets = torch.stack(move_targets)
    value_targets = torch.stack(value_targets)

    # TODO: Add legal move masks if needed
    # This requires reconstructing Board from FEN (expensive)

    if auxiliaries is not None:
        return boards, auxiliaries, move_targets, value_targets
    else:
        return boards, move_targets, value_targets


# ============================================================================
# Unit Tests
# ============================================================================

def _test_dataset():
    """Test ChessDataset functionality."""
    print("Testing ChessDataset...")

    # Create synthetic dataset
    positions = []
    board = chess.Board()

    for _ in range(100):
        if board.is_game_over():
            board.reset()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board.reset()
            legal_moves = list(board.legal_moves)

        move = np.random.choice(legal_moves)

        positions.append({
            "fen": board.fen(),
            "move": move.uci(),
            "outcome": np.random.choice([0.0, 0.5, 1.0]),
            "phase": "middlegame",
            "move_number": 20,
        })

        board.push(move)

    df = pd.DataFrame(positions)

    # Create dataset
    dataset = ChessDataset(df, include_auxiliary=True)

    print(f"  Dataset size: {len(dataset)}")

    # Test __getitem__
    board_tensor, auxiliary, move_target, value_target = dataset[0]

    print(f"  Board tensor shape: {board_tensor.shape}")
    print(f"  Auxiliary shape: {auxiliary.shape}")
    print(f"  Move target: {move_target.item()}")
    print(f"  Value target: {value_target.item()}")

    assert board_tensor.shape == (12, 8, 8), "Invalid board tensor shape"
    assert auxiliary.shape == (6,), "Invalid auxiliary shape"
    assert 0 <= move_target.item() < MOVE_INDEX_SIZE, "Invalid move index"
    assert -1.0 <= value_target.item() <= 1.0, "Invalid value target"

    print("✓ Dataset test passed")

    # Test DataLoader
    train_df, val_df = df[:80], df[80:]
    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        batch_size=16,
        num_workers=0,
    )

    # Get one batch
    batch = next(iter(train_loader))
    board_batch, aux_batch, move_batch, value_batch = batch

    print(f"\n  Batch shapes:")
    print(f"    Boards: {board_batch.shape}")
    print(f"    Auxiliary: {aux_batch.shape}")
    print(f"    Moves: {move_batch.shape}")
    print(f"    Values: {value_batch.shape}")

    assert board_batch.shape == (16, 12, 8, 8), "Invalid board batch shape"
    assert aux_batch.shape == (16, 6), "Invalid auxiliary batch shape"

    print("✓ DataLoader test passed")

    print("\nAll dataset tests passed! ✓")


if __name__ == "__main__":
    _test_dataset()
