"""PyTorch Dataset for loading sharded chess position data.

Supports:
- Loading from multiple .pt shard files
- Phase-balanced sampling
- On-the-fly augmentation (file/rank flips)
- Efficient memory usage with lazy loading
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import random


class ShardedChessDataset(Dataset):
    """Dataset that loads chess positions from sharded .pt files."""

    def __init__(
        self,
        shard_dir: Path,
        phase_balanced: bool = True,
        augment_file_flip: bool = False,
        augment_rank_flip: bool = False,
        load_all_shards: bool = False,
        seed: int = 42,
    ):
        """
        Initialize sharded dataset.

        Args:
            shard_dir: Directory containing shard_*.pt files
            phase_balanced: Use phase-balanced sampling
            augment_file_flip: Apply random file flips
            augment_rank_flip: Apply random rank flips
            load_all_shards: Load all shards into memory (vs lazy loading)
            seed: Random seed
        """
        self.shard_dir = Path(shard_dir)
        self.phase_balanced = phase_balanced
        self.augment_file_flip = augment_file_flip
        self.augment_rank_flip = augment_rank_flip
        self.seed = seed

        # Find all shard files
        self.shard_paths = sorted(self.shard_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise ValueError(f"No shards found in {shard_dir}")

        print(f"Found {len(self.shard_paths)} shards in {shard_dir}")

        # Load shard metadata
        self.shard_sizes = []
        self.shard_phase_counts = []

        for shard_path in self.shard_paths:
            shard_data = torch.load(shard_path)
            self.shard_sizes.append(len(shard_data['boards']))

            # Count phases in this shard
            phases = shard_data['phases']
            phase_counts = {
                0: (phases == 0).sum().item(),
                1: (phases == 1).sum().item(),
                2: (phases == 2).sum().item(),
            }
            self.shard_phase_counts.append(phase_counts)

        self.total_positions = sum(self.shard_sizes)
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)

        print(f"Total positions: {self.total_positions:,}")

        # Optionally load all shards into memory
        self.loaded_shards = {}
        if load_all_shards:
            print("Loading all shards into memory...")
            for idx, shard_path in enumerate(self.shard_paths):
                self.loaded_shards[idx] = torch.load(shard_path)
            print("✓ All shards loaded")

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self) -> int:
        """Return total number of positions."""
        return self.total_positions

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single position.

        Args:
            idx: Position index

        Returns:
            Tuple of (board_tensor, move_index, value)
        """
        # Find which shard contains this index
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[shard_idx]

        # Load shard data (from memory or disk)
        if shard_idx in self.loaded_shards:
            shard_data = self.loaded_shards[shard_idx]
        else:
            shard_data = torch.load(self.shard_paths[shard_idx])

        # Extract position
        board = shard_data['boards'][local_idx]
        move = shard_data['moves'][local_idx]
        value = shard_data['values'][local_idx]

        # Apply augmentation
        if self.augment_file_flip and random.random() < 0.5:
            board = board.flip(2)  # Flip along file dimension

        if self.augment_rank_flip and random.random() < 0.5:
            board = board.flip(1)  # Flip along rank dimension

        return board, move, value

    def get_phase_weights(self) -> torch.Tensor:
        """
        Compute sampling weights for phase-balanced sampling.

        Returns:
            Tensor of weights for each position
        """
        # Count total positions per phase
        total_phase_counts = {0: 0, 1: 0, 2: 0}
        for phase_counts in self.shard_phase_counts:
            for phase, count in phase_counts.items():
                total_phase_counts[phase] += count

        # Compute inverse frequency weights
        phase_weights = {}
        for phase, count in total_phase_counts.items():
            if count > 0:
                phase_weights[phase] = 1.0 / count
            else:
                phase_weights[phase] = 0.0

        # Assign weight to each position based on its phase
        weights = []
        for shard_idx, shard_path in enumerate(self.shard_paths):
            if shard_idx in self.loaded_shards:
                shard_data = self.loaded_shards[shard_idx]
            else:
                shard_data = torch.load(shard_path)

            phases = shard_data['phases']
            for phase in phases:
                weights.append(phase_weights[phase.item()])

        return torch.tensor(weights, dtype=torch.float32)


def create_shard_dataloaders(
    train_shard_dir: Path,
    val_shard_dir: Path,
    batch_size: int = 256,
    num_workers: int = 0,
    phase_balanced: bool = True,
    augment_train: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from sharded data.

    Args:
        train_shard_dir: Directory containing training shards
        val_shard_dir: Directory containing validation shards
        batch_size: Batch size
        num_workers: Number of worker processes
        phase_balanced: Use phase-balanced sampling
        augment_train: Apply augmentation to training data
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("=" * 70)
    print("CREATING SHARD-BASED DATALOADERS")
    print("=" * 70)

    # Training dataset
    train_dataset = ShardedChessDataset(
        shard_dir=train_shard_dir,
        phase_balanced=phase_balanced,
        augment_file_flip=augment_train,
        augment_rank_flip=False,  # Keep False to avoid breaking move mapping
        load_all_shards=False,
    )

    # Validation dataset
    val_dataset = ShardedChessDataset(
        shard_dir=val_shard_dir,
        phase_balanced=False,  # No balancing for validation
        augment_file_flip=False,
        augment_rank_flip=False,
        load_all_shards=False,
    )

    # Create samplers
    if phase_balanced:
        print("Creating phase-balanced sampler...")
        weights = train_dataset.get_phase_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = None
    else:
        sampler = None
        train_shuffle = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=train_shuffle if sampler is None else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Phase-balanced sampling: {phase_balanced}")
    print(f"Augmentation: {augment_train}")
    print("=" * 70)

    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Test the dataset
    shard_dir = Path("artifacts/data/shards")

    if shard_dir.exists():
        dataset = ShardedChessDataset(shard_dir)
        print(f"\nDataset size: {len(dataset):,}")

        # Test getting a sample
        board, move, value = dataset[0]
        print(f"Board shape: {board.shape}")
        print(f"Move index: {move}")
        print(f"Value: {value}")

        print("\n✓ Shard dataset test passed!")
    else:
        print(f"Shard directory not found: {shard_dir}")
        print("Run the stream sampler first to create shards")
