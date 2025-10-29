"""Phase-stratified sampling utilities for balanced dataset creation."""

import pandas as pd
import numpy as np
import chess
import random
from typing import Tuple, Optional, Dict, List
from enum import Enum


class GamePhase(Enum):
    """Game phase enumeration."""
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


def stratify_by_phase(
    df: pd.DataFrame,
    target_size: int,
    phase_distribution: Optional[Dict[str, float]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample positions with stratification by game phase.

    Args:
        df: Positions DataFrame with 'phase' column
        target_size: Desired total number of positions
        phase_distribution: Optional dict mapping phase -> fraction
                           Default: {"opening": 0.25, "middlegame": 0.50, "endgame": 0.25}
        random_state: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    if phase_distribution is None:
        phase_distribution = {
            "opening": 0.25,
            "middlegame": 0.50,
            "endgame": 0.25,
        }

    # Validate distribution
    assert abs(sum(phase_distribution.values()) - 1.0) < 1e-6, "Phase fractions must sum to 1.0"

    sampled_dfs = []

    for phase, fraction in phase_distribution.items():
        phase_df = df[df["phase"] == phase]
        n_samples = int(target_size * fraction)

        # Sample with replacement if needed
        replace = len(phase_df) < n_samples

        if replace:
            print(f"Warning: {phase} has only {len(phase_df)} positions, sampling with replacement")

        sampled = phase_df.sample(n=n_samples, replace=replace, random_state=random_state)
        sampled_dfs.append(sampled)

    result = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"\nStratified sampling summary:")
    print(f"  Target size: {target_size:,}")
    print(f"  Actual size: {len(result):,}")
    print(f"  Phase distribution:")
    for phase in ["opening", "middlegame", "endgame"]:
        count = (result["phase"] == phase).sum()
        pct = count / len(result) * 100
        print(f"    {phase:12s}: {count:6,} ({pct:5.1f}%)")

    return result


def balance_dataset(
    df: pd.DataFrame,
    balance_by: str = "outcome",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance dataset by specified column.

    Args:
        df: Input DataFrame
        balance_by: Column name to balance by (e.g., 'outcome', 'phase')
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    min_count = df[balance_by].value_counts().min()

    balanced_dfs = []
    for value in df[balance_by].unique():
        subset = df[df[balance_by] == value]
        sampled = subset.sample(n=min_count, replace=False, random_state=random_state)
        balanced_dfs.append(sampled)

    result = pd.concat(balanced_dfs, ignore_index=True)
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"\nBalanced by '{balance_by}':")
    print(f"  Original size: {len(df):,}")
    print(f"  Balanced size: {len(result):,}")
    print(f"  Distribution:")
    print(result[balance_by].value_counts())

    return result


def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify_by: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        df: Input DataFrame
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        stratify_by: Optional column to stratify by
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if stratify_by is not None:
        # Stratified split
        train_dfs, val_dfs, test_dfs = [], [], []

        for value in df[stratify_by].unique():
            subset = df[df[stratify_by] == value]
            n = len(subset)

            train_end = int(n * train_frac)
            val_end = int(n * (train_frac + val_frac))

            train_dfs.append(subset.iloc[:train_end])
            val_dfs.append(subset.iloc[train_end:val_end])
            test_dfs.append(subset.iloc[val_end:])

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Shuffle splits
        train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        val_df = val_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    else:
        # Simple split
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_df):6,} ({len(train_df)/len(df)*100:5.1f}%)")
    print(f"  Val:   {len(val_df):6,} ({len(val_df)/len(df)*100:5.1f}%)")
    print(f"  Test:  {len(test_df):6,} ({len(test_df)/len(df)*100:5.1f}%)")

    return train_df, val_df, test_df


def sample_by_move_number(
    df: pd.DataFrame,
    move_ranges: Dict[str, Tuple[int, int]],
    samples_per_range: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample positions by move number ranges.

    Args:
        df: Positions DataFrame
        move_ranges: Dict mapping range name to (min_move, max_move) tuple
                     e.g., {"opening": (1, 15), "middle": (16, 40), "late": (41, 100)}
        samples_per_range: Number of samples per range
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    sampled_dfs = []

    for range_name, (min_move, max_move) in move_ranges.items():
        range_df = df[(df["move_number"] >= min_move) & (df["move_number"] <= max_move)]

        if len(range_df) == 0:
            print(f"Warning: No positions in range {range_name} ({min_move}-{max_move})")
            continue

        replace = len(range_df) < samples_per_range
        n_samples = min(samples_per_range, len(range_df)) if not replace else samples_per_range

        sampled = range_df.sample(n=n_samples, replace=replace, random_state=random_state)
        sampled_dfs.append(sampled)

        print(f"  {range_name:12s} ({min_move:3d}-{max_move:3d}): {n_samples:6,} samples")

    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return result


def phase_stratified_sample(
    positions: List[dict],
    n_samples: int,
    phase_weights: dict = None,
) -> List[dict]:
    """
    Sample positions stratified by game phase.

    Phase determined by piece count:
    - Opening: 28-32 pieces
    - Middlegame: 15-27 pieces
    - Endgame: 6-14 pieces

    Args:
        positions: List of position dicts with 'fen' key
        n_samples: Total number of samples
        phase_weights: Dict with 'opening', 'middlegame', 'endgame' weights

    Returns:
        Stratified sample of positions
    """
    if phase_weights is None:
        phase_weights = {'opening': 0.25, 'middlegame': 0.50, 'endgame': 0.25}

    # Categorize by phase
    phases = {'opening': [], 'middlegame': [], 'endgame': []}

    for pos in positions:
        board = chess.Board(pos['fen'])
        piece_count = len(board.piece_map())

        if piece_count >= 28:
            phases['opening'].append(pos)
        elif piece_count >= 15:
            phases['middlegame'].append(pos)
        else:
            phases['endgame'].append(pos)

    # Sample from each phase
    samples = []
    for phase, weight in phase_weights.items():
        n_phase = int(n_samples * weight)
        if len(phases[phase]) > 0:
            phase_samples = random.sample(
                phases[phase],
                min(n_phase, len(phases[phase]))
            )
            samples.extend(phase_samples)

    return samples


def mirror_board(board: chess.Board) -> chess.Board:
    """Mirror board horizontally (file flip)."""
    mirrored = chess.Board(None)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            mirror_square = chess.square_mirror(square)
            # Flip file: a->h, b->g, etc.
            file_flipped = chess.square(7 - chess.square_file(mirror_square), chess.square_rank(mirror_square))
            mirrored.set_piece_at(file_flipped, piece)

    mirrored.turn = board.turn
    mirrored.castling_rights = board.castling_rights
    mirrored.ep_square = board.ep_square
    mirrored.halfmove_clock = board.halfmove_clock
    mirrored.fullmove_number = board.fullmove_number

    return mirrored


# ============================================================================
# Example Usage
# ============================================================================

def _demo_sampling():
    """Demonstrate sampling utilities."""
    print("Demo: Sampling utilities")

    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 10000

    df = pd.DataFrame({
        "fen": [f"fen_{i}" for i in range(n_samples)],
        "move": [f"e2e4" for _ in range(n_samples)],
        "outcome": np.random.choice([0.0, 0.5, 1.0], size=n_samples, p=[0.4, 0.2, 0.4]),
        "phase": np.random.choice(["opening", "middlegame", "endgame"], size=n_samples, p=[0.3, 0.5, 0.2]),
        "move_number": np.random.randint(1, 100, size=n_samples),
    })

    print(f"Original dataset: {len(df):,} positions\n")

    # Stratify by phase
    stratified = stratify_by_phase(df, target_size=5000)

    # Balance by outcome
    balanced = balance_dataset(df, balance_by="outcome")

    # Train/val/test split
    train, val, test = train_val_test_split(df, stratify_by="phase")


if __name__ == "__main__":
    _demo_sampling()
