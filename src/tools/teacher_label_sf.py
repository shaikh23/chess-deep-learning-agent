"""Teacher-value labeling using Stockfish for supervised learning.

This tool generates value labels for chess positions using Stockfish evaluation.
The centipawn scores are normalized to [-1, 1] range for neural network training.

Usage:
    python -m src.tools.teacher_label_sf \
        --input data/positions.csv \
        --output data/teacher_labels.pt \
        --stockfish-path /usr/local/bin/stockfish \
        --depth 10 \
        --batch-size 1000
"""

import argparse
import chess
import chess.engine
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import json


def centipawn_to_value(centipawns: float, scale: float = 800.0) -> float:
    """
    Convert centipawn evaluation to value in [-1, 1].

    Args:
        centipawns: Stockfish centipawn evaluation
        scale: Scaling factor (default: 800 = ~8 pawns)

    Returns:
        Value in [-1, 1] range
    """
    return np.clip(centipawns / scale, -1.0, 1.0)


def label_positions_stockfish(
    fens: List[str],
    stockfish_path: str,
    depth: int = 10,
    time_limit: float = 0.1,
    mate_value: float = 10000.0,
) -> List[float]:
    """
    Generate value labels for positions using Stockfish.

    Args:
        fens: List of FEN strings
        stockfish_path: Path to Stockfish binary
        depth: Search depth
        time_limit: Time limit per position (seconds)
        mate_value: Centipawn value for mate positions

    Returns:
        List of value labels in [-1, 1]
    """
    values = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for fen in tqdm(fens, desc="Labeling positions"):
            try:
                board = chess.Board(fen)

                # Analyze position
                info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth, time=time_limit)
                )

                # Extract score
                score = info.get("score")
                if score is None:
                    # Default to 0 if no score available
                    cp_value = 0.0
                elif score.is_mate():
                    # Mate score
                    mate_moves = score.relative.moves
                    if mate_moves > 0:
                        cp_value = mate_value  # Winning
                    else:
                        cp_value = -mate_value  # Losing
                else:
                    # Regular centipawn score
                    cp_value = float(score.relative.score())

                # Convert to normalized value
                value = centipawn_to_value(cp_value)
                values.append(value)

            except Exception as e:
                print(f"Error analyzing {fen}: {e}")
                values.append(0.0)  # Default value on error

    return values


def create_teacher_dataset(
    input_path: str,
    output_path: str,
    stockfish_path: str,
    depth: int = 10,
    time_limit: float = 0.1,
    batch_size: int = 1000,
    max_positions: int = None,
):
    """
    Create teacher-labeled dataset from FEN list.

    Args:
        input_path: Path to input CSV/TXT with FENs
        output_path: Path to output .pt file
        stockfish_path: Path to Stockfish binary
        depth: Stockfish search depth
        time_limit: Time per position
        batch_size: Process in batches (for memory)
        max_positions: Maximum number of positions to label
    """
    print(f"Reading FENs from {input_path}...")

    # Read input
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        if 'fen' in df.columns:
            fens = df['fen'].tolist()
        else:
            fens = df.iloc[:, 0].tolist()
    else:
        # Plain text file, one FEN per line
        with open(input_path, 'r') as f:
            fens = [line.strip() for line in f if line.strip()]

    # Limit positions if requested
    if max_positions:
        fens = fens[:max_positions]

    print(f"Found {len(fens)} positions")

    # Label in batches
    all_values = []
    for i in range(0, len(fens), batch_size):
        batch_fens = fens[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}/{(len(fens) + batch_size - 1) // batch_size}...")

        batch_values = label_positions_stockfish(
            batch_fens,
            stockfish_path,
            depth=depth,
            time_limit=time_limit,
        )

        all_values.extend(batch_values)

    # Create dataset
    dataset = {
        'fens': fens,
        'value_teacher': all_values,
        'metadata': {
            'depth': depth,
            'time_limit': time_limit,
            'num_positions': len(fens),
            'stockfish_path': stockfish_path,
        }
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(dataset, output_path)
    print(f"\nSaved teacher dataset to {output_path}")

    # Print statistics
    values_array = np.array(all_values)
    print(f"\nValue statistics:")
    print(f"  Mean: {values_array.mean():.4f}")
    print(f"  Std: {values_array.std():.4f}")
    print(f"  Min: {values_array.min():.4f}")
    print(f"  Max: {values_array.max():.4f}")
    print(f"  Positive: {(values_array > 0).sum()} ({(values_array > 0).mean() * 100:.1f}%)")
    print(f"  Negative: {(values_array < 0).sum()} ({(values_array < 0).mean() * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher value labels using Stockfish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with FENs (CSV or TXT)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .pt file path",
    )

    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="/usr/local/bin/stockfish",
        help="Path to Stockfish binary (default: /usr/local/bin/stockfish)",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Stockfish search depth (default: 10)",
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=0.1,
        help="Time limit per position in seconds (default: 0.1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum number of positions to label (default: all)",
    )

    args = parser.parse_args()

    # Validate Stockfish path
    if not Path(args.stockfish_path).exists():
        print(f"Error: Stockfish not found at {args.stockfish_path}")
        print("Install: brew install stockfish (macOS)")
        return 1

    create_teacher_dataset(
        input_path=args.input,
        output_path=args.output,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        time_limit=args.time_limit,
        batch_size=args.batch_size,
        max_positions=args.max_positions,
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
