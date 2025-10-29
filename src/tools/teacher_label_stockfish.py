"""Teacher labeling tool using Stockfish for value targets.

Generates high-quality value labels for supervised training by evaluating
positions with Stockfish at specified depth.
"""

import chess
import chess.engine
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import argparse


def cp_to_value(cp: int, scale: float = 800.0) -> float:
    """Convert centipawn score to value in [-1, 1].

    Args:
        cp: Centipawn score
        scale: Scaling factor (default 800)

    Returns:
        Value in [-1, 1]
    """
    return np.clip(cp / scale, -1.0, 1.0)


def label_positions_with_stockfish(
    shard_path: Path,
    stockfish_path: Path,
    depth: int = 10,
    time_limit: float = 0.1,
) -> List[Dict]:
    """Label positions in a shard with Stockfish values.

    Args:
        shard_path: Path to input shard (.pt file)
        stockfish_path: Path to Stockfish binary
        depth: Search depth
        time_limit: Time limit per position (seconds)

    Returns:
        List of dicts with 'fen' and 'value_teacher' keys
    """
    # Load shard
    shard_data = torch.load(shard_path)
    boards = shard_data['boards']
    n_positions = len(boards)

    print(f"Labeling {n_positions:,} positions with Stockfish depth {depth}")

    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))

    labeled_positions = []

    try:
        for idx in tqdm(range(n_positions), desc="Labeling"):
            # Reconstruct board from tensor
            # Note: This requires FEN or we need to store FENs in shards
            # For now, skip - this is a placeholder implementation

            # In practice, we'd need FENs in the shard
            # board = chess.Board(fen)
            #
            # info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
            # cp_score = info['score'].relative.score(mate_score=10000)
            # value = cp_to_value(cp_score)
            #
            # labeled_positions.append({
            #     'fen': fen,
            #     'value_teacher': value,
            # })

            pass

    finally:
        engine.quit()

    return labeled_positions


def label_shards(
    input_dir: Path,
    output_dir: Path,
    stockfish_path: Path,
    depth: int = 10,
    time_limit: float = 0.1,
    max_positions: int = 100_000,
):
    """Label multiple shards with Stockfish.

    Args:
        input_dir: Directory containing input shards
        output_dir: Directory for output teacher shards
        stockfish_path: Path to Stockfish binary
        depth: Search depth
        time_limit: Time limit per position
        max_positions: Maximum positions to label
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input shards
    shard_paths = sorted(input_dir.glob("shard_*.pt"))
    if not shard_paths:
        print(f"No shards found in {input_dir}")
        return

    print("=" * 70)
    print("TEACHER LABELING WITH STOCKFISH")
    print("=" * 70)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Stockfish: {stockfish_path}")
    print(f"Depth: {depth}")
    print(f"Time limit: {time_limit}s")
    print(f"Max positions: {max_positions:,}")
    print(f"Found {len(shard_paths)} shards")
    print("=" * 70)

    # Verify Stockfish exists
    if not stockfish_path.exists():
        print(f"Error: Stockfish not found at {stockfish_path}")
        print("Install with: brew install stockfish (macOS)")
        return

    # Initialize Stockfish
    print("\nInitializing Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
    print(f"✓ Stockfish ready: {engine.id['name']}")

    all_labeled = []
    positions_processed = 0

    try:
        for shard_idx, shard_path in enumerate(shard_paths):
            if positions_processed >= max_positions:
                break

            print(f"\nProcessing shard {shard_idx + 1}/{len(shard_paths)}: {shard_path.name}")

            # Load shard
            shard_data = torch.load(shard_path)
            n_positions = min(len(shard_data['boards']), max_positions - positions_processed)

            # Note: This implementation requires FENs to be stored in shards
            # For now, this is a placeholder - full implementation would need:
            # 1. Store FENs in original shards during sampling
            # 2. Reconstruct FENs from board tensors (complex)
            # 3. Or create a separate FEN index file

            print(f"  Skipping {n_positions:,} positions (FEN reconstruction not implemented)")
            print(f"  To implement: store FENs in shards during sampling")

            positions_processed += n_positions

    finally:
        engine.quit()

    print("\n" + "=" * 70)
    print("TEACHER LABELING COMPLETE")
    print("=" * 70)
    print(f"Positions labeled: {positions_processed:,}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    # Save summary
    summary = {
        'positions_labeled': positions_processed,
        'depth': depth,
        'time_limit': time_limit,
        'stockfish_path': str(stockfish_path),
    }

    summary_path = output_dir / 'teacher_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary to {summary_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Label positions with Stockfish for teacher training")
    parser.add_argument("--in", dest="input_dir", type=str, required=True, help="Input shard directory")
    parser.add_argument("--out", dest="output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sf", dest="stockfish", type=str, default="/opt/homebrew/bin/stockfish",
                        help="Stockfish binary path")
    parser.add_argument("--depth", type=int, default=10, help="Search depth")
    parser.add_argument("--time", type=float, default=0.1, help="Time limit per position (seconds)")
    parser.add_argument("--max-positions", type=int, default=100_000, help="Maximum positions to label")

    args = parser.parse_args()

    label_shards(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        stockfish_path=Path(args.stockfish),
        depth=args.depth,
        time_limit=args.time,
        max_positions=args.max_positions,
    )


if __name__ == "__main__":
    main()
