"""Sanity check utility for Maia/Lc0 engine.

Validates that the Maia engine is working correctly by running simple test positions
and verifying that it returns legal moves.

Usage:
    python -m src.play.sanity --lc0-path /usr/local/bin/lc0 --maia-weights weights/maia1500.pb.gz
"""

import argparse
import json
import sys
import chess
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.play.maia_lc0_wrapper import MaiaLc0Engine


def sanity_check_maia(
    lc0_path: str,
    weights_path: str,
    threads: int = 1,
    movetime_ms: int = 200,
) -> Dict[str, str]:
    """
    Run sanity checks on Maia engine.

    Tests:
    1. Starting position: Request move from initial position
    2. Reply to e2e4: Play e2e4, then ask Maia to reply

    Args:
        lc0_path: Path to Lc0 binary
        weights_path: Path to Maia weights file
        threads: Number of threads (default: 1)
        movetime_ms: Time per move in ms (default: 200)

    Returns:
        Dictionary with test results:
        {
            "start_reply": "<uci_move>",
            "reply_to_e4": "<uci_move>"
        }

    Raises:
        RuntimeError: If any test fails
    """
    print("=" * 70)
    print("MAIA SANITY CHECK")
    print("=" * 70)
    print(f"Lc0 path: {lc0_path}")
    print(f"Weights: {weights_path}")
    print(f"Threads: {threads}")
    print(f"Movetime: {movetime_ms}ms")
    print("=" * 70)

    results = {}

    # Start engine
    print("\n[1/3] Starting Maia engine...")
    try:
        engine = MaiaLc0Engine(
            lc0_path=lc0_path,
            weights_path=weights_path,
            threads=threads,
            movetime_ms=movetime_ms,
        )
        engine._start_engine()
        engine._configure_engine()
        print(f"      ✓ Engine started: {engine.get_name()}")
    except Exception as e:
        print(f"      ✗ Failed to start engine: {e}")
        raise RuntimeError(f"Engine startup failed: {e}")

    try:
        # Test 1: Starting position
        print("\n[2/3] Test 1: Move from starting position")
        board = chess.Board()
        start_fen = board.fen()

        print(f"      FEN: {start_fen}")
        print(f"      Requesting move...")

        try:
            move_uci = engine.best_move(start_fen, moves_uci=None)
            print(f"      Move: {move_uci}")

            # Validate legality
            move_obj = chess.Move.from_uci(move_uci)
            if move_obj not in board.legal_moves:
                raise RuntimeError(f"Illegal move: {move_uci}")

            print(f"      ✓ Move is legal")
            results["start_reply"] = move_uci

            # Push our move to the board
            board.push(move_obj)

        except Exception as e:
            print(f"      ✗ Test 1 failed: {e}")
            engine.quit()
            raise RuntimeError(f"Test 1 failed: {e}")

        # Test 2: Reply to e2e4
        print("\n[3/3] Test 2: Reply to e2e4")

        # Reset board and play e2e4
        board = chess.Board()
        our_move = chess.Move.from_uci("e2e4")
        board.push(our_move)

        print(f"      We play: e2e4")
        print(f"      FEN: {board.fen()}")
        print(f"      Requesting Maia's reply...")

        try:
            # Ask Maia to reply (from starting FEN with e2e4 applied)
            reply_uci = engine.best_move(chess.STARTING_FEN, moves_uci=["e2e4"])
            print(f"      Reply: {reply_uci}")

            # Validate legality
            reply_obj = chess.Move.from_uci(reply_uci)
            if reply_obj not in board.legal_moves:
                raise RuntimeError(f"Illegal move: {reply_uci}")

            print(f"      ✓ Reply is legal")
            results["reply_to_e4"] = reply_uci

        except Exception as e:
            print(f"      ✗ Test 2 failed: {e}")
            engine.quit()
            raise RuntimeError(f"Test 2 failed: {e}")

    finally:
        # Clean up
        print("\n[Cleanup] Shutting down engine...")
        engine.quit()
        print("          ✓ Engine shut down")

    # Summary
    print("\n" + "=" * 70)
    print("SANITY CHECK PASSED")
    print("=" * 70)
    print(f"Start position reply: {results['start_reply']}")
    print(f"Reply to e2e4:        {results['reply_to_e4']}")
    print("=" * 70)

    return results


def main():
    """Command-line entry point for sanity check."""
    parser = argparse.ArgumentParser(
        description="Sanity check for Maia/Lc0 engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.play.sanity --lc0-path /usr/local/bin/lc0 --maia-weights weights/maia1500.pb.gz
  python -m src.play.sanity --lc0-path /usr/local/bin/lc0 --maia-weights weights/maia1900.pb.gz --threads 2
        """,
    )

    parser.add_argument(
        "--lc0-path",
        type=str,
        default="/usr/local/bin/lc0",
        help="Path to Lc0 binary (default: /usr/local/bin/lc0)",
    )

    parser.add_argument(
        "--maia-weights",
        type=str,
        required=True,
        help="Path to Maia weights file (e.g., weights/maia1500.pb.gz)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads (default: 1)",
    )

    parser.add_argument(
        "--movetime",
        type=int,
        default=200,
        help="Time per move in milliseconds (default: 200)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.lc0_path).exists():
        print(f"Error: Lc0 not found at {args.lc0_path}", file=sys.stderr)
        print("\nInstall Lc0:", file=sys.stderr)
        print("  macOS: brew install lc0", file=sys.stderr)
        print("  Linux: https://github.com/LeelaChessZero/lc0/releases", file=sys.stderr)
        sys.exit(1)

    if not Path(args.maia_weights).exists():
        print(f"Error: Maia weights not found at {args.maia_weights}", file=sys.stderr)
        print("\nDownload Maia weights from: https://maiachess.com/", file=sys.stderr)
        sys.exit(1)

    # Run sanity check
    try:
        results = sanity_check_maia(
            lc0_path=args.lc0_path,
            weights_path=args.maia_weights,
            threads=args.threads,
            movetime_ms=args.movetime,
        )

        # Output results
        if args.json:
            print("\n" + json.dumps(results, indent=2))

        sys.exit(0)

    except Exception as e:
        print(f"\n✗ SANITY CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
