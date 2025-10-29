"""Wrapper for Stockfish engine with configurable strength."""

import chess
import chess.engine
from pathlib import Path
from typing import Optional


class StockfishWrapper:
    """
    Wrapper for Stockfish engine.

    Supports skill level capping and time limits for fair comparisons.
    """

    def __init__(
        self,
        stockfish_path: str = "/opt/homebrew/bin/stockfish",
        skill_level: Optional[int] = None,
        time_limit: float = 0.2,
        depth_limit: Optional[int] = None,
    ):
        """
        Initialize Stockfish wrapper.

        Args:
            stockfish_path: Path to Stockfish binary
            skill_level: Skill level 0-20 (None = full strength)
            time_limit: Time limit per move in seconds
            depth_limit: Optional depth limit
        """
        self.stockfish_path = stockfish_path
        self.skill_level = skill_level
        self.time_limit = time_limit
        self.depth_limit = depth_limit

        # Verify Stockfish exists
        if not Path(stockfish_path).exists():
            print(f"Warning: Stockfish not found at {stockfish_path}")
            print("Please install Stockfish and update the path:")
            print("  macOS (Homebrew): brew install stockfish")
            print("  Linux: apt-get install stockfish")
            print("  Or download from: https://stockfishchess.org/download/")

        self.engine = None

    def __enter__(self):
        """Context manager entry."""
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

        # Set skill level if specified
        if self.skill_level is not None:
            self.engine.configure({"Skill Level": self.skill_level})

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.engine is not None:
            self.engine.quit()

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get best move from Stockfish.

        Args:
            board: Current board position

        Returns:
            Selected move
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Use 'with' statement.")

        if board.is_game_over():
            raise ValueError("Game is over")

        # Set search limits
        limit = chess.engine.Limit(time=self.time_limit)
        if self.depth_limit is not None:
            limit = chess.engine.Limit(depth=self.depth_limit)

        # Get move
        result = self.engine.play(board, limit)
        return result.move

    def get_name(self) -> str:
        """Get engine name."""
        if self.skill_level is not None:
            return f"Stockfish-Lv{self.skill_level}"
        else:
            return "Stockfish"

    def analyze(
        self,
        board: chess.Board,
        depth: int = 15,
        time_limit: float = 0.1,
    ) -> dict:
        """
        Analyze position.

        Args:
            board: Board to analyze
            depth: Analysis depth
            time_limit: Time limit in seconds

        Returns:
            Analysis info dictionary
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Use 'with' statement.")

        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=time_limit),
        )

        return info


# ============================================================================
# Unit Tests
# ============================================================================

def _test_stockfish_wrapper():
    """Test Stockfish wrapper."""
    print("Testing StockfishWrapper...")

    stockfish_path = "/opt/homebrew/bin/stockfish"

    # Check if Stockfish exists
    if not Path(stockfish_path).exists():
        print(f"Stockfish not found at {stockfish_path}, skipping test")
        print("Please install Stockfish to run this test")
        return

    # Test with skill level
    with StockfishWrapper(stockfish_path, skill_level=5, time_limit=0.1) as engine:
        board = chess.Board()
        move = engine.get_move(board)

        print(f"  Move: {move}")
        print(f"  Engine: {engine.get_name()}")

        assert move in board.legal_moves, "Move should be legal"
        print("  ✓ Stockfish wrapper test passed")

    print("\nAll Stockfish wrapper tests passed! ✓")


if __name__ == "__main__":
    _test_stockfish_wrapper()
