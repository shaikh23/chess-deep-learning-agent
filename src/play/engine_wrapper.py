"""Wrapper for neural network chess engine with search."""

import chess
import torch
from typing import Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from search.alphabeta import AlphaBetaSearcher, SearchConfig
from search.mcts_lite import MCTSLite, MCTSConfig


class NeuralEngineWrapper:
    """
    Wrapper for our neural network chess agent.

    Provides a uniform interface for playing games.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        search_type: str = "alphabeta",
        search_config: Optional[dict] = None,
    ):
        """
        Initialize neural engine wrapper.

        Args:
            model: Trained neural network model
            device: PyTorch device
            search_type: Type of search ('alphabeta' or 'mcts')
            search_config: Configuration for search algorithm
        """
        self.model = model
        self.device = device
        self.search_type = search_type

        # Initialize searcher
        if search_type == "alphabeta":
            config = SearchConfig(**(search_config or {}))
            self.searcher = AlphaBetaSearcher(model, device, config)
        elif search_type == "mcts":
            config = MCTSConfig(**(search_config or {}))
            self.searcher = MCTSLite(model, device, config)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get best move for current position.

        Args:
            board: Current board position

        Returns:
            Selected move
        """
        if board.is_game_over():
            raise ValueError("Game is over")

        if self.search_type == "alphabeta":
            move, score = self.searcher.search(board)
            return move
        elif self.search_type == "mcts":
            move, visit_counts = self.searcher.search(board)
            return move
        else:
            # Fallback: random legal move
            import random
            return random.choice(list(board.legal_moves))

    def get_name(self) -> str:
        """Get engine name."""
        search_name = self.search_type.upper()
        return f"NeuralAgent-{search_name}"

    def get_statistics(self) -> dict:
        """Get search statistics."""
        if hasattr(self.searcher, "get_statistics"):
            return self.searcher.get_statistics()
        return {}


# ============================================================================
# Unit Tests
# ============================================================================

def _test_engine_wrapper():
    """Test neural engine wrapper."""
    print("Testing NeuralEngineWrapper...")

    from model.nets import MiniResNetPolicyValue

    device = torch.device("cpu")
    model = MiniResNetPolicyValue(num_blocks=2, channels=32)
    model.to(device)
    model.eval()

    # Test with alpha-beta
    print("\n1. Testing with alpha-beta search...")
    engine = NeuralEngineWrapper(
        model,
        device,
        search_type="alphabeta",
        search_config={"max_depth": 2, "time_limit": 0.2},
    )

    board = chess.Board()
    move = engine.get_move(board)

    print(f"   Move: {move}")
    print(f"   Engine name: {engine.get_name()}")

    assert move in board.legal_moves, "Move should be legal"
    print("   ✓ Alpha-beta engine test passed")

    # Test with MCTS
    print("\n2. Testing with MCTS...")
    engine = NeuralEngineWrapper(
        model,
        device,
        search_type="mcts",
        search_config={"num_simulations": 20},
    )

    move = engine.get_move(board)
    print(f"   Move: {move}")

    assert move in board.legal_moves, "Move should be legal"
    print("   ✓ MCTS engine test passed")

    print("\nAll engine wrapper tests passed! ✓")


if __name__ == "__main__":
    _test_engine_wrapper()
