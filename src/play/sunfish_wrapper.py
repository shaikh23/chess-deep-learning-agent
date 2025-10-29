"""Wrapper for Sunfish engine (pure-Python chess engine).

Sunfish is a simple chess engine in ~100 lines of Python.
Source: https://github.com/thomasahle/sunfish
"""

import chess
from typing import Optional


# Sunfish engine code (simplified version)
# Full version: https://github.com/thomasahle/sunfish

class SunfishEngine:
    """
    Simplified Sunfish chess engine.

    This is a minimal pure-Python engine for baseline comparisons.
    """

    def __init__(self, depth: int = 2):
        """
        Initialize Sunfish.

        Args:
            depth: Search depth
        """
        self.depth = depth

        # Piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }

    def evaluate(self, board: chess.Board) -> float:
        """
        Simple material evaluation.

        Args:
            board: Board to evaluate

        Returns:
            Evaluation score
        """
        if board.is_checkmate():
            return -20000 if board.turn == chess.WHITE else 20000

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value

        return score

    def minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        """
        Minimax search with alpha-beta pruning.

        Args:
            board: Current board
            depth: Remaining depth
            alpha: Alpha value
            beta: Beta value
            maximizing: Whether maximizing or minimizing

        Returns:
            Evaluation score
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        legal_moves = list(board.legal_moves)

        if maximizing:
            max_eval = -float("inf")
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get best move using minimax.

        Args:
            board: Current board

        Returns:
            Best move
        """
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        best_value = -float("inf") if board.turn == chess.WHITE else float("inf")

        for move in legal_moves:
            board.push(move)
            value = self.minimax(
                board,
                self.depth - 1,
                -float("inf"),
                float("inf"),
                board.turn == chess.WHITE,
            )
            board.pop()

            if board.turn == chess.WHITE:
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_move


class SunfishWrapper:
    """
    Wrapper for Sunfish engine with consistent interface.
    """

    def __init__(self, depth: int = 2):
        """
        Initialize Sunfish wrapper.

        Args:
            depth: Search depth (1-3 recommended)
        """
        self.engine = SunfishEngine(depth=depth)
        self.depth = depth

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get best move from Sunfish.

        Args:
            board: Current board position

        Returns:
            Selected move
        """
        if board.is_game_over():
            raise ValueError("Game is over")

        move = self.engine.get_best_move(board)

        if move is None:
            # Fallback: pick first legal move
            import random
            return random.choice(list(board.legal_moves))

        return move

    def get_name(self) -> str:
        """Get engine name."""
        return f"Sunfish-D{self.depth}"


# ============================================================================
# Unit Tests
# ============================================================================

def _test_sunfish_wrapper():
    """Test Sunfish wrapper."""
    print("Testing SunfishWrapper...")

    engine = SunfishWrapper(depth=2)
    board = chess.Board()

    move = engine.get_move(board)

    print(f"  Move: {move}")
    print(f"  Engine: {engine.get_name()}")

    assert move in board.legal_moves, "Move should be legal"
    print("  ✓ Sunfish wrapper test passed")

    print("\nAll Sunfish wrapper tests passed! ✓")


if __name__ == "__main__":
    _test_sunfish_wrapper()
