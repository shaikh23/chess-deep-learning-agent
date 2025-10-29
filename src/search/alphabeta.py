"""Alpha-beta pruning search with neural network policy ordering and value evaluation.

Features:
- Iterative deepening
- Policy-based move ordering (descending by network logits)
- Value network evaluation at leaves
- Optional quiescence search on captures/checks
- Time and depth limits
"""

import chess
import torch
import time
import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.encoding import board_to_tensor, move_to_index
from utils.tt import TranspositionTable, NodeType
from search.static_eval import static_eval_normalized


@dataclass
class SearchConfig:
    """Configuration for alpha-beta search."""
    max_depth: int = 3
    time_limit: float = 1.0  # seconds
    movetime: Optional[float] = None  # If set, overrides time_limit (in seconds)
    use_policy_ordering: bool = True
    use_value_eval: bool = True
    quiescence_depth: int = 2  # depth of quiescence search
    enable_quiescence: bool = False
    use_transposition_table: bool = True
    tt_size: int = 100000  # max TT entries
    use_static_eval_blend: bool = True  # Blend network value with static eval
    static_eval_weight: float = 0.2  # Weight for static eval (0.8 network + 0.2 static)
    use_killer_moves: bool = True  # Enable killer move heuristic
    use_history_heuristic: bool = True  # Enable history heuristic


class AlphaBetaSearcher:
    """
    Alpha-beta pruning searcher with neural network integration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Optional[SearchConfig] = None,
    ):
        """
        Initialize searcher.

        Args:
            model: Neural network model (policy+value or policy-only)
            device: PyTorch device
            config: Search configuration
        """
        self.model = model
        self.device = device
        self.config = config or SearchConfig()

        self.model.eval()

        # Transposition table
        if self.config.use_transposition_table:
            self.tt = TranspositionTable(max_size=self.config.tt_size)
        else:
            self.tt = None

        # Killer moves (2 per depth level)
        self.killer_moves = {}  # depth -> [move1, move2]

        # History heuristic (from_square, to_square) -> score
        self.history = {}

        # Statistics
        self.nodes_searched = 0
        self.max_depth_reached = 0

    def search(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """
        Search for the best move using iterative deepening alpha-beta.

        Args:
            board: Current board position

        Returns:
            Tuple of (best_move, evaluation_score)
        """
        if board.is_game_over():
            raise ValueError("Cannot search in game-over position")

        self.nodes_searched = 0
        self.max_depth_reached = 0

        # Clear TT at start of search
        if self.tt is not None:
            self.tt.clear()

        # Clear killer moves and history
        self.killer_moves.clear()
        self.history.clear()

        start_time = time.time()
        best_move = None
        best_score = -float("inf")

        # Use movetime if specified, otherwise use time_limit
        time_budget = self.config.movetime if self.config.movetime is not None else self.config.time_limit

        # Iterative deepening
        for depth in range(1, self.config.max_depth + 1):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                break

            # Alpha-beta search at current depth
            move, score = self._alpha_beta_root(
                board,
                depth,
                -float("inf"),
                float("inf"),
                start_time,
            )

            if move is not None:
                best_move = move
                best_score = score
                self.max_depth_reached = depth

            # Check time limit again
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                break

        if best_move is None:
            # Fallback: pick first legal move
            best_move = next(iter(board.legal_moves))
            best_score = 0.0

        return best_move, best_score

    def _alpha_beta_root(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
    ) -> Tuple[Optional[chess.Move], float]:
        """
        Alpha-beta search at root node.

        Args:
            board: Current board
            depth: Search depth
            alpha: Alpha value
            beta: Beta value
            start_time: Search start time

        Returns:
            Tuple of (best_move, best_score)
        """
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None, 0.0

        # Order moves by policy + killer + history
        if self.config.use_policy_ordering:
            legal_moves = self._order_moves_by_policy(board, legal_moves, depth)

        best_move = legal_moves[0]
        best_score = -float("inf")

        for move in legal_moves:
            # Check time limit
            if time.time() - start_time >= self.config.time_limit:
                break

            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, start_time)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff

        return best_move, best_score

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
    ) -> float:
        """
        Recursive alpha-beta search with transposition table.

        Args:
            board: Current board
            depth: Remaining depth
            alpha: Alpha value
            beta: Beta value
            start_time: Search start time

        Returns:
            Evaluation score from current player's perspective
        """
        self.nodes_searched += 1

        # Probe transposition table
        alpha_orig = alpha
        if self.tt is not None:
            tt_entry = self.tt.probe(board, depth)
            if tt_entry is not None:
                if tt_entry.node_type == NodeType.EXACT:
                    return tt_entry.value
                elif tt_entry.node_type == NodeType.LOWERBOUND:
                    alpha = max(alpha, tt_entry.value)
                elif tt_entry.node_type == NodeType.UPPERBOUND:
                    beta = min(beta, tt_entry.value)

                if alpha >= beta:
                    return tt_entry.value

        # Check time limit
        time_budget = self.config.movetime if self.config.movetime is not None else self.config.time_limit
        if time.time() - start_time >= time_budget:
            return self._evaluate(board)

        # Terminal node checks
        if board.is_checkmate():
            return -10000.0  # Lost (from current player's perspective)

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0  # Draw

        if board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0.0  # Draw

        # Depth limit reached
        if depth <= 0:
            if self.config.enable_quiescence:
                return self._quiescence_search(board, alpha, beta, self.config.quiescence_depth, start_time)
            else:
                return self._evaluate(board)

        # Get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # Checkmate or stalemate (already handled above)
            return 0.0

        # Order moves by policy + killer + history
        if self.config.use_policy_ordering:
            legal_moves = self._order_moves_by_policy(board, legal_moves, depth)

        # Search moves
        best_value = -float("inf")
        best_move = None

        for move in legal_moves:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, start_time)
            board.pop()

            if score > best_value:
                best_value = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff - update killer moves and history
                if self.config.use_killer_moves and not board.is_capture(move):
                    self._update_killer_move(move, depth)

                if self.config.use_history_heuristic:
                    self._update_history(move, depth)

                break  # Beta cutoff

        # Store in transposition table
        if self.tt is not None:
            if best_value <= alpha_orig:
                node_type = NodeType.UPPERBOUND
            elif best_value >= beta:
                node_type = NodeType.LOWERBOUND
            else:
                node_type = NodeType.EXACT

            self.tt.store(board, depth, best_value, node_type, best_move)

        return alpha

    def _quiescence_search(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        depth: int,
        start_time: float,
    ) -> float:
        """
        Quiescence search to resolve tactical sequences.

        Only searches captures and checks.

        Args:
            board: Current board
            alpha: Alpha value
            beta: Beta value
            depth: Remaining quiescence depth
            start_time: Search start time

        Returns:
            Evaluation score
        """
        self.nodes_searched += 1

        # Stand pat evaluation
        stand_pat = self._evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Depth limit
        if depth <= 0:
            return stand_pat

        # Check time limit
        time_budget = self.config.movetime if self.config.movetime is not None else self.config.time_limit
        if time.time() - start_time >= time_budget:
            return stand_pat

        # Only search captures and checks
        tactical_moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]

        if not tactical_moves:
            return stand_pat

        for move in tactical_moves:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1, start_time)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using neural network value head or material count.

        Args:
            board: Board to evaluate

        Returns:
            Evaluation score from current player's perspective
        """
        if self.config.use_value_eval:
            return self._evaluate_with_network(board)
        else:
            return self._evaluate_material(board)

    def _evaluate_with_network(self, board: chess.Board) -> float:
        """
        Evaluate position using neural network value head, optionally blended with static eval.

        Args:
            board: Board to evaluate

        Returns:
            Value in range [-1, 1] from current player's perspective
        """
        with torch.no_grad():
            board_tensor = board_to_tensor(board, device=self.device).unsqueeze(0)

            # Get model prediction
            outputs = self.model(board_tensor, return_value=True)

            network_value = None
            if isinstance(outputs, tuple):
                if len(outputs) == 2:
                    _, value = outputs
                    if value is not None:
                        network_value = value.item()
                elif len(outputs) == 3:
                    _, _, value = outputs
                    if value is not None:
                        network_value = value.item()

            # If no value head, fallback to material
            if network_value is None:
                return self._evaluate_material(board)

            # Optionally blend with static eval
            if self.config.use_static_eval_blend:
                static_value = static_eval_normalized(board)
                blended_value = (
                    (1.0 - self.config.static_eval_weight) * network_value +
                    self.config.static_eval_weight * static_value
                )
                return blended_value
            else:
                return network_value

    def _evaluate_material(self, board: chess.Board) -> float:
        """
        Simple material evaluation.

        Args:
            board: Board to evaluate

        Returns:
            Material score from current player's perspective
        """
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }

        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece_values[piece.piece_type]
                if piece.color == board.turn:
                    score += value
                else:
                    score -= value

        return score / 39.0  # Normalize to roughly [-1, 1]

    def _order_moves_by_policy(self, board: chess.Board, moves: List[chess.Move], depth: int = 0) -> List[chess.Move]:
        """
        Order moves by policy network logits, killer moves, and history heuristic.

        Args:
            board: Current board
            moves: List of legal moves
            depth: Current search depth (for killer moves)

        Returns:
            Ordered list of moves
        """
        if not moves:
            return moves

        move_scores = []

        # Get policy scores
        with torch.no_grad():
            board_tensor = board_to_tensor(board, device=self.device).unsqueeze(0)

            # Get policy logits
            outputs = self.model(board_tensor, return_value=False)

            if isinstance(outputs, tuple):
                policy_logits = outputs[0]
            else:
                policy_logits = outputs

            # Extract logits for legal moves
            move_indices = [move_to_index(m) for m in moves]
            policy_scores = policy_logits[0, move_indices].cpu().numpy()

        # Score each move with policy + killer + history
        for i, move in enumerate(moves):
            score = float(policy_scores[i])

            # Bonus for killer moves
            if self.config.use_killer_moves and depth in self.killer_moves:
                if move in self.killer_moves[depth]:
                    score += 1000.0  # Large bonus

            # Bonus for history heuristic
            if self.config.use_history_heuristic:
                history_key = (move.from_square, move.to_square)
                history_score = self.history.get(history_key, 0)
                score += history_score * 0.01  # Small bonus based on history

            move_scores.append((move, score))

        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        ordered_moves = [move for move, _ in move_scores]

        return ordered_moves

    def _update_killer_move(self, move: chess.Move, depth: int):
        """
        Update killer moves table.

        Args:
            move: Move that caused a beta cutoff
            depth: Depth at which cutoff occurred
        """
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [move]
        elif move not in self.killer_moves[depth]:
            # Keep only 2 killer moves per depth
            self.killer_moves[depth] = [move] + self.killer_moves[depth][:1]

    def _update_history(self, move: chess.Move, depth: int):
        """
        Update history heuristic table.

        Args:
            move: Move that caused a beta cutoff
            depth: Depth at which cutoff occurred (used for scoring)
        """
        history_key = (move.from_square, move.to_square)
        if history_key not in self.history:
            self.history[history_key] = 0
        # Increment by depth^2 (deeper cutoffs are more valuable)
        self.history[history_key] += depth * depth

    def get_statistics(self) -> dict:
        """Get search statistics."""
        stats = {
            "nodes_searched": self.nodes_searched,
            "max_depth_reached": self.max_depth_reached,
        }

        # Add TT statistics if enabled
        if self.tt is not None:
            stats.update(self.tt.get_statistics())

        return stats


# ============================================================================
# Unit Tests
# ============================================================================

def _test_alphabeta():
    """Test alpha-beta search."""
    print("Testing AlphaBetaSearcher...")

    # Create a simple model for testing
    from model.nets import MiniResNetPolicyValue

    device = torch.device("cpu")
    model = MiniResNetPolicyValue(num_blocks=2, channels=32)
    model.to(device)
    model.eval()

    # Create searcher
    config = SearchConfig(max_depth=2, time_limit=0.5, use_policy_ordering=True, use_value_eval=True)
    searcher = AlphaBetaSearcher(model, device, config)

    # Test search from starting position
    board = chess.Board()
    move, score = searcher.search(board)

    print(f"  Best move: {move}")
    print(f"  Score: {score:.4f}")
    print(f"  Nodes searched: {searcher.nodes_searched}")
    print(f"  Max depth: {searcher.max_depth_reached}")

    assert move is not None, "Should return a move"
    assert move in board.legal_moves, "Move should be legal"

    print("✓ AlphaBeta test passed")


if __name__ == "__main__":
    _test_alphabeta()
