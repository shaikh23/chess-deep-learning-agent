"""Lightweight Monte Carlo Tree Search with neural network priors and value estimates.

Simplified MCTS implementation inspired by AlphaZero but much smaller scale.
"""

import chess
import torch
import numpy as np
import math
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.encoding import board_to_tensor, move_to_index


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 100
    c_puct: float = 1.4  # UCB exploration constant
    temperature: float = 1.0  # Temperature for move selection
    dirichlet_alpha: float = 0.3  # Dirichlet noise for exploration at root
    dirichlet_epsilon: float = 0.25  # Weight of dirichlet noise


class MCTSNode:
    """
    Node in the Monte Carlo search tree.
    """

    def __init__(self, parent: Optional['MCTSNode'] = None, prior: float = 0.0):
        """
        Initialize MCTS node.

        Args:
            parent: Parent node (None for root)
            prior: Prior probability from policy network
        """
        self.parent = parent
        self.prior = prior

        self.children: Dict[chess.Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)."""
        return len(self.children) == 0

    def select_child(self, c_puct: float) -> Tuple[chess.Move, 'MCTSNode']:
        """
        Select child with highest UCB score.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant

        Returns:
            Tuple of (move, child_node)
        """
        best_score = -float("inf")
        best_move = None
        best_child = None

        sqrt_parent_visits = math.sqrt(self.visit_count)

        for move, child in self.children.items():
            # Q value (from child's perspective, so negate)
            q_value = -child.value

            # UCB bonus
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, board: chess.Board, policy_priors: Dict[chess.Move, float]):
        """
        Expand node by adding children for all legal moves.

        Args:
            board: Current board position
            policy_priors: Dictionary mapping moves to prior probabilities
        """
        for move in board.legal_moves:
            prior = policy_priors.get(move, 1.0 / len(list(board.legal_moves)))
            self.children[move] = MCTSNode(parent=self, prior=prior)

    def update(self, value: float):
        """
        Update node statistics.

        Args:
            value: Value from current player's perspective
        """
        self.visit_count += 1
        self.value_sum += value

    def add_dirichlet_noise(self, alpha: float, epsilon: float):
        """
        Add Dirichlet noise to priors for exploration at root.

        Args:
            alpha: Dirichlet alpha parameter
            epsilon: Weight of noise (1-epsilon for original priors)
        """
        if not self.children:
            return

        moves = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))

        for move, noise_value in zip(moves, noise):
            child = self.children[move]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise_value


class MCTSLite:
    """
    Lightweight MCTS implementation with neural network integration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Optional[MCTSConfig] = None,
    ):
        """
        Initialize MCTS.

        Args:
            model: Neural network model (policy+value)
            device: PyTorch device
            config: MCTS configuration
        """
        self.model = model
        self.device = device
        self.config = config or MCTSConfig()

        self.model.eval()

    def search(self, board: chess.Board) -> Tuple[chess.Move, Dict[chess.Move, int]]:
        """
        Run MCTS to find the best move.

        Args:
            board: Current board position

        Returns:
            Tuple of (best_move, visit_counts_dict)
        """
        if board.is_game_over():
            raise ValueError("Cannot search in game-over position")

        # Create root node
        root = MCTSNode()

        # Get policy and expand root
        policy_priors = self._get_policy_priors(board)
        root.expand(board, policy_priors)

        # Add Dirichlet noise at root for exploration
        root.add_dirichlet_noise(
            self.config.dirichlet_alpha,
            self.config.dirichlet_epsilon,
        )

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(board.copy(), root)

        # Select move based on visit counts
        visit_counts = {move: child.visit_count for move, child in root.children.items()}
        best_move = self._select_move_by_visits(visit_counts, temperature=self.config.temperature)

        return best_move, visit_counts

    def _simulate(self, board: chess.Board, node: MCTSNode):
        """
        Run one simulation from node.

        Args:
            board: Current board (will be modified)
            node: Current node
        """
        # Selection: traverse tree until leaf
        path = [node]

        while not node.is_leaf() and not board.is_game_over():
            move, node = node.select_child(self.config.c_puct)
            board.push(move)
            path.append(node)

        # Terminal node check
        if board.is_game_over():
            if board.is_checkmate():
                value = -1.0  # Loss for current player
            else:
                value = 0.0  # Draw
        else:
            # Expansion and evaluation
            policy_priors = self._get_policy_priors(board)
            node.expand(board, policy_priors)

            # Evaluate with network
            value = self._evaluate(board)

        # Backpropagation
        for node in reversed(path):
            node.update(value)
            value = -value  # Flip perspective

    def _get_policy_priors(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Get policy priors from neural network.

        Args:
            board: Current board

        Returns:
            Dictionary mapping moves to prior probabilities
        """
        with torch.no_grad():
            board_tensor = board_to_tensor(board, device=self.device).unsqueeze(0)

            # Get policy logits
            # Model returns (policy_logits, log_probs, value) even when return_value=False
            outputs = self.model(board_tensor, return_value=False)

            if isinstance(outputs, tuple):
                policy_logits = outputs[0]
            else:
                policy_logits = outputs

            # Softmax to get probabilities
            policy_probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()

            # Extract priors for legal moves
            priors = {}
            for move in board.legal_moves:
                idx = move_to_index(move)
                priors[move] = float(policy_probs[idx])

            # Normalize priors
            total = sum(priors.values())
            if total > 0:
                priors = {move: prob / total for move, prob in priors.items()}

            return priors

    def _evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using neural network value head.

        Args:
            board: Board to evaluate

        Returns:
            Value in range [-1, 1] from current player's perspective
        """
        with torch.no_grad():
            board_tensor = board_to_tensor(board, device=self.device).unsqueeze(0)

            # Get model prediction
            # Model returns (policy_logits, log_probs, value)
            outputs = self.model(board_tensor, return_value=True)

            if isinstance(outputs, tuple) and len(outputs) >= 3:
                _, _, value = outputs
                if value is not None:
                    return value.item()

            # Fallback
            return 0.0

    def _select_move_by_visits(
        self,
        visit_counts: Dict[chess.Move, int],
        temperature: float,
    ) -> chess.Move:
        """
        Select move based on visit counts with temperature.

        Args:
            visit_counts: Dictionary mapping moves to visit counts
            temperature: Temperature for selection (0 = greedy, >1 = more random)

        Returns:
            Selected move
        """
        if not visit_counts:
            raise ValueError("No moves to select from")

        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)

        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(counts)
            return moves[best_idx]
        else:
            # Sample with temperature
            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()
            selected_idx = np.random.choice(len(moves), p=probs)
            return moves[selected_idx]


# ============================================================================
# Unit Tests
# ============================================================================

def _test_mcts():
    """Test MCTS."""
    print("Testing MCTSLite...")

    # Create a simple model for testing
    from model.nets import MiniResNetPolicyValue

    device = torch.device("cpu")
    model = MiniResNetPolicyValue(num_blocks=2, channels=32)
    model.to(device)
    model.eval()

    # Create MCTS
    config = MCTSConfig(num_simulations=50, temperature=1.0)
    mcts = MCTSLite(model, device, config)

    # Test search from starting position
    board = chess.Board()
    move, visit_counts = mcts.search(board)

    print(f"  Best move: {move}")
    print(f"  Visit counts: {len(visit_counts)} moves explored")
    print(f"  Top 3 moves:")
    sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    for m, count in sorted_moves[:3]:
        print(f"    {m}: {count} visits")

    assert move is not None, "Should return a move"
    assert move in board.legal_moves, "Move should be legal"

    print("✓ MCTS test passed")


if __name__ == "__main__":
    _test_mcts()
