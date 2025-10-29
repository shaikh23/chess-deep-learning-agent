"""Metrics for evaluating chess agents: Elo, ACPL, policy accuracy, confidence intervals."""

import math
import numpy as np
import torch
import chess
import chess.pgn
import chess.engine
from typing import List, Tuple, Optional, Dict
from pathlib import Path


def calculate_elo_difference(
    wins: int, draws: int, losses: int, confidence: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate Elo rating difference from match results with confidence interval.

    Uses the formula: Elo_diff = -400 * log10((L + 0.5*D) / (W + 0.5*D))

    Args:
        wins: Number of wins
        draws: Number of draws
        losses: Number of losses
        confidence: Confidence level for interval (default: 0.95)

    Returns:
        Tuple of (elo_difference, (lower_bound, upper_bound))
    """
    total = wins + draws + losses
    if total == 0:
        return 0.0, (0.0, 0.0)

    # Calculate win rate (draws count as 0.5)
    score = wins + 0.5 * draws
    win_rate = score / total

    # Avoid division by zero
    win_rate = max(0.001, min(0.999, win_rate))

    # Elo difference formula
    elo_diff = -400 * math.log10((1 - win_rate) / win_rate)

    # Wilson confidence interval for win rate
    lower_wr, upper_wr = wilson_confidence_interval(score, total, confidence)

    # Convert win rate bounds to Elo bounds
    lower_wr = max(0.001, min(0.999, lower_wr))
    upper_wr = max(0.001, min(0.999, upper_wr))

    elo_lower = -400 * math.log10((1 - lower_wr) / lower_wr)
    elo_upper = -400 * math.log10((1 - upper_wr) / upper_wr)

    # Ensure lower < upper (flip if needed due to negative relationship)
    if elo_lower > elo_upper:
        elo_lower, elo_upper = elo_upper, elo_lower

    return elo_diff, (elo_lower, elo_upper)


def elo_from_score(score: float, n_games: int) -> Tuple[float, float, float]:
    """
    Estimate Elo difference from match score with Wilson confidence interval.

    Args:
        score: Score (wins + 0.5 * draws)
        n_games: Total games

    Returns:
        Tuple of (elo_diff, ci_lower, ci_upper)
    """
    # Win percentage
    win_pct = score / n_games

    # Clip to avoid log(0)
    win_pct = np.clip(win_pct, 0.001, 0.999)

    # Elo formula: Elo_diff = -400 * log10(1/win_pct - 1)
    elo_diff = -400 * np.log10(1.0 / win_pct - 1.0)

    # Wilson score interval for win percentage
    z = 1.96  # 95% CI

    # Wilson interval
    denominator = 1 + z**2 / n_games
    centre = (win_pct + z**2 / (2 * n_games)) / denominator
    spread = z * np.sqrt((win_pct * (1 - win_pct) / n_games + z**2 / (4 * n_games**2))) / denominator

    lower_pct = max(centre - spread, 0.001)
    upper_pct = min(centre + spread, 0.999)

    # Convert to Elo
    elo_lower = -400 * np.log10(1.0 / lower_pct - 1.0)
    elo_upper = -400 * np.log10(1.0 / upper_pct - 1.0)

    return elo_diff, elo_lower, elo_upper


def wilson_confidence_interval(
    successes: float, trials: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.

    More accurate than normal approximation for small sample sizes.

    Args:
        successes: Number of successes (can be fractional for draws)
        trials: Total number of trials
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound) for proportion
    """
    if trials == 0:
        return 0.0, 1.0

    p = successes / trials

    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


def compute_acpl(
    moves: List[chess.Move],
    board: chess.Board,
    engine_path: str,
    depth: int = 15,
    time_limit: float = 0.1,
) -> float:
    """
    Compute Average Centipawn Loss (ACPL) for a sequence of moves.

    Args:
        moves: List of moves to evaluate
        board: Starting board position
        engine_path: Path to Stockfish binary
        depth: Search depth for evaluation
        time_limit: Time limit per position in seconds

    Returns:
        Average centipawn loss across all moves
    """
    if not moves:
        return 0.0

    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        centipawn_losses = []

        test_board = board.copy()

        for move in moves:
            # Get best move and evaluation before playing
            info_before = engine.analyse(
                test_board,
                chess.engine.Limit(depth=depth, time=time_limit),
            )

            score_before = info_before.get("score")
            if score_before is None:
                continue

            # Convert score to centipawns from current player's perspective
            score_before_cp = score_before.relative.score(mate_score=10000)

            # Play the move
            test_board.push(move)

            # Get evaluation after move (from opponent's perspective)
            info_after = engine.analyse(
                test_board,
                chess.engine.Limit(depth=depth, time=time_limit),
            )

            score_after = info_after.get("score")
            if score_after is None:
                continue

            # Convert to centipawns from opponent's perspective, then negate
            score_after_cp = -score_after.relative.score(mate_score=10000)

            # Centipawn loss (positive = loss)
            cp_loss = max(0, score_before_cp - score_after_cp)
            centipawn_losses.append(cp_loss)

        engine.quit()

        if not centipawn_losses:
            return 0.0

        return np.mean(centipawn_losses)

    except Exception as e:
        print(f"Error computing ACPL: {e}")
        return 0.0


def compute_acpl_by_phase(
    game: chess.pgn.Game,
    engine_path: str,
    depth: int = 15,
) -> Dict[str, float]:
    """
    Compute ACPL separately for opening, middlegame, and endgame phases.

    Phase boundaries:
    - Opening: moves 1-10
    - Middlegame: moves 11-30
    - Endgame: moves 31+

    Args:
        game: python-chess Game object
        engine_path: Path to Stockfish binary
        depth: Search depth

    Returns:
        Dictionary with keys 'opening', 'middlegame', 'endgame'
    """
    board = game.board()
    moves = list(game.mainline_moves())

    # Split moves by phase
    opening_moves = moves[:10]
    middlegame_moves = moves[10:30]
    endgame_moves = moves[30:]

    results = {}

    if opening_moves:
        results["opening"] = compute_acpl(opening_moves, board.copy(), engine_path, depth)

    if middlegame_moves:
        temp_board = board.copy()
        for m in opening_moves:
            temp_board.push(m)
        results["middlegame"] = compute_acpl(middlegame_moves, temp_board, engine_path, depth)

    if endgame_moves:
        temp_board = board.copy()
        for m in opening_moves + middlegame_moves:
            temp_board.push(m)
        results["endgame"] = compute_acpl(endgame_moves, temp_board, engine_path, depth)

    return results


def policy_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Calculate top-k accuracy for policy predictions.

    Args:
        logits: Model predictions [batch_size, num_moves]
        targets: Ground truth move indices [batch_size]
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as a fraction
    """
    with torch.no_grad():
        _, top_k_preds = logits.topk(k, dim=1)
        targets = targets.view(-1, 1)
        correct = (top_k_preds == targets).any(dim=1)
        return correct.float().mean().item()


def value_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.2,
) -> float:
    """
    Calculate accuracy for value predictions.

    Considers prediction correct if within threshold of target.

    Args:
        predictions: Model value predictions [batch_size]
        targets: Ground truth values [batch_size]
        threshold: Acceptable error margin

    Returns:
        Accuracy as a fraction
    """
    with torch.no_grad():
        errors = torch.abs(predictions - targets)
        correct = errors <= threshold
        return correct.float().mean().item()


def compute_move_ordering_quality(
    policy_logits: torch.Tensor,
    legal_moves: List[chess.Move],
    best_move: chess.Move,
    board: chess.Board,
) -> float:
    """
    Measure how well policy ranks the best move.

    Args:
        policy_logits: Policy network output
        legal_moves: List of legal moves in position
        best_move: Engine's best move
        board: Current board state

    Returns:
        Percentile rank of best move (1.0 = ranked first, 0.0 = ranked last)
    """
    from .encoding import move_to_index

    # Get indices of legal moves
    move_indices = [move_to_index(m) for m in legal_moves]
    best_idx = move_to_index(best_move)

    # Extract logits for legal moves
    legal_logits = policy_logits[move_indices].detach()

    # Sort by logits (descending)
    sorted_indices = torch.argsort(legal_logits, descending=True)

    # Find rank of best move
    best_move_position = (sorted_indices == move_indices.index(best_idx)).nonzero(as_tuple=True)[0].item()

    # Convert to percentile (1.0 = best)
    percentile = 1.0 - (best_move_position / len(legal_moves))

    return percentile


# ============================================================================
# Unit Tests
# ============================================================================

def _test_metrics():
    """Test metric calculations."""
    print("Testing metrics...")

    # Test Elo calculation
    elo_diff, (lower, upper) = calculate_elo_difference(wins=70, draws=10, losses=20)
    print(f"Elo difference: {elo_diff:.1f} ({lower:.1f}, {upper:.1f})")
    assert elo_diff > 0, "Expected positive Elo with more wins"
    assert lower < elo_diff < upper, "Elo should be within CI"
    print("✓ Elo calculation test passed")

    # Test Wilson CI
    lower, upper = wilson_confidence_interval(50, 100, 0.95)
    assert 0 <= lower <= 0.5 <= upper <= 1.0, f"Invalid CI: ({lower}, {upper})"
    print("✓ Wilson CI test passed")

    # Test policy accuracy
    logits = torch.randn(10, 100)
    targets = torch.randint(0, 100, (10,))
    acc = policy_top_k_accuracy(logits, targets, k=5)
    assert 0 <= acc <= 1.0, f"Invalid accuracy: {acc}"
    print(f"✓ Policy top-5 accuracy test passed (acc={acc:.3f})")

    print("All metric tests passed! ✓")


if __name__ == "__main__":
    _test_metrics()
