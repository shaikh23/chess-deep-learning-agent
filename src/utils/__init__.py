"""Utility modules for encoding, metrics, plotting, and reproducibility."""

from .encoding import (
    board_to_tensor,
    move_to_index,
    index_to_move,
    get_legal_move_mask,
    MOVE_INDEX_SIZE,
)
from .metrics import (
    calculate_elo_difference,
    wilson_confidence_interval,
    compute_acpl,
    policy_top_k_accuracy,
)
from .plotting import (
    plot_phase_distribution,
    plot_training_curves,
    plot_match_results,
    plot_acpl_by_phase,
)
from .seeds import set_seed

__all__ = [
    "board_to_tensor",
    "move_to_index",
    "index_to_move",
    "get_legal_move_mask",
    "MOVE_INDEX_SIZE",
    "calculate_elo_difference",
    "wilson_confidence_interval",
    "compute_acpl",
    "policy_top_k_accuracy",
    "plot_phase_distribution",
    "plot_training_curves",
    "plot_match_results",
    "plot_acpl_by_phase",
    "set_seed",
]
