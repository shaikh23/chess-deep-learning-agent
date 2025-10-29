"""Configuration module for chess-dl-agent."""

from .engines import (
    STOCKFISH_PATH,
    LC0_PATH,
    MAIA_1500_WEIGHTS,
    STOCKFISH_DEFAULTS,
    SUNFISH_DEFAULTS,
    MAIA_DEFAULTS,
    MATCH_DEFAULTS,
    validate_engines,
    get_stockfish_path,
    print_engine_status,
)

__all__ = [
    "STOCKFISH_PATH",
    "LC0_PATH",
    "MAIA_1500_WEIGHTS",
    "STOCKFISH_DEFAULTS",
    "SUNFISH_DEFAULTS",
    "MAIA_DEFAULTS",
    "MATCH_DEFAULTS",
    "validate_engines",
    "get_stockfish_path",
    "print_engine_status",
]
