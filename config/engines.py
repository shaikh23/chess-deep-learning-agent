"""Engine configuration for chess-dl-agent.

This file contains paths and default settings for chess engines.
Update these paths to match your local installation.
"""

from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# Engine Paths
# ============================================================================

# Stockfish
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # macOS Homebrew
STOCKFISH_ALT_PATH = "/usr/local/bin/stockfish"  # Alternative path

# Lc0 (for Maia)
LC0_PATH = "/usr/local/bin/lc0"

# Maia Weights
MAIA_WEIGHTS_DIR = PROJECT_ROOT / "weights"
MAIA_1500_WEIGHTS = MAIA_WEIGHTS_DIR / "maia-1500.pb.gz"

# ============================================================================
# Engine Default Settings
# ============================================================================

# Stockfish
STOCKFISH_DEFAULTS = {
    "skill_level": 5,       # 0-20, lower is weaker
    "time_limit": 0.3,      # seconds per move
    "depth_limit": None,    # or set to 3-5 for depth-limited search
}

# Sunfish
SUNFISH_DEFAULTS = {
    "depth": 2,             # search depth (1-4)
}

# Maia/Lc0
MAIA_DEFAULTS = {
    "movetime_ms": 300,     # milliseconds per move
    "threads": 1,           # number of threads
    "nn_backend": "cpu",    # "cpu", "cuda", "metal", etc.
    "depth": None,          # optional depth limit
}

# ============================================================================
# Match Configuration
# ============================================================================

MATCH_DEFAULTS = {
    "num_games": 100,               # games per match
    "max_moves_per_game": 200,      # max moves before draw
    "alternate_colors": True,       # alternate colors each game
    "enable_desync_guard": True,    # validate moves before applying
    "log_per_ply": False,           # detailed per-move logging
}

# ============================================================================
# Validation Functions
# ============================================================================

def validate_engines() -> dict:
    """
    Check which engines are available.

    Returns:
        Dictionary with engine availability status
    """
    status = {
        "stockfish": Path(STOCKFISH_PATH).exists() or Path(STOCKFISH_ALT_PATH).exists(),
        "lc0": Path(LC0_PATH).exists(),
        "maia_1500": MAIA_1500_WEIGHTS.exists(),
    }

    return status


def get_stockfish_path() -> str:
    """Get valid Stockfish path or raise error."""
    if Path(STOCKFISH_PATH).exists():
        return STOCKFISH_PATH
    elif Path(STOCKFISH_ALT_PATH).exists():
        return STOCKFISH_ALT_PATH
    else:
        raise FileNotFoundError(
            f"Stockfish not found at {STOCKFISH_PATH} or {STOCKFISH_ALT_PATH}\n"
            f"Install with: brew install stockfish (macOS)"
        )


def print_engine_status():
    """Print status of all engines."""
    status = validate_engines()

    print("=" * 70)
    print("ENGINE STATUS")
    print("=" * 70)

    print(f"Stockfish:    {'✓ Available' if status['stockfish'] else '✗ Not found'}")
    if status['stockfish']:
        path = STOCKFISH_PATH if Path(STOCKFISH_PATH).exists() else STOCKFISH_ALT_PATH
        print(f"              {path}")

    print(f"Lc0:          {'✓ Available' if status['lc0'] else '✗ Not found'}")
    if status['lc0']:
        print(f"              {LC0_PATH}")

    print(f"Maia-1500:    {'✓ Available' if status['maia_1500'] else '✗ Not found'}")
    if status['maia_1500']:
        print(f"              {MAIA_1500_WEIGHTS}")

    print("=" * 70)

    if not status['stockfish']:
        print("\nTo install Stockfish:")
        print("  macOS: brew install stockfish")
        print("  Linux: apt-get install stockfish")

    if not status['lc0']:
        print("\nTo install Lc0:")
        print("  macOS: brew install lc0")
        print("  Linux: https://github.com/LeelaChessZero/lc0/releases")

    if not status['maia_1500']:
        print("\nTo download Maia weights:")
        print("  Visit: https://maiachess.com/")
        print(f"  Save to: {MAIA_WEIGHTS_DIR}/")

    return status


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print_engine_status()
