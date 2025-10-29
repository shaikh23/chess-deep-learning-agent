# Chess-DL-Agent: Complete Implementation Guide

This document contains all the code needed to implement the improvements for achieving 15-20/100 wins vs Sunfish.

## Summary of Improvements

1. ✅ **Model**: Value head added, clean APIs with log-probs
2. ✅ **Loss**: PolicyValueLoss with λ=0.7, ε=0.05
3. ✅ **Teacher Distillation**: Stockfish depth-10 labeling tool created
4. 🔄 **Search**: Need to add killer moves, history heuristic, quiescence, TT
5. 🔄 **Static Eval**: Need material + PST evaluation
6. 🔄 **Opening Book**: Need 20 quiet lines
7. 🔄 **Sampling**: Need phase stratification and mirroring
8. 🔄 **Metrics**: Need Elo estimation and ACPL
9. 🔄 **Sanity Checks**: Need comprehensive test suite
10. 🔄 **Presets**: Need fast/full configuration

## Files Completed

### 1. src/model/nets.py ✅
- Added `masked_log_softmax()` utility
- Updated `MiniResNetPolicyValue.forward()` to return `(logits, log_probs, value)`
- All models (MLP, CNN, ResNet) support both policy and value heads

### 2. src/model/loss.py ✅
- `PolicyValueLoss` with default λ=0.7, ε=0.05
- `topk_accuracy()` for top-k metrics
- `calibration_bins()` for confidence analysis

### 3. src/tools/teacher_label_sf.py ✅
- Stockfish depth-10 labeling (configurable)
- Centipawn → value conversion: `v = clip(cp/800, -1, 1)`
- Batch processing with progress bars
- Saves to `.pt` format with metadata

## Files To Implement

Due to the large scope, I recommend implementing these in priority order:

### Priority 1: Search Improvements (Biggest Impact)

**src/utils/tt.py** - Transposition Table
```python
"""Transposition table with Zobrist hashing for chess search."""

import chess
import random
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Initialize Zobrist hashing keys
random.seed(42)  # Reproducible
ZOBRIST_PIECES = {}
for piece_type in chess.PIECE_TYPES:
    for color in chess.COLORS:
        for square in chess.SQUARES:
            piece = chess.Piece(piece_type, color)
            ZOBRIST_PIECES[(piece, square)] = random.getrandbits(64)

ZOBRIST_TURN = random.getrandbits(64)
ZOBRIST_CASTLING = [random.getrandbits(64) for _ in range(16)]
ZOBRIST_EP = [random.getrandbits(64) for _ in range(8)]


class TTFlag(Enum):
    """Transposition table entry type."""
    EXACT = 0
    LOWER = 1  # Alpha cutoff
    UPPER = 2  # Beta cutoff


@dataclass
class TTEntry:
    """Transposition table entry."""
    hash_key: int
    depth: int
    score: float
    flag: TTFlag
    best_move: Optional[chess.Move]


def zobrist_hash(board: chess.Board) -> int:
    """Compute Zobrist hash for board position."""
    h = 0

    # Pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            h ^= ZOBRIST_PIECES[(piece, square)]

    # Side to move
    if board.turn == chess.BLACK:
        h ^= ZOBRIST_TURN

    # Castling rights
    h ^= ZOBRIST_CASTLING[board.castling_rights]

    # En passant
    if board.ep_square:
        h ^= ZOBRIST_EP[chess.square_file(board.ep_square)]

    return h


class TranspositionTable:
    """Transposition table for search."""

    def __init__(self, size: int = 1000000):
        self.size = size
        self.table = {}
        self.hits = 0
        self.stores = 0

    def store(self, hash_key: int, depth: int, score: float, flag: TTFlag, best_move: Optional[chess.Move]):
        """Store entry in TT."""
        index = hash_key % self.size
        entry = TTEntry(hash_key, depth, score, flag, best_move)
        self.table[index] = entry
        self.stores += 1

    def probe(self, hash_key: int, depth: int, alpha: float, beta: float) -> Tuple[Optional[float], Optional[chess.Move]]:
        """Probe TT for entry."""
        index = hash_key % self.size
        entry = self.table.get(index)

        if entry and entry.hash_key == hash_key and entry.depth >= depth:
            self.hits += 1

            if entry.flag == TTFlag.EXACT:
                return entry.score, entry.best_move
            elif entry.flag == TTFlag.LOWER and entry.score >= beta:
                return entry.score, entry.best_move
            elif entry.flag == TTFlag.UPPER and entry.score <= alpha:
                return entry.score, entry.best_move

            # Return best move even if score not usable
            return None, entry.best_move

        return None, None

    def clear(self):
        """Clear table."""
        self.table.clear()
        self.hits = 0
        self.stores = 0

    def hit_rate(self) -> float:
        """Get hit rate."""
        if self.stores == 0:
            return 0.0
        return self.hits / self.stores
```

**src/search/static_eval.py** - Static Evaluation
```python
"""Static evaluation for chess positions.

Uses material balance + piece-square tables (PST) + simple mobility.
"""

import chess
import numpy as np

# Material values (centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-Square Tables (from Sunfish, flipped for white perspective)
# Format: [rank 0-7][file 0-7], from white's perspective
PST = {
    chess.PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ],
}


def static_eval(board: chess.Board) -> float:
    """
    Static evaluation of position.

    Returns value from white's perspective in centipawns.
    Range: approximately [-10000, 10000]

    Components:
    - Material balance
    - Piece-square tables
    - Simple mobility (number of legal moves)
    """
    if board.is_checkmate():
        return -10000.0 if board.turn == chess.WHITE else 10000.0

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    score = 0.0

    # Material and PST
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Material value
            value = PIECE_VALUES[piece.piece_type]

            # PST value (flip square for black pieces)
            pst_square = square if piece.color == chess.WHITE else chess.square_mirror(square)
            pst_value = PST[piece.piece_type][pst_square]

            piece_score = value + pst_value

            if piece.color == chess.WHITE:
                score += piece_score
            else:
                score -= piece_score

    # Mobility (simplified)
    mobility = len(list(board.legal_moves))
    if board.turn == chess.WHITE:
        score += mobility * 0.5
    else:
        score -= mobility * 0.5

    return score


def static_eval_normalized(board: chess.Board, scale: float = 800.0) -> float:
    """
    Static evaluation normalized to [-1, 1].

    Args:
        board: Chess board
        scale: Scaling factor (default: 800 centipawns)

    Returns:
        Evaluation in [-1, 1] from current player's perspective
    """
    cp_score = static_eval(board)

    # Convert to current player's perspective
    if board.turn == chess.BLACK:
        cp_score = -cp_score

    # Normalize
    return np.clip(cp_score / scale, -1.0, 1.0)
```

### Priority 2: Opening Book

**Updated src/play/opening_book.py** - Add 20 quiet lines:

```python
# Add these lines to the OPENING_LINES list (replace existing):

OPENING_LINES = [
    # Starting position
    ("Starting Position", [], 0),

    # King's Pawn - Quiet
    ("Italian Game", ["e4", "e5", "Nf3", "Nc6", "Bc4"], 5),
    ("Giuoco Piano", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"], 6),
    ("Two Knights", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"], 6),

    # French Defense
    ("French Defense", ["e4", "e6", "d4", "d5"], 4),
    ("French Tarrasch", ["e4", "e6", "d4", "d5", "Nd2"], 5),

    # Caro-Kann
    ("Caro-Kann", ["e4", "c6", "d4", "d5"], 4),
    ("Caro-Kann Classical", ["e4", "c6", "d4", "d5", "Nc3", "dxe4"], 6),

    # Sicilian - Closed
    ("Sicilian Closed", ["e4", "c5", "Nc3"], 3),
    ("Sicilian Grand Prix", ["e4", "c5", "Nc3", "Nc6", "f4"], 5),

    # Queen's Pawn - Quiet
    ("London System", ["d4", "d5", "Bf4"], 3),
    ("London vs d5", ["d4", "d5", "Bf4", "Nf6", "e3"], 5),
    ("Colle System", ["d4", "d5", "Nf3", "Nf6", "e3"], 5),

    # Queen's Gambit
    ("Queen's Gambit", ["d4", "d5", "c4"], 3),
    ("QGD Orthodox", ["d4", "d5", "c4", "e6", "Nc3", "Nf6"], 6),
    ("Slav Defense", ["d4", "d5", "c4", "c6"], 4),
    ("Semi-Slav", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6"], 8),

    # Indian Defenses
    ("King's Indian", ["d4", "Nf6", "c4", "g6"], 4),
    ("Nimzo-Indian", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"], 6),

    # Flank Openings
    ("English Opening", ["c4", "e5"], 2),
    ("Réti Opening", ["Nf3", "d5", "c4"], 3),
]
```

### Priority 3: Enhanced Sampling

Add to **src/data/sampling.py**:

```python
def phase_stratified_sample(
    positions: List[dict],
    n_samples: int,
    phase_weights: dict = None,
) -> List[dict]:
    """
    Sample positions stratified by game phase.

    Phase determined by piece count:
    - Opening: 28-32 pieces
    - Middlegame: 15-27 pieces
    - Endgame: 6-14 pieces

    Args:
        positions: List of position dicts with 'fen' key
        n_samples: Total number of samples
        phase_weights: Dict with 'opening', 'middlegame', 'endgame' weights

    Returns:
        Stratified sample of positions
    """
    if phase_weights is None:
        phase_weights = {'opening': 0.25, 'middlegame': 0.50, 'endgame': 0.25}

    # Categorize by phase
    phases = {'opening': [], 'middlegame': [], 'endgame': []}

    for pos in positions:
        board = chess.Board(pos['fen'])
        piece_count = len(board.piece_map())

        if piece_count >= 28:
            phases['opening'].append(pos)
        elif piece_count >= 15:
            phases['middlegame'].append(pos)
        else:
            phases['endgame'].append(pos)

    # Sample from each phase
    samples = []
    for phase, weight in phase_weights.items():
        n_phase = int(n_samples * weight)
        if len(phases[phase]) > 0:
            phase_samples = random.sample(
                phases[phase],
                min(n_phase, len(phases[phase]))
            )
            samples.extend(phase_samples)

    return samples


def mirror_board(board: chess.Board) -> chess.Board:
    """Mirror board horizontally (file flip)."""
    mirrored = chess.Board(None)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            mirror_square = chess.square_mirror(square)
            # Flip file: a->h, b->g, etc.
            file_flipped = chess.square(7 - chess.square_file(mirror_square), chess.square_rank(mirror_square))
            mirrored.set_piece_at(file_flipped, piece)

    mirrored.turn = board.turn
    mirrored.castling_rights = board.castling_rights
    mirrored.ep_square = board.ep_square
    mirrored.halfmove_clock = board.halfmove_clock
    mirrored.fullmove_number = board.fullmove_number

    return mirrored
```

### Priority 4: Metrics and Benchmarking

Add to **src/utils/metrics.py**:

```python
def elo_from_score(score: float, n_games: int) -> Tuple[float, float, float]:
    """
    Estimate Elo difference from match score with Wilson confidence interval.

    Args:
        score: Score (wins + 0.5 * draws)
        n_games: Total games

    Returns:
        Tuple of (elo_diff, ci_lower, ci_upper)
    """
    from scipy import stats

    # Win percentage
    win_pct = score / n_games

    # Clip to avoid log(0)
    win_pct = np.clip(win_pct, 0.001, 0.999)

    # Elo formula: Elo_diff = -400 * log10(1/win_pct - 1)
    elo_diff = -400 * np.log10(1.0 / win_pct - 1.0)

    # Wilson score interval for win percentage
    z = stats.norm.ppf(0.975)  # 95% CI

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


def compute_acpl(
    game_moves: List[str],
    start_fen: str,
    stockfish_path: str,
    depth: int = 10,
) -> Tuple[float, List[float]]:
    """
    Compute Average Centipawn Loss (ACPL) for a game.

    Args:
        game_moves: List of UCI moves
        start_fen: Starting FEN
        stockfish_path: Path to Stockfish
        depth: Analysis depth

    Returns:
        Tuple of (acpl, move_losses)
    """
    import chess.engine

    losses = []
    board = chess.Board(start_fen)

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for move_uci in game_moves:
            # Get eval before move
            info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
            score_before = info_before['score'].relative.score(mate_score=10000)

            # Make move
            move = chess.Move.from_uci(move_uci)
            board.push(move)

            # Get eval after move (from opponent's perspective, so negate)
            info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
            score_after = -info_after['score'].relative.score(mate_score=10000)

            # Loss = score_before - score_after
            loss = max(0, score_before - score_after)
            losses.append(loss)

    acpl = np.mean(losses) if losses else 0.0
    return acpl, losses
```

##  Quick Start Commands

```bash
# 1. Generate teacher labels (50k positions, ~2 hours)
python -m src.tools.teacher_label_sf \
  --input data/processed/train_positions.csv \
  --output data/processed/teacher_labels_50k.pt \
  --stockfish-path /usr/local/bin/stockfish \
  --depth 10 \
  --max-positions 50000

# 2. Train with teacher distillation (in notebook)
# See notebook 02_train_supervised.ipynb for training loop

# 3. Run benchmarks
python -m src.play.match_runner \
  --opponent sunfish \
  --games 100 \
  --movetime 300 \
  --preset full

# 4. Analyze results
jupyter notebook notebooks/04_benchmarks_and_analysis.ipynb
```

## Expected Results

After implementing all improvements:

| Metric | Before | After (Target) |
|--------|--------|----------------|
| vs Sunfish wins | 2/100 | 15-20/100 |
| Policy top-1 | ~35% | ~50% |
| Value MSE | N/A | ~0.15 |
| Search depth (300ms) | 2 ply | 3-4 ply |
| TT hit rate | 0% | >20% |

## Implementation Priority

1. **Week 1**: Teacher labeling + training with value head
2. **Week 2**: Search improvements (TT, killer, quiescence)
3. **Week 3**: Opening book + static eval blending
4. **Week 4**: Full benchmarking + analysis

## Notes

- All code is MacBook-friendly (MPS support, moderate memory usage)
- UCI-only for engines (no XBoard issues)
- Minimal dependencies (torch, chess, scipy, pandas)
- Reasonable defaults documented throughout

---

*This is a living document. Update as implementation progresses.*
