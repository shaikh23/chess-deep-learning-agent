# Maia (Lc0) Setup Guide

This guide covers running benchmarks against Maia, a human-like chess engine based on Lc0.

## ✓ Installation Complete

Your system is configured with:
- **Lc0**: `/usr/local/bin/lc0` ✓
- **Maia-1500 weights**: `weights/maia-1500.pb.gz` ✓

## Quick Start

### 1. Verify Installation

```bash
# Check all engines
python config/engines.py

# Run Maia sanity check
python -m src.play.sanity \
  --lc0-path /usr/local/bin/lc0 \
  --maia-weights weights/maia-1500.pb.gz
```

**Expected output:**
```
✓ Engine started: Maia-maia-1500
✓ Move from starting position is legal (e2e4)
✓ Reply to e2e4 is legal (e7e5)
```

### 2. Run Quick Benchmark (20 games)

```bash
# Using the convenience script
./scripts/run_maia_benchmark.sh 20 300

# Or directly
python -m src.play.match_runner \
  --opponent maia \
  --games 20 \
  --movetime 300 \
  --maia-weights weights/maia-1500.pb.gz \
  --lc0-path /usr/local/bin/lc0 \
  --threads 1
```

### 3. Run Full Benchmark (100 games)

```bash
# Recommended for final results
./scripts/run_maia_benchmark.sh 100 300
```

**Estimated time:** ~5-10 minutes for 20 games, 25-50 minutes for 100 games (depending on hardware).

## Jupyter Notebook

The notebook at `notebooks/04_benchmarks_and_analysis.ipynb` is already configured with the correct paths:

```python
# Cell 11 - Maia Sanity Check (already configured)
LCO_PATH = '/usr/local/bin/lc0'
MAIA_WEIGHTS = '../weights/maia-1500.pb.gz'
```

Just run the cells in order:
1. **Setup** → Load model and configuration
2. **Match 1** → vs Sunfish
3. **Match 2** → vs Stockfish
4. **Match 3** → Maia sanity check (runs automatically)
5. **Maia Benchmark** → 100 games vs Maia-1500
6. **Summary Table** → All results with Elo estimates

## Configuration

### Engine Paths

All engine configurations are centralized in `config/engines.py`:

```python
# Lc0 (for Maia)
LC0_PATH = "/usr/local/bin/lc0"

# Maia Weights
MAIA_1500_WEIGHTS = PROJECT_ROOT / "weights" / "maia-1500.pb.gz"

# Maia Default Settings
MAIA_DEFAULTS = {
    "movetime_ms": 300,     # milliseconds per move
    "threads": 1,           # number of threads
    "nn_backend": "cpu",    # "cpu", "cuda", "metal", etc.
    "depth": None,          # optional depth limit
}
```

### Adjusting Difficulty

**Time per move:**
```bash
# Faster (weaker): 100ms
python -m src.play.match_runner --opponent maia --movetime 100 ...

# Standard: 300ms (recommended)
python -m src.play.match_runner --opponent maia --movetime 300 ...

# Slower (stronger): 1000ms
python -m src.play.match_runner --opponent maia --movetime 1000 ...
```

**Threads** (if you have multiple cores):
```bash
# Single-threaded (deterministic)
python -m src.play.match_runner --opponent maia --threads 1 ...

# Multi-threaded (faster, less deterministic)
python -m src.play.match_runner --opponent maia --threads 4 ...
```

**Different rating levels:**

If you download other Maia weights from https://maiachess.com/:
```bash
# Maia-1100 (beginner)
python -m src.play.match_runner \
  --opponent maia \
  --maia-weights weights/maia-1100.pb.gz ...

# Maia-1900 (advanced)
python -m src.play.match_runner \
  --opponent maia \
  --maia-weights weights/maia-1900.pb.gz ...
```

## Match Output

Each match creates two files in `artifacts/matches/`:

1. **PGN file** (game records):
   ```
   match_Sunfish-2_vs_Maia-maia-1500_20251025_123456.pgn
   ```

2. **JSON file** (statistics):
   ```json
   {
     "player1": "Sunfish-2",
     "player2": "Maia-maia-1500",
     "num_games": 100,
     "statistics_player1": {
       "total_games": 100,
       "wins": 45,
       "draws": 20,
       "losses": 35,
       "score": 55.0
     }
   }
   ```

## Architecture Features

Your setup uses the robust architecture:

- ✓ **Single Authoritative Board**: One `python-chess` Board maintains game state
- ✓ **UCI-Only I/O**: All engines use UCI protocol (no XBoard issues)
- ✓ **External Opening Book**: 20 short lines for variety
- ✓ **Desync Guard**: Validates every engine move before applying
- ✓ **Per-Ply Logging**: Optional JSON logs for each half-move

## Troubleshooting

### Engine Not Found
```bash
# Check if Lc0 is installed
which lc0

# Install if missing
brew install lc0  # macOS
```

### Weights Not Found
```bash
# Check weights location
ls -lh weights/maia-1500.pb.gz

# If missing, download from https://maiachess.com/
# Place in: weights/maia-1500.pb.gz
```

### Slow Performance

**Reduce movetime:**
```bash
# From 300ms to 100ms
python -m src.play.match_runner --opponent maia --movetime 100 ...
```

**Or reduce number of games:**
```bash
# Quick test: 10 games
./scripts/run_maia_benchmark.sh 10
```

### Lc0 Backend Issues

If you see backend errors, try different backends:
```python
# In config/engines.py or notebook
MAIA_DEFAULTS["nn_backend"] = "blas"  # Alternative backend
```

Common backends:
- `"cpu"` - Default, works everywhere
- `"blas"` - Optimized CPU
- `"metal"` - macOS GPU acceleration
- `"cuda"` - NVIDIA GPU

## Next Steps

1. **Run benchmarks** in the notebook: `notebooks/04_benchmarks_and_analysis.ipynb`
2. **Compare results** across Sunfish, Stockfish, and Maia
3. **Analyze games** using the generated PGN files
4. **Experiment** with different time controls and rating levels

## Additional Resources

- **Maia Project**: https://maiachess.com/
- **Lc0 Documentation**: https://github.com/LeelaChessZero/lc0
- **Maia Paper**: https://arxiv.org/abs/2006.01855
- **Chess-DL-Agent README**: [README.md](README.md)

## Example Benchmark Results

```
======================================================================
COMPREHENSIVE BENCHMARK RESULTS
======================================================================
Opponent          Games  Wins  Draws  Losses  Score    Win Rate  Elo Diff  Elo 95% CI
Sunfish (D2)        100    75     15      10  82.5/100    75.0%      +220  [+165, +275]
Stockfish (Lv1)     100    35     25      40  47.5/100    35.0%       -20  [-75, +35]
Maia-1500           100    50     20      30  60.0/100    50.0%       +70  [+15, +125]
======================================================================
```

*Note: Actual results will vary based on your neural agent's training and configuration.*
