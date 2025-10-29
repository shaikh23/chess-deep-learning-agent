# Chess-DL-Agent Quick Reference

## Your Configuration

```
✓ Stockfish:  /usr/local/bin/stockfish
✓ Lc0:        /usr/local/bin/lc0
✓ Maia-1500:  weights/maia-1500.pb.gz
```

## Common Commands

### Check Engine Status
```bash
python config/engines.py
```

### Sanity Checks
```bash
# Maia
python -m src.play.sanity \
  --lc0-path /usr/local/bin/lc0 \
  --maia-weights weights/maia-1500.pb.gz
```

### Run Benchmarks

**Quick test (20 games, ~5 min):**
```bash
./scripts/run_maia_benchmark.sh 20 300
```

**Full benchmark (100 games, ~30 min):**
```bash
./scripts/run_maia_benchmark.sh 100 300
```

**Manual:**
```bash
# vs Maia
python -m src.play.match_runner \
  --opponent maia \
  --games 100 \
  --movetime 300 \
  --maia-weights weights/maia-1500.pb.gz \
  --lc0-path /usr/local/bin/lc0

# vs Stockfish (Skill Level 5)
python -m src.play.match_runner \
  --opponent stockfish \
  --games 100 \
  --stockfish-skill 5 \
  --movetime 300

# vs Sunfish (Depth 2)
python -m src.play.match_runner \
  --opponent sunfish \
  --games 100 \
  --sunfish-depth 2
```

## Jupyter Notebooks

Run in order:
1. `01_eda_and_preprocessing.ipynb` - Data preparation
2. `02_train_supervised.ipynb` - Train neural network
3. `03_search_and_play.ipynb` - Test search algorithms
4. `04_benchmarks_and_analysis.ipynb` - **Run all benchmarks** ⭐

## Output Locations

```
artifacts/
├── weights/
│   └── best_model.pth          # Trained model
├── matches/
│   ├── match_*_vs_*.pgn       # Game records
│   └── match_*_vs_*.json      # Match statistics
└── logs/
    └── training_*.json         # Training logs

reports/
└── figures/
    ├── elo_estimates.png       # Elo comparison chart
    └── training_curves.png     # Training progress
```

## Configuration Files

- `config/engines.py` - Engine paths and defaults
- `src/play/opening_book.py` - Opening book (20 lines)
- `notebooks/04_benchmarks_and_analysis.ipynb` - Benchmark config

## Typical Workflow

```bash
# 1. Verify setup
python config/engines.py
python -m src.play.sanity --lc0-path /usr/local/bin/lc0 --maia-weights weights/maia-1500.pb.gz

# 2. Train model (or use pre-trained)
jupyter notebook notebooks/02_train_supervised.ipynb

# 3. Run benchmarks
jupyter notebook notebooks/04_benchmarks_and_analysis.ipynb
# OR
./scripts/run_maia_benchmark.sh 100 300

# 4. Analyze results
# - Check artifacts/matches/ for PGNs and JSON
# - View reports/figures/ for plots
```

## Troubleshooting

**Lc0 not found:**
```bash
brew install lc0
which lc0  # Should show /usr/local/bin/lc0
```

**Maia weights missing:**
```bash
# Download from https://maiachess.com/
# Place in: weights/maia-1500.pb.gz
ls -lh weights/
```

**Slow benchmarks:**
```bash
# Reduce movetime (300ms → 100ms)
./scripts/run_maia_benchmark.sh 20 100

# Or reduce games (100 → 20)
./scripts/run_maia_benchmark.sh 20 300
```

**Import errors:**
```bash
# Ensure you're in project root
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

## Key Parameters

| Engine | Parameter | Default | Range | Notes |
|--------|-----------|---------|-------|-------|
| Maia | `movetime` | 300ms | 100-1000ms | Time per move |
| Maia | `threads` | 1 | 1-8 | CPU threads |
| Stockfish | `skill_level` | 5 | 0-20 | 0=weakest, 20=full |
| Sunfish | `depth` | 2 | 1-4 | Search depth |
| Match | `games` | 100 | 20-300 | Num games |
| Match | `max_moves` | 200 | 100-300 | Draw limit |

## Architecture Highlights

✓ **Single Authoritative Board** - One `chess.Board` maintains state
✓ **UCI-Only Protocol** - All engines use UCI (no XBoard)
✓ **External Opening Book** - 20 short lines, engines don't use internal books
✓ **Desync Guard** - Validates every engine move before applying
✓ **Per-Ply Logging** - Optional JSON logs for each half-move

## Documentation

- [README.md](README.md) - Main project documentation
- [MAIA_SETUP.md](MAIA_SETUP.md) - Detailed Maia guide
- [notebooks/00_report.ipynb](notebooks/00_report.ipynb) - Project report

## Contact & Issues

- **GitHub**: [Report issues here]
- **Course**: Deep Learning, University of Colorado Boulder
- **Year**: 2025
