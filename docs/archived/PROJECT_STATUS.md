# Chess-DL-Agent Project Status

**Last Updated:** October 25, 2025
**Status:** ✓ Ready for Benchmarking

---

## Installation Status

### ✓ Engines Configured

| Engine | Status | Path | Notes |
|--------|--------|------|-------|
| **Stockfish** | ✓ Installed | `/usr/local/bin/stockfish` | Skill levels 0-20 |
| **Lc0** | ✓ Installed | `/usr/local/bin/lc0` | For Maia |
| **Maia-1500** | ✓ Configured | `weights/maia-1500.pb.gz` | Human-like ~1500 Elo |
| **Sunfish** | ✓ Built-in | Python package | Pure Python baseline |

**Verify:** `python config/engines.py`

---

## Project Structure

```
Chess app/
├── config/                      ✨ NEW - Engine configuration
│   ├── __init__.py
│   └── engines.py              # Centralized engine paths & settings
├── src/
│   ├── play/
│   │   ├── maia_lc0_wrapper.py ✨ NEW - Maia/Lc0 UCI wrapper
│   │   ├── sanity.py           ✨ NEW - Sanity check utility
│   │   ├── match_runner.py     ✏️ UPDATED - Robust architecture, Maia support
│   │   ├── opening_book.py     ✏️ UPDATED - UCI helpers
│   │   ├── stockfish_wrapper.py
│   │   └── sunfish_wrapper.py
│   ├── utils/
│   │   └── encoding.py         ✏️ UPDATED - san_to_uci(), validate_engine_move()
│   ├── data/                   # PGN parsing, datasets
│   ├── model/                  # Neural networks
│   └── search/                 # Alpha-beta, MCTS
├── scripts/
│   └── run_maia_benchmark.sh   ✨ NEW - Quick benchmark script
├── weights/
│   └── maia-1500.pb.gz         ✨ NEW - Maia weights
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_train_supervised.ipynb
│   ├── 03_search_and_play.ipynb
│   └── 04_benchmarks_and_analysis.ipynb ✏️ UPDATED - Maia support
├── MAIA_SETUP.md               ✨ NEW - Detailed Maia guide
├── QUICK_REFERENCE.md          ✨ NEW - Command reference
└── README.md                   ✏️ UPDATED - Maia docs, GNU removed
```

**Legend:**
- ✨ NEW - Newly created
- ✏️ UPDATED - Modified
- ❌ DELETED - `src/play/gnuchess_wrapper.py` (removed XBoard issues)

---

## Completed Tasks

### ✅ Phase 1: Maia Integration
- [x] Created `maia_lc0_wrapper.py` with UCI protocol
- [x] Implemented robust move validation
- [x] Added timeout handling and error guards
- [x] Tested with Maia-1500 weights

### ✅ Phase 2: Sanity Checks
- [x] Created `sanity.py` utility
- [x] Implemented CLI with `--lc0-path` and `--maia-weights`
- [x] Added JSON output option
- [x] Verified: Start position → e2e4, Reply to e2e4 → e7e5 ✓

### ✅ Phase 3: Match Runner Refactor
- [x] Single authoritative `chess.Board`
- [x] UCI-only I/O (no XBoard)
- [x] External opening book only
- [x] Desync guard with detailed logging
- [x] Per-ply JSON logging (optional)
- [x] Added Maia to engine registry
- [x] CLI with `--opponent maia`

### ✅ Phase 4: GNU Removal
- [x] Deleted `gnuchess_wrapper.py`
- [x] Removed all GNU imports
- [x] Updated notebook cells
- [x] Updated README references

### ✅ Phase 5: Documentation
- [x] Updated `README.md` with Maia sections
- [x] Created `MAIA_SETUP.md` guide
- [x] Created `QUICK_REFERENCE.md`
- [x] Updated notebook with correct paths
- [x] Created `config/engines.py` for centralized config

### ✅ Phase 6: Testing
- [x] Sanity check passed ✓
- [x] Engine validation script created
- [x] Quick benchmark script (`run_maia_benchmark.sh`)

---

## Architecture Improvements

### Before (Issues)
- ❌ XBoard protocol issues with GNU Chess
- ❌ Multiple move representations (SAN, UCI, algebraic)
- ❌ No move validation between engines
- ❌ Potential board state desync

### After (Robust)
- ✓ **UCI-only protocol** for all engines
- ✓ **Single authoritative Board** (python-chess)
- ✓ **Desync guard** validates every move
- ✓ **External opening book** only (no internal books)
- ✓ **Detailed error logging** with FEN, move history
- ✓ **Per-ply JSON logs** (optional)

---

## Ready to Use

### Quick Test
```bash
# 1. Verify all engines
python config/engines.py

# 2. Sanity check Maia
python -m src.play.sanity \
  --lc0-path /usr/local/bin/lc0 \
  --maia-weights weights/maia-1500.pb.gz

# 3. Quick benchmark (20 games, ~5 min)
./scripts/run_maia_benchmark.sh 20 300
```

### Full Workflow
```bash
# 1. Open Jupyter
jupyter notebook

# 2. Run notebooks in order:
#    - 01_eda_and_preprocessing.ipynb
#    - 02_train_supervised.ipynb
#    - 03_search_and_play.ipynb
#    - 04_benchmarks_and_analysis.ipynb (includes Maia)

# 3. Results will be in:
#    - artifacts/matches/*.pgn (game records)
#    - artifacts/matches/*.json (statistics)
#    - reports/figures/*.png (visualizations)
```

---

## Expected Results

With MiniResNet 6×64 trained on 100k positions:

| Opponent | Win Rate | Elo Difference | Notes |
|----------|----------|----------------|-------|
| **Sunfish (D2)** | 70-80% | +150 to +250 | Baseline opponent |
| **Stockfish (Lv5)** | 30-40% | -100 to +50 | Limited strength |
| **Maia-1500** | ~50% | ±50 | Human-like play |

*Actual results depend on training data, model architecture, and search depth.*

---

## Next Steps

### Immediate (Ready Now)
1. ✅ Run sanity checks
2. ✅ Run quick benchmark (20 games)
3. ⏭️ Run full benchmarks in notebook (100 games each)

### Analysis (After Benchmarks)
4. Compare Elo estimates across opponents
5. Analyze PGN files for interesting games
6. Review per-ply logs (if enabled)
7. Generate visualizations

### Future Improvements
- [ ] Add Maia-1100, 1300, 1700, 1900 weights
- [ ] Multi-threaded Lc0 for faster benchmarks
- [ ] ACPL (Average Centipawn Loss) analysis
- [ ] Opening book expansion
- [ ] Self-play reinforcement learning

---

## Known Limitations

1. **Training Data**: Limited to 100k positions (vs millions in AlphaZero)
2. **Search Depth**: 3 ply (vs 20+ in strong engines)
3. **No Opening Book**: Relies on training data
4. **No Tablebases**: Endgame play not optimal
5. **Single-Threaded**: No parallelization (by design for determinism)

---

## Configuration Defaults

All defaults are documented in `config/engines.py`:

```python
MAIA_DEFAULTS = {
    "movetime_ms": 300,     # Time per move
    "threads": 1,           # CPU threads (1 = deterministic)
    "nn_backend": "cpu",    # Neural network backend
    "depth": None,          # Optional depth limit
}

MATCH_DEFAULTS = {
    "num_games": 100,               # Games per match
    "max_moves_per_game": 200,      # Draw limit
    "alternate_colors": True,       # Fair matchup
    "enable_desync_guard": True,    # Validate moves
    "log_per_ply": False,           # Detailed logging
}
```

---

## Troubleshooting

See [MAIA_SETUP.md](MAIA_SETUP.md#troubleshooting) for detailed troubleshooting.

**Common issues:**
- Engine not found → Run `python config/engines.py`
- Slow performance → Reduce `--movetime` or `--games`
- Import errors → Ensure in project root: `cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"`

---

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Main project documentation |
| [MAIA_SETUP.md](MAIA_SETUP.md) | Detailed Maia guide |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Command reference |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | This file |

---

## Summary

✓ **All engines installed and tested**
✓ **Maia integration complete**
✓ **Robust architecture implemented**
✓ **GNU Chess removed**
✓ **Documentation complete**
✓ **Ready for full benchmarking**

**Next Action:** Run `./scripts/run_maia_benchmark.sh 20 300` for a quick test, then proceed to full benchmarks in the notebook! 🚀
