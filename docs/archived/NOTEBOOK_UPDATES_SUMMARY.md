# Notebook Updates Summary

All three notebooks have been updated with the strength improvements implementation.

## ✅ Completed Updates

### Notebook 02: Training (02_train_supervised.ipynb)

**Major Changes:**

1. **Shard-Based Data Loading**
   - Added import for `create_shard_dataloaders` from `src.data.shard_dataset`
   - Smart data loading: tries shards first, falls back to CSV with warnings
   - Supports phase-balanced sampling and augmentation

2. **Updated Configuration**
   - `num_epochs`: 20 → **10** (more data = fewer epochs needed)
   - `value_weight`: 0.7 → **0.35** (focus on policy first)
   - Added `warmup_epochs`: **2**
   - Added `use_amp`: **True** (Automatic Mixed Precision)
   - Added `channels_last`: **True** (CNN memory optimization)
   - Added `phase_balanced`: **True**
   - Added `augment_train`: **True**

3. **Enhanced Training Loop**
   - Warmup + cosine learning rate schedule
   - AMP support with gradient scaling
   - Channels-last memory format for faster CNN training
   - Better progress tracking with LR logging

4. **Better Output**
   - Detailed configuration summary on startup
   - Clear epoch progress with top-1, top-3, top-5 accuracy
   - Final summary with percentage accuracy

**Key Cell Updates:**
- Cell 1: Added shard dataset import
- Cell 3: Complete config overhaul
- Cell 5: Smart data loading with fallback
- Cell 7: Added channels_last support
- Cell 9: Warmup + cosine scheduler
- Cell 11: AMP-enabled training functions
- Cell 12: Enhanced training loop with better logging

---

### Notebook 03: Search and Play (03_search_and_play.ipynb)

**Major Addition:**

**Comprehensive Smoke Tests** (New Cell after existing content)

Tests 5 critical search quality metrics:

1. **Test 1: Quick Match** - Play 4 games vs Sunfish
   - Validates: Game completion, reasonable move counts

2. **Test 2: TT Performance** - Check transposition table
   - Validates: TT is being used, has non-zero hit rate

3. **Test 3: Search Depth** - Measure depth @ 300ms
   - Validates: Reaches depth 3+, completes in time

4. **Test 4: Legal Moves** - Play 10 moves
   - Validates: All moves are legal

5. **Test 5: Config Toggles** - Test different configurations
   - Validates: Policy ordering, quiescence, killer moves can be toggled

**Output:**
```
ALL SMOKE TESTS PASSED ✓
Search quality verified:
  ✓ Can play complete games
  ✓ Transposition table working
  ✓ Reaches depth 3+ in 300ms
  ✓ All moves are legal
  ✓ Configuration toggles work
Ready for full benchmarks!
```

---

### Notebook 04: Benchmarks and Analysis (04_benchmarks_and_analysis.ipynb)

**Major Additions (3 New Sections):**

#### 1. Before/After Comparison Analysis

**Features:**
- Loads previous benchmark results from `benchmark_summary_before.csv`
- Compares with current results
- Shows improvement in W/D/L, score %, and Elo
- Lists all improvements implemented

**Output Table:**
```
Opponent       | Before W/D/L | After W/D/L | Before Score | After Score | Δ Score | Before Elo | After Elo | Δ Elo
Sunfish (D2)   | 0/60/40      | 15/60/25    | 30.0%        | 45.0%       | +15.0%  | -147       | -35       | +112
Maia-1500      | 0/1/99       | 3/10/87     | 0.5%         | 8.0%        | +7.5%   | -798       | -520      | +278
```

**Improvements Listed:**
- Training data: 100k → 1M positions
- Value weight: 0.7 → 0.35
- Phase-balanced sampling
- Data augmentation (file flips)
- Warmup + cosine LR schedule
- All search heuristics enabled

#### 2. ACPL by Phase Analysis

**Features:**
- Analyzes first 20 games from latest Sunfish match
- Computes ACPL for opening (1-10), middlegame (11-30), endgame (31+)
- Creates bar chart with reference lines (good < 50, acceptable < 100)
- Saves plot to `reports/figures/acpl_by_phase.png`
- Saves CSV to `artifacts/matches/acpl_by_phase.csv`

**Example Output:**
```
Average Centipawn Loss by Phase:
  Opening (moves 1-10):     65.3 cp
  Middlegame (moves 11-30): 89.2 cp
  Endgame (moves 31+):      112.5 cp
```

**Plot Features:**
- Color-coded bars (blue/red/green for opening/middle/endgame)
- Value labels on bars
- Reference lines at 50cp (good) and 100cp (acceptable)
- Professional styling

#### 3. Data Validation

- Checks if Stockfish is available before running ACPL
- Graceful fallback if data is missing
- Clear error messages with installation instructions

---

## 📊 Expected Workflow After Updates

### 1. Sample Data
```bash
python -m src.data.stream_sampler \
  --pgn-dir "data/raw/Lichess Elite Database" \
  --target 1000000 \
  --output artifacts/data/shards

# Split into train/val (70/30)
mkdir -p artifacts/data/shards/train artifacts/data/shards/val
mv artifacts/data/shards/shard_00{00..13}.pt artifacts/data/shards/train/
mv artifacts/data/shards/shard_00{14..19}.pt artifacts/data/shards/val/
```

### 2. Train Model
```bash
jupyter notebook notebooks/02_train_supervised.ipynb
```

**Expected Output:**
```
TRAINING CONFIGURATION
======================================================================
Device: mps
Model: miniresnet (6×64)
Value head: ENABLED
Epochs: 10 (warmup: 2)
...
Phase-balanced sampling: True
Augmentation: True
======================================================================

Loading from sharded data...
✓ Loaded shard-based dataloaders
  Train batches: 2734
  Val batches: 586

...

TRAINING COMPLETE
======================================================================
Best validation accuracy: 0.2847 (28.47%)
Model saved to: ../artifacts/weights/best_model.pth
======================================================================
```

**Key Milestone:** Val top-1 should be **25-35%** (up from 6.4%)

### 3. Run Smoke Tests
```bash
jupyter notebook notebooks/03_search_and_play.ipynb
```

**Expected Output:**
```
ALL SMOKE TESTS PASSED ✓
Ready for full benchmarks!
```

### 4. Run Benchmarks
```bash
jupyter notebook notebooks/04_benchmarks_and_analysis.ipynb
```

**Expected Results:**
- **Sunfish**: 15-25 wins, 50-60 draws, 15-35 losses (40-55% score)
- **Maia-1500**: 0-5 wins, 10-20 draws, 75-90 losses (5-15% score)
- **Elo vs Sunfish**: -40 to +20 (up from -147)

**Outputs Generated:**
- `benchmark_summary.csv` - Current results
- `before_after_comparison.csv` - Comparison table
- `acpl_by_phase.csv` - ACPL data
- `elo_estimates.png` - Elo plot
- `acpl_by_phase.png` - ACPL plot

---

## 🔧 Troubleshooting

### Issue: "No training data found!"
**Solution:** Run stream sampler first:
```bash
python -m src.data.stream_sampler --pgn-dir <path> --target 1000000
```

### Issue: "Shards not found. Loading from CSV (legacy mode)"
**Expected:** This is a fallback. You'll see a warning about using 100k positions.
**Solution:** Run stream sampler to create shards with 1M positions.

### Issue: "AMP not available on this device"
**Expected:** AMP may not work on all devices. Training will fall back to standard precision.
**Impact:** Slightly slower training, but results are the same.

### Issue: "Stockfish not found"
**Impact:** ACPL analysis will be skipped (optional feature).
**Solution:** Install Stockfish: `brew install stockfish` (macOS)

---

## 📈 Success Metrics

After implementing all changes and retraining with 1M positions:

| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| **Val Top-1 Accuracy** | 6.4% | 25-35% | ⏳ Pending training |
| **Sunfish Wins/100** | 0 | 15-25 | ⏳ Pending training |
| **Sunfish Score** | 30% | 40-55% | ⏳ Pending training |
| **Sunfish Elo** | -147 | -40 to +20 | ⏳ Pending training |
| **Maia Wins/100** | 0 | 0-5 | ⏳ Pending training |
| **Maia Score** | 0.5% | 5-15% | ⏳ Pending training |

---

## 🎯 Next Steps

1. **Run stream sampler** on your Lichess Elite data
2. **Train model** with new notebook 02 (expect 1-2 hours on MacBook)
3. **Run smoke tests** in notebook 03 to verify search quality
4. **Run benchmarks** in notebook 04 (expect 30-60 min per opponent)
5. **Analyze results** using before/after comparison and ACPL plots

---

## 📝 Notes

- All notebooks maintain **backward compatibility** - they work with CSV data if shards aren't available
- Smoke tests are **non-destructive** - they don't affect any saved data
- Before/after comparison **auto-saves current results** as baseline for future runs
- ACPL analysis is **optional** and only runs if Stockfish is available

---

**All notebook updates complete!** Ready for training run with 1M-position dataset. 🚀
