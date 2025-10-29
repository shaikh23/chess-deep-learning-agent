# Chess-DL-Agent: Improvements Summary

## Goal
Improve from **~2/100 wins vs Sunfish** to **15-20/100 wins vs Sunfish**, with a few wins vs Maia-1500.

## Status: Foundation Complete ✅

### Completed (Copy-Paste Ready)

#### 1. Model Architecture ✅
**File**: `src/model/nets.py`
- ✅ Added `masked_log_softmax()` utility function
- ✅ Updated `MiniResNetPolicyValue.forward()` to return `(logits, log_probs, value)`
- ✅ Value head with `tanh` activation for [-1, 1] range
- ✅ Clean API: `forward(board, legal_mask, return_value, return_logprobs)`

**Test**:
```python
from src.model.nets import MiniResNetPolicyValue
model = MiniResNetPolicyValue(num_blocks=6, channels=64)
# Returns: (logits, log_probs, value)
```

#### 2. Loss Function ✅
**File**: `src/model/loss.py`
- ✅ `PolicyValueLoss` with default λ=0.7, ε=0.05
- ✅ `topk_accuracy()` for metrics
- ✅ `calibration_bins()` for confidence analysis
- ✅ Label smoothing cross-entropy

**Usage**:
```python
from src.model.loss import PolicyValueLoss, topk_accuracy
criterion = PolicyValueLoss(value_weight=0.7, policy_smoothing=0.05)
loss, loss_dict = criterion(policy_logits, value_pred, policy_targets, value_targets)
```

#### 3. Teacher-Value Distillation ✅
**File**: `src/tools/teacher_label_sf.py`
- ✅ Stockfish depth-10 labeling (configurable)
- ✅ Centipawn → value: `v = clip(cp/800, -1, 1)`
- ✅ Batch processing with tqdm progress
- ✅ Saves to `.pt` format

**Run**:
```bash
python -m src.tools.teacher_label_sf \
  --input data/positions.csv \
  --output data/teacher_labels_50k.pt \
  --stockfish-path /usr/local/bin/stockfish \
  --depth 10 \
  --max-positions 50000
```

#### 4. Configuration Presets ✅
**File**: `configs/presets.yaml`
- ✅ `fast`: 250k pos, 5 epochs, 100ms moves, ~2h total
- ✅ `full`: 1M pos, 15 epochs, 300ms moves, ~10h total
- ✅ `experimental`: 2M pos, deeper search
- ✅ All parameters documented

**Usage**: `--preset fast` or `--preset full`

### To Implement (Code Provided in IMPLEMENTATION_COMPLETE.md)

#### 5. Search Upgrades 🔄
**Files**: `src/utils/tt.py`, `src/search/alphabeta.py`, `src/search/static_eval.py`

**Features**:
- Transposition Table with Zobrist hashing
- Killer moves heuristic (per depth)
- History heuristic
- Quiescence search (captures/checks)
- Static evaluation (material + PST + mobility)
- Eval blending: `E = 0.8*V_net + 0.2*E_static`
- Iterative deepening

**Impact**: Depth 2→3+ in 300ms, better tactical awareness

#### 6. Opening Book 🔄
**File**: `src/play/opening_book.py`

**Features**:
- 20 quiet lines (Italian, London, Slav, French, Caro-Kann, etc.)
- `book_moves_uci(board, max_plies=8)` function
- SAN→UCI validation
- External book for both sides

**Impact**: Better opening play, more variety

#### 7. Enhanced Sampling 🔄
**File**: `src/data/sampling.py`

**Features**:
- Phase-stratified sampling (opening/middle/endgame)
- Balance side-to-move
- Board mirroring augmentation
- Teacher label merging: `v = alpha*v_teacher + (1-alpha)*v_result`

**Impact**: Better training data quality

#### 8. Metrics & Benchmarking 🔄
**File**: `src/utils/metrics.py`

**Features**:
- Elo estimation with Wilson 95% CI
- ACPL (Average Centipawn Loss) by phase
- Top-k accuracy tracking
- Calibration analysis

**Impact**: Better evaluation and diagnostics

#### 9. Sanity Checks 🔄
**File**: `src/tests/sanity_checks.py`

**Tests**:
- Legal mask: top-1 always legal (10k positions)
- Value bounds: endgames near expected
- Search depth histogram under 300ms
- TT hit rate >20%
- Desync guard

**Impact**: Catch bugs early

#### 10. Match Runner Upgrades 🔄
**File**: `src/play/match_runner.py`

**Features**:
- Adjudication (resign at ±5.0 for 3 plies)
- Parallel matches (`--workers`)
- Per-game PGN + compact JSON
- CLI with all toggles

**Impact**: Faster benchmarking

## Implementation Roadmap

### Phase 1: Core Improvements (Week 1) 🎯
Priority: Teacher distillation + value head training

1. **Generate teacher labels** (2 hours)
   ```bash
   python -m src.tools.teacher_label_sf \
     --input data/train.csv \
     --output data/teacher_50k.pt \
     --depth 10 \
     --max-positions 50000
   ```

2. **Train with value head** (4 hours)
   - Edit `notebooks/02_train_supervised.ipynb`
   - Add teacher distillation section
   - Train with `PolicyValueLoss`
   - Target: policy top-1 45%→50%, value MSE ~0.15

**Expected**: 2/100 → 5-8/100 wins vs Sunfish

### Phase 2: Search & Static Eval (Week 2) 🔍
Priority: Transposition table + static eval

1. **Implement TT** (`src/utils/tt.py`)
2. **Implement static eval** (`src/search/static_eval.py`)
3. **Update alpha-beta** with TT + killer + history
4. **Add quiescence search**

**Expected**: 5-8/100 → 10-12/100 wins vs Sunfish

### Phase 3: Opening Book & Polish (Week 3) 📚
Priority: Opening book + adjudication

1. **Expand opening book** (20 quiet lines)
2. **Add adjudication** to match runner
3. **Implement parallel workers**
4. **Phase-stratified sampling**

**Expected**: 10-12/100 → 15-18/100 wins vs Sunfish

### Phase 4: Full Benchmarking (Week 4) 📊
Priority: Comprehensive evaluation

1. **Run 100-game matches** vs Sunfish, Maia, Stockfish
2. **Compute ACPL by phase**
3. **Generate Elo estimates** with 95% CI
4. **Create visualizations**

**Expected**: Final 15-20/100 wins vs Sunfish, 3-5/100 vs Maia-1500

## Quick Start (Right Now)

### Option 1: Generate Teacher Labels (Immediate)
```bash
# Create a sample of 10k positions for quick test
python -m src.tools.teacher_label_sf \
  --input data/processed/train_positions.csv \
  --output data/processed/teacher_test_10k.pt \
  --stockfish-path /usr/local/bin/stockfish \
  --depth 10 \
  --max-positions 10000 \
  --batch-size 500

# Should take ~30 minutes
```

### Option 2: Test Model Updates (Immediate)
```python
import torch
from src.model.nets import MiniResNetPolicyValue
from src.model.loss import PolicyValueLoss

# Create model
model = MiniResNetPolicyValue(num_blocks=6, channels=64)
print(f"Parameters: {model.count_parameters():,}")

# Test forward pass
board = torch.randn(8, 12, 8, 8)
legal_mask = torch.randint(0, 2, (8, 4672), dtype=torch.bool)

logits, log_probs, value = model(board, legal_mask, return_value=True, return_logprobs=True)

print(f"Logits: {logits.shape}")
print(f"Log probs: {log_probs.shape}")
print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")

# Test loss
criterion = PolicyValueLoss(value_weight=0.7, policy_smoothing=0.05)
targets = torch.randint(0, 4672, (8,))
value_targets = torch.randn(8)

loss, loss_dict = criterion(logits, value, targets, value_targets)
print(f"Loss: {loss_dict}")
```

### Option 3: Use Presets (Immediate)
```bash
# Load preset in Python
import yaml

with open('configs/presets.yaml') as f:
    config = yaml.safe_load(f)

fast_config = config['fast']
print(fast_config['training'])
print(fast_config['search'])
```

## Key Metrics to Track

| Metric | Baseline | Target | Current |
|--------|----------|--------|---------|
| **vs Sunfish (100g)** |
| Wins | 2 | 15-20 | TBD |
| Score | 2.5/100 | 20/100 | TBD |
| **vs Maia-1500 (100g)** |
| Wins | 0 | 3-5 | TBD |
| Score | 10/100 | 15/100 | TBD |
| **Training Metrics** |
| Policy top-1 | 35% | 50% | TBD |
| Policy top-3 | 55% | 70% | TBD |
| Value MSE | N/A | 0.15 | TBD |
| **Search Metrics** |
| Avg depth (300ms) | 2 | 3-4 | TBD |
| Nodes/sec | 5k | 10k+ | TBD |
| TT hit rate | 0% | 20%+ | TBD |
| **Quality Metrics** |
| ACPL Opening | N/A | <60 | TBD |
| ACPL Middle | N/A | <90 | TBD |
| ACPL Endgame | N/A | <70 | TBD |

## Files Reference

### Implemented ✅
- `src/model/nets.py` - Value head, log-probs
- `src/model/loss.py` - PolicyValueLoss, metrics
- `src/tools/teacher_label_sf.py` - Teacher labeling
- `configs/presets.yaml` - Fast/full presets

### To Implement (Code in IMPLEMENTATION_COMPLETE.md) 🔄
- `src/utils/tt.py` - Transposition table
- `src/search/static_eval.py` - Static evaluation
- `src/search/alphabeta.py` - Killer, history, quiescence
- `src/play/opening_book.py` - 20 quiet lines
- `src/data/sampling.py` - Phase stratification
- `src/utils/metrics.py` - Elo, ACPL
- `src/tests/sanity_checks.py` - Comprehensive tests

### To Update 📝
- `notebooks/02_train_supervised.ipynb` - Teacher section
- `notebooks/03_search_and_play.ipynb` - Smoke tests
- `notebooks/04_benchmarks_and_analysis.ipynb` - ACPL plots
- `notebooks/00_report.ipynb` - Before/after tables

## Documentation

- **IMPLEMENTATION_COMPLETE.md** - Full code for all remaining files
- **MAIA_SETUP.md** - Maia installation and usage
- **QUICK_REFERENCE.md** - Command cheat sheet
- **README.md** - Main project docs

## Next Steps

1. **Today**: Generate 10k teacher labels for testing
2. **This Week**: Implement Phase 1 (teacher + value training)
3. **Next Week**: Implement Phase 2 (search improvements)
4. **Week 3**: Implement Phase 3 (opening book)
5. **Week 4**: Full benchmarking and analysis

## Success Criteria

✅ **Minimum Success**: 15/100 wins vs Sunfish
🎯 **Target Success**: 18/100 wins vs Sunfish + 3/100 vs Maia
🚀 **Stretch Goal**: 20/100 wins vs Sunfish + 5/100 vs Maia

---

**Status**: Foundation complete, ready for implementation
**Time to Target**: 3-4 weeks with focused work
**Confidence**: High (proven techniques, solid foundation)
