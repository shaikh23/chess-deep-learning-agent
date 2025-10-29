# Getting Started with Improvements

## 🎯 Goal
Improve chess agent from **~2/100 wins vs Sunfish** to **15-20/100 wins vs Sunfish**.

## ✅ What's Done (Ready to Use)

### 1. Enhanced Model Architecture
**File**: `src/model/nets.py`

The model now has:
- ✅ Policy head with masked softmax
- ✅ Value head with tanh activation [-1, 1]
- ✅ Clean API returning (logits, log_probs, value)
- ✅ MiniResNetPolicyValue with 6-8 blocks

**Test it now**:
```python
from src.model.nets import MiniResNetPolicyValue
import torch

model = MiniResNetPolicyValue(num_blocks=6, channels=64)
board = torch.randn(8, 12, 8, 8)  # Batch of 8 boards
mask = torch.randint(0, 2, (8, 4672), dtype=torch.bool)

logits, log_probs, value = model(board, mask, return_value=True, return_logprobs=True)
print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")  # Should be [-1, 1]
```

### 2. PolicyValueLoss with Label Smoothing
**File**: `src/model/loss.py`

Features:
- ✅ Combined policy + value loss
- ✅ Default λ=0.7 (value weight)
- ✅ Label smoothing ε=0.05
- ✅ Top-k accuracy functions

**Test it now**:
```python
from src.model.loss import PolicyValueLoss, topk_accuracy

criterion = PolicyValueLoss(value_weight=0.7, policy_smoothing=0.05)

# Dummy data
policy_logits = torch.randn(32, 4672)
value_pred = torch.randn(32, 1)
policy_targets = torch.randint(0, 4672, (32,))
value_targets = torch.randn(32)

loss, loss_dict = criterion(policy_logits, value_pred, policy_targets, value_targets)
print(loss_dict)

# Compute top-1 accuracy
acc = topk_accuracy(policy_logits, policy_targets, k=1)
print(f"Top-1 accuracy: {acc:.2%}")
```

### 3. Teacher-Value Labeling Tool
**File**: `src/tools/teacher_label_sf.py`

Generate value labels using Stockfish:
```bash
# Quick test with 1000 positions (~2 minutes)
python -m src.tools.teacher_label_sf \
  --input data/sample_fens.txt \
  --output data/teacher_test_1k.pt \
  --stockfish-path /usr/local/bin/stockfish \
  --depth 10 \
  --max-positions 1000

# Full run with 50k positions (~2 hours)
python -m src.tools.teacher_label_sf \
  --input data/train_positions.csv \
  --output data/teacher_50k.pt \
  --depth 10 \
  --max-positions 50000
```

**Output format**:
```python
import torch
data = torch.load('data/teacher_test_1k.pt')
print(data['fens'][:5])  # FENs
print(data['value_teacher'][:5])  # Values in [-1, 1]
print(data['metadata'])  # Depth, time, etc.
```

### 4. Configuration Presets
**File**: `configs/presets.yaml`

Three presets ready:
- **fast**: 250k pos, 5 epochs, 100ms moves (~2 hours)
- **full**: 1M pos, 15 epochs, 300ms moves (~10 hours)
- **experimental**: 2M pos, deeper search (requires good hardware)

**Load in Python**:
```python
import yaml

with open('configs/presets.yaml') as f:
    config = yaml.safe_load(f)

# Use fast preset
fast = config['fast']
print(f"Training epochs: {fast['training']['epochs']}")
print(f"Search depth: {fast['search']['max_depth']}")
print(f"Games to play: {fast['match']['games']}")
```

## 🔄 What's Next (Prioritized)

### Priority 1: Train with Value Head (THIS WEEK)

**Goal**: Get policy+value model working

**Steps**:
1. Edit `notebooks/02_train_supervised.ipynb`
2. Add this training loop:

```python
from src.model.nets import MiniResNetPolicyValue
from src.model.loss import PolicyValueLoss, topk_accuracy
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Create model
model = MiniResNetPolicyValue(num_blocks=6, channels=64)
model = model.to(device)

# Loss and optimizer
criterion = PolicyValueLoss(value_weight=0.7, policy_smoothing=0.05)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_policy_acc = 0

    for batch in train_loader:
        boards, legal_masks, policy_targets, value_targets = batch

        # Forward
        logits, _, value = model(boards, legal_masks, return_value=True)

        # Loss
        loss, loss_dict = criterion(logits, value, policy_targets, value_targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss_dict['total_loss']
        total_policy_acc += topk_accuracy(logits, policy_targets, k=1)

    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Loss: {total_loss/len(train_loader):.4f}")
    print(f"  Policy top-1: {total_policy_acc/len(train_loader):.2%}")
    print(f"  Value loss: {loss_dict['value_loss']:.4f}")
```

**Expected results**:
- Policy top-1: 40-50%
- Value MSE: ~0.15
- Training time: 2-4 hours for 250k positions

### Priority 2: Implement Search Improvements (NEXT WEEK)

**Goal**: Add TT, static eval, killer moves

**Files to create** (code in [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)):
1. `src/utils/tt.py` - Transposition table
2. `src/search/static_eval.py` - Material + PST evaluation
3. Update `src/search/alphabeta.py` - Add killer/history/quiescence

**Impact**: Depth 2→3+ plies in 300ms, better tactics

### Priority 3: Opening Book & Adjudication (WEEK 3)

**Goal**: Better opening play, faster benchmarks

1. Update `src/play/opening_book.py` with 20 quiet lines
2. Add adjudication to `src/play/match_runner.py`
3. Enable parallel workers

**Impact**: More varied games, faster evaluation

### Priority 4: Full Benchmarking (WEEK 4)

**Goal**: Measure improvements

1. Run 100-game matches
2. Compute Elo with 95% CI
3. Calculate ACPL by phase
4. Generate visualizations

## 📊 Progress Tracker

| Milestone | Target | Status | Notes |
|-----------|--------|--------|-------|
| **Phase 0: Foundation** |
| Model with value head | ✅ | DONE | `nets.py` updated |
| PolicyValueLoss | ✅ | DONE | `loss.py` updated |
| Teacher labeling tool | ✅ | DONE | `teacher_label_sf.py` |
| Presets config | ✅ | DONE | `presets.yaml` |
| **Phase 1: Training** |
| Train policy+value | 50% top-1 | TODO | Edit notebook |
| Generate 50k teacher labels | N/A | TODO | Run tool |
| Train with teacher | 0.15 MSE | TODO | Add to notebook |
| vs Sunfish | 5-8/100 | TODO | Benchmark |
| **Phase 2: Search** |
| Implement TT | N/A | TODO | `tt.py` |
| Implement static eval | N/A | TODO | `static_eval.py` |
| Update alpha-beta | Depth 3+ | TODO | `alphabeta.py` |
| vs Sunfish | 10-12/100 | TODO | Benchmark |
| **Phase 3: Polish** |
| Opening book | 20 lines | TODO | `opening_book.py` |
| Adjudication | N/A | TODO | `match_runner.py` |
| vs Sunfish | 15-18/100 | TODO | Benchmark |
| **Phase 4: Final** |
| Full benchmarks | 100 games | TODO | All opponents |
| ACPL analysis | <80 avg | TODO | Metrics |
| **Final** vs Sunfish | 15-20/100 | TODO | 🎯 |

## 🚀 Run This Right Now

### Option 1: Test the New Model (5 minutes)
```bash
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"
python -c "
from src.model.nets import MiniResNetPolicyValue
from src.model.loss import PolicyValueLoss
import torch

model = MiniResNetPolicyValue(num_blocks=6, channels=64)
print(f'✓ Model created: {model.count_parameters():,} parameters')

criterion = PolicyValueLoss(value_weight=0.7, policy_smoothing=0.05)
print(f'✓ Loss created: lambda={criterion.value_weight}, epsilon={criterion.policy_loss_fn.smoothing}')

print('✓ All imports working!')
"
```

### Option 2: Generate Teacher Labels (30 minutes)
```bash
# First, create a sample FEN file
head -1000 data/processed/train_positions.csv > data/sample_1000.csv

# Generate teacher labels
python -m src.tools.teacher_label_sf \
  --input data/sample_1000.csv \
  --output data/teacher_sample_1k.pt \
  --stockfish-path /usr/local/bin/stockfish \
  --depth 10

# Check output
python -c "
import torch
data = torch.load('data/teacher_sample_1k.pt')
import numpy as np
values = np.array(data['value_teacher'])
print(f'✓ Labeled {len(data[\"fens\"])} positions')
print(f'  Mean value: {values.mean():.3f}')
print(f'  Std value: {values.std():.3f}')
print(f'  Range: [{values.min():.3f}, {values.max():.3f}]')
"
```

### Option 3: Load a Preset (1 minute)
```bash
python -c "
import yaml
with open('configs/presets.yaml') as f:
    config = yaml.safe_load(f)

print('=== FAST PRESET ===')
print(f'Training: {config[\"fast\"][\"training\"][\"epochs\"]} epochs, {config[\"fast\"][\"training\"][\"num_positions\"]} positions')
print(f'Search: depth {config[\"fast\"][\"search\"][\"max_depth\"]}, {config[\"fast\"][\"search\"][\"movetime_ms\"]}ms')
print(f'Match: {config[\"fast\"][\"match\"][\"games\"]} games, {config[\"fast\"][\"match\"][\"workers\"]} workers')
"
```

## 📚 Documentation

- **IMPROVEMENTS_SUMMARY.md** - High-level overview
- **IMPLEMENTATION_COMPLETE.md** - Full code for all remaining files
- **MAIA_SETUP.md** - Maia installation and usage
- **QUICK_REFERENCE.md** - Command cheat sheet
- **README.md** - Main project documentation

## 💡 Tips

1. **Start Small**: Test with 1k positions before running 50k
2. **Use Presets**: `--preset fast` for iteration, `--preset full` for final
3. **Check Device**: Ensure MPS is available on Mac for faster training
4. **Monitor Metrics**: Track top-1 accuracy and value MSE
5. **Iterate**: Train → Benchmark → Analyze → Improve

## 🎯 Success Criteria

- ✅ **Minimum**: 15/100 wins vs Sunfish
- 🎯 **Target**: 18/100 wins vs Sunfish + 3/100 vs Maia-1500
- 🚀 **Stretch**: 20/100 wins vs Sunfish + 5/100 vs Maia-1500

## ❓ FAQ

**Q: Where do I start?**
A: Run the test model script above, then generate 1k teacher labels.

**Q: How long will Phase 1 take?**
A: ~4 hours total (2h teacher labels + 2h training).

**Q: Do I need a GPU?**
A: No, Mac MPS or CPU is fine. Training is manageable on MacBook.

**Q: What if I get import errors?**
A: Ensure you're in project root and Python path is correct.

**Q: Can I skip teacher distillation?**
A: Yes, but you'll get better results with it. Start without, add later.

---

**Next Action**: Run one of the three "Run This Right Now" options above! 🚀
