# Troubleshooting Guide

Common issues and their solutions.

---

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: `ModuleNotFoundError: No module named 'chess'`

**Solution**: Install python-chess
```bash
pip install python-chess
```

### Issue: `AttributeError: module 'chess' has no attribute 'pgn'`

**Solution**: Import chess.pgn explicitly
```python
import chess
import chess.pgn  # Must import separately
```

This has been fixed in all source files. If you still see this, make sure you're using python-chess version 1.999.

---

## PyTorch Issues

### Issue: "MPS backend not available"

**Expected on**: Intel Macs, non-Mac systems

**Solution**: The code automatically falls back to CPU. This is normal and okay.

```python
# Code handles this automatically
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

**Impact**: Training will be 2-3× slower on CPU vs MPS, but still works fine.

### Issue: "CUDA not available"

**Expected on**: Macs and systems without NVIDIA GPUs

**Solution**: Use CPU or MPS (Mac). No action needed.

### Issue: "Out of memory"

**Solutions**:

1. **Reduce batch size**:
   ```python
   CONFIG['batch_size'] = 128  # Instead of 256
   ```

2. **Use smaller model**:
   ```python
   CONFIG['channels'] = 32  # Instead of 64
   CONFIG['num_blocks'] = 4  # Instead of 6
   ```

3. **Reduce dataset size**:
   ```python
   TARGET_SIZE = 50_000  # Instead of 100_000
   ```

---

## Import Errors

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution**: Add src to Python path

In notebooks:
```python
import sys
sys.path.append('../src')  # Make sure this cell runs first
```

In Python scripts:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))
```

### Issue: `ImportError: cannot import name 'board_to_tensor'`

**Solution**: Check that utils/__init__.py exists and is properly configured

```bash
# Verify file exists
ls src/utils/__init__.py

# If missing, create it
touch src/utils/__init__.py
```

---

## Data Issues

### Issue: "PGN file not found"

**Solution**: Either download real data or use synthetic fallback

**Option 1**: Download from Lichess
```bash
# Download and extract
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
zstd -d lichess_db_standard_rated_2024-01.pgn.zst

# Move to data directory
mkdir -p data/raw
mv lichess_db_standard_rated_2024-01.pgn data/raw/
```

**Option 2**: Use synthetic data (automatic in notebook 01)

The notebook will automatically generate synthetic data if PGN not found.

### Issue: "No games extracted from PGN"

**Causes**:
1. PGN file is corrupted
2. No games meet Elo threshold (default: 1800+)
3. File format not recognized

**Solutions**:

1. **Lower Elo threshold**:
   ```python
   MIN_ELO = 1200  # Instead of 1800
   ```

2. **Check PGN format**:
   ```bash
   head -20 data/raw/your_file.pgn
   ```
   Should show headers like `[Event "..."]`

3. **Try smaller sample**:
   ```python
   MAX_GAMES = 1000  # Instead of 10000
   ```

---

## Training Issues

### Issue: Training is very slow

**Expected**: 10 epochs should take ~30 mins on M1 Mac, ~60-90 mins on CPU

**Solutions**:

1. **Verify device**:
   ```python
   import torch
   print(torch.backends.mps.is_available())  # Should be True on M1/M2/M3 Mac
   ```

2. **Reduce epochs**:
   ```python
   CONFIG['num_epochs'] = 5  # Quick test
   ```

3. **Use smaller dataset**:
   ```python
   TARGET_SIZE = 50_000
   ```

### Issue: Loss is NaN

**Causes**:
1. Learning rate too high
2. Gradient explosion
3. Bad data

**Solutions**:

1. **Lower learning rate**:
   ```python
   CONFIG['learning_rate'] = 0.0001  # Instead of 0.001
   ```

2. **Check gradient clipping** (already in code):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```

3. **Verify data**:
   ```python
   # Check for NaNs in dataset
   print(train_df.isnull().sum())
   ```

### Issue: Accuracy not improving

**Expected**: Policy top-1 accuracy plateaus around 45-50%

**This is normal**: Even experts only agree on the "best" move ~50% of the time. Top-5 accuracy should be ~70-75%.

**If accuracy is < 30%**:
1. Train longer (more epochs)
2. Increase model capacity (more channels/blocks)
3. Collect more data
4. Check data quality

---

## Search Issues

### Issue: "ValueError: Cannot search in game-over position"

**Solution**: Check board state before searching

```python
if not board.is_game_over():
    move = agent.get_move(board)
else:
    print("Game is over")
```

### Issue: Search takes too long

**Solutions**:

1. **Reduce depth**:
   ```python
   SearchConfig(max_depth=2, ...)  # Instead of 3
   ```

2. **Lower time limit**:
   ```python
   SearchConfig(time_limit=0.1, ...)  # Instead of 0.2
   ```

3. **Disable quiescence**:
   ```python
   SearchConfig(enable_quiescence=False, ...)
   ```

### Issue: Agent makes illegal moves

**This should never happen**: Legal move masking is built-in.

**If it does**:
1. Check that `legal_mask` is being applied
2. Verify move encoding/decoding
3. Report as bug

```python
# Debug
from utils.encoding import get_legal_move_mask
mask = get_legal_move_mask(board)
print(f"Legal moves: {mask.sum().item()}")
```

---

## Benchmarking Issues

### Issue: Stockfish not found

**Solutions**:

**macOS**:
```bash
brew install stockfish
```

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install stockfish
```

**Manual**:
1. Download from [stockfishchess.org](https://stockfishchess.org/download/)
2. Extract to a directory
3. Update path in notebook:
   ```python
   STOCKFISH_PATH = '/path/to/stockfish'
   ```

### Issue: "Stockfish analysis taking too long"

**Solution**: Reduce depth or skip ACPL analysis

```python
# In notebook 04
# Option 1: Lower depth
phase_acpl = compute_acpl_by_phase(game, STOCKFISH_PATH, depth=10)  # Instead of 15

# Option 2: Skip ACPL analysis
# Comment out the ACPL section
```

### Issue: Matches timeout

**Solution**: Increase time limits or reduce game count

```python
# Increase time per move
SearchConfig(time_limit=0.5, ...)  # Instead of 0.2

# Or reduce games
NUM_GAMES = 50  # Instead of 100
```

---

## Notebook Issues

### Issue: Jupyter kernel crashes

**Causes**:
1. Out of memory
2. Infinite loop
3. GPU/MPS issue

**Solutions**:

1. **Restart kernel**: Kernel → Restart
2. **Clear outputs**: Cell → All Output → Clear
3. **Reduce memory usage**: See "Out of memory" section above
4. **Use CPU**: `device = torch.device('cpu')`

### Issue: Cell takes forever to run

**Expected long-running cells**:
- Training (notebook 02): ~30 mins
- Benchmarking (notebook 04): ~60 mins for 100 games

**If unexpectedly slow**:
1. Check if cell has progress bar (tqdm)
2. Interrupt kernel: Kernel → Interrupt
3. Reduce problem size (fewer games, epochs, etc.)

### Issue: Plots not showing

**Solution**: Add matplotlib backend

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

---

## Verification Script Issues

### Issue: `python verify_setup.py` fails

**Solutions**:

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.11+
   ```

2. **Install dependencies first**:
   ```bash
   pip install -r requirements.txt
   python verify_setup.py
   ```

3. **Run with verbose output**:
   ```bash
   python verify_setup.py 2>&1 | tee verify.log
   ```

---

## Performance Optimization

### Slow Training

**Tips**:
1. Use MPS on Mac: Ensure `torch.backends.mps.is_available() == True`
2. Increase batch size (if memory allows): 256 → 512
3. Reduce data loading workers: Keep at 0 on Mac (multiprocessing issues)
4. Disable unnecessary features during training

### Slow Search

**Tips**:
1. Enable policy ordering: `use_policy_ordering=True` (default)
2. Lower search depth: 3 → 2
3. Reduce time limit: 0.2 → 0.1
4. Disable quiescence: `enable_quiescence=False` (default)

### Slow Matches

**Tips**:
1. Reduce games: 200 → 100 or 50
2. Use fixed depth instead of time: `SearchConfig(depth_limit=2)`
3. Run on faster hardware
4. Parallelize (requires code modification)

---

## Common Warnings (Safe to Ignore)

### ✅ "MPS not available, using CPU"
- **Impact**: Slower training
- **Action**: None if on Intel Mac or Linux

### ✅ "Stockfish not found"
- **Impact**: Can't run ACPL analysis or Stockfish matches
- **Action**: Install Stockfish or skip those sections

### ✅ "No PGN file found, using synthetic data"
- **Impact**: Won't train a useful model (synthetic data is random)
- **Action**: Download real Lichess data for actual training

### ✅ "Sampling with replacement"
- **Impact**: Some positions duplicated in training set
- **Action**: None if dataset is large enough; otherwise collect more data

---

## Getting More Help

1. **Check documentation**:
   - [README.md](README.md) - Comprehensive guide
   - [QUICKSTART.md](QUICKSTART.md) - Fast setup
   - [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) - Presentation guide

2. **Run verification**:
   ```bash
   python verify_setup.py
   ```

3. **Check source code**:
   - All modules have docstrings
   - Inline unit tests at bottom of each file
   - Example usage in `if __name__ == "__main__":` blocks

4. **Minimal test**:
   ```python
   # Test imports
   import sys
   sys.path.append('src')

   from utils.encoding import board_to_tensor
   import chess

   board = chess.Board()
   tensor = board_to_tensor(board)
   print(f"Success! Board tensor shape: {tensor.shape}")
   ```

5. **Report issues**:
   - Check error message carefully
   - Try minimal reproducible example
   - Include Python version, OS, and full error traceback

---

## Quick Diagnostic Checklist

Run these commands to diagnose most issues:

```bash
# 1. Python version
python --version

# 2. Verify setup
python verify_setup.py

# 3. Check imports
python -c "import torch; import chess; import chess.pgn; import pandas; print('All imports OK')"

# 4. Check PyTorch device
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"

# 5. Test encoding
python -c "import sys; sys.path.append('src'); from utils.encoding import board_to_tensor; import chess; print(board_to_tensor(chess.Board()).shape)"

# 6. Check directories
ls -la src/ notebooks/ artifacts/

# 7. Check Stockfish
which stockfish
```

If all of the above pass, your setup is correct!

---

## Still Stuck?

If you've tried everything above and still have issues:

1. **Start fresh**:
   ```bash
   # Remove virtual environment
   rm -rf venv/

   # Recreate
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

   # Verify
   python verify_setup.py
   ```

2. **Try minimal example**: Run the "Quick Test" from QUICKSTART.md

3. **Check system resources**:
   - Disk space: Need ~5GB for data + models
   - RAM: Need ~8GB for training
   - Python: 3.11 or 3.12 recommended

Good luck! 🚀
