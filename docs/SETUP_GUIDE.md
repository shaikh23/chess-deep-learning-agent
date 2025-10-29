# Setup Guide - Chess Deep Learning Agent

Complete setup instructions for getting started with the Chess AI project.

## Prerequisites

- Python 3.11+
- 8GB+ RAM
- Optional: Apple Silicon Mac with MPS support or CUDA GPU

## Quick Start (10 minutes)

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd "Chess app"

# Option A: pip (recommended for Mac)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate chess-dl-agent
```

### Step 2: Verify Setup (1 minute)

```bash
python verify_setup.py
```

You should see:
```
✓ Python version: 3.11.x
✓ PyTorch
✓ python-chess
✓ NumPy
✓ All checks passed! You're ready to go.
```

**Troubleshooting**:
- If MPS not available: OK, will use CPU (slower but works)
- If Stockfish not found: Optional, install with `brew install stockfish` (macOS)

### Step 3: Obtain Training Data (3 minutes)

#### Option A: Download Lichess Data (Recommended)

1. Go to [https://database.lichess.org/](https://database.lichess.org/)
2. Download a monthly PGN file (e.g., `lichess_db_standard_rated_2024-01.pgn.zst`)
3. Extract: `zstd -d lichess_db_standard_rated_2024-01.pgn.zst`
4. Place in `data/raw/` directory:
   ```bash
   mkdir -p data/raw
   mv lichess_db_standard_rated_2024-01.pgn data/raw/
   ```

#### Option B: Use Synthetic Data (Quick Demo)

The notebooks will generate synthetic data if no PGN is found. Good for testing, not for real training.

### Step 4: Run Notebooks (30-60 minutes)

```bash
jupyter notebook
```

Run in order:

1. **`01_eda_and_preprocessing.ipynb`** (5-10 mins)
   - Update `PGN_PATH` to your data file
   - Run all cells
   - Output: Train/val/test splits in `artifacts/data/`

2. **`02_train_supervised.ipynb`** (20-30 mins)
   - Adjust `CONFIG` if needed (default is good)
   - Run all cells
   - Output: Trained model in `artifacts/weights/best_model.pth`

3. **`03_search_and_play.ipynb`** (5 mins)
   - Run all cells to test search and play a quick game
   - Output: Verified working agent

4. **`04_benchmarks_and_analysis.ipynb`** (30-60 mins)
   - Set `NUM_GAMES = 100` (or more for better stats)
   - Run all cells
   - Output: Match results, PGNs, plots

## Installation Details

### Required Dependencies

Core packages (from requirements.txt):
- `torch>=2.0.0` - Deep learning framework
- `python-chess>=1.999` - Chess rules and board representation
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Plotting
- `jupyter>=1.0.0` - Interactive notebooks

### Optional Dependencies

For benchmarking:
- **Stockfish**: `brew install stockfish` (macOS) or download from [stockfishchess.org](https://stockfishchess.org/download/)
- **Lc0** (for Maia): `brew install lc0` (macOS)
- **Maia weights**: Download from [maiachess.com](https://maiachess.com/)

### Configuration

Default settings work for most users. See Configuration Options section for customization.

## Expected Timeline

| Task | Time | Output |
|------|------|--------|
| Setup | 5 mins | Environment ready |
| Download data | 5 mins | PGN file |
| EDA notebook | 10 mins | Data splits |
| Training notebook | 30 mins | Trained model |
| Search notebook | 5 mins | Verified agent |
| Benchmark notebook | 60 mins | Match results |
| **Total** | **~2 hours** | Complete project |

**Note**: Times assume MacBook M1/M2/M3. CPU training takes 2-3× longer.

## Configuration Options

### Training (`02_train_supervised.ipynb`)

```python
CONFIG = {
    'model_type': 'miniresnet',  # or 'mlp', 'cnn'
    'num_blocks': 6,             # ResNet depth (4-10)
    'channels': 64,              # Width (32-128)
    'num_epochs': 10,            # Training epochs (5-20)
    'batch_size': 256,           # Batch size (128-512)
}
```

**Tip**: If out of memory, reduce `batch_size` to 128 or `channels` to 32.

### Search (`03_search_and_play.ipynb`)

```python
SearchConfig(
    max_depth=3,           # Search depth (1-5)
    time_limit=0.2,        # Time per move (0.1-1.0)
    use_policy_ordering=True,
    use_value_eval=True,
)
```

**Tip**: Increase `max_depth` to 4-5 for stronger play (slower).

### Benchmarks (`04_benchmarks_and_analysis.ipynb`)

```python
NUM_GAMES = 100  # Games per opponent (50-300)
```

**Tip**: 100 games give reasonable Elo estimates; 300+ for publication-quality results.

## Troubleshooting

### "MPS not available"
→ **OK**: Will use CPU, slightly slower but works fine.

### "Stockfish not found"
→ **Optional**: Install with `brew install stockfish` (macOS) or skip Stockfish matches.

### "Out of memory"
→ **Reduce** `batch_size` to 128 or `channels` to 32 in training config.

### "Import error: No module named 'src'"
→ **Fix**: Run cells that add `sys.path.append('../src')` first.

### "PGN file not found"
→ **Option 1**: Download from Lichess (see Step 3)
→ **Option 2**: Use synthetic data (automatic fallback in notebook 01)

### "Training is slow"
→ **Expected** on CPU. Reduce `num_epochs` to 5 or use smaller model (`channels=32`).

## Quick Test (Without Training)

Want to test the code without training?

```python
# In a Python shell or notebook
import sys
sys.path.append('src')

from utils.encoding import board_to_tensor
import chess

board = chess.Board()
tensor = board_to_tensor(board)
print(f"Board tensor shape: {tensor.shape}")  # Should be (12, 8, 8)

from model.nets import MiniResNetPolicyValue
model = MiniResNetPolicyValue(num_blocks=2, channels=32)
print(f"Model parameters: {model.count_parameters():,}")  # Should be ~100k

from search.alphabeta import AlphaBetaSearcher, SearchConfig
import torch
device = torch.device('cpu')
model.eval()
config = SearchConfig(max_depth=2, time_limit=0.1)
searcher = AlphaBetaSearcher(model, device, config)
move, score = searcher.search(board)
print(f"Best move: {move}")  # Should return a legal move
```

If all of the above runs without errors, your setup is working!

## Next Steps

After setup is complete:

1. **Run all notebooks** in order (1-4)
2. **Experiment with hyperparameters**: Try deeper models, more epochs
3. **Collect more data**: Use 500k-1M positions for stronger agent
4. **Review results**: Check `artifacts/` and `reports/` directories

## Getting Help

- **Main documentation**: See [README.md](../README.md)
- **Project overview**: See [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)
- **Troubleshooting**: Run `python verify_setup.py` to diagnose issues

---

Happy training! ♟️🤖
