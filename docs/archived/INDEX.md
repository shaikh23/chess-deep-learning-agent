# Documentation Index

Quick navigation to all project documentation.

---

## 🚀 Start Here

| Document | Purpose | Time |
|----------|---------|------|
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | High-level overview and status | 5 min |
| **[QUICKSTART.md](QUICKSTART.md)** | Get up and running fast | 10 min |
| **[USE_YOUR_DATA.md](USE_YOUR_DATA.md)** | ✨ How to use your Lichess Elite Database | 5 min |
| **[README.md](README.md)** | Complete documentation | 30 min |

---

## 📖 Documentation

### For First-Time Setup
1. [QUICKSTART.md](QUICKSTART.md) - 10-minute setup guide
2. Run `python verify_setup.py` - Verify installation
3. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - If you encounter issues

### For Understanding the Project
1. [README.md](README.md) - Comprehensive guide
   - Project structure
   - Installation instructions
   - Usage examples
   - Configuration options
   - Expected results
2. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Quick overview
   - What's included
   - Key features
   - Status and verification

### For Resolving Issues
1. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common problems and solutions
   - Installation issues
   - PyTorch configuration
   - Data problems
   - Training issues
   - Search and benchmarking
   - Performance optimization

### For Presentation
1. [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) - 15-minute demo outline
   - Slide structure
   - Talking points
   - Timing allocation
   - Visual recommendations
   - Q&A preparation

---

## 💻 Code

### Jupyter Notebooks (Run in Order)
1. [notebooks/01_eda_and_preprocessing.ipynb](notebooks/01_eda_and_preprocessing.ipynb)
   - Load and explore Lichess data
   - Create stratified train/val/test splits
   - ~10 minutes

2. [notebooks/02_train_supervised.ipynb](notebooks/02_train_supervised.ipynb)
   - Train policy + value network
   - Monitor training curves
   - ~30 minutes

3. [notebooks/03_search_and_play.ipynb](notebooks/03_search_and_play.ipynb)
   - Load trained model
   - Test alpha-beta and MCTS search
   - Play sample games
   - ~5 minutes

4. [notebooks/04_benchmarks_and_analysis.ipynb](notebooks/04_benchmarks_and_analysis.ipynb)
   - Run matches vs Sunfish and Stockfish
   - Compute Elo and ACPL statistics
   - Visualize results
   - ~60 minutes

### Source Code Modules

#### Data Processing (`src/data/`)
- [pgn_to_positions.py](src/data/pgn_to_positions.py) - Extract positions from PGN
- [dataset.py](src/data/dataset.py) - PyTorch Dataset and DataLoader
- [sampling.py](src/data/sampling.py) - Stratified sampling utilities

#### Models (`src/model/`)
- [nets.py](src/model/nets.py) - Neural network architectures
  - MLPPolicy
  - CNNPolicyValue
  - MiniResNetPolicyValue
- [loss.py](src/model/loss.py) - Loss functions
  - PolicyValueLoss
  - LabelSmoothingCrossEntropy

#### Search (`src/search/`)
- [alphabeta.py](src/search/alphabeta.py) - Alpha-beta pruning with policy ordering
- [mcts_lite.py](src/search/mcts_lite.py) - Lightweight MCTS

#### Game Play (`src/play/`)
- [engine_wrapper.py](src/play/engine_wrapper.py) - Neural agent wrapper
- [stockfish_wrapper.py](src/play/stockfish_wrapper.py) - Stockfish integration
- [sunfish_wrapper.py](src/play/sunfish_wrapper.py) - Sunfish baseline
- [match_runner.py](src/play/match_runner.py) - Tournament runner

#### Utilities (`src/utils/`)
- [encoding.py](src/utils/encoding.py) - Board/move encoding
- [metrics.py](src/utils/metrics.py) - Elo, ACPL, accuracy
- [plotting.py](src/utils/plotting.py) - Visualization
- [seeds.py](src/utils/seeds.py) - Reproducibility

---

## ⚙️ Configuration

| File | Purpose |
|------|---------|
| [requirements.txt](requirements.txt) | Python dependencies (pip) |
| [environment.yml](environment.yml) | Conda environment |
| [.gitignore](.gitignore) | Git ignore rules |
| [verify_setup.py](verify_setup.py) | Installation verification |

---

## 📊 Expected Outputs

### After Training (Notebook 02)
- `artifacts/weights/best_model.pth` - Trained model
- `artifacts/logs/training_log.json` - Training metrics
- `reports/figures/training_curves.png` - Loss and accuracy plots

### After Benchmarking (Notebook 04)
- `artifacts/matches/*.pgn` - Game records
- `artifacts/matches/*.json` - Match statistics
- `reports/figures/match_results.png` - Results bar chart
- `reports/figures/acpl_by_phase.png` - ACPL analysis

---

## 🔍 Quick Reference

### Installation
```bash
pip install -r requirements.txt
python verify_setup.py
```

### Running Notebooks
```bash
jupyter notebook
# Run notebooks 01 → 02 → 03 → 04
```

### Testing Code
```python
# Test encoding
import sys
sys.path.append('src')
from utils.encoding import board_to_tensor
import chess
board = chess.Board()
tensor = board_to_tensor(board)
print(f"Board shape: {tensor.shape}")  # Should be (12, 8, 8)
```

### Common Commands
```bash
# Verify setup
python verify_setup.py

# Check PyTorch device
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Check Stockfish
which stockfish

# List artifacts
ls -la artifacts/weights/ artifacts/matches/
```

---

## 📁 Directory Structure

```
chess-dl-agent/
│
├── 📘 Documentation (you are here)
│   ├── INDEX.md (this file)
│   ├── PROJECT_SUMMARY.md
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── TROUBLESHOOTING.md
│   └── VIDEO_SCRIPT.md
│
├── 📓 Notebooks
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_train_supervised.ipynb
│   ├── 03_search_and_play.ipynb
│   └── 04_benchmarks_and_analysis.ipynb
│
├── 💻 Source Code
│   ├── data/
│   ├── model/
│   ├── search/
│   ├── play/
│   └── utils/
│
├── ⚙️ Configuration
│   ├── requirements.txt
│   ├── environment.yml
│   ├── .gitignore
│   └── verify_setup.py
│
└── 📂 Outputs (generated)
    ├── artifacts/
    │   ├── weights/
    │   ├── logs/
    │   └── matches/
    └── reports/
        └── figures/
```

---

## 🎯 Use Case Guide

### I want to...

**...get started quickly**
→ Read [QUICKSTART.md](QUICKSTART.md)

**...understand the project in depth**
→ Read [README.md](README.md)

**...fix an error**
→ Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**...prepare my presentation**
→ Follow [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md)

**...see what's included**
→ Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

**...verify my installation**
→ Run `python verify_setup.py`

**...train a model**
→ Run notebook 02

**...benchmark my agent**
→ Run notebook 04

**...understand the code**
→ Read docstrings in `src/` modules

**...customize hyperparameters**
→ Edit CONFIG dict in notebooks

**...publish results**
→ Check `artifacts/` and `reports/` directories

---

## 📊 Reading Order

### For Quick Start (30 minutes)
1. PROJECT_SUMMARY.md (5 min)
2. QUICKSTART.md (10 min)
3. Run verify_setup.py (2 min)
4. Skim notebook 01 (5 min)
5. Review expected results (5 min)

### For Deep Understanding (2 hours)
1. README.md - Complete guide (30 min)
2. Source code docstrings (30 min)
3. Run notebooks 01-04 (60 min)

### For Presentation Prep (1 hour)
1. VIDEO_SCRIPT.md (20 min)
2. Review training curves and match results (20 min)
3. Practice with slides (20 min)

---

## ✅ Verification Checklist

Before running notebooks:
- [ ] Python 3.11+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `python verify_setup.py` passes all checks
- [ ] Lichess PGN data downloaded (or using synthetic)
- [ ] Stockfish installed (optional but recommended)

Before presentation:
- [ ] Read VIDEO_SCRIPT.md
- [ ] Review training results
- [ ] Run benchmark matches (at least 50 games)
- [ ] Prepare slides with visuals
- [ ] Test demo notebook
- [ ] Anticipate Q&A questions

---

## 🆘 Getting Help

1. **Check documentation** (in this order):
   - QUICKSTART.md for setup
   - TROUBLESHOOTING.md for errors
   - README.md for details

2. **Run verification**:
   ```bash
   python verify_setup.py
   ```

3. **Check code examples**:
   - All modules have `if __name__ == "__main__"` examples
   - Notebooks have markdown explanations

4. **Review error messages**:
   - Most common issues covered in TROUBLESHOOTING.md
   - Error messages include helpful hints

---

## 📞 Quick Links

| Resource | Link |
|----------|------|
| Lichess Database | https://database.lichess.org/ |
| Stockfish Download | https://stockfishchess.org/download/ |
| PyTorch Docs | https://pytorch.org/docs/stable/ |
| python-chess Docs | https://python-chess.readthedocs.io/ |

---

## 🎓 Course Alignment

This project fulfills all requirements:

✅ **Deliverable 1**: Jupyter notebooks with EDA, training, results, conclusions
✅ **Deliverable 2**: Video demo outline (VIDEO_SCRIPT.md)
✅ **Deliverable 3**: GitHub-ready repository with README and reproducibility

**Plus**:
- Production-quality code
- Comprehensive documentation
- Troubleshooting guide
- Verification script
- Benchmarking against real engines

---

**Last Updated**: January 2025
**Project Status**: ✅ Complete and Ready
**Verification**: Run `python verify_setup.py`
