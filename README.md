# Chess Deep Learning Agent

A PyTorch-based chess playing agent using policy and value networks with lightweight search (alpha-beta pruning or MCTS). Trained via supervised learning from Lichess games and benchmarked against existing engines.

---

## Features

- **Neural Networks**: MLP, CNN, and Mini-ResNet architectures with policy + value heads
- **Search Algorithms**: Alpha-beta pruning with policy-based move ordering; lightweight MCTS
- **Training**: Supervised learning from Lichess PGN data with phase stratification
- **Benchmarking**: Head-to-head matches against Sunfish and Stockfish with PGN logging
- **Analysis**: Elo estimation, ACPL (Average Centipawn Loss), confidence intervals
- **Mac-Optimized**: Runs locally on MacBook with MPS (Metal Performance Shaders) support

---


---

## Requirements

### Hardware
- **Recommended**: MacBook with Apple Silicon (M1/M2/M3) for MPS acceleration
- **Minimum**: Any modern CPU (training will be slower)

### Software
- Python 3.11+
- PyTorch 2.2+ (with MPS support on Mac)
- Stockfish (optional, for ACPL analysis and benchmarking)

---

## Installation

### Option 1: pip (Recommended for Mac)

```bash
# Clone or download the repository
cd "Chess app"

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda

```bash
# Create conda environment
conda env create -f environment.yml
conda activate chess-dl-agent
```

### Install Engines (Optional but Recommended for Benchmarking)

#### Stockfish
```bash
# macOS (Homebrew)
brew install stockfish

# Linux (Debian/Ubuntu)
sudo apt-get install stockfish

# Or download from: https://stockfishchess.org/download/
```

#### Lc0 (for Maia)
```bash
# macOS (Homebrew)
brew install lc0

# Linux: Download from https://github.com/LeelaChessZero/lc0/releases
```

#### Maia Weights
Download human-like neural network weights from [https://maiachess.com/](https://maiachess.com/)
- Available rating levels: 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900
- Download `.pb.gz` files (e.g., `maia1500.pb.gz`) to `weights/` directory

**Your Installation:**
- ✓ Lc0: `/usr/local/bin/lc0`
- ✓ Maia-1500: `weights/maia-1500.pb.gz`
- See [MAIA_SETUP.md](MAIA_SETUP.md) for detailed usage instructions

---


### 1. Obtain Training Data

**Lichess Open Database**: [https://database.lichess.org/](https://database.lichess.org/)

1. Download PGN files (e.g., from Lichess Elite or monthly databases)
2. Extract if compressed: `zstd -d lichess_db_standard_rated_2024-01.pgn.zst`
3. Place in `data/raw/` directory (create if needed)
4. **Important**: Add `data/raw/` to `.gitignore` (large files)

**Licensing**: Lichess data is public domain (CC0). Cite source in publications.

### 2. Stream Sample Positions (~1M)

Use the stream sampler to extract and shard positions from PGN files:

```bash
# Sample 1M positions from PGN files (full mode)
python -m src.data.stream_sampler \
  --pgn-dir data/raw/Lichess\ Elite\ Database \
  --target 1000000 \
  --output artifacts/data/shards \
  --min-elo 2000 \
  --min-moves 20 \
  --min-time 300

# Quick mode (250k positions for fast iteration)
python -m src.data.stream_sampler \
  --pgn-dir data/raw/Lichess\ Elite\ Database \
  --target 250000 \
  --output artifacts/data/shards_fast \
  --min-elo 2000 \
  --min-moves 15
```

The sampler will:
- Filter games (rated, min Elo 2000, min 20 moves, 5+ minute time control)
- Skip first 8 plies (opening book moves)
- Balance white/black 50/50
- Tag phases (opening/middle/endgame)
- Save sharded `.pt` files (50k positions each)
- Generate `summary.json` with statistics

Expected output:
```
SAMPLING COMPLETE
Total positions: 1,000,000
Shards created: 20
Phase distribution:
  Opening:    250,000 ( 25.0%)
  Middlegame: 500,000 ( 50.0%)
  Endgame:    250,000 ( 25.0%)
```

### 3. (Optional) Generate Teacher Labels with Stockfish

For higher-quality value targets, label positions with Stockfish:

```bash
python -m src.tools.teacher_label_stockfish \
  --in artifacts/data/shards \
  --out artifacts/data/teacher \
  --sf /opt/homebrew/bin/stockfish \
  --depth 10 \
  --max-positions 100000
```

**Note**: This is time-consuming (~1-2 hours for 100k positions). Skip for initial training.

### 4. Train Model

Open and run the training notebook:

```bash
jupyter notebook
```

Open and run:

1. **01_eda_and_preprocessing.ipynb**
   - Load PGN data and extract positions
   - Perform EDA (phase distribution, outcomes, piece counts)
   - Create stratified train/val/test splits

2. **02_train_supervised.ipynb**
   - Train MiniResNet policy + value network
   - ~10 epochs, batch size 256
   - Achieves ~45-50% top-1 policy accuracy
   - Saves best model to `artifacts/weights/best_model.pth`

3. **03_search_and_play.ipynb**
   - Load trained model
   - Test alpha-beta and MCTS search
   - Play quick test games
   - Verify policy ordering improves search efficiency

4. **04_benchmarks_and_analysis.ipynb**
   - Run 100-300 games vs Sunfish, Stockfish, and Maia
   - Compute Elo differences with 95% CI
   - Analyze ACPL by game phase
   - Visualize results and sample games

### 3. Expected Results

With default configuration (MiniResNet 6 blocks × 64 channels, 100k training positions):

- **Training**: Top-1 policy accuracy ~45-50%, value MSE ~0.15
- **vs Sunfish (depth 2)**: Win rate ~70-80%, Elo difference +150 to +250
- **vs Stockfish (skill 5)**: Win rate ~30-40%, Elo difference -100 to +50
- **vs Maia-1500**: Competitive against human-like play (~1500 rating level)
- **ACPL**: Opening ~50, Middlegame ~80, Endgame ~65 (lower is better)

---

## Acknowledgments

- **Lichess**: For providing open game data
- **AlphaZero**: Inspiration for architecture and search
- **Sunfish**: Simple baseline engine
- **Stockfish**: Analysis and benchmarking
- **PyTorch**: Deep learning framework
- **python-chess**: Chess logic and PGN handling

---

## Video Demo

See [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) for presentation outline and talking points for the 5-15 minute video demo.
