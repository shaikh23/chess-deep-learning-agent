# Chess Deep Learning Agent

A PyTorch-based chess playing agent using policy and value networks with lightweight search (alpha-beta pruning or MCTS). Trained via supervised learning from Lichess games and benchmarked against existing engines.

**Course Project**: Deep Learning (University of Colorado Boulder)

---

## Features

- **Neural Networks**: MLP, CNN, and Mini-ResNet architectures with policy + value heads
- **Search Algorithms**: Alpha-beta pruning with policy-based move ordering; lightweight MCTS
- **Training**: Supervised learning from Lichess PGN data with phase stratification
- **Benchmarking**: Head-to-head matches against Sunfish and Stockfish with PGN logging
- **Analysis**: Elo estimation, ACPL (Average Centipawn Loss), confidence intervals
- **Mac-Optimized**: Runs locally on MacBook with MPS (Metal Performance Shaders) support

---

## Project Structure

```
chess-dl-agent/
├── README.md                          # This file
├── PROJECT_SUMMARY.md                 # High-level project overview
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── verify_setup.py                    # Setup verification script
│
├── docs/                              # Documentation
│   ├── README.md                      # Documentation index
│   ├── SETUP_GUIDE.md                 # Complete setup instructions
│   ├── QUICKSTART.md                  # Quick start guide
│   ├── TROUBLESHOOTING.md             # Common issues and solutions
│   ├── USE_YOUR_DATA.md               # Custom dataset guide
│   └── archived/                      # Historical documentation
│
├── presentation/                      # Presentation materials
│   ├── Chess_AI_Presentation.pptx     # PowerPoint slides
│   ├── PRESENTATION_SCRIPT.md         # Speaking notes and script
│   ├── VIDEO_SCRIPT.md                # Video demo outline
│   └── create_presentation.py         # Presentation generation script
│
├── notebooks/                         # Jupyter notebooks (run in order)
│   ├── 01_eda_and_preprocessing.ipynb # Data exploration and preparation
│   ├── 02_train_supervised.ipynb      # Model training
│   ├── 03_search_and_play.ipynb       # Search algorithms and gameplay
│   ├── 04_benchmarks_and_analysis.ipynb # Match evaluation
│   └── 00_report_submission.ipynb     # Final comprehensive report
│
├── src/                               # Source code
│   ├── data/                          # PGN parsing, dataset, sampling
│   ├── model/                         # Neural network architectures and loss
│   ├── search/                        # Alpha-beta and MCTS implementations
│   ├── play/                          # Engine wrappers and match runner
│   └── utils/                         # Encoding, metrics, plotting, seeds
│
├── config/                            # Configuration files
│   ├── engines.py                     # Engine paths and settings
│   ├── presets.yaml                   # Training presets
│   └── run_maia_benchmark.sh          # Benchmark script
│
├── artifacts/                         # Generated outputs
│   ├── weights/                       # Saved model checkpoints
│   ├── data/                          # Processed training data
│   ├── logs/                          # Training logs (JSON)
│   ├── matches/                       # PGN game records
│   └── maia-1500.pb.gz                # Maia neural network weights
│
├── reports/                           # Analysis outputs
│   ├── figures/                       # Plots and visualizations
│   └── final/                         # Final report PDFs
│
├── backups/                           # Data backups
│   ├── shards_backup.tar.gz           # Training data backup
│   └── src_code.tar.gz                # Source code archive
│
└── data/
    └── raw/                           # Raw Lichess PGN files
```

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

## Quick Start

### 0. Verify Engine Setup

```bash
# Check all engines are available
python config/engines.py

# Test Maia installation
python -m src.play.sanity \
  --lc0-path /usr/local/bin/lc0 \
  --maia-weights weights/maia-1500.pb.gz
```

See [MAIA_SETUP.md](MAIA_SETUP.md) for complete Maia documentation.

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

## Configuration Options

### Model Architecture (in `02_train_supervised.ipynb`)

```python
CONFIG = {
    'model_type': 'miniresnet',  # 'mlp', 'cnn', 'miniresnet'
    'num_blocks': 6,             # ResNet depth
    'channels': 64,              # Channel width
    'train_value_head': True,    # Enable value head
}
```

### Search Configuration (in `03_search_and_play.ipynb`)

```python
# Alpha-beta
SearchConfig(
    max_depth=3,                 # Search depth (1-5)
    time_limit=0.2,              # Time per move (seconds)
    use_policy_ordering=True,    # Policy-based move ordering
    use_value_eval=True,         # Value network evaluation
)

# MCTS
MCTSConfig(
    num_simulations=100,         # Number of simulations (50-500)
    temperature=1.0,             # Exploration temperature
)
```

### Match Configuration (in `04_benchmarks_and_analysis.ipynb`)

```python
NUM_GAMES = 100  # Increase to 200-300 for final benchmarks
```

---

## Project Components

### Data Processing (`src/data/`)
- **pgn_to_positions.py**: Extract positions from PGN with metadata (phase, outcome, move)
- **dataset.py**: PyTorch Dataset for chess positions
- **sampling.py**: Phase-stratified sampling and train/val/test split

### Models (`src/model/`)
- **nets.py**: MLPPolicy, CNNPolicyValue, MiniResNetPolicyValue
- **loss.py**: Label-smoothing cross-entropy + MSE for policy + value

### Search (`src/search/`)
- **alphabeta.py**: Iterative deepening alpha-beta with policy ordering and value evaluation
- **mcts_lite.py**: Lightweight MCTS with UCB and neural priors

### Game Play (`src/play/`)
- **engine_wrapper.py**: Unified interface for neural agent
- **stockfish_wrapper.py**: Stockfish with skill level capping
- **sunfish_wrapper.py**: Pure-Python Sunfish engine (baseline)
- **match_runner.py**: Tournament runner with PGN logging and statistics

### Utilities (`src/utils/`)
- **encoding.py**: Board → tensor (12 × 8 × 8), move → index (4672-dim)
- **metrics.py**: Elo calculation, Wilson CI, ACPL, policy accuracy
- **plotting.py**: Visualization for EDA, training curves, match results
- **seeds.py**: Reproducibility (Python, NumPy, PyTorch)

---

## Running Maia (Lc0) Benchmarks

Maia is a human-like chess engine based on Lc0 with specialized weights trained to mimic human play at specific rating levels.

### Prerequisites
```bash
# Install Lc0
brew install lc0  # macOS
# Or download from https://github.com/LeelaChessZero/lc0/releases

# Download Maia weights
# Visit https://maiachess.com/ and download desired rating level
# Example: maia1500.pb.gz for ~1500 Elo playing strength
```

### Sanity Check
```bash
python -m src.play.sanity \
  --lc0-path /usr/local/bin/lc0 \
  --maia-weights weights/maia1500.pb.gz
```

Expected output:
```
✓ Engine started: Maia-maia1500
✓ Move from starting position is legal
✓ Reply to e2e4 is legal
```

### Running Matches
```bash
python -m src.play.match_runner \
  --opponent maia \
  --games 100 \
  --movetime 300 \
  --maia-weights weights/maia1500.pb.gz \
  --lc0-path /usr/local/bin/lc0 \
  --threads 1
```

**Notes:**
- External opening book only (engines don't use internal books)
- UCI-only I/O for all engines
- Single authoritative `chess.Board` as source of truth
- Desync guard enabled by default (validates every engine move)

## Benchmarking Details

### Architecture
- **Single Authoritative Board**: One python-chess Board maintains game state
- **UCI-Only Protocol**: All engines communicate via UCI (no XBoard)
- **External Opening Book**: 20 short opening lines; engines must disable internal books
- **Desync Guard**: Validates every engine move before applying to board
- **Per-Ply Logging**: Optional JSON logs for each half-move

### Opening Book
- 20 short opening lines for game variety
- Alternates colors each game for fairness
- UCI format only

### Time Controls
- Fixed time per move: 200-300 ms (recommended)
- Or fixed depth: 2-3 ply
- Equal budgets across all engines

### Evaluation Metrics
- **Score**: Wins + 0.5 × Draws
- **Elo Difference**: With 95% confidence interval (Wilson method)
- **ACPL**: Average centipawn loss (via Stockfish analysis)

---

## Reproducing Results

1. Set random seed: `set_seed(42)` (already in notebooks)
2. Use same training data split
3. Train for same number of epochs with same hyperparameters
4. Run at least 100 games per pairing for stable Elo estimates

**Note**: Minor variations expected due to:
- Search time limits (hardware-dependent)
- Stockfish version (for ACPL)
- MPS vs CPU backend

---

## Limitations and Future Work

### Current Limitations
- **Training Data**: Limited to 100k positions (vs millions in AlphaZero)
- **Search Depth**: 3 ply (vs 20+ in strong engines)
- **No Opening Book**: Relies on training data for opening knowledge
- **No Tablebases**: Endgame play not optimal
- **Single-Threaded**: No parallelization

### Future Improvements
1. **Self-Play RL**: AlphaZero-style reinforcement learning
2. **Larger Networks**: 20+ ResNet blocks with 256 channels
3. **Data Augmentation**: Board rotations, color flipping
4. **Better Search**: Parallel MCTS, deeper alpha-beta
5. **Opening Books**: Integrate ECO opening database
6. **Syzygy Tablebases**: Perfect endgame play with ≤7 pieces
7. **NNUE Architecture**: Stockfish-style efficiently updatable networks

---

## Citation and Licensing

### Training Data
- **Source**: Lichess Open Database ([database.lichess.org](https://database.lichess.org/))
- **License**: CC0 (Public Domain)
- **Citation**: "Games from the Lichess Open Database (https://database.lichess.org/)"

### Code
This project is for educational purposes (course project). If you use or adapt this code, please cite:
```
Chess Deep Learning Agent
University of Colorado Boulder
Deep Learning Course Project, 2025
```

### Dependencies
- **PyTorch**: BSD License
- **python-chess**: GPL-3.0
- **Stockfish**: GPL-3.0 (optional dependency)

---

## Troubleshooting

### Common Issues

#### 1. MPS Not Available
```python
# Falls back to CPU automatically
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

#### 2. Stockfish Not Found
Update path in notebooks:
```python
STOCKFISH_PATH = '/path/to/stockfish'  # Update this
```

#### 3. Out of Memory
Reduce batch size or model size:
```python
CONFIG['batch_size'] = 128  # Instead of 256
CONFIG['channels'] = 32     # Instead of 64
```

#### 4. Slow Training
- Ensure MPS is enabled on Mac
- Reduce dataset size
- Use fewer ResNet blocks

#### 5. Import Errors
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

## Contact and Contribution

**Author**: [Your Name]
**Course**: Deep Learning, University of Colorado Boulder
**Date**: 2025

For questions or issues, please open an issue in the repository or contact the author.

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
