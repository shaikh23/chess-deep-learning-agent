## Current status: 
Creating an app where you can play against the model. Will deploy via Vercel. Using Claude for assistance and refining my code where necessary. 

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

## Installation

### pip 

```bash
# Clone or download the repository
cd "Chess app"

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


### Install Engines

#### Stockfish
```bash
# macOS (Homebrew)
brew install stockfish

```

#### Lc0 (for Maia)
```bash
# macOS (Homebrew)
brew install lc0

```

#### Maia Weights
Download human-like neural network weights from [https://maiachess.com/](https://maiachess.com/)

---

### 1. Obtain Training Data

**Lichess Open Database**: [https://database.lichess.org/](https://database.lichess.org/)

1. Download PGN files (e.g., from Lichess Elite or monthly databases)
2. Extract: `zstd -d lichess_db_standard_rated_2024-01.pgn.zst`
3. Place in `data/raw/` directory (create if needed)

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

### 3. Generate Teacher Labels with Stockfish

For higher-quality value targets, label positions with Stockfish:

```bash
python -m src.tools.teacher_label_stockfish \
  --in artifacts/data/shards \
  --out artifacts/data/teacher \
  --sf /opt/homebrew/bin/stockfish \
  --depth 10 \
  --max-positions 100000
```

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


---

## Video Demo

See [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) for presentation outline and talking points for the 5-15 minute video demo.
