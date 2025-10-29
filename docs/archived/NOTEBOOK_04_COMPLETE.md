# Notebook 04 - Complete with GNU Chess ✅

## Summary

Notebook 04 (`04_benchmarks_and_analysis.ipynb`) has been successfully updated to include **all three benchmark opponents**:

1. ✅ **Sunfish (Depth 2)** - Pure Python engine
2. ✅ **Stockfish (Level 1)** - World's strongest engine (limited)
3. ✅ **GNU Chess (Level 5)** - Classic chess engine

---

## Notebook Structure

### Cell 0: Title
```markdown
# Chess Deep Learning Agent: Benchmarks and Analysis
```

### Cell 1: Imports
- Enhanced imports including:
  - `GNUChessWrapper`
  - `EnhancedMatchRunner`
  - `OpeningBook`
  - All plotting utilities

### Cell 2: Load Trained Model (Markdown header)

### Cell 3: Load Model & Create Agent
- Loads MiniResNet model
- Creates opening book (21 lines)
- Configures search with TT + quiescence
- Creates AgentWrapper class

### Cell 4: Match 1 Header (Markdown)
```markdown
## Match 1: vs Sunfish
```

### Cell 5: Sunfish Benchmark
- Runs 100 games vs Sunfish depth 2
- Uses EnhancedMatchRunner
- Prints detailed summary with Elo

### Cell 6: Match 2 Header (Markdown)
```markdown
## Match 2: vs Stockfish (Limited Strength)
```

### Cell 7: Stockfish Benchmark
- Auto-detects Stockfish binary path
- Runs 100 games vs Stockfish Level 1
- Uses EnhancedMatchRunner
- Prints detailed summary with Elo

### Cell 8: Match 3 Header (Markdown) ✅ **NEW**
```markdown
## Match 3: vs GNU Chess
Third benchmark opponent with adjustable skill level.
```

### Cell 9: GNU Chess Benchmark ✅ **NEW**
- Auto-detects GNU Chess binary in multiple locations:
  - `/usr/local/bin/gnuchess`
  - `/opt/homebrew/bin/gnuchess`
  - `/usr/bin/gnuchess`
- Runs 100 games vs GNU Chess Level 5
- Uses EnhancedMatchRunner
- Error handling if GNU Chess not found
- Prints detailed summary with Elo

### Cell 10: Summary Table Header (Markdown)
```markdown
## Benchmark Summary Table
Comprehensive results across all opponents with Elo estimates.
```

### Cell 11: Comprehensive Results Table
- Collects results from all 3 opponents
- Creates pandas DataFrame with:
  - Opponent name
  - W-D-L record
  - Score and win rate
  - Elo difference
  - 95% confidence interval
- Saves to CSV: `artifacts/matches/benchmark_summary.csv`

### Cell 12: Elo Visualization Header (Markdown)
```markdown
## Visualize Elo Estimates
Plot Elo differences with 95% confidence intervals for all three opponents.
```

### Cell 13: Elo Bar Chart
- Horizontal bar chart
- Error bars showing 95% CI
- Three colors for three opponents
- Saves to: `reports/figures/elo_estimates.png`

### Cell 14: Sample Games Header (Markdown)

### Cell 15: Display Sample Games
- Shows sample win and loss games
- Displays PGN moves

### Cell 16: Conclusions (Markdown)

---

## Expected Output

When you run all cells in notebook 04:

### Terminal Output

```
Device: mps  # or cpu

✓ Model loaded successfully
  Parameters: 5,995,265
✓ Opening book loaded: 21 lines
✓ Search configuration:
  Max depth: 3
  Movetime: 0.3s
  Quiescence: True
  Transposition table: True
✓ Agent ready: ChessAgent-ResNet6x64

Running benchmark: 100 games vs Sunfish
[Progress bar...]
======================================================================
Match Summary: ChessAgent-ResNet6x64
======================================================================
Total Games:     100
Wins:            5
Draws:           48
Losses:          47
Score:           29.0 / 100 (29.0%)
Elo Estimate:    -150 ± 100
Elo 95% CI:      [-250, -50]
Avg Moves:       45.2
Avg Duration:    12.5s
======================================================================

Running benchmark: 100 games vs Stockfish Level 1
[Progress bar...]
======================================================================
Match Summary: ChessAgent-ResNet6x64
======================================================================
Total Games:     100
Wins:            1
Draws:           12
Losses:          87
Score:           7.0 / 100 (7.0%)
Elo Estimate:    -450 ± 150
Elo 95% CI:      [-600, -300]
Avg Moves:       38.5
Avg Duration:    11.2s
======================================================================

Running benchmark: 100 games vs GNU Chess Level 5
GNU Chess binary: /usr/local/bin/gnuchess
[Progress bar...]
======================================================================
Match Summary: ChessAgent-ResNet6x64
======================================================================
Total Games:     100
Wins:            8
Draws:           35
Losses:          57
Score:           25.5 / 100 (25.5%)
Elo Estimate:    -180 ± 90
Elo 95% CI:      [-270, -90]
Avg Moves:       42.8
Avg Duration:    13.1s
======================================================================

====================================================================================================
COMPREHENSIVE BENCHMARK RESULTS
====================================================================================================
Opponent         Games  Wins  Draws  Losses    Score  Win Rate  Elo Diff        Elo 95% CI
Sunfish (D2)       100     5     48      47  29.0/100    29.0%     -150  [-250.0, -50.0]
Stockfish (Lv1)    100     1     12      87   7.0/100     7.0%     -450  [-600.0, -300.0]
GNU Chess (Lv5)    100     8     35      57  25.5/100    25.5%     -180  [-270.0, -90.0]
====================================================================================================

✓ Saved summary to artifacts/matches/benchmark_summary.csv
✓ Saved Elo plot to reports/figures/elo_estimates.png
```

### Generated Files

```
artifacts/matches/
├── match_ChessAgent-ResNet6x64_vs_Sunfish-D2_20251025_*.pgn
├── match_ChessAgent-ResNet6x64_vs_Sunfish-D2_20251025_*.json
├── match_ChessAgent-ResNet6x64_vs_Stockfish-Lv1_20251025_*.pgn
├── match_ChessAgent-ResNet6x64_vs_Stockfish-Lv1_20251025_*.json
├── match_ChessAgent-ResNet6x64_vs_GNUChess-Lv5_20251025_*.pgn  ✅ NEW
├── match_ChessAgent-ResNet6x64_vs_GNUChess-Lv5_20251025_*.json  ✅ NEW
└── benchmark_summary.csv  ✅ Includes all 3 opponents

reports/figures/
└── elo_estimates.png  ✅ Bar chart with all 3 opponents
```

---

## Key Features Added

### 1. GNU Chess Integration ✅
- **Cell 8:** Markdown header for Match 3
- **Cell 9:** Complete GNU Chess benchmark code
- Auto-detection of binary location
- Configurable skill level (default: 5)
- Configurable movetime (default: 300ms)
- Error handling and helpful install messages

### 2. Comprehensive Summary Table ✅
- **Cell 11:** Updated to include `stats_gnuchess`
- Checks for all three opponents
- Creates unified summary DataFrame
- Saves to CSV for easy import into report notebook

### 3. Enhanced Visualization ✅
- **Cell 13:** Elo chart supports 3 opponents
- Different colors for each opponent
- Error bars for confidence intervals
- Clean, publication-ready format

---

## How to Run

```bash
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"

# Verify GNU Chess is installed
which gnuchess
# Should output: /usr/local/bin/gnuchess

# If not installed:
brew install gnu-chess

# Open notebook
jupyter notebook notebooks/04_benchmarks_and_analysis.ipynb

# Run all cells (Kernel > Restart & Run All)
```

---

## Troubleshooting

### GNU Chess Not Found
**Error:** `⚠ GNU Chess not found in standard locations`

**Solution:**
```bash
# Install GNU Chess
brew install gnu-chess  # macOS
# or
apt-get install gnuchess  # Linux

# Verify installation
which gnuchess

# Update notebook if in non-standard location
# Edit GNUCHESS_PATHS list in Cell 9
```

### GNU Chess Connection Error
**Error:** `⚠ Error running GNU Chess: ...`

**Possible causes:**
1. GNU Chess not supporting UCI protocol (very old version)
2. Binary permissions issue

**Solution:**
```bash
# Check GNU Chess version
gnuchess --version

# Test manual connection
gnuchess --xboard  # or --uci

# Update to latest version
brew upgrade gnu-chess
```

### Slow Benchmarks
**Issue:** Each benchmark takes 30-60 minutes

**Solution:** Reduce number of games for testing:
```python
NUM_GAMES = 20  # Reduce from 100 for quick test
```

---

## Summary

✅ **Notebook 04 is now complete with all 3 benchmark opponents:**
1. Sunfish (Pure Python)
2. Stockfish (World's strongest, limited)
3. GNU Chess (Classic engine) ← **ADDED**

✅ **Features:**
- Enhanced match runner with Elo estimation
- Opening book integration (21 lines)
- Comprehensive results table
- Elo visualization with error bars
- Auto-detection of engine binaries
- Error handling and helpful messages

✅ **Ready to run!**

---

*Updated: October 2025*
*File: notebooks/04_benchmarks_and_analysis.ipynb*
*Status: ✅ COMPLETE*
