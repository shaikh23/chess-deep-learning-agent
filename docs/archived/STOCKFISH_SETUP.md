# Stockfish Setup Guide

✅ **Stockfish is already installed on your system!**

---

## Installation Confirmed

**Version**: Stockfish 17.1
**Location**: `/usr/local/bin/stockfish`
**Status**: ✅ Ready to use

---

## Verify Installation

You can test Stockfish anytime:

```bash
# Check if installed
which stockfish

# Expected output: /usr/local/bin/stockfish
```

```bash
# Test it works
echo "position startpos moves e2e4 e7e5
go depth 10
quit" | stockfish
```

You should see Stockfish analyze the position and suggest best moves.

---

## Using Stockfish in Notebooks

### Notebook 04: Benchmarks and Analysis

When you reach notebook 04, you'll see a cell that defines `STOCKFISH_PATH`:

```python
# Stockfish configuration
STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'  # Default for M1/M2/M3 Mac
```

**Update it to your path**:

```python
# Your Stockfish path (Intel Mac)
STOCKFISH_PATH = '/usr/local/bin/stockfish'
```

Or use the auto-detection approach:

```python
import shutil

# Auto-detect Stockfish
STOCKFISH_PATH = shutil.which('stockfish') or '/usr/local/bin/stockfish'
print(f"Using Stockfish at: {STOCKFISH_PATH}")
```

---

## What Stockfish Is Used For

In this project, Stockfish serves two purposes:

### 1. Benchmarking Opponent (Notebook 04)
Play matches against limited-strength Stockfish:

```python
# Create limited-strength Stockfish for fair comparison
with StockfishWrapper(STOCKFISH_PATH, skill_level=5, time_limit=0.2) as stockfish:
    runner = MatchRunner(our_agent, stockfish, MATCHES_DIR)
    stats = runner.run_match(num_games=100)
```

**Skill levels**:
- Level 0-5: Beginner to intermediate (fair for our agent)
- Level 10-15: Advanced
- Level 20: Full strength (world champion level)

### 2. ACPL Analysis (Notebook 04)
Analyze move quality by computing Average Centipawn Loss:

```python
# Analyze our moves to measure quality
acpl = compute_acpl_by_phase(game, STOCKFISH_PATH, depth=15)
print(f"Opening ACPL: {acpl['opening']}")
print(f"Middlegame ACPL: {acpl['middlegame']}")
print(f"Endgame ACPL: {acpl['endgame']}")
```

Lower ACPL = better moves (stronger play).

---

## Stockfish Skill Levels Explained

| Skill Level | Approximate Elo | Description |
|-------------|-----------------|-------------|
| 0 | ~1000 | Beginner |
| 5 | ~1400-1500 | Intermediate (recommended for testing) |
| 10 | ~1900-2000 | Club player |
| 15 | ~2400-2500 | Master level |
| 20 | ~3500+ | World champion level |

**Recommendation**: Use **skill level 5** for fair comparison with your agent.

---

## Quick Test

Test Stockfish from Python:

```python
import chess
import chess.engine

# Your Stockfish path
STOCKFISH_PATH = '/usr/local/bin/stockfish'

# Create engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Analyze starting position
board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(depth=15))

print(f"Best move: {info['pv'][0]}")
print(f"Score: {info['score']}")

engine.quit()
```

Expected output: Something like "Best move: e2e4, Score: +0.3"

---

## Troubleshooting

### "Stockfish not found"

If you get this error in notebooks, update the path:

```python
# Try these in order:
STOCKFISH_PATH = '/usr/local/bin/stockfish'      # Your path (Intel Mac)
# STOCKFISH_PATH = '/opt/homebrew/bin/stockfish' # M1/M2/M3 Mac
# STOCKFISH_PATH = '/usr/bin/stockfish'          # Linux
```

### "Permission denied"

Make Stockfish executable:

```bash
chmod +x /usr/local/bin/stockfish
```

### "Stockfish crashed" or "Engine timeout"

Reduce search depth or time:

```python
# In ACPL analysis, reduce depth
compute_acpl_by_phase(game, STOCKFISH_PATH, depth=10)  # Instead of 15

# In matches, reduce time limit
StockfishWrapper(STOCKFISH_PATH, skill_level=5, time_limit=0.1)  # Instead of 0.2
```

---

## Upgrade Stockfish (Optional)

To get the latest version:

```bash
brew upgrade stockfish
```

Current version (17.1) is already excellent and sufficient for this project.

---

## Alternative: Skip Stockfish Features

If you encounter issues, you can skip Stockfish-dependent features:

### Skip Stockfish Matches
In notebook 04, comment out the Stockfish match section:

```python
# if Path(STOCKFISH_PATH).exists():
#     # ... Stockfish match code ...
# else:
#     print("Skipping Stockfish matches")
```

### Skip ACPL Analysis
Comment out the ACPL computation section:

```python
# if Path(STOCKFISH_PATH).exists():
#     # ... ACPL analysis code ...
# else:
#     print("Skipping ACPL analysis")
```

You can still:
- ✅ Train your model
- ✅ Play against Sunfish
- ✅ Compute Elo estimates
- ✅ Generate match PGNs

---

## Summary

✅ **Stockfish 17.1 is installed at**: `/usr/local/bin/stockfish`

✅ **Ready to use for**:
- Benchmarking matches (skill-limited)
- ACPL move quality analysis
- Position evaluation

✅ **Update notebooks**: Change `STOCKFISH_PATH` to `/usr/local/bin/stockfish`

✅ **Recommended settings**:
- Skill level: 5 (fair opponent)
- Time limit: 0.2 seconds/move
- ACPL depth: 10-15

---

## Quick Reference

```bash
# Verify Stockfish works
which stockfish
# Output: /usr/local/bin/stockfish

# Test it
echo "quit" | stockfish
# Should show: Stockfish 17.1 by the Stockfish developers

# Your path to use in notebooks
STOCKFISH_PATH = '/usr/local/bin/stockfish'
```

You're all set! Stockfish will help you benchmark your agent and measure move quality. 🚀♟️
