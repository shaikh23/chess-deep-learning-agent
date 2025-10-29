# Using Your Lichess Elite Database

✅ **Success!** Extracted 80 PGN files with **3.8 million elite games**!

---

## What You Have

📂 **Location**: `data/raw/Lichess Elite Database/`
📊 **Files**: 80 monthly PGN files (2013-2020)
🎮 **Games**: 3,819,130 elite games
👥 **Players**: Elo 2200+ (elite level)
💾 **Size**: 3.3 GB uncompressed

---

## Quick Start: Use One File

For fast training (recommended for first run), use a recent large file:

### Option 1: Use 2019 data (recommended)

```python
# In notebook 01_eda_and_preprocessing.ipynb, cell 3:
PGN_PATH = Path('../data/raw/Lichess Elite Database/lichess_elite_2019-12.pgn')
MAX_GAMES = 10000  # Extract 10k games for testing
```

### Option 2: Use all 2019 files

To combine multiple files, you can concatenate them:

```bash
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app/data/raw/Lichess Elite Database"

# Combine 2019 files into one
cat lichess_elite_2019-*.pgn > ../combined_2019.pgn

# Then use in notebook:
# PGN_PATH = Path('../data/raw/combined_2019.pgn')
```

### Option 3: Use ALL files (advanced)

For maximum training data, process all files sequentially:

```python
# In notebook 01, modify extraction to loop through files
from pathlib import Path
import pandas as pd

pgn_dir = Path('../data/raw/Lichess Elite Database')
all_positions = []

# Process each file
for pgn_file in sorted(pgn_dir.glob('lichess_elite_*.pgn')):
    print(f"Processing {pgn_file.name}...")
    df = extract_positions_from_pgn(
        pgn_file,
        max_games=5000,  # 5k per file = 400k total
        min_elo=2200,    # Elite players
    )
    all_positions.append(df)

    # Stop when we have enough
    if sum(len(df) for df in all_positions) > 500000:
        break

# Combine all
df_positions = pd.concat(all_positions, ignore_index=True)
```

---

## Recommended Settings

### For Quick Test (30 mins total)
```python
PGN_PATH = Path('../data/raw/Lichess Elite Database/lichess_elite_2019-12.pgn')
MAX_GAMES = 5000        # Quick extraction
MIN_ELO = 2200          # Elite players
TARGET_SIZE = 50000     # 50k positions for training
```

### For Production Training (2-3 hours)
```python
PGN_PATH = Path('../data/raw/Lichess Elite Database/lichess_elite_2019-12.pgn')
MAX_GAMES = 50000       # More games
MIN_ELO = 2200
TARGET_SIZE = 200000    # 200k positions
```

### For Maximum Quality (all data)
Use the loop approach above to process multiple files.

---

## File Sizes Guide

Here are some of the larger files (more games):

```
2019-12.pgn   - Most recent, large
2019-11.pgn   - Also good
2018-12.pgn   - Year-end has more games
2017-12.pgn   - Also large
```

Check sizes:
```bash
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app/data/raw/Lichess Elite Database"
ls -lh lichess_elite_201*.pgn | sort -k5 -hr | head -10
```

---

## Update Notebook 01

Open `notebooks/01_eda_and_preprocessing.ipynb` and find cell 3:

**Before**:
```python
PGN_PATH = Path('../data/raw/sample_games.pgn')  # Replace with actual path
MAX_GAMES = 10000
MIN_ELO = 1800
```

**After** (recommended):
```python
PGN_PATH = Path('../data/raw/Lichess Elite Database/lichess_elite_2019-12.pgn')
MAX_GAMES = 10000   # Start with 10k for testing
MIN_ELO = 2200      # These are elite games, so use 2200
SKIP_OPENING_MOVES = 3
```

---

## Data Quality Notes

✅ **Advantages of your data**:
- **High Elo (2200+)**: Better moves, stronger patterns
- **Large volume**: 3.8M games available
- **Time span**: 2013-2020, shows evolution of play
- **Verified games**: From Lichess with ratings

⚠️ **Considerations**:
- **Elite level**: Elo 2200+ is advanced (your agent will learn strong moves)
- **Processing time**: Full dataset would take hours; use subsets
- **Disk space**: Keep ~5-10 GB free for processed files

---

## Step-by-Step: First Run

1. **Open notebook 01**:
   ```bash
   jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
   ```

2. **Update cell 3**:
   ```python
   PGN_PATH = Path('../data/raw/Lichess Elite Database/lichess_elite_2019-12.pgn')
   MAX_GAMES = 10000
   MIN_ELO = 2200
   ```

3. **Run all cells** (takes ~10-15 mins)

4. **Check output**:
   ```bash
   ls -lh artifacts/data/
   # Should see: train.csv.gz, val.csv.gz, test.csv.gz
   ```

5. **Proceed to notebook 02** to train the model!

---

## Expected Results

With your elite data, expect **better performance** than the README estimates:

| Metric | Standard Data | Your Elite Data |
|--------|---------------|-----------------|
| Policy Top-1 | 45-50% | **50-55%** |
| vs Sunfish | +185 Elo | **+200-250 Elo** |
| vs Stockfish-Lv5 | -50 Elo | **0 to +50 Elo** |

Your model will learn from strong players (2200+ Elo), making it stronger!

---

## Quick Commands Summary

```bash
# 1. Check what you have
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app/data/raw/Lichess Elite Database"
ls -lh *.pgn | wc -l          # Should show 80 files
grep -c "^\[Event " *.pgn | awk -F: '{sum+=$2} END {print sum}'  # Total games

# 2. Pick a file (2019-12 recommended)
head -50 lichess_elite_2019-12.pgn

# 3. Update notebook and run
jupyter notebook

# 4. Train model (notebook 02)
# Expected time: ~30-60 mins on M1 Mac

# 5. Benchmark (notebook 04)
# Expected time: ~60 mins for 100 games
```

---

## Troubleshooting

### "File not found"
Make sure path is correct:
```python
from pathlib import Path
pgn_path = Path('../data/raw/Lichess Elite Database/lichess_elite_2019-12.pgn')
print(f"Exists: {pgn_path.exists()}")
print(f"Absolute: {pgn_path.absolute()}")
```

### "No games extracted"
Lower the MIN_ELO if needed:
```python
MIN_ELO = 2000  # Instead of 2200
```

### "Processing too slow"
Reduce MAX_GAMES:
```python
MAX_GAMES = 5000  # Instead of 10000
```

### "Out of memory"
Reduce TARGET_SIZE:
```python
TARGET_SIZE = 50000  # Instead of 100000
```

---

## Ready to Train! 🚀

Your data is ready. Now:

1. ✅ Open `notebooks/01_eda_and_preprocessing.ipynb`
2. ✅ Update PGN_PATH to point to your extracted file
3. ✅ Run all cells to extract positions
4. ✅ Proceed to notebook 02 for training
5. ✅ Enjoy strong results from elite game data!

Expected total time: **~2 hours** from EDA to trained model to benchmarks.

Good luck! ♟️🤖
