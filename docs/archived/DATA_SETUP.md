# Data Setup Guide: Lichess Elite Database

You have downloaded the Lichess Elite Database (582 MB). Let's extract and prepare it for training.

---

## Step 1: Install 7z Extractor

The file is in `.7z` format. Install the extractor:

### Option A: Homebrew (Recommended)
```bash
brew install p7zip
```

### Option B: MacPorts
```bash
sudo port install p7zip
```

### Option C: Manual Download
Download from: https://www.7-zip.org/download.html

---

## Step 2: Extract the Archive

Once p7zip is installed, extract the file:

```bash
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"

# Extract to data/raw directory
7z x "src/data/Lichess Elite Database.7z" -o"data/raw/"
```

This will extract the PGN files to `data/raw/`.

**Expected output**: One or more `.pgn` files containing chess games.

---

## Step 3: Verify Extraction

Check what was extracted:

```bash
ls -lh data/raw/
```

You should see `.pgn` files.

---

## Step 4: Update Notebook Path

Open `notebooks/01_eda_and_preprocessing.ipynb` and update the `PGN_PATH`:

```python
# Find this line (around cell 3):
PGN_PATH = Path('../data/raw/sample_games.pgn')

# Update to your extracted file:
PGN_PATH = Path('../data/raw/lichess_elite.pgn')  # Or whatever filename was extracted
```

---

## Quick Commands (Copy-Paste)

```bash
# 1. Install p7zip
brew install p7zip

# 2. Navigate to project
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"

# 3. Create data directory
mkdir -p data/raw

# 4. Extract archive
7z x "src/data/Lichess Elite Database.7z" -o"data/raw/"

# 5. Check extracted files
ls -lh data/raw/

# 6. Preview PGN content (first 50 lines)
head -50 data/raw/*.pgn
```

---

## Alternative: Move File Without Extraction

If the 7z archive actually contains a single PGN (already in text format), you might be able to just move it:

```bash
# Move to correct location
mv "src/data/Lichess Elite Database.7z" "data/raw/"

# Rename if needed
cd data/raw
mv "Lichess Elite Database.7z" "lichess_elite.pgn"
```

**Note**: This only works if the file is misnamed and is actually a PGN, not a compressed archive.

---

## Expected File Format

A valid PGN file should look like this:

```
[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[Date "2024.01.15"]
[Round "-"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1950"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 ...

[Event "Rated Blitz game"]
[Site "https://lichess.org/efgh5678"]
...
```

---

## Troubleshooting

### Issue: "7z command not found"
**Solution**: Install p7zip with `brew install p7zip`

### Issue: "Extraction failed" or "Archive corrupted"
**Solution**: Re-download the file or check if it's actually a different format

### Issue: "No space left on device"
**Solution**: The extracted PGN might be 1-2 GB. Free up disk space.

### Issue: "File is not a 7z archive"
**Solution**: Check the actual file type:
```bash
file "src/data/Lichess Elite Database.7z"
```

If it says it's already a text file, just rename it:
```bash
mv "src/data/Lichess Elite Database.7z" "data/raw/lichess_elite.pgn"
```

---

## After Extraction

Once you have the PGN file in `data/raw/`:

1. **Update notebook 01**:
   ```python
   PGN_PATH = Path('../data/raw/lichess_elite.pgn')
   MAX_GAMES = 10000  # Start with 10k games for testing
   ```

2. **Run notebook 01**: Extract positions and create train/val/test splits

3. **Proceed to notebook 02**: Train the model

---

## File Size Expectations

- **Compressed (.7z)**: 582 MB (current)
- **Extracted (.pgn)**: Likely 1-2 GB (text format is larger)
- **Processed positions**: ~100-500 MB (after extraction to CSV)

Make sure you have at least **5 GB free disk space** for the full pipeline.

---

## Quick Test

After extraction, test that the PGN is valid:

```bash
# Count number of games (each game starts with [Event])
grep -c "^\[Event " data/raw/*.pgn

# Show first game
sed -n '/^\[Event/,/^$/p' data/raw/*.pgn | head -20
```

Expected output: Should show a complete chess game with headers and moves.

---

## Ready to Go?

Once extracted, you're ready to run:

```bash
jupyter notebook
# Open and run: 01_eda_and_preprocessing.ipynb
```

The notebook will:
1. Load your PGN file
2. Extract 100k-500k positions
3. Create stratified train/val/test splits
4. Save to `artifacts/data/`

Then proceed to training (notebook 02)!

---

## Need Help?

If extraction issues persist, you can also:

1. **Use synthetic data** (automatic fallback in notebook 01)
2. **Download smaller sample** from Lichess: https://database.lichess.org/
3. **Ask me** to help troubleshoot specific errors

Good luck! 🚀
