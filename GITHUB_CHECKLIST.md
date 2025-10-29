# GitHub Upload Quick Checklist

## ✅ Pre-Upload Cleanup

```bash
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"

# 1. Remove duplicate Lichess data (582MB)
rm -rf src/data/Lichess

# 2. Verify .gitignore is set up (already done!)
cat .gitignore

# 3. Check what will be uploaded
git init
git add .
git status

# 4. Verify size (should be ~12MB, not 15GB!)
du -sh .git/objects/
```

## 📊 What Gets Uploaded (~12MB)

✅ **Source Code** (~10MB)
- `src/` - All Python modules (33 files)
- `notebooks/` - All 5 Jupyter notebooks

✅ **Documentation** (~100KB)
- `README.md`, `PROJECT_SUMMARY.md`
- `docs/` - Setup guides and documentation
- `GITHUB_GUIDE.md`, `GITHUB_CHECKLIST.md`

✅ **Presentation** (~380KB)
- `presentation/Chess_AI_Presentation.pptx`
- `presentation/PRESENTATION_SCRIPT.md`
- `presentation/VIDEO_SCRIPT.md`

✅ **Configuration** (~10KB)
- `requirements.txt`, `environment.yml`
- `config/` - Engine configs and scripts
- `.gitignore`

✅ **Visualizations** (~1.5MB)
- `reports/figures/` - All PNG charts (12 files)
- `reports/final/` - PDF reports (2 files)

## ❌ What Gets Excluded (15GB)

❌ **Training Data** (11GB)
- `artifacts/data/shards/` - 75 files × 147MB each
- Excluded via `.gitignore`
- Users will generate from Lichess data

❌ **Raw PGN Files** (3.1GB)
- `data/raw/Lichess Elite Database/`
- Excluded via `.gitignore`
- Users download from https://database.lichess.org/

❌ **Trained Models** (46MB)
- `artifacts/weights/*.pth`
- Excluded via `.gitignore`
- Users train their own (~40 minutes)

❌ **Backups** (158MB)
- `backups/*.tar.gz`
- Excluded via `.gitignore`
- Not needed on GitHub

❌ **Generated Outputs**
- `artifacts/matches/`, `artifacts/logs/`
- Excluded via `.gitignore`
- Users generate during benchmarks

## 🚀 Upload Commands

```bash
# 1. Initialize repository
git init
git add .
git commit -m "Initial commit: Chess Deep Learning Agent

Complete implementation of supervised learning chess AI:
- Policy-value neural network (MiniResNet, 6M params)
- Alpha-beta search with neural guidance
- Trained on Lichess master games
- ~2000 Elo playing strength
- Comprehensive notebooks and documentation
- Full presentation materials"

# 2. Create GitHub repo at https://github.com/new
#    Name: chess-deep-learning-agent
#    Visibility: Public (recommended for portfolio)
#    Don't initialize with README

# 3. Connect and push
git remote add origin https://github.com/YOUR-USERNAME/chess-deep-learning-agent.git
git branch -M main
git push -u origin main
```

## ⚡ Quick Verification

Before pushing, verify:

```bash
# Check total size (should be ~12MB, not GB!)
git count-objects -vH

# List all files to be committed
git ls-files

# Verify large files are excluded
git ls-files | xargs ls -lh | awk '$5 ~ /M|G/ {print $5, $9}'
```

**Expected output**: Only small files (<5MB each)

## 📝 Add to README

After uploading, add this section to your README.md:

```markdown
## 🚀 Quick Start

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/YOUR-USERNAME/chess-deep-learning-agent.git
cd chess-deep-learning-agent
\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
python verify_setup.py
\`\`\`

### 3. Download Training Data
Download Lichess PGN from https://database.lichess.org/
\`\`\`bash
# Extract and place in data/raw/
mkdir -p data/raw
mv lichess_db_*.pgn data/raw/
\`\`\`

### 4. Run Notebooks
\`\`\`bash
jupyter notebook
# Run in order: 01 → 02 → 03 → 04
\`\`\`

See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for detailed instructions.
```

## 🎯 Success Criteria

After uploading to GitHub, you should be able to:

1. ✅ Clone the repo on a fresh machine
2. ✅ Install dependencies from requirements.txt
3. ✅ Follow README to download data
4. ✅ Run all notebooks successfully
5. ✅ Train model from scratch
6. ✅ Repository size < 50MB
7. ✅ No large file warnings from GitHub

## ⚠️ Common Issues

### Issue: "File is too large (>100MB)"
**Solution**: Check `.gitignore` includes all large files
```bash
git rm --cached <large-file>
git commit --amend
```

### Issue: Repository is 1GB+
**Solution**: You forgot to exclude data/models
```bash
# Remove from git tracking
git rm -r --cached artifacts/data/
git rm -r --cached data/raw/
git commit --amend
```

### Issue: "Git LFS required"
**Solution**: Either use Git LFS or exclude .pth files
```bash
# Option 1: Exclude models
# (already done in .gitignore)

# Option 2: Use Git LFS
brew install git-lfs
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

## 📦 Optional: Include Sample Data

If you want to include a tiny sample for testing:

```bash
# Create small sample (not in .gitignore)
mkdir -p artifacts/data/samples
# Create 1-2MB sample dataset for quick testing

# Update .gitignore to allow samples/
echo '!artifacts/data/samples/' >> .gitignore
```

## 🎓 Portfolio Tips

To make your GitHub repo stand out:

1. ✅ **Clear README** with project overview
2. ✅ **Badges**: Add shields.io badges for Python, PyTorch
3. ✅ **Screenshots**: Include training curves, Elo chart in README
4. ✅ **License**: Add MIT or Apache 2.0 license
5. ✅ **Demo**: Link to presentation or video demo
6. ✅ **Documentation**: Your docs/ folder is excellent!
7. ✅ **Clean commits**: Meaningful commit messages

Example README badges:
```markdown
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

---

## Ready to Upload?

Run this final check:

```bash
# Verify cleanup
[ ! -d "src/data/Lichess" ] && echo "✅ Duplicate Lichess removed" || echo "❌ Remove src/data/Lichess"

# Verify .gitignore
git status | grep -E "artifacts/data|data/raw|backups" && echo "❌ Large files detected!" || echo "✅ No large files"

# Check size
SIZE=$(git count-objects -v | grep 'size-pack' | awk '{print $2}')
[ "$SIZE" -lt 50000 ] && echo "✅ Repo size OK (~$SIZE KB)" || echo "❌ Repo too large ($SIZE KB)"
```

If all checks pass: **You're ready to push to GitHub!** 🚀

---

**Time to upload**: ~30 seconds on fast internet
**Repository size**: ~12MB (professional and fast to clone!)
