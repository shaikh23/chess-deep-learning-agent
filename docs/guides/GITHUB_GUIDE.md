# GitHub Upload Guide

This guide explains what to upload to GitHub and what to exclude for best practices.

## Summary of File Sizes

```
Total Project Size: ~15GB

Breakdown:
- artifacts/      11GB  (processed data + trained models)
- data/           3.1GB (raw Lichess PGN files)
- src/            583MB (includes duplicate Lichess data)
- backups/        158MB (compressed archives)
- reports/        1.5MB (figures + PDFs)
- presentation/   380KB (slides + scripts)
```

## ✅ INCLUDE on GitHub (Essential Files)

### 1. Source Code & Notebooks (~10MB)
```
src/                          # All Python source code
  ├── data/
  ├── model/
  ├── search/
  ├── play/
  └── utils/

notebooks/                    # All Jupyter notebooks
  ├── 01_eda_and_preprocessing.ipynb
  ├── 02_train_supervised.ipynb
  ├── 03_search_and_play.ipynb
  ├── 04_benchmarks_and_analysis.ipynb
  └── 00_report_submission.ipynb
```

### 2. Documentation (~100KB)
```
README.md
PROJECT_SUMMARY.md
FILE_ORGANIZATION.md
GITHUB_GUIDE.md (this file)

docs/
  ├── README.md
  ├── SETUP_GUIDE.md
  ├── QUICKSTART.md
  ├── TROUBLESHOOTING.md
  └── USE_YOUR_DATA.md
  # Note: Can exclude docs/archived/ (optional)

presentation/
  ├── Chess_AI_Presentation.pptx
  ├── PRESENTATION_SCRIPT.md
  ├── VIDEO_SCRIPT.md
  └── create_presentation.py
```

### 3. Configuration Files (~10KB)
```
requirements.txt
environment.yml
verify_setup.py
.gitignore

config/
  ├── engines.py
  ├── presets.yaml
  └── run_maia_benchmark.sh
```

### 4. Small Reports & Figures (~1.5MB)
```
reports/
  ├── figures/                # Include all visualizations (12 PNG files)
  └── final/                  # Include PDFs (2 files, ~350KB total)
```

**Total to upload: ~12MB** ✅

## ❌ EXCLUDE from GitHub (Large Files)

### 1. Training Data (11GB) - TOO LARGE
```
artifacts/data/
  ├── shards/                 # 10.5GB (75 files × ~147MB each)
  └── positions_raw.csv.gz    # 11MB
```

**Why exclude**: Users should generate their own from Lichess data
**Alternative**: Provide download instructions in README

### 2. Raw Lichess PGNs (3.1GB) - TOO LARGE
```
data/raw/Lichess Elite Database/  # ~3.1GB of PGN files
src/data/Lichess                   # 582MB duplicate
```

**Why exclude**: Publicly available from Lichess, users should download directly
**Alternative**: Link to https://database.lichess.org/ in documentation

### 3. Trained Model Weights (46MB) - OPTIONAL
```
artifacts/weights/
  ├── best_model.pth          # 23MB
  └── final_model.pth         # 23MB
```

**Why consider excluding**: Large binary files, users can train their own
**Alternative options**:
- **Option A**: Use Git LFS (Git Large File Storage) for model files
- **Option B**: Upload to external hosting (Google Drive, Hugging Face Hub)
- **Option C**: Exclude and let users train from scratch (recommended)

### 4. Backups (158MB) - NOT NEEDED
```
backups/
  ├── shards_backup.tar.gz    # 158MB
  └── src_code.tar.gz         # 30KB
```

**Why exclude**: Redundant, source code is already on GitHub

### 5. Generated Match Files - OPTIONAL
```
artifacts/matches/            # PGN files and JSON stats
artifacts/logs/               # Training logs
```

**Why consider excluding**: Users will generate their own during benchmarks
**Could include**: A few sample matches as examples (<1MB)

### 6. External Model Weights (1.2MB) - LICENSING
```
artifacts/maia-1500.pb.gz     # 1.2MB
```

**Why exclude**: Third-party model, has its own licensing
**Alternative**: Provide download link to https://maiachess.com/

## 📝 Recommended .gitignore

Create or update `.gitignore`:

```gitignore
# Large data files
data/raw/
artifacts/data/
backups/
src/data/Lichess

# Trained models (optional - comment out if using Git LFS)
artifacts/weights/*.pth

# Generated outputs (users will create their own)
artifacts/matches/
artifacts/logs/

# External model weights
artifacts/maia-1500.pb.gz

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.log
*.tmp
```

## 🚀 GitHub Upload Steps

### Step 1: Clean Up Large Files

```bash
cd "/Users/aneesshaikh/colorado_masters/deep_learn/Chess app"

# Remove duplicate Lichess data in src/
rm -rf src/data/Lichess

# Optional: Remove trained models (or use Git LFS)
# rm -rf artifacts/weights/*.pth
```

### Step 2: Create/Update .gitignore

```bash
# Create .gitignore with recommended excludes
cat > .gitignore << 'EOF'
# Large data files
data/raw/
artifacts/data/
backups/

# Trained models
artifacts/weights/*.pth

# Generated outputs
artifacts/matches/
artifacts/logs/

# External weights
artifacts/maia-1500.pb.gz

# Python
__pycache__/
*.py[cod]
venv/
env/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# IDE
.vscode/
.idea/
EOF
```

### Step 3: Initialize Git Repository

```bash
# Initialize repo
git init

# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status

# Create initial commit
git commit -m "Initial commit: Chess Deep Learning Agent

- Supervised learning chess AI trained on Lichess master games
- MiniResNet policy-value network (6M parameters)
- Alpha-beta search with neural guidance
- ~2000 Elo strength
- Complete notebooks, documentation, and presentation materials"
```

### Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Create repository: `chess-deep-learning-agent` (or your preferred name)
3. Choose: Public (recommended for portfolio) or Private
4. Don't initialize with README (you already have one)

### Step 5: Push to GitHub

```bash
# Add remote repository (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/chess-deep-learning-agent.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 📦 Optional: Git LFS for Model Weights

If you want to include trained models (23MB each):

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or: apt-get install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes

# Now add and commit model files
git add artifacts/weights/*.pth
git commit -m "Add trained model weights via Git LFS"
git push
```

**Note**: GitHub LFS has storage limits (1GB free). Your 46MB models fit fine.

## 📊 What Users Will Download

With recommended exclusions:

**Repository size**: ~12MB (fast to clone!)

**Users will need to**:
1. Clone your repo
2. Download Lichess PGN data (instructions in README)
3. Run notebook 01 to preprocess data
4. Run notebook 02 to train model
5. Run notebooks 03-04 for evaluation

## 🎯 Alternative: Provide Pre-trained Models Externally

If you want to share trained models without GitHub LFS:

### Option A: Google Drive
```bash
# Upload to Google Drive, get shareable link
# Add to README:
Trained model weights available at:
https://drive.google.com/file/d/YOUR-FILE-ID/view?usp=sharing

Download and place in artifacts/weights/best_model.pth
```

### Option B: Hugging Face Hub
```bash
# Upload to Hugging Face Model Hub (https://huggingface.co/)
# Provides versioning, model cards, easy downloads

# In README:
pip install huggingface_hub
huggingface-cli download YOUR-USERNAME/chess-agent best_model.pth
```

### Option C: GitHub Releases
```bash
# After pushing to GitHub, go to Releases
# Create a new release and attach model files as assets
# Users can download from releases page
```

## 📋 Recommended README Updates

Add to your README.md:

```markdown
## Pre-trained Models

### Option 1: Train Your Own (Recommended)
Follow the notebooks to train from scratch (~40 minutes on M1 Mac).

### Option 2: Download Pre-trained
Pre-trained model (~2000 Elo) available at:
- [Google Drive Link] or [Hugging Face Link] or [GitHub Releases]

Download and place in `artifacts/weights/best_model.pth`

## Data Setup

Training data should be obtained from Lichess:

1. Visit https://database.lichess.org/
2. Download a monthly PGN file (e.g., lichess_db_standard_rated_2024-01.pgn.zst)
3. Extract: `zstd -d <file>.pgn.zst`
4. Place in `data/raw/`
5. Run notebook 01 to preprocess

Alternatively, the notebooks will generate synthetic data for testing.
```

## ✅ Final Checklist

Before pushing to GitHub:

- [ ] Remove `src/data/Lichess` (582MB duplicate)
- [ ] Create `.gitignore` with exclusions
- [ ] Verify file sizes: `du -sh * | sort -h`
- [ ] Check git status: `git status` (should be ~12MB of files)
- [ ] Test that notebooks still run after cleanup
- [ ] Update README with data download instructions
- [ ] Add badge/links to presentation materials
- [ ] Consider adding LICENSE file (MIT, Apache 2.0, etc.)

## 📈 Expected Upload Time

With recommended exclusions (~12MB):
- **Fast internet**: 10-30 seconds
- **Moderate internet**: 1-2 minutes
- **Slow internet**: 3-5 minutes

Without exclusions (15GB):
- **Would take hours and hit GitHub limits** ❌

## 🎓 GitHub Best Practices

1. **README.md**: Clear, comprehensive (you already have this!)
2. **LICENSE**: Add MIT or Apache 2.0 license
3. **requirements.txt**: Pinned versions (you have this)
4. **Documentation**: Organized in docs/ (you have this)
5. **Code organization**: Clean structure (you have this)
6. **.gitignore**: Exclude large/generated files (create this)
7. **Presentation**: Include in repo (you have this!)
8. **GitHub Pages**: Consider hosting presentation HTML

## 🔗 Bonus: GitHub Pages for Presentation

You can host your HTML presentation on GitHub Pages:

```bash
# Create docs/ branch for GitHub Pages
git checkout -b gh-pages
git push origin gh-pages

# In GitHub repo settings:
# Settings → Pages → Source: gh-pages branch → /docs folder

# Your presentation will be at:
# https://YOUR-USERNAME.github.io/chess-deep-learning-agent/
```

---

## Summary

**Upload**: ~12MB of code, notebooks, docs, figures
**Exclude**: 15GB of data/models (provide download instructions)
**Result**: Fast, professional GitHub repo that showcases your work!

Good luck with your GitHub upload! 🚀
