# Play Against Your Chess AI - Complete Guide

This guide covers two ways to play against your trained chess AI:
1. **Desktop GUI** (Pygame) - Play locally on your laptop
2. **Web App (PWA)** - Play in browser or on mobile phone

---

## Option 1: Desktop GUI (Quick Start)

### Play Now!

```bash
# Default: Play as white with best model
python play_gui.py

# Play as black
python play_gui.py --color black

# Use different model
python play_gui.py --model artifacts/weights/final_model.pth

# Increase AI strength (slower but smarter)
python play_gui.py --depth 5
```

### Controls

- **Click** piece, then click destination to move
- **R** - Reset game
- **U** - Undo last move (yours + AI's)
- **ESC** - Quit

### Features

- Visual chess board
- Legal move highlighting
- AI thinking status
- Move history
- Automatic pawn promotion

See [PLAY_GUI_README.md](PLAY_GUI_README.md) for detailed instructions.

---

## Option 2: Web App (PWA) - For Mobile & Browser

### Setup Steps

#### Step 1: Model Already Converted ✓

Your PyTorch model has been converted to ONNX format:
- Location: `web/model.onnx`
- Size: ~23 MB
- Ready to use in browsers

#### Step 2: Create App Icons

You need two PNG icons. Quick method using Python:

```bash
# Install Pillow if needed
pip install Pillow

# Create icons with Python
python -c "
from PIL import Image, ImageDraw, ImageFont
for size in [192, 512]:
    img = Image.new('RGBA', (size, size), '#181818')
    draw = ImageDraw.Draw(img)
    # Simple colored square as placeholder
    margin = size // 6
    draw.rectangle([margin, margin, size-margin, size-margin], fill='#4a90e2')
    img.save(f'web/icon-{size}.png')
print('Icons created!')
"
```

Or use any chess icon you like (PNG format, 192x192 and 512x512).

#### Step 3: Run Local Test Server

```bash
cd web
python -m http.server 8000
```

Then open: **http://localhost:8000**

#### Step 4: Deploy to Internet (Required for Mobile)

Choose one method:

**Option A: GitHub Pages (Free)**
```bash
# 1. Create new GitHub repo
# 2. Push web/ folder contents
# 3. Enable GitHub Pages in settings
# 4. Access at: https://yourusername.github.io/repo-name/
```

**Option B: Netlify (Free, Easiest)**
1. Go to [netlify.com](https://netlify.com)
2. Drag and drop `web/` folder
3. Get instant URL like: `https://your-chess-ai.netlify.app`

**Option C: Vercel (Free)**
```bash
npm install -g vercel
cd web
vercel
```

### Using on Your Phone

#### iPhone/iPad:
1. Open the deployed URL in Safari
2. Tap Share button → "Add to Home Screen"
3. App appears on home screen like native app

#### Android:
1. Open the deployed URL in Chrome
2. Tap menu → "Add to Home screen"
3. App appears on home screen

### How PWA Works

When you visit the URL in your phone's browser:
1. It loads the web page with chess board
2. Model downloads (one time, ~23 MB)
3. Everything caches for offline use
4. You can add to home screen
5. Runs like a native app!

**Benefits:**
- ✅ No app store needed
- ✅ Works on iPhone AND Android
- ✅ Plays offline after first load
- ✅ Updates automatically when you redeploy

---

## Comparison

| Feature | Desktop GUI | Web App (PWA) |
|---------|-------------|---------------|
| **Setup Time** | Instant | 10-15 mins |
| **Where to Use** | Your laptop only | Any device with browser |
| **Performance** | Fastest | Slightly slower |
| **Mobile Support** | ❌ No | ✅ Yes |
| **Sharing** | ❌ Can't share | ✅ Share URL |
| **Offline** | ✅ Always | ✅ After first load |
| **Updates** | Rerun script | Redeploy |

---

## Recommended Workflow

### 1. Test Locally First
```bash
# Play on your laptop to test model strength
python play_gui.py
```

### 2. Deploy to Web
```bash
# If model plays well, deploy PWA for mobile access
cd web
# Upload to Netlify/Vercel/GitHub Pages
```

### 3. Share with Others
Share the URL with friends/family to play against your AI!

---

## Troubleshooting

### Desktop GUI Issues

**"Model file not found"**
```bash
# Check available models
ls artifacts/weights/*.pth

# Use correct path
python play_gui.py --model artifacts/weights/best_model.pth
```

**"Pygame window doesn't open"**
```bash
pip install pygame
```

### Web App Issues

**"Model not loading"**
- Check `web/model.onnx` exists (~23 MB)
- Open browser console for errors
- Try different browser (Chrome recommended)

**"CORS errors"**
- Don't use `file://` protocol
- Must use HTTP server or deployed URL
- Use: `python -m http.server 8000`

**"Slow on mobile"**
- Normal! WASM is slower than native
- Expect 1-5 seconds per move on phone
- Desktop is faster (0.5-2 seconds)

**"Service worker not updating"**
- Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
- Or increment version in `service-worker.js`

---

## Next Steps

### Improve AI Strength

1. **Train longer**: More epochs = better play
2. **Larger model**: Increase blocks/channels
3. **Better data**: Use GM games instead of amateur
4. **Add search**: Implement minimax in JavaScript

### Customize Web App

Edit `web/styles.css`:
- Change colors
- Adjust board size
- Modify layout
- Customize dark mode

### Deploy Updates

After retraining model:
```bash
# Convert new model
python convert_model_to_onnx.py --input artifacts/weights/new_model.pth --output web/model.onnx

# Redeploy
# (Upload to Netlify/Vercel/GitHub Pages)
```

---

## File Locations

```
Chess app/
├── play_gui.py                 # Desktop GUI script
├── PLAY_GUI_README.md          # Desktop GUI docs
├── convert_model_to_onnx.py    # Model conversion script
│
└── web/                        # PWA files
    ├── index.html              # Main page
    ├── styles.css              # Styling
    ├── chess-ai.js             # Game logic + AI
    ├── manifest.json           # PWA config
    ├── service-worker.js       # Offline support
    ├── model.onnx              # AI model (~23 MB)
    ├── icon-192.png            # App icon
    ├── icon-512.png            # App icon
    └── README.md               # Detailed web app docs
```

---

## Questions?

- Desktop GUI: See [PLAY_GUI_README.md](PLAY_GUI_README.md)
- Web App: See [web/README.md](web/README.md)
- Model conversion: See `convert_model_to_onnx.py --help`

**Have fun playing against your AI!**
