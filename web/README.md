# Chess AI Web App (PWA)

A Progressive Web App that lets you play chess against your trained neural network model.

## Features

- Play chess against your AI model in the browser
- Works on desktop and mobile devices
- Installable as a PWA (add to home screen)
- Offline support after first load
- Responsive design
- Dark mode support

## Setup

### 1. Convert Your Model

First, convert your PyTorch model to ONNX format:

```bash
cd ..
python convert_model_to_onnx.py --input artifacts/weights/best_model.pth --output web/model.onnx
```

This will create `model.onnx` in the `web/` directory (~23 MB).

### 2. Create App Icons

Create two PNG icons:
- `icon-192.png` (192x192 pixels)
- `icon-512.png` (512x512 pixels)

You can use any chess-themed icon or logo. Quick option:
```bash
# Use ImageMagick or online tool to create icons
# Example with Unicode chess piece:
convert -size 192x192 -background "#181818" -fill white -font Arial -pointsize 120 -gravity center label:"♟️" icon-192.png
convert -size 512x512 -background "#181818" -fill white -font Arial -pointsize 320 -gravity center label:"♟️" icon-512.png
```

### 3. Run Local Server

You need a local web server to test (file:// protocol won't work due to CORS):

**Option A: Python HTTP Server**
```bash
python -m http.server 8000
```

**Option B: Node.js HTTP Server**
```bash
npx http-server -p 8000
```

**Option C: Live Server (VS Code Extension)**
- Install "Live Server" extension in VS Code
- Right-click `index.html` → "Open with Live Server"

### 4. Open in Browser

Navigate to: `http://localhost:8000`

## Deploy to Production

### Option 1: GitHub Pages

1. Create a new GitHub repository
2. Push the `web/` directory contents
3. Enable GitHub Pages in repository settings
4. Your app will be at: `https://yourusername.github.io/repo-name/`

### Option 2: Netlify

1. Sign up at [netlify.com](https://netlify.com)
2. Drag and drop the `web/` folder
3. Your app will be deployed instantly
4. Get a URL like: `https://your-chess-ai.netlify.app`

### Option 3: Vercel

```bash
npm install -g vercel
cd web
vercel
```

### Option 4: Custom Server

Upload the `web/` directory to any web host that serves static files:
- AWS S3 + CloudFront
- Google Cloud Storage
- Azure Static Web Apps
- Any shared hosting (cPanel, etc.)

## Using on Mobile

### iOS (iPhone/iPad)

1. Open the deployed URL in Safari
2. Tap the Share button (box with arrow)
3. Scroll down and tap "Add to Home Screen"
4. Tap "Add"
5. The app icon will appear on your home screen

### Android

1. Open the deployed URL in Chrome
2. Tap the menu (three dots)
3. Tap "Add to Home screen"
4. Tap "Add"
5. The app icon will appear on your home screen

## File Structure

```
web/
├── index.html           # Main HTML file
├── styles.css           # Styling
├── chess-ai.js          # Main JavaScript (chess logic + AI)
├── manifest.json        # PWA manifest
├── service-worker.js    # Service worker (offline support)
├── model.onnx          # Your converted AI model
├── icon-192.png        # App icon (192x192)
├── icon-512.png        # App icon (512x512)
└── README.md           # This file
```

## How It Works

1. **Chess Logic**: Uses [chess.js](https://github.com/jhlywa/chess.js) for game rules
2. **Board UI**: Uses [chessboard.js](https://chessboardjs.com/) for visual board
3. **AI Inference**: Uses [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) to run your model
4. **PWA**: Service worker caches all resources for offline use

## Customization

### Change AI Strength

Currently uses greedy policy (picks move with highest probability). To add search:

1. Implement minimax/alpha-beta in JavaScript
2. Use the model for position evaluation
3. Search N plies deep

### Styling

Edit `styles.css` to customize:
- Colors (change CSS variables in `:root`)
- Board size
- Layout
- Dark mode settings

### Model Updates

To use a different model:
```bash
python convert_model_to_onnx.py --input path/to/new_model.pth --output web/model.onnx
```

Then update cache version in `service-worker.js`:
```javascript
const CACHE_NAME = 'chess-ai-v2'; // Increment version
```

## Troubleshooting

### Model not loading
- Check browser console for errors
- Ensure `model.onnx` is in the `web/` directory
- Check CORS settings if hosting on custom server
- Try different browser (Chrome/Safari/Firefox)

### Slow performance
- ONNX Runtime Web uses WebAssembly (WASM)
- Performance depends on device CPU
- On mobile, expect 1-3 seconds per move
- Consider using smaller model (fewer blocks/channels)

### Service worker not updating
- Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Clear browser cache
- Increment `CACHE_NAME` version in service-worker.js

### Icons not showing
- Ensure `icon-192.png` and `icon-512.png` exist
- Check file paths in `manifest.json`
- Validate manifest at: https://manifest-validator.appspot.com/

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Safari: ✅ Full support (iOS 14.3+)
- Firefox: ✅ Full support
- Opera: ✅ Full support

## Performance

- **Model Size**: ~23 MB (loads once, then cached)
- **First Load**: 5-10 seconds (downloads model)
- **Subsequent Loads**: <1 second (from cache)
- **AI Move Time**:
  - Desktop: 0.5-2 seconds
  - Mobile: 1-5 seconds

## License

Same as your chess AI project.

## Credits

- [chess.js](https://github.com/jhlywa/chess.js) - Chess logic
- [chessboard.js](https://chessboardjs.com/) - Board UI
- [ONNX Runtime Web](https://onnxruntime.ai/) - Model inference
- Your trained neural network model

Enjoy playing against your AI!
