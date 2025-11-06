# Play Against Your Chess AI - GUI Guide

## Quick Start

### Play with default settings (you play white):
```bash
python play_gui.py
```

### Play as black:
```bash
python play_gui.py --color black
```

### Use a different model:
```bash
python play_gui.py --model artifacts/weights/final_model.pth
```

### Increase AI strength (slower but smarter):
```bash
python play_gui.py --depth 5
```

## Controls

- **Click** a piece, then click destination to move
- **R** - Reset game
- **U** - Undo last move (yours + AI's)
- **ESC** - Quit

## Features

- Visual chess board with drag-and-drop moves
- Legal move highlighting (green squares)
- Selected piece highlighting (yellow)
- AI thinking status
- Move history display
- Automatic pawn promotion to queen
- Game over detection (checkmate, stalemate, draw)

## Customization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to model weights | `artifacts/weights/best_model.pth` |
| `--color` | Your color (white/black) | `white` |
| `--depth` | AI search depth (1-10) | `3` |

## Search Depth Guide

- **Depth 1-2**: Very fast, weak play (~500 Elo)
- **Depth 3**: Good balance (default, ~1200 Elo)
- **Depth 4-5**: Strong play, takes longer (~1500 Elo)
- **Depth 6+**: Very strong, slow (~1800+ Elo)

## Troubleshooting

### Model not found
```
Error: Model file not found: artifacts/weights/best_model.pth
```

**Solution**: Check which models you have:
```bash
ls artifacts/weights/*.pth
```

Then specify the correct path:
```bash
python play_gui.py --model artifacts/weights/final_model.pth
```

### Pygame window doesn't open
Make sure pygame is installed:
```bash
pip install pygame
```

### AI takes too long
Reduce search depth:
```bash
python play_gui.py --depth 2
```

## Example Sessions

### Quick casual game:
```bash
python play_gui.py --depth 2
```

### Challenging match:
```bash
python play_gui.py --depth 5
```

### Play as black:
```bash
python play_gui.py --color black --depth 3
```

Enjoy playing against your AI!
