# Interactive Chess Demos

Quick ways to play against the trained model.

## GUI Version (play_gui.py)

Pygame-based chess interface with visual board.

```bash
# Download piece images first
python download_chess_pieces_png.py

# Play as white
python play_gui.py --model ../artifacts/weights/best_model.pth

# Play as black
python play_gui.py --model ../artifacts/weights/best_model.pth --color black
```

## Web Version (../web/)

Browser-based PWA with ONNX model.

```bash
# Convert PyTorch model to ONNX
python convert_model_to_onnx.py --model ../artifacts/weights/best_model.pth --output ../web/model.onnx

# Start local server
./START_WEB_SERVER.sh
```

Then open http://localhost:8000

## Notes

The GUI needs pygame and the piece PNG files. Web version needs the model converted to ONNX format for in-browser inference.
