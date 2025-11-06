# Chess Piece Images Setup

## What Changed

The GUI has been updated to use proper PNG chess piece images instead of Unicode symbols, making the pieces much clearer and easier to distinguish during play.

## Files Modified

- **play_gui.py**: Updated to load and display PNG images
  - Added `_load_piece_images()` method to load images from the `pieces/` directory
  - Modified `draw_pieces()` to display images instead of Unicode symbols
  - Removed dependency on Unicode font rendering for pieces

## Files Added

- **download_chess_pieces_png.py**: Script to download chess piece images from Wikimedia Commons
- **test_piece_images.py**: Test script to verify all images load correctly
- **pieces/**: Directory containing 12 PNG images (6 white pieces + 6 black pieces)

## Chess Piece Images

The images are from the Cburnett chess set, licensed under CC BY-SA 3.0 from Wikimedia Commons.

All 12 pieces (180x180 pixels each):
- White: King, Queen, Rook, Bishop, Knight, Pawn
- Black: King, Queen, Rook, Bishop, Knight, Pawn

## How to Use

Simply run the GUI as before:

```bash
python play_gui.py --model artifacts/weights/best_model.pth
```

The piece images will automatically load from the `pieces/` directory.

## Re-downloading Images

If you ever need to re-download the images:

```bash
python download_chess_pieces_png.py
```

## Testing

To verify all images load correctly without starting a game:

```bash
python test_piece_images.py
```

## Note on Pygame Warning

The warning about `pkg_resources` is from pygame's internal code and doesn't affect functionality. It's a deprecation notice for a future pygame version.
