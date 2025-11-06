"""Download free chess piece images from Wikimedia Commons.

This script downloads high-quality PNG chess piece images that are free to use.
The pieces are from the Cburnett chess set, which is licensed under CC BY-SA 3.0.
"""

import urllib.request
import os
from pathlib import Path

# Wikimedia Commons URLs for Cburnett chess pieces (CC BY-SA 3.0)
# These are high-quality, professional-looking pieces
PIECE_URLS = {
    'white_king': 'https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg',
    'white_queen': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg',
    'white_rook': 'https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg',
    'white_bishop': 'https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg',
    'white_knight': 'https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg',
    'white_pawn': 'https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg',
    'black_king': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg',
    'black_queen': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg',
    'black_rook': 'https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg',
    'black_bishop': 'https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg',
    'black_knight': 'https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg',
    'black_pawn': 'https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg',
}


def download_pieces(output_dir: str = "pieces"):
    """Download chess piece images.

    Args:
        output_dir: Directory to save the images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Downloading chess pieces to {output_path.absolute()}...")

    for piece_name, url in PIECE_URLS.items():
        # Use SVG extension for now
        output_file = output_path / f"{piece_name}.svg"

        try:
            print(f"  Downloading {piece_name}...")
            urllib.request.urlretrieve(url, output_file)
            print(f"    ✓ Saved to {output_file}")
        except Exception as e:
            print(f"    ✗ Error downloading {piece_name}: {e}")

    print("\n✓ Download complete!")
    print(f"\nPieces saved to: {output_path.absolute()}")
    print("\nNote: The downloaded files are SVG format.")
    print("You'll need to install 'cairosvg' and 'pillow' to convert them:")
    print("  pip install cairosvg pillow")
    print("\nOr use the alternative PNG URLs (see download_chess_pieces_png.py)")


def main():
    download_pieces()


if __name__ == "__main__":
    main()
