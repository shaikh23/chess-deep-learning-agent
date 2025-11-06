"""Download free chess piece PNG images.

Downloads PNG format chess pieces that work directly with pygame.
Uses the Cburnett chess set from Wikimedia Commons (CC BY-SA 3.0).
"""

import urllib.request
import os
from pathlib import Path

# Direct PNG URLs from Wikimedia Commons (already rendered at good size)
PIECE_URLS = {
    'white_king': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Chess_klt45.svg/180px-Chess_klt45.svg.png',
    'white_queen': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Chess_qlt45.svg/180px-Chess_qlt45.svg.png',
    'white_rook': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chess_rlt45.svg/180px-Chess_rlt45.svg.png',
    'white_bishop': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Chess_blt45.svg/180px-Chess_blt45.svg.png',
    'white_knight': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Chess_nlt45.svg/180px-Chess_nlt45.svg.png',
    'white_pawn': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Chess_plt45.svg/180px-Chess_plt45.svg.png',
    'black_king': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Chess_kdt45.svg/180px-Chess_kdt45.svg.png',
    'black_queen': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Chess_qdt45.svg/180px-Chess_qdt45.svg.png',
    'black_rook': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Chess_rdt45.svg/180px-Chess_rdt45.svg.png',
    'black_bishop': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Chess_bdt45.svg/180px-Chess_bdt45.svg.png',
    'black_knight': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Chess_ndt45.svg/180px-Chess_ndt45.svg.png',
    'black_pawn': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Chess_pdt45.svg/180px-Chess_pdt45.svg.png',
}


def download_pieces(output_dir: str = "pieces"):
    """Download chess piece PNG images.

    Args:
        output_dir: Directory to save the images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Downloading chess piece PNG images to {output_path.absolute()}...")
    print("=" * 60)

    success_count = 0
    for piece_name, url in PIECE_URLS.items():
        output_file = output_path / f"{piece_name}.png"

        try:
            print(f"  Downloading {piece_name}...", end=" ")
            # Add user agent header to avoid 403 errors
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(output_file, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"✓")
            success_count += 1
        except Exception as e:
            print(f"✗ Error: {e}")

    print("=" * 60)
    print(f"\n✓ Successfully downloaded {success_count}/{len(PIECE_URLS)} pieces!")
    print(f"\nPieces saved to: {output_path.absolute()}")
    print("\nYou can now run play_gui.py and the pieces will be displayed properly!")
    print("\nLicense: CC BY-SA 3.0 (Cburnett chess set from Wikimedia Commons)")


def main():
    download_pieces()


if __name__ == "__main__":
    main()
