"""Quick test to verify chess piece images load correctly."""

import pygame
from pathlib import Path

def test_piece_images():
    """Test loading all chess piece images."""
    pygame.init()

    pieces_dir = Path(__file__).parent / "pieces"

    if not pieces_dir.exists():
        print(f"❌ Error: pieces directory not found at {pieces_dir}")
        print("Run 'python download_chess_pieces_png.py' first")
        return False

    piece_names = [
        'white_king', 'white_queen', 'white_rook', 'white_bishop', 'white_knight', 'white_pawn',
        'black_king', 'black_queen', 'black_rook', 'black_bishop', 'black_knight', 'black_pawn',
    ]

    all_ok = True
    print("Testing piece images:")
    print("=" * 60)

    for piece_name in piece_names:
        image_path = pieces_dir / f"{piece_name}.png"

        if not image_path.exists():
            print(f"  ❌ {piece_name}: File not found")
            all_ok = False
            continue

        try:
            image = pygame.image.load(str(image_path))
            width, height = image.get_size()
            print(f"  ✓ {piece_name}: {width}x{height} pixels")
        except Exception as e:
            print(f"  ❌ {piece_name}: Error loading - {e}")
            all_ok = False

    print("=" * 60)

    if all_ok:
        print("\n✓ All piece images loaded successfully!")
        print("\nYou can now run: python play_gui.py --model artifacts/weights/best_model.pth")
        return True
    else:
        print("\n❌ Some images failed to load")
        return False

if __name__ == "__main__":
    test_piece_images()
