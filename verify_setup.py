#!/usr/bin/env python3
"""Verify that the chess-dl-agent environment is correctly set up."""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro} (need 3.11+)")
        return False

def check_imports():
    """Check required packages."""
    required = {
        'torch': 'PyTorch',
        'chess': 'python-chess',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
    }

    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (not installed)")
            all_ok = False

    return all_ok

def check_pytorch():
    """Check PyTorch and MPS availability."""
    try:
        import torch
        print(f"\n✓ PyTorch version: {torch.__version__}")

        # Check backends
        if torch.backends.mps.is_available():
            print("✓ MPS (Metal) backend: Available")
            device = "mps"
        elif torch.cuda.is_available():
            print("✓ CUDA backend: Available")
            device = "cuda"
        else:
            print("⚠ Using CPU backend (training will be slower)")
            device = "cpu"

        # Test tensor creation
        x = torch.randn(2, 2, device=device)
        print(f"✓ PyTorch tensors working on {device}")

        return True
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def check_directories():
    """Check required directories."""
    base_dir = Path(__file__).parent
    required_dirs = [
        'src/data',
        'src/model',
        'src/search',
        'src/play',
        'src/utils',
        'notebooks',
        'artifacts/weights',
        'artifacts/logs',
        'artifacts/matches',
        'reports/figures',
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
            all_ok = False

    return all_ok

def check_stockfish():
    """Check if Stockfish is available."""
    common_paths = [
        '/opt/homebrew/bin/stockfish',
        '/usr/local/bin/stockfish',
        '/usr/bin/stockfish',
        '/usr/games/stockfish',
    ]

    for path in common_paths:
        if Path(path).exists():
            print(f"✓ Stockfish found at: {path}")
            return True

    print("⚠ Stockfish not found (optional, but recommended for benchmarks)")
    print("  Install with: brew install stockfish (macOS)")
    print("  or: apt-get install stockfish (Linux)")
    return False

def test_basic_functionality():
    """Test basic functionality."""
    try:
        import sys
        sys.path.append(str(Path(__file__).parent / 'src'))

        # Test chess library
        import chess
        import chess.pgn  # Must import separately

        # Test encoding
        from utils.encoding import board_to_tensor, move_to_index

        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (12, 8, 8), f"Board encoding failed: expected (12, 8, 8), got {tensor.shape}"

        move = chess.Move.from_uci("e2e4")
        idx = move_to_index(move)
        assert 0 <= idx < 4672, f"Move encoding failed: index {idx} out of range"

        print("\n✓ Basic functionality tests passed")
        return True
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("Chess Deep Learning Agent - Setup Verification")
    print("=" * 60)

    print("\n1. Checking Python version...")
    py_ok = check_python_version()

    print("\n2. Checking required packages...")
    imports_ok = check_imports()

    print("\n3. Checking PyTorch...")
    torch_ok = check_pytorch()

    print("\n4. Checking directory structure...")
    dirs_ok = check_directories()

    print("\n5. Checking Stockfish...")
    stockfish_ok = check_stockfish()

    print("\n6. Testing basic functionality...")
    func_ok = test_basic_functionality()

    print("\n" + "=" * 60)
    if all([py_ok, imports_ok, torch_ok, dirs_ok, func_ok]):
        print("✓ All checks passed! You're ready to go.")
        print("\nNext steps:")
        print("  1. Download Lichess PGN data (see README.md)")
        print("  2. Run notebooks in order:")
        print("     - 01_eda_and_preprocessing.ipynb")
        print("     - 02_train_supervised.ipynb")
        print("     - 03_search_and_play.ipynb")
        print("     - 04_benchmarks_and_analysis.ipynb")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        if not imports_ok:
            print("\nTo install missing packages:")
            print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
