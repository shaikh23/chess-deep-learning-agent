"""Reproducibility utilities for setting random seeds."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        # MPS (Metal) backend for Mac
        torch.mps.manual_seed(seed)

    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test seed setting
    set_seed(42)
    print(f"Random int: {random.randint(0, 100)}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")

    # Reset and verify reproducibility
    set_seed(42)
    print(f"Random int (again): {random.randint(0, 100)}")
    print(f"NumPy random (again): {np.random.rand()}")
    print(f"PyTorch random (again): {torch.rand(1).item()}")
