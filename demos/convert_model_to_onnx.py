"""Convert PyTorch chess model to ONNX format for web deployment.

This script converts the trained PyTorch model to ONNX format,
which can be run in web browsers using ONNX Runtime Web.

Usage:
    python convert_model_to_onnx.py --input artifacts/weights/best_model.pth --output web/model.onnx
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.model.nets import MiniResNetPolicyValue


def convert_model_to_onnx(
    input_path: str,
    output_path: str,
    model_config: dict = None
):
    """Convert PyTorch model to ONNX format.

    Args:
        input_path: Path to PyTorch model (.pth)
        output_path: Path to save ONNX model (.onnx)
        model_config: Model architecture configuration
    """
    # Default model config
    if model_config is None:
        model_config = {
            "num_blocks": 6,
            "channels": 64,
            "policy_head_hidden": 512,
            "value_head_hidden": 256,
            "dropout": 0.1
        }

    print("=" * 60)
    print("PyTorch to ONNX Model Conversion")
    print("=" * 60)

    # Create model
    print(f"\n1. Creating model architecture...")
    model = MiniResNetPolicyValue(**model_config)
    print(f"   Model parameters: {model.count_parameters():,}")

    # Load weights
    print(f"\n2. Loading weights from {input_path}...")
    device = torch.device("cpu")  # Use CPU for ONNX export
    checkpoint = torch.load(input_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   Loaded from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("   Loaded from 'state_dict'")
        else:
            model.load_state_dict(checkpoint)
            print("   Loaded directly")
    else:
        model.load_state_dict(checkpoint)
        print("   Loaded directly")

    model.eval()
    print("   Model loaded successfully!")

    # Create dummy input
    print(f"\n3. Creating dummy input...")
    batch_size = 1
    dummy_board = torch.randn(batch_size, 12, 8, 8)
    dummy_legal_mask = torch.ones(batch_size, 4672, dtype=torch.bool)

    print(f"   Board shape: {dummy_board.shape}")
    print(f"   Legal mask shape: {dummy_legal_mask.shape}")

    # Export to ONNX
    print(f"\n4. Exporting to ONNX format...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic axes for variable batch size
    dynamic_axes = {
        'board': {0: 'batch_size'},
        'legal_mask': {0: 'batch_size'},
        'policy_logits': {0: 'batch_size'},
        'value': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        (dummy_board, dummy_legal_mask, True, False),  # args
        str(output_path),
        export_params=True,
        opset_version=14,  # Use opset 14 for better browser compatibility
        do_constant_folding=True,
        input_names=['board', 'legal_mask'],
        output_names=['policy_logits', 'value'],  # Only 2 outputs (log_probs is None)
        dynamic_axes=dynamic_axes,
    )

    # Check file size
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   ONNX model saved to: {output_path}")
    print(f"   File size: {file_size:.2f} MB")

    # Verify the exported model
    print(f"\n5. Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model is valid!")

        # Print model info
        print(f"\n6. Model Information:")
        print(f"   Inputs:")
        for inp in onnx_model.graph.input:
            print(f"     - {inp.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]}")
        print(f"   Outputs:")
        for out in onnx_model.graph.output:
            print(f"     - {out.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]}")

    except ImportError:
        print("   ⚠ onnx package not installed. Install with: pip install onnx")
        print("   Skipping verification (model should still work)")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. The ONNX model is saved at: {output_path}")
    print(f"2. Use it in the web app with ONNX Runtime Web")
    print(f"3. Deploy to a web server or test locally")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch chess model to ONNX format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/weights/best_model.pth",
        help="Path to input PyTorch model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="web/model.onnx",
        help="Path to output ONNX model"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=6,
        help="Number of residual blocks (must match trained model)"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Number of channels (must match trained model)"
    )

    args = parser.parse_args()

    model_config = {
        "num_blocks": args.num_blocks,
        "channels": args.channels,
        "policy_head_hidden": 512,
        "value_head_hidden": 256,
        "dropout": 0.1
    }

    convert_model_to_onnx(args.input, args.output, model_config)


if __name__ == "__main__":
    main()
