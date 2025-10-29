"""Neural network architectures for chess: MLP, CNN, and ResNet variants.

All models output:
- Policy logits: [batch_size, 4672] (move indices)
- Value: [batch_size, 1] (position evaluation in range [-1, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.encoding import MOVE_INDEX_SIZE


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log-softmax with masking for illegal moves.

    Args:
        logits: Raw logits [batch_size, num_moves]
        mask: Boolean mask [batch_size, num_moves] where True = legal
        dim: Dimension to apply softmax

    Returns:
        Log probabilities [batch_size, num_moves]
    """
    # Mask illegal moves with large negative value
    masked_logits = logits.masked_fill(~mask, -1e9)
    return F.log_softmax(masked_logits, dim=dim)


class PolicyHead(nn.Module):
    """Policy head that outputs move probabilities with legal move masking."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = MOVE_INDEX_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional legal move masking.

        Args:
            x: Input features [batch_size, input_dim]
            legal_mask: Boolean mask [batch_size, output_dim] for legal moves

        Returns:
            Policy logits [batch_size, output_dim]
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        # Apply legal move mask (set illegal moves to large negative value)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)

        return logits


class ValueHead(nn.Module):
    """Value head that estimates position evaluation."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate position value.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Value estimate [batch_size, 1] in range [-1, 1]
        """
        x = F.relu(self.fc1(x))
        value = torch.tanh(self.fc2(x))
        return value


class MLPPolicy(nn.Module):
    """
    Simple MLP baseline for policy prediction only.

    Input: Flattened board [batch_size, 12*8*8 = 768]
    Output: Policy logits [batch_size, 4672]
    """

    def __init__(
        self,
        input_dim: int = 12 * 8 * 8,
        hidden_dims: Tuple[int, ...] = (1024, 512, 512),
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.policy_head = PolicyHead(prev_dim)

    def forward(self, board: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            board: Board tensor [batch_size, 12, 8, 8]
            legal_mask: Optional legal move mask

        Returns:
            Policy logits [batch_size, 4672]
        """
        # Flatten board
        x = board.reshape(board.size(0), -1)

        # Backbone
        x = self.backbone(x)

        # Policy head
        policy_logits = self.policy_head(x, legal_mask)

        return policy_logits


class MLPPolicyValue(nn.Module):
    """
    MLP with both policy and value heads.

    Input: Flattened board [batch_size, 12*8*8 = 768]
    Output: (policy_logits [batch_size, 4672], value [batch_size, 1])
    """

    def __init__(
        self,
        input_dim: int = 12 * 8 * 8,
        hidden_dims: Tuple[int, ...] = (1024, 512, 512),
        policy_head_hidden: int = 512,
        value_head_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.policy_head = PolicyHead(prev_dim, policy_head_hidden)
        self.value_head = ValueHead(prev_dim, value_head_hidden)

    def forward(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            board: Board tensor [batch_size, 12, 8, 8]
            legal_mask: Optional legal move mask
            return_value: Whether to compute value head

        Returns:
            Tuple of (policy_logits, value) or just policy_logits if return_value=False
        """
        # Flatten board
        x = board.reshape(board.size(0), -1)

        # Backbone
        x = self.backbone(x)

        # Policy head
        policy_logits = self.policy_head(x, legal_mask)

        if return_value:
            # Value head
            value = self.value_head(x)
            return policy_logits, value
        else:
            return policy_logits, None


class CNNPolicyValue(nn.Module):
    """
    CNN architecture with policy and value heads.

    Similar to early AlphaGo Zero but much smaller.
    """

    def __init__(
        self,
        num_channels: int = 128,
        num_layers: int = 4,
        policy_head_hidden: int = 512,
        value_head_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(12, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Residual tower (simplified - just conv blocks)
        conv_blocks = []
        for _ in range(num_layers):
            conv_blocks.extend([
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(),
                nn.Dropout2d(dropout),
            ])

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Compute flattened dimension after convolutions
        # Board is 8x8, channels = num_channels
        flat_dim = num_channels * 8 * 8

        # Policy and value heads
        self.policy_head = PolicyHead(flat_dim, policy_head_hidden)
        self.value_head = ValueHead(flat_dim, value_head_hidden)

    def forward(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            board: Board tensor [batch_size, 12, 8, 8]
            legal_mask: Optional legal move mask
            return_value: Whether to compute value head

        Returns:
            Tuple of (policy_logits, value) or just policy_logits if return_value=False
        """
        # Convolutional backbone
        x = F.relu(self.bn_in(self.conv_in(board)))
        x = self.conv_blocks(x)

        # Flatten for heads
        x_flat = x.view(x.size(0), -1)

        # Policy head
        policy_logits = self.policy_head(x_flat, legal_mask)

        if return_value:
            # Value head
            value = self.value_head(x_flat)
            return policy_logits, value
        else:
            return policy_logits, None


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and skip connection."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out += residual
        out = F.relu(out)

        return out


class MiniResNetPolicyValue(nn.Module):
    """
    Mini ResNet architecture inspired by AlphaZero.

    Compact version suitable for training on MacBook.

    Default: 6 residual blocks × 64 channels = ~300K parameters
    """

    def __init__(
        self,
        num_blocks: int = 6,
        channels: int = 64,
        policy_head_hidden: int = 512,
        value_head_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(12, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, dropout) for _ in range(num_blocks)
        ])

        # Compute flattened dimension
        flat_dim = channels * 8 * 8

        # Policy and value heads
        self.policy_head = PolicyHead(flat_dim, policy_head_hidden)
        self.value_head = ValueHead(flat_dim, value_head_hidden)

    def forward(
        self,
        board: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        return_value: bool = True,
        return_logprobs: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with flexible output options.

        Args:
            board: Board tensor [batch_size, 12, 8, 8]
            legal_mask: Optional legal move mask [batch_size, MOVE_INDEX_SIZE]
            return_value: Whether to compute value head
            return_logprobs: Whether to compute log probabilities

        Returns:
            Tuple of (policy_logits, log_probs, value)
            - log_probs and value are None if not requested
        """
        # Initial convolution
        x = F.relu(self.bn_in(self.conv_in(board)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Flatten (use reshape for channels_last compatibility)
        x_flat = x.reshape(x.size(0), -1)

        # Policy head (returns masked logits)
        policy_logits = self.policy_head(x_flat, legal_mask)

        # Compute log probabilities if requested
        log_probs = None
        if return_logprobs and legal_mask is not None:
            log_probs = masked_log_softmax(policy_logits, legal_mask)

        # Value head
        value = None
        if return_value:
            value = self.value_head(x_flat)

        return policy_logits, log_probs, value

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Kaiming initialization.

    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ============================================================================
# Unit Tests
# ============================================================================

def _test_models():
    """Test all model architectures."""
    print("Testing model architectures...")

    batch_size = 8
    board = torch.randn(batch_size, 12, 8, 8)
    legal_mask = torch.randint(0, 2, (batch_size, MOVE_INDEX_SIZE), dtype=torch.bool)

    # Test MLP (policy only)
    print("\n1. Testing MLPPolicy...")
    mlp = MLPPolicy()
    initialize_weights(mlp)
    policy_logits = mlp(board, legal_mask)
    assert policy_logits.shape == (batch_size, MOVE_INDEX_SIZE)
    print(f"   ✓ Output shape: {policy_logits.shape}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in mlp.parameters()):,}")

    # Test MLP (policy + value)
    print("\n2. Testing MLPPolicyValue...")
    mlp_pv = MLPPolicyValue()
    initialize_weights(mlp_pv)
    policy_logits, value = mlp_pv(board, legal_mask, return_value=True)
    assert policy_logits.shape == (batch_size, MOVE_INDEX_SIZE)
    assert value.shape == (batch_size, 1)
    print(f"   ✓ Policy shape: {policy_logits.shape}")
    print(f"   ✓ Value shape: {value.shape}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in mlp_pv.parameters()):,}")

    # Test CNN
    print("\n3. Testing CNNPolicyValue...")
    cnn = CNNPolicyValue(num_channels=64, num_layers=3)
    initialize_weights(cnn)
    policy_logits, value = cnn(board, legal_mask, return_value=True)
    assert policy_logits.shape == (batch_size, MOVE_INDEX_SIZE)
    assert value.shape == (batch_size, 1)
    print(f"   ✓ Policy shape: {policy_logits.shape}")
    print(f"   ✓ Value shape: {value.shape}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    # Test Mini ResNet
    print("\n4. Testing MiniResNetPolicyValue...")
    resnet = MiniResNetPolicyValue(num_blocks=6, channels=64)
    initialize_weights(resnet)
    policy_logits, value = resnet(board, legal_mask, return_value=True)
    assert policy_logits.shape == (batch_size, MOVE_INDEX_SIZE)
    assert value.shape == (batch_size, 1)
    print(f"   ✓ Policy shape: {policy_logits.shape}")
    print(f"   ✓ Value shape: {value.shape}")
    print(f"   ✓ Parameters: {resnet.count_parameters():,}")

    # Test value range
    print("\n5. Testing value head range...")
    assert (value >= -1.0).all() and (value <= 1.0).all(), "Value out of range [-1, 1]"
    print(f"   ✓ Value range: [{value.min():.3f}, {value.max():.3f}]")

    print("\nAll model tests passed! ✓")


if __name__ == "__main__":
    _test_models()
