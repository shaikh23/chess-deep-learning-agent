"""Loss functions for chess policy and value network training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Prevents overconfidence by distributing some probability mass
    to non-target classes.
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing CE loss.

        Args:
            smoothing: Smoothing factor in [0, 1]. 0 = no smoothing (standard CE)
        """
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]

        Returns:
            Scalar loss
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Standard NLL loss with confidence
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # Smoothing loss (uniform distribution over all classes)
        smooth_loss = -log_probs.mean(dim=-1)

        # Combine losses
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


class PolicyValueLoss(nn.Module):
    """
    Combined loss for policy and value network training.

    Loss = policy_loss + lambda * value_loss

    - Policy loss: Cross-entropy with label smoothing
    - Value loss: Mean squared error
    """

    def __init__(
        self,
        value_weight: float = 0.35,
        policy_smoothing: float = 0.05,
    ):
        """
        Initialize combined loss.

        Args:
            value_weight: Weight for value loss component (lambda). Default 0.35 optimized for supervised pretraining.
            policy_smoothing: Label smoothing factor for policy CE
        """
        super().__init__()
        self.value_weight = value_weight
        self.policy_loss_fn = LabelSmoothingCrossEntropy(smoothing=policy_smoothing)

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: Optional[torch.Tensor],
        policy_targets: torch.Tensor,
        value_targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            policy_logits: Policy predictions [batch_size, num_moves]
            value_pred: Value predictions [batch_size, 1] or None
            policy_targets: Move indices [batch_size]
            value_targets: Outcome values [batch_size] or None

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Policy loss
        policy_loss = self.policy_loss_fn(policy_logits, policy_targets)

        loss_dict = {
            "policy_loss": policy_loss.item(),
        }

        total_loss = policy_loss

        # Value loss (if provided)
        if value_pred is not None and value_targets is not None:
            # MSE between predicted value and outcome
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)

            loss_dict["value_loss"] = value_loss.item()
            total_loss = total_loss + self.value_weight * value_loss

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in move prediction.

    Focuses training on hard examples by down-weighting easy examples.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth indices [batch_size]

        Returns:
            Scalar loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Get probabilities of target classes
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1.0 - target_probs) ** self.gamma

        # NLL loss
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Combine
        loss = self.alpha * focal_weight * nll_loss

        return loss.mean()


class ValueLoss(nn.Module):
    """
    Standalone value loss with optional Huber loss for robustness.
    """

    def __init__(self, use_huber: bool = False, delta: float = 1.0):
        """
        Initialize value loss.

        Args:
            use_huber: Use Huber loss instead of MSE
            delta: Huber loss delta parameter
        """
        super().__init__()
        self.use_huber = use_huber
        self.delta = delta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute value loss.

        Args:
            predictions: Value predictions [batch_size, 1]
            targets: Ground truth values [batch_size]

        Returns:
            Scalar loss
        """
        predictions = predictions.squeeze(-1)

        if self.use_huber:
            return F.huber_loss(predictions, targets, delta=self.delta)
        else:
            return F.mse_loss(predictions, targets)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth indices [batch_size]
        k: Number of top predictions to consider

    Returns:
        Accuracy as float in [0, 1]
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = logits.topk(k, dim=-1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return (correct_k.item() / batch_size)


def calibration_bins(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute calibration bins for model confidence analysis.

    Args:
        probs: Predicted probabilities [batch_size, num_classes]
        targets: Ground truth indices [batch_size]
        num_bins: Number of calibration bins

    Returns:
        Tuple of (bin_confidences, bin_accuracies, bin_counts)
    """
    with torch.no_grad():
        # Get max probability and predicted class
        max_probs, preds = probs.max(dim=-1)

        # Check if predictions are correct
        correct = (preds == targets).float()

        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_confidences = torch.zeros(num_bins)
        bin_accuracies = torch.zeros(num_bins)
        bin_counts = torch.zeros(num_bins)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            bin_count = in_bin.float().sum()

            if bin_count > 0:
                bin_confidences[i] = max_probs[in_bin].mean()
                bin_accuracies[i] = correct[in_bin].mean()
                bin_counts[i] = bin_count

        return bin_confidences, bin_accuracies, bin_counts


# ============================================================================
# Unit Tests
# ============================================================================

def _test_losses():
    """Test loss functions."""
    print("Testing loss functions...")

    batch_size = 32
    num_classes = 4672

    # Create synthetic data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    value_pred = torch.randn(batch_size, 1)
    value_targets = torch.randn(batch_size)

    # Test label smoothing CE
    print("\n1. Testing LabelSmoothingCrossEntropy...")
    ls_ce = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = ls_ce(logits, targets)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"
    print(f"   ✓ Loss value: {loss.item():.4f}")

    # Test standard CE (no smoothing)
    ce = LabelSmoothingCrossEntropy(smoothing=0.0)
    loss_no_smooth = ce(logits, targets)
    print(f"   ✓ Loss without smoothing: {loss_no_smooth.item():.4f}")

    # Test policy-value loss
    print("\n2. Testing PolicyValueLoss...")
    pv_loss = PolicyValueLoss(value_weight=1.0, policy_smoothing=0.1)
    total_loss, loss_dict = pv_loss(logits, value_pred, targets, value_targets)

    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "total_loss" in loss_dict

    print(f"   ✓ Policy loss: {loss_dict['policy_loss']:.4f}")
    print(f"   ✓ Value loss: {loss_dict['value_loss']:.4f}")
    print(f"   ✓ Total loss: {loss_dict['total_loss']:.4f}")

    # Test policy-only loss
    print("\n3. Testing policy-only loss...")
    total_loss, loss_dict = pv_loss(logits, None, targets, None)
    assert "value_loss" not in loss_dict
    print(f"   ✓ Policy-only loss: {loss_dict['total_loss']:.4f}")

    # Test focal loss
    print("\n4. Testing FocalLoss...")
    focal = FocalLoss(alpha=1.0, gamma=2.0)
    loss = focal(logits, targets)
    assert loss.ndim == 0
    print(f"   ✓ Focal loss: {loss.item():.4f}")

    # Test value loss
    print("\n5. Testing ValueLoss...")
    value_loss_fn = ValueLoss(use_huber=False)
    loss = value_loss_fn(value_pred, value_targets)
    print(f"   ✓ MSE value loss: {loss.item():.4f}")

    value_loss_fn_huber = ValueLoss(use_huber=True, delta=1.0)
    loss_huber = value_loss_fn_huber(value_pred, value_targets)
    print(f"   ✓ Huber value loss: {loss_huber.item():.4f}")

    print("\nAll loss tests passed! ✓")


if __name__ == "__main__":
    _test_losses()
