"""
Loss Functions for Multi-Label Classification
- Focal Loss with per-class alpha
- Weighted BCEWithLogitsLoss
- Label smoothing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Focuses on hard examples by down-weighting easy ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Per-class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (higher = more focus on hard examples)
        label_smoothing: Label smoothing value
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (batch, num_classes)
            targets: Ground truth (batch, num_classes)
        
        Returns:
            Focal loss value
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Sigmoid probabilities
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Focal weight: (1 - p_t)^gamma
        # p_t = p if target=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha (per-class weight)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            # alpha_t = alpha if target=1, else 1-alpha (but we use full alpha for both)
            alpha_weight = alpha.unsqueeze(0)  # (1, num_classes)
            focal_loss = alpha_weight * focal_weight * bce
        else:
            focal_loss = focal_weight * bce
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss with pos_weight support.
    
    Args:
        pos_weight: Positive weights per class (neg_count / pos_count)
        label_smoothing: Label smoothing value
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (batch, num_classes)
            targets: Ground truth (batch, num_classes)
        
        Returns:
            Loss value
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # BCE with pos_weight
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight,
            reduction=self.reduction
        )
        
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Better handles the class imbalance by asymmetrically focusing on
    positive vs negative examples. Good for rare positive labels.
    
    ASL = L+ * y + L- * (1-y)
    where L+ = (1-p)^gamma+ * log(p)
          L- = p^gamma- * log(1-p)
    
    Args:
        gamma_neg: Focusing parameter for negatives
        gamma_pos: Focusing parameter for positives
        clip: Probability margin for negatives
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # Clipping for negatives
        probs_neg = probs.clamp(max=1 - self.clip)
        
        # Losses
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        
        # Asymmetric focusing
        loss_pos = loss_pos * ((1 - probs) ** self.gamma_pos)
        loss_neg = loss_neg * (probs_neg ** self.gamma_neg)
        
        loss = -(loss_pos + loss_neg)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def create_loss(
    loss_type: str = "focal",
    alpha: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05
) -> nn.Module:
    """
    Create loss function.
    
    Args:
        loss_type: 'focal', 'bce', or 'asymmetric'
        alpha: Per-class weights for focal loss
        pos_weight: Positive weights for BCE
        gamma: Focusing parameter
        label_smoothing: Label smoothing value
    
    Returns:
        Loss function
    """
    if loss_type == "focal":
        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing
        )
    elif loss_type == "bce":
        return WeightedBCELoss(
            pos_weight=pos_weight,
            label_smoothing=label_smoothing
        )
    elif loss_type == "asymmetric":
        return AsymmetricLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    num_classes = 13
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test Focal Loss
    alpha = torch.ones(num_classes)
    focal = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.05)
    loss_focal = focal(logits, targets)
    print(f"Focal Loss: {loss_focal.item():.4f}")
    
    # Test Weighted BCE
    pos_weight = torch.ones(num_classes) * 2
    bce = WeightedBCELoss(pos_weight=pos_weight, label_smoothing=0.05)
    loss_bce = bce(logits, targets)
    print(f"Weighted BCE Loss: {loss_bce.item():.4f}")
    
    # Test Asymmetric Loss
    asl = AsymmetricLoss()
    loss_asl = asl(logits, targets)
    print(f"Asymmetric Loss: {loss_asl.item():.4f}")
