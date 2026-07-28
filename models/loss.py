import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=255):
        """
        Focal Loss for Dense Object Detection.
        alpha: Weighting factor for each class (tensor or list).
        gamma: Focusing parameter (default=2).
        ignore_index: Label to ignore in loss calculation.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) -> logits
        # targets: (B, H, W) -> class indices
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Mask out ignore_index for reduction (cross_entropy already sets ignored pixels to 0 loss if reduction='none'?)
        # Verified: F.cross_entropy with reduction='none' returns 0 for ignored indices.
        # So focal_loss will be 0 at those pixels.

        if self.reduction == 'mean':
            # We should only average over non-ignored pixels
            # Create mask of valid pixels
            valid_mask = (targets != self.ignore_index).float()
            return focal_loss.sum() / (valid_mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=None, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) -> logits
        # targets: (B, H, W) -> labels
        
        inputs = F.softmax(inputs, dim=1)
        
        # Clone targets to avoid modifying original
        targets_copy = targets.clone()
        # Replace ignore_index with a dummy valid class (0) for one_hot, then mask it out later
        # Or better: create a mask
        valid_mask = (targets != self.ignore_index)
        targets_copy[~valid_mask] = 0 # Safe dummy
        
        # One-hot encode
        targets_one_hot = F.one_hot(targets_copy, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply mask to both inputs and targets (zero out ignored regions)
        # valid_mask is (B, H, W), unsqueeze to (B, 1, H, W)
        mask_expanded = valid_mask.unsqueeze(1).float()
        
        inputs = inputs * mask_expanded
        targets_one_hot = targets_one_hot * mask_expanded
        
        # Intersection and Union
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over batch and classes (1 - dice)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, dice_weight=0.5, num_classes=None, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight # Weight for dice term (0.0 to 1.0)

    def forward(self, inputs, targets):
        focal = self.focal(inputs, targets)
        dice = self.dice(inputs, targets)
        
        # Weighted sum
        return (1 - self.dice_weight) * focal + self.dice_weight * dice
