"""
Loss functions
"""

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum()
        denominator = preds.sum() + targets.sum() + self.smooth
        dice = (2 * intersection + self.smooth) / denominator

        return 1 - dice


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        self.dice = SoftDiceLoss()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, preds, targets):
        # Compute the complement mask
        complement = torch.logical_not(targets).float()

        # Compute the alpha factor
        alpha = max(complement.sum() / targets.sum(), 1)

        # Compute the adaptive weight
        weights = alpha**2 * targets + alpha * complement + 1

        # Compute the weighted cross entropy loss
        bce_loss = (weights * self.bce(preds, targets)).mean()

        return self.dice(preds, targets) + bce_loss
