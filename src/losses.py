"""
Loss functions
"""

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
    def __init__(self, weight):
        super(Loss, self).__init__()

        self.dice = SoftDiceLoss()
        self.bce = nn.BCELoss(weight=weight)

    def forward(self, preds, targets):
        return self.dice(preds, targets) - self.bce(preds, targets)
