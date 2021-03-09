"""
Loss functions
"""

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """
    The soft dice loss function.

    Attributes
    ----------
    smooth : float (default=1.)
        The smoothing value.

    Methods
    -------
    forward(preds, targets)
        Compute the soft dice loss between the predictions and targets.
    """

    def __init__(self, smooth=1.):
        super().__init__()

        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute the soft dice loss between the prediction and targets.

        Parameters
        ----------
        preds : Tensor
            The predicted masks.
        targets : Tensor
            The ground-truth masks.

        Return
        ------
        dice : float
            The soft dice loss between the predictions and the targets.
        """

        intersection = (preds * targets).sum()
        denominator = preds.sum() + targets.sum() + self.smooth
        dice = (2. * intersection + self.smooth) / denominator

        return 1. - dice


class Loss(nn.Module):
    """
    The loss function of the NuClick model.

    Methods
    -------
    forward(preds, targets)
        Compute the loss between the predictions and targets.

    Notes
    -----
    The loss function is a combination of the soft dice loss with a weighted
    binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()

        self.dice = SoftDiceLoss()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, preds, targets):
        """
        Compute the loss between the prediction and targets.

        Parameters
        ----------
        preds : Tensor
            The predicted masks.
        targets : Tensor
            The ground-truth masks.

        Return
        ------
        loss : float
            The loss between the predictions and the targets.
        """

        # Compute the complement mask
        complement = torch.logical_not(targets).float()

        # Compute the alpha factor
        alpha = max(complement.sum() / targets.sum(), 1)

        # Compute the adaptive weight
        weights = alpha**2 * targets + alpha * complement + 1

        # Compute the weighted cross entropy loss
        bce_loss = (weights * self.bce(preds, targets)).mean()

        return self.dice(preds, targets) + bce_loss
