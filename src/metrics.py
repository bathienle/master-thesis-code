"""
Metrics for the evaluation of the model
"""

import torch.nn as nn


class IoU(nn.Module):
    """
    Class computing the intersection over the union (IoU) metric.

    Attributes
    ----------
    smooth : float
        The smoothing value.

    Methods
    -------
    forward(preds, targets)
        Compute the IoU between the predictions and targets.
    """

    def __init__(self, smooth=1.):
        """
        Initialises the class.

        Parameters
        ----------
        smooth : float (default=1.)
            The smooting value.
        """

        super(IoU, self).__init__()

        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute the IoU between the prediction and targets.

        Parameters
        ----------
        preds : torch tensor
            The predicted masks.
        targets : torch tensor
            The ground-truth masks.

        Return
        ------
        iou : float
            The IoU between the predictions and the targets.
        """

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        return (intersection + self.smooth) / (union + self.smooth)


class DiceCoefficient(nn.Module):
    """
    Class computing the dice coefficient metric.

    Attributes
    ----------
    smooth : float
        The smoothing value.

    Methods
    -------
    forward(preds, targets)
        Compute the dice coefficient between the predictions and targets.
    """

    def __init__(self, smooth=1.):
        """
        Initialises the class.

        Parameters
        ----------
        smooth : float (default=1.)
            The smooting value.
        """

        super(DiceCoefficient, self).__init__()

        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute the dice coefficient between the prediction and targets.

        Parameters
        ----------
        preds : torch tensor
            The predicted masks.
        targets : torch tensor
            The ground-truth masks.

        Return
        ------
        dice : float
            The dice coefficient between the predictions and the targets.
        """

        intersection = (preds * targets).sum()
        denominator = preds.sum() + targets.sum() + self.smooth

        return (2. * intersection + self.smooth) / denominator
