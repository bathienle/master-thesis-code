"""
Metrics for the evaluation of the model
"""

import torch
import torch.nn as nn

from scipy.ndimage.morphology import distance_transform_edt


class IoU(nn.Module):
    """
    Class computing the intersection over the union (IoU).

    Attributes
    ----------
    smooth : float (default=1.)
        The smoothing value.
    """

    def __init__(self, smooth=1.):
        super().__init__()

        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute the IoU between the predictions and targets.

        Parameters
        ----------
        preds : Tensor
            The predicted masks.
        targets : Tensor
            The ground truth masks.

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
    Class computing the dice coefficient.

    Attributes
    ----------
    smooth : float (default=1.)
        The smoothing value.
    """

    def __init__(self, smooth=1.):
        super().__init__()

        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute the dice coefficient between the prediction and targets.

        Parameters
        ----------
        preds : Tensor
            The predicted masks.
        targets : Tensor
            The ground truth masks.

        Return
        ------
        dice : float
            The dice coefficient between the predictions and the targets.
        """

        intersection = (preds * targets).sum()
        denominator = preds.sum() + targets.sum() + self.smooth

        return (2. * intersection + self.smooth) / denominator


class HausdorffDistance(nn.Module):
    """
    Class computing the Hausdorff distance.

    Notes
    -----
    Reference paper implementation: https://arxiv.org/pdf/1904.10030.pdf
    """

    def hd(self, p, q):
        """
        Compute the Hausdorff distance between two masks.

        Parameters
        ----------
        p : Tensor
            The first mask.
        q : Tensor
            The second mask.

        Return
        ------
        hausdorff : float
            The hausdorff distance between p and q.
        """

        edt = torch.as_tensor(distance_transform_edt(q), dtype=torch.float32)

        return torch.max(torch.abs(p - q) * edt)

    def forward(self, preds, targets):
        """
        Compute the Hausdorff distance between the predictions and targets.

        Parameters
        ----------
        preds : Tensor
            The predicted masks.
        targets : Tensor
            The ground truth masks.

        Return
        ------
        hausdorff : float
            The hausdorff distance between the predictions and the targets.
        """

        # Distance transform is not supported on GPU!
        preds, targets = preds.cpu(), targets.cpu()

        return torch.max(self.hd(preds, targets), self.hd(targets, preds))
