"""
Data augmentation functions
"""

import random
import torch
import torch.nn as nn


class Transform(nn.Module):
    """
    Apply a random transform on the input and target.

    Attributes
    ----------
    transforms : list
        A list of transforms for both the input and target.
    input_only : list
        A list of transforms only for the input.
    """

    def __init__(self, transforms=None, input_only=None):
        super().__init__()

        self.transforms = transforms
        self.input_only = input_only

    def forward(self, input, target):
        """
        Apply a random transform on the input and target.

        Parameters
        ----------
        input : Tensor
            The input to transform.
        target : Tensor
            The target to transform.

        Return
        ------
        input : Tensor
            The input after applying the transform.
        target : Tensor
            The target after applying the transform.
        """

        # Apply a transform for both the input and target
        if self.transforms:
            t = random.choice(self.transforms)

            return t(input, target)

        # Apply a transform on the input only
        if self.input_only:
            t = random.choice(self.input_only)

            return t(input), target

        return input, target


class RandomHorizontalFlip(nn.Module):
    """
    Flip horizontally a given image and its target.

    Attributes
    ----------
    p : float (default=0.5)
        The probability to flip the image and its target.
    """

    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def forward(self, image, target):
        """
        Flip the image and its target horizontally given a probability.

        Parameters
        ----------
        image : Tensor
            The image to flip.
        target : Tensor
            The target to flip.

        Return
        ------
        image : Tensor
            The flipped or original image given the probability.
        target : Tensor
            The flipped or original target given the probability.
        """

        if torch.rand(1) < self.p:
            return image.flip(-1), target.flip(-1)

        return image, target


class RandomVerticalFlip(nn.Module):
    """
    Flip vertically a given image and its target.

    Attributes
    ----------
    p : float (default=0.5)
        The probability to flip the image and its target.
    """

    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def forward(self, image, target):
        """
        Flip the image and its target vertically given a probability.

        Parameters
        ----------
        image : Tensor
            The image to flip.
        target : Tensor
            The target to flip.

        Return
        ------
        image : Tensor
            The flipped or original image given the probability.
        target : Tensor
            The flipped or original target given the probability.
        """

        if torch.rand(1) < self.p:
            return image.flip(-2), target.flip(-2)

        return image, target
