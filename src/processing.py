"""
Pre-processing and post-processing
"""

import numpy as np
import scipy.ndimage as ndi
import torch

from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import (
    skeletonize_3d, remove_small_objects, remove_small_holes
)


def create_inclusion_map(mask):
    """Create an inclusion map given a mask.

    Parameters:
    -----------
    mask : torch tensor
        A binary mask.

    Return:
    -------
    inclusion : torch tensor
        The inclusion map.
    """
    # Compute the Euclidian distance transform of the mask
    edt = distance_transform_edt(mask)

    # Compute its mean and standard deviation
    mean = ndi.mean(edt)
    std = ndi.standard_deviation(edt)

    # Compute the threshold tau
    threshold = np.random.uniform(0, mean + std)

    # Compute the new mask based on the threshold
    mask = edt > threshold

    # Apply a morphological skeleton on the new mask
    inclusion = skeletonize_3d(mask)

    return torch.tensor(inclusion)


def create_exclusion_map(mask):
    """Create an exclusion map given a mask.

    Parameters:
    -----------
    mask : torch tensor
        A binary mask.

    Return:
    -------
    exclusion : torch tensor
        The exclusion map.
    """
    return torch.zeros(mask.shape)


def post_process(preds, threshold=0.5, min_size=10, area_threshold=30):
    """Post-process a prediction.

    Parameters:
    -----------
    preds : torch tensor
        The predicted mask.
    threshold : int (default=0.5)
        The threshold to remove small predictions.
    min_size : int (default=10)
        The minimum size of an object.
    area_threshold : int (default=30)
        The minimum area of an object.

    Return:
    -------
    masks : torch tensor
        The final prediction mask.
    """
    # Remove small output number
    masks = preds > threshold

    # Remove small objects
    masks = remove_small_objects(masks, min_size=min_size)

    # Remove small holes
    masks = remove_small_holes(masks, area_threshold=area_threshold)

    return masks
