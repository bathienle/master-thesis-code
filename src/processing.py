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


def create_signal(mask):
    """
    Create the guiding signal.

    Parameters
    ----------
    mask : Tensor
        The binary mask to transform.

    Return
    ------
    signal : Tensor
        The guiding signal.
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
    signal = skeletonize_3d(mask)

    return torch.as_tensor(signal)


def post_process(preds, threshold=0.5, min_size=10, area_threshold=30):
    """
    Post-process the predictions.

    Parameters
    ----------
    preds : Tensor
        The predicted masks.
    threshold : float (default=0.5)
        The mininum threshold for the taking the output into account.
    min_size : int (default=10)
        The minimum size of a prediction.
    area_threshold : int (default=30)
        The minimum area of a prediction.

    Return
    ------
    masks : Tensor
        The post-processed predictions.
    """

    # Convert to CPU if needed
    device = preds.device
    preds = preds.cpu()

    # Remove small output number
    masks = (preds > threshold).numpy()

    # Remove small objects
    masks = remove_small_objects(masks, min_size=min_size)

    # Remove small holes
    masks = remove_small_holes(masks, area_threshold=area_threshold)

    return torch.as_tensor(masks, dtype=torch.float32, device=device)
