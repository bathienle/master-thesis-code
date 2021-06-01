"""
Squiggle representing a geometric shape generator
"""

import cv2
import numpy as np

from itertools import combinations
from math import ceil, dist


def generate_shape(mask, contour, shape='c'):
    """
    Generate a squiggle in a geometric shape

    Parameters
    ----------
    mask : NumPy array
        The mask of an object.
    shape : string (default='circle')
        The shape to use for the squiggle.

    Return
    ------
    squiggle : NumPy array
        The generated shape inside the mask.

    Notes
    -----
    The available shape are 'c' circle and 's' square.
    """

    # Compute the image moments of the contour
    moments = cv2.moments(contour)

    # Compute the centroid of the mask
    xc = int(moments['m10'] // moments['m00'])
    yc = int(moments['m01'] // moments['m00'])

    squiggle = np.zeros(mask.shape, dtype=mask.dtype)

    # Get the four extreme points of the mask
    points = [
        tuple(contour[contour[:, :, 0].argmin()][0]),
        tuple(contour[contour[:, :, 0].argmax()][0]),
        tuple(contour[contour[:, :, 1].argmin()][0]),
        tuple(contour[contour[:, :, 1].argmax()][0]),
    ]

    # Compute the radius of the circle
    distances = [ceil(dist(a, b)) for (a, b) in combinations(points, 2)]
    sigma = ceil(np.std(distances))

    if shape == 'c':
        radius = np.random.randint(sigma // 2, sigma)

        cv2.circle(squiggle, (xc, yc), radius, 255, 2)
    elif shape == 's':
        length = np.random.randint(sigma, 1.5 * sigma)
        half = length // 2
        topleft = (xc - half, yc - half)
        botright = (xc + half, yc + half)

        cv2.rectangle(squiggle, topleft, botright, 255, 2)

    return squiggle
