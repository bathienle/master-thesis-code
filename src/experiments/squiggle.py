"""
Random squiggles generator
"""

import numpy as np

from math import ceil, dist
from skimage.draw import disk, line_aa


def generate_squiggle(mask, origin, dest, min_line, n_step=5):
    """
    Generate a random squiggle from a source point to a destination point
    inside a given mask.

    Parameters
    ----------
    mask : NumPy array
        The mask of an object.
    origin : tuple of int
        The source coordinate.
    dest : tuple of int
        The destination coordinate.
    min_line : int
        The minimum length of a line segment.
    n_steps : int (default=5)
        The number of line segment to draw.

    Return
    ------
    squiggle : NumPy array
        The random generated squiggle.
    """

    circleA = np.zeros(mask.shape, dtype=mask.dtype)
    circleB = np.zeros(mask.shape, dtype=mask.dtype)
    squiggle = np.zeros(mask.shape, dtype=mask.dtype)

    # Randomize the step
    steps = [i for i in range(1, n_step)]
    np.random.shuffle(steps)

    points = {}
    prev_step = 0
    curr_x, curr_y = origin
    xb, yb = dest
    outside = False

    for step in steps:
        # Choose either the origin or the previous point as starting point
        if prev_step > step:
            if step-1 in points:
                curr_x, curr_y = points[step-1]
            else:
                curr_x, curr_y = origin

        # Compute the radius of the two circle
        radiusA = step * min_line
        radiusB = (n_step - step) * min_line

        # Clear the two arrays
        circleA.fill(0)
        circleB.fill(0)

        # Create the first circle
        rr, cc = disk((curr_y, curr_x), radiusA+1, shape=circleA.shape)
        circleA[rr, cc] = 255

        # Create the second circle
        rr, cc = disk((yb, xb), radiusB+1, shape=circleB.shape)
        circleB[rr, cc] = 255

        # Get the intersection of the two circles and the mask of the object
        image = circleA & circleB & mask

        # Get the coordinates of the nonzero value
        indices = np.nonzero(image)

        if len(indices[0]) == 0 or len(indices[1]) == 0:
            outside = True
            indices = np.nonzero(circleA & circleB)

        # Sample the next point inside the intersection region
        next_x = np.random.choice(indices[1])
        next_y = np.random.choice(indices[0])
        da = ceil(dist((next_x, next_y), (curr_x, curr_y)))
        db = ceil(dist((next_x, next_y), dest))

        # Sample a point until it respects all the constraints
        if outside:
            while da > radiusA+1 and db > radiusB+1:
                next_x = np.random.choice(indices[1])
                next_y = np.random.choice(indices[0])

                da = ceil(dist((next_x, next_y), (curr_x, curr_y)))
                db = ceil(dist((next_x, next_y), dest))

            outside = False
        else:
            while mask[next_y, next_x] != 255 or (da > radiusA+1 and db > radiusB+1):
                next_x = np.random.choice(indices[1])
                next_y = np.random.choice(indices[0])

                da = ceil(dist((next_x, next_y), (curr_x, curr_y)))
                db = ceil(dist((next_x, next_y), dest))

        # Save the sampled point
        points[step] = (next_x, next_y)

        # Update the state of the algorithm
        curr_x = next_x
        curr_y = next_y
        prev_step = step

    # Sort the point according to the step
    points = dict(sorted(points.items()))

    # Connect the point together
    curr_x, curr_y = origin
    for step, point in points.items():
        next_x, next_y = point
        rr, cc, val = line_aa(curr_y, curr_x, next_y, next_x)
        squiggle[rr, cc] = val * 255
        curr_x, curr_y = next_x, next_y

    return squiggle
