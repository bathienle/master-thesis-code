"""
Quality analysis of the annotations
"""

import csv
import cv2
import numpy as np
import torch
import os

from argparse import ArgumentParser
from itertools import combinations
from math import ceil, dist
from PIL import Image
from torch.utils.data import DataLoader

from src import (
    NuClick, TestDataset, IoU, DiceCoefficient, HausdorffDistance,
    post_process, to_uint8
)
from squiggle import generate_squiggle
from shape import generate_shape


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    """
    Parse the arguments of the program.

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(description="Evaluate the annotations' quality.")

    parser.add_argument(
        '--data',
        help="Path to the test data for evaluation."
    )
    parser.add_argument(
        '--stat',
        help="Path to save statistic about the evaluation."
    )
    parser.add_argument(
        '--weight',
        help="Path to weight of the model."
    )
    parser.add_argument(
        '--save',
        default=None,
        help="Path to save the output images."
    )
    parser.add_argument(
        '--times',
        type=int,
        default=10,
        help="Number of variations for the same image."
    )
    parser.add_argument(
        '--step',
        type=int,
        default=5,
        help="The number of lines to use."
    )
    parser.add_argument(
        '--min_size',
        type=int,
        default=100,
        help="The smalled allowed object for the post processing."
    )
    parser.add_argument(
        '--area_threshold',
        type=int,
        default=300,
        help="The maximum area to fill for the post processing."
    )
    parser.add_argument(
        '--squiggle',
        default='random',
        choices=['random', 'shape'],
        help="The type of squiggle to generate."
    )

    return parser.parse_args()


def compute_min_distance(contour):
    """
    Compute the minimum distance from the contour of a mask.

    Parameters
    ----------
    contour : NumPy array
        An array of coordinates representing the contour of the mask.

    Return
    ------
    min_d : int
        The distance between two extreme points in the mask.
    """

    # Get the four extreme points of the mask
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    # Compute the distances for each pair of extreme points
    points = [leftmost, rightmost, topmost, bottommost]
    distances = [ceil(dist(a, b)) for (a, b) in combinations(points, 2)]

    return ceil(np.mean(distances))


def create_random_signal(mask, steps, min_area=1000):
    """
    Create random squiggles to imitate the user.

    Parameters
    ----------
    mask : NumPy array
        The segmentation mask of the image.
    steps : int
        The number of line segment to draw.
    min_area : int (default=1000)
        The minimum area to be considered as an object.

    Return
    ------
    signals : list of NumPy array
        A list of squiggles.
    """

    # Remove small noises
    mask[mask != 255] = 0

    # Get the number of masks in the image
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Generate a squiggle per object
    signals = []
    for contour in contours:
        # Fill the mask of this particular object
        object = np.zeros(mask.shape, dtype=mask.dtype)
        cv2.fillPoly(object, [contour], 255)

        # Get the coordinates of the mask
        indices = np.nonzero(object)

        # Get the minimum distance between the origin and destination point
        min_d = compute_min_distance(contour)

        # Sample the origin and destination points
        x1, x2 = np.random.choice(indices[1], 2)
        y1, y2 = np.random.choice(indices[0], 2)
        d = ceil(dist((x1, y1), (x2, y2)))

        while d < min_d or object[y1, x1] == 0 or object[y2, x2] == 0:
            x1, x2 = np.random.choice(indices[1], 2)
            y1, y2 = np.random.choice(indices[0], 2)
            d = ceil(dist((x1, y1), (x2, y2)))

        # Compute the minimum line length
        min_line = ceil(d / steps) + 1

        # Generate the squiggle based on the origin and destination point
        signal = generate_squiggle(object, (x1, y1), (x2, y2), min_line, steps)
        signals.append(signal)

    return signals


def create_shape_signal(mask, shape='c', min_area=1000):
    """
    Create shaped squiggles.

    Parameters
    ----------
    mask : NumPy array
        The segmentation mask of the image.
    shape : string
        The shape to draw.

    Return
    ------
    signals : list of NumPy array
        A list of squiggles.
    """

    # Remove small noises
    mask[mask != 255] = 0

    # Get the number of masks in the image
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Generate a squiggle per object
    signals = []
    for contour in contours:
        # Generate the squiggle based on the origin and destination point
        signal = generate_shape(mask, contour, shape=shape)
        signals.append(signal)

    return signals


def create_inputs(image, mask, steps, min_area=1000, random='shape', shape='c'):
    """
    Create the inputs of the network.

    Parameters
    ----------
    image : NumPy array
        The input image.
    mask : NumPy array
        The segmentation mask of the image.
    steps : int
        The number of line segment to draw.
    min_area : int (default=1000)
        The minimum area to be considered as an object.

    Return
    ------
    input : torch Tensor
        The random generated squiggle.
    signals : list of NumPy array
        A list of squiggles.
    """

    if random == 'shape':
        signals = create_shape_signal(mask, shape=shape)
    else:
        signals = create_random_signal(mask, steps, min_area=min_area)

    # Create the inclusion and exclusion map
    inclusion = torch.as_tensor(signals[0], dtype=torch.uint8).unsqueeze(0)
    exclusion = np.zeros(mask.shape, dtype=mask.dtype)
    for signal in signals[1:]:
        exclusion |= signal
    exclusion = torch.as_tensor(exclusion, dtype=torch.uint8).unsqueeze(0)

    # Create the input of the network
    input = torch.cat([image, inclusion, exclusion], dim=0)

    return input, signals


if __name__ == "__main__":
    args = parse_arguments()

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Statistics
    header = [
        'filename', 'variation', 'iou', 'dice', 'haus', 'steps', 'min_size',
        'area_threshold'
    ]
    if not os.path.exists(args.stat):
        with open(args.stat, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

    # Create the model and load its weight
    model = NuClick().to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    # Build the dataset
    test_data = TestDataset(os.path.join(args.data, 'test'), ret_filename=True)
    testloader = DataLoader(test_data)

    # Metrics
    iou = IoU()
    dice = DiceCoefficient()
    haus = HausdorffDistance()

    # Create the output dirs if save is enabled
    if args.save:
        for subdir in ['drawings', 'squiggles', 'outputs']:
            os.makedirs(os.path.join(args.save, subdir), exist_ok=True)

    for index, (inputs, targets, filenames) in enumerate(testloader):
        image = inputs[0]
        mask = to_uint8(targets[0].squeeze(0))
        filename = filenames[0]
        targets = targets.to(device)

        for i in range(args.times):
            # If shape, alternate between a circle and a square
            if i % 2:
                shape = 's'
            else:
                shape = 'c'

            # Create the inputs of the network
            input, signals = create_inputs(
                image, mask, args.step, random=args.squiggle, shape=shape
            )
            inputs = input.unsqueeze(0).to(device)

            with torch.no_grad():
                predictions = model(inputs)

            outputs = post_process(
                predictions,
                min_size=args.min_size,
                area_threshold=args.area_threshold
            )

            if args.save:
                # Merge all the squiggle of the image
                drawing = np.copy(image).transpose(1, 2, 0)
                drawing = to_uint8(drawing)
                squiggle = np.zeros(mask.shape, dtype=mask.dtype)

                for signal in signals:
                    squiggle |= signal
                    drawing[signal != 0, :] = [0, 255, 0]

                # Save the image with the squiggle
                path = os.path.join(args.save, 'drawings', f'{i}-{filename}')
                drawing = Image.fromarray(drawing)
                drawing.save(path)

                # Save the squiggle only
                path = os.path.join(args.save, 'squiggles', f'{i}-{filename}')
                squiggle = Image.fromarray(squiggle)
                squiggle.save(path)

                # Save the predicted mask
                path = os.path.join(args.save, 'outputs', f'{i}-{filename}')
                result = to_uint8(outputs[0].squeeze(0).cpu())
                result = Image.fromarray(result)
                result.save(path)

            # Save the performance
            with open(args.stat, 'a', newline='') as file:
                csv.writer(file).writerow([
                    filename,
                    i,
                    iou(outputs, targets).item(),
                    dice(outputs, targets).item(),
                    haus(outputs, targets).item(),
                    args.step,
                    args.min_size,
                    args.area_threshold
                ])
