"""
Quantity analysis
"""

import numpy as np
import os

from argparse import ArgumentParser


def parse_arguments():
    """
    Parse the arguments of the program.

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(
        description="Produce several script for the quantity analysis."
    )

    parser.add_argument(
        '--data',
        help="Path to the dataset."
    )
    parser.add_argument(
        '--dest',
        help="Destination path of the SLURM training scripts."
    )
    parser.add_argument(
        '--type',
        help="Type of object."
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help="Whether to shuffle the training images or not."
    )
    parser.add_argument(
        '--step',
        type=float,
        default=0.1,
        help="Percentage of the dataset."
    )
    parser.add_argument(
        '--time',
        default="1-00:00:00",
        help="The maximal time to run this script."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Count the training images
    n_images = len(os.listdir(os.path.join(args.data, 'train', 'images')))

    # Split the training images
    split = np.append(
        np.arange(1, n_images, np.floor(n_images * args.step), dtype=int),
        n_images
    )

    # Create the script for a growing number of training images
    for index, n_images in enumerate(split):
        # Create the directories
        path = os.path.join(args.dest, f"{args.type}-{index}")
        os.makedirs(path, exist_ok=True)

        # Create the SLURM script for this specific training
        code = os.path.abspath("../src/")
        command = f"python3 train.py --dest {path} --code {code}"\
            f" --data {args.data} --epoch 300 --size {n_images}"\
            f" --time {args.time} --partition tesla --type {args.type}"\
            f" --shuffle {args.shuffle}"

        os.system(command)
