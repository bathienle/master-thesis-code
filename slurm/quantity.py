"""
Quantity analysis
"""

import numpy as np
import os

from argparse import ArgumentParser

from src import str2bool


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
        type=str2bool,
        default=True,
        help="Whether to shuffle the training images or not."
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.1,
        help="Percentage of the dataset."
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=300,
        help="Number of epoch to train the models."
    )
    parser.add_argument(
        '--time',
        default="14-00:00:00",
        help="The maximal time to run this script."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Count the training images
    n_images = len(os.listdir(os.path.join(args.data, 'train', 'images')))

    # Split the training images
    step = int(n_images * args.ratio)
    split = np.linspace(1, n_images, n_images // step, dtype=int)

    # Create the script for a growing number of training images
    for index, n_images in enumerate(split):
        # Create the directories
        path = os.path.join(args.dest, f"{args.type}-{index}")
        os.makedirs(path, exist_ok=True)

        # Create the SLURM script for this specific training
        code = os.path.abspath("../src/")
        command = f"python3 train.py --dest {path} --code {code}"\
            f" --data {args.data} --epoch {args.epoch} --size {n_images}"\
            f" --time {args.time} --partition tesla --type {args.type}"\
            f" --shuffle {args.shuffle}"

        os.system(command)

        # Create the SLURM script for the evaluation
        weight_path = os.path.join(path, f"{args.type}_model.pth")
        command = f"python3 eval.py --dest {path} --code {code}"\
            f" --data {args.data} --weight {weight_path} --size {n_images}"\
            f" --time {args.time} --partition all --type {args.type}"\
            f" --stat {args.dest}\n"

        os.system(command)
