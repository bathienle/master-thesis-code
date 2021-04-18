"""
Create a SLURM script for training
"""

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

    parser = ArgumentParser(description="Create a SLURM script for training.")

    # Paths
    parser.add_argument(
        '--dest',
        help="Destination path of the outputs of the run script."
    )
    parser.add_argument(
        '--code',
        help="The path to the source code."
    )
    parser.add_argument(
        '--data',
        help="The path to the dataset."
    )

    # Training parameters
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help="The number of epochs if training."
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help="The batch size."
    )
    parser.add_argument(
        '--size',
        type=int,
        default=0,
        help="The number of images to use for training."
    )
    parser.add_argument(
        '--shuffle',
        type=str2bool,
        default=True,
        help="Whether to shuffle the training images or not."
    )

    # SLURM parameters
    parser.add_argument(
        '--time',
        default="1-00:00:00",
        help="The maximal time to run this script."
    )
    parser.add_argument(
        '--task',
        type=int,
        default=12,
        help="The number of tasks per CPU."
    )
    parser.add_argument(
        '--partition',
        choices=['all', 'quadro', 'tesla', 'debug'],
        default='all',
        help="The partition to use for the program."
    )
    parser.add_argument(
        '--env',
        default='thesis',
        help="The anaconda environment."
    )
    parser.add_argument(
        '--type',
        help="Type of object."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Create the header of the script
    HEADER = "#!/usr/bin/env bash\n\n"\
        f"#SBATCH --job-name={args.bs}-{args.type}\n"\
        "#SBATCH --export=ALL\n"\
        f"#SBATCH --output={args.type}.log\n"\
        f"#SBATCH --cpus-per-task={args.task}\n"\
        "#SBATCH --mem-per-cpu=4G\n"\
        "#SBATCH --gres=gpu:1\n"\
        f"#SBATCH --time={args.time}\n"\
        f"#SBATCH --partition={args.partition}\n\n"\

    # Create the environment command
    ENV = f"conda activate {args.env}\n\n"

    # Create the command to perform
    COMMAND = f"cd {args.code}\n"\
        f"python3 -u train.py --epochs {args.epoch} --bs {args.bs} "\
        f"--size {args.size} --data {args.data} --type {args.type} "\
        f"--dest {args.dest} --shuffle {args.shuffle} "\
        f"--stat {os.path.join(args.dest, f'{args.type}-statistics.csv')}\n"

    script = HEADER + ENV + COMMAND

    with open(os.path.join(args.dest, f'train-{args.type}.sh'), 'w') as file:
        file.write(script)
