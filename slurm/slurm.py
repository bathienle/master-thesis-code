"""
Create a SLURM script for training
"""

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
        choices=['gland', 'bronchus', 'tumor', 'infiltration', 'inflammation'],
        help="Type of object."
    )

    return parser.parse_args()


# Main

args = parse_arguments()

# Create the script

HEADER = f"""#!/usr/bin/env bash

#SBATCH --job-name={args.bs}-{args.type}
#SBATCH --export=ALL
#SBATCH --output={args.type}.log
#SBATCH --cpus-per-task={args.task}
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time={args.time}
#SBATCH --partition={args.partition}

"""

ENV = f"conda activate {args.env}"

COMMAND = f"""

cd {args.code}
python3 -u train.py --epochs {args.epoch} --bs {args.bs} --size {args.size} --path {args.data} --dest {args.dest} --stat {os.path.join(args.dest, f'{args.type}-statistics.csv')}
"""

script = HEADER + ENV + COMMAND

with open(os.path.join(args.dest, f'train-{args.type}.sh'), 'w') as file:
    file.write(script)
