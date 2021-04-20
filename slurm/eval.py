"""
Create a SLURM script for evaluation
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

    parser = ArgumentParser(
        description="Create a SLURM script for evaluation."
    )

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
        '--weight',
        help="The path weight of the model to evaluate."
    )
    parser.add_argument(
        '--data',
        help="The path to the dataset."
    )
    parser.add_argument(
        '--stat',
        help="Path to save statistics about the evaluation."
    )

    # SLURM parameters
    parser.add_argument(
        '--time',
        default="14-00:00:00",
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
        default='all',
        help="The partition to use for the program."
    )

    # Project parameters
    parser.add_argument(
        '--env',
        default='thesis',
        help="The anaconda environment."
    )
    parser.add_argument(
        '--type',
        help="Type of object."
    )
    parser.add_argument(
        '--size',
        type=int,
        help="The number of images used to train the model."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Create the header of the script
    HEADER = "#!/usr/bin/env bash\n\n"\
        f"#SBATCH --job-name={args.type}-eval\n"\
        "#SBATCH --export=ALL\n"\
        f"#SBATCH --output={args.type}-eval.log\n"\
        f"#SBATCH --cpus-per-task={args.task}\n"\
        "#SBATCH --mem-per-cpu=4G\n"\
        "#SBATCH --gres=gpu:1\n"\
        f"#SBATCH --time={args.time}\n"\
        f"#SBATCH --partition={args.partition}\n\n"\

    ENV = f"conda activate {args.env}\n\n"

    COMMAND = f"cd {args.code}\n"\
        f"python3 -u evaluate.py --data {args.data} --stat {args.stat} "\
        f"--weight {args.weight} --type {args.type} --size {args.size}\n"

    script = HEADER + ENV + COMMAND

    with open(os.path.join(args.dest, f'eval-{args.type}.sh'), 'w') as file:
        file.write(script)
