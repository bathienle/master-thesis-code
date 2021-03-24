"""
Evaluation of the model using metrics
"""

import csv
import numpy as np
import torch
import os

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from datasets import CytomineDataset
from metrics import IoU, DiceCoefficient, HausdorffDistance
from model import NuClick
from processing import post_process


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

    parser = ArgumentParser(description="Evaluate a model.")

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
        '--type',
        help="Type of object."
    )
    parser.add_argument(
        '--bs',
        dest='batch_size',
        type=int,
        default=16,
        help="The batch size."
    )
    parser.add_argument(
        '--size',
        type=int,
        default=0,
        help="The number of images used to train the model."
    )

    return parser.parse_args()


def evaluate(model, testloader, criterion):
    """
    Evaluate the model with a metric evaluation.

    Parameters
    ----------
    model : torch model
        The model to train.
    testloader : torch DataLoader
        The test dataset.
    criterion : torch metric
        The metric.

    Return
    ------
    metric : float
        The performance resulted from the metric.
    """

    model.eval()
    metric = 0.0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model(inputs)
            outputs = post_process(predictions)

            metric += criterion(outputs, targets).item()

    return metric / len(testloader)


if __name__ == "__main__":
    args = parse_arguments()

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Statistics
    stat_path = os.path.join(args.stat, f'{args.type}-evaluation.csv')
    header = ['dataset', 'size', 'iou', 'dice', 'hausdorff']
    if not os.path.exists(stat_path):
        with open(stat_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

    # Build the test set
    test_set = CytomineDataset(os.path.join(args.data, 'test'))
    testloader = DataLoader(test_set, args.batch_size)

    # Load the model
    model = NuClick().to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))

    # Evaluation
    iou = evaluate(model, testloader, IoU())
    dice = evaluate(model, testloader, DiceCoefficient())
    hausdorff = evaluate(model, testloader, HausdorffDistance())

    with open(stat_path, 'a', newline='') as file:
        csv.writer(file).writerow([
            os.path.basename(os.path.normpath(args.data)),
            args.size,
            iou,
            dice,
            hausdorff
        ])
