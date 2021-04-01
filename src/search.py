"""
Cross validation of the hyperparameters
"""

import numpy as np
import torch
import os

from argparse import ArgumentParser
from skorch.net import NeuralNet
from skorch.helper import predefined_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from torch.optim import Adam

from datasets import CytomineDataset
from losses import Loss
from metrics import IoU, DiceCoefficient, HausdorffDistance
from model import NuClick


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

    parser = ArgumentParser(
        description="Cros validation for the hyperparameters."
    )

    parser.add_argument(
        '--wd',
        type=float,
        default=1e-5,
        help="The weight decay of the optimizer."
    )
    parser.add_argument(
        '--type',
        help="The type of object to detect."
    )
    parser.add_argument(
        '--data',
        help="Path to the dataset."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Build the training and validation set
    train_data = CytomineDataset(os.path.join(args.data, 'train'))
    val_data = CytomineDataset(os.path.join(args.data, 'val'))

    network = NeuralNet(
        NuClick,
        criterion=Loss,
        optimizer=Adam,
        optimizer__weight_decay=args.wd,
        batch_size=16,
        train_split=predefined_split(val_data),
        device=device,
        iterator_train__shuffle=True,
    )

    # Set the hyperparameters to test
    parameters = {
        'lr': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'max_epochs': [100, 200, 300]
    }

    # Set the scoring metrics
    scoring = {
        'iou': make_scorer(IoU),
        'dice': make_scorer(DiceCoefficient),
        'haus': make_scorer(HausdorffDistance, greater_is_better=False)
    }

    # Perform the cross validation
    grid = GridSearchCV(network, parameters, refit=False, scoring=scoring)
    grid.fit(train_data)

    # Print the results
    print("Type:", args.type)
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)
