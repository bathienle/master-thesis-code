"""
Cross validation of the hyperparameters
"""

import csv
import numpy as np
import torch
import os

from argparse import ArgumentParser
from itertools import product
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from datasets import CrossValidationDataset, CytomineDataset
from evaluate import evaluate
from losses import Loss
from metrics import IoU, DiceCoefficient, HausdorffDistance
from model import NuClick
from train import train, validate

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
        '--bs',
        dest='batch_size',
        type=int,
        default=16,
        help="The batch size for the dataset."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        '--data',
        help="Path to the dataset."
    )
    parser.add_argument(
        '--dest',
        default='./',
        help="The path to save the CSV results."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Load the dataset
    dirnames = next(os.walk(args.data))[1]
    dirnames.remove('test')
    dirnames = [os.path.join(args.data, dirname) for dirname in dirnames]

    cross = []
    for dirname in dirnames:
        folds = dirnames.copy()
        folds.remove(dirname)

        cross.append((folds, dirname))

    # Create the test set
    test_data = CytomineDataset(os.path.join(args.data, 'test'))
    testloader = DataLoader(test_data, args.batch_size, shuffle=True)

    # Loss function and metrics
    criterion = Loss()

    # Statistics
    header = ['epoch', 'train_mean_loss', 'train_std_loss', 'val_mean_loss',
              'val_std_loss', 'iou', 'dice', 'haus']
    data_name = os.path.basename(os.path.normpath(args.data))

    # Parameters to test
    optimizers = [Adam, SGD]
    lrs = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    wds = [1e-1, 1e-2, 1e-3]

    # The cross-validation
    for index, (folds, val) in enumerate(cross):
        print(f'Fold {index}')

        # Create the train and validation sets for this combination of folds
        train_data = CrossValidationDataset(folds)
        trainloader = DataLoader(train_data, args.batch_size, shuffle=True)
        val_data = CytomineDataset(val)
        valloader = DataLoader(val_data, args.batch_size, shuffle=True)

        for (optim, lr, wd) in product(optimizers, lrs, wds):
            print()
            print(f'{optim.__name__:>4}, lr: {lr:>6}, wd: {wd:>5}')
            print('-' * 27)

            # Statistics
            csv_name = f'cv-{data_name}-{optim.__name__}-lr={lr}-wd={wd}.csv'
            stat_path = os.path.join(args.dest, csv_name)
            with open(stat_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()

            # Create the model
            model = NuClick().to(device)

            # Create the optimizer
            optimizer = optim(model.parameters(), lr=lr, weight_decay=wd)

            print(" epoch   train_loss    val_loss      iou     dice      haus")
            print("------  -----------  ----------  -------  -------  --------")

            # Training the model
            for epoch in range(args.epochs):
                # Train the model for one epoch
                train_losses = train(model, trainloader, criterion, optimizer)

                # Perform the validation test on the model
                val_losses = validate(model, valloader, criterion)

                # Perform the validation test on the metrics
                iou = evaluate(model, valloader, IoU())
                dice = evaluate(model, valloader, DiceCoefficient())
                haus = evaluate(model, valloader, HausdorffDistance())

                # Statistics
                with open(stat_path, 'a', newline='') as file:
                    csv.writer(file).writerow([
                        epoch,
                        np.mean(train_losses),
                        np.std(train_losses),
                        np.mean(val_losses),
                        np.std(val_losses),
                        iou,
                        dice,
                        haus
                    ])

                print(
                    f"{epoch:>6}", ' ',
                    f"{np.mean(train_losses):>10.4f} ",
                    f"{np.mean(val_losses):>10.4f}", ' ',
                    f"{iou:>5.4f}", ' ',
                    f"{dice:>5.4f}", ' ',
                    f"{haus:>7.4f}"
                )
