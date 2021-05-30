"""
Training with dark inclusion and exclusion map
"""

import csv
import numpy as np
import time
import torch
import os

from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader

from src import (
    NuClick, TestDataset, Loss, convert_time, str2bool
)


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

    parser = ArgumentParser(description="Train a model with dark signal maps.")

    # Training parameters
    parser.add_argument(
        '--shuffle',
        type=str2bool,
        default=True,
        help="Whether to shuffle the training images or not."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        '--bs',
        dest='batch_size',
        type=int,
        default=16,
        help="The batch size for the training"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-3,
        help="The learning rate of the optimizer."
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=5e-5,
        help="The weight decay of the optimizer."
    )
    parser.add_argument(
        '--size',
        type=int,
        default=0,
        help="The number of images to use for training."
    )

    # Misc parameters
    parser.add_argument(
        '--type',
        help="The type of object to detect."
    )
    parser.add_argument(
        '--data',
        help="Path to the dataset."
    )
    parser.add_argument(
        '--dest',
        default='./',
        help="The path to save the weights of the model."
    )
    parser.add_argument(
        '--resume',
        type=bool,
        default=False,
        help="Resume the training of the model.")
    parser.add_argument(
        '--checkpoint',
        help="Checkpoint of the state of the training."
    )
    parser.add_argument(
        '--step',
        type=int,
        default=20,
        help="Save a checkpoint every step epoch."
    )
    parser.add_argument(
        '--stat',
        dest='stat_path',
        default='./statistics.csv',
        help="Path to save statistic about training."
    )

    return parser.parse_args()


def train(model, trainloader, criterion, optimizer, batch_size):
    """
    Train the model for one epoch using gradient accumulation technique.
    Parameters
    ----------
    model : torch model
        The model to train.
    trainloader : torch DataLoader
        The training dataset.
    criterion : torch loss
        The loss function.
    optimizer : torch optimizer
        The optimizer.
    batch_size : int
        The real batch size.
    Return
    ------
    losses : list of float
        The losses during the training.
    """

    model.train()
    losses = []

    for index, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Add black inclusion/exclusion map
        shape = (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3])
        signal = torch.zeros(shape, device=inputs.device)
        inputs = torch.cat([inputs, signal], dim=1)

        predictions = model(inputs)

        loss = criterion(predictions, targets)

        (loss / batch_size).backward()

        losses.append(loss.item() * inputs.size(0))

        if (index + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    return losses


def validate(model, valloader, criterion):
    """
    Validate the model for one epoch.

    Parameters
    ----------
    model : torch model
        The model to train.
    valloader : torch DataLoader
        The validation dataset.
    criterion : torch loss
        The loss function.

    Return
    ------
    losses : list of float
        The losses during the validation.
    """

    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, targets)
            losses.append(loss.item() * inputs.size(0))

    return losses


if __name__ == "__main__":
    args = parse_arguments()

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Statistics
    header = ['epoch', 'train_mean_loss', 'train_std_loss', 'val_mean_loss',
              'val_std_loss', 'duration']
    if not os.path.exists(args.stat_path):
        with open(args.stat_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

    # Build the training and validation set
    train_data = TestDataset(os.path.join(args.data, 'train'), n_image=args.size)
    val_data = TestDataset(os.path.join(args.data, 'val'))
    trainloader = DataLoader(train_data, 4, shuffle=args.shuffle)
    valloader = DataLoader(val_data, args.batch_size, shuffle=args.shuffle)

    model = NuClick()
    model = model.to(device)

    optimizer = Adam(model.parameters(), args.lr, weight_decay=args.wd)
    criterion = Loss()

    # Check if resume the training
    if args.resume:
        state = torch.load(args.checkpoint, map_location=device)

        start_epoch = state['epoch']
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
    else:
        start_epoch = 0

    total_time = 0.0

    # Training the model
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Train the model for one epoch
        train_losses = train(
            model, trainloader, criterion, optimizer, args.batch_size
        )

        # Perform the validation test on the model
        val_losses = validate(model, valloader, criterion)

        # Compute the time taken for one epoch
        elapsed_time = time.time() - start_time
        minutes, seconds = convert_time(elapsed_time)
        total_time += elapsed_time

        # Statistics
        with open(args.stat_path, 'a', newline='') as file:
            csv.writer(file).writerow([
                epoch,
                np.mean(train_losses),
                np.std(train_losses),
                np.mean(val_losses),
                np.std(val_losses),
                f"{minutes:.0f}m{seconds:.0f}s"
            ])

        # Checkpoint save
        if epoch % args.step and args.checkpoint:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            torch.save(
                state,
                os.path.join(args.dest, f'{args.type}_checkpoint.pth')
            )

    minutes, seconds = convert_time(total_time)
    print(f"Training complete in {minutes:.0f}m {seconds:.0f}s")

    # Save the trained model
    torch.save(
        model.state_dict(),
        os.path.join(args.dest, f'{args.type}_model.pth')
    )

    # Save the training state for further training
    if args.checkpoint:
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(
            state,
            os.path.join(args.dest, f'{args.type}_checkpoint.pth')
        )
