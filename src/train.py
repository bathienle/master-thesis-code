"""
Training procedure for the neural network
"""

import csv
import numpy as np
import time
import torch
import os

from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CytomineDataset
from losses import Loss
from model import NuClick
from utils import convert_time


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    """Parse the arguments of the program. 

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(description="Train a model.")

    # Training parameters
    parser.add_argument(
        '--epochs',
        dest='epochs',
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
        dest='lr',
        type=float,
        default=3e-3,
        help="The learning rate of the optimizer."
    )
    parser.add_argument(
        '--wd',
        dest='wd',
        type=float,
        default=5e-5,
        help="The weight decay of the optimizer."
    )

    # Misc parameters
    parser.add_argument(
        '--path',
        dest='path',
        type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        '--dest',
        dest='dest',
        type=str,
        default='./',
        help="The path to save the weights of the model."
    )
    parser.add_argument(
        '--resume',
        dest='resume',
        type=bool,
        default=False,
        help="Resume the training of the model.")
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        type=str,
        default='./checkpoint.pth',
        help="Checkpoint of the state of the training."
    )
    parser.add_argument(
        '--stat',
        dest='state_path',
        type=str,
        default='./statistics.csv',
        help="Path to save statistic about training."
    )

    return parser.parse_args()


def train(model, trainloader, criterion, optimizer):
    """Train the model for one epoch.

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

    Return
    ------
    losses : python list of floats
        The losses during the training.
    """

    model.train()
    losses = []

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)

        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item() * inputs.size(0))

    return losses


def validate(model, valloader, criterion):
    """Validate the model for one epoch.

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
    losses : python list of floats
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

    torch.manual_seed(0)  # Reproducibility

    # Statistics
    header = ['epoch', 'train_mean_loss', 'train_std_loss', 'val_mean_loss',
              'val_std_loss']
    if not os.path.exists(args.state_path):
        with open(args.state_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

    # Transform for images and masks
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Build the training and validation set
    datasets = {x: CytomineDataset(os.path.join(args.path, x), transform)
                for x in ['train', 'val']}
    trainloader = DataLoader(datasets['train'], args.batch_size, shuffle=True)
    valloader = DataLoader(datasets['val'], args.batch_size, shuffle=True)

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
        print(f"Epoch {epoch+1}/{args.epochs}")
        print('-' * 10)

        start_time = time.time()

        # Train the model for one epoch
        train_losses = train(model, trainloader, criterion, optimizer)

        # Perform the validation test on the model
        val_losses = validate(model, valloader, criterion)

        # Loss
        training_loss = sum(train_losses) / len(trainloader.dataset)
        validation_loss = sum(val_losses) / len(valloader.dataset)
        print(f"Training loss: {training_loss:.4f}",
              f"Validation loss: {validation_loss:.4f}")

        # Statistics
        with open(args.state_path, 'a', newline='') as file:
            csv.writer(file).writerow([
                epoch,
                np.mean(train_losses),
                np.std(train_losses),
                np.mean(val_losses),
                np.std(val_losses)
            ])

        # Compute the time taken for one epoch
        elapsed_time = time.time() - start_time
        minutes, seconds = convert_time(elapsed_time)
        print(f"{minutes:.0f}m {seconds:.0f}s")
        total_time += elapsed_time

    minutes, seconds = convert_time(total_time)
    print(f"Training complete in {minutes:.0f}m {seconds:.0f}s")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.dest, 'model.pth'))

    # Save the training state for further training
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(state, os.path.join(args.dest, 'checkpoint.pth'))