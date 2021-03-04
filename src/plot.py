"""
Plot functions
"""

import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser


def plot_loss(n_epochs, train_loss, val_loss, filename):
    """Plot the losses of a training over the epochs.

    Parameters
    ----------
    n_epochs : list of int
        The range of epochs of the training.
    train_loss : list of float
        The training loss.
    val_loss : list of float
        The validation loss.
    filename : str
        The filename of the output plot.
    """

    # Set up the plot
    plt.figure()
    plt.grid()

    # Labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot the training and validation losses
    plt.plot(n_epochs, train_loss)
    plt.plot(n_epochs, val_loss)
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')

    # Save the plot
    plt.savefig(filename)


def parse_arguments():
    """Parse the arguments of the program. 

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(description="Plot images.")

    parser.add_argument(
        '--path',
        default='./statistics.csv',
        help="Path to the statistics file (CSV)."
    )
    parser.add_argument(
        '--dest',
        default='./',
        help="Destination path of the plot."
    )
    parser.add_argument(
        '--type',
        default='gland',
        choices=['gland', 'bronchus', 'tumor'],
        help="The type of object to detect."
    )
    parser.add_argument(
        '--loss',
        default='mean',
        choices=['mean', 'std'],
        help="The mean or std loss."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Read the data from the CSV
    df = pd.read_csv(args.path)

    if args.loss == 'mean':
        train_loss = df['train_mean_loss']
        val_loss = df['val_mean_loss']
    else:
        train_loss = df['train_std_loss']
        val_loss = df['val_std_loss']

    filename = f'{args.type}_{args.loss}_loss.png'

    # Plot the loss
    plot_loss(df['epoch'], train_loss, val_loss, filename)