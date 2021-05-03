"""
Plot functions
"""

import matplotlib.pyplot as plt
import numpy as np
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


def plot_quantity_analysis(filename, df):
    # Set up the plot
    plt.figure()
    plt.grid()

    # Labels
    plt.xlabel('Number of annotations')
    plt.ylabel('Metric')

    # Plot the training and validation losses
    plt.plot(df['size'], df['iou'])
    plt.plot(df['size'], df['dice'])
    plt.legend(['IoU', 'Dice'], loc='lower left')

    # Set proper limit
    xticks = np.arange(0, df['size'].iloc[-1], 50, dtype=int)
    xticks[0] = 1

    plt.xticks(xticks)
    plt.xlim(left=1)
    plt.ylim([0, 1])

    # Save the figure
    plt.savefig('similarity-' + filename, bbox_inches='tight')


def plot_hausdorff(filename, df):
    # Set up the plot
    plt.figure()
    plt.grid()

    # Labels
    plt.xlabel('Number of annotations')
    plt.ylabel('Hausdorff distance')

    # Plot the training and validation losses
    plt.plot(df['size'], df['hausdorff'])

    # Set proper limit
    xticks = np.arange(0, df['size'].iloc[-1], 50, dtype=int)
    xticks[0] = 1

    plt.xticks(xticks)
    plt.xlim(left=1)

    # Save the figure
    plt.savefig('haus-' + filename, bbox_inches='tight')


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
        help="The type of object to detect."
    )
    parser.add_argument(
        '--loss',
        default='mean',
        choices=['mean', 'std'],
        help="The mean or std loss."
    )
    parser.add_argument(
        '--plot',
        default='loss',
        choices=['loss', 'quantity', 'quality', 'robustness'],
        help="Option to choose what to plot."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Read the data from the CSV
    df = pd.read_csv(args.path)

    if args.plot == 'loss':
        if args.loss == 'mean':
            train_loss = df['train_mean_loss']
            val_loss = df['val_mean_loss']
        else:
            train_loss = df['train_std_loss']
            val_loss = df['val_std_loss']

        filename = f'{args.type}_{args.loss}_loss.pdf'

        # Plot the loss
        plot_loss(df['epoch'], train_loss, val_loss, filename)
    elif args.plot == 'quantity':
        filename = f'{args.type}.pdf'

        plot_quantity_analysis(filename, df)
        plot_hausdorff(filename, df)
