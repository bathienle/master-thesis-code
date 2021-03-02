"""
Dataset processing
"""

import random
import torch
import os

from argparse import ArgumentParser
from PIL import Image
from torch.utils.data.dataset import Dataset

from processing import create_signal


class CytomineDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        # Keep the filename of each image
        self.filenames = os.listdir(os.path.join(path, 'images'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Load the image
        image_path = os.path.join(self.path, 'images', self.filenames[index])
        image = Image.open(image_path)

        # Load the mask
        mask_path = os.path.join(self.path, 'masks', self.filenames[index])
        mask = Image.open(mask_path).convert("L")

        # Load the pseudo inclusion map
        inc_path = os.path.join(self.path, 'inclusions', self.filenames[index])
        inclusion = Image.open(inc_path).convert("L")

        # Load the pseudo exclusion map
        exc_path = os.path.join(self.path, 'exclusions', self.filenames[index])
        exclusion = Image.open(exc_path).convert("L")

        # Apply transform on image and mask
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            inclusion = self.transform(inclusion)
            exclusion = self.transform(exclusion)

        # Create the inclusion and exclusion map
        inclusion = create_signal(inclusion)
        exclusion = create_signal(exclusion)

        # Concatenate the inclusion and exclusion map along the RGB channels
        image = torch.cat([image, inclusion, exclusion], 0)

        return image, mask


def parse_arguments():
    """Parse the arguments of the program. 

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(
        description="Split test dataset into test/validation set."
    )

    parser.add_argument(
        '--path',
        type=str,
        default='../../117286674/',
        help="Path to the dataset."
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.8,
        help="The train/test split ratio of the dataset."
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=False,
        help="Whether to shuffle the test set or not."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Get the filenames and the number of images
    filenames = os.listdir(os.path.join(args.path, 'images'))
    n_images = len(filenames)
    print(n_images)

    # Shuffle the images
    if args.shuffle:
        random.shuffle(filenames)

    # Split the dataset with the ratio
    split_train_test = round(n_images * args.ratio)
    split_train_val = round(split_train_test * 0.8)  # Split 80/20

    test_set = filenames[split_train_test:]
    train_set = filenames[:split_train_test]
    val_set = train_set[split_train_val:]
    train_set = train_set[:split_train_val]

    subdirs = ['images', 'masks', 'inclusions', 'exclusions']
    sets = ['train', 'val', 'test']

    # Create the directories
    for s in sets:
        for subdir in subdirs:
            os.makedirs(os.path.join(args.path, s, subdir), exist_ok=True)

    # Move the images and masks to the destination directory
    for dataset, name in zip([train_set, val_set, test_set], sets):
        for filename in dataset:
            for subdir in subdirs:
                os.rename(os.path.join(args.path, subdir, filename),
                          os.path.join(args.path, name, subdir, filename))

    # Delete empty directories
    for subdir in subdirs:
        os.rmdir(os.path.join(args.path, subdir))
