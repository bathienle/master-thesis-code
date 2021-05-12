"""
Dataset processing
"""

import torch
import os

from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor, resize

from src.processing import create_signal


def load_image(path, dim, binary=False):
    """
    Load an image from a given path.

    Parameters
    ----------
    path : str
        The path to the image.
    dim : tuple
        The output dimension of the image.
    binary : bool (default=False)
        Whether to convert the image to binary mode or not.

    Return
    ------
    image : PIL Image
        The loaded image.
    """

    image = Image.open(path)

    if binary:
        image = image.convert('L')

    image = to_tensor(resize(image, dim))

    return image


class CytomineDataset(Dataset):
    """
    Class representing a dataset from Cytomine.
    Attributes
    ----------
    path : str
        The path to the dataset.
    transform : Transform (default=None)
        The transform to apply for data augmentation.
    dim : tuple (default=(512, 512))
        The output dimension of the images.
    n_image : int (default=0)
        The maximum number of images to use from the dataset.
    """

    def __init__(self, path, transform=None, dim=(512, 512), n_image=0):
        self.path = path
        self.transform = transform
        self.dim = dim

        # Keep the filename of each image
        self.filenames = os.listdir(os.path.join(path, 'images'))

        # Compute the number of images to work with
        self.length = n_image if n_image else len(self.filenames)
        self.length = min(self.length, len(self.filenames))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filename = self.filenames[index]

        # Load the image
        image_path = os.path.join(self.path, 'images', filename)
        image = load_image(image_path, self.dim)

        # Load the mask
        mask_path = os.path.join(self.path, 'masks', filename)
        mask = load_image(mask_path, self.dim, binary=True)

        # Load the pseudo inclusion map
        inc_path = os.path.join(self.path, 'inclusions', filename)
        inclusion = load_image(inc_path, self.dim, binary=True)

        # Load the pseudo exclusion map
        exc_path = os.path.join(self.path, 'exclusions', filename)
        exclusion = load_image(exc_path, self.dim, binary=True)

        # Create the inclusion and exclusion map
        inclusion = create_signal(inclusion)
        exclusion = create_signal(exclusion)

        # Concatenate the inclusion and exclusion map along the RGB channels
        input = torch.cat([image, inclusion, exclusion], dim=0)

        if self.transform:
            input, mask = self.transform(input, mask)

        return input, mask


class TestDataset(Dataset):
    """
    Class representing a simple dataset composed of images and masks.

    Attributes
    ----------
    path : str
        The path to the dataset.
    dim : tuple (default=(512, 512))
        The output dimension of the images.
    ret_filename : bool (default=False)
        Whether to return the filename or not.
    """

    def __init__(self, path, dim=(512, 512), ret_filename=False):
        self.path = path
        self.dim = dim
        self.ret_filename = ret_filename

        # Keep the filename of each image
        self.filenames = os.listdir(os.path.join(path, 'images'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        # Load the image
        image_path = os.path.join(self.path, 'images', filename)
        image = load_image(image_path, self.dim)

        # Load the mask
        mask_path = os.path.join(self.path, 'masks', filename)
        mask = load_image(mask_path, self.dim, binary=True)

        if self.ret_filename:
            return image, mask, filename

        return image, mask


class CrossValidationDataset(Dataset):
    """
    Class representing a dataset for the cross-validation.

    Attributes
    ----------
    paths : list of str
        A list of paths.
    dim : tuple (default=(512, 512))
        The output dimension of the images.
    """

    def __init__(self, paths, dim=(512, 512)):
        self.paths = paths
        self.dim = dim

        self.filenames = []
        # Get all the absolute path of the filenames
        for base_path in paths:
            path = os.path.join(base_path, 'images')
            filenames = [
                os.path.join(path, filename) for filename in os.listdir(path)
            ]
            self.filenames.extend(filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        absolute_path = self.filenames[index]

        # Split the directory path and the filename of the image
        path = os.path.dirname(os.path.dirname(absolute_path))
        filename = os.path.basename(absolute_path)

        # Load the image
        image_path = os.path.join(path, 'images', filename)
        image = load_image(image_path, self.dim)

        # Load the mask
        mask_path = os.path.join(path, 'masks', filename)
        mask = load_image(mask_path, self.dim, binary=True)

        # Load the pseudo inclusion map
        inc_path = os.path.join(path, 'inclusions', filename)
        inclusion = load_image(inc_path, self.dim, binary=True)

        # Load the pseudo exclusion map
        exc_path = os.path.join(path, 'exclusions', filename)
        exclusion = load_image(exc_path, self.dim, binary=True)

        # Create the inclusion and exclusion map
        inclusion = create_signal(inclusion)
        exclusion = create_signal(exclusion)

        # Concatenate the inclusion and exclusion map along the RGB channels
        input = torch.cat([image, inclusion, exclusion], dim=0)

        return input, mask
