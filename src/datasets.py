"""
Dataset processing
"""

import torch
import os

from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor, resize

from processing import create_signal


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
    """

    def __init__(self, path, transform=None, dim=(512, 512)):
        self.path = path
        self.transform = transform
        self.dim = dim

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

        # Resize to the correct dimension and convert to tensor
        image = to_tensor(resize(image, self.dim))
        mask = to_tensor(resize(mask, self.dim))
        inclusion = to_tensor(resize(inclusion, self.dim))
        exclusion = to_tensor(resize(exclusion, self.dim))

        # Create the inclusion and exclusion map
        inclusion = create_signal(inclusion)
        exclusion = create_signal(exclusion)

        # Concatenate the inclusion and exclusion map along the RGB channels
        input = torch.cat([image, inclusion, exclusion], dim=0)

        if self.transform:
            input, mask = self.transform(input, mask)

        return input, mask
