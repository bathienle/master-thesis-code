"""
Dataset processing
"""

import torch
import os

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class GlandDataset(Dataset):
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
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
