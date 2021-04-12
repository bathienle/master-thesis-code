"""
Utility functions
"""

import numpy as np
import torch

from tkinter import Tk
from tkinter.filedialog import askopenfilename


def str2bool(value):
    """
    Convert a string boolean to a boolean value.

    Parameters
    ----------
    value : str or bool
        The value to convert.

    Return
    ------
    value : bool
        The string converted into a boolean value.
    """

    if isinstance(value, bool):
        return value
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        return False


def to_uint8(array):
    """
    Convert an array of value in range [0, 1] to range [0, 255].

    Parameters
    ----------
    array : NumPy ndarray or Tensor
        The array to convert.

    Return
    ------
    array : NumPy ndarray
        The converted array.
    """

    if isinstance(array, torch.Tensor):
        return (array * 255).type(torch.uint8).permute(1, 2, 0).numpy()

    if isinstance(array, np.ndarray):
        return (array * 255).astype(np.uint8)


def convert_time(milliseconds):
    """
    Convert milliseconds to minutes and seconds.

    Parameters
    ----------
    milliseconds : int
        The time expressed in milliseconds.

    Return
    ------
    minutes : int
        The minutes.
    seconds : int
        The seconds.
    """

    minutes = milliseconds // 60
    seconds = milliseconds % 60

    return minutes, seconds


def get_image_path():
    """
    Get the path of an image.

    Return
    ------
    path : str
        The path to the image.
    """

    root = Tk()
    root.withdraw()  # Hide the useless tk window that appear

    return askopenfilename(
        filetypes=[('PNG', '*.png'), ('JPG', '*.jpg'), ('BMP', '*.bmp'),
                   ('TIF', '*.tif'), ('All files', '*')],
        parent=root
    )
