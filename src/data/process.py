"""
Clean and process the dataset.
"""

import numpy as np
import shutil
import os

from argparse import ArgumentParser
from PIL import Image


def parse_arguments():
    """
    Parse the arguments of the program.

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(description="Clean and process the dataset.")

    parser.add_argument(
        '--path',
        help="Path to the dataset."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    image_path = os.path.join(args.path, 'images')
    mask_path = os.path.join(args.path, 'masks')
    inclusion_path = os.path.join(args.path, 'inclusions')
    exclusion_path = os.path.join(args.path, 'exclusions')

    # Get the filenames
    filenames = os.listdir(image_path)

    for filename in filenames:
        mask = Image.open(os.path.join(mask_path, filename)).convert("L")

        # If no mask then remove this image
        if not os.path.exists(os.path.join(mask_path, filename)):
            print(f'[{filename}]', "Remove no mask images...")

            for path in [image_path, inclusion_path, exclusion_path]:
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)

            continue

        # Case 1: the target mask is completely white
        n_pixel = mask.height * mask.width
        nonzero = np.count_nonzero(mask)

        # If the mask is completely white
        if nonzero == n_pixel:
            print(f'[{filename}]', "Replace white mask by inclusion mask...")

            # Simply keep the inclusion mask as the target mask
            shutil.copy(
                os.path.join(inclusion_path, filename),
                os.path.join(mask_path, filename)
            )

            # There will be no exclusion map
            shape = (mask.height, mask.width)
            exclusion = Image.fromarray(np.zeros(shape, dtype=np.uint8))
            exclusion.save(os.path.join(exclusion_path, filename))

        # Case 2: the inclusion mask is completely dark
        path = os.path.join(inclusion_path, filename)
        inclusion = Image.open(path).convert('L')

        # If the inclusion map is completely dark
        if not np.count_nonzero(inclusion):
            print(f'[{filename}]', "Remove dark inclusion mask...")

            # Remove the image, mask, inclusion and exclusion map
            for path in [image_path, mask_path, inclusion_path, exclusion_path]:
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
