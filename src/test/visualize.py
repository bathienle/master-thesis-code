"""
Visualize the predictions in the image
"""

import cv2
import numpy as np
import os
import torch

from argparse import ArgumentParser
from skimage.color import label2rgb
from torchvision.transforms.functional import to_tensor

from drawing import Painter
from src import NuClick, create_signal, post_process, get_image_path, to_uint8


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

    parser = ArgumentParser(description="Test a model.")

    parser.add_argument(
        '--weight',
        help="Path to the weight of the model."
    )
    parser.add_argument(
        '--dest',
        default='./',
        help="Path to save the segmentations."
    )
    parser.add_argument(
        '--min_size',
        type=int,
        default=100,
        help="The smalled allowed object for the post processing."
    )
    parser.add_argument(
        '--area_threshold',
        type=int,
        default=300,
        help="The maximum area to fill for the post processing."
    )

    return parser.parse_args()


def get_patches(image, squiggles, signal, dim=(512, 512)):
    """
    Get patches from the image based on the squiggles.

    Parameters
    ----------
    image : numpy array of shape (h, w, 3)
        The complete image.
    squiggles : list of coordinates
        The squiggles coordinates done by the user.
    signal : numpy array of shape (h, w)
        The complete drawing of the user.

    Return
    ------
    inputs : Tensor
        The inputs to the network.
    offsets : Tensor
        The offset of each patches from the original image.
    """

    inputs = []
    offsets = []
    height, width = dim
    shape = image.shape[:2]

    for squiggle in squiggles:
        x_max, y_max = max(squiggle[:, 0]), max(squiggle[:, 1])

        # Compute the top left corner coordinates of the crop
        x = min(max(0, x_max - width // 2), shape[1] - width)
        y = min(max(0, y_max - height // 2), shape[0] - height)

        offsets.append([x, y])

        # Create a crop of the image based on the squiggle
        crop = image[y:y+height, x:x+width, :]

        # Create the corresponding crop signal
        crop_signal = signal[y:y+height, x:x+width]

        # Create the inclusion map
        incl = np.zeros(signal.shape, dtype=np.uint8)
        for i in range(len(squiggle)-1):
            cv2.line(incl, tuple(squiggle[i]), tuple(squiggle[i+1]), 255, 1)

        inclusion = incl[y:y+height, x:x+width]

        # Create the exclusion map
        exclusion = crop_signal - inclusion

        # Create the input of the network
        crop = to_tensor(crop)
        inclusion = torch.as_tensor(inclusion).unsqueeze(0)
        exclusion = torch.as_tensor(exclusion).unsqueeze(0)

        input = torch.cat([crop, inclusion, exclusion], dim=0).unsqueeze(0)
        inputs.append(input)

    return torch.cat(inputs, dim=0).float(), torch.as_tensor(offsets)


def merge_masks(shape, masks, offsets):
    """
    Merge all masks in one mask.

    Parameters
    ----------
    shape : tuple
        The width and height of the mask.
    masks : Tensor
        A tensor of masks.
    offsets : Tensor
        A tensor of offsets from the masks.

    Return
    ------
    mask : Tensor
        The merged mask.
    """

    height, width = shape
    mask = torch.zeros((height, width), dtype=torch.uint8)

    for m, offset in zip(masks, offsets):
        h, w = m.shape[1:]
        x, y = offset

        # Merge the sub masks
        mask[y:y+h, x:x+w] |= m.squeeze().type(torch.uint8)

    return mask.type(torch.float32)


if __name__ == "__main__":
    args = parse_arguments()

    # Load the model weights
    model = NuClick()
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    # Get the path of the image
    path = get_image_path()
    if path == ():
        exit(1)

    # Load the image
    image = cv2.imread(path)

    # Draw on the image
    painter = Painter(path, image)
    signal, squiggles = painter.draw_on_image()

    # Extract patches of the image and create the input to the network
    inputs, offsets = get_patches(image, squiggles, signal)

    # Prediction
    with torch.no_grad():
        predictions = model(inputs)

    # Post-process the predictions
    masks = post_process(
        predictions, min_size=args.min_size, area_threshold=args.area_threshold
    )

    # Construct the complete segmentation
    mask = merge_masks(image.shape[:2], masks, offsets)

    # Save the resulting image with the mask
    filename = os.path.basename(os.path.normpath(path))
    result = label2rgb(mask.numpy(), image, colors=['yellow'], bg_label=0)

    cv2.imwrite(os.path.join(args.dest, f'mask-{filename}'), to_uint8(mask))
    cv2.imwrite(os.path.join(args.dest, f'result-{filename}'), to_uint8(result))
