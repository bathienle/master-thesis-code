"""
Get the images and the masks from Cytomine
"""

import logging
import os

from argparse import ArgumentParser

from shapely import wkt
from shapely.affinity import affine_transform

from cytomine import Cytomine
from cytomine.models import (
    AnnotationCollection, ImageInstanceCollection, TermCollection
)


def parse_arguments():
    """
    Parse the arguments of the program.

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(description="Get a dataset from Cytomine.")

    parser.add_argument(
        '--host',
        help="The Cytomine host."
    )
    parser.add_argument(
        '--public_key',
        help="The Cytomine public key."
    )
    parser.add_argument(
        '--private_key',
        help="The Cytomine private key."
    )
    parser.add_argument(
        '--project_id',
        help="The project from which we want the images."
    )
    parser.add_argument(
        '--path',
        default='./',
        help="Where to store the dataset."
    )
    parser.add_argument(
        '--term',
        help="Get specific term annotation."
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    base_path = os.path.join(args.path, str(args.project_id))

    # Connect to Cytomine
    Cytomine(args.host, args.public_key, args.private_key, logging.INFO)

    # Get the term ID
    terms = TermCollection().fetch_with_filter('project', args.project_id)
    term_id = next(
        (term.id for term in terms if term.name.lower() == args.term.lower())
    )

    # Get all images of a Cytomine project
    images = ImageInstanceCollection()
    images = images.fetch_with_filter('project', args.project_id)

    # Get the all the annotations of all the images of the project
    annotations = AnnotationCollection(
        project=args.project_id,
        term=term_id,
        showWKT=True,
        showMeta=True,
        showGIS=True
    )
    annotations.fetch()

    # Sort the annotations for each image
    image_to_label = {
        image: [annot for annot in annotations if annot.image == image.id]
        for image in images
    }

    # Download patches of image, mask, pseudo inclusion and exclusion masks
    for image, labels in image_to_label.items():
        for annotation in labels:
            geometry = wkt.loads(annotation.location)
            coords = affine_transform(geometry, [1, 0, 0, -1, 0, image.height])
            x_min, y_min, x_max, y_max = tuple(map(int, coords.bounds))

            # Remove the annotation being the whole image
            if image.width == x_max and image.height == y_max:
                labels.remove(annotation)
                continue

            # Compute the coordinates of the crops
            x = min(x_min, max(0, x_max - 512))
            y = min(y_min, max(0, y_max - 512))
            width = 512 if x_max-x_min < 512 else x_max-x_min
            height = 512 if y_max-y_min < 512 else y_max-y_min

            # Download the image patch
            path = os.path.join(base_path, 'images', f'{annotation.id}.jpg')
            image.window(x, y, width, height, path)

            # Get the annotation ids
            ids = [label.id for label in labels]

            # Download the mask patch
            path = os.path.join(base_path, 'masks', f'{annotation.id}.jpg')
            image.window(x, y, width, height, path, mask=True, annotations=ids)

            # Download the pseudo inclusion mask
            path = os.path.join(base_path, 'inclusions', f'{annotation.id}.jpg')
            image.window(x, y, width, height, path, mask=True, annotations=[annotation.id])

            # Remove the current annotation
            ids.remove(annotation.id)

            # Download the pseudo exclusion mask
            path = os.path.join(base_path, 'exclusions', f'{annotation.id}.jpg')
            image.window(x, y, width, height, path, mask=True, annotations=ids)
