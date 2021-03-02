"""
Get the images and their ground truth from Cytomine
"""

import logging
import os

from argparse import ArgumentParser

from shapely import wkt
from shapely.affinity import affine_transform

from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstanceCollection


def parse_arguments():
    """Parse the arguments of the program. 

    Return
    ------
    args : class argparse.Namespace
        The parsed arguments.
    """

    parser = ArgumentParser(description="Get a dataset from Cytomine.")

    parser.add_argument(
        '--host',
        default='research.cytomine.be',
        help="The Cytomine host"
    )
    parser.add_argument(
        '--public_key',
        help="The Cytomine public key"
    )
    parser.add_argument(
        '--private_key',
        help="The Cytomine private key"
    )
    parser.add_argument(
        '--project_id',
        help="The project from which we want the images"
    )
    parser.add_argument(
        '--path',
        default='./',
        help="Where to store the dataset"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    base_path = os.path.join(args.path, str(args.project_id))

    # Connect to Cytomine
    Cytomine(args.host, args.public_key, args.private_key, logging.INFO)

    # Get all images of a Cytomine project
    images = ImageInstanceCollection()
    images = images.fetch_with_filter('project', args.project_id)

    # Get the all the annotations of all the images of the project
    annotations = AnnotationCollection(
        project=args.project_id,
        showWKT=True,
        showMeta=True,
        showGIS=True
    )
    annotations.fetch()

    # Sort the annotations for each image
    image_to_label = {image: [annotation for annotation in annotations
                              if annotation.image == image.id]
                      for image in images}

    # Download patches of image, mask, pseudo inclusion and exclusion masks
    for image, labels in image_to_label.items():
        for annotation in labels:
            geometry = wkt.loads(annotation.location)
            coords = affine_transform(geometry, [1, 0, 0, -1, 0, image.height])
            bbox = tuple(map(int, coords.bounds))

           # Remove the annotation being the whole image
            if image.width == bbox[2] and image.height == bbox[3]:
                labels.remove(annotation)
                continue

            x = max(0, bbox[2] - 512)
            y = max(0, bbox[3] - 512)

            # Download the image patch
            path = os.path.join(base_path, 'images', f'{annotation.id}.jpg')
            image.window(x, y, 512, 512, path)

            # Get the annotation ids
            ids = [label.id for label in labels]

            # Download the mask patch
            path = os.path.join(base_path, 'masks', f'{annotation.id}.jpg')
            image.window(x, y, 512, 512, path, mask=True, annotations=ids)

            # Download the pseudo inclusion mask
            path = os.path.join(base_path, 'inclusions',
                                f'{annotation.id}.jpg')
            image.window(x, y, 512, 512, path, mask=True,
                         annotations=[annotation.id])

            # Remove the current annotation
            ids.remove(annotation.id)

            # Download the pseudo exclusion mask
            path = os.path.join(base_path, 'exclusions',
                                f'{annotation.id}.jpg')
            image.window(x, y, 512, 512, path, mask=True, annotations=ids)
