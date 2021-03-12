"""
Split the dataset into train/val/test sets from a Cytomine dataset.
"""

import logging
import os

from argparse import ArgumentParser

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

    parser = ArgumentParser(
        description="Split dataset into train/val/test sets."
    )

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
        help="The project from the dataset."
    )
    parser.add_argument(
        '--path',
        help="Path to the dataset."
    )
    parser.add_argument(
        '--term',
        help="The specific term of the annotations."
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.8,
        help="The train/test split ratio of the dataset."
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
        showMeta=True,
    )
    annotations.fetch()

    # Map the annotations to the corresponding image
    image_to_label = {
        image.id: [annot.id for annot in annotations if annot.image == image.id]
        for image in images
    }

    # Remove no label images
    image_to_label = {
        image: labels for image, labels in image_to_label.items() if labels
    }

    # Sort the image by the number of annotations
    image_to_label = {
        image: labels for image, labels in
        sorted(
            image_to_label.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )
    }

    # Get the number of annotations
    n_labels = sum([len(v) for v in image_to_label.values()])

    # Split the dataset with the ratio
    split_train_test = round(n_labels * args.ratio)
    split_train_val = round(split_train_test * 0.8)  # Split 80/20

    train_set = []
    test_set = []

    for image, labels in image_to_label.items():
        if len(train_set) < split_train_test:
            train_set.extend(labels)
        else:
            test_set.extend(labels)

    # Create the filenames
    train_set = [f'{id}.jpg' for id in train_set]
    test_set = [f'{id}.jpg' for id in test_set]

    val_set = train_set[split_train_val:]
    train_set = train_set[:split_train_val]

    # Split the annotations to the corresponding directories
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
                os.rename(
                    os.path.join(args.path, subdir, filename),
                    os.path.join(args.path, name, subdir, filename)
                )

    # Delete empty directories
    for subdir in subdirs:
        os.rmdir(os.path.join(args.path, subdir))