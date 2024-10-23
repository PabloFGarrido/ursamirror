"""
This module contains the function to transform the online version of the star
    to the standard format used in the project.
"""

# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from numpy import argwhere, dstack
from skimage import io, img_as_ubyte

from ursamirror.utils import inner_star


def online2std(path_to_image, new_path="none", save=False):
    """
    Transform an image from the webpage application to the standard format
    used in this project

    Parameters
    ----------
    path_to_image : str
        Path to the saved original file
    new_path : str
        Path to the new transformed file
    save : bool
        Parameter to indicate whether or not to save the image. True for saving,
        False for just returning it as a 3D array of shape (n, m, 4). By default, False.


    Returns
    -------
    numpy.ndarray
        3D array of shape (n, m, 4) containing image transformed to the 
        standard format. Channels 0, 1, 2, and 3 correspond to the path, borders, 
        inner star, and all of the elements toghether, by order.
    """

    image = io.imread(path_to_image)[:, :, :3]
    image = image/image.max()

    # The image of the star form the online version may have a thick line at the
    # bottom of the image with the time. The following 5 lines detect it and remove
    # that part of the image for the analysis

    red_aux = image[:, :, 0]
    black_aux = red_aux.sum(axis=1) == 0
    if any(black_aux):
        limit_row = argwhere(black_aux)[0, 0]
        image = image[:limit_row, :, :]

    red, green = image[:, :, 0], image[:, :, 1]

    path = (red - green) > 0  # The drawn line is in red.
    border = red < 1
    inside = inner_star(border)

    complete_image = path | inside | border

    transformed_image = img_as_ubyte(
        dstack((path, inside, border, complete_image)))

    if save:
        if new_path != "none":
            io.imsave(new_path, transformed_image)

        else:
            raise ValueError("Save path must be provided if save is True.")

    return transformed_image


if __name__ == "__main__":
    parser = ArgumentParser(description="Online image to standard format")
    parser.add_argument('input_file', type=str,
                        help='Path to online image file')
    parser.add_argument('output_file', type=str, help='Path to output file')
    args = parser.parse_args()

    online2std(args.input_file, args.output_file, True)
