"""
This module contains the function to transform the digitized paper version 
    of the star to the standard format used in the project.
"""

# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from numpy import dstack
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from ursamirror.utils import fill_path, inner_star,  valid_regions


def paper2std(path_to_image, new_path="none", path_color="auto", save=True, 
              border_size_limit=100, path_size_limit=16):
    """
    Transform an digitized image from a paper version to the standard format
    used in this project

    Parameters
    ----------
    path_to_image : str
        Path to the saved original file
    path_color : str
        Color of the drawn path. Possibilities: "red", "green", "blue", "auto".
        If "auto" is selected, the algorithm will suggest one. By default, "auto"
    save : bool
        Parameter to indicate whether or not to save the image. True for saving,
        False for just returning it as a 3D array of shape (n, m, 4).
    new_path : str
        Path to the new transformed file

    Returns
    -------
    numpy.ndarray
        3D array of shape (n, m, 4) containing image transformed to the 
        standard format. Channels 0, 1, 2, and 3 correspond to the path, borders, 
        inner star, and all of the elements toghether, by order.
    """
    image = io.imread(path_to_image)[:, :, :3]
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    color_dict = {"red": red, "green": green, "blue": blue}

    image_gray = rgb2gray(image)
    mask_background = rgb2gray(image) < threshold_otsu(image_gray)

    if path_color == "auto":
        color_values = {"red": (red*mask_background).sum(),
                        "green": (green*mask_background).sum(),
                        "blue": (blue*mask_background).sum()}
        path_color = max(color_values, key=color_values.get)

    main_color = color_dict[path_color]

    # First suggestion for the border
    pre_border = main_color < threshold_otsu(main_color)

    other_colors = [value for channel, value in color_dict.items()
                    if channel != path_color]

    # First suggestion for the drawn path
    pre_path = main_color - (other_colors[0]*0.5+other_colors[1]*0.5)
    pre_path = pre_path > threshold_otsu(pre_path)

    border = pre_border*valid_regions(pre_border, border_size_limit)
    path = fill_path(pre_path, border, path_size_limit)
    inside = inner_star(border)
    complete_image = path | (inside.astype(bool)) | (border)

    transformed_image = dstack((path, inside, border, complete_image))

    if save:
        if new_path != "none":
            io.imsave(new_path, img_as_ubyte(transformed_image))

        else:
            raise ValueError("Save path must be provided if save is True.")

    return transformed_image


if __name__ == "__main__":
    parser = ArgumentParser(description="Paper image to standard format")
    parser.add_argument('input_file', type=str,
                        help='Path to paper image file')
    parser.add_argument('output_file', type=str, help='Path to output file')
    args = parser.parse_args()

    paper2std(args.input_file, args.output_file, "auto", True)
