"""
This module contains the main comon utilities for the package.

"""

# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve
from scipy.spatial.distance import cdist
from skimage.draw import line
from skimage.measure import label as skimage_label
from skimage.morphology import binary_dilation, binary_closing, skeletonize, disk
from skimage.segmentation import flood


def path_thickness(path_image):
    """
    Estimate the thickness of the drawn path. 

    Parameters
    ----------
    image : numpy.ndarray
        2D array of shape (n, m) containing the image of the drawing path. 

    Returns
    -------
    float
        Estimated path thickness

    """
    if len(path_image.shape) != 2:
        raise ValueError(
            "The image has more than 1 channel. Convert it to a (n, m) numpy array")

    binary_image = path_image > 0.

    # Determine the distance of every pixel > 0 to the foreground.
    # Find the shape's skeleton and then calculate the mean distance * 2 (thickness, not radius)

    distances = distance_transform_edt(binary_image)
    skeleton = skeletonize(binary_image)
    skeleton_thickness = distances[skeleton] * 2

    return np.mean(skeleton_thickness)


def split_borders(border_image):
    """
    Split the inner and outer borders of the star from an image containing both.

    Parameters
    ----------
    border_image : numpy.ndarray
        2D array of shape (n, m) containing the image of the borders. 

    Returns
    -------
    list
        A list of two elements containing the image with the inner
        and the outer borders

    """
    labeled_array = skimage_label(border_image)

    if np.sum(labeled_array == 1) > np.sum(labeled_array == 2):
        outer_border = labeled_array == 1
        inner_border = labeled_array == 2
    else:
        outer_border = labeled_array == 2
        inner_border = labeled_array == 1

    return [inner_border, outer_border]


def inner_star(border_image):
    """
    Determine the pixels between the borders of the star

    Parameters
    ----------
    border_image : numpy.ndarray
        2D array of shape (n, m) containing the image of the borders. 

    Returns
    -------
    numpy.ndarray
        2D array of shape (n, m) containing the binary image
        of the space between the borders.
    """

    inn, out = split_borders(border_image)

    inn_fill = flood(inn, tuple(np.array(inn.shape)//2))
    out_fill = flood(out, tuple(np.array(out.shape)//2))

    inner = ~inn_fill * out_fill

    # Remove borders
    inner = (inner * ~inn) * ~out

    return inner


def endpoints(sk_image):
    """
    Locate the ending points for the different pieces of the drawn path

    Parameters
    ----------
    sk_image : numpy.ndarray
        2D array of shape (n, m) containing skeleton of the drawn path 

    Returns
    -------
    numpy.ndarray
        2D array of shape (n, m) containing the binary image
        of ending pixels for each piece of path
    """

    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    padding = ((1, 1), (1, 1))

    sums = convolve(np.pad(sk_image.astype(int), padding),
                    mask, mode='constant')

    return (sums[1:-1, 1:-1]*sk_image) == 1


def valid_regions(path_image, min_size=16):
    """
    Find the independent pieces of the path larger than a certain size. Imput 
        image must be a boolean matrix.

    Parameters
    ----------
    path_image : numpy.ndarray
        2D array of shape (n, m) containing the initial suggestion of the
        drawn path 
    min_size : int
        Minimun size of an independent path to be considered in the analysis,
        by default 16

    Returns
    -------
    numpy.ndarray
        2D array of shape (n, m) containing the mask with all the valid regions
    """

    labeled_image = skimage_label(path_image)
    labels, pixel_counts = np.unique(labeled_image, return_counts=True)
    valid_labels = labels[pixel_counts > min_size]

    return np.isin(labeled_image, valid_labels)


def expand_through_border(points_coordinates, distance_matrix, border, path_thick):
    """
    Connects the points in an image by expanding them along the border of
    the shape. This function has been created to connect the parts of the path 
    that are cut when crossing the edge of the figure.

    Parameters
    ----------
    points_coordinates : numpy.ndarray
        2D array of shape (2, l) containing the pixel coordinates of the 
        points to be connected.
    distance_matrix : numpy.ndarray
        2D array of shape (l, l) containing the distances between all the 
        points.
    border : numpy.ndarray
        2D array of shape (n, m) containing the borders of the shape
    path_thi : float
        Thickness of the drawn path.

    Returns
    -------
    numpy.ndarray
        2D array of shape (n, m) containing the mask with all the valid regions
    """

    # Find every pair of endpoints that should be connected. Solved by,
    # finding the closest that is not part of the same piece
    connections = []
    for i in range(distance_matrix.shape[0]):
        closest_neighbor_index = np.argmin(distance_matrix[i, :])
        if np.argmin(distance_matrix[closest_neighbor_index, :]) == i:
            connections.append([i, closest_neighbor_index])
    connections = np.array(connections)

    border_new = np.zeros_like(border)
    for connection in connections:
        border_new_aux = np.zeros_like(border)
        # Create a straight line to connect the points
        rr, cc = line(
            *points_coordinates[:, connection[0]],
            *points_coordinates[:, connection[1]])
        border_new_aux[rr, cc] = True

        # not allowing connections too far from the border
        if ((border_new_aux & border).sum()/border_new_aux.sum()) > .25:
            border_new[rr, cc] = True

    return binary_dilation(border_new, disk(path_thick/2))


def fill_path(pre_path, border, min_size=16):
    """
    Complete a drawn path that is interrupted by the intersection with the edges

    Parameters
    ----------
    pre_path : numpy.ndarray
        2D array of shape (n, m) containing the initial suggestion of the
        drawn path 
    border : numpy.ndarray
        2D array of shape (n, m) containing the borders of the shape
    min_size : int
        Minimun size of an independent path to be considered in the analysis,
        by default 16

    Returns
    -------
    numpy.ndarray
        2D array of shape (n, m) containing the binary image
        of the completed drawn path
    """

    # Clean the path, get its thickness and label the skeleton.
    pre_path_clean = pre_path * valid_regions(pre_path, min_size)
    path_thick = path_thickness(pre_path_clean)
    pp_clean_sk = skeletonize(pre_path_clean)
    labeled_sk = skimage_label(pp_clean_sk, connectivity=2)

    # Get the endpoints of every piece of the drawn path.
    endpoints_coordinates = np.array(np.nonzero(endpoints(pp_clean_sk)))
    endpoints_label = labeled_sk[*endpoints_coordinates]

    # Determine the distance between all the endpoints.
    # Self-distance and distance to other points of the same piece of the path
    # are set to inf
    distance_matrix = cdist(endpoints_coordinates.transpose(),
                            endpoints_coordinates.transpose(), 'euclidean')
    mask = endpoints_label[:, None] == endpoints_label[None, :]
    distance_matrix[mask] = np.inf

    filled_gaps = expand_through_border(endpoints_coordinates,
                                        distance_matrix, border, path_thick)

    completed_path = (filled_gaps & border) | pre_path_clean
    completed_path *= valid_regions(completed_path, min_size)
    completed_path = binary_closing(completed_path, disk(path_thick))

    return completed_path
