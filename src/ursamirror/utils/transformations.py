"""
This module contains the change of coordinates between pixel and polar
"""

# -*- coding: utf-8 -*-

import numpy as np


def pixel2polar(points, mean_x, mean_y):
    """
    Calculate polar coordinates (angles and distances) from pixel coordinates points.
    In this approach, the angle is considered starting from the top point of the star
    and measured clockwise.

    Parameters
    ----------
    points : numpy.ndarray
        2D array of shape (n, 2) containing the pixel coordinates points (I, J). 
    mean_x : float
        X-coordinate of the center of the star (equivalent to J)
    mean_y : float
        Y-coordinate of the center of the star (equivalent to I)

    Returns
    -------
    list
        A list of two elements containing the angles and distances arrays. 
        Angles are measured in radians and distance in pixels.

    """

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points should be a 2D array with shape (n, 2).")

    x_centered = points[:, 1]-mean_x
    y_centered = points[:, 0]-mean_y

    distances = np.sqrt(y_centered**2 + x_centered**2)
    # angles are transformed to start from top and clockwise
    angles = np.pi-np.arctan2(x_centered, y_centered)

    return [angles, distances]


def polar2pixel(angle, distance, mean_x, mean_y):
    """
    Calculate pixel coordinates (i, j) from polar coordinates (distance and angles).
    In this approach, the angle is considered starting from the top point of the star
    and measured clockwise.

    Parameters
    ----------
    distance : float
        Distance value, in pixels, from the center of the star
    angle : float
        Angle value of polar coordinates, in radians.
    mean_x : float
        X-coordinate of the center of the star (equivalent to J)
    mean_y : float
        Y-coordinate of the center of the star (equivalent to I)

    Returns
    -------
    list
        A list of two elements containing the (i, j) pixel coordinates.

    """

    angle = np.pi-angle  # The angles are rotated to follow our convention

    j = distance * np.sin(angle) + mean_x
    i = distance * np.cos(angle) + mean_y

    return [i, j]
