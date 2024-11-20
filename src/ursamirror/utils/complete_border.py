"""
This module contains has been created specifically for completing those
borders that may have gaps. It makes the code slower.
"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from skimage.morphology import binary_dilation, skeletonize, disk

from .star_equations import fitting_star_eq, star_eq_dephase
from .transformations import polar2pixel
from .utils import path_thickness, split_borders


def prepare_fit(border_image, peaks=5, fit_k=False):
    """
    get the fitted coefficients fotr the outer and inner borders

    Parameters
    ----------
    border_image : numpy.ndarray
        2D array of shape (n, m) containing the image of the borders. 
    peaks : int, optional
        Number of peaks of the star. By default, 5.
    fit_k : bool, optional
        Wheteher to fit or not the k parameter in the star equation. By default, False.
    Returns
    -------
    list
        List containing three (3) elements: 
            a tuple with X and Y coordinates of the center
            numpy array with the outer star coefficients
            numpy array with the inner star coefficients
    """
    inn, out = split_borders(border_image)

    outer_points = np.transpose(np.nonzero(out))
    inner_points = np.transpose(np.nonzero(inn))

    ly, lx = border_image.shape[:2]
    center_y, center_x = np.mean(
        np.transpose(np.nonzero(border_image)), axis=0)
    if fit_k:
        seed = [center_x,
                center_y,
                0.5*0.5*(center_x+center_y),
                0.5*0.5*(center_x+center_y),
                2,
                0,
                1]
        bnds = ((0, lx),
                (0, ly),
                (0, lx),
                (0, ly),
                (-10, 10),
                (-np.pi, np.pi),
                (-1, 1))
        fitted_coef = minimize(fitting_star_eq, x0=seed,
                               method='L-BFGS-B', bounds=bnds,
                               args=(outer_points, inner_points, fit_k, peaks)).x
        center_x, center_y, rho1, rho2, m, deph, k = fitted_coef

    else:
        seed = [center_x,
                center_y,
                0.5*0.5*(center_x+center_y),
                0.5*0.5*(center_x+center_y),
                2,
                0]
        bnds = ((0, lx),
                (0, ly),
                (0, lx),
                (0, ly),
                (-10, 10),
                (-np.pi, np.pi))
        fitted_coef = minimize(fitting_star_eq, x0=seed,
                               method='Powell', bounds=bnds,
                               args=(outer_points, inner_points, fit_k, peaks)).x
        center_x, center_y, rho1, rho2, m, deph = fitted_coef
        k = 1
    return [(center_x, center_y),
            np.array([rho1, m, deph, k, peaks]),
            np.array([rho2, m, deph, k, peaks])]


def predict_border(border_image, center_x, center_y, coef, path_thick):
    """
    Get the expected border according to the fitted coefficients and path thickness

    Parameters
    ----------
    border_image : numpy.ndarray
        2D array of shape (n, m) containing the image of the borders. 
    center_x : float
        X coordinate for the center of the star 
    center_y : float
        Ycoordinate for the center of the star 
    coef : numpy.ndarray
        1D numpy array containing [rho, m, deph, k, peaks], the parameters for
        the star equation with dephase
    Returns
    -------
    numpy.ndarray
        2D array of shape (n, m) containing the predicted border.
    """

    # Array of angles to be used to reconstruct the image, and initial image
    angles = np.linspace(0, 2*np.pi, 36000)
    image_aux = np.zeros(border_image.shape[:2])

    # Ideal star distance
    distances = star_eq_dephase(angles, *coef)

    ideal_star_coordinates = np.array(polar2pixel(
        angles, distances, center_x, center_y)).transpose().astype(int)

    image_aux[ideal_star_coordinates[:, 0],
              ideal_star_coordinates[:, 1]] = 1

    dilated_image = binary_dilation(
        skeletonize(image_aux), disk(path_thick/3))  # /3 to not overestimate

    return dilated_image


def complete_border(border_image, peaks=5, fit_k=False):
    """
    Complete the borders of the star to avoid gaps

    Parameters
    ----------
    border_image : numpy.ndarray
        2D array of shape (n, m) containing the image of the borders. 
    peaks : int, optional
        Number of peaks of the star. By default, 5.
    fit_k : bool, optional
        Wheteher to fit or not the k parameter in the star equation. By default, False.
    Returns
    -------
    numpy.ndarray
        2D array of shape (n, m) containing the new complete borders.
    """

    # The following if/else fits the borders of the star to the proposed equation.

    inn, out = split_borders(border_image)
    (center_x, center_y), border_out_coef, border_in_coef = prepare_fit(
        border_image, peaks, fit_k)
    path_thick = 0.5*path_thickness(inn) + 0.5*path_thickness(out)

    inn_expanded = predict_border(inn, center_x, center_y,
                                  border_in_coef, path_thick)

    out_expanded = predict_border(out, center_x, center_y,
                                  border_out_coef, path_thick)

    return border_image | inn_expanded | out_expanded
