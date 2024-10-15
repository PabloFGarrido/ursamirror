"""
This module contains the functions that define the geometry of the star and the
measuresments using them
"""

# -*- coding: utf-8 -*-

import numpy as np

from .transformations import pixel2polar

def star_eq(phi, rho, m, k=1, n=5):
    """
    Compute the equation that describes the star.

    Parameters
    ----------
    phi : float
        The value of the angle around the center of the star
    rho : float
        Main distance from the center of the star to the borders
    m : float
        Parameter that controls the side bending or deepness
    k : float, optional
        Roundness of the peaks. Values between [-1,1]
    n : int, optional
        Number of vertices 

    Returns
    -------
    float
        The distance from the border of the star to the center.

    Notes
    -----
    The equation that describes the shape of the star is given by:
        .. math:: 

            \\rho(\\phi) = \\rho_0 \\frac{\\cos{\\left (\\frac{2 \\sin^{-1}{(k)}+\\pi m}{2n} \\right)}}
            {\\cos\\left(\\frac{2 \\sin^{-1}{\\left(k\\cos{n \\phi} \\right) }+\\pi m}{2n}\\right)}

    The angles are reversed in the calculations so that the solution is clockwise.
    This last note may not be necessary if the shape is symmetrical
    """

    phi_rot = -phi

    num = np.cos((2*np.arcsin(k) + np.pi*m) / (2*n))
    den = np.cos((2*np.arcsin(k*np.cos(n*phi_rot)) + np.pi*m) / (2*n))

    return rho * num / den


def star_eq_dephase(phi, rho, m, deph, k=1, n=5):
    """
    Compute the equation that describes the star with a rotation (dephase).

    Parameters
    ----------
    phi : float
        The value of the angle around the center of the star
    rho : float
        Main distance from the center of the star to the borders
    m : float
        Parameter that controls the side bending or deepness
    deph : float
        dephase added to compensate a rotation
    k : float, optional
        Roundness of the peaks. Values between [-1,1]
    n : int, optional
        Number of vertices 

    Returns
    -------
    float
        The distance from the border of the star to the center.

    Notes
    -----
    The equation that describes the shape of the star is given by:
        .. math:: 

            \\rho(\\phi) = \\rho_0 \\frac{\\cos{\\left (\\frac{2 \\sin^{-1}{(k)}+\\pi m}{2n} \\right)}}
            {\\cos\\left(\\frac{2 \\sin^{-1}{\\left(k\\cos{n \\left[\\phi + \\omega\\right]} \\right) }+\\pi m}{2n}\\right)}

    """
    return star_eq(phi+deph, rho, m, k, n)


def fitting_star_eq(params, outer_points, inner_points, fit_k=False, n_prop=5):
    """
    Prepare the star equation to be minimized for both edges of the star, 
    sharing the m, k and n parameters and the center of the star. 

    Parameters
    ----------
    params : numpy.ndarray
        1D array of size 6 or 7, depending on if fit_k is True/False. Contains 
        the values for: X coordinate of the center of the star, Y coordinate of 
        the center of the star, inner radius, outer radius, parameter m, dephase,
        and, if fit_k==True, the k parameter.
    outer_points : numpy.ndarray
        2D array of shape (n, 2) containing the pixel coordinates points (I, J)
        for the outer border of the star. 
    inner_points : numpy.ndarray
        2D array of shape (n, 2) containing the pixel coordinates points (I, J)
        for the inner border of the star. 
    fit_k : bool, optional
        Wheter or not to fit the k parameter. By default, False.
    n_prop : int, optional
        Number of peaks of the star. By default, 5.


    Returns
    -------
    float
        The sum of chi² values for the inner and outer border.

    """
    if fit_k:
        center_x, center_y, rho1, rho2, m, deph, k = params

        angles_out, distances_out = pixel2polar(
            outer_points, center_x, center_y)
        chi1 = np.sum((distances_out - star_eq_dephase(angles_out,
                      rho1, m, deph, k, n=n_prop)) ** 2)

        angles_in, distances_in = pixel2polar(
            inner_points, center_x, center_y)
        chi2 = np.sum((distances_in - star_eq_dephase(angles_in,
                      rho2, m, deph, k, n=n_prop)) ** 2)

    else:
        k_prop = 1
        center_x, center_y, rho1, rho2, m, deph = params

        angles_out, distances_out = pixel2polar(
            outer_points, center_x, center_y)
        chi1 = np.sum((distances_out - star_eq_dephase(angles_out,
                      rho1, m, deph, k=k_prop, n=n_prop)) ** 2)

        angles_in, distances_in = pixel2polar(
            inner_points, center_x, center_y)
        chi2 = np.sum((distances_in - star_eq_dephase(angles_in,
                      rho2, m, deph, k=k_prop, n=n_prop)) ** 2)

    return chi1 + chi2

def angular_width_star(angle, border_out_coef, border_in_coef):
    """
    Compute the star width for a given angle, providing the coefficients corresponding
    to the dephase equation that fits the outer and eht inner borders.

    Parameters
    ----------
    angle : float
        Angle value of polar coordinates, in radians.
    border_out_coef : numpy.ndarray
        1D array of shape (1, 5) containing the (rho, m, deph, k, n) values corresponding
        to the outer border of the star.
    border_in_coef : numpy.ndarray
        1D array of shape (1, 5) containing the (rho, m, deph, k, n) values corresponding
        to the inner border of the star.
    Returns
    -------
    float
        The widht of the star, in pixels, for the given angle

    """

    return star_eq_dephase(angle, *border_out_coef) - star_eq_dephase(angle, *border_in_coef)

def residuals_mean_star(angle, distance, border_out_coef, border_in_coef):
    """
    Calculate the distance, in pixels, between the provided pixel in polar coordinates
    and the ideal star (the mean star). 
    The distance is normalized by the width of the star for that angle.

    Parameters
    ----------
    angle : float
        Angle value of polar coordinates, in radians, and measured clockwise
    distance : float
        Distance value, in pixels, from the center of the star
    border_out_coef : numpy.ndarray
        1D array of shape (1, 5) containing the (rho, m, deph, k, n) values corresponding
        to the outer border of the star.
    border_in_coef : numpy.ndarray
        1D array of shape (1, 5) containing the (rho, m, deph, k, n) values corresponding
        to the inner border of the star.

    Returns
    -------
    float
        The normalized residuals for the given pixel.

    """

    y_theo = 0.5 * star_eq_dephase(angle, *border_out_coef) + \
        0.5 * star_eq_dephase(angle, *border_in_coef)

    return (-y_theo + distance) / angular_width_star(angle, border_out_coef, border_in_coef)


def residuals_by_interval(angles_residuals, residuals, centered_angles):
    """
    Calculate the mean residuals and mean squared residuals for given angle bins

    Parameters
    ----------
    angles_residuals : numpy.ndarray
        1D array containing the corresponding angles of every pixel of the path.
    residuals : numpy.ndarray
        1D array containing the corresponding residuals of every pixel of the path
    centered_angles : numpy.ndarray
        1D array containing the centered angles from the density analysis

    Returns
    -------
    list
        A list of two elements containing the mean residuals and
        mean squared residuals, corresponding to the centered angles.
    """
    bin_size = centered_angles[1]-centered_angles[0]

    index_sort = angles_residuals.argsort()
    residuals_sort = residuals[index_sort]
    angles_residuals_sort = angles_residuals[index_sort]

    mean_residuals = np.zeros_like(centered_angles)
    mean_sqrd_residuals = np.zeros_like(centered_angles)

    for i, angle in enumerate(centered_angles):

        mean_residuals[i] = np.mean(residuals_sort[
            (angles_residuals_sort > (angle - bin_size/2)) &
            (angles_residuals_sort < (angle + bin_size/2))])

        mean_sqrd_residuals[i] = np.mean(residuals_sort[
            (angles_residuals_sort > (angle - bin_size/2)) &
            (angles_residuals_sort < (angle + bin_size/2))]**2)

    mean_residuals = np.nan_to_num(mean_residuals)
    mean_sqrd_residuals = np.nan_to_num(mean_sqrd_residuals)

    return [mean_residuals, mean_sqrd_residuals]
