"""
This module contains the STAR class which allows for analysis and processing
of star-shaped drawings within images.
"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from skimage import io
from skimage.morphology import skeletonize, binary_dilation, disk
import matplotlib.pyplot as plt
import pandas as pd

from ursamirror.utils import (fitting_star_eq, path_thickness, pixel2polar,
                              polar2pixel, residuals_by_interval,
                              residuals_mean_star, split_borders, star_eq_dephase)


class STAR:
    """
    STAR class for processing and analysis of Mirror Tracing Task star-like drawings.


    Args
    ----------
    path_to_image : str
        Path to the image in the standardized format: path, inner part, and borders
        in the red, green and blue channel respectively.
    number_angles : int, optional
        Number of angle divisions over which to calculate density and mean residuals.
        By default, 360
    peaks : int, optional
        Number of peaks of the star. By default, 5.
    fit_k : bool, optional
        Wheteher to fit or not the k parameter in the star equation. By default, False.

    Attributes
    ----------
    original : numpy.ndarray
        3D array of shape (n, m, 3) or (n, m, 4) containing the original image.
    number_angles : int
        Number of angle divisions over which to calculate density and mean residuals.
    path : numpy.ndarray
        2D array of shape (n, m) containing the image of the drawing path. 
    inner : numpy.ndarray
        2D array of shape (n, m) containing the image of the inner part of the star. 
    border : numpy.ndarray
        2D array of shape (n, m) containing the image of the borders of the star. 
    path_skeleton : numpy.ndarray
        2D boolean array of shape (n, m) containing the image of the skeleton of drawing path. 
    border_in : numpy.ndarray
        2D boolean array of shape (n, m) containing the inner border of the star.
    border_out : numpy.ndarray
        2D boolean array of shape (n, m) containing the outer border of the star.
    border_in_coef : numpy.ndarray
        1D array of shape (1, 5) containing the (rho, m, deph, k, n) values corresponding
        to the inner border of the star.    
    border_out_coef : numpy.ndarray
        1D array of shape (1, 5) containing the (rho, m, deph, k, n) values corresponding
        to the outer border of the star.
    center_x : float
        X coordinate (J in pixels) of the center of the star.
    center_y : float
        Y coordinate (I in pixels) of the center of the star.
    path_thick : float
        Estimated thickness, in pixels, of the dranw path.

    Methods
    -------
    __init__(path_to_image, number_angles=360, peaks=5, fit_k=False)
        Initializes the STAR object
    path_polar()
        Transform the path pixel coordinates to polar coordinates.
    path_skeleton_polar()
        Transform the path pixel coordinates of the skeletonized path 
        to polar coordinates.
    residuals(path="Complete")
        Calculate the distance, in pixels, between the provided pixels in
        polar coordinates and the ideal star (the mean star). 
        The distance is normalized by the width of the star for that angle.
    expected_image(number_of_angles=3600)
        Reconstruct the ideal expected star 
    density()
        Calculate the density as the ratio of drawn pixels and expected pixels
        per unit of angular resolution.
    export()
        Create a pandas DataFrame containing the Angles, Density, Mean Residuals, 
        and the Mean Squared Residuals of the star drawing. 
        Angles are measured in radians.
    plot()
        Plot original image, drawn path in polar coordinates, normalized residuals,
        and density.

    """

    def __init__(self, path_to_image, number_angles=360, peaks=5, fit_k=False):
        """
        Initializes the STAR object with the provided image path and parameters.

        Parameters
        ----------
        path_to_image : str
            Path to the image in the standardized format: path, inner part, and borders
            in the red, green and blue channel respectively.
        number_angles : int, optional
            Number of angle divisions over which to calculate density and mean residuals.
            By default, 360
        peaks : int, optional
            Number of peaks of the star. By default, 5.
        fit_k : bool, optional
            Wheteher to fit or not the k parameter in the star equation. By default, False.

        """
        self.original = io.imread(path_to_image)
        self.number_angles = number_angles

        im = self.original/self.original.max()
        self.path = im[:, :, 0]
        self.inner = im[:, :, 1]
        self.border = im[:, :, 2]
        self.path_skeleton = skeletonize(self.path)

        border_points = np.transpose(np.nonzero(self.border))
        center_y, center_x = np.mean(border_points, axis=0)

        self.border_in, self.border_out = split_borders(self.border)

        outer_points = np.transpose(np.nonzero(self.border_out))
        inner_points = np.transpose(np.nonzero(self.border_in))

        # The following if/else fits the borders of the star to the proposed equation.
        ly, lx = self.original.shape[:2]
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

        self.border_out_coef = np.array([rho1, m, deph, k, peaks])
        self.border_in_coef = np.array([rho2, m, deph, k, peaks])
        self.center_x, self.center_y = center_x, center_y
        self.path_thick = path_thickness(self.path)

    def path_polar(self):
        """
        Transform the path pixel coordinates to polar coordinates.

        Parameters
        ----------

        Returns
        -------
        list
            A list of two elements containing the angles and distances arrays. 
            Angles are measured in radians and distance in pixels.

        """
        return pixel2polar(np.transpose(np.nonzero(self.path)),
                           self.center_x,
                           self.center_y)

    def path_skeleton_polar(self):
        """
        Transform the path pixel coordinates of the skeletonized path 
        to polar coordinates.

        Parameters
        ----------

        Returns
        -------
        list
            A list of two elements containing the angles and distances arrays
            of the skeletonized path. 
            Angles are measured in radians and distance in pixels.

        """
        return pixel2polar(np.transpose(np.nonzero(self.path_skeleton)),
                           self.center_x,
                           self.center_y)

    def residuals(self, path="Complete"):
        """
        Calculate the distance, in pixels, between the provided pixels in
        polar coordinates and the ideal star (the mean star). 
        The distance is normalized by the width of the star for that angle.

        Parameters
        ----------
        path : str, optional
            Type of drawn path used to determine the residuals: "Complete" will use
            all the pixels; "Skeleton" will use just the path skeleton.
            By default, "Complete"

        Returns
        -------
        list
            A list of two elements containing the angles and residuals arrays. 
            Angles are measured in radians.

        """

        if path == "Complete":
            return [self.path_polar()[0],
                    residuals_mean_star(*self.path_polar(),
                                        self.border_out_coef,
                                        self.border_in_coef)]
        elif path == "Skeleton":
            return [self.path_skeleton_polar()[0],
                    residuals_mean_star(*self.path_skeleton_polar(),
                                        self.border_out_coef,
                                        self.border_in_coef)]
        else:
            return "Unrecognized kind of path"

    def expected_image(self, number_of_angles=3600):
        """
        Reconstruct the ideal expected star using the same path thickness as 
        the original.

        Parameters
        ----------
        number_of_angles : int, optional
            Number of angles used to reconstruct the image. By default, 3600

        Returns
        -------
        numpy.ndarray
            2D binary array of shape (n, m) containing the expected ideal star.
        """
        # Array of angles to be used to reconstruct the image, and initial image
        angles = np.linspace(0, 2*np.pi, number_of_angles)
        image_aux = np.zeros(self.original.shape[:2])

        # Ideal star distance, correspondent to the mean value between the edges
        distances = 0.5*star_eq_dephase(angles, *self.border_out_coef) + \
            0.5*star_eq_dephase(angles, *self.border_in_coef)

        ideal_star_coordinates = np.array(polar2pixel(
            angles, distances, self.center_x, self.center_y)).transpose().astype(int)

        image_aux[ideal_star_coordinates[:, 0],
                  ideal_star_coordinates[:, 1]] = 1

        # The ideal star is dilated to have the same thickness as the original
        dilated_image = binary_dilation(
            skeletonize(image_aux), disk(self.path_thick/2))

        return dilated_image

    def density(self):
        """
        Calculate the density as the ratio of drawn pixels and expected pixels
        per unit of angular resolution.

        Parameters
        ----------

        Returns
        -------
        list
            A list of two elements containing the angles and distances arrays. 
            Angles are measured in radians and distance in pixels.

        """
        # Get, for every pixel, the corresponding angle, both in the drawn
        # and expected image
        path_angles = self.path_polar()[0]
        expected_angles = pixel2polar(np.transpose(np.nonzero(self.expected_image())),
                                      self.center_x,
                                      self.center_y)[0]

        # Count the number of pixels within every range of angles
        counts_path = np.histogram(path_angles, bins=np.linspace(
            0, 2*np.pi, self.number_angles, endpoint=True))
        counts_expected = np.histogram(expected_angles, bins=np.linspace(
            0, 2*np.pi, self.number_angles, endpoint=True))

        return [0.5*counts_path[1][1:]+0.5*counts_path[1][:-1],
                counts_path[0]/counts_expected[0]]

    def export(self):
        """
        Create a pandas DataFrame containing the Angles, Density, Mean Residuals, 
        and the Mean Squared Residuals of the star drawing. 
        Angles are measured in radians.

        Parameters
        ----------

        Returns
        -------
        pandas.core.frame.DataFrame
            A pandas DataFrame with columns
            ["Angles", "Density", "Residuals", "Residuals_sqrd"]
        """

        centered_angles, density = self.density()
        angles_residuals, residuals = self.residuals()

        mean_residuals, mean_sqrd_residuals = residuals_by_interval(angles_residuals,
                                                                    residuals, centered_angles)

        data = {"Angles": centered_angles,
                "Angles_degrees": np.rad2deg(centered_angles),
                "Density": density,
                "Residuals": mean_residuals,
                "Residuals_sqrd": mean_sqrd_residuals}

        return pd.DataFrame(data)

    def plot(self):
        """
        Plot original image, drawn path in polar coordinates, normalized residuals,
        and density.
        """
        circlex = np.linspace(0, 2*np.pi, num=3600, endpoint=True)
        circley = np.zeros(3600)

        fig, ax = plt.subplots(
            1, 4, subplot_kw={'projection': 'polar'}, figsize=(48, 8))
        ax[0].remove()
        ax0 = fig.add_subplot(1, 4, 1)
        ax0.imshow(self.original)
        ax0.axis('off')
        ax0.set_title("Original image", fontsize=20)

        ax[1].plot(*pixel2polar(np.transpose(np.nonzero(self.border)),
                                self.center_x,
                                self.center_y),
                   ".", color="firebrick", label="border")

        ax[1].plot(*self.path_polar(), "go", label="Path")
        ax[1].plot(*self.path_skeleton_polar(), "y.", label="Skeleton path")
        ax[1].legend()
        ax[1].set_title("Star Drawing (polar coordinates)", fontsize=20)
        ax[1].set_theta_zero_location("N")
        ax[1].set_theta_direction(-1)

        angles_in, distance_in = pixel2polar(np.transpose(np.nonzero(self.border_in)),
                                             self.center_x,
                                             self.center_y)
        angles_out, distance_out = pixel2polar(np.transpose(np.nonzero(self.border_out)),
                                               self.center_x,
                                               self.center_y)
        ax[2].plot(angles_in,
                   residuals_mean_star(angles_in, distance_in,
                                       self.border_out_coef,
                                       self.border_in_coef),
                   ".", color="firebrick", label="border")
        ax[2].plot(angles_out,
                   residuals_mean_star(angles_out, distance_out,
                                       self.border_out_coef,
                                       self.border_in_coef),
                   ".", color="firebrick")
        ax[2].plot(*self.residuals(), "go", label="residuals")
        ax[2].plot(circlex, circley, "k", label="0 value")
        ax[2].set_title(
            "Standarized Residuals (polar coordinates)", fontsize=20)
        ax[2].legend()
        ax[2].set_theta_zero_location("N")
        ax[2].set_theta_direction(-1)

        ax[3].plot(*self.density(), ".", ms=10)
        ax[3].plot(circlex, circley+1, "k")
        ax[3].set_title(
            "Standarized Dots density (polar coordinates)", fontsize=20)
        ax[3].set_theta_zero_location("N")
        ax[3].set_theta_direction(-1)

        fig.tight_layout()
        fig.show()
