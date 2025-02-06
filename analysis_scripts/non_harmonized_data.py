#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:00:56 2024

@author: Pablo F. Garrido
"""

import numpy as np
import matplotlib.pyplot as plt
import ursamirror as um

def app2paper(app):
    app_transf = np.copy(app)
    app_transf[:,:,3] = np.zeros(app_transf.shape[:2])
    app_transf[:,:,3] = app_transf[:,:,0].astype(bool) + app_transf[:,:,2].astype(bool)
    app_transf[:,:,1] = np.zeros(app_transf.shape[:2])
    app_transf[:,:,2] = np.zeros(app_transf.shape[:2])
    return app_transf

def residuals_mean_star_raw(path_polar, border_out_coef, border_in_coef):
    """
    Calculate the distance difference (in radial coordinate) from a given point
    in polar coordinates to the mean star. The distance is normalized by the star width at that angle.
    
    Parameters
    ----------
    path_polar : numpy array
         2D array containing the polar coordinates of the point (angle, distance).
    border_out_coef : numpy.ndarray
        numpy array containing the outer border coeficientes for star_eq_dephase 
    border_in_coef : numpy.ndarray
        numpy array containing the inner border coeficientes for star_eq_dephase 

    Returns
    -------
    float
        A computed value for the distance. Negative is closer to the center.
        Positive is towards the outside of the star. 

    """
    aux_ang = path_polar[0]
    aux_rho = path_polar[1]
    y_theo = 0.5 * um.utils.star_eq_dephase(aux_ang, *border_out_coef) + 0.5 * um.utils.star_eq_dephase(aux_ang, *border_in_coef)
    return (-y_theo + aux_rho) 

def residuals_raw(star):
    angles_array = np.linspace(0, 2*np.pi, star.number_angles, endpoint=True)
    aux_ang = star.path_polar()[0]        

    H, H_bins = np.histogram(aux_ang, bins=angles_array)
    centered_angles, density = star.density()
    
    residuals_sort = residuals_mean_star_raw(star.path_polar(),
                               star.border_out_coef,
                               star.border_in_coef)[aux_ang.argsort()]
    residuals_end = np.zeros(len(density))
    residuals_end2 = np.zeros(len(density))
    aux_ang_sort = aux_ang[aux_ang.argsort()]
    
    
    
    for i in range(len(H_bins)-1):
        residuals_end[i] = np.mean(residuals_sort[(aux_ang_sort > H_bins[i]) & (aux_ang_sort < H_bins[i+1])])
        residuals_end2[i] = np.mean((residuals_sort[(aux_ang_sort > H_bins[i]) & (aux_ang_sort < H_bins[i+1])])**2)
         
    residuals_end = np.nan_to_num(residuals_end)  
    return(residuals_end)

def density_raw(star):
    coef1 = star.border_out_coef
    coef2 = star.border_in_coef
    path_angle = star.path_polar()[0]
    
    angle_aux = np.linspace(0, 2*np.pi,3600)
    R_aux = 0.5*um.utils.star_eq_dephase(angle_aux,*coef1)+0.5*um.utils.star_eq_dephase(angle_aux,*coef2)
    im_aux = np.zeros(star.original.shape)[:,:,0]
    coord_aux = np.array(um.utils.polar2pixel(R_aux,angle_aux,star.center_x,star.center_y)).transpose()
    for l in coord_aux.astype(int):
        im_aux[l[1],l[0]] = 1
    
    bins = star.number_angles
    H = np.histogram(path_angle,bins=np.linspace(0, 2*np.pi,bins,endpoint=True))

        
    return(0.5*H[1][1:]+0.5*H[1][:-1], H[0])
#%%
online_class = um.STAR("../processed_data/online/00_online_cut.png")
paper1_class = um.STAR("../processed_data/paper/02_Paper_thin.png")
paper2_class = um.STAR("../processed_data/paper/01_Paper_thick.png")
app_class = um.STAR("../raw_data/app/04_App.png")

plt.close()
fig, ax = plt.subplots(1,2,figsize=(24,12),subplot_kw={'projection': 'polar'})
ANGLES = online_class.density()[0]
ax[0].plot(ANGLES,np.zeros(len(ANGLES)),color="k",label="0 value",lw=2)
ax[0].plot(ANGLES,residuals_raw(online_class),"o",label="Online",c="firebrick")
ax[0].plot(ANGLES,residuals_raw(paper1_class),"o",label="Paper_thin",c="limegreen")
ax[0].plot(ANGLES,residuals_raw(paper2_class),"o",label="Paper_thick",c="forestgreen")
ax[0].plot(ANGLES,residuals_raw(app_class),"o",label="App",c="slateblue")
lim =  ax[0].get_ylim()
# ax[0].vlines(np.linspace(0,2*np.pi,6,endpoint=True),*lim,color="forestgreen")
ax[0].vlines(np.linspace(0,2*np.pi,6,endpoint=True)+np.deg2rad(36.0),*lim,color="goldenrod",lw=3)
ax[0].set_ylim(lim)
ax[0].set_theta_zero_location("N")
ax[0].set_theta_direction(-1)
ax[0].set_yticklabels([])
# ax[0].legend()


ax[1].plot(*density_raw(online_class),"o",c="firebrick")
ax[1].plot(*density_raw(paper1_class), "o",c="limegreen")
ax[1].plot(*density_raw(paper2_class),"o",c="forestgreen")
ax[1].plot(*density_raw(app_class),"o",c="slateblue")
lim =  ax[1].get_ylim()
# ax[0].vlines(np.linspace(0,2*np.pi,6,endpoint=True),*lim,color="firebrick",lw=3)
ax[1].vlines(np.linspace(0,2*np.pi,6,endpoint=True)+np.deg2rad(36.0),*lim)
ax[1].set_ylim(lim)
ax[1].set_theta_zero_location("N")
ax[1].set_theta_direction(-1)
ax[1].set_yticklabels([])

plt.tight_layout()
# plt.savefig("Figures/standrad_comp.png", dpi=200)
# plt.savefig("Figures/standrad_comp.svg")


#%%
plt.close("all")
fig, ax = plt.subplots(1,4,figsize=(20,6))
ax = ax.flatten()
ax[0].imshow(app2paper(online_class.original))
ax[0].plot(np.nonzero(online_class.path)[1],np.nonzero(online_class.path)[0],".",c="firebrick")
ax[0].axis('off')
ax[0].set_title("Online",fontsize=20)

ax[1].imshow(app2paper(paper1_class.original))
ax[1].plot(np.nonzero(paper1_class.path)[1],np.nonzero(paper1_class.path)[0],".",c="limegreen")
ax[1].axis('off')
ax[1].set_title("Paper-thin",fontsize=20)

ax[2].imshow(app2paper(paper2_class.original))
ax[2].plot(np.nonzero(paper2_class.path)[1],np.nonzero(paper2_class.path)[0],".",c="forestgreen")
ax[2].axis('off')
ax[2].set_title("Paper-thick",fontsize=20)


ax[3].imshow(app2paper(app_class.original))
ax[3].plot(np.nonzero(app_class.path)[1],np.nonzero(app_class.path)[0],".",c="slateblue")
ax[3].axis('off')
ax[3].set_title("App",fontsize=20)
fig.tight_layout()

# plt.savefig("Figures/stars_legend.png", dpi=200)
# plt.savefig("Figures/stars_legend.svg")
