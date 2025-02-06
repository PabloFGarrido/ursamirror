#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:05:44 2024

@author: Pablo F. Garrido
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io
import ursamirror as um


def app2paper(app_image):
    """
    Converts an app image to a transparent paper-like image.
    
    Parameters:
    - app_image: Input image array
    
    Returns:
    - Transformed image array
    """
    app_transf = np.copy(app_image/255)
    app_transf[:,:,3] = np.zeros(app_transf.shape[:2])
    app_transf[:,:,3] = app_transf[:,:,0].astype(bool) + app_transf[:,:,2].astype(bool)
    app_transf[:,:,1] = np.zeros(app_transf.shape[:2])
    app_transf[:,:,2] = np.zeros(app_transf.shape[:2])
    return app_transf

def star_blue(image):
    """
    Swaps the red and blue channels of an image.
    
    Parameters:
    - image: Input image array
    
    Returns:
    - Image array with red and blue channels swapped
    """
    
    app_transf = np.copy(image)
    app_transf[:,:,0] = np.copy(image[:,:,2])
    app_transf[:,:,2] = np.copy(image[:,:,0])
    return app_transf

# Function to plot a set of images
def plot_images(images, titles, transformed_func=None, figsize=(20, 6)):
    fig, ax = plt.subplots(1, len(images), figsize=figsize)
    ax = ax.flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        if transformed_func:
            img = transformed_func(img)
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_title(title, fontsize=20)
    fig.tight_layout()
    plt.show()
    
#%% Emulated stars-4 types
online = io.imread("../raw_data/online/00_online_cut.png")
online_procs = io.imread("../processed_data/online/00_online_cut.png")
paper1 = io.imread("../raw_data/paper/02_Paper_thin.png")
paper1_procs = io.imread("../processed_data/paper/02_Paper_thin.png")
paper2 = io.imread("../raw_data/paper/01_Paper_thick.png")
paper2_procs = io.imread("../processed_data/paper/01_Paper_thick.png")
app = io.imread("../raw_data/app/04_App.png")

# Plot raw images
plot_images(
    [online, paper1, paper2, app],
    ["Online", "Paper-thin", "Paper-thick", "App"],
    transformed_func=app2paper
)

# Plot processed images
plot_images(
    [online_procs, paper1_procs, paper2_procs, app],
    ["Online", "Paper-thin", "Paper-thick", "App"]
)

# Plot combined raw and processed images
fig_all, ax_all = plt.subplots(2, 4, figsize=(20, 12))
for i, (raw, proc) in enumerate(zip([online, paper1, paper2, app], [online_procs, paper1_procs, paper2_procs, app])):
    ax_all[0, i].imshow(app2paper(raw))
    ax_all[0, i].set_title("Raw - {0}".format(["Online", "Paper-thin", "Paper-thick", "App"][i]), fontsize=20)
    ax_all[1, i].imshow(proc)
ax_all[0, 0].axis('off')
ax_all[1, 0].axis('off')
fig_all.tight_layout()
plt.show()


#%%  Different shapes form star-equation

X = np.linspace(-np.pi, np.pi, 3600)
star_params = [
    {"title": "Number of peaks", "sweep_var": "n", "sweep_values": np.linspace(3, 11, 9), "fixed_params": {"r1": 1, "r2": 1.2, "m": 1.5, "k": 1}},
    {"title": "Side deepness", "sweep_var": "m", "sweep_values": np.linspace(1, 3.5, 9), "fixed_params": {"n": 5, "r1": 1, "r2": 1.2, "k": 1}},
    {"title": "Star width", "sweep_var": "r2", "sweep_values": np.linspace(1, 2, 9), "fixed_params": {"n": 5, "r1": 1, "m": 2.5, "k": 1}},
    {"title": "Peak smoothness", "sweep_var": "k", "sweep_values": np.linspace(0, 1, 9), "fixed_params": {"n": 5, "r1": 1, "r2": 1.2, "m": 2.5}}
]

for params in star_params:
    fig, ax = plt.subplots(3, 3, subplot_kw={'projection': 'polar'})
    ax = ax.flatten()
    i = -1
    for val in params["sweep_values"]:
        i += 1
        current_params = params["fixed_params"].copy()
        current_params[params["sweep_var"]] = val
        ax[i].plot(X, um.utils.star_equations.star_eq(X, **current_params), "k")
        ax[i].plot(X, um.utils.star_equations.star_eq(X, r2=current_params["r2"], **current_params), "k")
        ax[i].axis("off")
        ax[i].set_theta_zero_location("N")
    fig.suptitle(params["title"], fontsize=20)
    fig.tight_layout()
    plt.show()



#%% anomalous stars

good = um.STAR("../raw_data/app/A.png")
osc_low = um.STAR("../raw_data/app/B.png")
out = um.STAR("../raw_data/app/C.png")
inn = um.STAR("../raw_data/app/D.png")
osc_high = um.STAR("../raw_data/app/E.png")
cut = um.STAR("../raw_data/app/F.png")
err = um.STAR("../raw_data/app/G.png")

lista = [good,osc_low,out,inn,osc_high,cut, err]
names = ["Good","Good_oscillatory","Outer","Inner","Bad_oscillatory","Cut","Strug"]
counts = [0, 0 , 1, 2, 61, 2, 12]
label = "ABCDEFG"

fig= plt.figure(figsize=(24,12))
gs = gridspec.GridSpec(2, 8)
ax = []

for i in range(4):
    ax.append(fig.add_subplot(gs[0, 2 * i:2 * i + 2]))
    
for i in range(3):
    ax.append(fig.add_subplot(gs[1, 2 * i - 7:2 * i + 2 - 7]))


ax = np.array(ax)

ax[0].imshow(app2paper(good.original/255))
ax[1].imshow(app2paper(osc_low.original))
ax[2].imshow(app2paper(out.original))
ax[3].imshow(app2paper(inn.original))
ax[4].imshow(app2paper(osc_high.original))
ax[5].imshow(app2paper(cut.original))
ax[6].imshow(app2paper(err.original))

for i in range(len(ax)):
    ax[i].axis('off')
    ax[i].set_xlim((150,850))
    ax[i].set_ylim((830,130))
    # ax[i].set_title("Star %i: %i counts" %(i+1,counts[i]),fontsize=20)
    ax[i].set_title(label[i]+") %i counts" %(counts[i]),fontsize=20)
    
fig.tight_layout()

# plt.savefig("Figures/examples.png", dpi=200)
# plt.savefig("Figures/examples.svg")

#%% matrix plot-4 stars
online_class = um.STAR("../processed_data/online/00_online_cut.png")
paper1_class = um.STAR("../processed_data/paper/02_Paper_thin.png")
paper2_class = um.STAR("../processed_data/paper/01_Paper_thick.png")
app_class = um.STAR("../raw_data/app/04_App.png")

lista = [online_class,paper1_class,paper2_class,app_class]

fig, ax = plt.subplots(5,5,figsize=(30,30),subplot_kw={'projection': 'polar'})

ax[0,0].remove()
ax[0,0] = fig.add_subplot(5,5, 1)
ax[0,0].imshow(app2paper(online_class.original))
ax[0,0].axis('off')

for i in range(1,len(lista)+1):
    ax[i,i].remove()


for i,el in enumerate(lista):
    ax[i+1,0].remove()
    ax[i+1,0] = fig.add_subplot(5,5, 6+i*5)
    ax[i+1,0].imshow(app2paper(el.original))
    ax[i+1,0].axis('off')
    
    ax[0,i+1].remove()
    ax[0,i+1] = fig.add_subplot(5,5, 2+i)
    ax[0,i+1].imshow(star_blue(app2paper(el.original)))
    ax[0,i+1].axis('off')
    # ax[i,0].set_title("Original image",fontsize=20)


circlex = np.linspace(0,2*np.pi,num=3600,endpoint=True)
circley = np.zeros(3600)
for i,el in enumerate(lista):
    for j in range(i+1,len(lista)):
        el2 = lista[j]
        ax[i+1,j+1].plot(*el.residuals(),".",ms=10,c="firebrick")
        ax[i+1,j+1].plot(*el2.residuals(),"o",ms=10,mfc="none",mec="royalblue")
        ax[i+1,j+1].plot(circlex,circley,"k")
        ax[i+1,j+1].set_xticklabels([])
        ax[i+1,j+1].set_yticklabels([])
        ax[i+1,j+1].set_theta_zero_location("N")
        ax[i+1,j+1].set_theta_direction(-1)
        # ax[i+1,j+1].set_xlim(0,np.pi/2)

circlex = np.linspace(0,2*np.pi,num=3600,endpoint=True)
circley = np.ones(3600)
for i,el in enumerate(lista):
    for j in range(i):
        el2 = lista[j]
        ax[i+1,j+1].plot(*el.density(),".",ms=10, color="firebrick")
        ax[i+1,j+1].plot(*el2.density(),"o",ms=10,mfc="none",mec="royalblue")
        ax[i+1,j+1].plot(circlex,circley,"k")
        ax[i+1,j+1].set_xticklabels([])
        ax[i+1,j+1].set_yticklabels([])
        ax[i+1,j+1].set_theta_zero_location("N")
        ax[i+1,j+1].set_theta_direction(-1)
        # ax[i+1,j+1].set_xlim(0,np.pi/2)
fig.suptitle("Residuals and density",fontsize=50)
fig.tight_layout()



#%% matrix plot-anomalous stars

lista = [good,osc_low,out,inn,osc_high,cut, err]
names = ["Good","Good_oscillatory","Outer","Inner","Bad_oscillatory","Cut","Strug"]
angles_array = np.linspace(0, 2*np.pi, 360, endpoint=True)

fig, ax = plt.subplots(8,8,figsize=(48,48),subplot_kw={'projection': 'polar'})

ax[0,0].remove()
ax[0,0] = fig.add_subplot(8,8, 1)
ax[0,0].imshow(app2paper(good.original)[160:830,160:830])
ax[0,0].axis('off')

for i in range(1,len(lista)+1):
    ax[i,i].remove()


for i,el in enumerate(lista):
    ax[i+1,0].remove()
    ax[i+1,0] = fig.add_subplot(8,8, 9+i*8)
    ax[i+1,0].imshow(app2paper(el.original)[160:830,160:830])
    ax[i+1,0].axis('off')
    
    ax[0,i+1].remove()
    ax[0,i+1] = fig.add_subplot(8,8, 2+i)
    ax[0,i+1].imshow(star_blue(app2paper(el.original))[160:830,160:830])
    ax[0,i+1].axis('off')
    # ax[i,0].set_title("Original image",fontsize=20)


circlex = np.linspace(0,2*np.pi,num=3600,endpoint=True)
circley = np.zeros(3600)
for i,el in enumerate(lista):
    for j in range(i+1,len(lista)):
        el2 = lista[j]
        ax[i+1,j+1].plot(*el.residuals(),".",ms=10,c="firebrick")
        ax[i+1,j+1].plot(*el2.residuals(),"o",ms=10,mfc="none",mec="royalblue")
        ax[i+1,j+1].plot(circlex,circley,"k")
        ax[i+1,j+1].set_xticklabels([])
        ax[i+1,j+1].set_yticklabels([])
        ax[i+1,j+1].set_theta_zero_location("N")
        ax[i+1,j+1].set_theta_direction(-1)
        # ax[i+1,j+1].set_xlim(0,np.pi/2)

circlex = np.linspace(0,2*np.pi,num=3600,endpoint=True)
circley = np.ones(3600)
for i,el in enumerate(lista):
    for j in range(i):
        el2 = lista[j]
        ax[i+1,j+1].plot(*el.density(),".",ms=10, color="firebrick")
        ax[i+1,j+1].plot(*el2.density(),"o",ms=10,mfc="none",mec="royalblue")
        ax[i+1,j+1].plot(circlex,circley,"k")
        ax[i+1,j+1].set_xticklabels([])
        ax[i+1,j+1].set_yticklabels([])
        ax[i+1,j+1].set_theta_zero_location("N")
        ax[i+1,j+1].set_theta_direction(-1)
        # ax[i+1,j+1].set_xlim(0,np.pi/2)
fig.suptitle("Residuals and density",fontsize=50)
fig.tight_layout()
