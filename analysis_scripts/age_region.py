#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:50:20 2024

@author: Pablo F. Garrido
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
from scipy.stats import boxcox, false_discovery_control, pearsonr
from sklearn.preprocessing import StandardScaler
from ursamirror.utils import star_eq
import matplotlib.colors as mcolors
import matplotlib.cm as cm

scaler = StandardScaler()

# Star coefficients
star_out = np.array([ 1.2, -2.5,  1, 5])
star_in = np.array([ 1., -2.5,  1, 5])

#%% Read and concatenate data
LIST = np.read("participants_path.txt")

dataframes = []                
for file_path in LIST:
    df = pd.read_csv(file_path)
    dataframes.append(df)
    
combined_df = pd.concat(dataframes,ignore_index=True) 
merged_df = combined_df.groupby('Angles').apply(lambda x: x.reset_index(drop=True))
merged_df = merged_df[["Density","Residuals_sqrd","ID","Age","Residuals"]] #Relevant columns

angles = merged_df.index.get_level_values(0).unique()

# Analysis array. It will store the p-values and the mean density and residuals per angle
analysis = np.zeros((len(angles),5))
analysis[:,0] += angles

#%% Data analysis. p-value computation

for i,ang in enumerate(angles):
    sel = merged_df.xs(ang)
    sel = sel.dropna()
    sel = sel[sel["Age"]>0]
    sel["Transformed_Residuals"] = scaler.fit_transform(np.array(boxcox(np.abs(sel.Residuals_sqrd)+1e-5)[0]).reshape(-1, 1))
    sel["Transformed_Density"] = scaler.fit_transform(np.array(boxcox(sel.Density+1e-5)[0]).reshape(-1, 1))
    
    r, p = pearsonr(sel["Age"], sel["Transformed_Residuals"])
    analysis[i,1] += p
    r, p = pearsonr(sel["Age"], sel["Transformed_Density"])
    
    analysis[i,2] += p
    analysis[i,3] += sel.Density.mean()
    analysis[i,4] += sel.Residuals.mean()

analysis[:,0] = analysis[:,0]

# To include FDR control
p_val_res = analysis[:,1]
p_val_dens = analysis[:,2]

#Benjaminini-Yekutieli correction
p_val_res_corr = false_discovery_control(p_val_res, method="by")
p_val_dens_corr = false_discovery_control(p_val_dens, method="by")

analysis[:,1] = p_val_res_corr
analysis[:,2] = p_val_dens_corr
#%% figure plots


fig, ax = plt.subplots(1,3,figsize=(30,12),subplot_kw={'projection': 'polar'})
A = np.linspace(0,2*np.pi,3600)
for i in range(3):
    ax[i].set_theta_zero_location("N")
    ax[i].set_theta_direction(-1)
    ax[i].set_yticklabels([])
    ax[i].yaxis.grid(False)
    ax[i].plot(A,star_eq(A,*star_out),"k",lw=3)
    ax[i].plot(A,star_eq(A,*star_in),"k",lw=3)
    ax[i].tick_params(axis='x', which='major', labelsize=15)

normalize_d = mcolors.Normalize(vmin=0, vmax=max(analysis[:,3]))

val = 0
if -min(analysis[:,4])>max(analysis[:,4]):
    val = -min(analysis[:,4])
else:
    val = max(analysis[:,4])
normalize_r = mcolors.Normalize(vmin=-val, vmax=val)

cmap_d = cm.get_cmap('YlGnBu')
cmap_r = cm.get_cmap('coolwarm')


dp = analysis[0,0]

for line in analysis:
    x = np.array([line[0]-dp,line[0]+dp])
    y1 = star_eq(x,*star_out)
    y2 = star_eq(x,*star_in)
    
    ax[0].fill_between(x, y1, y2,color=cmap_d(normalize_d(line[3])))
    ax[1].fill_between(x, y1, y2,color=cmap_r(normalize_r(line[4])))
    
    if (line[1]<.05) and (line[2]<.05):
        ax[2].fill_between(x, y1, y2,color="firebrick")
    
sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=normalize_d)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax[0], orientation='horizontal')    
cbar.set_label('Mean Density', fontsize=20)
cbar.ax.tick_params(labelsize=15)


sm = plt.cm.ScalarMappable(cmap=cmap_r, norm=normalize_r)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax[1], orientation='horizontal')   
cbar.ax.tick_params(labelsize=15)
cbar.set_label('Mean Residuals', fontsize=20) 
 
cbar = fig.colorbar(sm, ax=ax[2], orientation='horizontal') 
cbar.set_label('Significant angles', fontsize=20)       
cbar.ax.tick_params(labelsize=15)

plt.tight_layout()


# plt.savefig("Figures/Star_regions_T1W1_etal_bycorr.png", dpi=200)
# plt.savefig("Figures/Star_regions_T1W1_etal_bycorr.svg")
