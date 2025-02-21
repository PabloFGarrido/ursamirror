#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:08:44 2024

@author: Pablo F. Garrido
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os as os
import shutil

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from ursamirror.utils import star_eq

star_out = np.array([ 1.2, -2.5,  1, 5])
star_in = np.array([ 1., -2.5,  1, 5])

#Using "star_eq" needs '+ np.deg2rad(36)' dephase to be North oriented when plotting

def star_eq_mod(phi, rho, m, k=1, n=5):
    return star_eq(phi+ np.deg2rad(36), rho, m, k=1, n=5)


#%%

LIST = []
for participant in os.listdir("../simulated_data/"):
    if "sim_participant" in participant:
        LIST.append("../simulated_data/"+participant)

            
DATA = []

for file in LIST:
    code = file.split("/")[-1].split("_analysis")[0]
    aux_df = pd.read_csv(file)[["Density","Residuals"]]
    aux_df = aux_df.fillna(0)
    DATA.append(aux_df)

NAMES = np.zeros(len(LIST)).astype(str)
for i,f in enumerate(LIST):
    NAMES[i] = f.split("/")[-1].split("_analysis")[0]
    
DATA = np.array(DATA)
DATA_scaled = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(DATA)

#%% Time Series Clustering-Model init
from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging,euclidean_barycenter

n_clusters = 3
n_jobs = 12

model_dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw",n_jobs=n_jobs)

model_euc = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean",n_jobs=n_jobs)

model_cdtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw",
                              metric_params={"global_constraint":"sakoe_chiba", 
                                             "sakoe_chiba_radius":18},n_jobs=n_jobs)


model_dtw.fit(DATA)
model_euc.fit(DATA)
model_cdtw.fit(DATA)

labels_dtw= model_dtw.labels_
labels_cdtw= model_cdtw.labels_
labels_euc= model_euc.labels_


#%% Cluster plotting
normalize_d = mcolors.Normalize(vmin=0, vmax=5)    
cmap_d = cm.get_cmap('YlGnBu')


fig, ax = plt.subplots(3,3,figsize=(24,24),subplot_kw={'projection': 'polar'})
H = np.linspace(0,2*np.pi,360,endpoint=True)  
aux_ang = 0.5*H[1:]+0.5*H[:-1]
aux_ang2 = np.linspace(0,2*np.pi,3600,endpoint=True) 

dp = aux_ang[0]

for cluster in range(3):
    #EUC
    ax[0,cluster].set_theta_zero_location("N")
    ax[0,cluster].set_theta_direction(-1)
    ax[0,cluster].set_yticklabels([])
    ax[0,cluster].yaxis.grid(False)
    ax[0,cluster].plot(aux_ang2,star_eq_mod(aux_ang2,*star_out),"k",lw=3)
    ax[0,cluster].plot(aux_ang2,star_eq_mod(aux_ang2,*star_in),"k",lw=3)
    ax[0,cluster].tick_params(axis='x', which='major', labelsize=15)
    
    data_subset = DATA[labels_euc==cluster]
    residuals = euclidean_barycenter(data_subset[:,:,1])
    residuals = residuals.transpose()[0]
    density = euclidean_barycenter(data_subset[:,:,0])
    density = density.transpose()[0]
    
    for i,ang in enumerate(aux_ang):
        x = np.array([ang+dp,ang-dp])
        y1 = star_eq_mod(x,*star_out)
        y2 = star_eq_mod(x,*star_in)
        
        ax[0,cluster].fill_between(x, y1, y2,color=cmap_d(normalize_d(density[i])))
    y_mean = (star_eq_mod(aux_ang,*star_out)*.5+star_eq_mod(aux_ang,*star_in)*.5)+residuals*(star_eq_mod(aux_ang,*star_out)-star_eq_mod(aux_ang,*star_in))
    ax[0,cluster].plot(aux_ang,y_mean,"firebrick",lw=2)
    
    #C-DTW
    ax[1,cluster].set_theta_zero_location("N")
    ax[1,cluster].set_theta_direction(-1)
    ax[1,cluster].set_yticklabels([])
    ax[1,cluster].yaxis.grid(False)
    ax[1,cluster].plot(aux_ang2,star_eq_mod(aux_ang2,*star_out),"k",lw=3)
    ax[1,cluster].plot(aux_ang2,star_eq_mod(aux_ang2,*star_in),"k",lw=3)
    ax[1,cluster].tick_params(axis='x', which='major', labelsize=15)
    
    data_subset = DATA[labels_cdtw==cluster]
    residuals = dtw_barycenter_averaging(data_subset[:,:,1],
                                         metric_params={"global_constraint":"sakoe_chiba", 
                                                        "sakoe_chiba_radius":18})
    residuals = residuals.transpose()[0]
    density = dtw_barycenter_averaging(data_subset[:,:,0],
                                       metric_params={"global_constraint":"sakoe_chiba", 
                                                      "sakoe_chiba_radius":18})
    density = density.transpose()[0]
    
    for i,ang in enumerate(aux_ang):
        x = np.array([ang+dp,ang-dp])
        y1 = star_eq_mod(x,*star_out)
        y2 = star_eq_mod(x,*star_in)
        
        ax[1,cluster].fill_between(x, y1, y2,color=cmap_d(normalize_d(density[i])))
    y_mean = (star_eq_mod(aux_ang,*star_out)*.5+star_eq_mod(aux_ang,*star_in)*.5)+residuals*(star_eq_mod(aux_ang,*star_out)-star_eq_mod(aux_ang,*star_in))
    ax[1,cluster].plot(aux_ang,y_mean,"firebrick",lw=2)
    
    
    #DTW
    ax[2,cluster].set_theta_zero_location("N")
    ax[2,cluster].set_theta_direction(-1)
    ax[2,cluster].set_yticklabels([])
    ax[2,cluster].yaxis.grid(False)
    ax[2,cluster].plot(aux_ang2,star_eq_mod(aux_ang2,*star_out),"k",lw=3)
    ax[2,cluster].plot(aux_ang2,star_eq_mod(aux_ang2,*star_in),"k",lw=3)
    ax[2,cluster].tick_params(axis='x', which='major', labelsize=15)
    
    data_subset = DATA[labels_dtw==cluster]
    residuals = dtw_barycenter_averaging(data_subset[:,:,1])
    residuals = residuals.transpose()[0]
    density = dtw_barycenter_averaging(data_subset[:,:,0])
    density = density.transpose()[0]
    
    for i,ang in enumerate(aux_ang):
        x = np.array([ang+dp,ang-dp])
        y1 = star_eq_mod(x,*star_out)
        y2 = star_eq_mod(x,*star_in)
        
        ax[2,cluster].fill_between(x, y1, y2,color=cmap_d(normalize_d(density[i])))
    y_mean = (star_eq_mod(aux_ang,*star_out)*.5+star_eq_mod(aux_ang,*star_in)*.5)+residuals*(star_eq_mod(aux_ang,*star_out)-star_eq_mod(aux_ang,*star_in))
    ax[2,cluster].plot(aux_ang,y_mean,"firebrick",lw=2)

fig.tight_layout()

# plt.savefig("Figures/Cluster_comp_2.png", dpi=200)
# plt.savefig("Figures/Cluster_comp_2.svg")


#%% Cluster demographics
IDs = np.array([x.replace(".csv","") for x in NAMES])
df_clust = pd.DataFrame(data=np.array([IDs,labels_euc,labels_cdtw,labels_dtw]).transpose(),
                        columns=["ID", "Ecuclidean Cluster", "c-DTW Cluster","DTW Cluster" ])
metadata = pd.read_csv("../simulated_data/simulated_metadata.csv",sep=",")
metadata = metadata.rename(columns={"age": "Age (years)", "sex": "Sex"})
df_merged = pd.merge(metadata, df_clust, on="ID")

fig, ax = plt.subplots(1,3,figsize=(24,12),sharey=True)
for i, label in enumerate(['Ecuclidean Cluster', 'c-DTW Cluster','DTW Cluster']):
    sns.violinplot(df_merged,x=label,y="Age (years)",hue="Sex", split=True,
                    ax=ax[i])
    ax[i].xaxis.label.set_size(15)
ax[0].yaxis.label.set_size(20)
plt.tight_layout()
# plt.savefig("Figures/Cluster_dem.png", dpi=200)
# plt.savefig("Figures/Cluster_dem.svg")

