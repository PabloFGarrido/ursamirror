#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:54:07 2024

@author: Pablo F. Garrido
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox
import scipy as sp
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

plt.close("all")

metadata = pd.read_csv("../simulated_data/simulated_metadata.csv",sep=",")

sum_squared_residuals = []
mean_density = []

for index, row in metadata.iterrows():
    file_path = f"../simulated_data/{row['ID']}.csv" 
    data = pd.read_csv(file_path)
    
    sum_squared_residuals.append(data.Residuals_sqrd.sum())
    mean_density.append(data.Density.mean())

data_participants = metadata.rename(columns={"age": "Age (years)",
                          "time": "Time (s)",
                          'counts_out': "Times_outside",
                          "sex":"Sex"})

data_participants['Residuals'] = sum_squared_residuals
data_participants['Density'] = mean_density


data_participants["Transformed_Residuals"] = boxcox(data_participants.Residuals)[0]
data_participants["Transformed_Density"] = boxcox(data_participants.Density)[0]
data_participants["Transformed_Time"] = boxcox(data_participants["Time (s)"]+.1)[0]
data_participants["Transformed_Errors"] = boxcox(data_participants.Times_outside+1)[0]


data_participants["Residuals (Z-scored)"] = scaler.fit_transform(np.array(data_participants["Transformed_Residuals"]).reshape(-1, 1))
data_participants["Density (Z-scored)"] = scaler.fit_transform(np.array(data_participants["Transformed_Density"]).reshape(-1, 1))
data_participants["Time (Z-scored)"] = scaler.fit_transform(np.array(data_participants["Transformed_Time"]).reshape(-1, 1))
data_participants["Errors (Z-scored)"] = scaler.fit_transform(np.array(data_participants["Transformed_Errors"]).reshape(-1, 1))



#%% Density-Residuals-Counts comparison by Wave 1 and 1st trial

fig, ax = plt.subplots(1,3,figsize=(20,6))
# sns.regplot(data=data_participants, x= "Age (years)",y="Transformed_Time",ax=ax[0,0])
sns.regplot(data=data_participants, x= "Errors (Z-scored)",y="Density (Z-scored)",ax=ax[0],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(data_participants["Errors (Z-scored)"], data_participants["Density (Z-scored)"])
print(r,p)
ax[0].set_title("PCC=%.3f" %(r),fontsize=25)
ax[0].set_xlabel("Errors (Z-scored)" ,fontsize=20)
ax[0].set_ylabel("Density (Z-scored)" ,fontsize=20)
ax[0].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=data_participants, x= "Errors (Z-scored)",y="Residuals (Z-scored)",ax=ax[1],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(data_participants["Errors (Z-scored)"], data_participants["Residuals (Z-scored)"])
print(r,p)
ax[1].set_title("PCC=%.3f" %(r),fontsize=25)
ax[1].set_xlabel("Errors (Z-scored)" ,fontsize=20)
ax[1].set_ylabel("Residuals (Z-scored)" ,fontsize=20)
ax[1].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=data_participants, x= "Density (Z-scored)",y="Residuals (Z-scored)",ax=ax[2],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(data_participants["Density (Z-scored)"], data_participants["Residuals (Z-scored)"])
print(r,p)
ax[2].set_title("PCC=%.3f" %(r),fontsize=25)
ax[2].set_ylabel("Residuals (Z-scored)" ,fontsize=20)
ax[2].set_xlabel("Density (Z-scored)" ,fontsize=20)
ax[2].tick_params(axis='both', which='major', labelsize=15)

fig.tight_layout()


#%% Split by sex

sexes = data_participants['Sex'].unique()

fig, axes = plt.subplots(len(sexes), 3, figsize=(20, 6 * len(sexes)))


if len(sexes) == 1:
    axes = [axes]

for i, sex in enumerate(sexes):
    data_participants_sex = data_participants[data_participants['Sex'] == sex]

    sns.regplot(data=data_participants_sex, x="Errors (Z-scored)", y="Density (Z-scored)", ax=axes[i][0],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(data_participants_sex["Errors (Z-scored)"], data_participants_sex["Density (Z-scored)"])
    print(f"Sex: {sex}, PCC between Errors (Z-scored) and Density (Z-scored): {r}, p-value: {p}")
    axes[i][0].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][0].set_xlabel("Errors (Z-scored)", fontsize=20)
    axes[i][0].set_ylabel("Density (Z-scored)", fontsize=20)
    axes[i][0].tick_params(axis='both', which='major', labelsize=15)


    sns.regplot(data=data_participants_sex, x="Errors (Z-scored)", y="Residuals (Z-scored)", ax=axes[i][1],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(data_participants_sex["Errors (Z-scored)"], data_participants_sex["Residuals (Z-scored)"])
    print(f"Sex: {sex}, PCC between Errors (Z-scored) and Residuals (Z-scored): {r}, p-value: {p}")
    axes[i][1].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][1].set_xlabel("Errors (Z-scored)", fontsize=20)
    axes[i][1].set_ylabel("Residuals (Z-scored)", fontsize=20)
    axes[i][1].tick_params(axis='both', which='major', labelsize=15)


    sns.regplot(data=data_participants_sex, x="Density (Z-scored)", y="Residuals (Z-scored)", ax=axes[i][2],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(data_participants_sex["Density (Z-scored)"], data_participants_sex["Residuals (Z-scored)"])
    print(f"Sex: {sex}, PCC between Density (Z-scored) and Residuals (Z-scored): {r}, p-value: {p}")
    axes[i][2].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][2].set_xlabel("Density (Z-scored)", fontsize=20)
    axes[i][2].set_ylabel("Residuals (Z-scored)", fontsize=20)
    axes[i][2].tick_params(axis='both', which='major', labelsize=15)


fig.tight_layout()
plt.show()



#%% Time, Residuals, Density, Errors vs Age

fig, ax = plt.subplots(2,2,figsize=(16,12),sharex=True)
# sns.regplot(data=data_participants, x= "Age (years)",y="Transformed_Time",ax=ax[0,0])
sns.regplot(data=data_participants, x= "Age (years)",y="Time (Z-scored)",ax=ax[0,0],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="slateblue"))
r, p = sp.stats.pearsonr(data_participants["Age (years)"], data_participants["Time (Z-scored)"])
print(r,p)
ax[0,0].set_title("PCC=%.3f" %(r),fontsize=25)
ax[0,0].set_ylabel("Time (Z-scored)",fontsize=20)
ax[0,0].set_xlabel("")
ax[0,0].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=data_participants, x= "Age (years)",y="Residuals (Z-scored)",ax=ax[0,1],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="forestgreen"))
r, p = sp.stats.pearsonr(data_participants["Age (years)"], data_participants["Residuals (Z-scored)"])
print(r,p)
ax[0,1].set_title("PCC=%.3f" %(r),fontsize=25)
ax[0,1].set_ylabel("Residuals (Z-scored)",fontsize=20)
ax[0,1].set_xlabel("")
ax[0,1].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=data_participants, x= "Age (years)",y="Density (Z-scored)",ax=ax[1,0],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="forestgreen"))
r, p = sp.stats.pearsonr(data_participants["Age (years)"], data_participants["Density (Z-scored)"])
print(r,p)
ax[1,0].set_title("PCC=%.3f" %(r),fontsize=25)
ax[1,0].set_ylabel("Density (Z-scored)",fontsize=20)
ax[1,0].set_xlabel("Age (years)",fontsize=20)
ax[1,0].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=data_participants, x= "Age (years)",y="Errors (Z-scored)",ax=ax[1,1],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(data_participants["Age (years)"], data_participants["Errors (Z-scored)"])
print(r,p)
ax[1,1].set_title("PCC=%.3f" %(r),fontsize=25)
ax[1,1].set_ylabel("Errors (Z-scored)",fontsize=20)
ax[1,1].set_xlabel("Age (years)",fontsize=20)
ax[1,1].tick_params(axis='both', which='major', labelsize=15)

fig.tight_layout()
# plt.savefig("Figures/comparison.png", dpi=200)
# plt.savefig("Figures/comparison.svg")

#%% Split by sex

sexes = data_participants['Sex'].unique()

fig, axes = plt.subplots(len(sexes), 4, figsize=(16, 12 * len(sexes)), sharex=True)

if len(sexes) == 1:
    axes = [axes]

for i, sex in enumerate(sexes):
    data_participants_sex = data_participants[data_participants['Sex'] == sex]

    sns.regplot(data=data_participants_sex, x="Age (years)", y="Time (Z-scored)", ax=axes[i][0],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="slateblue"))
    r, p = sp.stats.pearsonr(data_participants_sex["Age (years)"], data_participants_sex["Time (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Time (Z-scored): {r}, p-value: {p}")
    axes[i][0].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][0].set_ylabel("Time (Z-scored)", fontsize=20)
    axes[i][0].set_xlabel("")
    axes[i][0].tick_params(axis='both', which='major', labelsize=15)

    sns.regplot(data=data_participants_sex, x="Age (years)", y="Residuals (Z-scored)", ax=axes[i][1],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="forestgreen"))
    r, p = sp.stats.pearsonr(data_participants_sex["Age (years)"], data_participants_sex["Residuals (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Residuals (Z-scored): {r}, p-value: {p}")
    axes[i][1].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][1].set_ylabel("Residuals (Z-scored)", fontsize=20)
    axes[i][1].set_xlabel("")
    axes[i][1].tick_params(axis='both', which='major', labelsize=15)


    sns.regplot(data=data_participants_sex, x="Age (years)", y="Density (Z-scored)", ax=axes[i][2],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="forestgreen"))
    r, p = sp.stats.pearsonr(data_participants_sex["Age (years)"], data_participants_sex["Density (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Density (Z-scored): {r}, p-value: {p}")
    axes[i][2].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][2].set_ylabel("Density (Z-scored)", fontsize=20)
    axes[i][2].set_xlabel("")
    axes[i][2].tick_params(axis='both', which='major', labelsize=15)
    axes[i][2].set_xlabel("Age (years)", fontsize=20)


    sns.regplot(data=data_participants_sex, x="Age (years)", y="Errors (Z-scored)", ax=axes[i][3],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(data_participants_sex["Age (years)"], data_participants_sex["Errors (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Errors (Z-scored): {r}, p-value: {p}")
    axes[i][3].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][3].set_ylabel("Errors (Z-scored)", fontsize=20)
    axes[i][3].set_xlabel("Age (years)", fontsize=20)
    axes[i][3].tick_params(axis='both', which='major', labelsize=15)


fig.tight_layout()
plt.show()
