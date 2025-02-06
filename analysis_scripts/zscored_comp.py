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

MTT_trials = pd.read_csv("Data/MTT_results_240823.csv")
times = pd.read_csv("Data/times.csv")
times = times.rename(columns={"Code": "ID"})
core = pd.read_csv("Data/core_s2c.csv",sep=",")
core = core.rename(columns={"subject_id": "Subject","wave_code": "Wave"})
counts = pd.read_csv("Data/Counts_trials.csv")

MTT = pd.merge(MTT_trials, counts, on=['ID', 'Subject', 'Wave', 'Trial'])
MTT_time = pd.merge(MTT, times, on=['ID', 'Subject', 'Wave', 'Trial'])

MTT_all = pd.merge(MTT_time,core,on=["Subject","Wave"])

sel = MTT_all[['ID', 'Subject', 'Wave', 'Trial',"visit_age","SumSquaredResiduals",
               "SumResiduals","MeanDensity",'Time_ms','Counts',"Version",'subject_sex',"Zygo"]]

sel.loc[:, 'Time_ms'] = sel['Time_ms'] / 1000
sel = sel.rename(columns={"visit_age": "Age (years)",
                          "SumSquaredResiduals": "Residuals",                          
                          "SumResiduals": "Residuals_mean",
                          "MeanDensity": "Density",
                          "Time_ms": "Time (s)",
                          'Counts': "Times_outside",
                          "subject_sex":"Sex"})

sel = sel.dropna()
sel = sel[sel["Age (years)"]!=0]
sel = sel[sel["Time (s)"]!=0]

sel = sel[sel.Trial<7]

sel["Transformed_Residuals"] = boxcox(sel.Residuals)[0]
sel["Transformed_Density"] = boxcox(sel.Density)[0]
sel["Transformed_Time"] = boxcox(sel["Time (s)"]+.1)[0]
sel["Transformed_Errors"] = boxcox(sel.Times_outside+1)[0]


sel["Residuals (Z-scored)"] = scaler.fit_transform(np.array(sel["Transformed_Residuals"]).reshape(-1, 1))
sel["Residuals_mean (Z-scored)"] = scaler.fit_transform(np.array(sel["Residuals_mean"]).reshape(-1, 1))
sel["Density (Z-scored)"] = scaler.fit_transform(np.array(sel["Transformed_Density"]).reshape(-1, 1))
sel["Time (Z-scored)"] = scaler.fit_transform(np.array(sel["Transformed_Time"]).reshape(-1, 1))
sel["Errors (Z-scored)"] = scaler.fit_transform(np.array(sel["Transformed_Errors"]).reshape(-1, 1))

#%%Wave 1, Trial 1
sel2 = sel[sel.Wave==1][sel.Trial==1]

sel2 = sel2.dropna()
sel2 = sel2[sel2["Age (years)"]!=0]
sel2 = sel2[sel2["Time (s)"]!=0]




#%% Density-Residuals-Counts comparison by Wave 1 and 1st trial

fig, ax = plt.subplots(1,3,figsize=(20,6))
# sns.regplot(data=sel2, x= "Age (years)",y="Transformed_Time",ax=ax[0,0])
sns.regplot(data=sel2, x= "Errors (Z-scored)",y="Density (Z-scored)",ax=ax[0],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(sel2["Errors (Z-scored)"], sel2["Density (Z-scored)"])
print(r,p)
ax[0].set_title("PCC=%.3f" %(r),fontsize=25)
ax[0].set_xlabel("Errors (Z-scored)" ,fontsize=20)
ax[0].set_ylabel("Density (Z-scored)" ,fontsize=20)
ax[0].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=sel2, x= "Errors (Z-scored)",y="Residuals (Z-scored)",ax=ax[1],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(sel2["Errors (Z-scored)"], sel2["Residuals (Z-scored)"])
print(r,p)
ax[1].set_title("PCC=%.3f" %(r),fontsize=25)
ax[1].set_xlabel("Errors (Z-scored)" ,fontsize=20)
ax[1].set_ylabel("Residuals (Z-scored)" ,fontsize=20)
ax[1].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=sel2, x= "Density (Z-scored)",y="Residuals (Z-scored)",ax=ax[2],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(sel2["Density (Z-scored)"], sel2["Residuals (Z-scored)"])
print(r,p)
ax[2].set_title("PCC=%.3f" %(r),fontsize=25)
ax[2].set_ylabel("Residuals (Z-scored)" ,fontsize=20)
ax[2].set_xlabel("Density (Z-scored)" ,fontsize=20)
ax[2].tick_params(axis='both', which='major', labelsize=15)

fig.tight_layout()


#%% Split by sex

sexes = sel2['Sex'].unique()

fig, axes = plt.subplots(len(sexes), 3, figsize=(20, 6 * len(sexes)))


if len(sexes) == 1:
    axes = [axes]

for i, sex in enumerate(sexes):
    sel2_sex = sel2[sel2['Sex'] == sex]

    sns.regplot(data=sel2_sex, x="Errors (Z-scored)", y="Density (Z-scored)", ax=axes[i][0],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(sel2_sex["Errors (Z-scored)"], sel2_sex["Density (Z-scored)"])
    print(f"Sex: {sex}, PCC between Errors (Z-scored) and Density (Z-scored): {r}, p-value: {p}")
    axes[i][0].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][0].set_xlabel("Errors (Z-scored)", fontsize=20)
    axes[i][0].set_ylabel("Density (Z-scored)", fontsize=20)
    axes[i][0].tick_params(axis='both', which='major', labelsize=15)


    sns.regplot(data=sel2_sex, x="Errors (Z-scored)", y="Residuals (Z-scored)", ax=axes[i][1],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(sel2_sex["Errors (Z-scored)"], sel2_sex["Residuals (Z-scored)"])
    print(f"Sex: {sex}, PCC between Errors (Z-scored) and Residuals (Z-scored): {r}, p-value: {p}")
    axes[i][1].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][1].set_xlabel("Errors (Z-scored)", fontsize=20)
    axes[i][1].set_ylabel("Residuals (Z-scored)", fontsize=20)
    axes[i][1].tick_params(axis='both', which='major', labelsize=15)


    sns.regplot(data=sel2_sex, x="Density (Z-scored)", y="Residuals (Z-scored)", ax=axes[i][2],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(sel2_sex["Density (Z-scored)"], sel2_sex["Residuals (Z-scored)"])
    print(f"Sex: {sex}, PCC between Density (Z-scored) and Residuals (Z-scored): {r}, p-value: {p}")
    axes[i][2].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][2].set_xlabel("Density (Z-scored)", fontsize=20)
    axes[i][2].set_ylabel("Residuals (Z-scored)", fontsize=20)
    axes[i][2].tick_params(axis='both', which='major', labelsize=15)


fig.tight_layout()
plt.show()



#%% Time, Residuals, Density, Errors vs Age

fig, ax = plt.subplots(2,2,figsize=(16,12),sharex=True)
# sns.regplot(data=sel2, x= "Age (years)",y="Transformed_Time",ax=ax[0,0])
sns.regplot(data=sel2, x= "Age (years)",y="Time (Z-scored)",ax=ax[0,0],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="slateblue"))
r, p = sp.stats.pearsonr(sel2["Age (years)"], sel2["Time (Z-scored)"])
print(r,p)
ax[0,0].set_title("PCC=%.3f" %(r),fontsize=25)
ax[0,0].set_ylabel("Time (Z-scored)",fontsize=20)
ax[0,0].set_xlabel("")
ax[0,0].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=sel2, x= "Age (years)",y="Residuals (Z-scored)",ax=ax[0,1],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="forestgreen"))
r, p = sp.stats.pearsonr(sel2["Age (years)"], sel2["Residuals (Z-scored)"])
print(r,p)
ax[0,1].set_title("PCC=%.3f" %(r),fontsize=25)
ax[0,1].set_ylabel("Residuals (Z-scored)",fontsize=20)
ax[0,1].set_xlabel("")
ax[0,1].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=sel2, x= "Age (years)",y="Density (Z-scored)",ax=ax[1,0],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="forestgreen"))
r, p = sp.stats.pearsonr(sel2["Age (years)"], sel2["Density (Z-scored)"])
print(r,p)
ax[1,0].set_title("PCC=%.3f" %(r),fontsize=25)
ax[1,0].set_ylabel("Density (Z-scored)",fontsize=20)
ax[1,0].set_xlabel("Age (years)",fontsize=20)
ax[1,0].tick_params(axis='both', which='major', labelsize=15)

sns.regplot(data=sel2, x= "Age (years)",y="Errors (Z-scored)",ax=ax[1,1],
            scatter_kws={'color': "white", 'edgecolors': 'black'},line_kws=dict(color="firebrick"))
r, p = sp.stats.pearsonr(sel2["Age (years)"], sel2["Errors (Z-scored)"])
print(r,p)
ax[1,1].set_title("PCC=%.3f" %(r),fontsize=25)
ax[1,1].set_ylabel("Errors (Z-scored)",fontsize=20)
ax[1,1].set_xlabel("Age (years)",fontsize=20)
ax[1,1].tick_params(axis='both', which='major', labelsize=15)

fig.tight_layout()
# plt.savefig("Figures/comparison.png", dpi=200)
# plt.savefig("Figures/comparison.svg")

#%% Split by sex

sexes = sel2['Sex'].unique()

fig, axes = plt.subplots(len(sexes), 4, figsize=(16, 12 * len(sexes)), sharex=True)

if len(sexes) == 1:
    axes = [axes]

for i, sex in enumerate(sexes):
    sel2_sex = sel2[sel2['Sex'] == sex]

    sns.regplot(data=sel2_sex, x="Age (years)", y="Time (Z-scored)", ax=axes[i][0],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="slateblue"))
    r, p = sp.stats.pearsonr(sel2_sex["Age (years)"], sel2_sex["Time (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Time (Z-scored): {r}, p-value: {p}")
    axes[i][0].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][0].set_ylabel("Time (Z-scored)", fontsize=20)
    axes[i][0].set_xlabel("")
    axes[i][0].tick_params(axis='both', which='major', labelsize=15)

    sns.regplot(data=sel2_sex, x="Age (years)", y="Residuals (Z-scored)", ax=axes[i][1],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="forestgreen"))
    r, p = sp.stats.pearsonr(sel2_sex["Age (years)"], sel2_sex["Residuals (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Residuals (Z-scored): {r}, p-value: {p}")
    axes[i][1].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][1].set_ylabel("Residuals (Z-scored)", fontsize=20)
    axes[i][1].set_xlabel("")
    axes[i][1].tick_params(axis='both', which='major', labelsize=15)


    sns.regplot(data=sel2_sex, x="Age (years)", y="Density (Z-scored)", ax=axes[i][2],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="forestgreen"))
    r, p = sp.stats.pearsonr(sel2_sex["Age (years)"], sel2_sex["Density (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Density (Z-scored): {r}, p-value: {p}")
    axes[i][2].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][2].set_ylabel("Density (Z-scored)", fontsize=20)
    axes[i][2].set_xlabel("")
    axes[i][2].tick_params(axis='both', which='major', labelsize=15)
    axes[i][2].set_xlabel("Age (years)", fontsize=20)


    sns.regplot(data=sel2_sex, x="Age (years)", y="Errors (Z-scored)", ax=axes[i][3],
                scatter_kws={'color': "white", 'edgecolors': 'black'}, line_kws=dict(color="firebrick"))
    r, p = sp.stats.pearsonr(sel2_sex["Age (years)"], sel2_sex["Errors (Z-scored)"])
    print(f"Sex: {sex}, PCC between Age (years) and Errors (Z-scored): {r}, p-value: {p}")
    axes[i][3].set_title("PCC=%.3f" % (r), fontsize=25)
    axes[i][3].set_ylabel("Errors (Z-scored)", fontsize=20)
    axes[i][3].set_xlabel("Age (years)", fontsize=20)
    axes[i][3].tick_params(axis='both', which='major', labelsize=15)


fig.tight_layout()
plt.show()
