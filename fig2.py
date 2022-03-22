#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 2: Matrices of relative transfer of information tau / correlation coefficient R between sea-ice area and its potential drivers - SSP5-8.5
Fig. S3: Matrices of relative transfer of information tau / correlation coefficient R between sea-ice area and its potential drivers - SSP1-1.9
Fig. S4: Matrices of relative transfer of information tau / correlation coefficient R between sea-ice volume and its potential drivers - SSP5-8.5
Fig. S5: Matrices of relative transfer of information tau / correlation coefficient R between sea-ice volume and its potential drivers - SSP1-1.9

Compute Liang index on the SMHI-LENS experiments (Wyser et al., 2021)
Multiple variables (Liang, 2021)

Computations for each member (time series = years)

Last updated: 12/01/2022

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import combine_pvalues # for combining p values (Fisher test)
import seaborn as sns # for creating a matrix plot
from matplotlib.patches import Rectangle # for drawing rectangles around elements in a matrix

# Import my functions
sys.path.append('/home/sm_davdo/Projects/scripts/ROADMAP/')
from function_liang_nvar import compute_liang_nvar

# Working directories
dir_input = '/nobackup/rossby24/proj/rossby/joint_exp/oseaice/SMHI-LENS/'
dir_fig = '/nobackup/rossby24/proj/rossby/joint_exp/oseaice/SMHI-LENS/'

# Parameters
save_fig = True
scenario = 'ssp585' # ssp585; ssp119
use_sia = True # True: SIA; False: SIV
nvar = 7 # number of variables (1: SIA, 2: T2m, 3: SST, 4: Arctic OHT, 5: OHT 70N, 6: AHT 70N, 7: AOI)
n_members = 50 # number of ensemble members
nmy = 12 # number of months in a year
dt = 1 # time step (years)
n_iter = 1000 # number of repetitions for the bootstrapping
conf = 1.96 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
last_year_mar = 2080 # last year for March
last_year_sep = 2040 # last year for September

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if (var-conf*error < 0. and var+conf*error < 0.) or (var-conf*error > 0. and var+conf*error > 0.):
        sig = 1
    else:
        sig = 0
    z = var / error # z score
    pval = np.exp(-0.717 * z - 0.416 * z**2.) # p value for 95% confidence interval (https://www.bmj.com/content/343/bmj.d2304)
    return sig,pval

# Compute error based on p value
def compute_ci(pval,var):
    z = -0.862 + np.sqrt(0.743 - 2.404 * np.log(pval)) # z score (https://www.bmj.com/content/343/bmj.d2090)
    se = var / z # standard error
    return se

# Load monthly mean Arctic sea-ice area / volume from SMHI-LENS (1970-2014)
if use_sia == True:
    filename = dir_input + 'SIA_SMHI-LENS_historical.npy'
else:
    filename = dir_input + 'SIV_SMHI-LENS_historical.npy'
sia_hist = np.load(filename,allow_pickle=True)
nyears_hist = np.size(sia_hist,1)

# Load monthly mean Arctic sea-ice area / volume from SMHI-LENS (2015-2100)
if use_sia == True:
    filename = dir_input + 'SIA_SMHI-LENS_' + str(scenario) + '.npy'
else:
    filename = dir_input + 'SIV_SMHI-LENS_' + str(scenario) + '.npy'
sia_ssp585 = np.load(filename,allow_pickle=True)
nyears_ssp585 = np.size(sia_ssp585,1)
    
# Load monthly mean near-surface temperature from SMHI-LENS (1970-2014)
filename = dir_input + 'tas_Arctic_SMHI-LENS_historical.npy' # Arctic mean
tas_hist = np.load(filename,allow_pickle=True)

# Load monthly mean near-surface temperature from SMHI-LENS (2015-2100)
filename = dir_input + 'tas_Arctic_SMHI-LENS_' + str(scenario) + '.npy' # Arctic mean
tas_ssp585 = np.load(filename,allow_pickle=True)

# Load monthly mean SST from SMHI-LENS (1970-2014)
filename = dir_input + 'sst_Arctic_SMHI-LENS_historical.npy' # Arctic mean
sst_hist = np.load(filename,allow_pickle=True)

# Load monthly mean SST from SMHI-LENS (2015-2100)
filename = dir_input + 'sst_Arctic_SMHI-LENS_' + str(scenario) + '.npy' # Arctic mean
sst_ssp585 = np.load(filename,allow_pickle=True)

# Load monthly mean Arctic OHT from SMHI-LENS (1970-2014)
filename = dir_input + 'OHT_Total_Omon_EC-Earth3_historical_gn_1970-2014.npy'
oht_arctic_hist = np.load(filename,allow_pickle=True)

# Load monthly mean Arctic OHT from SMHI-LENS (2015-2100)
filename = dir_input + 'OHT_Total_Omon_EC-Earth3_' + str(scenario) + '_gn_2015-2100.npy'
oht_arctic_ssp585 = np.load(filename,allow_pickle=True)

# Load annual mean heat transports at 70N from SMHI-LENS (1970-2014)
filename = dir_input + 'ht70_SMHI-LENS_historical.npy'
notused,aht_70N_hist,oht_70N_hist = np.load(filename,allow_pickle=True)

# Load annual mean heat transports at 70N from SMHI-LENS (2015-2100)
filename = dir_input + 'ht70_SMHI-LENS_' + str(scenario) + '.npy'
notused,aht_70N_ssp585,oht_70N_ssp585 = np.load(filename,allow_pickle=True)

# Load AO index from SMHI-LENS (1970-2014)
filename = dir_input + 'AOI_SMHI-LENS_historical.npy'
aoi_hist = np.load(filename,allow_pickle=True)

# Load AO index from SMHI-LENS (2015-2100)
filename = dir_input + 'AOI_SMHI-LENS_' + str(scenario) + '.npy'
aoi_ssp585 = np.load(filename,allow_pickle=True)

# Concatenate historical and future periods (1970-2100)
nyears = nyears_hist + nyears_ssp585
sia = np.zeros((n_members,nyears,nmy))
tas = np.zeros((n_members,nyears,nmy))
sst = np.zeros((n_members,nyears,nmy))
oht_arctic = np.zeros((n_members,nyears,nmy))
oht_70N = np.zeros((n_members,nyears))
aht_70N = np.zeros((n_members,nyears))
aoi = np.zeros((n_members,nyears,nmy))
for m in np.arange(n_members):
    for i in np.arange(nmy):
        sia[m,:,i] = np.concatenate((sia_hist[m,:,i],sia_ssp585[m,:,i]))
        tas[m,:,i] = np.concatenate((tas_hist[m,:,i],tas_ssp585[m,:,i]))
        sst[m,:,i] = np.concatenate((sst_hist[m,:,i],sst_ssp585[m,:,i]))
        oht_arctic[m,:,i] = np.concatenate((oht_arctic_hist[m,:,i],oht_arctic_ssp585[m,:,i]))
        aoi[m,:,i] = np.concatenate((aoi_hist[m,:,i],aoi_ssp585[m,:,i]))
    oht_70N[m,:] = np.concatenate((oht_70N_hist[m,:],oht_70N_ssp585[m,:]))
    aht_70N[m,:] = np.concatenate((aht_70N_hist[m,:],aht_70N_ssp585[m,:]))

# Compute annual mean T2m, SST, OHT_Arctic
tas_annmean = np.zeros((n_members,nyears))
sst_annmean = np.zeros((n_members,nyears))
oht_arctic_annmean = np.zeros((n_members,nyears))
for m in np.arange(n_members):
    tas_annmean[m,:] = np.nanmean(tas[m,:,:],axis=1)
    sst_annmean[m,:] = np.nanmean(sst[m,:,:],axis=1)
    oht_arctic_annmean[m,:] = np.nanmean(oht_arctic[m,:,:],axis=1)
    
# Take SIA/SIV of the month
sia_mar = sia[:,:,2] # March SIA/SIV
sia_sep = sia[:,:,8] # September SIA/SIV

# Compute mean JFM AO index
aoi_jfm = np.zeros((n_members,nyears))
for m in np.arange(n_members):
    aoi_jfm[m,:] = np.nanmean(aoi[m,:,0:3],axis=1)

# Compute ensemble mean
sia_mar_ensmean = np.nanmean(sia_mar,axis=0)
sia_sep_ensmean = np.nanmean(sia_sep,axis=0)
tas_ensmean = np.nanmean(tas_annmean,axis=0)
sst_ensmean = np.nanmean(sst_annmean,axis=0)
oht_arctic_ensmean = np.nanmean(oht_arctic_annmean,axis=0)
oht_70N_ensmean = np.nanmean(oht_70N,axis=0)
aht_70N_ensmean = np.nanmean(aht_70N,axis=0)
aoi_ensmean = np.nanmean(aoi_jfm,axis=0)

# Take years of interest and save variables - March
ind_last_year_mar = int(last_year_mar-1970+1)
nyears_mar = ind_last_year_mar
sia_mar2 = np.zeros((n_members,nyears_mar))
tas_mar = np.zeros((n_members,nyears_mar))
sst_mar = np.zeros((n_members,nyears_mar))
oht_arctic_mar = np.zeros((n_members,nyears_mar))
oht_70N_mar = np.zeros((n_members,nyears_mar))
aht_70N_mar = np.zeros((n_members,nyears_mar))
aoi_mar = np.zeros((n_members,nyears_mar))
for m in np.arange(n_members):
    sia_mar2[m,:] = sia_mar[m,0:ind_last_year_mar]
    tas_mar[m,:] = tas_annmean[m,0:ind_last_year_mar]
    sst_mar[m,:] = sst_annmean[m,0:ind_last_year_mar]
    oht_arctic_mar[m,:] = oht_arctic_annmean[m,0:ind_last_year_mar]
    oht_70N_mar[m,:] = oht_70N[m,0:ind_last_year_mar]
    aht_70N_mar[m,:] = aht_70N[m,0:ind_last_year_mar]
    aoi_mar[m,:] = aoi_jfm[m,0:ind_last_year_mar]

# Take years of interest and save variables - September
ind_last_year_sep = int(last_year_sep-1970+1)
nyears_sep = ind_last_year_sep
sia_sep2 = np.zeros((n_members,nyears_sep))
tas_sep = np.zeros((n_members,nyears_sep))
sst_sep = np.zeros((n_members,nyears_sep))
oht_arctic_sep = np.zeros((n_members,nyears_sep))
oht_70N_sep = np.zeros((n_members,nyears_sep))
aht_70N_sep = np.zeros((n_members,nyears_sep))
aoi_sep = np.zeros((n_members,nyears_sep))
for m in np.arange(n_members):
    sia_sep2[m,:] = sia_sep[m,0:ind_last_year_sep]
    tas_sep[m,:] = tas_annmean[m,0:ind_last_year_sep]
    sst_sep[m,:] = sst_annmean[m,0:ind_last_year_sep]
    oht_arctic_sep[m,:] = oht_arctic_annmean[m,0:ind_last_year_sep]
    oht_70N_sep[m,:] = oht_70N[m,0:ind_last_year_sep]
    aht_70N_sep[m,:] = aht_70N[m,0:ind_last_year_sep]
    aoi_sep[m,:] = aoi_jfm[m,0:ind_last_year_sep]
    
# Detrend data
for m in np.arange(n_members):
    sia_mar2[m,:] = sia_mar2[m,:] - sia_mar_ensmean[0:ind_last_year_mar]
    tas_mar[m,:] = tas_mar[m,:] - tas_ensmean[0:ind_last_year_mar]
    sst_mar[m,:] = sst_mar[m,:] - sst_ensmean[0:ind_last_year_mar]
    oht_arctic_mar[m,:] = oht_arctic_mar[m,:] - oht_arctic_ensmean[0:ind_last_year_mar]
    oht_70N_mar[m,:] = oht_70N_mar[m,:] - oht_70N_ensmean[0:ind_last_year_mar]
    aht_70N_mar[m,:] = aht_70N_mar[m,:] - aht_70N_ensmean[0:ind_last_year_mar]
    aoi_mar[m,:] = aoi_mar[m,:] - aoi_ensmean[0:ind_last_year_mar]
    sia_sep2[m,:] = sia_sep2[m,:] - sia_sep_ensmean[0:ind_last_year_sep]
    tas_sep[m,:] = tas_sep[m,:] - tas_ensmean[0:ind_last_year_sep]
    sst_sep[m,:] = sst_sep[m,:] - sst_ensmean[0:ind_last_year_sep]
    oht_arctic_sep[m,:] = oht_arctic_sep[m,:] - oht_arctic_ensmean[0:ind_last_year_sep]
    oht_70N_sep[m,:] = oht_70N_sep[m,:] - oht_70N_ensmean[0:ind_last_year_sep]
    aht_70N_sep[m,:] = aht_70N_sep[m,:] - aht_70N_ensmean[0:ind_last_year_sep]
    aoi_sep[m,:] = aoi_sep[m,:] - aoi_ensmean[0:ind_last_year_sep]
    
# Compute absolute and relative transfers of information (T and tau) and correlation coefficient (R) and their errors using function_liang_nvar
T_mar = np.zeros((n_members,nvar,nvar))
tau_mar = np.zeros((n_members,nvar,nvar))
R_mar = np.zeros((n_members,nvar,nvar))
error_T_mar = np.zeros((n_members,nvar,nvar))
error_tau_mar = np.zeros((n_members,nvar,nvar))
error_R_mar = np.zeros((n_members,nvar,nvar))
T_sep = np.zeros((n_members,nvar,nvar))
tau_sep = np.zeros((n_members,nvar,nvar))
R_sep = np.zeros((n_members,nvar,nvar))
error_T_sep = np.zeros((n_members,nvar,nvar))
error_tau_sep = np.zeros((n_members,nvar,nvar))
error_R_sep = np.zeros((n_members,nvar,nvar))
for t in np.arange(n_members):
    print(t)
    xx_mar = np.array((sia_mar2[t,:],tas_mar[t,:],sst_mar[t,:],oht_arctic_mar[t,:],oht_70N_mar[t,:],aht_70N_mar[t,:],aoi_mar[t,:]))
    T_mar[t,:,:],tau_mar[t,:,:],R_mar[t,:,:],error_T_mar[t,:,:],error_tau_mar[t,:,:],error_R_mar[t,:,:] = compute_liang_nvar(xx_mar,dt,n_iter)
    xx_sep = np.array((sia_sep2[t,:],tas_sep[t,:],sst_sep[t,:],oht_arctic_sep[t,:],oht_70N_sep[t,:],aht_70N_sep[t,:],aoi_sep[t,:]))
    T_sep[t,:,:],tau_sep[t,:,:],R_sep[t,:,:],error_T_sep[t,:,:],error_tau_sep[t,:,:],error_R_sep[t,:,:] = compute_liang_nvar(xx_sep,dt,n_iter)
        
# Compute significance of different members (different from 0) based on the confidence interval
sig_T_mar = np.zeros((n_members,nvar,nvar))
sig_tau_mar = np.zeros((n_members,nvar,nvar))
sig_R_mar = np.zeros((n_members,nvar,nvar))
pval_T_mar = np.zeros((n_members,nvar,nvar))
pval_tau_mar = np.zeros((n_members,nvar,nvar))
pval_R_mar = np.zeros((n_members,nvar,nvar))
sig_T_sep = np.zeros((n_members,nvar,nvar))
sig_tau_sep = np.zeros((n_members,nvar,nvar))
sig_R_sep = np.zeros((n_members,nvar,nvar))
pval_T_sep = np.zeros((n_members,nvar,nvar))
pval_tau_sep = np.zeros((n_members,nvar,nvar))
pval_R_sep = np.zeros((n_members,nvar,nvar))
for t in np.arange(n_members):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            sig_T_mar[t,i,j],pval_T_mar[t,i,j] = compute_sig(T_mar[t,i,j],error_T_mar[t,i,j],conf)
            sig_tau_mar[t,i,j],pval_tau_mar[t,i,j] = compute_sig(tau_mar[t,i,j],error_tau_mar[t,i,j],conf)
            sig_R_mar[t,i,j],pval_R_mar[t,i,j] = compute_sig(R_mar[t,i,j],error_R_mar[t,i,j],conf)
            sig_T_sep[t,i,j],pval_T_sep[t,i,j] = compute_sig(T_sep[t,i,j],error_T_sep[t,i,j],conf)
            sig_tau_sep[t,i,j],pval_tau_sep[t,i,j] = compute_sig(tau_sep[t,i,j],error_tau_sep[t,i,j],conf)
            sig_R_sep[t,i,j],pval_R_sep[t,i,j] = compute_sig(R_sep[t,i,j],error_R_sep[t,i,j],conf)

# Count number of significant members
count_T_mar = np.count_nonzero(sig_T_mar==1,axis=0)
count_tau_mar = np.count_nonzero(sig_tau_mar==1,axis=0)
count_R_mar = np.count_nonzero(sig_R_mar==1,axis=0)
count_T_sep = np.count_nonzero(sig_T_sep==1,axis=0)
count_tau_sep = np.count_nonzero(sig_tau_sep==1,axis=0)
count_R_sep = np.count_nonzero(sig_R_sep==1,axis=0)

# Combine p-values (Fisher test) of different members
pval_T_fisher_mar = np.zeros((nvar,nvar))
pval_tau_fisher_mar = np.zeros((nvar,nvar))
pval_R_fisher_mar = np.zeros((nvar,nvar))
pval_T_fisher_sep = np.zeros((nvar,nvar))
pval_tau_fisher_sep = np.zeros((nvar,nvar))
pval_R_fisher_sep = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        pval_T_fisher_mar[i,j] = combine_pvalues(pval_T_mar[:,i,j],method='fisher')[1]
        pval_tau_fisher_mar[i,j] = combine_pvalues(pval_tau_mar[:,i,j],method='fisher')[1]
        pval_R_fisher_mar[i,j] = combine_pvalues(pval_R_mar[:,i,j],method='fisher')[1]
        pval_T_fisher_sep[i,j] = combine_pvalues(pval_T_sep[:,i,j],method='fisher')[1]
        pval_tau_fisher_sep[i,j] = combine_pvalues(pval_tau_sep[:,i,j],method='fisher')[1]
        pval_R_fisher_sep[i,j] = combine_pvalues(pval_R_sep[:,i,j],method='fisher')[1]

# Compute ensemble mean T, tau and R
T_ensmean_mar = np.nanmean(T_mar,axis=0)
tau_ensmean_mar = np.nanmean(tau_mar,axis=0)
R_ensmean_mar = np.nanmean(R_mar,axis=0)
T_ensmean_sep = np.nanmean(T_sep,axis=0)
tau_ensmean_sep = np.nanmean(tau_sep,axis=0)
R_ensmean_sep = np.nanmean(R_sep,axis=0)

# Labels
if use_sia == True:
    label_names_mar = ['MSIA','T$_{2m}$','SST','OHT$_{A}$','OHT$_{70N}$','AHT$_{70N}$','AOI']
    label_names_sep = ['SSIA','T$_{2m}$','SST','OHT$_{A}$','OHT$_{70N}$','AHT$_{70N}$','AOI']
else:
    label_names_mar = ['MSIV','T$_{2m}$','SST','OHT$_{A}$','OHT$_{70N}$','AHT$_{70N}$','AOI']
    label_names_sep = ['SSIV','T$_{2m}$','SST','OHT$_{A}$','OHT$_{70N}$','AHT$_{70N}$','AOI'] 


# Plot options
fig,ax = plt.subplots(2,2,figsize=(24,26))
fig.subplots_adjust(left=0.05,bottom=0.01,right=0.95,top=0.92,wspace=0.15,hspace=0.15)
cmap_tau = plt.cm.YlOrRd._resample(15)
cmap_R = plt.cm.bwr._resample(16)
sns.set(font_scale=1.8)

# Matrix of tau (ensemble mean) - March
tau_annotations = np.round(np.abs(tau_ensmean_mar),2)
tau_plot = sns.heatmap(np.abs(tau_ensmean_mar),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=70,
    xticklabels=label_names_mar,yticklabels=label_names_mar,linewidths=0.1,linecolor='gray',ax=ax[0,0])
#tau_annotations = tau_ensmean_mar.astype(int)
#tau_plot = sns.heatmap(tau_ensmean_mar,annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_R,
#    cbar_kws={'label':r'$\tau$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=-70,vmax=70,
#    xticklabels=label_names_mar,yticklabels=label_names_mar,linewidths=0.1,linecolor='gray',ax=ax[0,0])
tau_plot.set_title(r'Relative transfer of information $\|\tau\|$ - March' + '\n',fontsize=24)
#tau_plot.set_title(r'Relative transfer of information $\tau$ - March' + '\n',fontsize=24)
tau_plot.set_title('a \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if pval_tau_fisher_mar[j,i] < 0.05:
            tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)
                   
# Matrix of R (ensemble mean) - March
R_annotations = np.round(R_ensmean_mar,2)
R_plot = sns.heatmap(R_ensmean_mar,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names_mar,yticklabels=label_names_mar,linewidths=0.1,linecolor='gray',ax=ax[0,1])
R_plot.set_title('Correlation coefficient $R$ - March \n ',fontsize=24)
R_plot.set_title('b \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if pval_R_fisher_mar[j,i] < 0.05 and j != i: 
            R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=20)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=20)

# Matrix of tau (ensemble mean) - September
tau_annotations = np.round(np.abs(tau_ensmean_sep),2)
tau_plot = sns.heatmap(np.abs(tau_ensmean_sep),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=70,
    xticklabels=label_names_sep,yticklabels=label_names_sep,linewidths=0.1,linecolor='gray',ax=ax[1,0])
#tau_annotations = tau_ensmean_sep.astype(int)
#tau_plot = sns.heatmap(tau_ensmean_sep,annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_R,
#    cbar_kws={'label':r'$\tau$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=-70,vmax=70,
#    xticklabels=label_names_sep,yticklabels=label_names_sep,linewidths=0.1,linecolor='gray',ax=ax[1,0])
tau_plot.set_title(r'Relative transfer of information $\|\tau\|$ - September' + '\n',fontsize=24)
#tau_plot.set_title(r'Relative transfer of information $\tau$ - September' + '\n',fontsize=24)
tau_plot.set_title('c \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if pval_tau_fisher_sep[j,i] < 0.05:
            tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)
                   
# Matrix of R (ensemble mean) - September
R_annotations = np.round(R_ensmean_sep,2)
R_plot = sns.heatmap(R_ensmean_sep,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names_sep,yticklabels=label_names_sep,linewidths=0.1,linecolor='gray',ax=ax[1,1])
R_plot.set_title('Correlation coefficient $R$ - September \n ',fontsize=24)
R_plot.set_title('d \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if pval_R_fisher_sep[j,i] < 0.05 and j != i: 
            R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=20)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=20)

# Save figure
if save_fig == True:
    if use_sia == True:
        if scenario == 'ssp585':
            fig.savefig(dir_fig + 'fig2.jpg',dpi=300)
        elif scenario == 'ssp119':
            fig.savefig(dir_fig + 'figS3.jpg',dpi=300)
    elif use_sia == False:
        if scenario == 'ssp585':
            fig.savefig(dir_fig + 'figS4.jpg',dpi=300)
        elif scenario == 'ssp119':
            fig.savefig(dir_fig + 'figS5.jpg',dpi=300)
