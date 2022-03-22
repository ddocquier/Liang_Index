#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 3: Time evolution of tau for March sea-ice area - OHT_Arctic - SSP5-8.5
Fig. S6: Time evolution of tau for September sea-ice area - OHT_Arctic - SSP5-8.5
Fig. S7: Time evolution of tau for March sea-ice area - OHT_Arctic - SSP1-1.9
Fig. S8: Time evolution of tau for September sea-ice area - OHT_Arctic - SSP1-1.9
Fig. 4: Time evolution of tau for March sea-ice volume - OHT_Arctic - SSP5-8.5
Fig. S9: Time evolution of tau for September sea-ice volume - OHT_Arctic - SSP5-8.5
Fig. S10: Time evolution of tau for March sea-ice volume - OHT_Arctic - SSP1-1.9
Fig. S11: Time evolution of tau for September sea-ice volume - OHT_Arctic - SSP1-1.9

Compute Liang index on the SMHI-LENS experiments (Wyser et al., 2021)
Multiple variables (Liang, 2021)

Computations for each member (time series = years) or for each range of years (time series = members)

Last updated: 17/01/2022

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

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
var = 1 # for plot option 2; 1: T2m, 2: SST, 3: Arctic OHT, 4: OHT 70N, 5: AHT 70N, 6: AOI
var2 = 3 # for plot option 2; 1: T2m, 2: SST, 3: Arctic OHT, 4: OHT 70N, 5: AHT 70N, 6: AOI
mon = 2 # month for SIA (2: March; 8: September)
n_members = 50 # number of ensemble members
range_years = 5 # range of years (for option = 2)
nmy = 12 # number of months in a year
dt = 1 # time step (years)
n_iter = 1000 # number of repetitions for the bootstrapping
conf = 1.96 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%

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
    
# Take SIA of the month
sia_mon = sia[:,:,mon]

# Compute mean JFM AO index
aoi_jfm = np.zeros((n_members,nyears))
for m in np.arange(n_members):
    aoi_jfm[m,:] = np.nanmean(aoi[m,:,0:3],axis=1)
    
# Compute absolute and relative transfers of information (T and tau) and correlation coefficient (R) and their errors using function_liang_nvar
nt = int(nyears / range_years)
T = np.zeros((nt,nvar,nvar))
tau = np.zeros((nt,nvar,nvar))
R = np.zeros((nt,nvar,nvar))
error_T = np.zeros((nt,nvar,nvar))
error_tau = np.zeros((nt,nvar,nvar))
error_R = np.zeros((nt,nvar,nvar))
for t in np.arange(nt):
    print(t)
    if t == (nt - 1):
        sia2 = sia_mon[:,t*range_years::]
        tas2 = tas_annmean[:,t*range_years::]
        sst2 = sst_annmean[:,t*range_years::]
        oht_arctic2 = oht_arctic_annmean[:,t*range_years::]
        oht_70N2 = oht_70N[:,t*range_years::]
        aht_70N2 = aht_70N[:,t*range_years::]
        aoi2 = aoi_jfm[:,t*range_years::]
    else:
        sia2 = sia_mon[:,t*range_years:t*range_years+range_years]
        tas2 = tas_annmean[:,t*range_years:t*range_years+range_years]
        sst2 = sst_annmean[:,t*range_years:t*range_years+range_years]
        oht_arctic2 = oht_arctic_annmean[:,t*range_years:t*range_years+range_years]
        oht_70N2 = oht_70N[:,t*range_years:t*range_years+range_years]
        aht_70N2 = aht_70N[:,t*range_years:t*range_years+range_years]
        aoi2 = aoi_jfm[:,t*range_years:t*range_years+range_years]
    n = np.size(sia2)
    sia3 = np.reshape(sia2,n)
    tas3 = np.reshape(tas2,n)
    sst3 = np.reshape(sst2,n)
    oht_arctic3 = np.reshape(oht_arctic2,n)
    oht_70N3 = np.reshape(oht_70N2,n)
    aht_70N3 = np.reshape(aht_70N2,n)
    aoi3 = np.reshape(aoi2,n)
    
    xx = np.array((sia3,tas3,sst3,oht_arctic3,oht_70N3,aht_70N3,aoi3))
    T[t,:,:],tau[t,:,:],R[t,:,:],error_T[t,:,:],error_tau[t,:,:],error_R[t,:,:] = compute_liang_nvar(xx,dt,n_iter) 
    

# Plot options
index = np.arange(nt)
bar_width = 1

# Labels
if range_years == 5:
    xrange = np.arange(2,27,4)
    name_xticks = ['1975-1979','1995-1999','2015-2019','2035-2039','2055-2059','2075-2079','2095-2100']
elif range_years == 4:
    xrange = np.arange(2,33,4)
    name_xticks = ['1974-1977','1990-1993','2006-2009','2022-2025','2038-2041','2054-2057','2070-2073','2086-2089']

# Figure
fig,ax = plt.subplots(2,1,figsize=(12,10))
fig.subplots_adjust(left=0.1,bottom=0.1,right=0.85,top=0.95,hspace=0.3)
ax2 = ax[0].twinx()
ax3 = ax[1].twinx()

# Label panel a
if mon == 2:
    if use_sia == True:
        if var == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},MSIA}$'
        elif var == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,MSIA}$'
        elif var == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,MSIA}$'
        elif var == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},MSIA}$'
        elif var == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},MSIA}$'
        elif var == 6:
            var_label1 = r'$\|\tau_{AOI \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,MSIA}$'
    else:
        if var == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},MSIV}$'
        elif var == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,MSIV}$'
        elif var == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,MSIV}$'
        elif var == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},MSIV}$'
        elif var == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},MSIV}$'
        elif var == 6:
            var_label1 = r'$\|\tau_{AOI \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,MSIV}$'
elif mon == 8:
    if use_sia == True:
        if var == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},SSIA}$'
        elif var == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,SSIA}$'
        elif var == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,SSIA}$'
        elif var == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},SSIA}$'
        elif var == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},SSIA}$'
        elif var == 6:
            var_label1 = r'$\tau_{AOI \longrightarrow SSIA}\|$'
            var_label2 = r'$\tau_{SSIA \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,SSIA}$'
    else:
        if var == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},SSIV}$'
        elif var == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,SSIV}$'
        elif var == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,SSIV}$'
        elif var == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},SSIV}$'
        elif var == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},SSIV}$'
        elif var == 6:
            var_label1 = r'$\|\tau_{AOI \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,SSIV}$'
    
# X2 --> X1 - var1
ax[0].errorbar(index[0]+1,np.abs(tau[0,var,0]),yerr=conf*error_tau[0,var,0],fmt='ro',markersize=8,label=var_label1)
for i in np.arange(np.size(index)-1):
    ax[0].errorbar(index[i+1]+1,np.abs(tau[i+1,var,0]),yerr=conf*error_tau[i+1,var,0],fmt='ro',markersize=8)

# X1 --> X2 - var1
ax[0].errorbar(index[0]+1.2,np.abs(tau[0,0,var]),yerr=conf*error_tau[0,0,var],fmt='bo',markersize=8,label=var_label2)
for i in np.arange(np.size(index)-1):
    ax[0].errorbar(index[i+1]+1.2,np.abs(tau[i+1,0,var]),yerr=conf*error_tau[i+1,0,var],fmt='bo',markersize=8)
    
## X2 --> X1 - var1
#ax[0].errorbar(index[0]+1,tau[0,var,0],yerr=conf*error_tau[0,var,0],fmt='ro',markersize=8,label=var_label1)
#for i in np.arange(np.size(index)-1):
#    ax[0].errorbar(index[i+1]+1,tau[i+1,var,0],yerr=conf*error_tau[i+1,var,0],fmt='ro',markersize=8)
#
## X1 --> X2 - var1
#ax[0].errorbar(index[0]+1.2,tau[0,0,var],yerr=conf*error_tau[0,0,var],fmt='bo',markersize=8,label=var_label2)
#for i in np.arange(np.size(index)-1):
#    ax[0].errorbar(index[i+1]+1.2,tau[i+1,0,var],yerr=conf*error_tau[i+1,0,var],fmt='bo',markersize=8)

# R - var1
ax2.plot(index[0]+1,R[0,var,0],'kx',markersize=8,label=var_label3)
for i in np.arange(np.size(index)-1):
    ax2.plot(index[i+1]+1,R[i+1,var,0],'kx',markersize=8)

# Labels and legend - var1
ax[0].set_ylabel('Transfer of information ($\%$)',fontsize=20)
ax2.set_ylabel('Correlation coefficient',fontsize=20)
ax[0].tick_params(axis='both',labelsize=14)
ax2.tick_params(axis='both',labelsize=14)
#ax.axhline(c='k')
ax[0].set_xlabel('Time period',fontsize=20)
if scenario == 'ssp585' and mon == 8:
    ax[0].legend(loc='upper right',fontsize=14,shadow=True,frameon=False)
    ax2.legend(loc='lower right',fontsize=14,shadow=True,frameon=False)
else:
    ax[0].legend(loc='upper left',fontsize=14,shadow=True,frameon=False)
    ax2.legend(loc='upper center',fontsize=14,shadow=True,frameon=False)
ax[0].set_xticks(xrange)
ax2.set_xticks(xrange)
ax[0].set_xticklabels(name_xticks)
ax[0].grid(linestyle='--')
ax[0].set_ylim(0,45)
#ax[0].set_ylim(-45,45)
ax2.set_ylim(-1,-0.15)
ax[0].set_title('a',loc='left',fontsize=25,fontweight='bold')

# Label panel b
if mon == 2:
    if use_sia == True:
        if var2 == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},MSIA}$'
        elif var2 == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,MSIA}$'
        elif var2 == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,MSIA}$'
        elif var2 == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},MSIA}$'
        elif var2 == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},MSIA}$'
        elif var2 == 6:
            var_label1 = r'$\|\tau_{AOI \longrightarrow MSIA}\|$'
            var_label2 = r'$\|\tau_{MSIA \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,MSIA}$'
    else:
        if var2 == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},MSIV}$'
        elif var2 == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,MSIV}$'
        elif var2 == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,MSIV}$'
        elif var2 == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},MSIV}$'
        elif var2 == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},MSIV}$'
        elif var2 == 6:
            var_label1 = r'$\|\tau_{AOI \longrightarrow MSIV}\|$'
            var_label2 = r'$\|\tau_{MSIV \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,MSIV}$'
elif mon == 8:
    if use_sia == True:
        if var2 == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},SSIA}$'
        elif var2 == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,SSIA}$'
        elif var2 == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,SSIA}$'
        elif var2 == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},SSIA}$'
        elif var2 == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow SSIA}\|$'
            var_label2 = r'$\|\tau_{SSIA \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},SSIA}$'
        elif var2 == 6:
            var_label1 = r'$\tau_{AOI \longrightarrow SSIA}\|$'
            var_label2 = r'$\tau_{SSIA \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,SSIA}$'
    else:
        if var2 == 1:
            var_label1 = r'$\|\tau_{T_{2m} \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow T_{2m}}\|$'
            var_label3 = '$R_{T_{2m},SSIV}$'
        elif var2 == 2:
            var_label1 = r'$\|\tau_{SST \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow SST}\|$'
            var_label3 = '$R_{SST,SSIV}$'
        elif var2 == 3:
            var_label1 = r'$\|\tau_{OHT_A \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow OHT_A}\|$'
            var_label3 = '$R_{OHT_A,SSIV}$'
        elif var2 == 4:
            var_label1 = r'$\|\tau_{OHT_{70N} \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow OHT_{70N}}\|$'
            var_label3 = '$R_{OHT_{70N},SSIV}$'
        elif var2 == 5:
            var_label1 = r'$\|\tau_{AHT_{70N} \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow AHT_{70N}}\|$'
            var_label3 = '$R_{AHT_{70N},SSIV}$'
        elif var2 == 6:
            var_label1 = r'$\|\tau_{AOI \longrightarrow SSIV}\|$'
            var_label2 = r'$\|\tau_{SSIV \longrightarrow AOI}\|$'
            var_label3 = '$R_{AOI,SSIV}$'

# X2 --> X1 - var2
ax[1].errorbar(index[0]+1,np.abs(tau[0,var2,0]),yerr=conf*error_tau[0,var2,0],fmt='ro',markersize=8,label=var_label1)
for i in np.arange(np.size(index)-1):
    ax[1].errorbar(index[i+1]+1,np.abs(tau[i+1,var2,0]),yerr=conf*error_tau[i+1,var2,0],fmt='ro',markersize=8)

# X1 --> X2 - var2
ax[1].errorbar(index[0]+1.2,np.abs(tau[0,0,var2]),yerr=conf*error_tau[0,0,var2],fmt='bo',markersize=8,label=var_label2)
for i in np.arange(np.size(index)-1):
    ax[1].errorbar(index[i+1]+1.2,np.abs(tau[i+1,0,var2]),yerr=conf*error_tau[i+1,0,var2],fmt='bo',markersize=8)
    
## X2 --> X1 - var2
#ax[1].errorbar(index[0]+1,tau[0,var2,0],yerr=conf*error_tau[0,var2,0],fmt='ro',markersize=8,label=var_label1)
#for i in np.arange(np.size(index)-1):
#    ax[1].errorbar(index[i+1]+1,tau[i+1,var2,0],yerr=conf*error_tau[i+1,var2,0],fmt='ro',markersize=8)
#
## X1 --> X2 - var2
#ax[1].errorbar(index[0]+1.2,tau[0,0,var2],yerr=conf*error_tau[0,0,var2],fmt='bo',markersize=8,label=var_label2)
#for i in np.arange(np.size(index)-1):
#    ax[1].errorbar(index[i+1]+1.2,tau[i+1,0,var2],yerr=conf*error_tau[i+1,0,var2],fmt='bo',markersize=8)

# R - var2
ax3.plot(index[0]+1,R[0,var2,0],'kx',markersize=8,label=var_label3)
for i in np.arange(np.size(index)-1):
    ax3.plot(index[i+1]+1,R[i+1,var2,0],'kx',markersize=8)

# Labels and legend - var2
ax[1].set_ylabel('Transfer of information ($\%$)',fontsize=20)
ax3.set_ylabel('Correlation coefficient',fontsize=20)
ax[1].tick_params(axis='both',labelsize=14)
ax3.tick_params(axis='both',labelsize=14)
#ax.axhline(c='k')
ax[1].set_xlabel('Time period',fontsize=20)
if scenario == 'ssp585' and mon == 8:
    ax[1].legend(loc='upper right',fontsize=14,shadow=True,frameon=False)
    ax3.legend(loc='lower right',fontsize=14,shadow=True,frameon=False)
else:
    ax[1].legend(loc='upper left',fontsize=14,shadow=True,frameon=False)
    ax3.legend(loc='upper center',fontsize=14,shadow=True,frameon=False)
ax[1].set_xticks(xrange)
ax3.set_xticks(xrange)
ax[1].set_xticklabels(name_xticks)
ax[1].grid(linestyle='--')
ax[1].set_ylim(0,45)
#ax[1].set_ylim(-45,45)
ax3.set_ylim(-1,-0.15)
ax[1].set_title('b',loc='left',fontsize=25,fontweight='bold')

# Save figure
if save_fig == True:
    if use_sia == True:
        if scenario == 'ssp585':
            if mon == 2:
                fig.savefig(dir_fig + 'fig3.jpg',dpi=300)
            elif mon == 8:
                fig.savefig(dir_fig + 'figS6.jpg',dpi=300)
        elif scenario == 'ssp119':
            if mon == 2:
                fig.savefig(dir_fig + 'figS7.jpg',dpi=300)
            elif mon == 8:
                fig.savefig(dir_fig + 'figS8.jpg',dpi=300)
    elif use_sia == False:
        if scenario == 'ssp585':
            if mon == 2:
                fig.savefig(dir_fig + 'fig4.jpg',dpi=300)
            elif mon == 8:
                fig.savefig(dir_fig + 'figS9.jpg',dpi=300)
        elif scenario == 'ssp119':
            if mon == 2:
                fig.savefig(dir_fig + 'figS10.jpg',dpi=300)
            elif mon == 8:
                fig.savefig(dir_fig + 'figS11.jpg',dpi=300)
