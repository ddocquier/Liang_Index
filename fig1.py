#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 1: Time series of original variables - SSP5-8.5
Fig. S1: Time series of original variables - SSP1-1.9
Fig. S2: Time series of detrended variables - SSP5-8.5

Last updated: 17/01/2022

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Working directories
dir_input = '/home/dadocq/Documents/SMHI-LENS/input/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/Liang_Index/LaTeX/figures/'

# Parameters
save_fig = True
nmy = 12 # number of months in a year
last_year = 2080 # last year for detrending March SIA
last_year2 = 2040 # last year for detrending September SIA
n_members = 50 # number of members
arctic = True # T2m and SST north of 70N (Arctic mean); False: global mean
scenario = 'ssp585' # ssp585; ssp119

# Load monthly Arctic sea-ice area from SMHI-LENS (1970-2014)
filename = dir_input + 'SIA_SMHI-LENS_historical.npy'
sia_hist = np.load(filename,allow_pickle=True)
nyears_hist = np.size(sia_hist,1)

# Load monthly Arctic sea-ice area from SMHI-LENS (2015-2100)
filename = dir_input + 'SIA_SMHI-LENS_' + str(scenario) + '.npy'
sia_ssp585 = np.load(filename,allow_pickle=True)
nyears_ssp585 = np.size(sia_ssp585,1)

# Load monthly Arctic sea-ice volume from SMHI-LENS (1970-2014)
filename = dir_input + 'SIV_SMHI-LENS_historical.npy'
siv_hist = np.load(filename,allow_pickle=True)

# Load monthly Arctic sea-ice volume from SMHI-LENS (2015-2100)
filename = dir_input + 'SIV_SMHI-LENS_' + str(scenario) + '.npy'
siv_ssp585 = np.load(filename,allow_pickle=True)
    
# Load monthly mean near-surface temperature from SMHI-LENS (1970-2014)
if arctic == True:
    filename = dir_input + 'tas_Arctic_SMHI-LENS_historical.npy' # Arctic mean
else:
    filename = dir_input + 'tas_SMHI-LENS_historical.npy' # global mean
tas_hist = np.load(filename,allow_pickle=True)

# Load monthly mean near-surface temperature from SMHI-LENS (2015-2100)
if arctic == True:
    filename = dir_input + 'tas_Arctic_SMHI-LENS_' + str(scenario) + '.npy' # Arctic mean
else:
    filename = dir_input + 'tas_SMHI-LENS_' + str(scenario) + '.npy' # global mean
tas_ssp585 = np.load(filename,allow_pickle=True)

# Load monthly mean SST from SMHI-LENS (1970-2014)
if arctic == True:
    filename = dir_input + 'sst_Arctic_SMHI-LENS_historical.npy' # Arctic mean
else:
    filename = dir_input + 'sst_SMHI-LENS_historical.npy' # global mean
sst_hist = np.load(filename,allow_pickle=True)

# Load monthly mean SST from SMHI-LENS (2015-2100)
if arctic == True:
    filename = dir_input + 'sst_Arctic_SMHI-LENS_' + str(scenario) + '.npy' # Arctic mean
else:
    filename = dir_input + 'sst_SMHI-LENS_' + str(scenario) + '.npy' # global mean
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
siv = np.zeros((n_members,nyears,nmy))
tas = np.zeros((n_members,nyears,nmy))
sst = np.zeros((n_members,nyears,nmy))
oht_arctic = np.zeros((n_members,nyears,nmy))
oht_70N = np.zeros((n_members,nyears))
aht_70N = np.zeros((n_members,nyears))
aoi = np.zeros((n_members,nyears,nmy))
for m in np.arange(n_members):
    for i in np.arange(nmy):
        sia[m,:,i] = np.concatenate((sia_hist[m,:,i],sia_ssp585[m,:,i]))
        siv[m,:,i] = np.concatenate((siv_hist[m,:,i],siv_ssp585[m,:,i]))
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
    
# March and September SIA
sia_mar = sia[:,:,2]
sia_sep = sia[:,:,8]

# March and September SIV
siv_mar = siv[:,:,2]
siv_sep = siv[:,:,8]

# Compute mean JFM AO index
aoi_jfm = np.zeros((n_members,nyears))
for m in np.arange(n_members):
    aoi_jfm[m,:] = np.nanmean(aoi[m,:,0:3],axis=1)
    
# Compute change between 2015 and 2100
print('March SIA:',np.nanmean(sia_mar[:,-1],axis=0)-np.nanmean(sia_mar[:,0],axis=0))
print('March SIA (%):',100*(np.nanmean(sia_mar[:,-1],axis=0)-np.nanmean(sia_mar[:,0],axis=0))/np.nanmean(sia_mar[:,0],axis=0))
print('September SIA:',np.nanmean(sia_sep[:,-1],axis=0)-np.nanmean(sia_sep[:,0],axis=0))
print('September SIA (%):',100*(np.nanmean(sia_sep[:,-1],axis=0)-np.nanmean(sia_sep[:,0],axis=0))/np.nanmean(sia_sep[:,0],axis=0)) 
print('March SIV:',np.nanmean(siv_mar[:,-1],axis=0)-np.nanmean(siv_mar[:,0],axis=0))
print('March SIV (%):',100*(np.nanmean(siv_mar[:,-1],axis=0)-np.nanmean(siv_mar[:,0],axis=0))/np.nanmean(siv_mar[:,0],axis=0))
print('September SIV:',np.nanmean(siv_sep[:,-1],axis=0)-np.nanmean(siv_sep[:,0],axis=0))
print('September SIV (%):',100*(np.nanmean(siv_sep[:,-1],axis=0)-np.nanmean(siv_sep[:,0],axis=0))/np.nanmean(siv_sep[:,0],axis=0)) 
print('T2m:',np.nanmean(tas_annmean[:,-1],axis=0)-np.nanmean(tas_annmean[:,0],axis=0))
print('T2m (%):',100*(np.nanmean(tas_annmean[:,-1],axis=0)-np.nanmean(tas_annmean[:,0],axis=0))/np.abs(np.nanmean(tas_annmean[:,0],axis=0)))
print('SST:',np.nanmean(sst_annmean[:,-1],axis=0)-np.nanmean(sst_annmean[:,0],axis=0))
print('SST (%):',100*(np.nanmean(sst_annmean[:,-1],axis=0)-np.nanmean(sst_annmean[:,0],axis=0))/np.abs(np.nanmean(sst_annmean[:,0],axis=0)))
print('OHT:',np.nanmean(oht_arctic_annmean[:,-1],axis=0)-np.nanmean(oht_arctic_annmean[:,0],axis=0))
print('OHT (%):',100*(np.nanmean(oht_arctic_annmean[:,-1],axis=0)-np.nanmean(oht_arctic_annmean[:,0],axis=0))/np.nanmean(oht_arctic_annmean[:,0],axis=0))
print('OHT 70N:',np.nanmean(oht_70N[:,-1],axis=0)-np.nanmean(oht_70N[:,0],axis=0))
print('OHT 70N (%):',100*(np.nanmean(oht_70N[:,-1],axis=0)-np.nanmean(oht_70N[:,0],axis=0))/np.nanmean(oht_70N[:,0],axis=0))
print('AHT 70N:',np.nanmean(aht_70N[:,-1],axis=0)-np.nanmean(aht_70N[:,0],axis=0))
print('AHT 70N (%):',100*(np.nanmean(aht_70N[:,-1],axis=0)-np.nanmean(aht_70N[:,0],axis=0))/np.nanmean(aht_70N[:,0],axis=0)) 
print('AOI:',np.nanmean(aoi_jfm[:,-1],axis=0)-np.nanmean(aoi_jfm[:,0],axis=0))
print('AOI (%):',100*(np.nanmean(aoi_jfm[:,-1],axis=0)-np.nanmean(aoi_jfm[:,0],axis=0))/np.abs(np.nanmean(aoi_jfm[:,0],axis=0)))


# Plot options
xrange = np.arange(11,141,20)
name_xticks = ['1980','2000','2020','2040','2060','2080','2100']

# Time series of original variables
fig,ax = plt.subplots(3,2,figsize=(24,24))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)

# March and September sea-ice area
#ax[0,0].plot(np.arange(nyears)+1,sia_sep[0,:],'r--',linewidth=0.5,label='March SIA')
for m in np.arange(n_members):
    ax[0,0].plot(np.arange(nyears)+1,sia_mar[m,:],'b--',linewidth=0.5)
    ax[0,0].plot(np.arange(nyears)+1,sia_sep[m,:],'r--',linewidth=0.5)
ax[0,0].plot(np.arange(nyears)+1,np.nanmean(sia_mar,axis=0),'-',color='darkblue',linewidth=4,label='March SIA')
ax[0,0].plot(np.arange(nyears)+1,np.nanmean(sia_sep,axis=0),'-',color='darkred',linewidth=4,label='September SIA')
ax[0,0].set_ylabel('Arctic sea-ice area (10$^6$ km$^2$)',fontsize=26)
ax[0,0].set_xticks(xrange)
ax[0,0].set_xticklabels(name_xticks)
ax[0,0].tick_params(labelsize=20)
ax[0,0].legend(loc='upper right',fontsize=24,shadow=True,frameon=False)
ax[0,0].grid(linestyle='--')
ax[0,0].axis([-1, 133, -1, 20])
ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# March and September sea-ice volume
for m in np.arange(n_members):
    ax[0,1].plot(np.arange(nyears)+1,siv_mar[m,:],'b--',linewidth=0.5)
    ax[0,1].plot(np.arange(nyears)+1,siv_sep[m,:],'r--',linewidth=0.5)
ax[0,1].plot(np.arange(nyears)+1,np.nanmean(siv_mar,axis=0),'-',color='darkblue',linewidth=4,label='March SIV')
ax[0,1].plot(np.arange(nyears)+1,np.nanmean(siv_sep,axis=0),'-',color='darkred',linewidth=4,label='September SIV')
ax[0,1].set_ylabel('Arctic sea-ice volume (10$^3$ km$^3$)',fontsize=26)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].tick_params(labelsize=20)
ax[0,1].legend(loc='upper right',fontsize=24,shadow=True,frameon=False)
ax[0,1].grid(linestyle='--')
ax[0,1].axis([-1, 133, -3, 65])
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# Near-surface and sea-surface temperature
for m in np.arange(n_members):
    ax[1,0].plot(np.arange(nyears)+1,sst_annmean[m,:],'r--',linewidth=0.5)
    ax[1,0].plot(np.arange(nyears)+1,tas_annmean[m,:],'b--',linewidth=0.5)
ax[1,0].plot(np.arange(nyears)+1,np.nanmean(sst_annmean,axis=0),'-',color='darkred',linewidth=4,label='SST')
ax[1,0].plot(np.arange(nyears)+1,np.nanmean(tas_annmean,axis=0),'-',color='darkblue',linewidth=4,label='T$_{2m}$')
if arctic == True:
    ax[1,0].set_ylabel('Arctic mean temperature ($^\circ$C)',fontsize=26)
else:
    ax[1,0].set_ylabel('Global mean temperature ($^\circ$C)',fontsize=26)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].tick_params(labelsize=20)
ax[1,0].legend(loc='upper left',fontsize=24,shadow=True,frameon=False)
ax[1,0].grid(linestyle='--')
if arctic == True:
    ax[1,0].axis([-1, 133, -20, 10])
else:
    ax[1,0].axis([-1, 133, 12, 24])
ax[1,0].set_title('c',loc='left',fontsize=30,fontweight='bold')

# Total Arctic OHT
for m in np.arange(n_members):
    ax[1,1].plot(np.arange(nyears)+1,oht_arctic_annmean[m,:],'r--',linewidth=0.5)
ax[1,1].plot(np.arange(nyears)+1,np.nanmean(oht_arctic_annmean,axis=0),'-',color='darkred',linewidth=4,label='OHT$_A$')
ax[1,1].set_ylabel('Total Arctic Ocean heat transport (TW)',fontsize=26)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].tick_params(labelsize=20)
ax[1,1].legend(loc='upper left',fontsize=24,shadow=True,frameon=False)
ax[1,1].grid(linestyle='--')
ax[1,1].axis([-1, 133, 25, 350])
ax[1,1].set_title('d',loc='left',fontsize=30,fontweight='bold')

# Ocean and atmospheric HT at 70N
for m in np.arange(n_members):
    ax[2,0].plot(np.arange(nyears)+1,aht_70N[m,:],'b--',linewidth=0.5)
    ax[2,0].plot(np.arange(nyears)+1,oht_70N[m,:],'r--',linewidth=0.5)
ax[2,0].plot(np.arange(nyears)+1,np.nanmean(aht_70N,axis=0),'-',color='darkblue',linewidth=4,label='AHT$_{70N}$')
ax[2,0].plot(np.arange(nyears)+1,np.nanmean(oht_70N,axis=0),'-',color='darkred',linewidth=4,label='OHT$_{70N}$')
ax[2,0].set_ylabel('Heat transport at 70$^{\circ}$N (PW)',fontsize=26)
ax[2,0].set_xlabel('Year',fontsize=26)
ax[2,0].set_xticks(xrange)
ax[2,0].set_xticklabels(name_xticks)
ax[2,0].tick_params(labelsize=20)
ax[2,0].legend(loc='center left',fontsize=24,shadow=True,frameon=False)
ax[2,0].grid(linestyle='--')
ax[2,0].axis([-1, 133, 0, 1.8])
ax[2,0].set_title('e',loc='left',fontsize=30,fontweight='bold')

# Arctic Oscillation index
for m in np.arange(n_members):
    ax[2,1].plot(np.arange(nyears)+1,aoi_jfm[m,:],'r--',linewidth=0.5)
ax[2,1].plot(np.arange(nyears)+1,np.nanmean(aoi_jfm,axis=0),'-',color='darkred',linewidth=4,label='AOI')
ax[2,1].set_ylabel('Arctic Oscillation index (JFM)',fontsize=26)
ax[2,1].set_xlabel('Year',fontsize=26)
ax[2,1].set_xticks(xrange)
ax[2,1].set_xticklabels(name_xticks)
ax[2,1].tick_params(labelsize=20)
ax[2,1].legend(loc='upper left',fontsize=24,shadow=True,frameon=False)
ax[2,1].grid(linestyle='--')
ax[2,1].axis([-1, 133, -4.5, 4.5])
ax[2,1].set_title('f',loc='left',fontsize=30,fontweight='bold')

# Save Fig.
if save_fig == True:
    if scenario == 'ssp585':
        fig.savefig(dir_fig + 'fig1.jpg',dpi=300)
    elif scenario == 'ssp119':
       fig.savefig(dir_fig + 'figS1.jpg',dpi=300) 

# Take years of interest - March SIA/SIV
ind_last_year = int(last_year-1970+1)
nyears2 = ind_last_year
sia_mar2 = np.zeros((n_members,nyears2))
siv_mar2 = np.zeros((n_members,nyears2))
tas2 = np.zeros((n_members,nyears2))
sst2 = np.zeros((n_members,nyears2))
oht_arctic2 = np.zeros((n_members,nyears2))
oht_70N2 = np.zeros((n_members,nyears2))
aht_70N2 = np.zeros((n_members,nyears2))
aoi2 = np.zeros((n_members,nyears2))
for m in np.arange(n_members):
    sia_mar2[m,:] = sia_mar[m,0:ind_last_year]
    siv_mar2[m,:] = siv_mar[m,0:ind_last_year]
    tas2[m,:] = tas_annmean[m,0:ind_last_year]
    sst2[m,:] = sst_annmean[m,0:ind_last_year]
    oht_arctic2[m,:] = oht_arctic_annmean[m,0:ind_last_year]
    oht_70N2[m,:] = oht_70N[m,0:ind_last_year]
    aht_70N2[m,:] = aht_70N[m,0:ind_last_year]
    aoi2[m,:] = aoi_jfm[m,0:ind_last_year]
    
# Take years of interest - September SIA/SIV
ind_last_year2 = int(last_year2-1970+1)
nyears3 = ind_last_year2
sia_sep2 = np.zeros((n_members,nyears3))
siv_sep2 = np.zeros((n_members,nyears3))
for m in np.arange(n_members):
    sia_sep2[m,:] = sia_sep[m,0:ind_last_year2]
    siv_sep2[m,:] = siv_sep[m,0:ind_last_year2]
   
# Compute ensemble mean    
sia_mar_ensmean = np.nanmean(sia_mar2,axis=0)
sia_sep_ensmean = np.nanmean(sia_sep2,axis=0)
siv_mar_ensmean = np.nanmean(siv_mar2,axis=0)
siv_sep_ensmean = np.nanmean(siv_sep2,axis=0)
tas_ensmean = np.nanmean(tas2,axis=0)
sst_ensmean = np.nanmean(sst2,axis=0)
oht_arctic_ensmean = np.nanmean(oht_arctic2,axis=0)
oht_70N_ensmean = np.nanmean(oht_70N2,axis=0)
aht_70N_ensmean = np.nanmean(aht_70N2,axis=0)
aoi_ensmean = np.nanmean(aoi2,axis=0)
    
# Detrend data - Remove ensemble mean
for m in np.arange(n_members):
    sia_mar2[m,:] = sia_mar2[m,:] - sia_mar_ensmean
    sia_sep2[m,:] = sia_sep2[m,:] - sia_sep_ensmean
    siv_mar2[m,:] = siv_mar2[m,:] - siv_mar_ensmean
    siv_sep2[m,:] = siv_sep2[m,:] - siv_sep_ensmean
    tas2[m,:] = tas2[m,:] - tas_ensmean
    sst2[m,:] = sst2[m,:] - sst_ensmean
    oht_arctic2[m,:] = oht_arctic2[m,:] - oht_arctic_ensmean
    oht_70N2[m,:] = oht_70N2[m,:] - oht_70N_ensmean
    aht_70N2[m,:] = aht_70N2[m,:] - aht_70N_ensmean
    aoi2[m,:] = aoi2[m,:] - aoi_ensmean
            
# Time series of detrended variables
fig,ax = plt.subplots(3,2,figsize=(24,24))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)

# March and September sea-ice area
for m in np.arange(n_members):
    ax[0,0].plot(np.arange(nyears2)+1,sia_mar2[m,:],'b--',linewidth=0.5)
    ax[0,0].plot(np.arange(nyears3)+1,sia_sep2[m,:],'r--',linewidth=0.5)
ax[0,0].plot(np.arange(nyears2)+1,np.nanmean(sia_mar2,axis=0),'b-',linewidth=4,label='March SIA')
ax[0,0].plot(np.arange(nyears3)+1,np.nanmean(sia_sep2,axis=0),'r-',linewidth=4,label='September SIA')
ax[0,0].set_ylabel('Det. Arctic sea-ice area (10$^6$ km$^2$)',fontsize=26)
ax[0,0].set_xticks(xrange)
ax[0,0].set_xticklabels(name_xticks)
ax[0,0].tick_params(labelsize=20)
ax[0,0].legend(loc='upper right',fontsize=24,shadow=True,frameon=False)
ax[0,0].grid(linestyle='--')
ax[0,0].axis([-1, 113, -3.5, 3.5])
ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# March and September sea-ice volume
for m in np.arange(n_members):
    ax[0,1].plot(np.arange(nyears2)+1,siv_mar2[m,:],'b--',linewidth=0.5)
    ax[0,1].plot(np.arange(nyears3)+1,siv_sep2[m,:],'r--',linewidth=0.5)
ax[0,1].plot(np.arange(nyears2)+1,np.nanmean(siv_mar2,axis=0),'b-',linewidth=4,label='March SIV')
ax[0,1].plot(np.arange(nyears3)+1,np.nanmean(siv_sep2,axis=0),'r-',linewidth=4,label='September SIV')
ax[0,1].set_ylabel('Det. Arctic sea-ice volume (10$^3$ km$^3$)',fontsize=26)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].tick_params(labelsize=20)
ax[0,1].legend(loc='upper right',fontsize=24,shadow=True,frameon=False)
ax[0,1].grid(linestyle='--')
ax[0,1].axis([-1, 113, -20, 20])
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# Near-surface and sea-surface temperature
for m in np.arange(n_members):
    ax[1,0].plot(np.arange(nyears2)+1,sst2[m,:],'r--',linewidth=0.5)
    ax[1,0].plot(np.arange(nyears2)+1,tas2[m,:],'b--',linewidth=0.5)
ax[1,0].plot(np.arange(nyears2)+1,np.nanmean(sst2,axis=0),'r-',linewidth=4,label='SST')
ax[1,0].plot(np.arange(nyears2)+1,np.nanmean(tas2,axis=0),'b-',linewidth=4,label='T$_{2m}$')
if arctic == True:
    ax[1,0].set_ylabel('Det. Arctic mean temperature ($^\circ$C)',fontsize=26)
else:
    ax[1,0].set_ylabel('Det. global mean temperature ($^\circ$C)',fontsize=26)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].tick_params(labelsize=20)
ax[1,0].legend(loc='upper left',fontsize=24,shadow=True,frameon=False)
ax[1,0].grid(linestyle='--')
if arctic == True:
    ax[1,0].axis([-1, 113, -3.5, 3.5])
else:
    ax[1,0].axis([-1, 113, -0.7, 0.7])
ax[1,0].set_title('c',loc='left',fontsize=30,fontweight='bold')

# Total Arctic OHT
for m in np.arange(n_members):
    ax[1,1].plot(np.arange(nyears2)+1,oht_arctic2[m,:],'r--',linewidth=0.5)
ax[1,1].plot(np.arange(nyears2)+1,np.nanmean(oht_arctic2,axis=0),'r-',linewidth=4,label='OHT$_A$')
ax[1,1].set_ylabel('Det. Arctic Ocean heat transport (TW)',fontsize=26)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].tick_params(labelsize=20)
ax[1,1].legend(loc='upper left',fontsize=24,shadow=True,frameon=False)
ax[1,1].grid(linestyle='--')
ax[1,1].axis([-1, 113, -100, 100])
ax[1,1].set_title('d',loc='left',fontsize=30,fontweight='bold')

# Ocean and atmospheric HT at 70N
for m in np.arange(n_members):
    ax[2,0].plot(np.arange(nyears2)+1,aht_70N2[m,:],'b--',linewidth=0.5)
    ax[2,0].plot(np.arange(nyears2)+1,oht_70N2[m,:],'r--',linewidth=0.5)
ax[2,0].plot(np.arange(nyears2)+1,np.nanmean(aht_70N2,axis=0),'-',linewidth=4,label='AHT$_{70N}$')
ax[2,0].plot(np.arange(nyears2)+1,np.nanmean(oht_70N2,axis=0),'r-',linewidth=4,label='OHT$_{70N}$')
ax[2,0].set_ylabel('Det. heat transport at 70$^{\circ}$N (PW)',fontsize=26)
ax[2,0].set_xlabel('Year',fontsize=26)
ax[2,0].set_xticks(xrange)
ax[2,0].set_xticklabels(name_xticks)
ax[2,0].tick_params(labelsize=20)
ax[2,0].legend(loc='upper left',fontsize=24,shadow=True,frameon=False)
ax[2,0].grid(linestyle='--')
ax[2,0].axis([-1, 113, -0.2, 0.2])
ax[2,0].set_title('e',loc='left',fontsize=30,fontweight='bold')

# Arctic Oscillation index
for m in np.arange(n_members):
    ax[2,1].plot(np.arange(nyears2)+1,aoi2[m,:],'r--',linewidth=0.5)
ax[2,1].plot(np.arange(nyears2)+1,np.nanmean(aoi2,axis=0),'r-',linewidth=4,label='AOI')
ax[2,1].set_ylabel('Det. Arctic Oscillation index (JFM)',fontsize=26)
ax[2,1].set_xlabel('Year',fontsize=26)
ax[2,1].set_xticks(xrange)
ax[2,1].set_xticklabels(name_xticks)
ax[2,1].tick_params(labelsize=20)
ax[2,1].legend(loc='upper left',fontsize=24,shadow=True,frameon=False)
ax[2,1].grid(linestyle='--')
ax[2,1].axis([-1, 113, -4.5, 4.5])
ax[2,1].set_title('f',loc='left',fontsize=30,fontweight='bold')

# Save Fig.
if save_fig == True:
    if scenario == 'ssp585':
        fig.savefig(dir_fig + 'figS2.jpg',dpi=300)
