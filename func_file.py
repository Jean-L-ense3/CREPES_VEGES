"""
Last update on December 2025

@author: jlittaye
Python 3.11.5
"""
import os
from os import listdir
import time
import datetime
import sys

import matplotlib.pyplot as plt
from math import *
import random
import numpy as np
import pandas as pd
import torch
import scipy.signal
from scipy.interpolate import RectBivariateSpline
from torch.utils.data import TensorDataset, DataLoader, Subset


def init_NNPZD(T0, Nz) :
    """ Return initial Nitrogen concentration vertical profile for NO3, NH4, P, Z and D
    - T0: Reference Nitrogen concentration over the water column/
    - Nz: Number of layers.
    """
    return torch.cat((T0*torch.ones([1, Nz])*1.0, T0*torch.ones([1, Nz])*.01, T0*torch.ones([1, Nz])*.01, T0*torch.ones([1, Nz])*.01, T0*torch.ones([1, Nz])*.01), dim = 0)


def linear_interpolate_1d(x, xp, fp):
    """ Linear interpolation for 1D data on GPU.
    - x: Target grid (interpolation points).
    - xp: Original grid.
    - fp: Values at original grid points.
    """
    fp = fp.moveaxis(len(fp.shape)-1, 0)
    indices = torch.searchsorted(xp, x).clamp(1, len(xp) - 1).cpu()
    x0, x1 = xp[indices - 1], xp[indices]
    f0, f1 = fp[indices - 1], fp[indices]    # Linear interpolation formula
    return f0.moveaxis(0, len(fp.shape)-1) + (f1 - f0).moveaxis(0, len(fp.shape)-1) * ((x - x0) / (x1 - x0))


def calcul_MLD(zrange, kz, dz=2.) :
    """ Calculate MLD as lowest depth where log10(Kz) > 2 m**2/d-1
    - zrange: vertical grid.
    - kz: diffusion coefficient.
    - dz: vertical resolution of MLD.
    """
    Kz_interpo_step = 10**linear_interpolate_1d(torch.arange(0, 350, dz), zrange, torch.log10(kz))
    MLD_step = torch.argmax(1*(torch.log10(Kz_interpo_step) < 2), dim = -1)*dz
    return MLD_step

def calcul_MLDmax(zrange, kz, dz = 2.) :
    return calcul_MLD(zrange, kz, dz).max(dim=-1).values
    

def Hovmoller_plot(I, Kz, NO3, NH4, P, Z, D, trange, zrange, dzT, ti_i = 0, ti_f = -1, zmax = None, vmin = None, vmax = None) :
    """ Hovmoller plot for NO3, NH4, P, Z and D as states, the vertical diffusion Kz nd irradiance Itz.
    - I: surface PAR.
    - Kz: vertical diffusion.
    - NO3, NH4, P, Z and D: states (nitrate, ammonium, phytoplankton, zooplankton, detritus).
    - trange: time space.
    - zrange: vertical space.
    - dzT: layer width for sotck calculation.
    """
    dt = trange[1]-trange[0]
    fig, ax = plt.subplots(figsize = [6.4*1.5*2, 4.8*0.7*4], nrows = 4, ncols = 2)
    dstart, dend = trange[ti_i], trange[ti_f]
    if zmax == None :
        zi_max = -1
    else :
        zi_max = np.argmin(zrange < zmax)
    plot_Itz = ax[0, 0].imshow(np.log10(I[:zi_max, ti_i:ti_f]), cmap = "jet", aspect = 'auto', extent = [round(dstart/365), round(dend/365), -zrange[zi_max], -zrange[0]], vmin = -8)
    plot_Kz = ax[0, 1].imshow(np.log10(Kz[:zi_max, ti_i:ti_f]/(3600*24)), cmap = "jet", aspect = 'auto', extent = [round(dstart/365), round(dend/365), -zrange[zi_max], -zrange[0]])
    plot_NO3 = ax[1, 0].imshow(NO3[:zi_max, ti_i:ti_f], cmap = "jet", aspect = 'auto', extent = [round(dstart/365), round(dend/365), -zrange[zi_max], -zrange[0]], vmin = vmin, vmax = vmax)
    plot_NH4 = ax[1, 1].imshow(NH4[:zi_max, ti_i:ti_f], cmap = "jet", aspect = 'auto', extent = [round(dstart/365), round(dend/365), -zrange[zi_max], -zrange[0]], vmin = vmin, vmax = vmax)
    plot_P = ax[2, 0].imshow(P[:zi_max, ti_i:ti_f], cmap = "jet", aspect = 'auto', extent = [round(dstart/365), round(dend/365), -zrange[zi_max], -zrange[0]], vmin = vmin, vmax = vmax)
    plot_Z = ax[2, 1].imshow(Z[:zi_max, ti_i:ti_f], cmap = "jet", aspect = 'auto', extent = [round(dstart/365), round(dend/365), -zrange[zi_max], -zrange[0]], vmin = vmin, vmax = vmax)
    plot_D = ax[3, 0].imshow(D[:zi_max, ti_i:ti_f], cmap = "jet", aspect = 'auto', extent = [round(dstart/365), round(dend/365), -zrange[zi_max], -zrange[0]], vmin = vmin, vmax = vmax)
   
    for t in range(ti_i, ti_f) :
        if trange[t]%365 == 30 or trange[t]%365 == 60 :
            for i in range(2, 7) :
                ax[i//2, i%2].axvline(x = trange[t]/365, linestyle = ':', color = 'white', lw = 2)

    fig.colorbar(plot_Itz, ax=ax[0, 0], location = 'right', label=r"log($W.m^{-2}$)")
    fig.colorbar(plot_Kz, ax=ax[0, 1], location = 'right', label=r"log($m^2.s^{-1}$)")
    fig.colorbar(plot_NO3, ax=ax[1, 0], location = 'right', label=r"$(mmol N).m^{-3}$")
    fig.colorbar(plot_NH4, ax=ax[1, 1], location = 'right', label=r"$(mmol N).m^{-3}$")
    fig.colorbar(plot_P, ax=ax[2, 0], location = 'right', label=r"$(mmol N).m^{-3}$")
    fig.colorbar(plot_Z, ax=ax[2, 1], location = 'right', label=r"$(mmol N).m^{-3}$")
    fig.colorbar(plot_D, ax=ax[3, 0], location = 'right', label=r"$(mmol N).m^{-3}$")

    for i in range(7) :
        ax[i//2, i%2].grid()
        ax[i//2, i%2].set_ylabel("Profondeur (m)")
        ax[i//2, i%2].set_xlabel("Temps (année)")
        ax[i//2, i%2].set_title([r"$I_0$", "Kz", "NO3", "NH4", "P", "Z", "D", "Verification"][i])
        if i >= 2 :
            ax[i//2, i%2].contour(trange[ti_i:ti_f]/365, -zrange[:zi_max], np.log10(Kz[:zi_max, ti_i:ti_f]), levels=[2.], colors='w', linestyles='-', alpha = 0.5)
    
    ax[3, 1].plot(trange[ti_i:ti_f], np.sum((NO3.T*dzT), axis = 1)[ti_i:ti_f]/10, label = 'NO3/10')
    ax[3, 1].plot(trange[ti_i:ti_f], np.sum((NH4.T*dzT), axis = 1)[ti_i:ti_f], label = "NH4")
    ax[3, 1].plot(trange[ti_i:ti_f], np.sum((P.T*dzT), axis = 1)[ti_i:ti_f], label = "P")
    ax[3, 1].plot(trange[ti_i:ti_f], np.sum((Z.T*dzT), axis = 1)[ti_i:ti_f], label = "Z")
    ax[3, 1].plot(trange[ti_i:ti_f], np.sum((D.T*dzT), axis = 1)[ti_i:ti_f], label = "D")
    ax[3, 1].grid()
    for t in trange[ti_i:ti_f] :
        if t%365 == 30 or t%365 == 60 :
            ax[3, 1].axvline(x = t, ls = '--', color = 'k', alpha = 0.5)
            ax[3, 1].axvline(x = t, ls = '--', color = 'k', alpha = 0.5)
    ax[3, 1].legend(loc = 'upper right')

    fig.tight_layout()
    return fig


# def find_next_min(serie, local_win_length = 1) :
#     for i_search in range(local_win_length, len(serie)-local_win_length) :
#         if serie[i_search] < torch.min(serie[i_search-local_win_length:i_search]) and serie[i_search] < torch.min(serie[i_search+1:i_search+local_win_length+1]) :
#             return i_search
#     print("No minimum")
#     return None


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def pointValue(x,y,power,smoothing,xv,yv,values):  
    """ Interpolates the value of a 2D variable with respect to surrounding values.
    - (x, y): location of the point to interpolate.
    - (xv, yv): surrounding values location.
    - power, smooting: hyperparameters to interpolation. 
    - values: value of the surrounding points.
    """
    nominator=0  
    denominator=0  
    for i in range(0,len(values)):  
        dist = sqrt((x-xv[i])*(x-xv[i])+(y-yv[i])*(y-yv[i])+smoothing*smoothing)
        #If the point is really close to one of the data points, return the data point value to avoid singularities  
        if(dist<0.0000000001):  
            return values[i]  
        nominator=nominator+(values[i]/pow(dist,power))  
        denominator=denominator+(1/pow(dist,power))  
    #Return NODATA if the denominator is zero  
    if denominator > 0:  
        value = nominator/denominator  
    else:  
        value = -9999  
    return value  


def side_pts(ptA, ptB, nb_pt = 22) :
    """ Divides the [ptA, ptB] segment into nb_pt points and returns their respective location"""
    return np.array([[ptA[0]*icut+ptB[0]*(1-icut) for icut in np.linspace(0., 1., nb_pt)], [ptA[1]*icut+ptB[1]*(1-icut) for icut in np.linspace(0., 1., nb_pt)]])



def select_forcing_gen(path, lat, lon, year, zw_grid, Kz_bank, case, lat_lon_F = None, nb_interpo = 4, clip = True, method = 'lin', alphaAR=0.7) :
    """ Generates two forcing (Kz, I0) profiles: a True one, which is interpolated at (lat/lon) location according to existing profiles in Kzbank, and a False profile which is interpolated at each time step at a location (lat+laglat/lon+laglon) where laglat/laglon are location uncertainties are generated according to a AR signal and a level of uncertainty.
    - path: path of the reference forcing profiles.
    - lat, lon: location of the true profile.
    - year: number of the studied year (7 years available, so 0-6).
    - zw_grid: vertical grid of the generated Kz profiles.
    - Kz_bank: already existing Kz profiles from the polgyr simulation.
    - case: uncertainty level (between 0 and 3) that defines how far the uncertain location can be from the true location.
    - lat_lon_F: if one wants to reuse already generated locations for the false profile.
    - nb_interpo: number of existing profiles to interpolate a value.
    - clip: while defining uncertain position (lat+laglat/lon+laglon), it is possible to keep the location within the shape defined by the nb_interpo interpolating profiles.
    - method: 'lin' to interpolate values linearly w.r.t. nb_interpo values; 'KNN' to set locations to the nearest existing location (nearest Kz profile location).
    - alphaAR: to manage the frequence of the AR signal. 0.7 for periods of weeks (given that a low pass mitigates high frequencies).
    """
    lat_lon_Kz = np.load(path+"lat_lon_index.npy") ## Location of already existing Kz profiles (from polgyr simulation)
    laglat, laglon = np.array([[0.], [0.]])
    for i in range(365*3) :
        laglat = np.concatenate((laglat, laglat[-1:]*alphaAR + random.random()))
        laglon = np.concatenate((laglon, laglon[-1:]*alphaAR + random.random()))
    if case == 0 : # No location uncertainty
        distance_zone = 1000000.
        sigmaAR = 0.
    elif case == 1 : # Location uncertainty <=0.3° for 95% of samples
        sigmaAR = 0.15
        distance_zone = sigmaAR*3
    elif case == 2 : # Location uncertainty <=0.5° for 95% of samples
        sigmaAR = 0.25
        distance_zone = sigmaAR*3
    elif case == 3 : # Location uncertainty <=1° for 95% of samples
        sigmaAR = 0.5
        distance_zone = sigmaAR*3
    laglat = sigmaAR*(laglat[366:]-np.mean(laglat[366:]))/np.std(laglat)
    laglon = sigmaAR*(laglon[366:]-np.mean(laglon[366:]))/np.std(laglon)
    ## If not defined, the uncertainty locations of the generated profile are computed
    lag_lat_lon = np.concatenate((lowpass(laglat, cutoff = 1/3, sample_rate = 2, poles = 3)[None], lowpass(laglon, cutoff = 1/3, sample_rate = 2, poles = 3)[None]), axis = 0)
    try :
        if lat_lon_F == None :
            lat_lon_F = lag_lat_lon + np.array([[lat], [lon]]).repeat(lag_lat_lon.shape[1], 1)
    except :
        print("Already existing path")

    ## lat/lon locations are stored
    if method == 'KNN' : ## Select closest existing Kz profile location
        lat_lon_Kz_fit = np.zeros([2, 730])
    elif method == 'lin' and not clip : ## Keep the generated location (lat_lon_F)
        lat_lon_Kz_fit = np.zeros([4, 730])
    else : ## Selects a location in the area defined by the interpolating profiles that is the closest from the generated location (lat_lon_F)
        lat_lon_Kz_fit = np.zeros([2, 730])

    ## Sorts the existing profiles that are the closest from the generated true location (for interpolation)
    dist_profiles = np.sqrt((lat-lat_lon_Kz[0])**2+(lon-lat_lon_Kz[1])**2)
    i_square_Kz = np.argsort(dist_profiles)[:np.sum(dist_profiles<distance_zone)]

    
    ##################################################################
    ########################## GENERATES Kz ##########################
    ##################################################################
    ## Generates true Kz profile (at lat/lon)
    i_closer_profiles_Kz_T = np.argsort(np.sqrt((lat-lat_lon_Kz[0])**2+(lon-lat_lon_Kz[1])**2))[:nb_interpo]
    if method == 'KNN' :
        Kz_interpo_T = Kz_bank[i_closer_profiles_Kz_T[0]]
    elif method == 'lin' :
        Kz_interpo_T = 10**pointValue(lon, lat, 8, 1/5, lat_lon_Kz[1, i_closer_profiles_Kz_T], lat_lon_Kz[0, i_closer_profiles_Kz_T], np.log10(Kz_bank[i_closer_profiles_Kz_T]))
        
    ## Generates false Kz profile (at lat+laglat/lon+laglon)
    Kz_interpo_F = np.zeros([730, len(zw_grid)])
    for t in range(730) :
        if method == 'KNN' :
            if clip :
                i_closer_profiles_Kz_F = i_square_Kz[np.argmin(np.sqrt((lat_lon_F[0, t]-lat_lon_Kz[0])**2+(lat_lon_F[1, t]-lat_lon_Kz[1])**2))]
                lat_lon_Kz_fit[:, t] = np.array([lat_lon_Kz[:, i_closer_profiles_Kz_F]])
                Kz_interpo_F[t] = Kz_bank[i_closer_profiles_Kz_F, t]
            else :
                i_closer_profiles_Kz_F = np.argmin(np.sqrt((lat_lon_F[0, t]-lat_lon_Kz[0])**2+(lat_lon_F[1, t]-lat_lon_Kz[1])**2))
                lat_lon_Kz_fit[:, t] = np.array([lat_lon_Kz[:, i_closer_profiles_Kz_F]])
                Kz_interpo_F[t] = Kz_bank[i_closer_profiles_Kz_F, t]
        elif method == 'lin' :
            if clip :
                lat_lon_Kz_fit[:, t] = clipin_area(lat_lon_Kz, lat_lon_Kz[:, i_square_Kz], lat_lon_F[:, t])
                surrounding_Kz_profiles = i_square_Kz[np.argsort(np.sqrt((lat_lon_Kz_fit[0, t]-lat_lon_Kz[0, i_square_Kz])**2+(lat_lon_Kz_fit[1, t]-lat_lon_Kz[1, i_square_Kz])**2))[:nb_interpo]]
                Kz_interpo_F[t] = 10**pointValue(lat_lon_Kz_fit[1, t], lat_lon_Kz_fit[0, t], 8, 1/5, lat_lon_Kz[1, surrounding_Kz_profiles], lat_lon_Kz[0, surrounding_Kz_profiles], np.log10(Kz_bank[surrounding_Kz_profiles, t]))
            else :
                i_closer_profiles_Kz_F = np.argsort(np.sqrt((lat_lon_F[0, t]-lat_lon_Kz[0])**2+(lat_lon_F[1, t]-lat_lon_Kz[1])**2))[:nb_interpo]
                lat_lon_Kz_fit[:, t] = i_closer_profiles_Kz_F
                Kz_interpo_F[t] = 10**pointValue(lat_lon_F[1, t], lat_lon_F[0, t], 8, 1/5, lat_lon_Kz[1, i_closer_profiles_Kz_F], lat_lon_Kz[0, i_closer_profiles_Kz_F], np.log10(Kz_bank[i_closer_profiles_Kz_F, t]))

    ##################################################################
    ########################## GENERATES I0 ##########################
    ##################################################################
    I0 = np.load(path+"SSR.npy")[:, year*365:(year+1)*365]
    lat_lon_I0 = np.load(path+"lat_lon_index_for.npy")

    ## Generates true I0 profile
    if method == 'KNN' :
        i_closer_profiles_I0_T = np.argsort(np.sqrt((lat_lon_Kz[0, i_closer_profiles_Kz_T[0]]-lat_lon_I0[0])**2+(lat_lon_Kz[1, i_closer_profiles_Kz_T[0]]-lat_lon_I0[1])**2))[:nb_interpo]
        I0_interpo_T = pointValue(lat_lon_Kz[1, i_closer_profiles_Kz_T[0]], lat_lon_Kz[0, i_closer_profiles_Kz_T[0]], 8, 1/5, lat_lon_I0[1, i_closer_profiles_I0_T], lat_lon_I0[0, i_closer_profiles_I0_T], I0[i_closer_profiles_I0_T])
    elif method == 'lin' :
        i_closer_profiles_I0_T = np.argsort(np.sqrt((lat-lat_lon_I0[0])**2+(lon-lat_lon_I0[1])**2))[:nb_interpo]
        I0_interpo_T = pointValue(lon, lat, 8, 1/5, lat_lon_I0[1, i_closer_profiles_I0_T], lat_lon_I0[0, i_closer_profiles_I0_T], I0[i_closer_profiles_I0_T])
    
    ## Generates false I0 profile
    I0_interpo_F = np.zeros([365])
    for t in range(365) :
        if method == 'KNN' :
            if clip :
                i_closer_profiles_Kz_F = i_square_Kz[np.argmin(np.sqrt((lat_lon_F[0, 2*t]-lat_lon_Kz[0])**2+(lat_lon_F[1, 2*t]-lat_lon_Kz[1])**2))]
                i_closer_profiles_I0_F = np.argsort(np.sqrt((lat_lon_Kz[0, i_closer_profiles_Kz_F]-lat_lon_I0[0])**2+(lat_lon_Kz[1, i_closer_profiles_Kz_F]-lat_lon_I0[1])**2))[:nb_interpo]
                I0_interpo_F[t] = pointValue(lat_lon_Kz[1, i_closer_profiles_Kz_F], lat_lon_Kz[0, i_closer_profiles_Kz_F], 8, 1/5, lat_lon_I0[1, i_closer_profiles_I0_F], lat_lon_I0[0, i_closer_profiles_I0_F], I0[i_closer_profiles_I0_F, t])
            else :
                i_closer_profiles_Kz_F = np.argmin(np.sqrt((lat_lon_F[0, 2*t]-lat_lon_Kz[0])**2+(lat_lon_F[1, 2*t]-lat_lon_Kz[1])**2))
                i_closer_profiles_I0_F = np.argsort(np.sqrt((lat_lon_Kz[0, i_closer_profiles_Kz_F]-lat_lon_I0[0])**2+(lat_lon_Kz[1, i_closer_profiles_Kz_F]-lat_lon_I0[1])**2))[:nb_interpo]
                I0_interpo_F[t] = pointValue(lat_lon_Kz[1, i_closer_profiles_Kz_F], lat_lon_Kz[0, i_closer_profiles_Kz_F], 8, 1/5, lat_lon_I0[1, i_closer_profiles_I0_F], lat_lon_I0[0, i_closer_profiles_I0_F], I0[i_closer_profiles_I0_F, t])
        elif method == 'lin' :
            if clip :
                i_closer_profiles_I0_F = np.argsort(np.sqrt((lat_lon_Kz_fit[0, 2*t]-lat_lon_I0[0])**2+(lat_lon_Kz_fit[1, 2*t]-lat_lon_I0[1])**2))[:nb_interpo]
                I0_interpo_F[t] = pointValue(lat_lon_Kz_fit[1, 2*t], lat_lon_Kz_fit[0, 2*t], 8, 1/5, lat_lon_I0[1, i_closer_profiles_I0_F], lat_lon_I0[0, i_closer_profiles_I0_F], I0[i_closer_profiles_I0_F, t])
            else :
                i_closer_profiles_I0_F = np.argsort(np.sqrt((lat_lon_F[0, 2*t]-lat_lon_I0[0])**2+(lat_lon_F[1, 2*t]-lat_lon_I0[1])**2))[:nb_interpo]
                I0_interpo_F[t] = pointValue(lat_lon_F[1, 2*t], lat_lon_F[0, 2*t], 8, 1/5, lat_lon_I0[1, i_closer_profiles_I0_F], lat_lon_I0[0, i_closer_profiles_I0_F], I0[i_closer_profiles_I0_F, t])

    return Kz_interpo_T, Kz_interpo_F, I0_interpo_T, I0_interpo_F, lat_lon_Kz_fit, lat_lon_F, np.array([lat, lon])


def clipin_area(lat_lon_bank, lat_lon_box, lat_lon_ptest) :
    """ Returns the projected location of a point on the area defined by interpolating profiles.
    - lat_lon_bank: Location of the already existing profiles.
    - lat_lon_box: Location of the existing profiles that are used for interpolation.
    - lat_lon_ptest: Location of the point to interpolate.
    """
    square_ptest = lat_lon_bank[:, np.argsort(((lat_lon_ptest[0]-lat_lon_bank[0])**2 + (lat_lon_ptest[1]-lat_lon_bank[1])**2))[:4]]
    common_pt = 0
    for pt in square_ptest.T :
        if np.min((pt[1]-lat_lon_box[1])**2 + (pt[1]-lat_lon_box[1])**2) == 0 :
            common_pt += 1
    if common_pt >= 3 and np.max(np.sqrt((lat_lon_ptest[0]-square_ptest[0])**2 + (lat_lon_ptest[1]-square_ptest[1])**2)) <= 0.8 :
        return lat_lon_ptest
    else :
        points_to_clipbetween = np.argsort((lat_lon_ptest[0]-lat_lon_box[0])**2 + (lat_lon_ptest[1]-lat_lon_box[1])**2)[:2]
        section_to_clipon = side_pts(lat_lon_box[:, points_to_clipbetween[0]], lat_lon_box[:, points_to_clipbetween[1]])
        return section_to_clipon[:, np.argmin((lat_lon_ptest[0]-section_to_clipon[0])**2 + (lat_lon_ptest[1]-section_to_clipon[1])**2)]


def Gen_data(path_name, nb_subset, nb_sample, case, device, save_stats = False) :
    """Generates several subsets of samples (true and false forcings (Kz/I0)and associated states for a given uncertainty case.
    - path_name: path of directory where data are saved.
    - nb_subset, nb_sample: number of subset to generate and number of sample per subset. Avoid dealing with too large datasets that may crash the machine.
    - case: level of uncertainty (between 0 to 3, 0: no uncertainty, 1: 0.3°, 2: 0.5° and 3: 1° location uncertainty).
    - device: either cpu or cuda if available.
    - save_stats: if True, the function computes and saves mean and standard deviation of forcings and states.
    """
    if not os.path.isdir(path_name+"Loading_files/") :
        os.makedirs(path_name+"Loading_files/")
    ## Reference value of BGC parameters
    alpha = 0.025 # (W.m-2.d)-1 initial slope P_I
    Xi = 0.05 # d-1 phyto specific mortality rate
    rho = 2. # d-1 zoo max grazing
    gamma = 0.2 # fraction of zoo excreted and mortality (Z to D)
    Gamma = 0.03 # d-1 zoo egestion (going into NH4)
    phi = 0.03  # d-1 remin (detritus decomp)
    omega = 10. # m.d-1 D sinking
    param_ref = torch.tensor([alpha, Xi, rho, gamma, Gamma, phi, omega])
    
    forcing_path = 'FORCING_40km/' # Path of profiles generated from polgyr
    dt = (30/60)/24 # integration time step
    year = 3 # number of simulated years
    trange = torch.arange(0, 365*year, dt)
    zmax = 350 # maximum depth studied
    Nz = np.argmin(np.load(forcing_path+"depth.npy")[0] < zmax) # number of layers between the surface and 350m
    Kz_bank = torch.from_numpy(np.load(forcing_path+"akt.npy")) # Kz profiles from polgyr
    lat_lon_Kz = np.load(forcing_path+"lat_lon_index.npy") # location (lat/lon) of profiles from polgyr
    zrangew_Kz = torch.from_numpy(np.load(forcing_path+"depthw.npy")) # layer depth related to Kz profiles
    
    z_grid = torch.from_numpy(np.load(forcing_path+"depth.npy")[0, :Nz]) # depth of layer centers (where tracers are located), between 0-350 m
    zw_grid = torch.from_numpy(np.load(forcing_path+"depthw.npy")[0, :Nz+1]) # depth of layer borders (where kz is located), between 0-350 m
    ## interpolate the polgyr Kz profiles on the same zw_grid
    Kz_bank_interpo = torch.zeros([Kz_bank.shape[0], Kz_bank.shape[1], len(zw_grid)])
    for i_profile in range(Kz_bank.shape[0]) :
        Kz_bank_interpo[i_profile] = 10**linear_interpolate_1d(zw_grid, zrangew_Kz[i_profile], torch.log10(Kz_bank[i_profile]))
        
    ti = time.time()
    for nb_file in range(nb_subset) :
        stud_yr = torch.tensor([random.randint(0, 6) for yr in range(nb_sample)]) # randomly select one year among the 7 years available
        Kz_batch = torch.zeros([2, nb_sample, 365*2, Nz+1]) # to store true/false Kz pofile
        I0_batch = torch.zeros([2, nb_sample, 365]) # to store true/false I0 profile
        lat_lon_batch = torch.zeros([nb_sample, 6, 730]) # to store lat_lon of true profile and lat_lon of false profile
    
        ########################################################################
        ########################## FORCING SIMULATION ##########################
        ########################################################################
        ## For each sample, generates two one-year forcing profiles: a true profile (at lat/lon location) and a false profile (at lat/lon+epsilon location).
        for nsamp in range(nb_sample) :
            Kz_interpo_T, Kz_interpo_F, I0_interpo_T, I0_interpo_F, lat_lon_F_fit, lat_lon_F, lat_lon_T = select_forcing_gen(path=forcing_path, lat=47.5+random.random()*3, lon=-18+random.random()*3, year=stud_yr[nsamp], zw_grid=np.array(zw_grid.cpu()), Kz_bank=np.array(Kz_bank_interpo[:, 730*stud_yr[nsamp]:730*(stud_yr[nsamp]+1)].cpu()), case=case)
        
            Kz_batch[0, nsamp] = torch.from_numpy(Kz_interpo_T)
            I0_batch[0, nsamp] = torch.from_numpy(I0_interpo_T)
            Kz_batch[1, nsamp] = torch.from_numpy(Kz_interpo_F)
            I0_batch[1, nsamp] = torch.from_numpy(I0_interpo_F)
            lat_lon_batch[nsamp, :2] = torch.from_numpy(lat_lon_T[:, None]) # True location
            lat_lon_batch[nsamp, 2:4] = torch.from_numpy(lat_lon_F) # False location
            lat_lon_batch[nsamp, 4:] = torch.from_numpy(lat_lon_F_fit) # False location inside restricted area defined by interpolating profiles
            sys.stdout.write(f"\rProfile n°{nsamp+1}/{nb_sample} generated")
    
        ## Generates set of BGC parameters around reference value +/- 20%
        prct_err_theta = 0.2
        parameters_batch = param_ref.repeat(2, nb_sample, 1)
        for itheta in range(parameters_batch.shape[-1]) :
            parameters_batch[0, :, itheta] += torch.tensor([(random.random()*2*prct_err_theta-prct_err_theta)*param_ref[itheta] for i in range(nb_sample)]) # True set of parameters
            parameters_batch[1, :, itheta] += torch.tensor([(random.random()*2*prct_err_theta-prct_err_theta)*param_ref[itheta] for i in range(nb_sample)]) # False set of parameters
        
        ## Calculates initial states and nudging profile of Nitrate concentration, defined according to MLDmax and MLD/NO3 curve from polgyr simulations
        X_in = torch.zeros([nb_sample, 5, Nz]) # Stores initial states
        beta_z_curve = torch.from_numpy(np.load(forcing_path+"NO3_start.npy")).to(torch.float32).to(device)
        zMLD_T = calcul_MLDmax(zw_grid.to(device), Kz_batch[0])
        zMLD_F = calcul_MLDmax(zw_grid.to(device), Kz_batch[1])
        beta_T = beta_z_curve[1, [torch.argmin(1*(beta_z_curve[0] < z)) for z in zMLD_T]]
        beta_F = beta_z_curve[1, [torch.argmin(1*(beta_z_curve[0] < z)) for z in zMLD_F]]
        X_in = torch.cat([init_NNPZD(beta_T[nsamp], Nz)[None] for nsamp in range(nb_sample)], dim = 0)
        
        parameters_batch = torch.cat((parameters_batch, torch.cat((beta_T[None], beta_F[None]), dim=0)[:, :, None]), dim=2)
        
        ## BGC model is defined according to a set of parameters, time/space domain and forcings Kz/I0
        Model_BGC = Model_NNPZD_1D(parameters_batch[0].clone(), trange, z_grid.to(device), zw_grid.to(device), Kz_batch[0].repeat(1, year, 1), I0_batch[0].repeat(1, year))
    
        with torch.no_grad() :
            Model_BGC.NO3_0 = X_in[:, 0, None].clone().to(device)
            Model_BGC.NH4_0 = X_in[:, 1, None].clone().to(device)
            Model_BGC.P_0 = X_in[:, 2, None].clone().to(device)
            Model_BGC.Z_0 = X_in[:, 3, None].clone().to(device)
            Model_BGC.D_0 = X_in[:, 4, None].clone().to(device)
            
        Model_BGC.eval()
        Model_BGC.to(device)
        Model_BGC.device = device

        torch.save(Kz_batch, path_name+f"Loading_files/Kz_{nb_file}.pt")
        torch.save(I0_batch, path_name+f"Loading_files/I0_{nb_file}.pt")
        torch.save(parameters_batch, path_name+f"Loading_files/Params_{nb_file}.pt")
        torch.save(stud_yr, path_name+f"Loading_files/Years_{nb_file}.pt")
        torch.save(lat_lon_batch, path_name+f"Loading_files/Latlon_{nb_file}.pt")
        
        ########################################################################
        ########################### STATE SIMULATION ###########################
        ########################################################################
        with torch.no_grad() :
            States_ref, It0_ref, Kz_grid = Model_BGC(sub_windows = torch.arange(0, 365*3, 1)[None], info_time=True)
        States_ref = States_ref[:, :, 0].clone()
        torch.save(States_ref, path_name+f"Loading_files/States_{nb_file}.pt")
        print(f"Subset n°{nb_file+1}/{nb_subset} generated in {datetime.timedelta(seconds=time.time()-ti)}s")
        ti = time.time()
        
    ########################################################################
    ########################### DATA CONCATENATE ###########################
    ########################################################################
    del States_ref, Kz_batch, I0_batch, parameters_batch, It0_ref, Kz_grid # free storage
    States_cat = torch.tensor([])
    for i_sub in range(nb_subset) :
        States_cat = torch.cat((States_cat, torch.load(path_name+f"Loading_files/States_{i_sub}.pt", weights_only=False)), dim = 0)
    torch.save(States_cat, path_name+f"States.pt")
    if save_stats :
        torch.save(torch.std(States_cat, dim=(0, 2, 3)), path_name+f"States_std.pt")
        torch.save(torch.mean(States_cat, dim=(0, 2, 3)), path_name+f"States_mean.pt")
    del States_cat
    
    Kz_cat = torch.tensor([])
    for i_sub in range(nb_subset) :
        Kz_cat = torch.cat((Kz_cat, torch.load(path_name+f"Loading_files/Kz_{i_sub}.pt", weights_only=False)), dim = 1)
    torch.save(Kz_cat, path_name+f"Kz.pt")
    if save_stats :
        torch.save(torch.std(torch.log10(Kz_cat)), path_name+f"Kzlog10_std.pt")
        torch.save(torch.mean(torch.log10(Kz_cat)), path_name+f"Kzlog10_mean.pt")
    del Kz_cat
    
    I0_cat = torch.tensor([])
    for i_sub in range(nb_subset) :
        I0_cat = torch.cat((I0_cat, torch.load(path_name+f"Loading_files/I0_{i_sub}.pt", weights_only=False)), dim = 1)
    torch.save(I0_cat, path_name+f"I0.pt")
    if save_stats :
        torch.save(torch.std(I0_cat), path_name+f"I0_std.pt")
        torch.save(torch.mean(I0_cat), path_name+f"I0_mean.pt")
    del I0_cat
    
    Params_cat = torch.tensor([])
    for i_sub in range(nb_subset) :
        Params_cat = torch.cat((Params_cat, torch.load(path_name+f"Loading_files/Params_{i_sub}.pt", weights_only=False)), dim = 1)
    torch.save(Params_cat, path_name+f"Params.pt")
    if save_stats :
        torch.save(torch.std(Params_cat, dim=(0, 1)), path_name+f"Params_std.pt")
        torch.save(torch.mean(Params_cat, dim=(0, 1)), path_name+f"Params_mean.pt")
    del Params_cat
    
    Years_cat = torch.tensor([])
    for i_sub in range(nb_subset) :
        Years_cat = torch.cat((Years_cat, torch.load(path_name+f"Loading_files/Years_{i_sub}.pt", weights_only=False)), dim = 0)
    torch.save(Years_cat, path_name+f"Years.pt")
    del Years_cat
    
    Latlon_cat = torch.tensor([])
    for i_sub in range(nb_subset) :
        Latlon_cat = torch.cat((Latlon_cat, torch.load(path_name+f"Loading_files/Latlon_{i_sub}.pt", weights_only=False)), dim = 0)
    torch.save(Latlon_cat, path_name+f"Latlon.pt")
    del Latlon_cat
    
    for file in os.listdir(path_name+"/Loading_files/") :
        os.remove(path_name+"/Loading_files/"+file)
    os.rmdir(path_name+"Loading_files")


def get_topcorner(ax, lim = 0.9) :
    ## computes y location of the plot to display subscript
    return ax.get_ybound()[0]+(ax.get_ybound()[1]-ax.get_ybound()[0])*lim
def get_rightcorner(ax, lim = 0.9) :
    ## computes x location of the plot to display subscript
    return ax.get_xbound()[0]+(ax.get_xbound()[1]-ax.get_xbound()[0])*lim


def load_best_model(name_file, criterion = "valid_loss") :
    """Returns the path of the best NN model version. """
    selection_model = pd.DataFrame(listdir(name_file+f"every_n_epochs/"), columns = ["file_name"])
    selection_model["epoch"] = [int(path_model.lstrip("chkpt_epoch=").rstrip(".ckpt")) for path_model in selection_model.file_name]

    logs = pd.read_csv(name_file+f"lightning_logs/version_0/metrics.csv", header = 0)
    tensor_logs = torch.from_numpy(np.array([logs.epoch.unique()]))
    list_cols = ["epoch", "train_loss_epoch", "valid_loss"]
    for name_col in list_cols[1:] :
        new_col = torch.from_numpy(np.array([[np.nanmin(logs[logs.epoch == i_epoch][name_col]) for i_epoch in logs.epoch.unique()]]))
        tensor_logs = torch.cat((tensor_logs, new_col), dim = 0)
    logs_clean = pd.DataFrame(tensor_logs.transpose(0, 1), columns = list_cols)

    selection_model[criterion] = [logs_clean[logs_clean.epoch == top_model_epoch][criterion].item() for top_model_epoch in selection_model.epoch]
    chckpt_path = name_file+f"every_n_epochs/"+selection_model[selection_model[criterion] == min(selection_model[criterion])].file_name.item()
    print(f"Selected model: {selection_model[selection_model[criterion] == min(selection_model[criterion])].file_name.item()}")
    return chckpt_path


def data_sampling(DS_in, sampling_patt, i_z_ch, dt=1.) :
    """ Puts zeros to non-observed data of the TensorDataset. 
    - DS_in: Input tensor dataset.
    - sampling_patt: list of the time sampling for each state of the dataset (5 first channels).
    - i_z_ch: list of the vertical space sampling for each state.
    - dt: timestep of the dataset.
    """
    DL = DataLoader(DS_in, batch_size = 10000)
    sampling_patt_t =(1/dt)*torch.tensor(sampling_patt)
    for x, y in DL :
        x_replace = torch.zeros(x.shape)
        y_replace = torch.clone(y)
        for i in range(len(sampling_patt_t)) :
            for day in range(x_replace.shape[2]) :
                if day%sampling_patt_t[i] == 0 :
                    x_replace[:, i, day] = x[:, i, day]
            if len(i_z_ch[i]) > 0 :
                x_replace[:, i, :, i_z_ch[i]] = x[:, i, :, i_z_ch[i]]
    x_replace[:, 5:, :] = x[:, 5:, :]
    return TensorDataset(x_replace, y_replace)


def data_sampling_rdmshift(DS_in, sampling_patt, i_z_ch, dt=1., list_firstobsday=[]) :
    """ Does the same as data_sampling but randomly selects the first observed day for each channel.
    - list_firstobsday: if empty, generates a list of first observed date, randomly (uniformly) selected between 0 and X day, X being the time sampling.
    """
    DL = DataLoader(DS_in, batch_size = 10000)
    sampling_patt_t = int(1/dt)*torch.tensor(sampling_patt, dtype=torch.int)
    for x, y in DL :
        x_replace = x.clone()
        y_replace = y.clone()
        if not len(list_firstobsday) :
            list_firstobsday = [int(random.randint(0, max(sampling_patt)-1)/dt) for sample in range(x.shape[0])]
        mask_dtdz = torch.ones([5, x.shape[2], x.shape[3]])*torch.nan
        for ch in range(len(sampling_patt_t)) :
            if sampling_patt_t[ch] :
                for day in range(x.shape[2]) :
                    if day%sampling_patt_t[ch] == 0 :
                        mask_dtdz[ch, day, i_z_ch[ch]] = 1
        mask_dtdz_rdm = torch.ones([x.shape[0], 5, x.shape[2], x.shape[3]])*torch.nan
        for sample in range(x.shape[0]) :
            if list_firstobsday[sample] :
                mask_dtdz_rdm[sample, :, list_firstobsday[sample]:] = mask_dtdz[:, :-list_firstobsday[sample]]
            else :
                mask_dtdz_rdm[sample] = mask_dtdz
    x_replace[:, :5] = x[:, :5]*mask_dtdz_rdm
    return TensorDataset(x_replace, y_replace)


def data_sampling_rdmshift_ens(DS_in, sampling_patt, i_z_ch, dt=1., list_firstobsday=[]) :
    """ Does the same as data_sampling_rdmshift, for an ensemble (dimensions: n_ensemble, n_sample, n_channel, n_t, n_z). """
    DL = DataLoader(DS_in, batch_size = 10000)
    sampling_patt_t = int(1/dt)*torch.tensor(sampling_patt, dtype=torch.int)
    
    for x, y in DL :
        x_replace = x.clone()
        y_replace = y.clone()
        if not len(list_firstobsday) :
            list_firstobsday = [int(random.randint(0, max(sampling_patt)-1)/dt) for sample in range(x.shape[0])]
        mask_dtdz = torch.ones([5, x.shape[3], x.shape[4]])*torch.nan
        for ch in range(len(sampling_patt_t)) :
            if sampling_patt_t[ch] :
                for day in range(x.shape[3]) :
                    if day%sampling_patt_t[ch] == 0 :
                        mask_dtdz[ch, day, i_z_ch[ch]] = 1
        mask_dtdz_rdm = torch.ones([x.shape[0], x.shape[1], 5, x.shape[3], x.shape[4]])*torch.nan
        for sample in range(x.shape[0]) :
            if list_firstobsday[sample] :
                mask_dtdz_rdm[sample, :, :, list_firstobsday[sample]:] = mask_dtdz[None, :, :-list_firstobsday[sample]].repeat(x.shape[1], 1, 1, 1)
            else :
                mask_dtdz_rdm[sample] = mask_dtdz[None].repeat(x.shape[1], 1, 1, 1)
    x_replace[:, :, :5] = x[:, :, :5]*mask_dtdz_rdm
    return TensorDataset(x_replace, y_replace)



def metrics_BGC(data_obs, data_pred, dt = 1) :
    """Computes correlation, shift and amplitude metrics between two datasets"""
    def torch_crosscorr(x, y) :
        return torch.nn.functional.conv1d(x[None, None], y[None, None], padding=y.shape[0] - 1)[0, 0]
    corr, shift, ampl = torch.zeros([3, data_obs.shape[0], data_obs.shape[1]])
    for isamp in range(data_obs.shape[0]) :
        for ich in range(data_obs.shape[1]) :
            corr[isamp, ich] = torch.corrcoef(torch.cat((data_obs[None, isamp, ich], data_pred[None, isamp, ich]), dim = 0))[1, 0]
            shift[isamp, ich] = dt*(torch_crosscorr(data_obs[isamp, ich], data_obs[isamp, ich]).argmax() - torch_crosscorr(data_obs[isamp, ich], data_pred[isamp, ich]).argmax())
            ampl[isamp, ich] = (torch.max(data_pred[isamp, ich])-torch.min(data_pred[isamp, ich]))/(torch.max(data_obs[isamp, ich])-torch.min(data_obs[isamp, ich]))
    return corr, shift, ampl


class Model_NNPZD_1D(torch.nn.Module) :
    def __init__(self, params_value, trange, zrange, zrangew, Kz, I0) :
        super().__init__()        
        ## Domain variable
        self.trange = (trange).clone().detach() # time domain
        self.dt = (self.trange[1] - self.trange[0]).item() # time sampling
        self.zT = zrange.clone().detach() # vertical space for tracers
        self.zw = zrangew.clone().detach() # vertical space for Kz
        self.Nz = self.zT.shape[0] # number of layer
        self.dzT = (self.zw[1:]-self.zw[:-1]).clone().detach() # layer width
        self.dzw = torch.cat(((self.dzT[0]/2)[None], self.zT[1:]-self.zT[:-1], (self.dzT[self.Nz-1]/2)[None])) # distance between the center of each layer
        self.Nsample = params_value.shape[0]
        ## BGC parameters
        self.xi     = torch.ones([params_value.shape[0]])*0.067 # => \xi (water attenuation)
        self.zeta     = torch.ones([params_value.shape[0]])*9.5e-3 # => \zeta (phyto attenuation)
        self.Lambda = torch.ones([params_value.shape[0]])*0.06 # => \Lambda (ivlev)
        self.Psi    = torch.ones([params_value.shape[0]])*1.46 # => \Psi (NH4 inhib)
        self.mu  = torch.ones([params_value.shape[0]])*0.25 # => \mu (NH4 oxyd)
        self.eta     = torch.ones([params_value.shape[0]])*1.5 # => \eta (phyto max N uptake rate)
        self.kappa     = torch.ones([params_value.shape[0]])*1.0 # => \kappa (half saturation)
        self.alpha = torch.nn.Parameter(params_value[:, 0]) # => \alpha (initial slope P-I curve)
        self.Xi    = torch.nn.Parameter(params_value[:, 1]) # => \Xi (P mortality rate)
        self.rho    = torch.nn.Parameter(params_value[:, 2]) # => \rho (Zoo max grazing rate)
        self.gamma = torch.nn.Parameter(params_value[:, 3]) # => \gamma (Z excretion/mortality Z->D)
        self.Gamma = torch.nn.Parameter(params_value[:, 4]) # => \Gamma (frac Z grazing egested Z->NH4)
        self.varphi   = torch.nn.Parameter(params_value[:, 5]) # => \varphi (D decomposition/remin)
        self.omega    = torch.nn.Parameter(params_value[:, 6]) # => \omega (D sinking rate)
        self.beta    = torch.nn.Parameter(params_value[:, 7]) # => \beta (nudging profile of N concentration)
        ## Initial conditions
        self.X0 = torch.autograd.Variable(torch.zeros((self.Nsample, 5, self.Nz)), requires_grad = True)
        ## Forcings
        self.Kz = Kz.clone().detach()
        self.I0 = I0.clone().detach()
    
    
    def bio_step(self, NO3_step, NH4_step, P_step, Z_step, D_step, It_step) :
        """ Returns the BGC state variation due to BGC processes. """
        G = self.eta[:, None, None] * self.alpha[:, None, None]*It_step / torch.sqrt(torch.pow(self.eta[:, None, None], 2) + torch.pow(self.alpha[:, None, None]* It_step, 2))
        bio_NO3 = self.mu[:, None, None]*NH4_step - G*torch.exp(-self.Psi[:, None, None]*NH4_step)*P_step*(NO3_step/(self.kappa[:, None, None]+NO3_step))
        bio_NH4 = self.varphi[:, None, None]*D_step + self.Gamma[:, None, None]*Z_step - G*P_step*(NH4_step/(self.kappa[:, None, None]+NH4_step)) - self.mu[:, None, None]*NH4_step
        bio_P = G*P_step*(torch.exp(-self.Psi[:, None, None]*NH4_step)*(NO3_step/(self.kappa[:, None, None]+NO3_step)) + (NH4_step/(self.kappa[:, None, None]+NH4_step))) - self.rho[:, None, None]*(1-torch.exp(-self.Lambda[:, None, None]*P_step))*Z_step - self.Xi[:, None, None]*P_step
        bio_Z = (1-self.gamma[:, None, None])*self.rho[:, None, None]*(1-torch.exp(-self.Lambda[:, None, None]*P_step))*Z_step - self.Gamma[:, None, None]*Z_step
        bio_D = self.gamma[:, None, None]*self.rho[:, None, None]*(1-torch.exp(-self.Lambda[:, None, None]*P_step))*Z_step + self.Xi[:, None, None]*P_step - self.varphi[:, None, None]*D_step
        return bio_NO3, bio_NH4, bio_P, bio_Z, bio_D

    
    def diff_step(self, comp_i, Kz, A_u = None) :
        """ Returns diffusion matrix from Kz profile, and BGC state variation due to diffusion. """
        if A_u == None :
            a_dif_sup = - self.dt * Kz/(self.dzT[:self.Nz-1]*self.dzw[1:self.Nz])
            a_dif_inf = - self.dt * Kz/(self.dzT[1:self.Nz]*self.dzw[1:self.Nz])
            a_dif_diag = torch.cat(((1 - a_dif_sup[:, :, 0])[:, :, None], 1 - a_dif_inf[:, :, :self.Nz-2] - a_dif_sup[:, :, 1:self.Nz-1], (1 - a_dif_inf[:, :, self.Nz-2])[:, :, None]), dim = 2)
            A_u = torch.zeros([self.Nsample, self.n_subwindow, self.Nz, self.Nz]).to(self.device)
            for nsample in range(self.Nsample) :
                for nwin in range(self.n_subwindow) :
                    A_u[nsample, nwin] = torch.diagflat(a_dif_inf[nsample, nwin], -1) +\
                        torch.diagflat(a_dif_diag[nsample, nwin]) +\
                        torch.diagflat(a_dif_sup[nsample, nwin], 1)
        return A_u, torch.linalg.solve(A_u, comp_i)

    def weno3z(self, qm, q0, qp):
        """ Resolves the weno3z scheme to integrate a tracer in the sedimentation process. """
        eps = 1e-14
        qi1 = -1./2.*qm + 3./2.*q0
        qi2 = 1./2.*(q0 + qp)
        beta1 = (q0-qm)**2
        beta2 = (qp-q0)**2
        tau = abs(beta2-beta1)
        g1, g2 = 1./3., 2./3.
        w1 = g1 * (1. + tau / (beta1 + eps))
        w2 = g2 * (1. + tau / (beta2 + eps))
        qi_weno3 = (w1*qi1 + w2*qi2) / (w1 + w2)
        return qi_weno3
    
    def sed_step(self, T, sed_coef) :
        """ Returns the sedimentation flux from the tracer profile and the sedimentation coefficient. """
        flux_sed = torch.zeros([self.Nsample, self.n_subwindow, self.Nz+1]).to(self.device)
        flux_sed[:, :, 2:-1] = sed_coef*self.weno3z(T[:, :, :-2]*self.dzT[:-2], T[:, :, 1:-1]*self.dzT[1:-1], T[:, :, 2:]*self.dzT[2:])/self.dzw[2:-1]
        flux_sed[:, :, 1:2] = sed_coef*T[:, :, :1]
        flux_sed[:, :, -1:] = sed_coef*T[:, :, -1:]
        return flux_sed
        
    def reinjection_step(self, NO3_step, rappel_t = 0) :
        """ Nudging of nitrate profile. """
        return NO3_step.clone() + (self.beta[:, None, None]*torch.ones([self.Nsample, self.n_subwindow, self.Nz]).to(self.device)-NO3_step.clone())*(rappel_t[:, None]*torch.ones([1, self.Nz]))

    def forward(self, sub_windows, sampling_pattern = [1, 1, 1, 1, 1], dt = (30/60)/24, info_time = False) :
        """ Resolves the model over a time window. 
        - sub_windows: time space of the sub-time-windows (adapted to weak constrained DA where 120 days are divided as 12 10-days subwindows).
        - sampling_pattern: number of day when states are stored.
        """
        ## Select forcings on the studied time-windows
        self.n_subwindow = sub_windows.shape[0] ## number of subwindow
        self.len_subwindow = sub_windows.shape[1] ## size of subwindows
        Kz_slice = torch.cat([self.Kz[:, None, int(2*sub_windows[i_win, 0]):2+int(2*sub_windows[i_win, -1])] for i_win in range(self.n_subwindow)], dim = 1)
        I0_slice = torch.cat([self.I0[:, None, int(sub_windows[i_win, 0]):1+int(sub_windows[i_win, -1])] for i_win in range(self.n_subwindow)], dim = 1)

        ## Defines when nudging shall be performed
        reinject_win = torch.zeros([self.n_subwindow, self.len_subwindow]).to(self.device)
        for t in range(0, round(self.trange[-1].item()/365)) :
            reinject_win += (1/20)*(torch.relu(sub_windows-(30+365*t)) - torch.relu(sub_windows-(31+365*t)) - torch.relu(sub_windows-(60+365*t)) + torch.relu(sub_windows-(61+365*t)))

        NO3_i = self.NO3_0
        NH4_i = self.NH4_0
        P_i = self.P_0
        Z_i = self.Z_0
        D_i = self.D_0
        NO3 = torch.cat((NO3_i.clone()[:, :, None], ))
        NH4 = torch.cat((NH4_i.clone()[:, :, None], ))
        P = torch.cat((P_i.clone()[:, :, None], ))
        Z = torch.cat((Z_i.clone()[:, :, None], ))
        D = torch.cat((D_i.clone()[:, :, None], ))
        Itz = torch.zeros([self.Nsample, self.n_subwindow, self.len_subwindow+1, self.Nz+1]).to(self.device)
        Itz_i = I0_slice[:, :, :1]*torch.exp(-self.xi[:, None, None]*self.zT-self.zeta[:, None, None]*torch.cumsum(P_i*self.dzT, dim = 2))
        
        A_ui = None ## Diffusion matrix, yet not computed
        
        step_duration = time.time()
        t = sub_windows[0, 0].item()
        for step in range(1, self.len_subwindow) :
            compteur = trunc(2*t) ## Count number of half-day (Kz time step), if Kz changes, A_ui is computed again.
            while t-sub_windows[0, 0] < sub_windows[0, step]-sub_windows[0, 0] :
                ## Biology
                Itz_i = I0_slice[:, :, int(trunc(t-sub_windows[0, 0].item()))][:, :, None]*torch.exp(-self.xi[:, None, None]*self.zT-self.zeta[:, None, None]*torch.cumsum(P_i*self.dzT, dim = 2))
                bio_NO3, bio_NH4, bio_P, bio_Z, bio_D = self.bio_step(NO3_i, NH4_i, P_i, Z_i, D_i, Itz_i)
                ## Diffusion
                if trunc(2*t) == compteur :
                    compteur += 1
                    A_ui, diff_NO3_i = self.diff_step(NO3_i+self.dt*bio_NO3, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz])
                    A_ui, diff_NH4_i = self.diff_step(NH4_i+self.dt*bio_NH4, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                    A_ui, diff_P_i = self.diff_step(P_i+self.dt*bio_P, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                    A_ui, diff_Z_i = self.diff_step(Z_i+self.dt*bio_Z, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                    A_ui, diff_D_i = self.diff_step(D_i+self.dt*bio_D, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                else :
                    A_ui, diff_NO3_i = self.diff_step(NO3_i+self.dt*bio_NO3, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                    A_ui, diff_NH4_i = self.diff_step(NH4_i+self.dt*bio_NH4, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                    A_ui, diff_P_i = self.diff_step(P_i+self.dt*bio_P, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                    A_ui, diff_Z_i = self.diff_step(Z_i+self.dt*bio_Z, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                    A_ui, diff_D_i = self.diff_step(D_i+self.dt*bio_D, Kz = Kz_slice[:, :, int(trunc(2*(t-sub_windows[0, 0].item()))), 1:self.Nz], A_u=A_ui)
                ## Sedimentation
                flux_sed_D = self.sed_step(diff_D_i, self.omega[:, None, None])
                sed_D_i = diff_D_i + self.dt*(flux_sed_D[:, :, :-1]-flux_sed_D[:, :, 1:])/self.dzT
                ## Nudging
                rej_NO3_i = self.reinjection_step(diff_NO3_i, reinject_win[:, int(trunc((t-sub_windows[0, 0].item())))])
                NO3_i = rej_NO3_i.clone(); NH4_i = diff_NH4_i.clone(); P_i = diff_P_i.clone(); Z_i = diff_Z_i.clone(); D_i = sed_D_i.clone()
                t += self.dt
            ## Store the states 
            NO3 = torch.cat((NO3, NO3_i[:, :, None]), dim = 2)
            NH4 = torch.cat((NH4, NH4_i[:, :, None]), dim = 2)
            P = torch.cat((P, P_i[:, :, None]), dim = 2)
            Z = torch.cat((Z, Z_i[:, :, None]), dim = 2)
            D = torch.cat((D, D_i[:, :, None]), dim = 2)

            Itz[:, :, step, 0] = I0_slice[:, :, int(trunc(t-sub_windows[0, 0].item()))].detach()
            Itz[:, :, step, 1:] = Itz_i.detach()
            if info_time :
                sys.stdout.write(f"\rStep {step}/{self.len_subwindow}. Ends in {datetime.timedelta(seconds=round((self.len_subwindow-step)*(time.time()-step_duration)))}s")
                step_duration = time.time()
        return torch.cat((NO3[:, None], NH4[:, None], P[:, None], Z[:, None], D[:, None]), dim = 1), Itz, Kz_slice

    def compute_cost(self, X_pred, Y_pred, Y_obs, std_ch) :
        """ Computes variational costs: observation error, model error. """
        return torch.mean(torch.moveaxis(torch.pow(Y_pred-Y_obs, 2), 1, 4)/torch.pow(std_ch, 2)), torch.mean(torch.moveaxis(torch.pow(X_pred[:, :, 1:-1, 0]-X_pred[:, :, :-2, -1], 2), 1, 3)/torch.pow(std_ch, 2))
