"""
Last update on December 2025

@author: jlittaye
Python 3.11.5
"""

## Import of libraries and functions
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from math import *
import time
import datetime
import sys
import pandas as pd
import scipy.signal

import torch
from torch import optim, nn, utils, Tensor
from torchviz import make_dot, make_dot_from_trace
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy.interpolate import RectBivariateSpline
from sklearn.linear_model import LinearRegression

from func_file import linear_interpolate_1d, Model_NNPZD_1D

devices = ['cpu']+[f"cuda:{idevice}" for idevice in range(torch.cuda.device_count())]
torch.set_default_device(devices[1])



## Defines the scenario configuration
case = 1 # forcing uncertainty level (0, 1, 2 or 3)
obs_ch = [10]*5 # States time sampling
obs_strat = 1 # Observation strategy, defines states vertical sampling
NNbase = True # True: Uses NN estimates (hybrid method), False: uses initial guesses

if obs_strat == 0 : # We observe the 5 states at every depth
    i_z_ch = [[i for i in range(34)], [i for i in range(34)], [i for i in range(34)], [i for i in range(34)], [i for i in range(34)]]
elif obs_strat == 1 : # CTD+rosette: NO3/P/D observed continuously, NH4/Z observed discretely
    i_z_ch = [[i for i in range(34)], [5, 11, 15, 20, 24, 27, 33], [i for i in range(34)], [5, 11, 15, 20, 24, 27, 33], [i for i in range(34)]]
elif obs_strat == 2 : # CTD: NO3/P/D observed continuously
    i_z_ch = [[i for i in range(34)], [], [i for i in range(34)], [], [i for i in range(34)]]
elif obs_strat == 3 : # Rosette: NO3/NH4/P/Z/D observed discretely
    i_z_ch = [[5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33]]

## Ensures that time sampling and vertical sampling goes together
for ch in range(len(obs_ch)) :
    if len(i_z_ch[ch]) == 0 :
        obs_ch[ch] = 0

## Fixes assimilation variables
totalstep = 50
lr = 1e-1 #0, 1e-1 ## BGC parameter step correction
lr_x0 = 1e-2# 1e-2 # 1e-3 #lr*1e-3  ## initial state step correction
w_bg = 1e-1 ## weight for model error cost compared to observational error

if NNbase :
    path_save = f"Res/hybrid/Case{case}_1sample{obs_ch[0]}days_strat{obs_strat}/"
else :
    path_save = f"Res/DA/Case{case}_1sample{obs_ch[0]}days_strat{obs_strat}/"

nb_version = 0 # If ones wants to do ensemble
path_save += f"Version_{nb_version}/"
if not os.path.isdir(path_save) :
    os.makedirs(path_save)
print("File: ", path_save, "with machines:", devices)

if NNbase :
    nb_version_NN = -1
    path_save_NN_save = f"Res/NN/Case{case}_1sample{obs_ch[0]}days_strat{obs_strat}/"
    for file in os.listdir(path_save_NN_save) :
        if file[:8] == "Version_" :
            nb_version_NN += 1
    path_save_NN_save += f"Version_{nb_version_NN}/pred_var/"

forcing_path = 'FORCING_40km/'
zmax = 350 # maximum depth studied
Nz = np.argmin(np.load(forcing_path+"depth.npy")[0] < zmax) # number of layers between the surface and 350m    
z_grid = torch.from_numpy(np.load(forcing_path+"depth.npy")[0, :Nz]).to(devices[-1])
zw_grid = torch.from_numpy(np.load(forcing_path+"depthw.npy")[0, :Nz+1]).to(devices[-1])
z_interpo = z_grid.clone()
i_interpo = torch.tensor([abs(z_interpo[i]-z_grid).argmin() for i in range(len(z_interpo))]).cpu()

## Importing/Generating forcings
Kz = torch.load(f"Generated_Datasets/Case_{case}/Kz_ens.pt", weights_only=False, map_location=devices[-1])[1, nb_version, -100:].repeat(1, 3, 1)
I0 = torch.load(f"Generated_Datasets/Case_{case}/I0_ens.pt", weights_only=False, map_location=devices[-1])[1, nb_version, -100:].repeat(1, 3)
params_0 = torch.load(f"Generated_Datasets/Case_{case}/Params.pt", weights_only=False, map_location=devices[-1])[1, -100:]
params_ref = torch.load(f"Generated_Datasets/Case_{case}/Params.pt", weights_only=False, map_location=devices[-1])[0, -100:]
mean_params = torch.load(f"Generated_Datasets/Case_0/Params_mean.pt", weights_only=False, map_location=devices[-1])
mean_states = torch.load(f"Generated_Datasets/Case_0/States_mean.pt", weights_only=False, map_location=devices[-1])

if NNbase :
    params_0 = torch.load(path_save_NN_save+"Params_pred_ens.pt", weights_only=False, map_location=devices[-1])[nb_version]
    KzcorrNN = torch.load(path_save_NN_save+"Kz_pred_ens.pt", weights_only=False, map_location=devices[-1])[nb_version]
    I0corrNN = torch.load(path_save_NN_save+"I0_pred_ens.pt", weights_only=False, map_location=devices[-1])[nb_version]
    Kz[:, -335*2:-215*2] = KzcorrNN.clone()
    I0[:, -335:-215] = I0corrNN.clone()

## Resolution space for assimilation
N_sample = params_0.shape[0]
Nz = len(z_grid)
i0 = 365*2+30 # first day of assimilation window over 3 years
window_T = 120 # length of the assimilation window
subwindow_T = 10 # length of the subwindow (DA weakly constrained: 12 10-day subwindows)
assim_subwindows = torch.cat([torch.arange(i0+i_win, i0+i_win+subwindow_T+1, 1)[None] for i_win in range(0, window_T, subwindow_T)], dim=0)
obs_noise_std = torch.tensor([0.04, 0.001, 0.08, 0.12, 0.04])/2 # Observation noise for NO3, NH4, P, Z and D
prct_err_theta = 0.2 # parameter space (20% around reference value)
if os.path.exists(f"Generated_Datasets/Case_0/list_d0_sampling{obs_ch[0]}d.pt") :  # list of first observed day for N_sample
    list_d0 = torch.load(f"Generated_Datasets/Case_0/list_d0_sampling{obs_ch[0]}d.pt", weights_only=False, map_location=devices[-1])
else :
    list_d0 = [random.randint(0, obs_ch[0]-1) for samp in range(params_init.shape[0])]
    torch.save(torch.tensor(list_d0), f"Generated_Datasets/Case_0/list_d0_sampling{obs_ch[0]}d.pt")

## Observation matrix H (sampling): mask with 1 = observed data, 0 otherwise. Constructed from space/temporal sampling info and a lag randomly (uniform) picked between 0 and the lowest sampling frequency.
H = torch.zeros([N_sample, len(obs_ch), window_T, len(z_interpo)], device = devices[-1])
for samp in range(N_sample) :
    for i_ch in range(len(obs_ch)) :
        if obs_ch[i_ch] :
            pattern_t = torch.tensor([obs_ch[i_ch]*int((i-list_d0[samp])%obs_ch[i_ch] == 0) for i in range(window_T)])[:, None]
            pattern_z = torch.zeros([1, Nz])
            pattern_z[:, i_z_ch[i_ch]] = 1/len(i_z_ch[i_ch])
            H[samp, i_ch] = pattern_t*pattern_z
H = H.reshape([N_sample, 5, int(window_T/subwindow_T), subwindow_T, len(z_interpo)])

## Creates NNPZD model with simulation/assimilation space
Model_train = Model_NNPZD_1D(params_0.clone().to(devices[-1]), torch.arange(0, 365*3, (30/60)/24).to(devices[-1]), z_grid.to(devices[-1]), zw_grid.to(devices[-1]), Kz.to(devices[-1]), I0.to(devices[-1]))

## Initial states = mean state if no information, obs otherwise. And adds noise
States_target = torch.load(f"Generated_Datasets/Case_{case}/States.pt", weights_only=False, map_location=devices[-1])[-100:,:,-335:-215]
## Adding noise to observed states and imposing positive values
States_target += torch.cat(([torch.normal(0., obs_noise_std[ch]*torch.ones([N_sample, 1, window_T, States_target.shape[-1]])) for ch in range(5)]), dim = 1).to(devices[-1])
States_target = torch.max(States_target, torch.tensor([0.]))
States_target = States_target.reshape([N_sample, 5, int(window_T/subwindow_T), subwindow_T, Nz])

States0 = torch.zeros([N_sample, 5, int(window_T/subwindow_T), Nz])
for sample in range(N_sample) :
    for ch in range(5) :
        if obs_ch[ch] and list_d0[sample] == 0 :
            States0[sample, ch] = linear_interpolate_1d(z_grid.cpu(), z_grid[i_z_ch[ch]].cpu(), States_target[sample, ch, :, 0, i_z_ch[ch]].cpu()).to(devices[-1])
        elif obs_ch[ch] :
            sample_zinterpo = linear_interpolate_1d(z_grid.cpu(), z_grid[i_z_ch[ch]].cpu(), States_target[sample, ch, :, :, i_z_ch[ch]].cpu()).to(devices[-1]) # dims = 12 x 10 x 34
            sample_zinterpo = torch.cat([sample_zinterpo[win] for win in range(int(window_T/subwindow_T))], dim = 0) # dims = 120 x 34
            sample_ztinterpo = linear_interpolate_1d(torch.arange(0, window_T, 1), torch.arange(list_d0[sample], window_T, obs_ch[ch]), torch.cat([sample_zinterpo[None, frame_t] for frame_t in range(list_d0[sample], window_T, obs_ch[ch])], dim=0).moveaxis(0,1)).moveaxis(1, 0)
            States0[sample, ch] = torch.cat([sample_ztinterpo[None, subwindow_T*win] for win in range(int(window_T/subwindow_T))], dim=0)
        else :
            States0[:, ch] = mean_states[ch]


if NNbase == True :
    States_predNN = torch.load(path_save_NN_save+"States_pred_ens.pt", weights_only=False, map_location=devices[-1])[nb_version]
    States0 = torch.cat([States_predNN[:, :, t, None] for t in range(0, window_T, subwindow_T)], dim = 2)
States0 = torch.max(States0, torch.tensor([0.])).moveaxis(1, 0)

with torch.no_grad() :
    Model_train.NO3_0 = States0[0].clone().requires_grad_(True)
    Model_train.NH4_0 = States0[1].clone().requires_grad_(True)
    Model_train.P_0 = States0[2].clone().requires_grad_(True)
    Model_train.Z_0 = States0[3].clone().requires_grad_(True)
    Model_train.D_0 = States0[4].clone().requires_grad_(True)
Model_train.train()

optim = torch.optim.Adam([{'params': Model_train.alpha, 'lr':mean_params[0]*lr}, {'params': Model_train.Xi, 'lr':mean_params[1]*lr}, {'params': Model_train.rho, 'lr':mean_params[2]*lr}, {'params': Model_train.gamma, 'lr':mean_params[3]*lr}, {'params': Model_train.Gamma, 'lr':mean_params[4]*lr}, {'params': Model_train.varphi, 'lr':mean_params[5]*lr}, {'params': Model_train.omega, 'lr':mean_params[6]*lr}, {'params': Model_train.beta, 'lr': mean_params[7]*lr}, {'params':Model_train.NO3_0, 'lr':lr_x0*States0[:, 0].std().item()}, {'params':Model_train.NH4_0, 'lr':lr_x0*States0[:, 1].std().item()}, {'params':Model_train.P_0, 'lr':lr_x0*States0[:, 2].std().item()}, {'params':Model_train.Z_0, 'lr':lr_x0*States0[:, 3].std().item()}, {'params':Model_train.D_0, 'lr':lr_x0*States0[:, 4].std().item()}], lr=lr, foreach=False) 
Model_train.to(devices[-1])
Model_train.device = devices[-1]

print("\nData loaded/generated")

#############################################################
####################### ASSIMILATION ########################
#############################################################

cost_visu = []
params_visu = torch.zeros([1, N_sample, 8])
iparam = 0
for param in Model_train.parameters() :
    params_visu[:, :, iparam] = param.detach()
    iparam += 1
States0_visu = States0.moveaxis(0, 1).detach()[None]

t_all = time.time()
t_step = time.time()
for step in range(totalstep) :
    States_pred, Itz_pred, Kzmodel_pred = Model_train(sub_windows = assim_subwindows)
    cost, cost_bg = Model_train.compute_cost(States_pred, States_pred[:, :, :, :-1, i_interpo]*H, States_target[:, :, :, :, i_interpo]*H, obs_noise_std)
    
    cost_visu.append([cost.detach(), cost_bg.detach()])

    optim.zero_grad()
    (cost+w_bg*cost_bg).backward()
    optim.step()

    param_save = torch.zeros(params_visu[0].shape)
    with torch.no_grad() :
        iparam = 0
        for param in Model_train.parameters() :
            if iparam <= 7 :
                # param[:].clamp_(min = mean_params[iparam].to(devices[-1])*(1-prct_err_theta), max = mean_params[iparam].to(devices[-1])*(1+prct_err_theta))
                param_save[:, iparam] = param.detach().to(devices[1])
            else :
                # param[:].clamp_(min = 0)
                param_save[:, iparam] = param.detach().to(devices[1])
            iparam += 1
        Model_train.NO3_0.clamp_(min = 0)
        Model_train.NH4_0.clamp_(min = 0)
        Model_train.P_0.clamp_(min = 0)
        Model_train.Z_0.clamp_(min = 0)
        Model_train.D_0.clamp_(min = 0)
        # Y0_visu = torch.cat((Y0_visu, torch.cat((Model_train.NO3_0[:, None], Model_train.NH4_0[:, None], Model_train.P_0[:, None], Model_train.Z_0[:, None], Model_train.D_0[:, None]), dim = 1)[:, :, :, i_interpo].to(Model_train.device).detach()[None]), dim = 0)
        States0_visu = torch.cat((States0_visu, torch.cat((Model_train.NO3_0[:, None], Model_train.NH4_0[:, None], Model_train.P_0[:, None], Model_train.Z_0[:, None], Model_train.D_0[:, None]), dim = 1).detach()[None]), dim=0)
        params_visu = torch.cat((params_visu, param_save[None]), dim = 0)
    if torch.isnan(torch.max(params_visu[-1])) or torch.isnan(torch.max(States0_visu[-1])) :
        print(f"Nan in params, stop")
        break
    print(f"Step {step}/{totalstep} - cost = {cost.detach():.2e}/{cost_bg.detach():.2e} - Fin dans {datetime.timedelta(seconds=(totalstep-step-1)*round(time.time()-t_step))}s.")
    t_step = time.time()


## Save used data
torch.save(Kz, path_save+"Kz.pt")
torch.save(I0, path_save+"I0.pt")
torch.save(params_ref, path_save+"Params_ref.pt")
torch.save(params_0, path_save+"Params_init.pt")
torch.save(params_visu, path_save+"Params_visu.pt")
torch.save(cost_visu, path_save+"cost_visu.pt")
torch.save(States0_visu, path_save+"States0_visu.pt")
torch.save(States_target, path_save+"States_noised.pt")
torch.save(list_d0, path_save+"list_d0.pt")
torch.save(States_pred, path_save+"States_pred.pt")
print(f"\nDA done in {datetime.timedelta(seconds=round(time.time()-t_all))}s.")

########################################################################
####################### PLOT cost & PARAMETERS #########################
########################################################################

fig = plt.figure(figsize = [6.4*1.5, 4.8*0.7])
plt.semilogy(torch.tensor(cost_visu)[:, 0].cpu(), label = 'Error obs')
plt.semilogy(torch.tensor(cost_visu)[:, 1].cpu(), label = 'Error mod')
plt.grid()
plt.legend()
plt.xlabel("Nb step")
plt.ylabel("cost value")
fig.savefig(path_save+"Plot_cost", dpi = 300)


fig, ax = plt.subplots(figsize = [6.4*1.5*2, 4.8*0.7*5], ncols = 2, nrows = 4)
for i in range(8) :
    ax[i%4, i//4].axhline(y = 0.4, color = 'k', ls = '--')
    ax[i%4, i//4].semilogy(torch.mean(abs(params_visu[:, :, i]-params_ref[:, i])/params_ref[:, i], dim = 1).cpu())
    ax[i%4, i//4].set_title(["alpha", "Xi", "rho", "gamma", "Gamma", "phi", "omega", "beta"][i])
    ax[i%4, i//4].grid()
fig.tight_layout()
fig.savefig(path_save+"Plot_params", dpi = 500)
