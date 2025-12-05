"""
Last update on December 2025

@author: jlittaye
Python 3.11.5
"""

## Import packages, functions
import os
from os import listdir
import datetime
from math import *
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import CSVLogger

from model_file import UNet_param_forcing_state
from func_file import Model_NNPZD_1D, init_NNPZD, linear_interpolate_1d, get_topcorner, get_rightcorner, metrics_BGC, load_best_model, data_sampling, data_sampling_rdmshift, data_sampling_rdmshift_ens

global devices
devices = ['cpu']+[f"cuda:{idevice}" for idevice in range(torch.cuda.device_count())]
torch.set_default_device(devices[-1])
print("Libs imported, device=", devices[-1])


## Scenario configuration
case = 1
obs_ch = [10]*5
obs_strat = 1

if obs_strat == 0 : # We observe the 5 states at every depth
    i_z_ch = [[i for i in range(34)], [i for i in range(34)], [i for i in range(34)], [i for i in range(34)], [i for i in range(34)]]
elif obs_strat == 1 : # CTD+rosette: NO3/P/D observed continuously, NH4/Z observed discretely
    i_z_ch = [[i for i in range(34)], [5, 11, 15, 20, 24, 27, 33], [i for i in range(34)], [5, 11, 15, 20, 24, 27, 33], [i for i in range(34)]]
elif obs_strat == 2 : # CTD: NO3/P/D observed continuously
    i_z_ch = [[i for i in range(34)], [], [i for i in range(34)], [], [i for i in range(34)]]
elif obs_strat == 3 : # Rosette: NO3/NH4/P/Z/D observed discretely
    i_z_ch = [[5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33], [5, 11, 15, 20, 24, 27, 33]]
for ch in range(len(obs_ch)) :
    if len(i_z_ch[ch]) == 0 :
        obs_ch[ch] = 0

## Training configuration
nb_epoch = 500
lr = 1e-3
wforcing = 10
wparam = 5
wstate = 10

path_save = f"Res/NN/Case{case}_1sample{obs_ch[0]}days_strat{obs_strat}/"
if not os.path.isdir(path_save) :
    os.makedirs(path_save)
nb_version = 0
for file in os.listdir(path_save) :
    if file[:8] == "Version_" :
        nb_version += 1
path_save += f"Version_{nb_version}/"
os.makedirs(path_save)

print("File: ", path_save, "with machines:", devices)

## Import variables: datasets, mean/std
DS_train = data_sampling_rdmshift(torch.load(f"Generated_Datasets/Case_{case}/DS_train", weights_only=False), obs_ch, i_z_ch, dt=1/2)
DS_valid = data_sampling_rdmshift(torch.load(f"Generated_Datasets/Case_{case}/DS_valid", weights_only=False), obs_ch, i_z_ch, dt=1/2)
DL_train = DataLoader(DS_train, batch_size=256, shuffle=True, generator = torch.Generator(devices[-1]))
DL_valid = DataLoader(DS_valid, batch_size=256, shuffle=True, generator = torch.Generator(devices[-1]))

params_mean, params_std = torch.load("Generated_Datasets/Case_0/Params_mean.pt", weights_only=False, map_location=devices[0]), torch.load("Generated_Datasets/Case_0/Params_std.pt", weights_only=False, map_location=devices[0])
Kzlog10_mean, Kzlog10_std = torch.load("Generated_Datasets/Case_0/Kzlog10_mean.pt", weights_only=False, map_location=devices[0]), torch.load("Generated_Datasets/Case_0/Kzlog10_std.pt", weights_only=False, map_location=devices[0])
I0_mean, I0_std = torch.load("Generated_Datasets/Case_0/I0_mean.pt", weights_only=False, map_location=devices[0]), torch.load("Generated_Datasets/Case_0/I0_std.pt", weights_only=False, map_location=devices[0])
State_mean, State_std = torch.load("Generated_Datasets/Case_0/States_mean.pt", weights_only=False, map_location=devices[0]), torch.load("Generated_Datasets/Case_0/States_std.pt", weights_only=False, map_location=devices[0])
output_mean, output_std = torch.cat((params_mean, Kzlog10_mean[None], I0_mean[None], State_mean)).to(devices[-1]), torch.cat((params_std, Kzlog10_std[None], I0_std[None], State_std)).to(devices[-1])

forcing_path = 'FORCING_40km/'
zmax = 350 # maximum depth studied
Nz = np.argmin(np.load(forcing_path+"depth.npy")[0] < zmax) # number of layers between the surface and 350m    
z_grid = torch.from_numpy(np.load(forcing_path+"depth.npy")[0, :Nz])
zw_grid = torch.from_numpy(np.load(forcing_path+"depthw.npy")[0, :Nz+1])


My_UNet = UNet_param_forcing_state(sampling_patt_t=obs_ch, i_z_ch=i_z_ch, z_grid=z_grid.to(devices[-1]), lr=lr)
My_UNet.to(devices[-1])
My_UNet.wparam = wparam
My_UNet.wforcing = wforcing
My_UNet.wstate = wstate
My_UNet.zw_grid = zw_grid.to(devices[-1])

## Checkpoints and logger
checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="valid_loss", mode="min", dirpath=path_save+f"top_10/", filename="chkpt_{epoch:02d}")
checkpoint_callback_2 = ModelCheckpoint(every_n_epochs = 50, save_top_k = -1, monitor="valid_loss", dirpath=path_save+f"every_n_epochs/", filename="chkpt_{epoch:02d}")
epoch_i = 0
logger = CSVLogger(save_dir = path_save, name = 'lightning_logs', version = 0)


#################################################
################### TRAINING ####################
#################################################
print("Start training\n")
trainer = pl.Trainer(check_val_every_n_epoch=10, default_root_dir = path_save, min_epochs = epoch_i+nb_epoch, max_epochs=epoch_i+nb_epoch, callbacks=[checkpoint_callback, checkpoint_callback_2], log_every_n_steps=None, logger = logger)
ti_train = time.time()
trainer.fit(model=My_UNet, train_dataloaders=DL_train, val_dataloaders=DL_valid)
print(f"Training ends after {datetime.timedelta(seconds=round(time.time()-ti_train))}s.")

trainer.save_checkpoint(path_save+f"final_chkpt.ckpt")
torch.save(My_UNet.state_dict(), path_save+"final_state_dict")


#######################################################
################### PLOTS training ####################
#######################################################
logs = pd.read_csv(path_save+f"lightning_logs/version_0/metrics.csv", header = 0, index_col = 'epoch')

valid_loss = logs[np.isnan(logs["valid_loss"]) == False]["valid_loss"]
training_loss = logs[np.isnan(logs["train_loss_epoch"]) == False]["train_loss_epoch"]
valid_loss_params = logs[np.isnan(logs["valid_loss_param"]) == False]["valid_loss_param"]
training_loss_params = logs[np.isnan(logs["train_loss_param_epoch"]) == False]["train_loss_param_epoch"]
valid_loss_forcing = logs[np.isnan(logs["valid_loss_forcing"]) == False]["valid_loss_forcing"]
training_loss_forcing = logs[np.isnan(logs["train_loss_forcing_epoch"]) == False]["train_loss_forcing_epoch"]
valid_loss_states = logs[np.isnan(logs["valid_loss_state"]) == False]["valid_loss_state"]
training_loss_states = logs[np.isnan(logs["train_loss_state_epoch"]) == False]["train_loss_state_epoch"]

prct_states = [logs[np.isnan(logs["valid_prct_"+ch]) == False]["valid_prct_"+ch] for ch in ["NO3", "NH4", "P", "Z", "D"]]
prct_forcings = [logs[np.isnan(logs["valid_prct_"+ch]) == False]["valid_prct_"+ch] for ch in ["Kz", "I0"]]

## Plot losses
fig, ax = plt.subplots(figsize = [6.4*1.5, 4.8*0.5*4], nrows = 4)
ax[0].semilogy(training_loss, label = 'Training')
ax[0].semilogy(valid_loss, label = 'Validation')
ax[1].semilogy(training_loss_params, label = 'Training')
ax[1].semilogy(valid_loss_params, label = 'Validation')
ax[2].semilogy(training_loss_forcing, label = 'Training forcing')
ax[2].semilogy(valid_loss_forcing, label = 'Validation forcing')
ax[3].semilogy(training_loss_states, label = 'Training')
ax[3].semilogy(valid_loss_states, label = 'Validation')
for i in range(4) :
    ax[i].set_xlabel("epoch")
    ax[i].set_ylabel("Loss value")
    ax[i].set_title(["Global", "BGC parameter", "Forcing", "BGC state"][i]+f" loss evolution, weight={[1, 1, wforcing, wstate][i]}")
    ax[i].legend()
    ax[i].grid()
plt.tight_layout()
plt.savefig(path_save+"Loss_plot", dpi = 200)

## Plot Reconstructed data errors
fig, ax = plt.subplots(figsize = [6.4*1.5, 4.8*0.5*2], nrows = 2)
for ch in range(5) :
    ax[0].semilogy(prct_states[ch], label = ["NO3", "NH4", "P", "Z", "D"][ch])
ax[0].axhline(y=100, ls = '--', color = 'k')
for ch in range(2) :
    ax[1].semilogy(prct_forcings[ch], label = ["Kz", "I0"][ch])
ax[1].axhline(y=100, ls = '--', color = 'k')
for i in range(2) :
    ax[i].set_xlabel("epoch")
    ax[i].set_ylabel("%")
    ax[i].set_title(f" Percentage of residual error for the "+["BGC states", "forcings"][i])
    ax[i].legend()
    ax[i].grid()
plt.tight_layout()
plt.savefig(path_save+"Prct_error_plot", dpi = 200)

print("Plot loss done")

######################################################
###################### TESTING #######################
######################################################

params_init = torch.load(f"Generated_Datasets/Case_{case}/Params.pt", weights_only=False, map_location=devices[-1])[1, -100:].clone()
if os.path.exists(f"Generated_Datasets/Case_0/list_d0_sampling{obs_ch[0]}d.pt") :
    list_d0 = torch.load(f"Generated_Datasets/Case_0/list_d0_sampling{obs_ch[0]}d.pt", weights_only=False, map_location=devices[-1])
else :
    list_d0 = [random.randint(0, obs_ch[0]-1) for samp in range(params_init.shape[0])]
    torch.save(torch.tensor(list_d0), f"Generated_Datasets/Case_0/list_d0_sampling{obs_ch[0]}d.pt")
DS_test = data_sampling_rdmshift_ens(torch.load(f"Generated_Datasets/Case_{case}/DS_test_ens", weights_only=False, map_location=devices[-1]), obs_ch, i_z_ch, dt=1/2, list_firstobsday=list_d0)
DL_test = DataLoader(DS_test, batch_size=100, shuffle=False)

chkpt = load_best_model(path_save, criterion = "valid_loss")
path_save_pred = path_save+"pred_var/"
if not os.path.isdir(path_save_pred) :
    os.makedirs(path_save_pred)
print(f"Loading file: {chkpt}")

My_UNet_test = UNet_param_forcing_state.load_from_checkpoint(chkpt, sampling_patt_t=obs_ch, i_z_ch=i_z_ch, z_grid=z_grid.to(devices[-1]), lr=lr).to(devices[-1])
# My_UNet_test.device = devices[-1]
My_UNet_test.to(devices[-1])
My_UNet_test.eval()
params, forcings, states, states_init = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
for x, y in DL_test :
    for m in range(x.shape[0]) :
        pred_param, pred_forcing, pred_state = My_UNet_test(x[m])
        params = torch.cat((params, pred_param.detach()[None]), dim=0)
        forcings = torch.cat((forcings, pred_forcing.detach()[None]), dim=0)
        states = torch.cat((states, pred_state.detach()[None]), dim=0)
        states_init = torch.cat((states_init, My_UNet_test(x[0], linear=True)[:, :5, :, :-1].detach()[None]), dim=0)

params_res = params*output_std[:8]+output_mean[:8]
Kz_res = (forcings[:, :, 0]*output_std[8]+output_mean[8])
I0_res = (forcings[:, :, 1, :, 0].unfold(2, 1, 2).flatten(2)*output_std[9]+output_mean[9])
Kz_res = 10**Kz_res
states_res = (states.unfold(3, 1, 2).flatten(4).moveaxis(2,4)*output_std[10:]+output_mean[10:]).moveaxis(4,2)
states_init_res = (states_init.unfold(3, 1, 2).flatten(4).moveaxis(2,4)*output_std[10:]+output_mean[10:]).moveaxis(4,2)

torch.save(params_res.clone().cpu(), path_save_pred+f"Params_pred_ens.pt")
torch.save(Kz_res.clone().cpu(), path_save_pred+f"Kz_pred_ens.pt")
torch.save(I0_res.clone().cpu(), path_save_pred+f"I0_pred_ens.pt")
torch.save(states_res.clone().cpu(), path_save_pred+f"States_pred_ens.pt")
torch.save(states_init_res.clone().cpu(), path_save_pred+f"States_init.pt")
print(f"Loss params: {torch.mean((params.detach()-y[:, :, :8, 0, 0].detach())**2)}\nLoss forcings: {torch.mean((forcings.detach()-y[:, :, 8:10].detach())**2)}\nLoss state: {torch.mean((states.detach()-y[:, :, 10:, :, :-1].detach())**2)}")
print(f"Initial error params: {torch.mean((((params_init-output_mean[:8])/output_std[:8])-y[:, :, :8, 0, 0].detach())**2)}\nInitial error forcings: {torch.mean((x[:, :, 5:7].detach()-y[:, :, 8:10].detach())**2)}\nInitial error state: {torch.mean((states_init-y[:, :, 10:, :, :-1].detach())**2)}")


## To reconstruct states from the mean estimated parameters
zmax = 350
Nz = np.argmin(np.load("FORCING_40km/depth.npy")[0] < zmax)
z_ref = torch.from_numpy(np.load("FORCING_40km/depth.npy")[0, :Nz]).to(devices[-1])
zw_ref = torch.from_numpy(np.load("FORCING_40km/depthw.npy")[0, :Nz+1]).to(devices[-1])

I0_test = torch.load(f"Generated_Datasets/Case_{case}/I0.pt", weights_only=False, map_location=devices[-1])[0, -100:]
Kz_test = torch.load(f"Generated_Datasets/Case_{case}/Kz.pt", weights_only=False, map_location=devices[-1])[0, -100:]
States_ref = torch.load(f"Generated_Datasets/Case_{case}/States.pt", weights_only=False, map_location=devices[-1])[-100:]

Model_BGC = Model_NNPZD_1D(params_res.mean(dim=0).clone().to(devices[-1]), torch.arange(0, 3*365, (30/60)/24), z_ref.to(devices[-1]), zw_ref.to(devices[-1]), Kz_test.repeat(1, 3, 1), I0_test.repeat(1, 3))
Model_BGC.eval()
Model_BGC.to(devices[-1])
Model_BGC.device = devices[-1]
with torch.no_grad() :
    Model_BGC.NO3_0 = States_ref[:, 0, 0, None].clone().to(devices[-1])
    Model_BGC.NH4_0 = States_ref[:, 1, 0, None].clone().to(devices[-1])
    Model_BGC.P_0 = States_ref[:, 2, 0, None].clone().to(devices[-1])
    Model_BGC.Z_0 = States_ref[:, 3, 0, None].clone().to(devices[-1])
    Model_BGC.D_0 = States_ref[:, 4, 0, None].clone().to(devices[-1])
    States, Itz, Kz = Model_BGC(sub_windows = torch.arange(0, 365*3, 1)[None], info_time=True)

torch.save(States.clone()[:, :, 0].cpu(), path_save_pred+f"States_reconstruct.pt")