"""
Last update on December 2025

@author: jlittaye
Python 3.11.5
"""
import time
from math import *
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import CSVLogger

from func_file import linear_interpolate_1d



class conv_block_1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x



class encoder_block_1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block_1D(in_c, out_c)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p



class decoder_block_1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block_1D(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x



class UNet_param_forcing_state(pl.LightningModule): # UNet_theta_phi35z_state_v3
    """ UNet to calibrate BGC parameters from observed states and forcings.
    - Input [N_sample, N_ch, N_t, N_z]: observed states, where unobserved data = zeros or nans; and physical forcing with uncertainties. Kz has 1/2 day time sampling whereas states and I0 are upsampled (doubled).
    - Output: parameters[N_sample, N_params], states[N_sample, N_ch, N_t, N_z], forcings[N_sample, 2, 2*N_t, N_z+1] estimated parameters, states and corrected forcings Kz (defined over 2*N_t and N_z+1) and I0 (defined over N_t but upsampled and repeated over N_z+1).
    """
    def __init__(self, sampling_patt_t, i_z_ch, z_grid, lr):
        super().__init__()
        self.sampling_patt_t = sampling_patt_t # time sampling of observed states
        self.i_z_ch = i_z_ch # vertical space sampling of observed states
        self.nb_depth = torch.cat([torch.tensor(i_z_ch[i]) for i in range(len(i_z_ch))]).shape[0]
        self.dt=1/2  # time sampling of Kz
        self.z_grid = z_grid  # vertical space of states
        self.lr=lr # learning rate
        
        ch_in = 2+torch.sum(torch.tensor(self.sampling_patt_t) > 0).item()
        self.layer_e1 = encoder_block_1D(ch_in*(len(self.z_grid)+1), 16)
        self.layer_e2 = encoder_block_1D(16, 32)
        self.layer_b = conv_block_1D(32, 64)
        self.layer_d1 = decoder_block_1D(64, 32)
        self.layer_d2 = decoder_block_1D(32, 16)
        self.layer_avgpooltheta = nn.AvgPool1d((240))
        self.dense1theta = nn.Linear(16, 8)

        self.conv1 = nn.Conv1d(16, 32, kernel_size = 5, padding = 2)
        self.conv2forcing = nn.Conv1d(32, 2*(len(self.z_grid)+1), kernel_size = 5, padding = 2) #7 because 2 forcings, 5 states
        self.conv2state = nn.Conv1d(32, 5*len(self.z_grid), kernel_size = 5, padding = 2) #7 because 2 forcings, 5 states

    
    def forward(self, x, linear=False):
        x_interpo = x.clone()

        pattern_t = [int(dt_ch/self.dt) for dt_ch in self.sampling_patt_t]
        # list_d0 = [(torch.isnan(x[sample, 0, :, 0])*1).argmin().item() for sample in range(x.shape[0])]
        for ch in range(5) : 
            if self.sampling_patt_t[ch] :
                if len(self.i_z_ch[ch]) < x.shape[-1] : # interpo in depth
                    x_interpo[:, ch, :, :-1] = linear_interpolate_1d(self.z_grid, self.z_grid[self.i_z_ch[ch]], x_interpo[:, ch, :, self.i_z_ch[ch]])
                x_interpo[:, ch, :, -1] = x_interpo[:, ch, :, -2]
                list_d0 = [(torch.isnan(x_interpo[sample, ch, :, 0])*1).argmin().item() for sample in range(x.shape[0])]
                
                for sample in range(x.shape[0]) : # interpo in time
                    if list_d0[sample] : # if the first observed day is not the first day of the time serie
                        x_interpo[sample, ch] = linear_interpolate_1d(torch.arange(0, x.shape[2], 1), torch.arange(list_d0[sample], 240, pattern_t[ch]), x_interpo[sample, ch, list_d0[sample]:].unfold(0, 1, pattern_t[ch])[:, :, 0].moveaxis(0, 1)).moveaxis(1, 0)
                    else :
                        x_interpo[sample, ch] = linear_interpolate_1d(torch.arange(0, x.shape[2], 1), torch.arange(0, 240, pattern_t[ch]), x_interpo[sample, ch].unfold(0, 1, pattern_t[ch])[:, :, 0].moveaxis(0, 1)).moveaxis(1, 0)

        y_base = x_interpo.clone()
        y_base[:, torch.where(torch.tensor(self.sampling_patt_t)==0)[0]] = 0.

        if linear :
            x_interpo[:, torch.where(torch.tensor(self.sampling_patt_t)==0)[0]] = 0.
            return x_interpo
        else :
            x_interpo = x_interpo[:, torch.cat((torch.where(torch.tensor(self.sampling_patt_t)>0)[0], torch.tensor([-2, -1])))] ## we remove unobserved states
            x_interpo = torch.cat([x_interpo[:, :, :, i] for i in range(x_interpo.shape[3])], dim = 1)
            """ Encoder """
            s1, p1 = self.layer_e1(x_interpo)
            s2, p2 = self.layer_e2(p1)   
            """ Bottleneck """
            b = self.layer_b(p2)
            """ Decoder """
            d1 = self.layer_d1(b, s2)
            d2 = self.layer_d2(d1, s1)
    
            """ We split theta/forcing/states """
            y1 = torch.flatten(self.layer_avgpooltheta(d2), 1)
            y1 = self.dense1theta(y1)
            y_ = F.relu(self.conv1(d2))
            y2 = self.conv2forcing(y_)
            y2 = torch.cat([y2[:, 2*layer:2*(layer+1), :, None] for layer in range(len(self.z_grid)+1)], dim=-1)
            y3 = self.conv2state(y_)
            y3 = torch.cat([y3[:, 5*layer:5*(layer+1), :, None] for layer in range(len(self.z_grid))], dim=-1)
            return y1, y_base[:, 5:] + y2, y_base[:, :5, :, :-1] + y3 # params, forcings, states


    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def mse_loss(self, predictions, targets, only_obs_states = False):
        """ Compute MSE loss.
        - only_obs_states: True = computes MSE only on observed states
        """
        if only_obs_states :
            mean = torch.zeros([1])
            for ch in range(5) :
                if self.sampling_patt_t[ch] :
                    mean += torch.mean(((predictions-targets)[:, ch, :, i_z_obs[ch]])**2)
            return torch.mean((diff)**2)
        else :
            return torch.mean((predictions-targets)**2)

    def training_step(self, x):
        ti = time.time()
        x_train = x[0]
        y_train_param = x[1][:, :8, 0, 0]
        y_train_forcing = x[1][:, 8:10]
        y_train_state = x[1][:, 10:15, :, :-1].unfold(2, 1, round(1/self.dt))[:, :, :, :, 0]
        
        pred_param, pred_forcing, pred_state = self(x_train)
        pred_state = pred_state.unfold(2, 1, round(1/self.dt))[:, :, :, :, 0]
        
        loss_param = self.mse_loss(pred_param, y_train_param)
        loss_forcing = self.mse_loss(pred_forcing, y_train_forcing)
        loss_state = self.mse_loss(pred_state, y_train_state)
        
        self.log('train_loss', loss_param+loss_forcing*self.wforcing+loss_state*self.wstate, on_epoch=True)
        self.log('train_loss_param', loss_param, on_epoch=True)
        self.log('train_loss_forcing', loss_forcing, on_epoch=True)
        self.log('train_loss_state', loss_state, on_epoch=True)
        
        for ch in range(2) :
            self.log(f'train_prct_'+['Kz','I0'][ch], 100*(torch.mean((pred_forcing-y_train_forcing)[:, ch]**2)/torch.mean((x_train[:, 5+ch]-y_train_forcing[:, ch])**2)), on_epoch=True)

        for ch in range(5) :
            if self.sampling_patt_t[ch] :
                prct = 100*(torch.mean((pred_state-y_train_state)[:, ch, :, self.i_z_ch[ch]]**2)/torch.mean((x_train[:, ch, :, :-1].unfold(1, 1, round(1/self.dt))[:, :, self.i_z_ch[ch], 0]-y_train_state[:, ch, :, self.i_z_ch[ch]])**2))
                self.log(f'train_prct_'+['NO3','NH4','P','Z','D'][ch], prct, on_epoch=True)

        if torch.isnan(loss_param) :
            print(f"Nan in param, loss = {loss_param.detach()}, Pred=", pred_param)
        if torch.isnan(loss_forcing) :
            print(f"Nan in forcing, loss = {loss_forcing.detach()}, Pred=", pred_forcing)
        if torch.isnan(loss_state) :
            print(f"Nan in state, loss = {loss_state.detach()}, Pred=", pred_state)
        return loss_param + loss_forcing*self.wforcing + loss_state*self.wstate

    def validation_step(self, x):
        x_valid = x[0]
        y_valid_param = x[1][:, :8, 0, 0]
        y_valid_forcing = x[1][:, 8:10]
        y_valid_state = x[1][:, 10:15, :, :-1].unfold(2, 1, round(1/self.dt))[:, :, :, :, 0]
        pred_param, pred_forcing, pred_state = self(x_valid)
        pred_state = pred_state.unfold(2, 1, round(1/self.dt))[:, :, :, :, 0]
        
        loss_param = self.mse_loss(pred_param, y_valid_param)
        loss_forcing = self.mse_loss(pred_forcing, y_valid_forcing)
        loss_state = self.mse_loss(pred_state, y_valid_state)
        
        self.log('valid_loss', loss_param+loss_forcing*self.wforcing+loss_state*self.wstate, on_epoch=True)
        self.log('valid_loss_param', loss_param, on_epoch=True, prog_bar=True)
        self.log('valid_loss_forcing', loss_forcing, on_epoch=True, prog_bar=True)
        self.log('valid_loss_state', loss_state, on_epoch=True, prog_bar=True)
        
        for ch in range(2) :
            self.log(f'valid_prct_'+['Kz','I0'][ch], 100*(torch.mean((pred_forcing-y_valid_forcing)[:, ch]**2)/torch.mean((x_valid[:, 5+ch]-y_valid_forcing[:, ch])**2)), on_epoch=True)

        for ch in range(5) :
            self.log(f'valid_prct_'+['NO3','NH4','P','Z','D'][ch], 100*(torch.mean((pred_state-y_valid_state)[:, ch]**2)/torch.mean((x_valid[:, ch, :, :-1].unfold(1, 1, round(1/self.dt))[:, :, :, 0]-y_valid_state[:, ch])**2)), on_epoch=True)
        
        if torch.isnan(loss_param) :
            print(f"Nan in param, loss = {loss_param.detach()}, Pred=", pred_param)
        if torch.isnan(loss_forcing) :
            print(f"Nan in forcing, loss = {loss_forcing.detach()}, Pred=", pred_forcing)
        if torch.isnan(loss_state) :
            print(f"Nan in state, loss = {loss_state.detach()}, Pred=", pred_state)
        return loss_param + self.wforcing*loss_forcing + self.wstate*loss_state


