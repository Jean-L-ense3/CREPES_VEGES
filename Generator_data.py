"""
Last update on December 2025

@author: jlittaye
Python 3.11.5
"""

## Packages and functions
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from func_file import Gen_data, select_forcing_gen, linear_interpolate_1d

## List devices (0: CPU, 1+: GPU if available)
devices = ['cpu']+[f"cuda:{idevice}" for idevice in range(torch.cuda.device_count())]
torch.set_default_device(devices[1])

obs_noise_std = torch.tensor([0.04, 0.001, 0.08, 0.12, 0.04])/2
nb_subset, nb_sample = 2, 21 #10, 310
train_size, valid_size, test_size = 30, 10, 2 # 2000, 1000, 100

for case in range(4) :
    path_save = f"Generated_Datasets_test/"
    print(f"Generation of dataset: ", path_save+f"Case_{case}/")
    if not case :
        Gen_data(path_save+f"Case_{case}/", nb_subset=nb_subset, nb_sample=nb_sample, case=case, save_stats = True, device=devices[-1])
        std_Kzlog10, mean_Kzlog10 = torch.load(path_save+f"Case_{case}/Kzlog10_std.pt", weights_only=False, map_location=devices[-1]), torch.load(path_save+f"Case_{case}/Kzlog10_mean.pt", weights_only=False, map_location=devices[-1])
        std_I0, mean_I0 = torch.load(path_save+f"Case_{case}/I0_std.pt", weights_only=False, map_location=devices[-1]), torch.load(path_save+f"Case_{case}/I0_mean.pt", weights_only=False, map_location=devices[-1])
        std_States, mean_States = torch.load(path_save+f"Case_{case}/States_std.pt", weights_only=False, map_location=devices[-1]), torch.load(path_save+f"Case_{case}/States_mean.pt", weights_only=False, map_location=devices[-1])
        std_params, mean_params = torch.load(path_save+f"Case_{case}/Params_std.pt", weights_only=False, map_location=devices[-1]), torch.load(path_save+f"Case_{case}/Params_mean.pt", weights_only=False, map_location=devices[-1])
    else :
        Gen_data(path_save+f"Case_{case}/", nb_subset=nb_subset, nb_sample=nb_sample, case=case, device=devices[-1])

    ## Select the third year (2y spin-up), 120 days from 1 february
    I0_ref, I0 = torch.load(path_save+f"Case_{case}/I0.pt", weights_only=False, map_location=devices[-1])[:, :, -335:-215].clone()
    Kz_ref, Kz = torch.load(path_save+f"Case_{case}/Kz.pt", weights_only=False, map_location=devices[-1])[:, :, None, -335*2:-215*2].clone()
    Params = torch.load(path_save+f"Case_{case}/Params.pt", weights_only=False, map_location=devices[-1])[0].clone()
    States = torch.load(path_save+f"Case_{case}/States.pt", weights_only=False, map_location=devices[-1])[:, :, -335:-215].clone()
    
    States_noise = States.clone()
    for ch in range(States.shape[1]) :
        States_noise[:, ch] += torch.normal(0., obs_noise_std[ch], [States_noise.shape[0], States_noise.shape[2], States_noise.shape[3]]).to(devices[-1])
    torch.save(States_noise, path_save+f"Case_{case}/States_noised.pt")
    
    ## Upsampling because states are generated every day, Kz is generated every 12h. Then padding to have save input depth dimension (states have Nz layers but Kz has Nz+1)
    States_noise = States_noise.repeat_interleave(2, 2)
    States = States.repeat_interleave(2, 2)
    States_noise = torch.cat((States_noise, torch.zeros([States.shape[0], States.shape[1], States.shape[2], 1])), dim=3)
    States = torch.cat((States, torch.zeros([States.shape[0], States.shape[1], States.shape[2], 1])), dim=3)
    
    ## Same for I0 that is only on surface and available every day
    I0 = I0[:, None, :, None].repeat_interleave(2, 2).repeat(1, 1, 1, Kz_ref.shape[-1])
    I0_ref = I0_ref[:, None, :, None].repeat_interleave(2, 2).repeat(1, 1, 1, Kz_ref.shape[-1])
    
    ## Input: Observed states, uncertain Kz, uncertain I0; and normalization
    X = torch.cat((((States_noise.moveaxis(1, 3)-mean_States)/std_States).moveaxis(3, 1),
              (torch.log10(Kz)-mean_Kzlog10)/std_Kzlog10, (I0-mean_I0)/std_I0), dim=1)
    ## Output: Correct parameters, Kz, I0, states; and normalization
    Y = torch.cat((((Params-mean_params)/std_params)[:, :, None, None].repeat(1, 1, Kz_ref.shape[2], Kz_ref.shape[3]),
                  (torch.log10(Kz_ref)-mean_Kzlog10)/std_Kzlog10, (I0_ref-mean_I0)/std_I0,
                  ((States.moveaxis(1, 3)-mean_States)/std_States).moveaxis(3, 1)), dim = 1)
    ## Create training/validation/test datasets
    DS_train = TensorDataset(X[:train_size].clone(), Y[:train_size].clone())
    DS_valid = TensorDataset(X[train_size:train_size+valid_size].clone(), Y[train_size:train_size+valid_size].clone())
    DS_test = TensorDataset(X[train_size+valid_size:train_size+valid_size+test_size].clone(), Y[train_size+valid_size:train_size+valid_size+test_size].clone())
    
    torch.save(DS_train, path_save+f"Case_{case}/DS_train")
    torch.save(DS_valid, path_save+f"Case_{case}/DS_valid")
    torch.save(DS_test, path_save+f"Case_{case}/DS_test")



## Generate ensembles: 10 forcing realisations for each uncertainty case, for assimilation (=> size of the test dataset)
n_member = 10
for case in range(1, 4) :
    print(f"Generating forcing members for case {case}")
    path_save = f"Generated_Datasets_test/"
    lat_lon = np.array(torch.load(path_save+f"Case_{case}/Latlon.pt", weights_only=False, map_location=devices[0])[-test_size:])
    year_to_sim = np.array(torch.load(path_save+f"Case_{case}/Years.pt", weights_only=False, map_location=devices[0])[-test_size:].to(dtype=torch.int))

    
    forcing_path = 'FORCING_40km/' # Path of profiles generated from polgyr
    zmax = 350 # maximum depth studied
    Nz = np.argmin(np.load(forcing_path+"depth.npy")[0] < zmax) # number of layers between the surface and 350m
    Kz_bank = torch.from_numpy(np.load(forcing_path+"akt.npy")) # Kz profiles from polgyr
    lat_lon_Kz = np.load(forcing_path+"lat_lon_index.npy") # location (lat/lon) of profiles from polgyr
    zw_grid_bank = torch.from_numpy(np.load(forcing_path+"depthw.npy")) # layer depth related to Kz profiles
    
    z_grid = torch.from_numpy(np.load(forcing_path+"depth.npy")[0, :Nz]) # depth of layer centers (where tracers are located), between 0-350 m
    zw_grid = torch.from_numpy(np.load(forcing_path+"depthw.npy")[0, :Nz+1]) # depth of layer borders (where kz is located), between 0-350 m
    Kz_bank_interpo = torch.zeros([Kz_bank.shape[0], Kz_bank.shape[1], Nz+1])
    for i_profile in range(Kz_bank.shape[0]) :
        Kz_bank_interpo[i_profile] = 10**linear_interpolate_1d(zw_grid, zw_grid_bank[i_profile], torch.log10(Kz_bank[i_profile]))

    Kz_batch_ens = np.zeros([2, n_member, test_size, 730, Nz+1])
    I0_batch_ens = np.zeros([2, n_member, test_size, 365])
    lat_lon_ens = np.zeros([n_member, test_size, 6, 730])
    for i_member in range(n_member) :
        for i_sample in range(lat_lon.shape[0]) :
            Kz_interpo_T, Kz_interpo_F, I0_interpo_T, I0_interpo_F, lat_lon_F_fit, lat_lon_F, lat_lon_T = select_forcing_gen(forcing_path, lat_lon[i_sample, 0, 0], lat_lon[i_sample, 1, 0], year_to_sim[i_sample], zw_grid, np.array(Kz_bank_interpo[:, year_to_sim[i_sample]*730:(year_to_sim[i_sample]+1)*730].to(devices[0])), case, clip = True)
            Kz_batch_ens[0, i_member, i_sample] = Kz_interpo_T
            Kz_batch_ens[1, i_member, i_sample] = Kz_interpo_F
            I0_batch_ens[0, i_member, i_sample] = I0_interpo_T
            I0_batch_ens[1, i_member, i_sample] = I0_interpo_F
            lat_lon_ens[i_member, i_sample, :2] = lat_lon_T[:, None]
            lat_lon_ens[i_member, i_sample, 2:4] = lat_lon_F
            lat_lon_ens[i_member, i_sample, 4:] = lat_lon_F_fit
    torch.save(torch.from_numpy(Kz_batch_ens).to(devices[-1]), path_save+f"Case_{case}/Kz_ens.pt")
    torch.save(torch.from_numpy(I0_batch_ens).to(devices[-1]), path_save+f"Case_{case}/I0_ens.pt")
    torch.save(torch.from_numpy(lat_lon_ens).to(devices[-1]), path_save+f"Case_{case}/Latlon_ens.pt")

    ## Generates test_dataset with ensemble forcing
    std_Kzlog10, mean_Kzlog10 = torch.load(path_save+f"Case_0/Kzlog10_std.pt", weights_only=False, map_location=devices[-1]), torch.load(f"Generated_Datasets_test/Case_{0}/"+"Kzlog10_mean.pt", weights_only=False, map_location=devices[-1])
    std_I0, mean_I0 = torch.load(path_save+f"Case_0/I0_std.pt", weights_only=False, map_location=devices[-1]), torch.load(f"Generated_Datasets_test/Case_{0}/"+"I0_mean.pt", weights_only=False, map_location=devices[-1])
    std_States, mean_States = torch.load(path_save+f"Case_0/States_std.pt", weights_only=False, map_location=devices[-1]), torch.load(f"Generated_Datasets_test/Case_{0}/"+"States_mean.pt", weights_only=False, map_location=devices[-1])
    std_params, mean_params = torch.load(path_save+f"Case_0/Params_std.pt", weights_only=False, map_location=devices[-1]), torch.load(f"Generated_Datasets_test/Case_{0}/"+"Params_mean.pt", weights_only=False, map_location=devices[-1])

    ## Select the third year (2y spin-up), 120 days from 1 february
    I0_ref, I0 = torch.load(path_save+f"Case_{case}/I0_ens.pt", weights_only=False, map_location=devices[-1])[:, :, -test_size:, None, -335:-215].clone()
    Kz_ref, Kz = torch.load(path_save+f"Case_{case}/Kz_ens.pt", weights_only=False, map_location=devices[-1])[:, :, -test_size:, None, -335*2:-215*2].clone()
    Params = torch.load(path_save+f"Case_{case}/Params.pt", weights_only=False, map_location=devices[-1])[0, -test_size:].clone()
    States = torch.load(path_save+f"Case_{case}/States.pt", weights_only=False, map_location=devices[-1])[-test_size:, :, -335:-215].clone()
    
    States_noise = States.clone()
    for ch in range(States.shape[1]) :
        States_noise[:, ch] += torch.normal(0., obs_noise_std[ch], [States_noise.shape[0], States_noise.shape[2], States_noise.shape[3]]).to(devices[-1])
    torch.save(States_noise, path_save+f"Case_{case}/States_noised.pt")
    
    ## Upsampling because states are generated every day, Kz is generated every 12h. Then padding to have save input depth dimension (states have Nz layers but Kz has Nz+1)
    States_noise = States_noise.repeat_interleave(2, 2)
    States = States.repeat_interleave(2, 2)
    States_noise = torch.cat((States_noise, torch.zeros([States.shape[0], States.shape[1], States.shape[2], 1])), dim=3)
    States = torch.cat((States, torch.zeros([States.shape[0], States.shape[1], States.shape[2], 1])), dim=3)
    
    ## Same for I0 that is only on surface and available every day
    I0 = I0[:, :, :, :, None].repeat_interleave(2, 3).repeat(1, 1, 1, 1, Kz_ref.shape[-1])
    I0_ref = I0_ref[:, :, :, :, None].repeat_interleave(2, 3).repeat(1, 1, 1, 1, Kz_ref.shape[-1])
    
    ## Input: Observed states, uncertain Kz, uncertain I0; and normalization
    X = torch.cat((((States_noise.moveaxis(1, 3)-mean_States)/std_States).moveaxis(3, 1)[None].repeat(n_member, 1, 1, 1, 1),
              (torch.log10(Kz)-mean_Kzlog10)/std_Kzlog10, (I0-mean_I0)/std_I0), dim=2)
    ## Output: Correct parameters, Kz, I0, states; and normalization
    Y = torch.cat((((Params-mean_params)/std_params)[None, :, :, None, None].repeat(n_member, 1, 1, Kz_ref.shape[3], Kz_ref.shape[4]),
                  (torch.log10(Kz_ref)-mean_Kzlog10)/std_Kzlog10, (I0_ref-mean_I0)/std_I0,
                  ((States.moveaxis(1, 3)-mean_States)/std_States).moveaxis(3, 1).repeat(n_member, 1, 1, 1, 1)), dim = 2)
    ## Create training/validation/test datasets
    DS_test = TensorDataset(X.clone(), Y.clone())
    
    torch.save(DS_test, path_save+f"Case_{case}/DS_test_ens")