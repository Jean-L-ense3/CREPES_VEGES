# CREPES_VEGES
Scripts for the VEGES project: Vertical Experiment Gauging Estimation Strategy

Description: This study is codenamed 'VEGES - Vertical Experiment Gauging Estimation Strategy'. It is part of the PhD project "CREPES - Carbon REconstructed Per an Emulator through Supervision" (Carbone REconstruit Par Emulateur Supervisé). This dataset contains 7 different files:

Calibration_DA.py to calibrate the BGC model using a 4Dvar-based scheme.
Calibration_NN.py to calibrate the BGC model using a UNet-based scheme.
environment.yaml to install the suitable python environment with correct package version.
func_file.py that contains the needed functions.
Generator_data.py to generate all the necessary data sets.
model_file.py that contains the UNet model.
Notebook_analysis_plot.ipynb to plot the paper results.

They goes with dataset from the zenodo repository 'Crepes_veges data: Solving calibration and reanalysis challenges of ocean BGC dynamics with neural schemes: a 1D NNPZD case-study'
It needs at least the FORCING_40km file that contains the raw data to generate the datasets. With data compressed in "Generated_Datasets.zip" and "Res.zip" it is possible to plot the results of the paper "Solving calibration and reanalysis challenges of ocean BGC dynamics with neural schemes: a 1D NNPZD case-study".
The Data are generated using a set of forcing profiles from polgyr, stored in "FORCING_40km.zip".

1. Install the correct packages with their associated version with the environment.yaml file.
2. Generate the different data sets: run Dataset_Generator.py
3. Use freely the different methods (run Calibration_DA.py, Calibration_NN.py)
