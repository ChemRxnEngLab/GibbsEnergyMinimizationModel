# Modell TKA_Mo_240503_1_(MethT_V11)
# calculation of methanation chemical equilibrium by Gibbs energy minimization
# calculation of fugacity coefficients by Soave-Redlich-Kwong EOS or ideal gas assumption

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from ICIW_Plots import cyclers as ICIW_cyclers
import matplotlib

from TKA_SHA_Mo_240503_2 import equilibrium_composition_methanation_moles

## Test the model by verification with data from Gao

## validation (Gao 2012, https://doi.org/10.1039/C2RA00632D)
p = np.array([1*1.01325])*1e5 # p in Pa
T = np.linspace(200 + 273.15, 800 + 273.15, 100) # T in K
type = 'real gas' # choose type of gas from 'ideal gas' and 'real gas'

# Parameter
x0 = np.empty(9)
x0[0] = 0.2       # initial mole fraction of CO2
x0[1] = 0.8-7e-20 # initial mole fraction of H2
x0[2] = 1e-20     # initial mole fraction of CH4
x0[3] = 1e-20     # initial mole fraction of H2O
x0[4] = 1e-20     # initial mole fraction of CO
x0[5] = 1e-20     # initial mole fraction of C
x0[6] = 1e-20     # initial mole fraction of He
x0[7] = 1e-20     # initial mole fraction of Ar
x0[8] = 1e-20     # initial mole fraction of N2

x,n = equilibrium_composition_methanation_moles(T,p,x0,type=type)

## data import - validation data
csv_data_Gao = pd.read_csv(r'data_Gao_CO2.csv',  # read csv file
                            sep = ';')

## convert read data into numpy array
data_Gao         = csv_data_Gao.to_numpy()

T_CO2_Gao        = data_Gao[:, 0]
x_CO2_Gao        = data_Gao[:, 1:7]

# plt.style.use('ICIWstyle')
font = {'size': 10}
matplotlib.rc('font', **font)

## CO2 methanation
fig, axs = plt.subplots()
for pp in range(p.shape[0]):
    cyc1 = ICIW_cyclers.ICIW_colormap_cycler('hsv', 9, start = 0.1, stop = 1)
    axs.set_prop_cycle(cyc1)
    axs.plot(T - 273.15,  x[pp, :, 0],     '-',                 label = 'CO$_2$')
    axs.plot(T - 273.15,  x[pp, :, 1],     '-',                 label = 'H$_2$')
    axs.plot(T - 273.15,  x[pp, :, 2],     '-',                 label = 'CH$_4$')
    axs.plot(T - 273.15,  x[pp, :, 3],     '-',                 label = 'H$_2$O')
    axs.plot(T - 273.15,  x[pp, :, 4],     '-',                 label = 'CO')
    axs.plot(T - 273.15,  x[pp, :, 5],     '-',                 label = 'C')
    axs.plot(T - 273.15,  x[pp, :, 6],     '-',                 label = 'He')
    axs.plot(T - 273.15,  x[pp, :, 7],     '-',                 label = 'Ar')
    axs.plot(T - 273.15,  x[pp, :, 8],     '-',                 label = 'N$_2$')
axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 0], 'o', markersize = 3, label = 'CO$_2$ (Gao)')
axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 1], 'o', markersize = 3, label = 'H$_2$ (Gao)')
axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 2], 'o', markersize = 3, label = 'CH$_4$ (Gao)')
axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 3], 'o', markersize = 3, label = 'H$_2$O (Gao)')
axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 4], 'o', markersize = 3, label = 'CO (Gao)')
axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 5], 'o', markersize = 3, label = 'C (Gao)')
axs.set_xlabel('$\mathit{T}$ / °C')
axs.set_ylabel('$\mathit{x}_{\mathit{i}}$ / 1')
axs.set_title('Equilibrium composition, H$_2$ / CO$_2$ = 4, 1 atm')
axs.set_ylim(0, 0.8)
axs.set_xlim(200, 800)
plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize = 8)
plt.tight_layout()
plt.show()
