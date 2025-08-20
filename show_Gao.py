# Modell TKA_SHA_Mo_240503_2 (MethT_V11)
# calculation of methanation chemical equilibrium by Gibbs energy minimization
# calculation of fugacity coefficients by Soave-Redlich-Kwong EOS or ideal gas assumption

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from ICIW_Plots import cyclers as ICIW_cyclers
import matplotlib

## data import - validation data
csv_data_Gao = pd.read_csv(r'data_Gao_CO.csv',  # read csv file
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
