# Simulation of Modell TKA_SHA_JGA_Mo_240503_3

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from ICIW_Plots import cyclers as ICIW_cyclers
import matplotlib

## import the model
from TKA_SHA_JGA_Mo_240503_3 import calc_eq_methanation

comps = np.array(['CO2','H2','CH4','H2O','CO','C','N2'])

## EQ calculation
def methanation_equilibrium(T_arr,p_arr,n0,gas_type='real gas'):
    '''
    Calculation of equilibrium composition at different temperatures (T_arr) and pressures (p_arr) for given initial molar amounts n0.
    Parameters:
    -----------
        T_arr: ND-array
        p_arr: ND-array
        n0: ND-array, shape: (7,)
    Returns:
    --------
        results: pd.DataFrame
            Containing T,p, n0, the equilibrium molar amounts and the success status of each calculation
    '''
    print('Evaluating methanation equilibrium...')
    results = pd.DataFrame()
    not_converged = 0
    for p in p_arr:
        for T in T_arr:
            n_eq,success = calc_eq_methanation(T,p,n0,type=gas_type)

            if success:
                x_eq = n_eq/sum(n_eq)
            else:
                n_eq = np.full_like(n0,np.nan)
                x_eq = np.full_like(n0,np.nan)
                not_converged +=1

            results_df = pd.DataFrame({
                'temperature': [T],
                'pressure': [p],
                'success': [success]
                }, 
                index=[0])
            for i,comp in enumerate(comps):
                results_df[f'n0_{comp}'] = n0[i]
                results_df[f'n_eq_{comp}'] = n_eq[i]
                results_df[f'x_eq_{comp}'] = x_eq[i]

            ## Fill dataframe
            results = pd.concat((results,results_df))
                
    print(f'\nConverged calculations: {(len(T_arr)-not_converged)/len(T_arr):.0%} ({len(T_arr) - not_converged}/{len(T_arr)})\n')

    return results


if __name__ == '__main__':
    ## Test the model by verification with data from Gao

    ## validation (Gao 2012, https://doi.org/10.1039/C2RA00632D)
    p = np.array([1*1.01325])*1e5 # p in Pa
    T = np.linspace(200 + 273.15, 800 + 273.15, 100) # T in K
    gas_type = 'real gas' # choose type of gas from 'ideal gas' and 'real gas'

    # Parameter
    n0 = np.full((7,),1e-20)
    n0[0] = 0.2       # initial mole fraction of CO2
    n0[1] = 0.8-7e-20 # initial mole fraction of H2
    n0[2] = 1e-20     # initial mole fraction of CH4
    n0[3] = 1e-20     # initial mole fraction of H2O
    n0[4] = 1e-20     # initial mole fraction of CO
    n0[5] = 1e-20     # initial mole fraction of C
    n0[6] = 1e-20     # initial mole fraction of N2

    ## EQ calculation
    results = methanation_equilibrium(T_arr=T,p_arr=p,n0=n0,gas_type=gas_type)

    

    # plt.style.use('ICIWstyle')
    font = {'size': 10}
    matplotlib.rc('font', **font)

    ## CO2 methanation
    fig, axs = plt.subplots()

    axs.plot(T - 273.15,  results['x_eq_CO2'].values,     '-',                 label = 'CO$_2$')
    axs.plot(T - 273.15,  results['x_eq_H2'].values,     '-',                 label = 'H$_2$')
    axs.plot(T - 273.15,  results['x_eq_CH4'].values,     '-',                 label = 'CH$_4$')
    axs.plot(T - 273.15,  results['x_eq_H2O'].values,     '-',                 label = 'H$_2$O')
    axs.plot(T - 273.15,  results['x_eq_CO'].values,     '-',                 label = 'CO')
    axs.plot(T - 273.15,  results['x_eq_C'].values,     '-',                 label = 'C')
    axs.plot(T - 273.15,  results['x_eq_N2'].values,     '-',                 label = 'N2')
    axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 0], 'o', markersize = 3, label = 'CO$_2$ (Gao)')
    axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 1], 'o', markersize = 3, label = 'H$_2$ (Gao)')
    axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 2], 'o', markersize = 3, label = 'CH$_4$ (Gao)')
    axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 3], 'o', markersize = 3, label = 'H$_2$O (Gao)')
    axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 4], 'o', markersize = 3, label = 'CO (Gao)')
    axs.plot(T_CO2_Gao,   x_CO2_Gao[:, 5], 'o', markersize = 3, label = 'C (Gao)')
    axs.set_xlabel('$T$ / °C')
    axs.set_ylabel('$x_i$ / 1')
    axs.set_title('Equilibrium composition, H$_2$ / CO$_2$ = 4, 1 atm')
    axs.set_ylim(0, 0.8)
    axs.set_xlim(200, 800)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize = 8)
    plt.tight_layout()
    plt.show()
