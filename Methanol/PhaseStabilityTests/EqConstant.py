from Gmin import dfg
import numpy as np
from scipy.constants import R

def K0_CO_to_MeOH(T):

    fG_i = dfg(T)

    # CO + 2 H2 --> CH3OH
    fG_CO = fG_i[3]
    fG_H2 = fG_i[1]
    fG_MeOH = fG_i[4]

    drG = fG_MeOH - fG_CO - 2*fG_H2

    K0 = np.exp(-drG/(R*T))

    return K0

def K0_CO2_to_MeOH(T):

    fG_i = dfg(T)

    # CO2 + 3 H2 --> CH3OH + H2O
    fG_CO2 = fG_i[0]
    fG_H2 = fG_i[1]
    fG_MeOH = fG_i[4]
    fG_H2O = fG_i[2]

    drG = fG_MeOH + fG_H2O - fG_CO2 - 3*fG_H2

    K0 = np.exp(-drG/(R*T))

    return K0

def K0_WGS(T):
    
    fG_i = dfg(T)

    # CO + H2O --> CO2 + H2
    fG_CO2 = fG_i[0]
    fG_H2 = fG_i[1]
    fG_CO = fG_i[3]
    fG_H2O = fG_i[2]

    drG = fG_CO2 + fG_H2 - fG_CO - fG_H2O

    K0 = np.exp(-drG/(R*T))

    return K0