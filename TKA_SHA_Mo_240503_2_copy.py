# Modell TKA_Mo_240503_1_(MethT_V11)
# calculation of methanation chemical equilibrium by Gibbs energy minimization
# calculation of fugacity coefficients by Soave-Redlich-Kwong EOS or ideal gas assumption

import numpy as np
from scipy.optimize import minimize
from TKA_Mo_240503_2_fugacity_coefficient_V2 import phi_Soave
import warnings
import scipy.constants as csts
from scipy.integrate import quad


def dfg(T):
    """
    function for determination of Gibbs free energy of formation of a species @ T from NIST-JANAF tables by a polynomic fit, range of validity: 0 - 6000 K

    :param T: temperature in K
    :return: Gibbs free energy of formation in J / mol
    """

    coeff = np.array([[ -4.4712899950965e-18, 7.69027397691646e-14, -4.83454740516897e-10, 2.06356409761266e-6, -4.28205970848287e-3, -3.93254377825293e2],  # CO2
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # H2
                      [-1.19480733487367e-16, 2.05212901276465e-12,  -1.30585665138136e-8, 3.77285316328865e-5,  6.34019284855666e-2, -7.14155077436689e1],  # CH4
                      [                    0, 1.02813542552485e-13,  -1.46197827311474e-9, 7.46216055871485e-6,  4.27766908921636e-2, -2.41321516149705e2],  # H2O
                      [                    0,                    0,                     0, 1.60737136748171e-6, -9.02786221059026e-2, -1.11442697976118e2],  # CO
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # C
                      [                    0,                    0,                     0,                   0,                    0,                   0]]) # N2

    T_poly = np.array([T**5, T**4, T**3, T**2, T, 1])

    res = np.matmul(T_poly, np.transpose(coeff)) * 1000
    return res


def g_T(n, T, p, type):
    """
    function for determination of the total Gibbs free energy to be minimized

    :param n: vector containing molar amounts of CO2, H2, CH4, H2O, CO, C and N2
    :param T: temperature in K
    :param p: pressure in bar
    :return: total Gibbs free energy in J / mol
    """

    for i in range(n.shape[0]):
        if n[i] <= 0:
            n[i] = 1e-20

    n_gas = np.delete(n, 5)   # array containing only the amounts of substance of gaseous species (CO2, H2, CH4, H2O, CO and N2) in mol
    n_sol = n[5] # array containing only the amounts of substance of solid species (C)

    y_gas = n_gas / np.sum(n_gas) # array containing gas phase molar fractions of gaseous species
    x_sol = n_sol / np.sum(n_sol) # array containing solid phase molar fractions of solid species

    dfgi = dfg(T)              # Gibbs free energy of formation of all species in J / mol
    phii = np.ones_like(n_gas) # default array for fugacity coefficients of gaseous species in 1

    if type == 'ideal gas':
        phii = phii
    elif type == 'real gas':
        phii = phi_Soave(y_gas, T, p * 1e5) # function needs pressure in Pa and molar fractions in gas phase
    else:
        print('Please choose type of gas from given options: ideal gas or real gas')

    R  = 8.314 # universal gas constant in J / mol K
    p0 = 1 # standard pressure in bar

    res = np.dot(n, dfgi) + R * T * (np.dot(n_gas, np.log(phii * p * y_gas / p0)) + np.dot(n_sol, np.log(x_sol)))
    return res


# setting constraints
def element_balance(n, n0):
    """
    function for checking the element balance as a constraint for the minimization

    :param n0: vector containing initial molar amounts of CO2, H2, CH4, H2O, CO, C and N2
    :return: residual -> 0
    """
    # element-species matrix (C, O, H, N)
    A = np.array([[1, 2, 0, 0],  # CO2
                  [0, 0, 2, 0],  # H2
                  [1, 0, 4, 0],  # CH4
                  [0, 1, 2, 0],  # H2O
                  [1, 1, 0, 0],  # CO
                  [1, 0, 0, 0],  # C
                  [0, 0, 0, 2]]) # N2
    res = np.matmul(n, A) - np.matmul(n0, A)
    return res

def comp_mole_numbers_to_element_mole_numbers(n):
    """
    function for converting the molar amounts of species to the molar amounts of elements

    Args:
        n (array): molar amounts of species [CO2 H2 CH4 H2O CO C N2]
    """
    # element-species matrix (C, O, H, N)
    A = np.array([[1, 2, 0, 0],  # CO2
                  [0, 0, 2, 0],  # H2
                  [1, 0, 4, 0],  # CH4
                  [0, 1, 2, 0],  # H2O
                  [1, 1, 0, 0],  # CO
                  [1, 0, 0, 0],  # C
                  [0, 0, 0, 2]]) # N2
    n_elements = np.matmul(n, A)
    
    return n_elements


def calc_bounds(x0):
    n0      = x0 * 1                            # initial molar amount in mol
    max_C   = n0[0] + n0[2] + n0[4] + n0[5]     # molar amount of carbon in the system in mol
    max_H   = 2 * n0[1] + 4 * n0[2] + 2 * n0[3] # molar amount of hydrogen in the system in mol
    max_O   = 2 * n0[0] + n0[3] + n0[4]         # molar amount of oxygen in the system in mol
    max_N   = 2 * n0[6]                         # molar amount of nitrogen in the system
    max_CO2 = min(max_C, 0.5 * max_O)           # maximum possible molar amount of CO2 in mol
    max_H2  = 0.5 * max_H                       # maximum possible molar amount of H2 in mol
    max_CH4 = min(max_C, 0.25 * max_H)          # maximum possible molar amount of CH4 in mol
    max_H2O = min(0.5 * max_H, max_O)           # maximum possible molar amount of H2O in mol
    max_CO  = min(max_C, max_O)                 # maximum possible molar amount of CO in mol
    max_N2  = 0.5 * max_N                       # maximum possible molar amount of N2 in mol

    bnds = ((0, max_CO2), (0, max_H2), (0, max_CH4), (0, max_H2O), (0, max_CO), (0, max_C), (0, max_N2))
    init = np.ones_like(n0)
    
    return n0,bnds,init 

Shomate_coeffs_CO2 = { # 298 - 1200 K
    "A": 24.997,
    "B": 55.187,
    "C": -33.691,
    "D": 7.948,
    "E": -0.137,
    "F": -403.608,
    "G": 228.243,
    "H": -393.522,
}

Shomate_coeffs_H2 = { # 298 - 1000 K
    "A": 33.066,
    "B": -11.363,
    "C": 11.433,
    "D": -2.772,
    "E": -0.159,
    "F": -9.981,
    "G": 172.708,
    "H": 0,
}

Shomate_coeffs_H2O = { # 500 - 1700 K
    "A": 30.092,
    "B": 6.833,
    "C": 6.793,
    "D": -2.534,
    "E": 0.082,
    "F": -250.881,
    "G": 223.397,
    "H": -241.826,
}

Shomate_coeffs_CO = { # 298 - 1300 K
    "A": 25.568,
    "B": 6.096,
    "C": 4.055,
    "D": -2.671,
    "E": 0.131,
    "F": -118.009,
    "G": 227.367,
    "H": -110.527,
}

Shomate_coeffs_CH4 = { # 298 - 1300 K
    "A": -0.703029,
    "B": 108.4773,
    "C": -42.52157,
    "D": 5.862788,
    "E": 0.678565,
    "F": -76.84376,
    "G": 158.7163,
    "H": -74.8731,
}

Glenn_coeffs_H2O = { # 200 - 1000 K
    "a_1": -3.948 * 1e4,
    "a_2": 5.756 * 1e2,
    "a_3": 9.318 * 1e-1,
    "a_4": 7.223 * 1e-3,
    "a_5": -7.343 * 1e-6,
    "a_6": 4.955 * 1e-9,
    "a_7": -1.337 * 1e-12,
    "b_1": -3.304 * 1e4,
    "b_2": 1.724 * 1e1,
}

coeffs_C = {
    "c_p": 10.68 # constant c_p for solid carbon
}

def c_p(T, coeffs):
    """
    function calculates the heat capacity of a compound at a given temperature
    :param T: temperature in K
    :param coeffs: dictionary of Shomate coefficients
    :return: heat capacity of the compound at the given temperature
    """
    
    t = T / 1000
    
    if "A" in coeffs:
        c_p_comp = coeffs["A"] + coeffs["B"] * t + coeffs["C"] * t**2 + coeffs["D"] * t**3 + coeffs["E"] / t**2
    elif "a_1" in coeffs:
        c_p_comp = csts.R * (coeffs["a_1"] / T ** 2 + coeffs["a_2"] / T + coeffs["a_3"] + coeffs["a_4"] * T + coeffs["a_5"] * T**2 + coeffs["a_6"] * T**3 + coeffs["a_7"] * T**4)
    else:
        c_p_comp = coeffs["c_p"] # calculation for carbon
    
    return c_p_comp 

def c_p_R(T, reaction):
    """
    function calculates the heat capacity of the reaction at a given temperature
    :param T: temperature in K
    :param reaction: string of the reaction
    :return: heat capacity of the reaction at the given temperature
    """

    c_p_CO2 = c_p(T, Shomate_coeffs_CO2)
    c_p_H2 = c_p(T, Shomate_coeffs_H2)
    c_p_H2O = c_p(T, Glenn_coeffs_H2O)
    c_p_CO = c_p(T, Shomate_coeffs_CO)
    c_p_CH4 = c_p(T, Shomate_coeffs_CH4)
    c_p_C = c_p(T, coeffs_C)

    if reaction == 'CO2 methanation':
        c_p_R = c_p_CH4 + 2 * c_p_H2O - c_p_CO2 - 4 * c_p_H2
    elif reaction == 'CO methanation':
        c_p_R = c_p_CH4 + c_p_H2O - c_p_CO - 3 * c_p_H2
    elif reaction == 'WGS':
        c_p_R = c_p_CO2 + c_p_H2 - c_p_CO - c_p_H2O
    elif reaction == 'Inversed Methane CO2 reforming':
        c_p_R = c_p_CO2 + c_p_CH4 - 2 * c_p_CO - 2 * c_p_H2
    elif reaction == 'Boudouard reaction':
        c_p_R = c_p_CO2 + c_p_C - 2 * c_p_CO
    elif reaction == 'Methane cracking':
        c_p_R = 2 * c_p_H2 + c_p_C - c_p_CH4
    elif reaction == 'Carbon monoxide reduction':
        c_p_R = c_p_C + c_p_H2O - c_p_CO - c_p_H2
    elif reaction == 'Carbon dioxide reduction':
        c_p_R = c_p_C + 2 * c_p_H2O - c_p_CO2 - 2 * c_p_H2


    return c_p_R

def Thermo_props_ref(reaction):
    """
    function calculates the standard enthalpy and entropy of the reaction at 298.15 K
    :param reaction: string of the reaction
    :return: standard enthalpy and entropy of the reaction at 298.15 K
    """

    dfH0_CH4 = Shomate_coeffs_CH4["H"] * 1000
    dfH0_CO2 = Shomate_coeffs_CO2["H"] * 1000
    dfH0_H2 = 0 #Shomate_coeffs_H2["H"] * 1000
    dfH0_CO = Shomate_coeffs_CO["H"] * 1000
    dfH0_H2O = Shomate_coeffs_H2O["H"] * 1000
    dfH0_C = 0

    dS0_CO2, dS0_H2, dS0_CO, dS0_H2O, dS0_CH4, dS0_C = 213.785, 130.68, 197.66, 188.84, 186.25, 6.201

    if reaction == 'CO2 methanation':
        dH0 = dfH0_CH4 + 2 * dfH0_H2O - dfH0_CO2 - 4 * dfH0_H2
        dS0 = dS0_CH4 + 2 * dS0_H2O - dS0_CO2 - 4 * dS0_H2
    elif reaction == 'CO methanation':
        dH0 = dfH0_CH4 + dfH0_H2O - dfH0_CO - 3 * dfH0_H2
        dS0 = dS0_CH4 + dS0_H2O - dS0_CO - 3 * dS0_H2
    elif reaction == 'WGS':
        dH0 = dfH0_CO2 + dfH0_H2 - dfH0_CO - dfH0_H2O
        dS0 = dS0_CO2 + dS0_H2 - dS0_CO - dS0_H2O
    elif reaction == 'Inversed Methane CO2 reforming':
        dH0 = dfH0_CO2 + dfH0_CH4 - 2 * dfH0_CO - 2 * dfH0_H2
        dS0 = dS0_CO2 + dS0_CH4 - 2 * dS0_CO - 2 * dS0_H2
    elif reaction == 'Boudouard reaction':
        dH0 = dfH0_CO2 + dfH0_C - 2 * dfH0_CO
        dS0 = dS0_CO2 + dS0_C - 2 * dS0_CO
    elif reaction == 'Methane cracking':
        dH0 = 2 * dfH0_H2 + dfH0_C - dfH0_CH4
        dS0 = 2 * dS0_H2 + dS0_C - dS0_CH4  
    elif reaction == 'Carbon monoxide reduction':
        dH0 = dfH0_C + dfH0_H2O - dfH0_CO - dfH0_H2
        dS0 = dS0_C + dS0_H2O - dS0_CO - dS0_H2
    elif reaction == 'Carbon dioxide reduction':
        dH0 = dfH0_C + 2 * dfH0_H2O - dfH0_CO2 - 2 * dfH0_H2
        dS0 = dS0_C + 2 * dS0_H2O - dS0_CO2 - 2 * dS0_H2

    return dH0, dS0

def dRH(T, reaction):
    """
    function calculates the enthalpy of the reaction at a given temperature
    :param T: temperature in K
    :param reaction: string of the reaction
    :return: enthalpy of the reaction at the given temperature
    """
    # reaction enthalpy at 298.15 K
    dH0 = Thermo_props_ref(reaction)[0]
    # integral of the heat capacity of the reaction
    def integrand(T_prime):
        return c_p_R(T_prime, reaction)
    # enthalpy of the reaction at the given temperature
    dRH = dH0 + quad(integrand, 298.15, T)[0]

    return dRH

def lnK0(reaction):
    """
    function calculates the logarithm of equilibrium constant of the reaction at 298.15 K
    :param reaction: string of the reaction
    :return: logarithm of equilibrium constant of the reaction at 298.15 K
    """
    T_ref = 298.15
    dH0, dS0 =  Thermo_props_ref(reaction)
    dG0 = dH0 - T_ref * dS0
    lnK0_ref = -dG0 / (csts.R * T_ref)

    return lnK0_ref

def K0(T, reaction):

    # Calculate ln(K0) at the reference temperature (298.15 K)
    lnK0_ref = lnK0(reaction)
    
    # Define the integrand for the Van't Hoff equation
    def integrand(T_prime):
        dRH_T_prime = dRH(T_prime, reaction)
        return dRH_T_prime / (csts.R * T_prime**2)
    # Perform the integration from 298.15 K to the desired temperature T
    integral_value, _ = quad(integrand, 298.15, T)
    # Calculate ln(K0(T)) using the Van't Hoff equation
    lnK0_T = lnK0_ref + integral_value
    # Calculate K0(T)
    K0_T = np.exp(lnK0_T)

    return K0_T

def check_eq_consts(T, p, y_GG, reaction):
    """
    function calculates the equilibrium constant of the reaction at a given temperature and pressure using Gibbs energy minimization model and Van't Hoff equation
    :param T: temperature in K
    :param p: pressure in Pa
    :param y_GG: molar fractions of all species in equilibrium calculated by Gibbs energy minimization model
    :param reaction: string of the reaction
    :param type: type of the gas (ideal or real)
    :return: equilibrium constant of the reaction calculated by Gibbs energy minimization model and Van't Hoff equation
    """

    p_bar = p * 1e-5 # p to bar for calculation of K_x pressure dependency bc p0 = 1 bar
    y_GG_fug_coeffs = np.delete(y_GG, 5) # removing C from the list  
    fug_coeffs = phi_Soave(y_GG_fug_coeffs, T, p) # calculating fugacity coefficients

    # set y_GG to at least 1e-20 to avoid division by zero
    for i in range(len(y_GG)):
        if y_GG[i] < 1e-20:
            y_GG[i] = 1e-20 # set to 1e-20 to avoid division by zero
    
    # calculating K_0 via Gibbs energy minimization model and Van't Hoff equation
    if reaction == 'CO2 methanation':
        K_0_sim = (y_GG[2] * y_GG[3] ** 2) / (y_GG[0] * y_GG[1] ** 4) / p_bar ** 2 * (fug_coeffs[2] * fug_coeffs[3]**2) / (fug_coeffs[0] * fug_coeffs[1]**4)
        K_0_vantHoff = K0(T, reaction)
    elif reaction == 'CO methanation':
        K_0_sim = (y_GG[2] * y_GG[3]) / (y_GG[4] * y_GG[1] ** 3) / p_bar**2 * (fug_coeffs[2] * fug_coeffs[3]) / (fug_coeffs[4] * fug_coeffs[1] ** 3)
        K_0_vantHoff = K0(T, reaction)
    elif reaction == 'WGS':
        K_0_sim = (y_GG[0] * y_GG[1]) / (y_GG[3] * y_GG[4]) * (fug_coeffs[0] * fug_coeffs[1]) / (fug_coeffs[3] * fug_coeffs[4])
        K_0_vantHoff = K0(T, reaction)
    elif reaction == 'Inversed Methane CO2 reforming':
        K_0_sim = (y_GG[2] * y_GG[0]) / (y_GG[4]**2 * y_GG[1]**2) / p_bar ** 2 * (fug_coeffs[2] * fug_coeffs[0]) / (fug_coeffs[4]**2 * fug_coeffs[1]**2)
        K_0_vantHoff = K0(T, reaction)
    elif reaction == 'Boudouard reaction':
        K_0_sim = y_GG[0] / y_GG[4]**2 / p_bar * fug_coeffs[0] / fug_coeffs[4]**2
        K_0_vantHoff = K0(T, reaction)
    elif reaction == 'Methane cracking':
        K_0_sim = y_GG[1]**2 / y_GG[2] * p_bar * fug_coeffs[1]**2 / fug_coeffs[2]
        K_0_vantHoff = K0(T, reaction)
    elif reaction == 'Carbon monoxide reduction': 
        K_0_sim = y_GG[3] / (y_GG[4] * y_GG[1]) / p_bar * fug_coeffs[3] / (fug_coeffs[4] * fug_coeffs[1])
        K_0_vantHoff = K0(T, reaction)
    elif reaction == 'Carbon dioxide reduction':
        K_0_sim = y_GG[3]**2 / (y_GG[0] * y_GG[1]**2) / p_bar * (fug_coeffs[3]**2) / (fug_coeffs[0] * fug_coeffs[1]**2)
        K_0_vantHoff = K0(T, reaction)
    else:
        raise ValueError('Reaction not found') 

    return K_0_sim, K_0_vantHoff

def calc_eq_methanation(T,p,x0,type='real gas'):
    '''
    Calculates the equilibrium composition of a gas mixture at one given temperature and pressure.

    Parameters
    ----------
        T: temperature in K (float)
        p: pressure in Pa (float)
        x0: inlet composition [CO2 H2 CH4 H2O CO C N2]
        guess: initial guess for the equilibrium composition for the minimization
        type: type of gas, choose from 'ideal gas' or 'real gas'

    Returns
    -------
        x_eq: equilibrium composition [CO2 H2 CH4 H2O CO C N2]
        success: boolean, True if the minimization was successful, False otherwise

    '''
    p = p*1e-5 # in bar

    n0,bnds,_ = calc_bounds(x0)
    cons = {'type': 'eq', 'fun': element_balance, 'args': [n0]}

    if round(np.sum(x0),5) != 1:
        ## Warning
        warnings.warn(f'WARNING: Please check inlet composition! Sum of x_i is not one but {np.sum(x0)} !')

    # use 3 different guesses for the minimization
    random_guess = np.random.dirichlet((1, 1, 1, 1, 1, 1, 1), 1)[0]
    guesses = [x0, np.ones_like(x0), random_guess]

    g_T_vals = np.zeros(len(guesses))
    x_eq_vals = np.zeros([len(guesses),len(x0)])
    n_eq_vals = np.zeros([len(guesses),len(x0)])
    
    for i, guess in enumerate(guesses):

        sol = minimize(g_T, guess, args=(T, p, type), method='SLSQP', constraints = cons, bounds=bnds, options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-5})

        if sol.success:
            g_T_vals[i] = sol.fun
            n_eq_vals[i, :] = sol.x
        else:
            g_T_vals[i] = np.nan
            n_eq_vals[i, :] = np.nan

    # now find the minimum g_T_value and respective x_eq
    try:
        min_idx = np.nanargmin(g_T_vals)
        n_eq = n_eq_vals[min_idx,:]
        x_eq = n_eq / np.sum(n_eq)
        success = True
    except ValueError:
        n_eq = np.nan
        x_eq = np.nan
        success = False

    if success:

        summed_K0_percentage_deviation = 0
        # check if the equilibrium constants are consistent
        for reaction in ['CO2 methanation', 'CO methanation', 'WGS', 'Inversed Methane CO2 reforming', 'Boudouard reaction', 'Methane cracking', 'Carbon monoxide reduction', 'Carbon dioxide reduction']:
            K_0_sim, K_0_vantHoff = check_eq_consts(T, p*1e5, x_eq, reaction)
            K0_percentage_deviation = 100 * (K_0_sim - K_0_vantHoff) / K_0_vantHoff
            summed_K0_percentage_deviation += abs(K0_percentage_deviation)

        # check if the equilibrium constants are consistent
        if T < 300+273.15:
            if summed_K0_percentage_deviation > 1500:
                n_eq_vals = np.nan
        elif 300+273.15 <= T < 600+273.15:
            if summed_K0_percentage_deviation > 2500:
                n_eq_vals = np.nan
        else:
            if summed_K0_percentage_deviation > 5000:
                n_eq_vals = np.nan

    # return p, T, x0, x_eq if successful else return only NaN
    if n_eq_vals is not np.nan:
        return p, T, x0, n_eq, success
    else:
        return np.nan, np.nan, np.nan, np.nan, False