# Modell TKA_Mo_240503_1_(MethT_V11)
# calculation of methanation chemical equilibrium by Gibbs energy minimization
# calculation of fugacity coefficients by Soave-Redlich-Kwong EOS or ideal gas assumption

import numpy as np
from scipy.optimize import minimize
from TKA_Mo_240503_2_fugacity_coefficient_V2 import phi_Soave
import warnings


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

    :param n0: vector containing initial molar amounts of CO2, H2, CH4, H2O, CO, C, He, Ar and N2
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

    for i, guess in enumerate(guesses):

        sol = minimize(g_T, guess, args=(T, p, type), method='SLSQP', constraints = cons, bounds=bnds, options = {'disp': 'False', 'maxiter': 1000, 'ftol': 1e-5})

        if sol.success:
            g_T_vals[i] = sol.fun
            success = True
            x_eq_vals[i, :] = sol.x / np.sum(sol.x)
        else:
            g_T_vals[i] = np.nan
            success = False
            x_eq_vals[i, :] = np.nan

    # now find the minimum g_T_value and respective x_eq
    min_idx = np.nanargmin(g_T_vals)
    g_T_value = g_T_vals[min_idx]
    x_eq = x_eq_vals[min_idx,:]

    return x_eq, success, g_T_value, random_guess