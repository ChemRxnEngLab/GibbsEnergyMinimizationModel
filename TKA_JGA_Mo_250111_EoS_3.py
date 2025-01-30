# Modell TKA_Mo_240503_1_(MethT_V11)
# calculation of methanation chemical equilibrium by Gibbs energy minimization
# calculation of fugacity coefficients by Soave-Redlich-Kwong EOS or ideal gas assumption

import numpy as np
from scipy.optimize import minimize, basinhopping, shgo, differential_evolution
from TKA_Mo_fug_coeffs_MeOH import phi_Soave, phi_Soave_2, phi_Soave_3
import warnings
from thermo_coeffs import delta_f_G


def dfg(T):
    """
    function for determination of Gibbs free energy of formation of a species @ T from NIST-JANAF tables by a polynomic fit, range of validity: 0 - 6000 K

    :param T: temperature in K
    :return: Gibbs free energy of formation in J / mol
    """

    coeff = np.array([[ -4.4712899950965e-18, 7.69027397691646e-14, -4.83454740516897e-10, 2.06356409761266e-6, -4.28205970848287e-3, -3.93254377825293e2],  # CO2
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # H2
                      [                    0, 1.02813542552485e-13,  -1.46197827311474e-9, 7.46216055871485e-6,  4.27766908921636e-2, -2.41321516149705e2],  # H2O
                      [                    0,                    0,                     0, 1.60737136748171e-6, -9.02786221059026e-2, -1.11442697976118e2]]) # CO

    T_poly = np.array([T**5, T**4, T**3, T**2, T, 1])

    res = np.matmul(T_poly, np.transpose(coeff)) * 1000
    # calculate DME and MEOH separately
    dfg_meoh = delta_f_G(T, 'CH3OH')
    dfg_n2 = 0
    res = np.append(res, [dfg_meoh, dfg_n2])
    
    return res


def g_T(n, T, p, type="real gas"):
    """
    function for determination of the total Gibbs free energy to be minimized

    :param n: vector containing molar amounts of CO2, H2, H2O, CO, DME, MeOH and N2
    :param T: temperature in K
    :param p: pressure in bar
    :return: total Gibbs free energy in J / mol
    """

    n_G = n[:6] # array containing amounts of substance in gas phase in mol
    n_L = n[6:] # array containing amounts of substance in liquid phase in mol

    n_G = np.maximum(n_G, 1e-20) # set all values below 0 to 0
    n_L = np.maximum(n_L, 1e-20) # set all values below 0 to 0

    y_gas = n_G / np.sum(n_G) # array containing gas phase molar fractions of gaseous species
    x_liq = n_L / np.sum(n_L) # array containing liquid phase molar fractions of gaseous species
    z_mix = np.hstack((y_gas, x_liq)) # array containing molar fractions of all species

    dfgi = dfg(T)                # Gibbs free energy of formation of all species in J / mol @ T in ideal gas state_

    if type == 'ideal gas':
        phii_V = np.ones_like(n_G)
        phii_L = np.ones_like(n_L)
    elif type == 'real gas':
        phii_L, phii_V, VLE = phi_Soave_3(z_mix, T, p * 1e5) # function needs pressure in Pa and molar fractions in gas phase
    else:
        print('Please choose type of gas from given options: ideal gas or real gas')

    R  = 8.314 # universal gas constant in J / mol K
    p0 = 1 # standard pressure in bar

    res_G = np.dot(n_G, dfgi) + R * T * np.dot(n_G, np.log(phii_V * p * y_gas / p0))   
    res_L = np.dot(n_L, dfgi) + R * T * np.dot(n_L, np.log(phii_L * p * x_liq / p0)) 

    res = res_G + res_L
    
    return res

# setting constraints
def element_balance(n, n0):
    """
    function for checking the element balance as a constraint for the minimization

    :param n0: vector containing initial molar amounts of CO2, H2, H2O, CO, CH3OH and N2
    :return: residual -> 0
    """

    # element-species matrix (C, O, H, N)
    A = np.array([[1, 2, 0, 0],  # CO2
                  [0, 0, 2, 0],  # H2
                  [0, 1, 2, 0],  # H2O
                  [1, 1, 0, 0],  # CO
                  [1, 1, 4, 0],  # MeOH
                  [0, 0, 0, 2]]) # N2
    
    res = np.matmul(n[:6], A) - np.matmul(n0[:6], A) + np.matmul(n[6:], A) - np.matmul(n0[6:], A)
    
    return res

def calc_bounds(x0):
    
    n0     = x0 * 1                                                                                                # initial molar amount in mol
    n_G    = n0[:6]                                                                                                 # initial molar amount in gas phase in mol
    n_L    = n0[6:]       
    n_t    = n_G + n_L                                                                                          # initial molar amount in liquid phase in mol
    max_C   = n_t[0] + n_t[3] + n_t[4] #+ n_L[1]                                                    # molar amount of carbon in the system in mol
    max_H   = 2 * n_t[1] + 2 * n_t[2] + 4 * n_t[4] #+ 2 * n_L[0] + 4 * n_L[1]
    max_O   = 2 * n_t[0] + n_t[2] + n_t[3] + n_t[4] #+ n_L[0] + n_L[1]        # molar amount of oxygen in the system in mol
    max_N   = 2 * n_t[5]                                                                 # molar amount of nitrogen in the system
    max_CO2 = min(max_C, 0.5 * max_O)                          # maximum possible molar amount of CO2 in mol
    max_H2  = 0.5 * max_H                                      # maximum possible molar amount of H2 in mol
    max_H2O = min(0.5 * max_H, max_O)                          # maximum possible molar amount of H2O in mol
    max_CO  = min(max_C, max_O)                                # maximum possible molar amount of CO in mol
    max_MeOH = min(max_C, max_O, 0.25 * max_H)                 # maximum possible molar amount of MeOH in mol
    max_N2  = 0.5 * max_N                                      # maximum possible molar amount of N2 in mol

    
    bnds = [(0, max_val) for max_val in [max_CO2, max_H2, max_H2O, max_CO, max_MeOH, max_N2]]
    bnds = tuple(bnds + bnds)  # Apply bounds to both gas and liquid phase
    
    init = np.ones_like(n0)
    
    return n0,bnds,init 

'''def isofug_cond(n, T, p):

    n_G = n[:6] # array containing amounts of substance in gas phase in mol
    n_L = n[6:] # array containing amounts of substance in liquid phase in mol

    n_G = np.maximum(n_G, 1e-20) # set all values below 0 to 0
    n_L = np.maximum(n_L, 1e-20) # set all values below 0 to 0

    y_gas = n_G / np.sum(n_G) # array containing gas phase molar fractions of gaseous species
    x_liq = n_L / np.sum(n_L) # array containing liquid phase molar fractions of gaseous species

    z_mix = np.hstack((y_gas, x_liq)) # array containing molar fractions of all species

    phii_L, phii_V, VLE = phi_Soave_3(z_mix, T, p * 1e5) # function needs pressure in Pa and molar fractions in gas phase

    if VLE:
        res = np.sum(y_gas * phii_V - x_liq * phii_L)
        print('VLE')
    else:
        res = 0

    return res'''

"""def calc_eq(T,p,x0,type='real gas'): # removed guess
    '''
    Calculates the equilibrium composition of a gas mixture at one given temperature and pressure.

    Parameters
    ----------
        T: temperature in K (float)
        p: pressure in Pa (float)
        x0: inlet composition [CO2 H2 CH4 H2O CO C He Ar N2]
        guess: initial guess for the equilibrium composition for the minimization
        type: type of gas, choose from 'ideal gas' or 'real gas'

    Returns
    -------
        x_eq: equilibrium composition [CO2 H2 CH4 H2O CO C He Ar N2]
        success: boolean, True if the minimization was successful, False otherwise

    '''
    p = p*1e-5 # in bar

    n0,bnds,_ = calc_bounds(x0)
    
    cons = [{'type': 'eq', 'fun': element_balance, 'args': [n0]},
            # set each component to be greater than 0
            {'type': 'ineq', 'fun': lambda n: n}]


    rndm_no = 10
    rndm_guesses = np.random.dirichlet(np.ones_like(x0), rndm_no)
    # make another rndm guess with n0 in G sums up to 1 and n0 in L equals 0 
    rndm_guesses_G = np.random.dirichlet(np.ones_like(x0[:6]), 1)[0]
    rndm_guesses_L = np.zeros_like(x0[6:])
    rndm_guesses_min3 = np.hstack((rndm_guesses_G, rndm_guesses_L))
    # another manual gess
    man_guess_G = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0])
    man_guess_L = np.array([0, 0.5, 0, 0, 0.5, 0])
    man_guess = np.hstack((man_guess_G, man_guess_L))

    guesses = np.zeros((rndm_no + 4, len(x0)))
    
    for i in range(rndm_no):
        
        guesses[i, :] = rndm_guesses[i]
    
    guesses[-4, :] = man_guess
    guesses[-3, :] = rndm_guesses_min3
    guesses[-2, :] = x0
    guesses[-1, :] = np.ones_like(x0)

    g_T_vals = np.zeros(len(guesses))
    x_eq_vals = np.zeros((len(guesses), x0.shape[0]))
    n_eq_vals = np.zeros((len(guesses), x0.shape[0]))

    for i, guess in enumerate(guesses):

        sol = minimize(g_T, guess, args=(T, p, type), method='SLSQP', constraints = cons, bounds=bnds, options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-5})
        # Define the local optimization method

        if sol.success:
            g_T_vals[i] = sol.fun
            success = True
            #x_eq_vals[i, :] = sol.x / np.sum(sol.x)
            n_eq_vals[i, :] = sol.x
        else:
            g_T_vals[i] = np.nan
            success = False
            #x_eq_vals[i, :] = np.nan
            n_eq_vals[i, :] = np.nan

    try:
        # now find the minimum g_T_value and respective x_eq
        min_idx = np.nanargmin(g_T_vals)
        g_T_value = g_T_vals[min_idx]
        #x_eq = x_eq_vals[min_idx,:]
        n_eq = n_eq_vals[min_idx,:]

    except ValueError:
        success = False
        #x_eq = np.nan
        g_T_value = np.nan
        n_eq = np.nan

    return _,success,g_T_value,n_eq"""

"""def calc_eq(T, p, x0, type='real gas'):
    '''
    Calculates the equilibrium composition of a gas mixture at a given temperature and pressure using basinhopping.

    Parameters
    ----------
        T: temperature in K (float)
        p: pressure in Pa (float)
        x0: inlet composition [CO2 H2 CH4 H2O CO C He Ar N2] (array-like)
        type: type of gas, choose from 'ideal gas' or 'real gas'

    Returns
    -------
        success: boolean, True if the minimization was successful, False otherwise
        g_T_value: minimum Gibbs free energy value found
        n_eq: equilibrium composition [CO2 H2 CH4 H2O CO C He Ar N2]
    '''
    p = p * 1e-5  # Convert pressure to bar

    n0, bnds, _ = calc_bounds(x0)
    
    cons = [{'type': 'eq', 'fun': element_balance, 'args': [n0]},
            {'type': 'eq', 'fun': isofug_cond, 'args': [T, p]},
            {'type': 'ineq', 'fun': lambda n: n}]  # Ensures non-negative values

    rndm_no = 1
    rndm_guesses = np.random.dirichlet(np.ones_like(x0), rndm_no)
    
    rndm_guesses_G = np.random.dirichlet(np.ones_like(x0[:6]), 1)[0]
    rndm_guesses_L = np.zeros_like(x0[6:])
    rndm_guesses_min3 = np.hstack((rndm_guesses_G, rndm_guesses_L))
    
    man_guess_G = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0])
    man_guess_L = np.array([0, 0.5, 0, 0, 0.5, 0])
    man_guess = np.hstack((man_guess_G, man_guess_L))
    
    guesses = np.zeros((rndm_no + 4, len(x0)))
    for i in range(rndm_no):
        guesses[i, :] = rndm_guesses[i]
    
    guesses[-4, :] = man_guess
    guesses[-3, :] = rndm_guesses_min3
    guesses[-2, :] = x0
    guesses[-1, :] = np.ones_like(x0)
    
    g_T_vals = np.zeros(len(guesses))
    n_eq_vals = np.zeros((len(guesses), x0.shape[0]))
    
    for i, guess in enumerate(guesses):
        minimizer_kwargs = {'method': 'SLSQP', 'constraints': cons, 'bounds': bnds, 'args': (T, p, type)}
        sol = basinhopping(g_T, guess, minimizer_kwargs=minimizer_kwargs, niter=100, stepsize=0.1)
        
        if sol.lowest_optimization_result.success:
            g_T_vals[i] = sol.fun
            n_eq_vals[i, :] = sol.x
        else:
            g_T_vals[i] = np.nan
            n_eq_vals[i, :] = np.nan
    
    try:
        min_idx = np.nanargmin(g_T_vals)
        g_T_value = g_T_vals[min_idx]
        n_eq = n_eq_vals[min_idx, :]
        success = True
    except ValueError:
        success = False
        g_T_value = np.nan
        n_eq = np.nan
    
    return success, g_T_value, n_eq"""

def calc_eq(T, p, x0, type='real gas'):
    '''
    Calculates the equilibrium composition of a gas mixture at one given temperature and pressure using global optimization.

    Parameters
    ----------
        T: temperature in K (float)
        p: pressure in Pa (float)
        x0: inlet composition [CO2 H2 CH4 H2O CO C He Ar N2]
        type: type of gas, choose from 'ideal gas' or 'real gas'

    Returns
    -------
        x_eq: equilibrium composition [CO2 H2 CH4 H2O CO C He Ar N2]
        success: boolean, True if the minimization was successful, False otherwise
    '''
    p = p * 1e-5  # Convert pressure to bar
    
    n0, bnds, _ = calc_bounds(x0)
    
    cons = [{'type': 'eq', 'fun': element_balance, 'args': [n0]},
            {'type': 'ineq', 'fun': lambda n: n}]  # Ensures all components are >= 0
    
    sol = shgo(g_T, bounds=bnds, args=(T, p, type), constraints=cons, n=100, iters=5, minimizer_kwargs={'method': 'SLSQP', 'constraints': cons, 'bounds': bnds})
    #sol = differential_evolution(g_T, bounds=bnds, args=(T, p, type), constraints=cons, tol=1e-5, x0=n0)

    if sol.success:
        g_T_value = sol.fun
        n_eq = sol.x
        success = True
    else:
        g_T_value = np.nan
        n_eq = np.nan
        success = False
    
    return success, g_T_value, n_eq

# testing of function
#p     = 30e5 #100e5          # enter pressure in Pa
#T     = 400# 200 + 273.15 # enter temperature in K
## enter mol fractions in 1:
#y_CO2  = 0.19
#y_H2  = 0.8-4e-20
#y_H2O  = 1e-20
#y_CO  = 0.01
#y_DME = 1e-20
#y_MeOH = 1e-20
#y_N2 = 1e-20
#
#n = np.array([y_CO2, y_H2, y_H2O, y_CO, y_DME, y_MeOH, y_N2]) # array containing amounts of substance in mol, assuming n = 1 mol
#

#print('eq', calc_eq(T,p,n))