import numpy as np
from scipy.optimize import minimize, basinhopping
from SRK import phi_SRK, phi_SRK_VLE
import warnings
from thermo_coeffs import delta_f_G
from PhaseStability import min_tpd, initial_guesses, min_TPD, check_phase_stability, init_2Ph_calc


def dfg(T):
    """
    function for determination of Gibbs free energy of formation of a species @ T from NIST-JANAF tables by a polynomic fit, range of validity: 0 - 6000 K

    :param T: temperature in K
    :return: Gibbs free energy of formation in J / mol
    """

    coeff = np.array([[ -4.4712899950965e-18, 7.69027397691646e-14, -4.83454740516897e-10, 2.06356409761266e-6, -4.28205970848287e-3, -3.93254377825293e2],  # CO2
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # H2
                      [                    0, 1.02813542552485e-13,  -1.46197827311474e-9, 7.46216055871485e-6,  4.27766908921636e-2, -2.41321516149705e2],  # H2O
                      [                    0,                    0,                     0, 1.60737136748171e-6, -9.02786221059026e-2, -1.11442697976118e2],  # CO
                      [-1.19480733487367e-16, 2.05212901276465e-12,  -1.30585665138136e-8, 3.77285316328865e-5,  6.34019284855666e-2, -7.14155077436689e1],  # CH4
                      ])

    T_poly = np.array([T**5, T**4, T**3, T**2, T, 1])

    res = np.matmul(T_poly, np.transpose(coeff)) * 1000
    # calculate DME and MEOH separately
    dfg_meoh = delta_f_G(T, 'CH3OH')
    dfg_n2 = 0
    #res = np.append(res, [dfg_meoh, dfg_n2])

    # insert CH3OH value at 4th index
    res = np.insert(res, 4, dfg_meoh)

    return res

def g_T(n, T, p):
    """
    function for determination of the total Gibbs free energy to be minimized

    :param n: vector containing molar amounts of CO2, H2, H2O, CO, MeOH and N2
    :param T: temperature in K
    :param p: pressure in Pa
    :return: total Gibbs free energy in J / mol
    """

    n = np.maximum(n, 1e-20) # avoid division by zero
    z = n / np.sum(n)        # molar fractions in gas phase
    dfgi = dfg(T)            # Gibbs free energy of formation of all species in J / mol @ T in ideal gas state_
    phii, _ = phi_SRK(z, T, p)  # function needs pressure in Pa and molar fractions in gas phase

    R  = 8.314 # universal gas constant in J / mol K
    p0 = 1e5 # standard pressure in Pa

    res = np.dot(n, dfgi) + R * T * np.dot(n, np.log(phii * p * z / p0))   
    
    return res

def g_T_VLE(n, T, p):
    """
    function for determination of the total Gibbs free energy to be minimized

    :param n: vector containing molar amounts of CO2, H2, H2O, CO, MeOH and N2
    :param T: temperature in K
    :param p: pressure in Pa
    :return: total Gibbs free energy in J / mol
    """

    n = np.maximum(n, 1e-20) # avoid division by zero
    n_G = n[:6]
    n_L = n[6:]

    z_G = n_G / np.sum(n_G)        # molar fractions in gas phase
    z_L = n_L / np.sum(n_L)        # molar fractions in liquid phase
    z = np.hstack((z_G, z_L))

    dfgi = dfg(T)            # Gibbs free energy of formation of all species in J / mol @ T in ideal gas state_
    phii = phi_SRK_VLE(z, T, p)  # function needs pressure in Pa and molar fractions in gas phase

    R  = 8.314 # universal gas constant in J / mol K
    p0 = 1e5 # standard pressure in Pa

    # check if phii contains NaN values
    #if np.isnan(phii).any():
    #    print("NaN values in phii @ T, p", T, p)

    res_G = np.dot(n_G, dfgi) + R * T * np.dot(n_G, np.log(phii[:6] * p * z_G / p0))
    res_L = np.dot(n_L, dfgi) + R * T * np.dot(n_L, np.log(phii[6:] * p * z_L / p0))

    res = res_G + res_L

    return res

def element_balance(n, n0):
    """
    function for checking the element balance as a constraint for the minimization

    :param n0: vector containing initial molar amounts of CO2, H2, H2O, CO, CH3OH and N2
    :return: residual -> 0
    """

    # element-species matrix (C, O, H, N)
    '''A = np.array([[1, 2, 0, 0],  # CO2
                  [0, 0, 2, 0],  # H2
                  [0, 1, 2, 0],  # H2O
                  [1, 1, 0, 0],  # CO
                  [1, 1, 4, 0],  # MeOH
                  [0, 0, 0, 2]]) # N2'''

    
    # element-species matrix (C, O, H)
    A = np.array([[1, 2, 0],  # CO2
                  [0, 0, 2],  # H2
                  [0, 1, 2],  # H2O
                  [1, 1, 0],  # CO
                  [1, 1, 4],  # MeOH
                  [1, 0, 4]]) # CH4
    
    res = np.matmul(n, A) - np.matmul(n0, A)
    
    return res

def element_balance_VLE(n, n0):
    """
    function for checking the element balance as a constraint for the minimization

    :param n0: vector containing initial molar amounts of CO2, H2, H2O, CO, CH3OH and N2
    :return: residual -> 0
    """

    # element-species matrix (C, O, H, N)
    '''A = np.array([[1, 2, 0, 0],  # CO2
                  [0, 0, 2, 0],  # H2
                  [0, 1, 2, 0],  # H2O
                  [1, 1, 0, 0],  # CO
                  [1, 1, 4, 0],  # MeOH
                  [0, 0, 0, 2]]) # N2'''
    
    A = np.array([[1, 2, 0],  # CO2
                  [0, 0, 2],  # H2
                  [0, 1, 2],  # H2O
                  [1, 1, 0],  # CO
                  [1, 1, 4],  # MeOH
                  [1, 0, 4]]) # CH4

    res = np.matmul(n[:6], A) - np.matmul(n0[:6], A) + np.matmul(n[6:], A) - np.matmul(n0[6:], A)

    return res

def calc_bounds(x0):
    
    n0     = x0 * 1                                            # initial molar amount in mol                          
    max_C   = n0[0] + n0[3] + n0[4] #+ n0[5]                            # molar amount of carbon in the system
    max_H   = 2 * n0[1] + 2 * n0[2] + 4 * n0[4]  #+ 4 * n0[5]              # molar amount of hydrogen in the system
    max_O   = 2 * n0[0] + n0[2] + n0[3] + n0[4]                # molar amount of oxygen in the system
    #max_N   = 2 * n0[5]                                        # molar amount of nitrogen in the system
    max_CO2 = min(max_C, 0.5 * max_O)                          # maximum possible molar amount of CO2 in mol
    max_H2  = 0.5 * max_H                                      # maximum possible molar amount of H2 in mol
    max_H2O = min(0.5 * max_H, max_O)                          # maximum possible molar amount of H2O in mol
    max_CO  = min(max_C, max_O)                                # maximum possible molar amount of CO in mol
    max_MeOH = min(max_C, max_O, 0.25 * max_H)                 # maximum possible molar amount of MeOH in mol
    max_CH4 = min(max_C, 0.25 * max_H)                            # maximum possible molar amount of CH4 in mol
    #max_N2  = 0.5 * max_N                                      # maximum possible molar amount of N2 in mol
    
    bnds = ((0, max_CO2), (0, max_H2), (0, max_H2O), (0, max_CO), (0, max_MeOH), (0, 1.0001 * n0[5]))#(0, max_N2))
    init = np.ones_like(n0)
    
    return n0,bnds,init 

def calc_bounds_VLE(x0):

    n0 = x0 * 1  # initial molar amount in mol
    n_G0 = n0[:6]
    n_L0 = n0[6:]
    n_t0 = n_G0 + n_L0

    max_C = n_t0[0] + n_t0[3] + n_t0[4] #+ n_t0[5]  # molar amount of carbon in the system
    max_H = 2 * n_t0[1] + 2 * n_t0[2] + 4 * n_t0[4] #+ 4 * n_t0[5]  # molar amount of hydrogen in the system
    max_O = 2 * n_t0[0] + n_t0[2] + n_t0[3] + n_t0[4]  # molar amount of oxygen in the system
    #max_N = 2 * n_t0[5]  # molar amount of nitrogen in the system
    max_CO2 = min(max_C, 0.5 * max_O)  # maximum possible molar amount of CO2 in mol
    max_H2 = 0.5 * max_H  # maximum possible molar amount of H2 in mol
    max_H2O = min(0.5 * max_H, max_O)  # maximum possible molar amount of H2O in mol
    max_CO = min(max_C, max_O)  # maximum possible molar amount of CO in mol
    max_MeOH = min(max_C, max_O, 0.25 * max_H)  # maximum possible molar amount of MeOH in mol
    max_CH4 = min(max_C, 0.25 * max_H)  # maximum possible molar amount of CH4 in mol
   # max_N2 = 0.5 * max_N  # maximum possible molar amount of N2 in mol

    bnds = ((0, max_CO2), (0, max_H2), (0, max_H2O), (0, max_CO), (0, max_MeOH), (0, 1.0001 * n_t0[5]))#(0, max_N2))
    bnds += bnds
    init = np.ones_like(n0)

    return n0, bnds, init

def isofug_cond(n, T, p):

    n = np.maximum(n, 1e-20)  # avoid division by zero

    n_G = n[:6]
    n_L = n[6:]

    z_G = n_G / np.sum(n_G)  # molar fractions in gas phase
    z_L = n_L / np.sum(n_L)  # molar fractions in liquid phase
    z = np.hstack((z_G, z_L))

    phii = phi_SRK_VLE(z, T, p)  # function needs pressure in Pa and molar fractions in gas phase

    phii_G = phii[:6]
    phii_L = phii[6:]

    res = np.zeros(6)

    for i in range(6):
        
        res[i] = z_G[i] * phii_G[i] - z_L[i] * phii_L[i]

    return res


def calc_eq(T, p, x0):
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
    #p = p * 1e-5  # Convert pressure to bar
    
    n0, bnds, _ = calc_bounds(x0)
    
    cons = [{'type': 'eq', 'fun': element_balance, 'args': [n0]},
            {'type': 'ineq', 'fun': lambda n: n},
            # set CH4 mole to equal CH4 n0 mole amounts
            {'type': 'eq', 'fun': lambda n: n[5] - n0[5]}
            ]  # Ensures all components are >= 0
    
    init_guess_1 = np.array([0.2, 0.5, 0.1, 0.0001, 0.10, 1e-10])
    init_guess_2 = x0

    init_guess = [init_guess_1, init_guess_2]

    g_T_values = np.zeros(len(init_guess))
    n_eq_values = np.zeros((len(init_guess), 6))

    for i in range(2):  
        sol = basinhopping(g_T, x0=init_guess[i], minimizer_kwargs={'method': 'SLSQP', 'bounds': bnds, 'constraints': cons, 'args': (T, p), 'options': {'disp': False, 'maxiter': 1000, 'ftol': 1e-5}})
    # solve with slsqp only
    #sol = minimize(g_T, x0=init_guess[i], bounds=bnds, constraints=cons, args=(T, p), method='SLSQP', options={'disp': False, 'maxiter': 1000, 'ftol': 1e-5})

    success = False
    if sol.success:
        g_T_values[i] = sol.fun
        n_eq_values[i] = sol.x
        success = True

    if success:
        g_T_value = g_T_values[np.argmin(g_T_values)]
        n_eq = n_eq_values[np.argmin(g_T_values)]
        x_eq = n_eq / np.sum(n_eq)

    #sol = minimize(g_T, x0=x0, bounds=bnds, args=(T, p), constraints=cons, method='SLSQP', options={'disp': False, 'maxiter': 1000, 'ftol': 1e-5})
    #sol = basinhopping(g_T, x0=x0, minimizer_kwargs={'method': 'SLSQP', 'bounds': bnds, 'constraints': cons, 'args': (T, p), 'options': {'disp': False, 'maxiter': 1000, 'ftol': 1e-5}})

    if sol.success:

        print("current T and p", T, p, "(after convergence of g_T one-phase)")
        
        g_T_value = sol.fun
        n_eq = sol.x
        x_eq = n_eq / np.sum(n_eq)
        success = True

        # check phase stability criterion
        #w = np.random.dirichlet(np.ones(6), size=5)
        w = np.zeros((2, 6))
        w[0] = initial_guesses(x_eq, T, p, 'vapor')
        w[1] = initial_guesses(x_eq, T, p, 'liquid')

        #print("w init", w)

        min_tpd_vals = np.zeros(2)
        x_L = np.zeros((len(w), 6))

        #print("min_tpd_vals init", min_tpd_vals)

        for i in range(len(min_tpd_vals)):
            
            #min_tpd_vals[i], x_L[i, :] = min_tpd(w[i], x_eq, T, p)
            #min_tpd_vals[i], x_L[i, :] = min_TPD(w[i], x_eq, T, p)
            if i == 0:
                min_tpd_vals[i], x_L[i, :] = min_tpd(w[i], x_eq, T, p, 'vapor')
            else:
                min_tpd_vals[i], x_L[i, :] = min_tpd(w[i], x_eq, T, p, 'liquid')
            #min_tpd_vals[i], x_L[i, :] = check_phase_stability(w, x_eq, T, p, 'vapor')

        #print("min_tpd_vals", min_tpd_vals)

        if min(min_tpd_vals) < 0:

            if min_tpd_vals[0] < min_tpd_vals[1]:
                trial_phase = 'vapor'
            else:
                trial_phase = 'liquid'

            #print("min_tpd_vals", min_tpd_vals)

            print("T in K", T)
            print("p in bae", p*1e-5)
            print("min_tpd", min(min_tpd_vals))
            #print("Z value init gas comp", phi_SRK(x_eq, T, p, 'vapor')[1], phi_SRK(x_eq, T, p, 'liquid')[1])
            
            warnings.warn("Phase stability criterion not fulfilled. Check results carefully.")
            PhaseStability = False
            x_L_init_guess = x_L[np.argmin(min_tpd_vals), :]
            x_L0 = np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20])
            x0_VLE = np.hstack((x0, x_L0))
            n0_VLE, bnds_VLE, _ = calc_bounds_VLE(x0_VLE)

            n_t0_VLE = np.sum(n0_VLE)

            cons_VLE = [{'type': 'eq', 'fun': element_balance_VLE, 'args': [n0_VLE]},
                        #{'type': 'ineq', 'fun': lambda n: n},
                        {'type': 'eq', 'fun': lambda n: n[5] + n[11] - n0_VLE[5] - n0_VLE[11]},
                        {'type': 'eq', 'fun': isofug_cond, 'args': (T, p)}]
            
            #print("Z value init gas comp", phi_SRK(x_eq, T, p, 'vapor'))
            #print("Z value init liq comp", phi_SRK(x_eq, T, p, 'liquid'))

            init_guess_1 = np.hstack((x_eq, x_L_init_guess))
            init_guess_2 = x0_VLE
            init_guess_3 = np.hstack((x_L_init_guess, x_L_init_guess))
            
            #np.hstack((x_eq, x_eq)) * 0.5 * n_t0_VLE
            init_guess_4 = init_2Ph_calc(x_L_init_guess, p, T, trial_phase)

            init_guess_5 = np.hstack((x_eq, x_eq))


            #init_guess = initial_guesses(x_eq, T, p)
            init_guess = [init_guess_1, init_guess_2, init_guess_3, init_guess_4, init_guess_5]
            g_T_values = np.zeros(len(init_guess))
            n_eq_values = np.zeros((len(init_guess), 12))

            for i in range(len(init_guess)):

                success = False

                sol_VLE = basinhopping(g_T_VLE, x0=init_guess[i], minimizer_kwargs={'method': 'SLSQP', 'bounds': bnds_VLE, 'constraints': cons_VLE, 'args': (T, p), 'options': {'disp': False, 'maxiter': 1000, 'ftol': 1e-5}})
                # solve with slsqp only
                #sol_VLE = minimize(g_T_VLE, x0=init_guess[i], bounds=bnds_VLE, constraints=cons_VLE, args=(T, p), method='SLSQP', options={'disp': False, 'maxiter': 1000, 'ftol': 1e-5})
                
                if sol_VLE.success:
                    
                    g_T_values[i] = sol_VLE.fun
                    n_eq_values[i] = sol_VLE.x
                    success = True

            
            if success:
                
                g_T_value = g_T_values[np.argmin(g_T_values)]
                n_eq = n_eq_values[np.argmin(g_T_values)]
                x_G_eq = n_eq[:6] / np.sum(n_eq[:6])
                x_L_eq = n_eq[6:] / np.sum(n_eq[6:])

                K_ij = x_G_eq / x_L_eq
                if np.sum(np.log(K_ij)**2) < 1e-4:

                    print("Trivial solution found.")

                x_eq = np.hstack((x_G_eq, x_L_eq))

            else:
                print("VLE calculation failed.")
                #success = False

            #sol_VLE = basinhopping(g_T_VLE, x0=init_guess, minimizer_kwargs={'method': 'SLSQP', 'bounds': bnds_VLE, 'constraints': cons_VLE, 'args': (T, p), 'options': {'disp': False, 'maxiter': 1000, 'ftol': 1e-5}})

            '''if sol_VLE.success:
                g_T_value = sol_VLE.fun
                n_eq = sol_VLE.x
                x_G_eq = n_eq[:6] / np.sum(n_eq[:6])
                x_L_eq = n_eq[6:] / np.sum(n_eq[6:])
                x_eq = np.hstack((x_G_eq, x_L_eq))
                success = True

            else:
                print("VLE calculation failed.")
                success = False'''

        else:
            
            PhaseStability = True

    else:
        g_T_value = np.nan
        n_eq = np.nan
        x_eq = np.nan
        success = False
        PhaseStability = False

        print("one phase calculation failed")

    if success and PhaseStability:
        zero_array = np.zeros(6)
        x_eq = np.hstack((x_eq, zero_array))
        n_eq = np.hstack((n_eq, zero_array))

        return success, g_T_value, n_eq, x_eq
    
    elif success and not PhaseStability:

        return success, g_T_value, n_eq, x_eq
    
    else:
        
        return success, np.nan, np.nan, np.nan

"""def calc_eq(T, p, x0, type='real gas'):
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
    #p = p * 1e-5  # Convert pressure to bar
    
    n0, bnds, _ = calc_bounds(x0)
    
    cons = [{'type': 'eq', 'fun': element_balance, 'args': [n0]},
            {'type': 'ineq', 'fun': lambda n: n}]  # Ensures all components are >= 0
    
    #sol = minimize(g_T, x0=x0, bounds=bnds, args=(T, p, type), constraints=cons, method='SLSQP', options={'disp': False, 'maxiter': 1000, 'ftol': 1e-10})

    sol = basinhopping(g_T, x0=x0, minimizer_kwargs={'method': 'SLSQP', 'bounds': bnds, 'constraints': cons, 'args': (T, p, type), 'options': {'disp': False, 'maxiter': 1000, 'ftol': 1e-10}})

    if sol.success:

        g_T_value = sol.fun
        n_eq = sol.x
        x_eq = n_eq / np.sum(n_eq)
        success = True
        
    else:
        g_T_value = np.nan
        n_eq = np.nan
        x_eq = np.nan
        success = False
    
    return success, g_T_value, n_eq, x_eq"""