import numpy as np
from SRK import phi_SRK, phi_SRK_VLE
from scipy.optimize import minimize, basinhopping

def initial_guesses(z, T, p, trial_phase):
                        # CO2, H2, H2O, CO, MeOH, CH4
    p_c_i = np.array([  73.74, 12.93,  220.64,   34.94, 80.97,  45.99 ]) * 1e5
    T_c_i   = np.array([304.12,  32.98, 647.14, 132.85, 512.64, 190.56 ])
    omega_i = np.array([0.225, -0.217,  0.344,  0.045, 0.565,   0.011 ])
    K_i = p_c_i / p * np.exp(5.37 * (1 + omega_i) * (1 - T_c_i / T))

    if trial_phase == "vapor":
        guesses = z * K_i
        S = np.sum(guesses) 
        guesses = guesses / S
    else:
        guesses = z / K_i
        S = np.sum(guesses)
        guesses = guesses / S

    #guess_1 = K_i * z
    #guess_2 = z / K_i
    #guess_3 = np.exp(np.log(z) + np.log(phi_z))
    #guess_4 = z / np.sum(z)

    #guesses = [guess_1, guess_2, guess_3, guess_4]

    return guesses

def init_2Ph_calc(y_tr, p, T, trial_phase):

    p_c_i = np.array([  73.74, 12.93,  220.64,   34.94, 80.97,  45.99 ]) * 1e5
    T_c_i   = np.array([304.12,  32.98, 647.14, 132.85, 512.64, 190.56 ])
    omega_i = np.array([0.225, -0.217,  0.344,  0.045, 0.565,   0.011 ])
    K_i = p_c_i / p * np.exp(5.37 * (1 + omega_i) * (1 - T_c_i / T))

    if trial_phase == "vapor":

        x_i = y_tr / K_i

    else:
        
        x_i = y_tr * K_i

    return np.hstack((x_i, y_tr))



def vapor_pressure(T):

    p_c_i = np.array([  73.74, 12.93,  220.64,   34.94, 80.97,  45.99 ]) * 1e5
    T_c_i   = np.array([304.12,  32.98, 647.14, 132.85, 512.64, 190.56 ])
    T_r_i = T / T_c_i
                    # CO2, H2, H2O, CO, MeOH, CH4
    A = np.array([-7.026565, - 4.836839, -7.870154 , -6.194175 , -8.726980 , -6.024057 ])
    B = np.array([1.527245, 0.943915, 1.906774, 1.319639, 1.450050, 1.268690])
    C = np.array([-2.246311, 0.763880, -2.310330, -0.943212, -2.771770, -0.570278   ])
    D = np.array([-2.630030, - 0.467794, -2.063390, -2.001545, -0.723874, -1.375360 ])

    p_i_s = p_c_i * np.exp((1 / T_r_i) * (A * (1- T_r_i) + B * (1 - T_r_i)**1.5 + C * (1 - T_r_i)**2.5 + D * (1 - T_r_i)**5))

    return p_i_s

def tpd(w, z, T, p, trial_phase):
    """function for calculating the reduced tangent plane distance function

    :param w: array containing the molar fractions of the trial phase in 1
    :param z: array containing the molar fractions of the reference phase in 1
    :return: reduced tangent plane distance in 1
    """

    phi_z, Z_z = phi_SRK(z, T, p)
    phi_w, Z_w = phi_SRK(w, T, p, phase=trial_phase)

    res = np.sum(w * (np.log(w) + np.log(phi_w) - np.log(z) - np.log(phi_z)))

    return res

def TPD(W, z, T, p, trial_phase):
    """function for calculating the reduced tangent plane distance function

    :param w: array containing the molar fractions of the trial phase in 1
    :param z: array containing the molar fractions of the reference phase in 1
    :return: reduced tangent plane distance in 1
    """

    w = W / np.sum(W)

    phi_w = phi_SRK(w, T, p, trial_phase)
    phi_z = phi_SRK(z, T, p)

    res = 1 + np.sum(W * (np.log(W) + np.log(phi_w) - np.log(z) - np.log(phi_z) - 1))

    return res

def summation_constraint(w):
    """function for the summation constraint

    :param w: array containing the molar fractions of the trial phase in 1
    :return: residual -> 0
    """

    return np.sum(w) - 1

def phase_stability_analysis(y_trial, z, T, p, trial_phase):

    phi_ref, _ = phi_SRK(z, T, p)
    
    #S = np.sum(Y_trial)
    #y_trial = Y_trial / S

    if trial_phase == "vapor":
        phi_trial, Z_trial = phi_SRK(y_trial, T, p, phase="vapor")
        f_trial = phi_trial * y_trial
        f_ref = phi_ref * z
        fug_ratio = f_ref / f_trial #* (1 / S)
    else:
        phi_trial, Z_trial = phi_SRK(y_trial, T, p, phase="liquid")
        f_trial = phi_trial * y_trial
        f_ref = phi_ref * z
        fug_ratio = f_trial / f_ref #* S

    res = np.sum(fug_ratio - 1) ** 2

    return res


def check_phase_stability(guess, z, T, p, trial_phase):
    
    # set x variables greater than 0 as constraints
    bnds = [(1e-20, None)] * len(guess)

    res = minimize(phase_stability_analysis, guess, args=(z, T, p, trial_phase), bounds=bnds, method="SLSQP", options={"disp": False, "maxiter": 1000, "ftol": 1e-10})

    Y_res = res.x
    y_res = Y_res / np.sum(Y_res)
    S = np.sum(Y_res)

    return S, y_res

def min_tpd(w, z, T, p, trial_phase):
    """function for minimizing the reduced tangent plane distance function

    :param z: array containing the molar fractions of the reference phase in 1
    :return: array containing the molar fractions of the trial phase in 1
    """

    cns = [{"type": "eq", "fun": summation_constraint},
            {"type": "ineq", "fun": lambda w: w}]
            #{"type": "eq", "fun": isofugacity_constraint, "args": (z, T, p)}]

    bnds = [(0, 1)] * len(w)
    #res = minimize(tpd, w, args=(z, T, p), method="SLSQP", bounds=bnds, constraints=cns, options={"disp": False, "maxiter": 1000, "ftol": 1e-5})
    res = basinhopping(tpd, w, minimizer_kwargs={"method": "SLSQP", "bounds": bnds, "constraints": cns, "options": {"disp": False, "maxiter": 1000, "ftol": 1e-5}, "args": (z, T, p, trial_phase)})

    return res.fun, res.x

def min_TPD(W, z, T, p, trial_phase):
    """function for minimizing the reduced tangent plane distance function

    :param z: array containing the molar fractions of the reference phase in 1
    :return: array containing the molar fractions of the trial phase in 1
    """

    bnds = [(1e-20, None)] * len(W)

    res = minimize(tpd, W, args=(z, T, p, trial_phase), method="SLSQP", bounds=bnds, options={"disp": False, "maxiter": 1000, "ftol": 1e-5})

    return res.fun, res.x


print(vapor_pressure(700))


