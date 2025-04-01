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

def summation_constraint(w):
    """function for the summation constraint

    :param w: array containing the molar fractions of the trial phase in 1
    :return: residual -> 0
    """

    return np.sum(w) - 1

def min_tpd(w, z, T, p, trial_phase):
    """function for minimizing the reduced tangent plane distance function

    :param z: array containing the molar fractions of the reference phase in 1
    :return: array containing the molar fractions of the trial phase in 1
    """

    cns = [{"type": "eq", "fun": summation_constraint},
            {"type": "ineq", "fun": lambda w: w}]
            #{"type": "eq", "fun": isofugacity_constraint, "args": (z, T, p)}]

    bnds = [(0, 1)] * len(w)
    res = minimize(tpd, w, args=(z, T, p, trial_phase), method="SLSQP", bounds=bnds, constraints=cns, options={"disp": False, "maxiter": 1000, "ftol": 1e-5})
    #res = basinhopping(tpd, w, minimizer_kwargs={"method": "SLSQP", "bounds": bnds, "constraints": cns, "options": {"disp": False, "maxiter": 1000, "ftol": 1e-5}, "args": (z, T, p, trial_phase)})

    return res.fun, res.x


