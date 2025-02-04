import numpy as np
from SRK import phi_SRK
from scipy.optimize import minimize, basinhopping

def tpd(w, z, T, p):
    """function for calculating the reduced tangent plane distance function

    :param w: array containing the molar fractions of the trial phase in 1
    :param z: array containing the molar fractions of the reference phase in 1
    :return: reduced tangent plane distance in 1
    """

    phi_z, Z_z = phi_SRK(z, T, p, phase="vapor")
    phi_w, Z_w = phi_SRK(w, T, p, phase="liquid")

    res = np.sum(w * (np.log(w) + np.log(phi_w) - np.log(z) - np.log(phi_z)))

    return res

def TPD(W, z, T, p, phase="vapor"):
    """function for calculating the reduced tangent plane distance function

    :param w: array containing the molar fractions of the trial phase in 1
    :param z: array containing the molar fractions of the reference phase in 1
    :return: reduced tangent plane distance in 1
    """

    w = W / np.sum(W)

    phi_w = phi_SRK(w, T, p, phase=phase)
    phi_z = phi_SRK(z, T, p, phase=phase)

    res = 1 + np.sum(W * (np.log(W) + np.log(phi_w) - np.log(z) - np.log(phi_z) - 1))

    return res

def summation_constraint(w):
    """function for the summation constraint

    :param w: array containing the molar fractions of the trial phase in 1
    :return: residual -> 0
    """

    return np.sum(w) - 1

def min_tpd(w, z, T, p):
    """function for minimizing the reduced tangent plane distance function

    :param z: array containing the molar fractions of the reference phase in 1
    :return: array containing the molar fractions of the trial phase in 1
    """

    cns = [{"type": "eq", "fun": summation_constraint},
            {"type": "ineq", "fun": lambda w: w}]

    bnds = [(0, 1)] * len(w)
    res = minimize(tpd, w, args=(z, T, p), method="SLSQP", bounds=bnds, constraints=cns, options={"disp": False, "maxiter": 1000, "ftol": 1e-5})
    #res = basinhopping(tpd, w, minimizer_kwargs={"method": "SLSQP", "bounds": bnds, "constraints": cns, "options": {"disp": False, "maxiter": 1000, "ftol": 1e-5}, "args": (z, T, p)})

    return res.fun, res.x

def min_TPD(W, z, T, p, phase="vapor"):
    """function for minimizing the reduced tangent plane distance function

    :param z: array containing the molar fractions of the reference phase in 1
    :return: array containing the molar fractions of the trial phase in 1
    """

    cns = {"type": "ineq", "fun": lambda W: W}

    res = minimize(tpd, W, args=(z, T, p, phase), method="SLSQP", constraints=cns, options={"disp": False, "maxiter": 1000, "ftol": 1e-5})

    return res.fun, res.x


