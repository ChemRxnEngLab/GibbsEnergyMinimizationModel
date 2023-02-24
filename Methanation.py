# Modell TKA-Mo-230203-1 (MethAmmT-V1)
# calculation of methanation chemical equilibrium by Gibbs free energy minimization
# calculation of fugacity coefficients by Soave-Redlich-Kwong EOS or ideal gas assumption

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd
from ICIW_Plots import cyclers as ICIW_cyclers
import matplotlib

def dfg(T):
    """
    function for determination of Gibbs free energy of formation of a species @ T from NIST-JANAF tables [1] by a polynomic fit, range of validity: 0 - 6000 K

    :param T: temperature in K
    :return: Gibbs free energy of formation in J / mol
    """

    coeff = np.array([[ -4.4712899950965e-18, 7.69027397691646e-14, -4.83454740516897e-10, 2.06356409761266e-6, -4.28205970848287e-3, -3.93254377825293e2],  # CO2
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # H2
                      [-1.19480733487367e-16, 2.05212901276465e-12,  -1.30585665138136e-8, 3.77285316328865e-5,  6.34019284855666e-2, -7.14155077436689e1],  # CH4
                      [                    0, 1.02813542552485e-13,  -1.46197827311474e-9, 7.46216055871485e-6,  4.27766908921636e-2, -2.41321516149705e2],  # H2O
                      [                    0,                    0,                     0, 1.60737136748171e-6, -9.02786221059026e-2, -1.11442697976118e2],  # CO
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # C
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # He
                      [                    0,                    0,                     0,                   0,                    0,                   0],  # Ar
                      [                    0,                    0,                     0,                   0,                    0,                   0]]) # N2

    T_poly = np.array([T**5, T**4, T**3, T**2, T, 1])

    res = np.matmul(T_poly, np.transpose(coeff)) * 1000
    return res

def phi_Soave(n, T, p):
    """
    function for determination of fugacity coefficients of all gaseous species @ T, p from Soave-Redlich-Kwong Equation of State [2]

    :param T: temperature in K
    :param p: pressure in bar
    :return: fugacity coefficients of CO2, H2, CH4, H2O, CO, He, Ar and N2 @ T, p in 1
    """
    n_gas = np.delete(n, 5) # removing carbon (solid) from vector containing molar amounts

    # parameters
    R       = 8.314 # universal gas constant in J / mol K
    M_i     = np.array([0.044,  0.002, 0.016, 0.018,  0.028,  0.004,   0.040, 0.028]) # vector containing molar masses of CO2, H2, CH4, H2O, CO, He, Ar and N2 in kg / mol
    omega_i = np.array([0.224, -0.215, 0.011, 0.343,  0.048, -0.388,   0.000, 0.037]) # vector containing acentric factors of CO2, H2, CH4, H2O, CO, He, Ar and N2 in 1 [3]
    T_c_i   = np.array([304.2,  33.18, 190.6, 647.0, 134.45,    5.2,  150.86, 126.2]) # vector containing critical temperatures of CO2, H2, CH4, H2O, CO, He, Ar and N2 in K [3]
    p_c_i   = np.array([ 73.8,  13.00,  46.1, 220.6,     35,  2.274, 48.9805,  33.9]) # vector containing critical pressures of CO2, H2, CH4, H2O, CO, He, Ar and N2 in bar [3]
    K_ij    = np.zeros([len(n_gas), len(n_gas)]) # matrix containing SRK Binary Interaction Parameters, rows and columns CO2, H2, CH4, H2O, CO, He, Ar and N2, respectively, in 1

    R_s_i = R / M_i               # vector containing specific gas constants of CO2, H2, CH4, H2O, CO, He, Ar and N2 in J / kg K
    T_r_i = T / T_c_i             # vector containing reduced temperatures of CO2, H2, CH4, H2O, CO, He, Ar and N2 in 1
    x_i   = n_gas / np.sum(n_gas) # vector containing mole fractions in the gas phase of CO2, H2, CH4, H2O, CO, He, Ar and N2 in 1
    R_s   = np.dot(x_i, R_s_i)    # average specific gas constant weighted by mole fractions in J / kg K

    m_i     = 0.480 + 1.574 * omega_i - 0.176 * omega_i**2    # vector containing SRK parameters m for CO2, H2, CH4, H2O, CO, He, Ar and N2 in 1
    alpha_i = (1 + m_i * (1 - T_r_i**0.5))**2                 # vector containing SRK parameters alpha for CO2, H2, CH4, H2O, CO, He, Ar and N2 in 1
    a_i     = 0.42747 * alpha_i * R_s_i**2 * T_c_i**2 / p_c_i # vector containing SRK parameters a for CO2, H2, CH4, H2O, CO, He, Ar and N2 in m^5 / s² kg
    b_i     = 0.08664 * R_s_i * T_c_i / p_c_i                 # vector containing SRK parameters b for CO2, H2, CH4, H2O, CO, He, Ar and N2 in m³ / kg
    A_ij    = (1-K_ij) * (np.transpose(np.ones([len(a_i), len(a_i)]) * a_i) * a_i)**0.5 # matrix containing a_ij, rows and columns CO2, H2, CH4, H2O, CO, He, Ar and N2, respectively, in 1

    a = np.sum(np.diag(np.matmul(np.transpose(np.transpose(np.ones([len(x_i), len(x_i)]) * x_i) * x_i), A_ij))) # SRK parameters a of the mixture in m^5 / s² kg
    b = np.dot(x_i, b_i)        # SRK parameters b of the mixture in m³ / kg
    A = a * p / (R_s**2 * T**2) # SRK parameters A in 1
    B = b * p / (R_s * T)       # SRK parameters B in 1

    # calculation of Z
    def Z_root(Z, A, B):
        res = Z**3 - Z**2 + Z * (A - B - B**2) - A * B
        return res

    Z_solve = root(Z_root, np.array([1]), args = (A, B))
    Z = Z_solve.x

    res = np.exp((Z - 1) * b_i / b - np.log(Z - B) - A / B * (2 * a_i**0.5 / a**0.5 - b_i / b) * np.log(1 + B / Z))
    res_C = np.insert(res, 5, 1) # fugacity coefficient of carbon is set to 1
    return res_C

def g_T(n, T, p, type):
    """
    function for determination of the total Gibbs free energy to be minimized [3, 4]

    :param n: vector containing molar amounts of CO2, H2, CH4, H2O, CO, C, He, Ar and N2
    :param T: temperature in K
    :param p: pressure in bar
    :return: total Gibbs free energy in J / mol
    """

    dfgi = dfg(T) # Gibbs free energy of formation in J / mol
    phii = np.ones_like(n)

    n_gas = np.delete(n, 5)  # removing carbon (solid) from vector containing molar amounts

    if type == 'ideal gas':
        phii = phii
    elif type == 'real gas':
        phii = phi_Soave(n, T, p)
    else:
        print('Please choose type of gas from given options: ideal gas or real gas')

    for i in range(n.shape[0]):
        if n[i] <= 0:
            n[i] = 1e-20

    R  = 8.314 # universal gas constant in J / mol K
    p0 = 1 # standard pressure in bar

    C_corr = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]) # vector for removal of carbon in the gas term in G

    res = np.dot(n, dfgi) + R * T * np.dot((n * C_corr), np.log(phii * p * n / (p0 * np.sum(n_gas))))
    return res


# Parameter
x0 = np.empty(9)
x0[0] = 0.2       # initial mole fraction of CO2
x0[1] = 0.8-7e-20 # initial mole fraction of H2
x0[2] = 1e-20     # initial mole fraction of CH4
x0[3] = 1e-20     # initial mole fraction of H2O
x0[4] = 1e-20     # initial mole fraction of CO
x0[5] = 1e-20     # initial mole fraction of C
x0[6] = 1e-20     # initial mole fraction of He
x0[7] = 1e-20     # initial mole fraction of Ar
x0[8] = 1e-20     # initial mole fraction of N2
n0    = x0 * 1    # initial molar amount in mol

# Solving
max_C   = n0[0] + n0[2] + n0[4] + n0[5]     # molar amount of carbon in the system in mol
max_H   = 2 * n0[1] + 4 * n0[2] + 2 * n0[3] # molar amount of hydrogen in the system in mol
max_O   = 2 * n0[0] + n0[3] + n0[4]         # molar amount of oxygen in the system in mol
max_He  = n0[6]                             # molar amount of helium in the system in mol
max_Ar  = n0[7]                             # molar amount of argon in the system in mol
max_N   = n0[8]                             # molar amount of nitrogen in the system
max_CO2 = min(max_C, 0.5 * max_O)           # maximum possible molar amount of CO2 in mol
max_H2  = 0.5 * max_H                       # maximum possible molar amount of H2 in mol
max_CH4 = min(max_C, 0.25 * max_H)          # maximum possible molar amount of CH4 in mol
max_H2O = min(0.5 * max_H, max_O)           # maximum possible molar amount of H2O in mol
max_CO  = min(max_C, max_O)                 # maximum possible molar amount of CO in mol
max_N2  = 0.5 * max_N                       # maximum possible molar amount of N2 in mol

bnds = ((0, max_CO2), (0, max_H2), (0, max_CH4), (0, max_H2O), (0, max_CO), (0, max_C), (0, max_He), (0, max_Ar), (0, np.inf))
init = np.ones_like(n0)

def element_balance(n, n0):
    """
    function for checking the element balance as a constraint for the minimization

    :param n0: vector containing initial molar amounts of CO2, H2, CH4, H2O, CO, C, He, Ar and N2
    :return: residual -> 0
    """
    # element-species matrix (C, O, H, He, Ar, N)
    A = np.array([[1, 2, 0, 0, 0, 0],  # CO2
                  [0, 0, 2, 0, 0, 0],  # H2
                  [1, 0, 4, 0, 0, 0],  # CH4
                  [0, 1, 2, 0, 0, 0],  # H2O
                  [1, 1, 0, 0, 0, 0],  # CO
                  [1, 0, 0, 0, 0, 0],  # C
                  [0, 0, 0, 1, 0, 0],  # He
                  [0, 0, 0, 0, 1, 0],  # Ar
                  [0, 0, 0, 0, 0, 2]]) # N2
    res = np.matmul(n, A) - np.matmul(n0, A)
    return res

cons = {'type': 'eq', 'fun': element_balance, 'args': [n0]}

# validation [3]
p = np.array([1.01325]) # p in bar
T = np.linspace(200 + 273.15, 800 + 273.15, 25) # T in K
type = 'real gas' # choose type of gas from 'ideal gas' and 'real gas'

x = np.ones([p.shape[0], T.shape[0], n0.shape[0]])

for i in range(p.shape[0]):
    for j in range(T.shape[0]):
        if np.sum(x0) != 1:
            print('Please check specified inlet composition!')

        else:
            sol = minimize(g_T, init, args=(T[j], p[i], type), method='SLSQP', bounds=bnds, constraints = cons, options = {'disp': 'True', 'maxiter': 1000, 'ftol': 1e-12})
            n = sol.x
            x[i, j] = n / np.sum(n)
            print(sol.success)
            if round(np.sum(n / np.sum(n)), 5) != 1:
                print('Error at ', p, 'bar and ', T, 'K!')
            if sol.success == False:
                sol = minimize(g_T, x[i, j - 1], args=(T[j], p[i], type), method='SLSQP', bounds=bnds, constraints=cons, options={'disp': 'True', 'maxiter': 1000, 'ftol': 1e-12})
                n = sol.x
                x[i, j] = n / np.sum(n)
                print(sol.success)

            print(x[i, j], '(', T[j], ')')

# data import - validation data [3]
csv_data_Gao = pd.read_csv('data_Gao_CO2.csv',  # read csv file
                           sep = ';')

# convert read data into numpy array
data_Gao         = csv_data_Gao.to_numpy()

T_CO2_Gao        = data_Gao[:, 0]
x_CO2_Gao        = data_Gao[:, 1:7]

plt.style.use('ICIWstyle')
font = {'size': 10}
matplotlib.rc('font', **font)

# CO2 methanation
fig, axs = plt.subplots()
cyc1 = ICIW_cyclers.ICIW_colormap_cycler('hsv', 9, start = 0.1, stop = 1)
axs.set_prop_cycle(cyc1)
axs.plot(T - 273.15,  x[0, :, 0],     '-',                 label = 'CO$_2$')
axs.plot(T - 273.15,  x[0, :, 1],     '-',                 label = 'H$_2$')
axs.plot(T - 273.15,  x[0, :, 2],     '-',                 label = 'CH$_4$')
axs.plot(T - 273.15,  x[0, :, 3],     '-',                 label = 'H$_2$O')
axs.plot(T - 273.15,  x[0, :, 4],     '-',                 label = 'CO')
axs.plot(T - 273.15,  x[0, :, 5],     '-',                 label = 'C')
axs.plot(T - 273.15,  x[0, :, 6],     '-',                 label = 'He')
axs.plot(T - 273.15,  x[0, :, 7],     '-',                 label = 'Ar')
axs.plot(T - 273.15,  x[0, :, 8],     '-',                 label = 'N$_2$')
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

# References
# [1] T.C. Allison, NIST-JANAF Thermochemical Tables - SRD 13, 2013 (accessed 16 November 2022). https://doi.org/10.18434/T42S31
# [2] G. Soave, Equilibrium constants from a modified Redlich-Kwong equation of state, Chemical Engineering Science 27 (1972) 1197–1203. https://doi.org/10.1016/0009-2509(72)80096-4.
# [3] R.H. Perry (Ed.), Perry's chemical engineers' handbook, 7th ed., McGraw-Hill, Montréal, 1997.
# [4] J. Gao, Y. Wang, Y. Ping, D. Hu, G. Xu, F. Gu, F. Su, A thermodynamic analysis of methanation reactions of carbon oxides for the production of synthetic natural gas, RSC Adv. 2 (2012) 2358. https://doi.org/10.1039/c2ra00632d
