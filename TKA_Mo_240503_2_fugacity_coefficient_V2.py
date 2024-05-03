# TKA_Mo_240503_2_fugacity_coefficient_V2
# function for calculation of fugacity coefficients according to Soave 1972 (https://doi.org/10.1016/0009-2509(72)80096-4) for mixtures
# contains CO2, H2, CH4, H2O, CO, He, Ar and N2

import numpy as np
from scipy.optimize import root


def phi_Soave(y, T, p):
    """
    function for calculation of fugacity coefficients @ T, p from Soave-Redlich-Kwong-equation of state according to 1972 (doi.org/10.1016/0009-2509(72)80096-4)

    :param n: array containing molar fractions in gas phase of CO2, H2, CH4, H2O, CO, He, Ar and N2 in mol
    :param T: temperature in K
    :param p: pressure in Pa
    :return: fugacity coefficients of CO2, H2, CH4, H2O, CO, He, Ar and N2 @ T, p in 1
    """

    y_i   = y # array of gas phase molar fractions (CO2, H2, CH4, H2O, CO, He, Ar and N2) in 1

    # Parameter         # CO2   # H2    # CH4    # H2O   # CO    # He    # Ar    # N2
    omega_i = np.array([ 0.224, -0.215,   0.011,  0.343,  0.048, -0.388,      0, 0.037]) # array containing acentric factors in 1 (Perry's)
    T_c_i   = np.array([304.21,  33.19, 190.564, 647.13, 132.92,    5.2, 150.86, 126.2]) # array contaning critical temperatures in K (Perry's)
    p_c_i   = np.array([  73.9,   13.2,    45.9,  219.4,   34.9,    2.3,   49.0,  33.9]) * 1e5 # array containing critical pressures in Pa (Perry's)
    # Green, Don W.; Perry, Robert H. (2003): Perry's chemical engineers' handbook. 7th ed., internat. ed., [Nachdr.]. New York: McGraw-Hill.

    T_r_i = T / T_c_i                                      # array containing reduced temperatures in 1
    m_i     = 0.480 + 1.574 * omega_i - 0.176 * omega_i**2 # array containing SRK parameters m in 1
    alpha_i = (1 + m_i * (1 - T_r_i**0.5))**2              # array containing SRK parameters alpha in 1

    ratio_a = (alpha_i**0.5 * T_c_i / p_c_i**0.5) / (np.dot(y_i, (alpha_i**0.5 * T_c_i / p_c_i**0.5))) # array containing ratios sqrt(a_i/a) in 1
    ratio_b = (T_c_i / p_c_i) / (np.dot(y_i, (T_c_i / p_c_i)))                                         # array containing ratios (b_i/b) in 1

    A = 0.42747 * p / T**2 * (np.dot(y_i, (T_c_i * alpha_i**0.5 / p_c_i**0.5)))**2 # parameter A
    B = 0.08664 * p / T * np.dot(y_i, (T_c_i / p_c_i))                             # parameter B

    # calculation of Z by root finding: Z³-Z²+Z(A-B-B²)-AB = 0
    def Z_root(Z, A, B):
        res = Z**3 - Z**2 + Z * (A - B - B**2) - A * B
        return res

    Z_solve = root(Z_root, np.array([1]), args = (A, B))
    Z = Z_solve.x

    # calculation of phi from ln phi_i = b_i/b*(Z-1)-ln(Z-B)-A/B*(2 sqrt(a_i/a)-b_i/b)*ln(1+B/Z)
    res = np.exp(ratio_b * (Z - 1) - np.log(Z - B) - A / B * (2 * ratio_a - ratio_b) * np.log(1 + B / Z)) # array of fugacity coefficients in 1

    # print('A', A)
    # print('B', B)
    # print('Z', Z)
    # print('m', m_i)
    # print('alpha', alpha_i)
    return res

# # testing of function
# p     = 8123232.79 #100e5          # enter pressure in Pa
# T     = 708.894416# 200 + 273.15 # enter temperature in K
# # enter mol fractions in 1:
# y_CO2  = 0.8
# y_H2  = 0.2
# y_CH4 = 0
# y_H2O  = 0
# y_CO  = 0
# y_He = 0
# y_Ar = 0
# y_N2 = 0
#
# n = np.array([y_CO2, y_H2, y_CH4, y_H2O, y_CO, y_He, y_Ar, y_N2]) # array containing amounts of substance in mol, assuming n = 1 mol
#
# print('phi', phi_Soave(n, T, p))