# TKA_Mo_240503_2_fugacity_coefficient_V2
# function for calculation of fugacity coefficients according to Soave 1972 (https://doi.org/10.1016/0009-2509(72)80096-4) for mixtures
# contains CO2, H2, CH4, H2O, CO, He, Ar and N2

import numpy as np
from scipy.optimize import root
import sympy as sp


def phi_Soave(y, T, p):
    """
    function for calculation of fugacity coefficients @ T, p from Soave-Redlich-Kwong-equation of state according to 1972 (doi.org/10.1016/0009-2509(72)80096-4)

    :param y: array containing molar fractions in gas phase of CO2, H2, H2O, CO, DME, MeOH and N2 in 1
    :param T: temperature in K
    :param p: pressure in Pa
    :return: fugacity coefficients of CO2, H2, H2O, CO, MeOH and N2 @ T, p in 1
    """

    y_i   = y # array of gas phase molar fractions (CO2, H2, H2O, CO, MeOH and N2) in 1

    # Parameter         # CO2   # H2     # H2O   # CO  # MeOH # N2
    omega_i = np.array([ 0.224, -0.215,  0.343,  0.048, 0.565, 0.037])      # array containing acentric factors in 1 (Perry's)
    T_c_i   = np.array([304.21,  33.19, 647.13, 132.92, 512.5, 126.2])      # array contaning critical temperatures in K (Perry's)
    p_c_i   = np.array([  73.9,   13.1,  219.4,   34.9, 80.8,   33.9]) * 1e5 # array containing critical pressures in Pa (Perry's)
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
    Z = Z_solve.x # largest positive root of Z
    #print('Z', Z_solve.x, 'T', T-273.15)
    
    # calculation of phi from ln phi_i = b_i/b*(Z-1)-ln(Z-B)-A/B*(2 sqrt(a_i/a)-b_i/b)*ln(1+B/Z)
    res = np.exp(ratio_b * (Z - 1) - np.log(Z - B) - A / B * (2 * ratio_a - ratio_b) * np.log(1 + B / Z)) # array of fugacity coefficients in 1

    # print('A', A)
    # print('B', B)
    # print('Z', Z)
    # print('m', m_i)
    # print('alpha', alpha_i)
    return res

'''# testing of function
p     = 100e5          # enter pressure in Pa
T     = 200 + 273.15 # enter temperature in K
# enter mol fractions in 1:
y_CO2  = 0.2
y_H2  = 0.1
y_H2O  = 0.1
y_CO  = 0.1
y_DME = 0.2
y_MeOH = 0.2
y_N2 = 0.1
#
n = np.array([y_CO2, y_H2, y_H2O, y_CO, y_DME, y_MeOH, y_N2]) # array containing amounts of substance in mol, assuming n = 1 mol
#
#print('phi', phi_Soave(n, T, p))'''


def phi_Soave_2(z, T, p):
    """
    function for calculation of fugacity coefficients @ T, p from Soave-Redlich-Kwong-equation of state according to 1972 (doi.org/10.1016/0009-2509(72)80096-4)

    :param y: array containing molar fractions in gas phase of CO2, H2, H2O, CO, DME, MeOH and N2 in 1
    :param T: temperature in K
    :param p: pressure in Pa
    :return: fugacity coefficients of CO2, H2, H2O, CO, MeOH and N2 @ T, p in 1
    """

    z_i_G   = z[:-2] # array of gas phase molar fractions (CO2, H2, H2O, CO, MeOH and N2) in 1
    z_i_L   = z[-2:] # array of liquid phase molar fractions (MeOH and N2) in 1

    # Parameter         # CO2   # H2     # H2O   # CO  # MeOH # N2
    omega_i = np.array([ 0.224, -0.215,  0.343,  0.048, 0.565, 0.037])      # array containing acentric factors in 1 (Perry's)
    T_c_i   = np.array([304.21,  33.19, 647.13, 132.92, 512.5, 126.2])      # array contaning critical temperatures in K (Perry's)
    p_c_i   = np.array([  73.9,   13.1,  219.4,   34.9, 80.8,   33.9]) * 1e5 # array containing critical pressures in Pa (Perry's)
    # Green, Don W.; Perry, Robert H. (2003): Perry's chemical engineers' handbook. 7th ed., internat. ed., [Nachdr.]. New York: McGraw-Hill.
    # params of liq
    omega_liq = np.array([0.343, 0.565]) # array containing acentric factors in 1 (Perry's)
    T_c_liq   = np.array([647.13, 512.5]) # array contaning critical temperatures in K (Perry's)
    p_c_liq   = np.array([219.4, 80.8]) * 1e5 # array containing critical pressures in Pa (Perry's)
    T_r_i_liq = T / T_c_liq                                      # array containing reduced temperatures in 1
    m_i_liq     = 0.480 + 1.574 * omega_liq - 0.176 * omega_liq**2 # array containing SRK parameters m in 1
    alpha_i_liq = (1 + m_i_liq * (1 - T_r_i_liq**0.5))**2              # array containing SRK parameters alpha in 1

    ratio_a_liq = (alpha_i_liq**0.5 * T_c_liq / p_c_liq**0.5) / (np.dot(z_i_L, (alpha_i_liq**0.5 * T_c_liq / p_c_liq**0.5))) # array containing ratios sqrt(a_i/a) in 1
    ratio_b_liq = (T_c_liq / p_c_liq) / (np.dot(z_i_L, (T_c_liq / p_c_liq)))                                         # array containing ratios (b_i/b) in 1

    A_L = 0.42747 * p / T**2 * (np.dot(z_i_L, (T_c_liq * alpha_i_liq**0.5 / p_c_liq**0.5)))**2 # parameter A
    B_L = 0.08664 * p / T * np.dot(z_i_L, (T_c_liq / p_c_liq))                             # parameter B


    T_r_i = T / T_c_i                                      # array containing reduced temperatures in 1
    m_i     = 0.480 + 1.574 * omega_i - 0.176 * omega_i**2 # array containing SRK parameters m in 1
    alpha_i = (1 + m_i * (1 - T_r_i**0.5))**2              # array containing SRK parameters alpha in 1

    ratio_a = (alpha_i**0.5 * T_c_i / p_c_i**0.5) / (np.dot(z_i_G, (alpha_i**0.5 * T_c_i / p_c_i**0.5))) # array containing ratios sqrt(a_i/a) in 1
    ratio_b = (T_c_i / p_c_i) / (np.dot(z_i_G, (T_c_i / p_c_i)))                                         # array containing ratios (b_i/b) in 1

    A_G = 0.42747 * p / T**2 * (np.dot(z_i_G, (T_c_i * alpha_i**0.5 / p_c_i**0.5)))**2 # parameter A
    B_G = 0.08664 * p / T * np.dot(z_i_G, (T_c_i / p_c_i))                             # parameter B

    # calculation of Z by root finding: Z³-Z²+Z(A-B-B²)-AB = 0
    def Z_root(Z, A, B):
        res = Z**3 - Z**2 + Z * (A - B - B**2) - A * B
        return res
    
    Z_solve = root(Z_root, np.array([1]), args = (A_G, B_G))
    Z_vap = Z_solve.x # largest positive root of Z

    Z_solve_liq = root(Z_root, np.array([0, 0.5, 1]), args = (A_L, B_L))
    # take smallest positive root of Z
    Z_liq = Z_solve_liq.x[np.where(Z_solve_liq.x > 0)[0][0]]

    #print('T', T, 'p', p, 'Z_liq', Z_solve_liq.x)

    res_liq = np.exp(ratio_b_liq * (Z_liq - 1) - np.log(Z_liq - B_L) - A_L / B_L * (2 * ratio_a_liq - ratio_b_liq) * np.log(1 + B_L / Z_liq)) # array of fugacity coefficients in 1
    res_vap = np.exp(ratio_b * (Z_vap - 1) - np.log(Z_vap - B_G) - A_G / B_G * (2 * ratio_a - ratio_b) * np.log(1 + B_G / Z_vap)) # array of fugacity coefficients in 1

    f_L = p * z_i_L * res_liq
    f_V = p * z_i_G * res_vap

    f_H2O = np.array([f_L[0], f_V[2]])
    f_MeOH = np.array([f_L[1], f_V[4]])

    return res_liq, res_vap, f_H2O, f_MeOH


# testing of function
p     = 100e5          # enter pressure in Pa
T     = 200 + 273.15 # enter temperature in K
# enter mol fractions in 1:
z_CO2  = 0.2
z_H2  = 0.1
z_H2O  = 0.1
z_CO  = 0.1
z_MeOH = 0.2
z_N2 = 0.1
x_H2O = 0.5
x_MeOH = 0.5    
#
z = np.array([z_CO2, z_H2, z_H2O, z_CO, z_MeOH, z_N2, x_H2O, x_MeOH]) # array containing amounts of substance in mol, assuming n = 1 mol
#
#print('phi', phi_Soave_2(z, T, p))
