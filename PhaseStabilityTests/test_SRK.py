def phi_SRK_test(z, T, p, phase="vapor"): # --> Gmehling 2019 example
    """
    function for calculation of fugacity coefficients @ T, p from Soave-Redlich-Kwong-equation of state according to 1972 (doi.org/10.1016/0009-2509(72)80096-4)

    :param z: array containing molar fractions in gas phase of CO2, H2, H2O, CO, MeOH and N2 in 1
    :param T: temperature in K
    :param p: pressure in Pa
    :return: fugacity coefficients of CO2, H2, H2O, CO, MeOH and N2 @ T, p in 1
    """

    # Parameter         # CO2   # H2     # H2O   # CO  # MeOH   # N2
    omega_i = np.array([0.01, 0.045])      # array containing acentric factors in 1 (Perry's)
    T_c_i   = np.array([190.63,126.15])      # array contaning critical temperatures in K (Perry's)
    p_c_i   = np.array([  46.17, 33.94]) * 1e5 # array containing critical pressures in Pa (Perry's)
    #p_i =    np.array([0, 0, 0.1277, 0, 0.2359, 0]) # array containing empirical polar parameters in 1 (Mathias 1983)
    p_i = np.zeros(2) # array containing empirical polar parameters in 1 (Mathias 1983)
    # Green, Don W.; Perry, Robert H. (2003): Perry's chemical engineers' handbook. 7th ed., internat. ed., [Nachdr.]. New York: McGraw-Hill.

    T_r_i = T / T_c_i                                               # array containing reduced temperatures in 1
    #m_i     = 0.48508 + 1.55171 * omega_i - 0.15613 * omega_i**2    # array containing SRK parameters m in 1
    m_i = 0.48 + 1.574 * omega_i - 0.176 * omega_i**2
    alpha_i = (1 + m_i * (1 - T_r_i**0.5) - p_i*(1-T_r_i)*(0.7-T_r_i))**2                       # array containing SRK parameters alpha in 1

    #print("alpha_i", alpha_i)

    # critical values a and b
    R = 8.3145 # J/(mol K)
    a_c_i = 0.42747 * R**2 * T_c_i**2 / p_c_i
    b_c_i = 0.08664 * R * T_c_i / p_c_i
    a_i_T = a_c_i * alpha_i

    print("a_i_T", a_i_T)

    # mixing rules
    a_ij = np.zeros((2, 2))
    #k_ij = np.array([[0, 0.1164, 0.3, 0.1164, 0.1, 0],
    #                [0.1164, 0, -0.745, -0.0007, -0.125, 0],
    #                [0.3, -0.745, 0, -0.474, -0.075, 0],
    #               [0.1164, -0.0007, -0.474, 0, -0.37, 0],
     #               [0.1, -0.125, -0.075, -0.37, 0, 0],
    #                [0, 0, 0, 0, 0, 0]])

    k_ij = np.zeros((2, 2))

    k_ij[0, 1] = 0.0267
    k_ij[1, 0] = 0.0267

    for i in range(2):
        for j in range(2):
            a_ij[i, j] = np.sqrt(a_i_T[i] * a_i_T[j]) * (1 - k_ij[i, j])

    #print("a_ij", a_ij)

    a = np.sum(np.dot(z[i], z[j] * a_ij[i, j]) for i in range(2) for j in range(2))
    b = np.dot(z, b_c_i)

    #print("b_ci", b_c_i)
    print("b", b)
    print("a", a)

    ratio_b = b_c_i / b
    ratio_a = np.dot(z, a_ij) / a

    A = a * p / R**2 / T**2
    B = b * p / R / T

    #print("A new: ", A)
    #print("B new: ", B)

    # use np.roots
    sol = ces.solve(1, -1, A - B - B**2, -A * B)
    sol = sol[np.isreal(sol)]
    Z = np.real(sol)

    if phase == "vapor":
        Z = np.max(Z)
    else:
        Z = np.min(Z)

    print("v_liq", Z * R * T / p)

    phi = np.exp(ratio_b * (Z - 1) - np.log(Z - B) - A / B * (2 * ratio_a - ratio_b) * np.log(1 + B / Z))

    return phi

def phi_SRK_VLE_old(z, T, p):
    '''
    function for calculation of fugacity coefficients @ T, p from Soave-Redlich-Kwong-equation of state according to 1972 (doi.org/10.1016/0009-2509(72)80096-4)

    :param z: array containing molar fractions in gas phase of CO2, H2, H2O, CO, MeOH and N2 in 1
    :param T: temperature in K
    :param p: pressure in Pa
    :return: fugacity coefficients of CO2, H2, H2O, CO, MeOH and N2 @ T, p in 1
    '''

    z_G = z[:6]
    z_L = z[6:]

    # Parameter         # CO2   # H2     # H2O   # CO  # MeOH # N2
    omega_i = np.array([ 0.225, -0.217,  0.343,  0.045, 0.565, 0.011])      # array containing acentric factors in 1 (Perry's)0.037
    T_c_i   = np.array([304.12,  32.98, 647.13, 132.85, 512.64, 190.56])      # array contaning critical temperatures in K (Perry's)126.2
    p_c_i   = np.array([  73.74, 12.93,  219.4,   34.94, 80.97,  45.99]) * 1e5 # array containing critical pressures in Pa (Perry's) 33.9
    p_i =    np.array([0, 0, 0.1277, 0, 0.2359, 0]) # array containing empirical polar parameters in 1 (Mathias 1983)
    # Green, Don W.; Perry, Robert H. (2003): Perry's chemical engineers' handbook. 7th ed., internat. ed., [Nachdr.]. New York: McGraw-Hill.

    T_r_i = T / T_c_i                                      # array containing reduced temperatures in 1
    m_i     = 0.480 + 1.574 * omega_i - 0.176 * omega_i**2 # array containing SRK parameters m in 1
    alpha_i = (1 + m_i * (1 - T_r_i**0.5)- p_i*(1-T_r_i)*(0.7-T_r_i))**2              # array containing SRK parameters alpha in 1

    ratio_a_G = (alpha_i**0.5 * T_c_i / p_c_i**0.5) / (np.dot(z_G, (alpha_i**0.5 * T_c_i / p_c_i**0.5))) # array containing ratios sqrt(a_i/a) in 1
    ratio_b_G = (T_c_i / p_c_i) / (np.dot(z_G, (T_c_i / p_c_i)))                                         # array containing ratios (b_i/b) in 1

    ratio_a_L = (alpha_i**0.5 * T_c_i / p_c_i**0.5) / (np.dot(z_L, (alpha_i**0.5 * T_c_i / p_c_i**0.5))) # array containing ratios sqrt(a_i/a) in 1
    ratio_b_L = (T_c_i / p_c_i) / (np.dot(z_L, (T_c_i / p_c_i)))                                         # array containing ratios (b_i/b) in 1

    A_G = 0.42747 * p / T**2 * (np.dot(z_G, (T_c_i * alpha_i**0.5 / p_c_i**0.5)))**2 # parameter A
    B_G = 0.08664 * p / T * np.dot(z_G, (T_c_i / p_c_i))                             # parameter B

    A_L = 0.42747 * p / T**2 * (np.dot(z_L, (T_c_i * alpha_i**0.5 / p_c_i**0.5)))**2 # parameter A
    B_L = 0.08664 * p / T * np.dot(z_L, (T_c_i / p_c_i))                             # parameter B

    # use np.roots
    sol_G = ces.solve(1, -1, A_G - B_G - B_G**2, -A_G * B_G)
    sol_G = sol_G[np.isreal(sol_G)]
    Z_G = np.real(sol_G)

    sol_L = ces.solve(1, -1, A_L - B_L - B_L**2, -A_L * B_L)
    sol_L = sol_L[np.isreal(sol_L)]
    Z_L = np.real(sol_L)

    Z_G = np.max(Z_G)
    Z_L = np.min(Z_L)

    print("Z_L", Z_L)
    print("Z_G", Z_G)

    phi_G = np.exp(ratio_b_G * (Z_G - 1) - np.log(Z_G - B_G) - A_G / B_G * (2 * ratio_a_G - ratio_b_G) * np.log(1 + B_G / Z_G))
    phi_L = np.exp(ratio_b_L * (Z_L - 1) - np.log(Z_L - B_L) - A_L / B_L * (2 * ratio_a_L - ratio_b_L) * np.log(1 + B_L / Z_L))

    phi = np.hstack((phi_G, phi_L))

    return phi

def phi_SRK_old(z, T, p, phase="vapor"):
    """
    function for calculation of fugacity coefficients @ T, p from Soave-Redlich-Kwong-equation of state according to 1972 (doi.org/10.1016/0009-2509(72)80096-4)

    :param z: array containing molar fractions in gas phase of CO2, H2, H2O, CO, MeOH and N2 in 1
    :param T: temperature in K
    :param p: pressure in Pa
    :return: fugacity coefficients of CO2, H2, H2O, CO, MeOH and N2 @ T, p in 1
    """

    # Parameter         # CO2   # H2     # H2O   # CO  # MeOH   # N2
    omega_i = np.array([0.225, -0.217,  0.344,  0.045, 0.565,   0.037])      # array containing acentric factors in 1 (Perry's)
    T_c_i   = np.array([304.12,  32.98, 647.14, 132.85, 512.64, 126.2])      # array contaning critical temperatures in K (Perry's)
    p_c_i   = np.array([  73.74, 12.93,  220.64,   34.94, 80.97,   33.9]) * 1e5 # array containing critical pressures in Pa (Perry's)
    p_i =    np.array([0, 0, 0.1277, 0, 0.2359, 0]) # array containing empirical polar parameters in 1 (Mathias 1983)
    #p_i = np.zeros(6) # array containing empirical polar parameters in 1 (Mathias 1983)
    # Green, Don W.; Perry, Robert H. (2003): Perry's chemical engineers' handbook. 7th ed., internat. ed., [Nachdr.]. New York: McGraw-Hill.

    T_r_i = T / T_c_i                                               # array containing reduced temperatures in 1
    m_i     = 0.48508 + 1.55171 * omega_i - 0.15613 * omega_i**2    # array containing SRK parameters m in 1
    alpha_i = (1 + m_i * (1 - T_r_i**0.5) - p_i*(1-T_r_i)*(0.7-T_r_i))**2                       # array containing SRK parameters alpha in 1

    ratio_a = (alpha_i**0.5 * T_c_i / p_c_i**0.5) / (np.dot(z, (alpha_i**0.5 * T_c_i / p_c_i**0.5))) # array containing ratios sqrt(a_i/a) in 1
    ratio_b = (T_c_i / p_c_i) / (np.dot(z, (T_c_i / p_c_i)))                                         # array containing ratios (b_i/b) in 1

    A = 0.42747 * p / T**2 * (np.dot(z, (T_c_i * alpha_i**0.5 / p_c_i**0.5)))**2 # parameter A
    B = 0.08664 * p / T * np.dot(z, (T_c_i / p_c_i))                             # parameter B

   #print("A old: ", A)
   # print("B old: ", B)

    # use np.roots
    sol = ces.solve(1, -1, A - B - B**2, -A * B)
    sol = sol[np.isreal(sol)]
    Z = np.real(sol)

    if phase == "vapor":
        Z = np.max(Z)
    else:
        Z = np.min(Z)

    phi = np.exp(ratio_b * (Z - 1) - np.log(Z - B) - A / B * (2 * ratio_a - ratio_b) * np.log(1 + B / Z))

    return phi