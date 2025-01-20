import numpy as np
import scipy.constants

Glenn_coeffs_CH3OH = { # 200 - 1000 K
    "a_1": -2.416642886e5,
    "a_2": 4.032147190e3,
    "a_3": -2.046415436e1,
    "a_4": 6.903698070e-2,
    "a_5": -7.598932690e-5,
    "a_6": 4.598208360e-8,
    "a_7": -1.158706744e-11,
    "b_1": -4.433261170e4,
    "b_2": 1.400142190e2,
}

Glenn_coeffs_CH3OCH3 = { # 200 - 1000 K
    "a_1": -2.693103242e5,
    "a_2": 4.300709710e3,
    "a_3": -2.152788028e1,
    "a_4": 8.131833390e-2,
    "a_5": -8.295671320e-5,
    "a_6": 4.801911510e-8,
    "a_7": -1.188699808e-11,
    "b_1": -4.410237090e4,
    "b_2": 1.467666934e2,
}

Glenn_coeffs_C_low = { # 200 - 600 K (graphite)
    "a_1": 1.132856760e5, 
    "a_2": -1.980421677e3,
    "a_3": 1.365384188e1,
    "a_4": -4.636096440e-2,
    "a_5": 1.021333011e-4,
    "a_6": -1.082893179e-7,
    "a_7": 4.472258860e-11,
    "b_1": 8.943859760e3,
    "b_2": -7.295824740e1
}

Glenn_coeffs_C_high = { # 600 - 2000 K (graphite)
    "a_1": 3.356004410e5, 
    "a_2": -2.596528368e3,
    "a_3": 6.948841910e0,
    "a_4": -3.484836090e-3,
    "a_5": 1.844192445e-6,
    "a_6": -5.055205960e-10,
    "a_7": 5.750639010e-14,
    "b_1": 1.398412456e4,
    "b_2": -4.477183040e1
}

Glenn_coeffs_H2 = { # 200 - 1000 K
    "a_1": 4.078323210e4,
    "a_2": -8.009186040e2,
    "a_3": 8.214702010e0,
    "a_4": -1.269714457e-2,
    "a_5": 1.753605076e-5,
    "a_6": -1.202860270e-8,
    "a_7": 3.368093490e-12,
    "b_1": 2.682484665e3,
    "b_2": -3.043788844e1,}

Glenn_coeffs_O2 = { # 200 - 1000 K
    "a_1": -3.425563420e4,
    "a_2": 4.847000970e2,
    "a_3": 1.119010961e0,
    "a_4": 4.293889240e-3,
    "a_5": -6.836300520e-7,
    "a_6": -2.023372700e-9,
    "a_7": 1.039040018e-12,
    "b_1": -3.391454870e3,
    "b_2": 1.849699470e1,
}

Glenn_coeffs_H2O = { # 200 - 1000 K
    "a_1": -3.948 * 1e4,
    "a_2": 5.756 * 1e2,
    "a_3": 9.318 * 1e-1,
    "a_4": 7.223 * 1e-3,
    "a_5": -7.343 * 1e-6,
    "a_6": 4.955 * 1e-9,
    "a_7": -1.337 * 1e-12,
    "b_1": -3.304 * 1e4,
    "b_2": 1.724 * 1e1,
}

Glenn_coeffs_H2O_L = { # 373.15 - 600 K
    "a_1": 1.263631001e9,
    "a_2": -1.680380249e7,
    "a_3": 9.278234790e4,
    "a_4": -2.722373950e2,
    "a_5": 4.479243760e-1,
    "a_6": -3.919397430e-4,
    "a_7": 1.425743266e-7,
    "b_1": 8.113176880e7,
    "b_2": -5.134418080e5,
}

Glenn_coeffs_CH3OH_L = { # 175.61 - 390 K
    "a_1": -1.302004763e6,
    "a_2": 3.166984180e4,
    "a_3": -3.031242152e2,
    "a_4": 1.602231130e0,
    "a_5": -4.594507340e-3,
    "a_6": 6.990178310e-6,
    "a_7": -4.207388950e-9,
    "b_1": -1.656168201e5,
    "b_2": 1.514346642e3
}

coeffs = {
    "H2": Glenn_coeffs_H2,
    "CH3OH": Glenn_coeffs_CH3OH,
    "CH3OCH3": Glenn_coeffs_CH3OCH3,
    "C_low": Glenn_coeffs_C_low,
    "C_high": Glenn_coeffs_C_high,
    "O2": Glenn_coeffs_O2,
    "H2O": Glenn_coeffs_H2O,
    "H2O_L": Glenn_coeffs_H2O_L,
    "CH3OH_L": Glenn_coeffs_CH3OH_L
}

def Thermo_props(comp, T): #in SI units

    if comp == "C" and T < 600:
        
        coeffs[comp] = Glenn_coeffs_C_low

    elif comp == "C" and T >= 600:
        
        coeffs[comp] = Glenn_coeffs_C_high

    # heat capacity
    c_p = scipy.constants.R * (coeffs[comp]["a_1"] / T ** 2 + coeffs[comp]["a_2"] / T + coeffs[comp]["a_3"] + coeffs[comp]["a_4"] * T \
                                + coeffs[comp]["a_5"] * T**2 + coeffs[comp]["a_6"] * T**3 + coeffs[comp]["a_7"] * T**4)
    # enthalpy
    H = scipy.constants.R * T * (-coeffs[comp]["a_1"] / T**2 + coeffs[comp]["a_2"] * np.log(T) / T + coeffs[comp]["a_3"] + coeffs[comp]["a_4"] * T / 2 \
                                + coeffs[comp]["a_5"] * T**2 / 3 + coeffs[comp]["a_6"] * T**3 / 4 + coeffs[comp]["a_7"] * T**4 / 5 + coeffs[comp]["b_1"] / T)
    # entropy
    S = scipy.constants.R * (-coeffs[comp]["a_1"] / T ** 2 / 2 - coeffs[comp]["a_2"] / T + coeffs[comp]["a_3"] * np.log(T) + coeffs[comp]["a_4"] * T \
                                + coeffs[comp]["a_5"] * T**2 / 2 + coeffs[comp]["a_6"] * T**3 / 3 + coeffs[comp]["a_7"] * T**4 / 4 + coeffs[comp]["b_2"])

    return c_p, H, S

def delta_f_G(T, comp):

    if T < 600:
        C_str = "C_low"
    else:
        C_str = "C_high"

    if comp == "CH3OH":

        delta_f_H = Thermo_props("CH3OH", T)[1] - Thermo_props(C_str, T)[1] - 2 * Thermo_props("H2", T)[1] - 0.5 * Thermo_props("O2", T)[1]
        delta_f_S = Thermo_props("CH3OH", T)[2] - Thermo_props(C_str, T)[2] - 2 * Thermo_props("H2", T)[2] - 0.5 * Thermo_props("O2", T)[2]

    elif comp == "CH3OCH3":

        delta_f_H = Thermo_props("CH3OCH3", T)[1] - 2 * Thermo_props(C_str, T)[1] - 0.5 * Thermo_props("O2", T)[1] - 3 * Thermo_props("H2", T)[1]
        delta_f_S = Thermo_props("CH3OCH3", T)[2] - 2 * Thermo_props(C_str, T)[2] - 0.5 * Thermo_props("O2", T)[2] - 3 * Thermo_props("H2", T)[2]

    elif comp == "H2O":

        delta_f_H = Thermo_props("H2O", T)[1] - Thermo_props("H2", T)[1] - 0.5 * Thermo_props("O2", T)[1]
        delta_f_S = Thermo_props("H2O", T)[2] - Thermo_props("H2", T)[2] - 0.5 * Thermo_props("O2", T)[2]

    elif comp == "H2O_L":
        
        delta_f_H = Thermo_props("H2O_L", T)[1] - Thermo_props("H2", T)[1] - 0.5 * Thermo_props("O2", T)[1]
        delta_f_S = Thermo_props("H2O_L", T)[2] - Thermo_props("H2", T)[2] - 0.5 * Thermo_props("O2", T)[2]

    elif comp == "CH3OH_L":

        delta_f_H = Thermo_props("CH3OH_L", T)[1] - Thermo_props("C_low", T)[1] - 2 * Thermo_props("H2", T)[1] - 0.5 * Thermo_props("O2", T)[1]
        delta_f_S = Thermo_props("CH3OH_L", T)[2] - Thermo_props("C_low", T)[2] - 2 * Thermo_props("H2", T)[2] - 0.5 * Thermo_props("O2", T)[2]

    dfG0 = delta_f_H - T * delta_f_S

    return dfG0

# dfG = dfH - T * dfS --> can be calculated from enthalpy and entropy of elements


print(delta_f_G(298.15, "H2O"))
print(delta_f_G(500, "H2O"))

print(delta_f_G(298.15, "H2O_L"))