import numpy as np

from Global_Params import V_OP

# ==========================================
# Table 1 Parameters for DPM Physics
# ==========================================
C_sc=2.134  # Weibull slope coefficient for DOT distribution (Table I parameter)

P_sc=0.847  # Weibull slope exponent for DOT distribution (Table I parameter)

W_mu=-0.3665  # Scale parameter for DOT distribution (Table I parameter)

mu_sc=0.914  # Scale factor for eta_DOT (Table I parameter)

# ==========================================
# Dynamic Percolation Model (DPM) Physics
# ==========================================

def calc_beta_DOT(S):
    """ Equation (2):
    Calculates the Weibull slope of the DOT distribution based on spacing S."""
    return C_sc * (S**P_sc)

def calc_eta_DOT(S):
    """ Equation (3):
    Calculates the local Weibull scale parameter for DOT distribution."""
    beta_DOT = calc_beta_DOT(S)
    return mu_sc * np.exp(-W_mu / beta_DOT)

def calc_Es(S):
    """
    Equation (17) 
    Calculates the characteristic E-field (Es) for a given spacing S. The paper's regression is based on the calibration data in Table II, which shows Es values for different spacings.

    """
    return 1 / (0.85 + 2.91 / S)

def calc_m(S):
    """
    Equation (18) 
    Calculates the spacing-dependent Weibull slope (m) for time-to-breakdown based on the characteristic E-field (Es) and the local Weibull scale parameter for DOT distribution (eta_DOT).

    CORRECTION: The paper lists this as m^-1, but Table II and experimentations shows the regression is actually for m.
    """
    return (0.5659 * (S**(-0.455)))

def calc_ma(S):
    """Equation (19): Spacing-dependent E-field acceleration factor"""
    return 20.66 / np.tanh(0.073 * S)

def calc_beta_tBD(S):
    """Calculates the local Weibull shape parameter for time-to-breakdown"""
    beta_DOT = calc_beta_DOT(S)
    return calc_m(S) * beta_DOT

def calc_eta_tBD(S):
    """
    Calculates the local Weibull scale parameter for time-to-breakdown.
    Corrected to scale from the S^2 supercell reference area down to 1 nm^2.
    """
    eta_DOT = calc_eta_DOT(S)
    
    # Local E-field assuming V/S matches the paper's scaling for 0.7V
    E_local = V_OP / S  
    
    m_val = calc_m(S)
    ma_val = calc_ma(S)
    Es_val = calc_Es(S)
    
    # Equation (8) yielding the tBD scale parameter for an area of S^2
    eta_tBD_supercell = (E_local / Es_val)**(-ma_val) * (eta_DOT**(1.0 / m_val))
    
    # Weibull slope
    beta_tBD = calc_beta_tBD(S)
    
    # Scale the eta parameter from S^2 down to 1 nm^2
    # Formula: eta_1 = eta_supercell * (A_supercell / A_1)^(1/beta)
    A_supercell = S**2
    A_unit = 1.0
    
    eta_tBD_1 = eta_tBD_supercell * (A_supercell / A_unit)**(1.0 / beta_tBD)
    
    return eta_tBD_1
