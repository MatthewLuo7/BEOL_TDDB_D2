import numpy as np
from Global_Params import V_OP

# ==========================================
# Table 1 Parameters for DPM Physics
# ==========================================
C_sc = 2.134   # Weibull slope coefficient for DOT distribution
P_sc = 0.847   # Weibull slope exponent for DOT distribution
W_mu = -0.3665 # Scale parameter for DOT distribution
mu_sc = 0.914  # Scale factor for eta_DOT

# ==========================================
# sqrt(E) Model Parameters (Spacing-Dependent)
# ==========================================
# The sqrt(E) model is motivated by Poole-Frenkel conduction
# in low-k dielectrics, where the leakage current (and hence
# degradation rate) depends on exp(-gamma * sqrt(E)).
#
# The acceleration factor relative to Es is:
#   AF = exp(gamma(S) * (sqrt(Es) - sqrt(E)))
#
# When E < Es:  sqrt(Es) - sqrt(E) > 0,  AF > 1  (longer life)
# When E = Es:  AF = 1  (reference condition)
#
# gamma(S) is derived by force-equivalence with the paper's
# calibrated power-law m_a(S):
#   m_a(S) * ln(E2/E1) = gamma(S) * (sqrt(E2) - sqrt(E1))
#   => gamma(S) = m_a(S) * ln(E2/E1) / (sqrt(E2) - sqrt(E1))
#
# With m_a(S) = 20.66 / tanh(0.073*S) and stress range
# 0.55-0.90 V/nm:
#   k = ln(0.90/0.55) / (sqrt(0.90) - sqrt(0.55)) = 2.3784 (V/nm)^(-1/2)
#
# Therefore: gamma(S) = 49.14 / tanh(0.073 * S)  [(V/nm)^(-1/2)]
#
# All E-fields in this code are in V/nm (consistent with V_OP/S).
# ==========================================
GAMMA_SQRTE_COEFF = 49.14  # [(V/nm)^(-1/2)] — coefficient for gamma(S)

def calc_gamma_sqrtE(S):
    """
    Spacing-dependent sqrt(E) field acceleration parameter.
    Derived from force-equivalence with the paper's power-law m_a(S).
    
    gamma(S) = 49.14 / tanh(0.073 * S)  [(V/nm)^(-1/2)]
    
    At large S, gamma -> 49.14 (V/nm)^(-1/2).
    At small S, gamma increases (stronger acceleration), mirroring
    the paper's spacing-dependent m_a from Eq. 19.
    
    Note on units: (V/nm)^(-1/2) = (10 MV/cm)^(-1/2)
    In MV/cm units: gamma = 15.54 / tanh(0.073*S) [(MV/cm)^(-1/2)]
    """
    return GAMMA_SQRTE_COEFF / np.tanh(0.073 * S)

# ==========================================
# Dynamic Percolation Model (DPM) Physics
# ==========================================

def calc_beta_DOT(S):
    """Equation (2): Weibull slope of the DOT distribution."""
    return C_sc * (S**P_sc)

def calc_eta_DOT(S):
    """Equation (3): Weibull scale parameter for DOT distribution."""
    beta_DOT = calc_beta_DOT(S)
    return mu_sc * np.exp(-W_mu / beta_DOT)

def calc_Es(S):
    """Equation (17): Characteristic E-field [V/nm] for spacing S [nm]."""
    return 1.0 / (0.85 + 2.91 / S)

def calc_m(S):
    """Equation (18): Spacing-dependent exponent m for DOT-tBD relation."""
    return 0.5659 * (S**(-0.455))

def calc_beta_tBD(S):
    """Weibull shape parameter for time-to-breakdown: beta_tBD = m * beta_DOT."""
    return calc_m(S) * calc_beta_DOT(S)

def calc_eta_tBD(S):
    """
    Weibull scale parameter for time-to-breakdown using the sqrt(E) model.
    
    The acceleration factor replaces the paper's power-law (E/Es)^(-m_a)
    with the sqrt(E) form:  AF = exp(gamma(S) * (sqrt(Es) - sqrt(E)))
    
    Physical basis: Poole-Frenkel conduction in porous low-k dielectrics.
    The PF mechanism gives a current proportional to exp(+gamma*sqrt(E)),
    so the time-to-breakdown scales as exp(-gamma*sqrt(E)), yielding
    the acceleration factor above when referenced to Es.
    
    Combined with the DOT-tBD relation (Eq. 6):
        eta_tBD_supercell = AF * eta_DOT^(1/m)
    
    Then area-scaled from the S^2 supercell to 1 nm^2.
    """
    eta_DOT = calc_eta_DOT(S)
    E_local = V_OP / S              # [V/nm]
    m_val   = calc_m(S)
    Es_val  = calc_Es(S)            # [V/nm]
    gamma   = calc_gamma_sqrtE(S)   # [(V/nm)^(-1/2)]
    
    # sqrt(E) acceleration factor (replaces (E/Es)^(-m_a) from Eq. 8)
    AF_sqrtE = np.exp(gamma * (np.sqrt(Es_val) - np.sqrt(E_local)))
    
    # tBD scale parameter for the S^2 supercell reference area
    eta_tBD_supercell = AF_sqrtE * (eta_DOT ** (1.0 / m_val))
    
    # Area-scale from S^2 down to 1 nm^2
    beta_tBD = calc_beta_tBD(S)
    A_supercell = S**2
    A_unit = 1.0
    eta_tBD_1 = eta_tBD_supercell * (A_supercell / A_unit) ** (1.0 / beta_tBD)
    
    return eta_tBD_1
