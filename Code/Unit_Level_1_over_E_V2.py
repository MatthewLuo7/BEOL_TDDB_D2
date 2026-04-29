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
# 1/E Model Parameters (Spacing-Dependent)
# ==========================================
# G_1E(S) is derived by forcing equivalence between the paper's
# calibrated power-law acceleration factor (Eq. 8 with Eq. 19)
# and the 1/E form:  AF = exp(G * (1/E - 1/Es))
#
# Matching condition at two stress fields E1, E2:
#   m_a(S) * ln(E2/E1) = G(S) * (1/E1 - 1/E2)
#   => G(S) = m_a(S) * ln(E2/E1) / (1/E1 - 1/E2)
#
# Since m_a(S) = 20.66 / tanh(0.073*S)  [Eq. 19, dimensionless],
# and the matching constant k = ln(E2/E1)/(1/E1 - 1/E2) has
# units of E-field, G(S) inherits those units.
#
# Using the paper's stress range of ~5.5-9.0 MV/cm (0.55-0.90 V/nm):
#   k = ln(0.90/0.55) / (1/0.55 - 1/0.90) = 0.6965 V/nm
#
# Therefore: G_1E(S) = 14.39 / tanh(0.073 * S)   [V/nm]
#            G_1E(S) = 143.9 / tanh(0.073 * S)    [MV/cm]
#
# All E-fields in this code are in V/nm (consistent with V_OP/S).
# ==========================================
G_1E_COEFF = 14.39  # [V/nm] — coefficient for spacing-dependent G

def calc_G_1E(S):
    """
    Spacing-dependent 1/E field acceleration constant.
    Derived from force-equivalence with the paper's power-law m_a(S).
    
    G_1E(S) = 14.39 / tanh(0.073 * S)  [V/nm]
    
    At large S, G_1E -> 14.39/1 = 14.39 V/nm (143.9 MV/cm).
    At small S, G_1E increases (stronger acceleration), mirroring
    the paper's finding that m_a increases at smaller spacings.
    """
    return G_1E_COEFF / np.tanh(0.073 * S)

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

def calc_ln_eta_tBD(S):
    """
    Natural log of the Weibull scale parameter for time-to-breakdown
    using the 1/E model.  Returns ln(eta_tBD) to avoid float64 overflow.
    
    IMPORTANT: The 1/E model produces exponents of O(200-800) at
    operating conditions (V_OP/S << Es), which overflow exp() in
    float64 (max ~709).  All arithmetic is therefore done in log-space.
    
    Derivation (all in log-space):
        ln(eta_supercell) = G/E - G/Es + (1/m)*ln(eta_DOT)
        ln(eta_1nm2)      = ln(eta_supercell) + (1/beta_tBD)*ln(S^2)
    """
    eta_DOT = calc_eta_DOT(S)
    E_local = V_OP / S          # [V/nm]
    m_val   = calc_m(S)
    Es_val  = calc_Es(S)        # [V/nm]
    G_val   = calc_G_1E(S)      # [V/nm]
    
    # Log of the 1/E acceleration factor
    ln_AF = G_val * (1.0 / E_local - 1.0 / Es_val)
    
    # Log of the tBD scale parameter for the S^2 supercell
    ln_eta_supercell = ln_AF + (1.0 / m_val) * np.log(eta_DOT)
    
    # Log of area-scaling from S^2 to 1 nm^2
    beta_tBD = calc_beta_tBD(S)
    ln_eta_1 = ln_eta_supercell + (1.0 / beta_tBD) * np.log(S**2)
    
    return ln_eta_1

def calc_eta_tBD(S):
    """
    Weibull scale parameter for time-to-breakdown using the 1/E model.
    
    WARNING: At operating voltages the 1/E acceleration factor is so
    large that eta_tBD overflows float64 for most spacings of interest.
    This function will return np.inf in those cases.  For downstream
    calculations (e.g. failure probability integration), use
    calc_ln_eta_tBD(S) instead and work in log-space.
    
    The acceleration factor replaces the paper's power-law (E/Es)^(-m_a)
    with the 1/E form:  AF = exp(G_1E(S) * (1/E_local - 1/Es))
    
    Combined with the DOT-tBD relation (Eq. 6):
        eta_tBD_supercell = eta_DOT^(1/m) * exp(G * (1/E - 1/Es))
    
    Then area-scaled from the S^2 supercell to 1 nm^2.
    """
    ln_eta = calc_ln_eta_tBD(S)
    
    # Guard against float64 overflow (max representable ~ exp(709))
    if np.isscalar(ln_eta):
        if ln_eta > 709:
            return np.inf
        return np.exp(ln_eta)
    else:
        result = np.full_like(ln_eta, np.inf, dtype=float)
        safe = ln_eta <= 709
        result[safe] = np.exp(ln_eta[safe])
        return result
