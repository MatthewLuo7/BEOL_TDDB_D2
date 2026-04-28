# ==========================================
# Layout & Operating Parameters
# ==========================================
V_OP = 0.7  # V, operating voltage

# ==========================================
# Reliability Targets
# ==========================================

T_TARGET = 10* 365.25 * 24 * 3600  # 10-year lifetime in seconds

F_TARGET = 100/1000000  # 100 ppm allowed failure rate

# ==========================================
# Spacing Parameters
# ==========================================

S_MAX = 12  # Maximum spacing to consider for predictions (in nm) (beyond this, full reliability is predicted)

S_MIN= 2  # Minimum spacing to consider for predictions (in nm) (below this, considered a yield issue)

