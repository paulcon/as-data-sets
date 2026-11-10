"""Speed-reducer shaft limit state (Du and Sudjianto 2004), closed form with exact gradients.

    g = S * 1e6 - 32 / (pi * (D/1e3)**3) * sqrt(F**2 * (L/1e3)**2 / 16 + T**2)      [Pa]

Strength minus the combined bending-torsion stress of a shaft; g < 0 is failure. The formula is
Eq. (34), p. 1205, and the input distributions are Table 8, p. 1205, of X. Du and A. Sudjianto,
AIAA J. 42(6):1199-1207, 2004, doi:10.2514/1.3877, with D, L converted from mm to m and S from MPa
to Pa (the article's Tables 10-11 thresholds are in Pa). The uqtestfuns 0.7.0 function
`RSSpeedReducerShaft` (MIT, D. Wicaksono) implements the same problem and serves as a cross-check
(check_accuracy.py). The code here is written from the formula; nothing is copied.

Inputs, in the order of the arrays (Table 8 gives marginals only; independence is this data set's reading):
    D  shaft diameter   mm    Normal(39, 0.1)
    L  shaft span       mm    Normal(400, 0.1)
    F  external force   N     Gumbel (max), mean 1500, sd 350 (location MU_F, scale BETA_F)
    T  torque           N m   Normal(250, 35)
    S  strength         MPa   Uniform[70, 80]

Besides g and its gradient, the module maps the inputs to and from independent standard normal
coordinates xi_i = Phi^{-1}(F_i(x_i)) (the "CDF" coordinates) and gives dx_i/dxi_i, and it has a
FORM design-point search in those coordinates. Only numpy and scipy are imported.
"""
import numpy as np
from scipy import optimize, special

NAMES = ('D', 'L', 'F', 'T', 'S')
UNITS = ('mm', 'mm', 'N', 'N m', 'MPa')
DESCRIPTIONS = ('shaft diameter', 'shaft span', 'external force', 'torque', 'strength')
QOI_NAME = 'g'
QOI_UNITS = 'Pa'

# Gumbel (max) parameters from mean and sd, as uqtestfuns' gumbel_max_beta / gumbel_max_mu
BETA_F = 350.0 * np.sqrt(6.0) / np.pi
MU_F = 1500.0 - BETA_F * np.euler_gamma
S_LO, S_HI = 70.0, 80.0

MEAN = np.array([39.0, 400.0, 1500.0, 250.0, 75.0])
SD = np.array([0.1, 0.1, 350.0, 35.0, (S_HI - S_LO) / np.sqrt(12.0)])
NORMAL_COLS = (0, 1, 3)
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def limit_state(X, grad=False):
    """g (Pa) at the rows of X (M, 5) in physical units; with grad=True also dg/dx (M, 5) in
    Pa per physical unit (Pa/mm, Pa/mm, Pa/N, Pa/(N m), Pa/MPa)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] != 5:
        raise ValueError('X must have 5 columns (D, L, F, T, S)')
    D, L, F, T, S = X.T
    d, l = D * 1e-3, L * 1e-3
    k = 32.0 / (np.pi * d**3)
    R = np.sqrt(F**2 * l**2 / 16.0 + T**2)
    g = S * 1e6 - k * R
    if not grad:
        return g
    G = np.empty_like(X)
    G[:, 0] = 3.0 * k * R / d * 1e-3            # dg/dD, per mm
    G[:, 1] = -k * F**2 * l / (16.0 * R) * 1e-3  # dg/dL, per mm
    G[:, 2] = -k * F * l**2 / (16.0 * R)          # dg/dF
    G[:, 3] = -k * T / R                          # dg/dT
    G[:, 4] = 1e6                                 # dg/dS, per MPa
    return g, G


def from_normal(Z):
    """Physical inputs from standard normal coordinates xi (M, 5): x_i = F_i^{-1}(Phi(xi_i))."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    X = np.empty_like(Z)
    for j in NORMAL_COLS:
        X[:, j] = MEAN[j] + SD[j] * Z[:, j]
    X[:, 2] = MU_F - BETA_F * np.log(-special.log_ndtr(Z[:, 2]))
    X[:, 4] = S_LO + (S_HI - S_LO) * special.ndtr(Z[:, 4])
    return X


def to_normal(X):
    """Standard normal coordinates xi_i = Phi^{-1}(F_i(x_i)) of physical inputs X (M, 5)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Z = np.empty_like(X)
    for j in NORMAL_COLS:
        Z[:, j] = (X[:, j] - MEAN[j]) / SD[j]
    t = np.exp(-(X[:, 2] - MU_F) / BETA_F)
    cdf = np.exp(-t)
    sf = -np.expm1(-t)
    Z[:, 2] = np.where(cdf < 0.5, special.ndtri(cdf), -special.ndtri(sf))
    Z[:, 4] = special.ndtri((X[:, 4] - S_LO) / (S_HI - S_LO))
    return Z


def dx_dxi(Z):
    """dx_i/dxi_i at standard normal coordinates Z (M, 5); multiply dg/dx by it for dg/dxi."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    J = np.empty_like(Z)
    for j in NORMAL_COLS:
        J[:, j] = SD[j]
    logphi = -0.5 * Z**2 - _LOG_SQRT_2PI
    # Gumbel: dx/dxi = phi(xi) / p(x), with log p = -log(beta) - t - exp(-t), t = (x - mu)/beta
    xF = MU_F - BETA_F * np.log(-special.log_ndtr(Z[:, 2]))
    t = (xF - MU_F) / BETA_F
    J[:, 2] = np.exp(logphi[:, 2] + np.log(BETA_F) + t + np.exp(-t))
    J[:, 4] = (S_HI - S_LO) * np.exp(logphi[:, 4])
    return J


def sample(M, rng):
    """M Monte Carlo samples: xi ~ N(0, I_5) from `rng` (a numpy Generator), x = from_normal(xi).
    Returns (X, xi)."""
    Z = rng.standard_normal((M, 5))
    return from_normal(Z), Z


def form(start=None, threshold=0.0):
    """FORM design point: min |xi| subject to g(x(xi)) = threshold (Pa; SLSQP with exact
    gradients), from `start` (default (0, 0, 1, 1, -1), toward larger load and lower strength).
    Returns (xi*, beta, alpha) with beta = |xi*| and alpha = xi*/beta, the unit vector from the
    origin to the design point. For threshold 0 the FORM estimate of P[g < 0] is Phi(-beta); for
    another threshold c it is Phi(-beta) if g > c at xi = 0 and Phi(beta) otherwise."""
    z0 = np.array([0.0, 0.0, 1.0, 1.0, -1.0]) if start is None else np.asarray(start, float)

    def con(z):
        return (limit_state(from_normal(z))[0] - threshold) * 1e-6

    def con_jac(z):
        _, G = limit_state(from_normal(z), grad=True)
        return (G * dx_dxi(z))[0] * 1e-6

    res = optimize.minimize(lambda z: 0.5 * z @ z, z0, jac=lambda z: z, method='SLSQP',
                            constraints=[{'type': 'eq', 'fun': con, 'jac': con_jac}],
                            options={'ftol': 1e-14, 'maxiter': 500})
    if not res.success:
        raise RuntimeError(f'FORM search did not converge: {res.message}')
    z = res.x
    beta = float(np.linalg.norm(z))
    return z, beta, z / beta
