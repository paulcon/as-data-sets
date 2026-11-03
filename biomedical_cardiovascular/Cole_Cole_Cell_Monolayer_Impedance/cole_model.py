"""Cole-Cole impedance of a cell monolayer (Zhuang et al., Molecules 27:5861, 2022, Eq. 1).

The cell-monolayer term of the paper's Eq. (1),

    Z(omega) = R_inf + (R_0 - R_inf) / (1 + (i omega tau)^alpha),

with omega the angular frequency in rad/s (see DATA.md for why the paper's frequency axis
is read as omega). The electrode-polarization term of Eq. (1) is left out: the paper's
uncertainty and sensitivity study (its Table 1, Figs. 1-3) is on the four cell-monolayer
parameters only, and it gives no values for the electrode term.

Inputs are rows x = (R_inf, R_0, alpha, tau) in physical units (ohm, ohm, 1, s).
`impedance` returns Z, `impedance_and_grad` returns Z and the exact gradient
dZ/dx (complex; its real and imaginary parts are the gradients of Re Z and Im Z).
(i omega tau)^alpha is evaluated in polar form, (omega tau)^alpha exp(i alpha pi/2),
which is the principal branch for omega, tau > 0.

Only numpy is imported.
"""
import numpy as np

NAMES = ('R_inf', 'R_0', 'alpha', 'tau')
UNITS = ('ohm', 'ohm', None, 's')
DESCRIPTIONS = ('resistance at infinite frequency', 'resistance at low frequency',
                'dispersion factor (CPE exponent)', 'characteristic time constant')

#: Table 1 of the paper (p. 4): Gaussian means and standard deviations ("Sigma").
MEAN = np.array([200.0, 1000.0, 0.7, 5e-6])
SD = np.array([60.0, 300.0, 0.21, 1.5e-6])

#: Lognormal with the same means and standard deviations (the notebook's distribution):
#: ln x_j ~ N(LOG_MU_j, LOG_SIGMA_j^2), LOG_SIGMA_j^2 = ln(1 + (SD_j/MEAN_j)^2).
LOG_SIGMA = np.sqrt(np.log1p((SD / MEAN)**2))
LOG_MU = np.log(MEAN) - LOG_SIGMA**2 / 2


def _check(X, omega):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f'X must have 4 columns (R_inf, R_0, alpha, tau); got shape {X.shape}')
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(omega))):
        raise ValueError('inputs and omega must be finite')
    if np.any(omega <= 0):
        raise ValueError('omega must be positive')
    if np.any(X[:, 3] <= 0):
        raise ValueError('tau must be positive (the CPE (i omega tau)^alpha needs tau > 0)')
    if np.any(X[:, 2] <= 0) or np.any(X[:, 2] >= 2):
        raise ValueError('alpha must lie in (0, 2) for the formula to be defined '
                         '(the physical range is 0 < alpha < 1)')
    return X, omega


def is_physical(X):
    """Rows inside the physical domain: R_inf > 0, R_0 > R_inf, 0 < alpha < 1, tau > 0."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return ((X[:, 0] > 0) & (X[:, 1] > X[:, 0]) & (X[:, 2] > 0) & (X[:, 2] < 1)
            & (X[:, 3] > 0))


def impedance(X, omega):
    """Complex impedance Z, shape (M, K), for M input rows and K angular frequencies (rad/s)."""
    X, omega = _check(X, omega)
    Ri, R0, a, t = (X[:, j][:, None] for j in range(4))
    s = np.exp(a * (np.log(omega[None, :] * t) + 0.5j * np.pi))   # (i omega tau)^alpha
    return Ri + (R0 - Ri) / (1.0 + s)


def impedance_and_grad(X, omega):
    """Z, shape (M, K), and dZ/dx, shape (M, K, 4), complex, w.r.t. the physical inputs."""
    X, omega = _check(X, omega)
    Ri, R0, a, t = (X[:, j][:, None] for j in range(4))
    L = np.log(omega[None, :] * t) + 0.5j * np.pi       # log(i omega tau), principal branch
    s = np.exp(a * L)                                    # (i omega tau)^alpha
    D = 1.0 + s
    dR = R0 - Ri
    Z = Ri + dR / D
    q = -dR * s / D**2                                   # dZ/ds * s
    G = np.stack([1.0 - 1.0 / D,                         # dZ/dR_inf
                  np.broadcast_to(1.0 / D, Z.shape),     # dZ/dR_0
                  q * L,                                 # dZ/dalpha
                  q * a / t], axis=-1)                   # dZ/dtau
    return Z, G
