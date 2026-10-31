"""Local shear buckling load of a hat-stiffened panel (Ko and Jackson 1994, formula as given by OpenTURNS).

The model is the closed-form critical shear load of the flat face-sheet strip between
two stiffener feet of a hat-stiffened panel, with a correction for the shear flow
carried by the hat. The source is W. L. Ko and R. H. Jackson, "Shear Buckling Analysis
of a Hat-Stiffened Panel", NASA TM-4644, 1994 (NTRS 19950011075). The formula and the
input distributions below were taken from the OpenTURNS "stiffened panel" use case
(documentation and `python/src/usecases/stiffened_panel.py`, tag v1.25.1, identical to
master on 2026-10-31), for the case in which the face sheet and the hat have the same
thickness t and the same material (E, nu). No OpenTURNS code (LGPL) was copied; this
module is written from the equations.

THICKNESS. OpenTURNS codes t ~ U(8.02e-5, 8.181e-5) m (0.0032 in), which gives a load
about 800 times below the memorandum's 900 lb/in finite-element local load; this appears
to be a factor-10 slip. The primary inputs here follow the memorandum's panel:
t ~ U(8.02e-4, 8.181e-4) m (about 0.032 in, a titanium sheet gauge), for which the
formula gives 1107 lb/in at the midpoint, "slightly higher" than 900 lb/in as the
memorandum says of its closed-form prediction. The OpenTURNS-as-coded thickness is
available as `variant='openturns'` (see `bounds`).

    (N_xy)_cr = k_xy (pi^2 / b0^2) E t^3 / (12 (1 - nu^2)) * Q,     k_xy = 5.35 + 4 (b0/a)^2,

    Q   = 1 + (1/4) [2 p (h - 2 h0) - h_c (f1 - f2)] / (h0 l),
    A   = l t,     Abar = A + t (p + (f1 - f2)/2),
    h0  = [A (h_c + 2 t) + t^2 (f1 - f2)] / (2 Abar),      h = h_c + t.

The QoI (N_xy)_cr is a shear force per unit length (shear flow), in N/m.

Inputs (column order of `NAMES`): E is a normal law N(110e9, 55e9) (mean, standard
deviation, Pa) truncated to [99e9, 121e9]; the other nine are uniform. f1 and f2 are
dependent through a normal (Gaussian) copula with Spearman rank correlation
SPEARMAN_F1F2 = -0.8, i.e. correlation parameter R_COPULA = 2 sin(pi (-0.8) / 6) = -0.8135,
as the OpenTURNS code does (it passes -0.8 through
NormalCopula.GetCorrelationFromSpearmanCorrelation); all other pairs are independent.

`bounds(variant)` gives the supports ('tm', the default, or 'openturns'), and the
distribution functions below take the same `variant` keyword.
`sample` draws the inputs by the inverse CDF (Rosenblatt order: f1 before f2).
`buckling_load` evaluates the QoI (real or complex arrays, so a complex step works),
`grad_buckling_load` its exact gradient with respect to the physical inputs.
`to_normal` maps x to z_j = Phi^{-1}(F_j(x_j)), which is N(0, R) with R the copula's
correlation matrix (`copula_correlation`), and `dx_dz` is dx_j/dz_j.
`mean_and_cov` gives the exact mean and covariance of the physical inputs.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy import special

#: Input names, in column order (OpenTURNS: E, nu, h_c, ell, f_1, f_2, t, a, b_0, p).
NAMES = ('E', 'nu', 'h_c', 'ell', 'f_1', 'f_2', 't', 'a', 'b_0', 'p')

#: Units of the inputs.
UNITS = ('Pa', '-', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm')

#: Descriptions (OpenTURNS use-case table).
DESCRIPTIONS = ('Young modulus', 'Poisson coefficient',
                'distance between the mean surfaces of the hat and the foot of the stiffener',
                'length of the stiffener side (flank)', 'width of the stiffener foot',
                'width of the stiffener hat', 'thickness of the panel and the stiffener',
                'width (long side) of the panel', 'distance between two stiffener feet',
                'half-width of the stiffener')

#: Truncated normal law of E: mean and standard deviation of the parent normal, bounds (Pa).
E_MU, E_SIGMA = 110.0e9, 55.0e9
E_LO, E_HI = 99.0e9, 121.0e9

#: Lower and upper bounds of every input (the truncation interval for E). The thickness t is
#: the memorandum's panel, ten times the OpenTURNS range (see the module docstring).
LOWER = np.array([E_LO, 0.3675, 0.0285, 0.04655, 0.0266, 0.00627, 8.02e-4, 0.6039, 0.04455, 0.03762])
UPPER = np.array([E_HI, 0.3825, 0.0315, 0.05145, 0.0294, 0.00693, 8.181e-4, 0.6161, 0.04545, 0.03838])

#: The thickness range as coded in OpenTURNS (variant 'openturns'), m.
T_OPENTURNS = (8.02e-5, 8.181e-5)
I_T = 6
LOWER_OPENTURNS = LOWER.copy()
UPPER_OPENTURNS = UPPER.copy()
LOWER_OPENTURNS[I_T], UPPER_OPENTURNS[I_T] = T_OPENTURNS

#: Distributions (OpenTURNS's, except t, which is ten times the OpenTURNS range).
DISTRIBUTIONS = ('TruncatedNormal(110e9, 55e9, [99e9, 121e9])',) + tuple(
    f'Uniform({lo:g}, {hi:g})' for lo, hi in zip(LOWER[1:], UPPER[1:]))

#: Spearman rank correlation of f_1 and f_2 (OpenTURNS: correlation_matrix[4, 5] = -0.8, converted).
SPEARMAN_F1F2 = -0.8
#: Correlation parameter of the normal copula between f_1 and f_2 (column indices 4, 5): 2 sin(pi rho_S / 6).
R_COPULA = 2.0 * np.sin(np.pi * SPEARMAN_F1F2 / 6.0)
I_F1, I_F2 = 4, 5

#: Reference point: the midpoint of every interval (also the mean of E, whose law is symmetric).
NOMINAL = 0.5 * (LOWER + UPPER)
NOMINAL_OPENTURNS = 0.5 * (LOWER_OPENTURNS + UPPER_OPENTURNS)

M = len(NAMES)


def bounds(variant='tm'):
    """(lower, upper) supports: 'tm' (default, the memorandum's thickness) or 'openturns' (t as coded there)."""
    if variant == 'tm':
        return LOWER, UPPER
    if variant == 'openturns':
        return LOWER_OPENTURNS, UPPER_OPENTURNS
    raise ValueError(f"variant must be 'tm' or 'openturns', got {variant!r}")


def copula_correlation(r=R_COPULA):
    """Correlation matrix R of the normal copula (identity except R[f1, f2] = r)."""
    R = np.eye(M)
    R[I_F1, I_F2] = R[I_F2, I_F1] = r
    return R


def spearman_of_copula(r=R_COPULA):
    """Spearman rank correlation of a normal copula with correlation parameter r: (6/pi) asin(r/2).

    For uniform marginals this is also the Pearson correlation of f1 and f2. With the default
    r = R_COPULA it returns SPEARMAN_F1F2 = -0.8 (to rounding).
    """
    return 6.0 / np.pi * np.arcsin(r / 2.0)


def _split(X):
    X = np.asarray(X)
    return [X[..., j] for j in range(M)]


def intermediates(X):
    """Dictionary of k_xy, the plate factor, h, h0, the hat correction Q and the strip load (no Q)."""
    E, nu, hc, ell, f1, f2, t, a, b0, p = _split(X)
    d = f1 - f2
    A = ell * t
    Abar = A + t * (p + d / 2)
    h0 = (A * (hc + 2 * t) + t**2 * d) / (2 * Abar)
    h = hc + t
    Q = 1 + 0.25 * (2 * p * (h - 2 * h0) - hc * d) / (h0 * ell)
    kxy = 5.35 + 4 * (b0 / a)**2
    D = E * t**3 / (12 * (1 - nu**2))
    strip = kxy * np.pi**2 / b0**2 * D
    return dict(kxy=kxy, D=D, h=h, h0=h0, Abar=Abar, Q=Q, strip=strip)


def buckling_load(X):
    """Critical shear buckling load (N_xy)_cr in N/m; X of shape (..., 10) (real or complex)."""
    q = intermediates(X)
    return q['strip'] * q['Q']


def grad_buckling_load(X):
    """Exact gradient of (N_xy)_cr with respect to the physical inputs, shape (..., 10), N/m per unit."""
    X = np.asarray(X, dtype=float)
    E, nu, hc, ell, f1, f2, t, a, b0, p = _split(X)
    d = f1 - f2
    # h0 = Nn / Dd, with Nn = l (hc + 2t) + t d and Dd = 2l + 2p + d (Abar and A share the factor t)
    Dd = 2 * ell + 2 * p + d
    h0 = (ell * (hc + 2 * t) + t * d) / Dd
    dh0 = dict(hc=ell / Dd, ell=((hc + 2 * t) - 2 * h0) / Dd, t=(2 * ell + d) / Dd,
               p=-2 * h0 / Dd, d=(t - h0) / Dd)
    num = 2 * p * (hc + t - 2 * h0) - hc * d
    dnum = dict(hc=2 * p * (1 - 2 * dh0['hc']) - d, ell=-4 * p * dh0['ell'], t=2 * p * (1 - 2 * dh0['t']),
                p=2 * (hc + t - 2 * h0) - 4 * p * dh0['p'], d=-4 * p * dh0['d'] - hc)
    den = h0 * ell
    dden = {k: ell * v for k, v in dh0.items()}
    dden['ell'] = dden['ell'] + h0
    Q = 1 + 0.25 * num / den
    dQ = {k: 0.25 * (dnum[k] * den - num * dden[k]) / den**2 for k in dh0}
    K = np.pi**2 * (5.35 / b0**2 + 4 / a**2)          # k_xy pi^2 / b0^2
    P = E * t**3 / (12 * (1 - nu**2))
    N = K * P * Q
    G = np.empty(X.shape)
    G[..., 0] = N / E
    G[..., 1] = N * 2 * nu / (1 - nu**2)
    G[..., 2] = K * P * dQ['hc']
    G[..., 3] = K * P * dQ['ell']
    G[..., 4] = K * P * dQ['d']
    G[..., 5] = -K * P * dQ['d']
    G[..., 6] = 3 * N / t + K * P * dQ['t']
    G[..., 7] = -8 * np.pi**2 / a**3 * P * Q
    G[..., 8] = -2 * 5.35 * np.pi**2 / b0**3 * P * Q
    G[..., 9] = K * P * dQ['p']
    return G


# ---------------------------------------------------------------- marginals and copula

_E_A, _E_B = (E_LO - E_MU) / E_SIGMA, (E_HI - E_MU) / E_SIGMA
_E_Z = special.ndtr(_E_B) - special.ndtr(_E_A)


def cdf(X, variant='tm'):
    """Marginal CDFs F_j(x_j), shape like X."""
    lo, hi = bounds(variant)
    X = np.asarray(X, dtype=float)
    U = (X - lo) / (hi - lo)
    U[..., 0] = (special.ndtr((X[..., 0] - E_MU) / E_SIGMA) - special.ndtr(_E_A)) / _E_Z
    return U


def ppf(U, variant='tm'):
    """Marginal inverse CDFs F_j^{-1}(u_j), shape like U."""
    lo, hi = bounds(variant)
    U = np.asarray(U, dtype=float)
    X = lo + (hi - lo) * U
    X[..., 0] = E_MU + E_SIGMA * special.ndtri(special.ndtr(_E_A) + U[..., 0] * _E_Z)
    return X


def pdf(X, variant='tm'):
    """Marginal densities rho_j(x_j), shape like X."""
    lo, hi = bounds(variant)
    X = np.asarray(X, dtype=float)
    R = np.broadcast_to(1.0 / (hi - lo), X.shape).copy()
    zE = (X[..., 0] - E_MU) / E_SIGMA
    R[..., 0] = np.exp(-0.5 * zE**2) / np.sqrt(2 * np.pi) / (E_SIGMA * _E_Z)
    return R


def sample(n, rng, r=R_COPULA, variant='tm'):
    """n samples of the physical inputs, shape (n, 10), by the inverse CDF.

    Draws V = rng.random((n, 10)); every column except f2 is F_j^{-1}(V_j). For the normal
    copula (Rosenblatt order f1, f2): z1 = Phi^{-1}(V_f1), z2 = r z1 + sqrt(1 - r^2) Phi^{-1}(V_f2),
    and f2 = F^{-1}(Phi(z2)). The two variants use the same draws V, so with the same rng state
    they differ only in the thickness column.
    """
    V = rng.random((n, M))
    U = V.copy()
    z1 = special.ndtri(V[:, I_F1])
    z2 = r * z1 + np.sqrt(1 - r**2) * special.ndtri(V[:, I_F2])
    U[:, I_F2] = special.ndtr(z2)
    return ppf(U, variant)


def to_normal(X, variant='tm'):
    """z_j = Phi^{-1}(F_j(x_j)); z is N(0, R) with R = copula_correlation()."""
    return special.ndtri(cdf(X, variant))


def dx_dz(X, variant='tm'):
    """dx_j/dz_j = phi(z_j) / rho_j(x_j) for the map of `to_normal`, shape like X."""
    Z = to_normal(X, variant)
    return np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi) / pdf(X, variant)


def mean_and_cov(r=R_COPULA, variant='tm'):
    """Exact mean and covariance of the physical inputs (truncated-normal moments in closed form)."""
    lo, hi = bounds(variant)
    mean = 0.5 * (lo + hi)
    var = (hi - lo)**2 / 12.0
    phi = lambda z: np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    a, b, Z = _E_A, _E_B, _E_Z
    mean[0] = E_MU + E_SIGMA * (phi(a) - phi(b)) / Z
    var[0] = E_SIGMA**2 * (1 + (a * phi(a) - b * phi(b)) / Z - ((phi(a) - phi(b)) / Z)**2)
    cov = np.diag(var)
    cov[I_F1, I_F2] = cov[I_F2, I_F1] = spearman_of_copula(r) * np.sqrt(var[I_F1] * var[I_F2])
    return mean, cov
