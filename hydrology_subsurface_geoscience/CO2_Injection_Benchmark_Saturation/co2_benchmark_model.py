"""Radial CO2 injection benchmark of Koeppel et al. (2019), exact Buckley-Leverett solution.

Model (Koeppel et al., Comput. Geosci. 23:339-354, 2019, Sec. 2-3, Eqs. 11, 13-15, Table 1):
CO2 is injected at the centre of a homogeneous, radially symmetric saline aquifer.
With incompressible phases, no capillary pressure and the analytic pressure of Eq. (13),
the CO2 saturation S(r, t) obeys the scalar conservation law

    phi dS/dt + (Q / r) d f(S; n) / dr = 0,   S(r, 0) = 0,   S(0, t) = 0.8,

with fractional flow f = lam_g / (lam_g + lam_w), lam_g = S**n / mu_g,
lam_w = (1 - S)**n / mu_w (Eqs. 14-15) and Q = Q0 (1 + theta1) (Eq. 13).
This module evaluates the exact entropy solution (method of characteristics with the
Welge/Oleinik construction): with xi = phi r**2 / (2 c_Q Q t), S is 0.8 for
xi <= f'(0.8), solves f'(S) = xi on the rarefaction, and drops to 0 at the shock
xi_s = f(S_s) / S_s, where f'(S_s) S_s = f(S_s).  The front radius is
r_f = sqrt(2 c_Q Q t xi_s / phi).

The benchmark's code (Zenodo 10.5281/zenodo.933827) is a central-upwind finite-volume
solver; it was not run here.  Two things not stated in the paper are fixed by fitting
this exact solution to the authors' Monte Carlo reference curves (MCrefs_MeanStd.csv)
and to Fig. 1 of the paper (see DATA.md): the affine map from the benchmark's coded
porosity samples to porosity (A_PHI, B_PHI) and a flux factor C_Q multiplying Q.

numpy and scipy only.
"""
import numpy as np
from scipy.optimize import brentq

# Table 1 of the paper
MU_G = 3.950e-5          # CO2 viscosity, Pa s
MU_W = 2.535e-4          # brine viscosity, Pa s
K_A = 2e-14              # absolute permeability, m^2
LAMBDA_MEAN = 1.0e4      # mean mobility, 1/(Pa s)
P_MAX = 320e5            # injection pressure, Pa
P_MIN = 300e5            # pressure at r_max, Pa
R_MAX = 500.0            # domain radius, m
T_END = 100 * 86400.0    # simulation time, s (100 days)
S_LEFT = 0.8             # saturation on the left boundary
PHI_NOMINAL = 0.15       # nominal porosity
# Q0 = Cp qCO2 = (pmax - pmin) K_A lambda / ln(r_max)  (Eq. 10 and the Cp definition)
Q0 = (P_MAX - P_MIN) * K_A * LAMBDA_MEAN / np.log(R_MAX)   # 6.436e-5 m^2/s

# Fitted to the benchmark's Monte Carlo reference (DATA.md, "Calibration")
C_Q = 0.64299            # flux factor multiplying Q (paper: 1; fitted, see DATA.md)
A_PHI = 0.09151          # porosity = A_PHI + B_PHI * po (po: coded sample in the archive)
B_PHI = 0.19647

NAMES = ("Q", "n", "phi")
UNITS = ("m^2/s", "1", "1")
DESCRIPTIONS = ("injection rate Q = Cp qCO2 (1 + theta1) (Eq. 13; Fig. 1 axis 'injection rate')",
                "relative permeability degree theta2 (Eqs. 14-15)",
                "reservoir porosity theta3")
RADII = (50.0, 75.0, 100.0, 125.0, 150.0)


def coded_to_physical(ir, kd, po):
    """Map the archive's coded samples (InputParameters.csv: ir, kd, po) to (Q, n, phi)."""
    ir, kd, po = (np.asarray(v, float) for v in (ir, kd, po))
    return Q0 * (1.0 + ir), 2.0 + 2.0 * kd, A_PHI + B_PHI * po


def frac_flow(S, n):
    a = (S ** n) / MU_G
    b = ((1.0 - S) ** n) / MU_W
    return a / (a + b)


def dfrac_flow(S, n):
    a = (S ** n) / MU_G
    b = ((1.0 - S) ** n) / MU_W
    da = n * S ** (n - 1.0) / MU_G
    db = -n * (1.0 - S) ** (n - 1.0) / MU_W
    return (da * b - a * db) / (a + b) ** 2


def shock(n):
    """Shock saturation S_s and shock speed xi_s = f(S_s)/S_s for exponent n."""
    g = lambda S: dfrac_flow(S, n) * S - frac_flow(S, n)
    if g(S_LEFT) >= 0.0:          # no rarefaction: a single shock from S_LEFT
        return S_LEFT, frac_flow(S_LEFT, n) / S_LEFT
    Ss = brentq(g, 1e-6, S_LEFT, xtol=1e-15, rtol=1e-15, maxiter=500)
    return Ss, frac_flow(Ss, n) / Ss


def _check(Q, n, phi):
    if not (np.isfinite(Q) and Q > 0):
        raise ValueError(f"injection rate Q must be positive, got {Q}")
    if not (np.isfinite(n) and n > 1):
        raise ValueError(f"relative permeability degree n must exceed 1, got {n}")
    if not (np.isfinite(phi) and 0 < phi <= 1):
        raise ValueError(f"porosity must be in (0, 1], got {phi}")


def saturation(r, Q, n, phi, t=T_END, c_q=None):
    """CO2 saturation at radii r (m) at time t (s) for one input (Q, n, phi).

    c_q: flux factor; default C_Q."""
    _check(Q, n, phi)
    c_q = C_Q if c_q is None else c_q
    r = np.atleast_1d(np.asarray(r, float))
    xi = phi * r ** 2 / (2.0 * c_q * Q * t)
    Ss, xs = shock(n)
    vL = dfrac_flow(S_LEFT, n)
    S = np.zeros_like(xi)
    S[xi <= vL] = S_LEFT
    mid = (xi > vL) & (xi < xs)
    if np.any(mid):
        lo = np.full(mid.sum(), Ss); hi = np.full(mid.sum(), S_LEFT); v = xi[mid]
        for _ in range(80):        # f' decreases on [S_s, 0.8]: bisection
            c = 0.5 * (lo + hi)
            big = dfrac_flow(c, n) > v
            lo = np.where(big, c, lo); hi = np.where(big, hi, c)
        S[mid] = 0.5 * (lo + hi)
    return S


def front_radius(Q, n, phi, t=T_END, c_q=None):
    """Radius (m) of the CO2 displacement front (the shock) at time t; c_q default C_Q."""
    _check(Q, n, phi)
    c_q = C_Q if c_q is None else c_q
    return np.sqrt(2.0 * c_q * Q * t * shock(n)[1] / phi)


def grad_front_radius(Q, n, phi, t=T_END, h=1e-4):
    """Gradient of front_radius with respect to (Q, n, phi).

    Q and phi: exact (r_f ~ sqrt(Q/phi)).  n: d xi_s/dn by a central difference
    of the shock speed with step h (the shock speed is smooth in n)."""
    rf = front_radius(Q, n, phi, t)
    xs = shock(n)[1]
    dxs = (shock(n + h)[1] - shock(n - h)[1]) / (2.0 * h)
    return np.array([0.5 * rf / Q, 0.5 * rf * dxs / xs, -0.5 * rf / phi])


def saturation_field(r, Q, n, phi, t=T_END):
    """Saturations at radii r (shape (R,)) for N inputs (arrays of shape (N,)); returns (N, R).

    The same exact solution as `saturation`, vectorized over the samples."""
    Q, n, phi = (np.atleast_1d(np.asarray(v, float)) for v in (Q, n, phi))
    for q, k, p in zip(Q, n, phi):
        _check(q, k, p)
    r = np.atleast_1d(np.asarray(r, float))
    xi = phi[:, None] * r[None, :] ** 2 / (2.0 * C_Q * Q[:, None] * t)
    sh = np.array([shock(k) for k in n])
    Ss, xs = sh[:, :1], sh[:, 1:]
    nn = n[:, None]
    vL = dfrac_flow(S_LEFT, nn)
    lo = np.broadcast_to(Ss, xi.shape).copy(); hi = np.full(xi.shape, S_LEFT)
    for _ in range(60):
        c = 0.5 * (lo + hi)
        big = dfrac_flow(c, nn) > xi
        lo = np.where(big, c, lo); hi = np.where(big, hi, c)
    S = 0.5 * (lo + hi)
    S = np.where(xi <= vL, S_LEFT, S)
    return np.where(xi >= xs, 0.0, S)


def qoi(Q, n, phi, radii=RADII, t=T_END):
    """Saturations at the radii and the front radius."""
    return np.concatenate([saturation(radii, Q, n, phi, t), [front_radius(Q, n, phi, t)]])
