"""Thermal consolidation around a point heat source (Booker & Savvidou 1985), with exact gradients.

A constant point heat source of power Q is buried deep in an infinite, water-saturated,
poro-thermo-elastic rock. The closed-form solution (Booker & Savvidou 1985, Eq. 23, in the
corrected form of Chaudhry et al. 2019, Eqs. 18-25) gives three fields of radius r and time t,
all relative to the initial state:

    dT(r, t)  = Q / (4 pi K r) * f(kappa)                         temperature change   [K]
    p(r, t)   = X Q / (4 pi K r) * [f(kappa) - f(c)] / (1 - c/kappa)  excess pore pressure [Pa]
    u_r(r, t) = Q / (4 pi K M) * { b' g(kappa) + X [g(kappa) - g(c)] / (1 - c/kappa) }
                                                                  radial displacement  [m]

with the front functions, for a diffusivity A (thermal kappa or hydraulic c),

    f(A) = erfc(z),  g(A) = tau + (1/2 - tau) erfc(z) - sqrt(tau/pi) exp(-z^2),
    tau = A t / r^2,  z = r / (2 sqrt(A t)) = 1 / (2 sqrt(tau)),

and the derived material constants

    kappa = K / (C rho)                    thermal diffusivity
    lam = E nu / ((1 + nu)(1 - 2 nu)), G = E / (2 (1 + nu)), M = lam + 2 G   (constrained modulus)
    c = k_s M / mu_w                       coefficient of consolidation (hydraulic diffusivity)
    a_u = phi a_w + (1 - phi) a_s          volumetric expansion of the mixture
    b' = (lam + 2 G / 3) a_s               thermal stress coefficient
    X = a_u M - b'                         thermal pressurisation coefficient.

The quotients [f(kappa) - f(c)] / (1 - c/kappa) and [g(kappa) - g(c)] / (1 - c/kappa) have a
removable singularity at c = kappa. They are written here as kappa * (h(kappa) - h(c)) /
(kappa - c), and near c = kappa the divided difference is evaluated as the integral of h'(A)
over the segment between c and kappa (24-point Gauss-Legendre rule); the exact gradients use
the same construction with h''(A). Elsewhere the direct quotient is used.

Inputs (8, independent), their distributions and the r-t grid are those of the UQTestFuns
test function ``SaturatedPointHeatSource`` (input set and parameter set "Chaudhry2021", from
Chaudhry, Buchwald & Nagel 2021, Table 1). That test function is not in the UQTestFuns 0.7.0
release on PyPI; it is in the unreleased GitHub source, https://github.com/damar-wicaksono/uqtestfuns,
branch dev, commit 82d5c8b1e2cc9e746ba7f22a53e3c7f1ad12d42f (2026-10-22). This file is an
independent numpy/scipy implementation written from the equations above (UQTestFuns is
MIT-licensed; its notice is reproduced at the end of this file because the input model and grid
are taken from it).

Imports numpy and scipy only.
"""

import numpy as np
from scipy import stats
from scipy.special import erf, erfc, erfcx

# ------------------------------------------------------------------ inputs
NAMES = ('K', 'C', 'rho', 'E', 'a_s', 'k_s', 'nu', 'phi')
UNITS = ('W/(m K)', 'J/(kg K)', 'kg/m^3', 'Pa', '1/K', 'm^2', '-', '-')
DESCRIPTIONS = ('total thermal conductivity', 'total specific heat capacity', 'total density',
                "Young's modulus (drained skeleton)", 'volumetric thermal expansion coefficient of the solid',
                'intrinsic permeability', "Poisson's ratio", 'porosity')
# (kind, parameters): truncnorm (mu, sigma, lower, upper) of the parent normal; normal (mu, sigma);
# uniform (lower, upper); triangular (lower, upper, mode).  UQTestFuns input set "Chaudhry2021".
MARGINALS = (('truncnorm', (1.79, 0.34, 1.29, 2.45)),
             ('normal', (978.0, 68.0)),
             ('truncnorm', (2480.0, 30.0, 2420.0, 2540.0)),
             ('truncnorm', (12.8e9, 3.7e9, 5.5e9, 20.1e9)),
             ('uniform', (3.0e-5, 7.5e-5)),
             ('truncnorm', (5.6e-20, 5.5e-20, 7.8e-21, 2.2e-19)),
             ('triangular', (0.2, 0.4, 0.3)),
             ('truncnorm', (0.15, 0.0276, 0.097, 0.185)))


def _fmt_dist(kind, p):
    if kind == 'truncnorm':
        return f'TruncNormal({p[0]:g}, {p[1]:g}; [{p[2]:g}, {p[3]:g}])'
    if kind == 'normal':
        return f'Normal({p[0]:g}, {p[1]:g})'
    if kind == 'uniform':
        return f'Uniform[{p[0]:g}, {p[1]:g}]'
    return f'Triangular[{p[0]:g}, {p[1]:g}], mode {p[2]:g}'


DISTRIBUTIONS = tuple(_fmt_dist(k, p) for k, p in MARGINALS)


def _frozen(kind, p):
    if kind == 'truncnorm':
        mu, s, lo, hi = p
        return stats.truncnorm((lo - mu) / s, (hi - mu) / s, loc=mu, scale=s)
    if kind == 'normal':
        return stats.norm(loc=p[0], scale=p[1])
    if kind == 'uniform':
        return stats.uniform(loc=p[0], scale=p[1] - p[0])
    lo, hi, mode = p
    return stats.triang((mode - lo) / (hi - lo), loc=lo, scale=hi - lo)


_DISTS = tuple(_frozen(k, p) for k, p in MARGINALS)

# Reference ("nominal") point: mu of the (truncated) normals, mid-range of the uniform, mode of
# the triangular, as in the UQTestFuns documentation figures.
NOMINAL = np.array([1.79, 978.0, 2480.0, 12.8e9, 5.25e-5, 5.6e-20, 0.3, 0.15])

# ------------------------------------------------------------------ held parameters and grid
Q = 3000.0          # W, source power
A_W = 4.0e-4        # 1/K, volumetric thermal expansion coefficient of water
MU_W = 1.0e-3       # Pa s, dynamic viscosity of water
DAY = 86400.0       # s
R_GRID = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0])     # m
T_DAYS = np.array([1, 7, 30, 90, 180, 365, 730, 1095, 1460, 1825])                # d (1 y = 365 d)
T_GRID = T_DAYS * DAY                                                             # s
T_LABELS = ('1d', '7d', '30d', '90d', '180d', '1y', '2y', '3y', '4y', '5y')
R_STAR = 6.0                # m,  radius of the scalar quantities of interest
T_STAR = 365 * DAY          # s,  time of the scalar quantities of interest (1 y)

_SQPI = np.sqrt(np.pi)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_GL_S = 0.5 * (_GL_X + 1.0)          # nodes on [0, 1]
_GL_W = 0.5 * _GL_W


# ------------------------------------------------------------------ distributions
def mean_and_sd():
    """Exact mean and standard deviation of each input marginal."""
    return (np.array([d.mean() for d in _DISTS]), np.array([d.std() for d in _DISTS]))


def bounds():
    """Support of each marginal (C is an untruncated normal: -inf, inf)."""
    s = np.array([d.support() for d in _DISTS], dtype=float)
    return s[:, 0], s[:, 1]


def ppf(U):
    """Map uniforms U in (0, 1)^(N x 8) to physical inputs by the inverse marginal CDFs."""
    U = np.atleast_2d(U)
    return np.column_stack([d.ppf(U[:, j]) for j, d in enumerate(_DISTS)])


def sample(n, rng):
    """n independent draws: inverse-CDF transform of rng.random((n, 8))."""
    return ppf(np.random.default_rng(rng).random((n, 8)))


def to_gaussian(X):
    """Map physical inputs to independent standard normals z_j = Phi^-1(F_j(x_j))."""
    X = np.atleast_2d(X)
    return np.column_stack([stats.norm.ppf(d.cdf(X[:, j])) for j, d in enumerate(_DISTS)])


def dx_dz(X):
    """Jacobian diagonal dx_j/dz_j of the map z -> x (multiply physical gradients by it)."""
    X = np.atleast_2d(X)
    Z = to_gaussian(X)
    return np.column_stack([stats.norm.pdf(Z[:, j]) / d.pdf(X[:, j]) for j, d in enumerate(_DISTS)])


# ------------------------------------------------------------------ derived constants
def derived(X, Q=Q, a_w=A_W, mu_w=MU_W):
    """Derived material constants (arrays of length N); Q, a_w, mu_w are the held parameters."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    K, C, rho, E, a_s, k_s, nu, phi = X.T
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    M = lam + 2 * G
    kappa = K / (C * rho)
    c = k_s / mu_w * M
    a_u = phi * a_w + (1 - phi) * a_s
    bp = (lam + 2 * G / 3) * a_s
    Xc = a_u * M - bp
    return {'kappa': kappa, 'lam': lam, 'G': G, 'M': M, 'c': c, 'a_u': a_u, 'bp': bp, 'Xc': Xc,
            'q': Q / (4 * np.pi * K)}


def _derived_jacobian(X, d, a_w=A_W, mu_w=MU_W):
    """d(q, kappa, c, M, bp, Xc)/dx, shape (N, 6, 8)."""
    K, C, rho, E, a_s, k_s, nu, phi = X.T
    N = len(X)
    J = np.zeros((N, 6, 8))
    dlnM_dnu = -1 / (1 - nu) - 1 / (1 + nu) + 2 / (1 - 2 * nu)     # M = E (1-nu)/((1+nu)(1-2nu))
    Kb = E / (3 * (1 - 2 * nu))                                       # bulk modulus; bp = Kb a_s
    J[:, 0, 0] = -d['q'] / K
    J[:, 1, 0] = d['kappa'] / K
    J[:, 1, 1] = -d['kappa'] / C
    J[:, 1, 2] = -d['kappa'] / rho
    J[:, 2, 3] = d['c'] / E
    J[:, 2, 5] = d['M'] / mu_w
    J[:, 2, 6] = d['c'] * dlnM_dnu
    J[:, 3, 3] = d['M'] / E
    J[:, 3, 6] = d['M'] * dlnM_dnu
    J[:, 4, 3] = Kb * a_s / E
    J[:, 4, 4] = Kb
    J[:, 4, 6] = Kb * a_s * 2 / (1 - 2 * nu)
    J[:, 5, 3] = d['Xc'] / E
    J[:, 5, 4] = (1 - phi) * d['M'] - Kb
    J[:, 5, 6] = d['a_u'] * J[:, 3, 6] - J[:, 4, 6]
    J[:, 5, 7] = (a_w - a_s) * d['M']
    return J


# ------------------------------------------------------------------ front functions of A
def _phi(kind, A, r, t):
    """f(A) or g(A) at radius r and time t (broadcasting)."""
    tau = A * t / r**2
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        z = 0.5 / np.sqrt(tau)
        if kind == 'f':
            return erfc(z)
        return tau + (0.5 - tau) * erfc(z) - np.sqrt(tau / np.pi) * np.exp(-z * z)


def _dphi(kind, A, r, t):
    """d f/dA or d g/dA."""
    tau = A * t / r**2
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        z = 0.5 / np.sqrt(tau)
        e = np.exp(-z * z)
        if kind == 'f':
            return z * e / (_SQPI * A)
        return (t / r**2) * (erf(z) - 2 / _SQPI * z * e)


def _d2phi(kind, A, r, t):
    """d^2 f/dA^2 or d^2 g/dA^2."""
    tau = A * t / r**2
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        z = 0.5 / np.sqrt(tau)
        e = np.exp(-z * z)
        if kind == 'f':
            return z * e * (z * z - 1.5) / (_SQPI * A * A)
        return -(t / r**2) * 2 / _SQPI * z**3 * e / A


def _g_remainder(A, r, t):
    """g(A) - tau = (1/2 - tau) erfc(z) - sqrt(tau/pi) exp(-z^2), evaluated with the scaled
    complementary error function so that no cancellation against tau occurs (used for z > 1)."""
    tau = A * t / r**2
    z = 0.5 / np.sqrt(tau)
    return np.exp(-z * z) / _SQPI * ((0.5 - tau) * _SQPI * erfcx(z) - 0.5 / z)


def _near(a, b, r, t):
    """Where the divided difference is evaluated by quadrature: |a - b| / min(a, b) * max(1, z_max^2) < 1,
    z_max = r / (2 sqrt(min(a, b) t)) (the integrand h'(A) then varies smoothly over [b, a])."""
    lo = np.minimum(a, b)
    zmax2 = r**2 / (4 * lo * t)
    return np.abs(a - b) / lo * np.maximum(1.0, zmax2) < 1.0


def divided(kind, a, b, r, t, grad=False):
    """D = (h(a) - h(b)) / (1 - b/a) = a (h(a) - h(b)) / (a - b) for h = f or g, and optionally
    dD/da, dD/db.  Arrays broadcast to a common shape."""
    a, b, r, t = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (a, b, r, t)))
    near = _near(a, b, r, t)
    far = ~near
    S = np.empty(a.shape)
    Sa = np.empty(a.shape) if grad else None
    Sb = np.empty(a.shape) if grad else None
    # direct quotient.  For g ahead of both fronts (z > 1 at a and at b), g(A) = tau + O(exp(-z^2)):
    # the linear part tau = A t / r^2 is divided exactly and only the remainder g - tau is
    # differenced, which avoids cancellation in dD/da and dD/db.
    af, bf, rf, tf = a[far], b[far], r[far], t[far]
    lin = np.zeros(af.shape)
    if kind == 'g':
        lin = np.where(rf**2 > 4 * np.maximum(af, bf) * tf, tf / rf**2, 0.0)
    if kind == 'g':
        use = lin > 0
        ha, hb = _phi(kind, af, rf, tf), _phi(kind, bf, rf, tf)
        ha[use] = _g_remainder(af[use], rf[use], tf[use])
        hb[use] = _g_remainder(bf[use], rf[use], tf[use])
    else:
        ha, hb = _phi(kind, af, rf, tf), _phi(kind, bf, rf, tf)
    q = (ha - hb) / (af - bf)
    S[far] = q + lin
    if grad:
        Sa[far] = (_dphi(kind, af, rf, tf) - lin - q) / (af - bf)
        Sb[far] = (q - _dphi(kind, bf, rf, tf) + lin) / (af - bf)
    # quadrature of h' (and h'') over the segment A = b + s (a - b), s in [0, 1]
    an, bn, rn, tn = a[near][:, None], b[near][:, None], r[near][:, None], t[near][:, None]
    A = bn + _GL_S * (an - bn)
    S[near] = _dphi(kind, A, rn, tn) @ _GL_W
    if grad:
        h2 = _d2phi(kind, A, rn, tn)
        Sa[near] = h2 @ (_GL_W * _GL_S)
        Sb[near] = h2 @ (_GL_W * (1 - _GL_S))
    D = a * S
    if not grad:
        return D
    return D, S + a * Sa, a * Sb


# ------------------------------------------------------------------ model
def _evaluate(X, r, t, grad, Q=Q, a_w=A_W, mu_w=MU_W):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    r = np.asarray(r, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if X.shape[1] != 8:
        raise ValueError('X must have 8 columns: ' + ', '.join(NAMES))
    if not np.all(np.isfinite(X)):
        raise ValueError('inputs must be finite')
    K, C, rho, E, a_s, k_s, nu, phi = X.T
    bad = ((K <= 0) | (C <= 0) | (rho <= 0) | (E <= 0) | (a_s < 0) | (k_s <= 0)
           | (nu <= -1) | (nu >= 0.5) | (phi < 0) | (phi > 1))
    if np.any(bad):
        raise ValueError(f'{int(bad.sum())} input rows outside the physical domain '
                         '(K, C, rho, E, k_s > 0; a_s >= 0; -1 < nu < 0.5; 0 <= phi <= 1)')
    if np.any(r <= 0) or np.any(t <= 0) or not (np.all(np.isfinite(r)) and np.all(np.isfinite(t))):
        raise ValueError('radii and times must be finite and positive')
    if not (Q > 0 and mu_w > 0 and a_w >= 0 and np.isfinite(Q + mu_w + a_w)):
        raise ValueError('held parameters need Q > 0, mu_w > 0, a_w >= 0')
    d = derived(X, Q, a_w, mu_w)
    kap = d['kappa'][:, None, None]
    c = d['c'][:, None, None]
    q = d['q'][:, None, None]
    M = d['M'][:, None, None]
    bp = d['bp'][:, None, None]
    Xc = d['Xc'][:, None, None]
    R = r[None, :, None]
    T = t[None, None, :]
    fk = _phi('f', kap, R, T)
    gk = _phi('g', kap, R, T)
    if grad:
        Df, Df_k, Df_c = divided('f', kap, c, R, T, grad=True)
        Dg, Dg_k, Dg_c = divided('g', kap, c, R, T, grad=True)
    else:
        Df = divided('f', kap, c, R, T)
        Dg = divided('g', kap, c, R, T)
    dT = q / R * fk
    p = Xc * q / R * Df
    brace = bp * gk + Xc * Dg
    u = q / M * brace
    if not grad:
        return dT, p, u
    shape = dT.shape
    # partial derivatives with respect to z = (q, kappa, c, M, bp, Xc), each of shape (N, nr, nt)
    zero = np.zeros(shape)
    fk_k = _dphi('f', kap, R, T)
    gk_k = _dphi('g', kap, R, T)
    P_dT = [fk / R + zero, q / R * fk_k, zero, zero, zero, zero]
    P_p = [Xc * Df / R + zero, Xc * q / R * Df_k, Xc * q / R * Df_c, zero, zero, q * Df / R + zero]
    P_u = [brace / M + zero, q / M * (bp * gk_k + Xc * Dg_k), q / M * Xc * Dg_c, -u / M,
           q / M * gk + zero, q / M * Dg]
    J = _derived_jacobian(X, d, a_w, mu_w)                          # (N, 6, 8)
    grads = []
    for P in (P_dT, P_p, P_u):
        Pz = np.stack(P, axis=-1)                        # (N, nr, nt, 6)
        grads.append(np.einsum('nijz,nzx->nijx', Pz, J))
    return (dT, p, u), tuple(grads)


def fields(X, r=R_GRID, t=T_GRID, Q=Q, a_w=A_W, mu_w=MU_W):
    """dT (K), p (Pa), u_r (m), each of shape (N, len(r), len(t)); t in seconds."""
    return _evaluate(X, r, t, False, Q, a_w, mu_w)


def fields_and_grads(X, r=R_GRID, t=T_GRID, Q=Q, a_w=A_W, mu_w=MU_W):
    """((dT, p, u_r), (grad dT, grad p, grad u_r)); gradients with respect to the physical
    inputs, shape (N, len(r), len(t), 8).  The values equal those of fields() bit for bit."""
    return _evaluate(X, r, t, True, Q, a_w, mu_w)


def headline(X):
    """Scalar QoIs at (R_STAR, T_STAR): F (N, 3) = (dT, p, u_r) and exact gradients G (N, 3, 8)."""
    (dT, p, u), (gT, gp, gu) = _evaluate(X, [R_STAR], [T_STAR], grad=True)
    F = np.column_stack([dT[:, 0, 0], p[:, 0, 0], u[:, 0, 0]])
    G = np.stack([gT[:, 0, 0], gp[:, 0, 0], gu[:, 0, 0]], axis=1)
    return F, G


def u_r_by_quadrature(x, r, t, n=400):
    """Independent check of u_r: in spherical symmetry M div(u) = p + b' dT, so
    u_r(r) = 1/(M r^2) * int_0^r s^2 (p + b' dT)(s) ds (Gauss-Legendre in s)."""
    x = np.asarray(x, dtype=float)[None]
    d = derived(x)
    xs, ws = np.polynomial.legendre.leggauss(n)
    s = 0.5 * r * (xs + 1)
    w = 0.5 * r * ws
    dT, p, _ = fields(x, s, [t])
    integrand = s**2 * (p[0, :, 0] + d['bp'][0] * dT[0, :, 0])
    return float(integrand @ w / (d['M'][0] * r**2))


def pde_residuals(x, r, t, h=1e-3):
    """Relative residuals of the field equations at (r, t), by central differences:
    dT_t - kappa lap(dT) = 0 and p_t - c lap(p) - X dT_t = 0 (lap = radial Laplacian)."""
    x = np.asarray(x, dtype=float)[None]
    d = derived(x)
    hr, ht = h * r, h * t
    rr = np.array([r - hr, r, r + hr])
    tt = np.array([t - ht, t, t + ht])
    dT, p, _ = fields(x, rr, tt)
    out = []
    for F, A, src in ((dT[0], d['kappa'][0], None), (p[0], d['c'][0], dT[0])):
        Ft = (F[1, 2] - F[1, 0]) / (2 * ht)
        Frr = (F[2, 1] - 2 * F[1, 1] + F[0, 1]) / hr**2
        Fr = (F[2, 1] - F[0, 1]) / (2 * hr)
        lap = Frr + 2 / r * Fr
        res = Ft - A * lap
        scale = abs(Ft) + abs(A * lap)
        if src is not None:
            St = d['Xc'][0] * (src[1, 2] - src[1, 0]) / (2 * ht)
            res -= St
            scale += abs(St)
        out.append(abs(res) / scale)
    return tuple(out)


# ------------------------------------------------------------------ UQTestFuns notice
# The input distributions (MARGINALS), held parameters (Q, A_W, MU_W) and the r-t grid are
# taken from UQTestFuns (src/uqtestfuns/test_functions/saturated_point_heat_source.yaml at
# https://github.com/damar-wicaksono/uqtestfuns, branch dev, commit
# 82d5c8b1e2cc9e746ba7f22a53e3c7f1ad12d42f; unreleased, not in the 0.7.0 PyPI release),
# distributed under the following licence:
#
# MIT License
#
# Copyright (c) 2022 Damar Wicaksono
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
