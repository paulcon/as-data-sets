"""Simplified river flood model of Lamboni, Iooss, Popelin and Gamboa (2013), Sec. 4.3.

A crude closed-form simplification of the 1-D Saint-Venant equations (uniform,
constant flow in a wide rectangular channel, Strickler friction law). The
maximal annual river height H and the maximal annual overflow S above the dyke
are (Eq. (4.20) of the paper)

    H = ( Q / (B Ks sqrt((Zm - Zv)/L)) )^0.6,      S = Zv + H - Hd - Cb      (m),

and the associated cost in million euros is (Eq. (4.21))

    Cp = 1{S > 0} + [0.2 + 0.8 (1 - exp(-1000/S^4))] 1{S <= 0}
         + (Hd 1{Hd > 8} + 8 1{Hd <= 8}) / 20.

The eight inputs are independent with the distributions of Table 4 of the paper
(p. 14 of the arXiv v2 manuscript):

    Q   truncated Gumbel G(1013, 558) on [500, 3000]   (location, scale)
    Ks  truncated normal N(30, 8) on [15, +inf)         (mean, standard deviation)
    Zv  triangular T(49, 50, 51)                         (min, mode, max)
    Zm  triangular T(54, 55, 56)
    Hd  uniform U[7, 9]
    Cb  triangular T(55, 55.5, 56)
    L   triangular T(4990, 5000, 5010)
    B   triangular T(295, 300, 305)

`sample` draws them by the inverse CDF. `grad_overflow` is the exact gradient
of S with respect to the physical inputs. `to_gaussian` maps the inputs to
independent standard normals z_j = Phi^{-1}(F_j(x_j)) (the transformed
variables of the notebook's second analysis), and `dx_dz` is dx_j/dz_j.
`exact_moments` computes Var(S), the DGSM nu_j = E[(dS/dx_j)^2] and the
first-order and total Sobol' indices of S by quadrature.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy import integrate, special

#: Input names, in column order (Table 4 of the paper).
NAMES = ('Q', 'Ks', 'Zv', 'Zm', 'Hd', 'Cb', 'L', 'B')

#: Units of the inputs (Table 4; Ks is listed there without a unit, it is m^(1/3)/s).
UNITS = ('m^3/s', 'm^(1/3)/s', 'm', 'm', 'm', 'm', 'm', 'm')

#: Descriptions (Table 4).
DESCRIPTIONS = ('maximal annual flowrate', 'Strickler coefficient', 'river downstream level',
                'river upstream level', 'dyke height', 'bank level',
                'length of the river stretch', 'river width')

#: Distributions as written in Table 4.
DISTRIBUTIONS = ('truncated Gumbel G(1013, 558) on [500, 3000]',
                 'truncated normal N(30, 8) on [15, +inf)',
                 'triangular T(49, 50, 51)', 'triangular T(54, 55, 56)', 'uniform U[7, 9]',
                 'triangular T(55, 55.5, 56)', 'triangular T(4990, 5000, 5010)',
                 'triangular T(295, 300, 305)')

Q_LOC, Q_SCALE, Q_LO, Q_HI = 1013.0, 558.0, 500.0, 3000.0
KS_MEAN, KS_SD, KS_LO = 30.0, 8.0, 15.0
HD_LO, HD_HI = 7.0, 9.0
#: Triangular inputs: (min, mode, max).
TRIANGULAR = {'Zv': (49.0, 50.0, 51.0), 'Zm': (54.0, 55.0, 56.0), 'Cb': (55.0, 55.5, 56.0),
              'L': (4990.0, 5000.0, 5010.0), 'B': (295.0, 300.0, 305.0)}

#: Lower and upper ends of the supports (Ks has no upper end).
LOWER = np.array([Q_LO, KS_LO, 49.0, 54.0, HD_LO, 55.0, 4990.0, 295.0])
UPPER = np.array([Q_HI, np.inf, 51.0, 56.0, HD_HI, 56.0, 5010.0, 305.0])

#: A reference point: the Gumbel location, the normal mean, the modes and the
#: centre of the uniform. Used as the EMA nominal point; it is not a quantity of the paper.
NOMINAL = np.array([1013.0, 30.0, 50.0, 55.0, 8.0, 55.5, 5000.0, 300.0])

#: Exponents of H = Q^0.6 Ks^-0.6 B^-0.6 L^0.3 (Zm - Zv)^-0.3 in Q, Ks, B and L.
_EXPONENT = {'Q': 0.6, 'Ks': -0.6, 'B': -0.6, 'L': 0.3}


# --------------------------------------------------------------------- model
def _cols(x):
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 8:
        raise ValueError(f'expected 8 inputs in the last axis, got shape {x.shape}')
    return x, [x[..., j] for j in range(8)]


def _check_domain(x):
    """Raise unless Q, Ks, L, B > 0 and Zm > Zv, where H is defined."""
    _, (Q, Ks, Zv, Zm, Hd, Cb, L, B) = _cols(x)
    if not np.all(np.isfinite(x)):
        raise ValueError('non-finite input')
    if np.any(Q <= 0) or np.any(Ks <= 0) or np.any(L <= 0) or np.any(B <= 0):
        raise ValueError('Q, Ks, L and B must be positive')
    if np.any(Zm <= Zv):
        raise ValueError('the upstream level Zm must exceed the downstream level Zv')


def river_height(x):
    """Maximal annual river height H in m, Eq. (4.20); x has shape (..., 8)."""
    x, (Q, Ks, Zv, Zm, Hd, Cb, L, B) = _cols(x)
    _check_domain(x)
    return (Q / (B * Ks * np.sqrt((Zm - Zv) / L)))**0.6


def overflow(x):
    """Maximal annual overflow S = Zv + H - Hd - Cb in m, Eq. (4.20)."""
    x, (Q, Ks, Zv, Zm, Hd, Cb, L, B) = _cols(x)
    return Zv + river_height(x) - Hd - Cb


def grad_overflow(x):
    """Exact gradient of S with respect to the physical inputs, shape (..., 8).

    dS/dQ = 0.6 H/Q, dS/dKs = -0.6 H/Ks, dS/dZv = 1 + 0.3 H/(Zm - Zv),
    dS/dZm = -0.3 H/(Zm - Zv), dS/dHd = dS/dCb = -1, dS/dL = 0.3 H/L, dS/dB = -0.6 H/B.
    """
    x, (Q, Ks, Zv, Zm, Hd, Cb, L, B) = _cols(x)
    H = river_height(x)
    dz = Zm - Zv
    one = np.ones_like(H)
    return np.stack([0.6 * H / Q, -0.6 * H / Ks, 1 + 0.3 * H / dz, -0.3 * H / dz,
                     -one, -one, 0.3 * H / L, -0.6 * H / B], axis=-1)


def _maintenance(S):
    """0.2 + 0.8 (1 - exp(-1000/S^4)) for S <= 0 (its limit at S = 0 is 1)."""
    S = np.asarray(S, dtype=float)
    with np.errstate(divide='ignore', over='ignore'):
        e = np.exp(-1000.0 / S**4)
    return 0.2 + 0.8 * (1.0 - e)


def cost(x):
    """Cost Cp in million euros, Eq. (4.21).

    Within the support of Table 4, S <= -3.6 m, so the flooding term 1{S > 0}
    never fires; the investment term max(Hd, 8)/20 has a kink at Hd = 8 m.
    """
    x, (Q, Ks, Zv, Zm, Hd, Cb, L, B) = _cols(x)
    S = overflow(x)
    return np.where(S > 0, 1.0, _maintenance(S)) + np.where(Hd > 8, Hd, 8.0) / 20.0


def grad_cost(x):
    """Gradient of Cp with respect to the physical inputs, where it exists, shape (..., 8).

    For S < 0, dCp/dS = -3200 S^-5 exp(-1000/S^4) (> 0); for S > 0 it is 0. The
    investment term adds 1/20 to dCp/dHd for Hd > 8 and 0 for Hd <= 8 (the
    one-sided derivative from the left at the kink Hd = 8).
    """
    x, (Q, Ks, Zv, Zm, Hd, Cb, L, B) = _cols(x)
    S = overflow(x)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        dS = np.where(S < 0, -3200.0 * S**-5.0 * np.exp(-1000.0 / S**4), 0.0)
    G = dS[..., None] * grad_overflow(x)
    G[..., 4] += np.where(Hd > 8, 1.0 / 20.0, 0.0)
    return G


# ------------------------------------------------------------- distributions
_Q_FA = np.exp(-np.exp(-(Q_LO - Q_LOC) / Q_SCALE))     # Gumbel CDF at 500
_Q_FB = np.exp(-np.exp(-(Q_HI - Q_LOC) / Q_SCALE))     # Gumbel CDF at 3000
_KS_ZA = (KS_LO - KS_MEAN) / KS_SD
_KS_PA = special.ndtr(_KS_ZA)                           # normal CDF at 15


def _tri_cdf(x, a, c, b):
    return np.where(x <= c, (x - a)**2 / ((b - a) * (c - a)), 1 - (b - x)**2 / ((b - a) * (b - c)))


def _tri_sf(x, a, c, b):
    return np.where(x <= c, 1 - (x - a)**2 / ((b - a) * (c - a)), (b - x)**2 / ((b - a) * (b - c)))


def _tri_pdf(x, a, c, b):
    return np.where(x <= c, 2 * (x - a) / ((b - a) * (c - a)), 2 * (b - x) / ((b - a) * (b - c)))


def _tri_ppf(u, a, c, b):
    fc = (c - a) / (b - a)
    return np.where(u < fc, a + np.sqrt(u * (b - a) * (c - a)),
                    b - np.sqrt((1 - u) * (b - a) * (b - c)))


def cdf_sf_pdf(x):
    """Marginal CDF F_j, survival 1 - F_j and density rho_j of each column, shape (..., 8) each."""
    x = np.asarray(x, dtype=float)
    F, SF, rho = np.empty_like(x), np.empty_like(x), np.empty_like(x)
    # Q: truncated Gumbel (maximum)
    t = np.exp(-(x[..., 0] - Q_LOC) / Q_SCALE)
    g = np.exp(-t)
    F[..., 0] = (g - _Q_FA) / (_Q_FB - _Q_FA)
    SF[..., 0] = (_Q_FB - g) / (_Q_FB - _Q_FA)
    rho[..., 0] = t * g / Q_SCALE / (_Q_FB - _Q_FA)
    # Ks: truncated normal
    z = (x[..., 1] - KS_MEAN) / KS_SD
    F[..., 1] = (special.ndtr(z) - _KS_PA) / (1 - _KS_PA)
    SF[..., 1] = special.ndtr(-z) / (1 - _KS_PA)
    rho[..., 1] = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi) / KS_SD / (1 - _KS_PA)
    # Hd: uniform
    F[..., 4] = (x[..., 4] - HD_LO) / (HD_HI - HD_LO)
    SF[..., 4] = (HD_HI - x[..., 4]) / (HD_HI - HD_LO)
    rho[..., 4] = 1 / (HD_HI - HD_LO)
    for j, n in ((2, 'Zv'), (3, 'Zm'), (5, 'Cb'), (6, 'L'), (7, 'B')):
        a, c, b = TRIANGULAR[n]
        F[..., j], SF[..., j], rho[..., j] = (_tri_cdf(x[..., j], a, c, b),
                                              _tri_sf(x[..., j], a, c, b),
                                              _tri_pdf(x[..., j], a, c, b))
    return F, SF, rho


def ppf(u):
    """Inverse marginal CDFs: physical inputs from u in [0, 1)^8, shape (..., 8)."""
    u = np.asarray(u, dtype=float)
    x = np.empty_like(u)
    x[..., 0] = Q_LOC - Q_SCALE * np.log(-np.log(_Q_FA + u[..., 0] * (_Q_FB - _Q_FA)))
    x[..., 1] = KS_MEAN + KS_SD * special.ndtri(_KS_PA + u[..., 1] * (1 - _KS_PA))
    x[..., 4] = HD_LO + u[..., 4] * (HD_HI - HD_LO)
    for j, n in ((2, 'Zv'), (3, 'Zm'), (5, 'Cb'), (6, 'L'), (7, 'B')):
        x[..., j] = _tri_ppf(u[..., j], *TRIANGULAR[n])
    return x


def sample(M, rng):
    """M independent draws of the eight inputs (Table 4), by the inverse CDF of rng.random((M, 8))."""
    return ppf(rng.random((M, 8)))


def to_gaussian(x):
    """z_j = Phi^{-1}(F_j(x_j)), independent standard normals, shape (..., 8)."""
    F, SF, _ = cdf_sf_pdf(x)
    return np.where(F < 0.5, special.ndtri(F), -special.ndtri(SF))


def dx_dz(x):
    """dx_j/dz_j = phi(z_j)/rho_j(x_j) for the map z = to_gaussian(x), shape (..., 8)."""
    z = to_gaussian(x)
    _, _, rho = cdf_sf_pdf(x)
    return np.exp(-z**2 / 2) / np.sqrt(2 * np.pi) / rho


# ------------------------------------------------------- moments by quadrature
def _gl(lo, hi, k=40):
    t, w = np.polynomial.legendre.leggauss(k)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def _tri_rule(a, c, b, k=40):
    """Nodes and probability weights for a triangular density (Gauss-Legendre on each side of the mode)."""
    x1, w1 = _gl(a, c, k)
    x2, w2 = _gl(c, b, k)
    x = np.concatenate([x1, x2])
    return x, np.concatenate([w1, w2]) * _tri_pdf(x, a, c, b)


def _expect(j, fun):
    """E[fun(X_j)] for one of the inputs Q (0), Ks (1), Hd (4), or a triangular one."""
    if j == 0:
        pdf = lambda q: cdf_sf_pdf(np.array([q, 30, 50, 55, 8, 55.5, 5000, 300]))[2][0]
        return integrate.quad(lambda q: fun(q) * pdf(q), Q_LO, Q_HI, epsabs=0, epsrel=1e-13,
                              limit=200)[0]
    if j == 1:
        c = 1 / (np.sqrt(2 * np.pi) * KS_SD * (1 - _KS_PA))
        return integrate.quad(lambda k: fun(k) * c * np.exp(-((k - KS_MEAN) / KS_SD)**2 / 2),
                              KS_LO, np.inf, epsabs=0, epsrel=1e-13, limit=200)[0]
    if j == 4:
        x, w = _gl(HD_LO, HD_HI)
        return np.sum(w * fun(x)) / (HD_HI - HD_LO)
    x, w = _tri_rule(*TRIANGULAR[NAMES[j]])
    return np.sum(w * fun(x))


def mean_and_sd():
    """Exact mean and standard deviation of each marginal, two arrays of shape (8,)."""
    m = np.array([_expect(j, lambda x: x) for j in range(8)])
    v = np.array([_expect(j, lambda x, mj=m[j]: (x - mj)**2) for j in range(8)])
    return m, np.sqrt(v)


def exact_moments():
    """Var(S), E[S], DGSM nu_j and Sobol' indices S_j, ST_j of the overflow, by quadrature.

    Write H = P G with P = Q^0.6 Ks^-0.6 B^-0.6 L^0.3 (independent factors) and
    G = (Zm - Zv)^-0.3; every quantity reduces to 1-D moments of the factors of P
    and 2-D Gauss-Legendre integrals over (Zv, Zm). Returns a dict.
    """
    idx = {n: j for j, n in enumerate(NAMES)}
    mom = {n: (lambda p, j=idx[n]: _expect(j, lambda x: x**p)) for n in _EXPONENT}

    def EP(k, skip=None):  # E[P^k], optionally without the factor of input `skip`
        return np.prod([mom[n](_EXPONENT[n] * k) for n in _EXPONENT if n != skip])

    zv, wv = _tri_rule(*TRIANGULAR['Zv'])
    zm, wm = _tri_rule(*TRIANGULAR['Zm'])
    V, M = np.meshgrid(zv, zm, indexing='ij')           # axis 0: Zv, axis 1: Zm
    W = np.outer(wv, wm)
    D2 = M - V
    E2 = lambda arr: np.sum(W * arr)                     # E over (Zv, Zm)
    G, EG, EG2 = D2**-0.3, E2(D2**-0.3), E2(D2**-0.6)
    var_zv = E2((V - 50.0)**2)
    var_hd = (HD_HI - HD_LO)**2 / 12
    xc, wc = _tri_rule(*TRIANGULAR['Cb'])
    var_cb = np.sum(wc * (xc - 55.5)**2)
    P1, P2 = EP(1), EP(2)
    cov_zv_g = E2((V - 50.0) * G)
    var_h = P2 * EG2 - P1**2 * EG**2
    D = var_zv + 2 * P1 * cov_zv_g + var_h + var_hd + var_cb
    ES = 50.0 + P1 * EG - 8.0 - 55.5

    nu, Vj, Tj = np.empty(8), np.empty(8), np.empty(8)
    for n, a in _EXPONENT.items():
        j = idx[n]
        Ex2a2 = mom[n](2 * a - 2)
        nu[j] = a**2 * Ex2a2 * EP(2, skip=n) * EG2
        var_p = mom[n](2 * a) - mom[n](a)**2
        Vj[j] = var_p * EP(1, skip=n)**2 * EG**2
        Tj[j] = var_p * EP(2, skip=n) * EG2
    E26, E13 = E2(D2**-2.6), E2(D2**-1.3)
    nu[idx['Zm']] = 0.09 * P2 * E26
    nu[idx['Zv']] = 1 + 0.6 * P1 * E13 + 0.09 * P2 * E26
    nu[idx['Hd']] = nu[idx['Cb']] = 1.0
    # Zm: conditional on Zm (average over Zv along axis 0), and on Zv (vary Zm along axis 1)
    g_m = np.sum(wv[:, None] * G, axis=0)                # E[G | Zm]
    Vj[idx['Zm']] = P1**2 * (np.sum(wm * g_m**2) - EG**2)
    g_given_v = np.sum(wm[None, :] * G, axis=1)          # E[G | Zv]
    var_g_given_v = np.sum(wm[None, :] * G**2, axis=1) - g_given_v**2
    Tj[idx['Zm']] = P2 * np.sum(wv * var_g_given_v)
    # Zv: E[S | Zv] = Zv + E[P] E[G | Zv] + const
    h = zv + P1 * g_given_v
    Vj[idx['Zv']] = np.sum(wv * h**2) - np.sum(wv * h)**2
    # total: E over (P, Zm) of Var over Zv of (Zv + P G)
    mv = np.sum(wv[:, None] * G, axis=0)                 # E[G | Zm] over Zv
    cov_v = np.sum(wv[:, None] * (zv[:, None] - 50.0) * G, axis=0)
    var_v = np.sum(wv[:, None] * G**2, axis=0) - mv**2
    Tj[idx['Zv']] = var_zv + 2 * P1 * np.sum(wm * cov_v) + P2 * np.sum(wm * var_v)
    Vj[idx['Hd']] = Tj[idx['Hd']] = var_hd
    Vj[idx['Cb']] = Tj[idx['Cb']] = var_cb
    return {'var': D, 'mean': ES, 'nu': nu, 'S': Vj / D, 'ST': Tj / D}


def poincare_constants():
    """The constant C_j = 4 C1^2 of the paper's bound Upsilon_j = C_j nu_j / Var(S), per input.

    Uniform: (b - a)^2 / pi^2 (Sobol and Kucherenko; paper Sec. 3). The other
    marginals are log-concave (truncation keeps log-concavity), and for them the
    Cheeger constant is C1 = 1/(2 rho(m)) at the median m (Theorem 3.2, Table 1),
    so C_j = 1/rho_j(m_j)^2. For the truncated normal Ks the paper's Table 1
    normal entry (C1 = sigma/2, C = sigma^2 = 64) is returned separately.
    """
    med = ppf(np.full(8, 0.5))
    rho = cdf_sf_pdf(med)[2]
    C = 1 / rho**2
    C[4] = (HD_HI - HD_LO)**2 / np.pi**2
    return C, KS_SD**2
