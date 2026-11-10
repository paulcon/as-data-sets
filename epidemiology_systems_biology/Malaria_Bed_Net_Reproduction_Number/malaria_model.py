"""Effective reproduction number of the two-strain malaria model of Bala and Gimba (2019).

S. Bala and B. Gimba, "Global sensitivity analysis to study the impacts of bed-nets, drug
treatment, and their efficacies on a two-strain malaria model", Mathematical and Computational
Applications 24(1):32, 2019, doi:10.3390/mca24010032.

The quantity of interest is the closed-form effective reproduction number of the scaled model
(Eq. 15), Eq. (19), p. 11:

    R_eff = xi1 k1 (1 - beta) / (p2 psi_v) + k1 xi1 beta (p2 + p3) / (p2 p1 psi_v),

with the biting rate a (Eq. 1, p. 4), xi1 = a b1 Nv/Nh = a b1 rho kv/kh and k1 = a b2 (p. 10),
the carrying capacities kv = (psi_v - mu_v - mu_v3 b gamma)/mu_v2 and kh = (psi_h - mu_h)/mu_h2
(proof of Lemma 1, p. 8), and p1 = r_s + delta_h + T_s + psi_h, p2 = r_r + delta_h + psi_h,
p3 = T_s (1 - e) (p. 10).

Twenty of the 23 parameters of Table 3 (p. 19) enter R_eff; alpha, alpha_s and alpha_r do not
(they appear only in the ODEs), and they are not inputs here. `INPUTS` lists the 20 in the order
of Table 3; `TABLE3` keeps all 23 rows.

Functions
---------
reff(x, grad=False)       R_eff at one input vector (physical units), optionally its exact gradient
reffs(X, grad=False)      the same for an (M, 20) array
strain_parts(x)           (R_r, R_s) of Eq. (20)
ngm(x)                    the next-generation matrix F V^{-1} of Eqs. (16)-(17)
sample(M, rng, lhs=True)  M draws from the Table 3 distributions (Latin hypercube or Monte Carlo)

Only numpy is imported.
"""
import numpy as np

# Table 3, p. 19: name, symbol, units (Table 2, p. 7), baseline, lower, upper, distribution, peak
# (peak only for the triangular rows). Units are those implied by the equations (Eqs. 1, 3-5):
# Table 2 prints psi_h, mu_h, mu_h2 as 'human x day^-1', psi_v, mu_v as 'mosquitoes x day^-1',
# mu_v2 as 'day^-1' and the efficacies e, gamma as 'day^-1', but psi_h, psi_v, mu_h, mu_v multiply
# a population to give a rate (per-capita rates, 1/day), mu_h2 and mu_v2 multiply a population
# to give a per-capita rate, and e, gamma are proportions in [0, 1] (Eq. 1).
TABLE3 = [
    ('psi_h', 'ψ_h', '1/day', 9.3614e-5, 6.0849e-5, 12.17e-5, 'uniform', None),
    ('psi_v', 'ψ_v', '1/day', 0.4478, 0.2911, 0.7, 'uniform', None),
    ('T_s', 'T_s', '1/day', 0.35, 0.2275, 0.455, 'uniform', None),
    ('b1', 'b_1', '-', 0.75, 0.1, 0.8, 'uniform', None),
    ('b2', 'b_2', '-', 0.5342, 0.072, 0.64, 'uniform', None),
    ('rho', 'ρ', '-', 7.0, 2.0, 8.0, 'uniform', None),
    ('b', 'b', '-', 0.53, 0.1325, 0.6625, 'triangular', 0.5),
    ('beta_max', 'β_max', '1/day', 0.6334, 0.1, 1.0, 'uniform', None),
    ('beta_min', 'β_min', '1/day', 0.0696, 0.0, 0.1, 'uniform', None),
    ('e', 'e', '-', 0.4, 0.01, 0.61, 'uniform', None),
    ('gamma', 'γ', '-', 0.5, 0.2, 1.0, 'uniform', None),
    ('alpha_s', 'α_s', '1/day', 0.0017, 0.001105, 0.00221, 'uniform', None),
    ('alpha_r', 'α_r', '1/day', 0.0017, 0.001105, 0.00221, 'triangular', 0.0017),
    ('alpha', 'α', '-', 0.3, 0.195, 0.39, 'uniform', None),
    ('r_r', 'r_r', '1/day', 0.0078, 0.00507, 0.01014, 'triangular', 0.0078),
    ('r_s', 'r_s', '1/day', 0.0078, 0.00507, 0.01014, 'uniform', None),
    ('beta', 'β', '-', 0.7, 0.455, 0.91, 'uniform', None),
    ('delta_h', 'δ_h', '1/day', 1e-3, 0.00065, 0.0013, 'uniform', None),
    ('mu_h2', 'μ_h2', '1/(human day)', 1e-7, 6.5e-8, 13e-8, 'uniform', None),
    ('mu_h', 'μ_h', '1/day', 4.212e-5, 2.74e-5, 5.48e-5, 'uniform', None),
    ('mu_v', 'μ_v', '1/day', 0.1429, 0.092885, 0.18577, 'uniform', None),
    ('mu_v2', 'μ_v2', '1/(mosquito day)', 2.28e-4, 1.48e-4, 2.96e-4, 'uniform', None),
    ('mu_v3', 'μ_v3', '1/day', 0.0995, 0.064675, 0.12935, 'uniform', None),
]
NOT_IN_REFF = ('alpha_s', 'alpha_r', 'alpha')

_rows = [r for r in TABLE3 if r[0] not in NOT_IN_REFF]
INPUTS = tuple(r[0] for r in _rows)                  # the 20 inputs, Table 3 order
NAMES = INPUTS
SYMBOLS = tuple(r[1] for r in _rows)
UNITS = tuple(r[2] for r in _rows)
BASELINE = np.array([r[3] for r in _rows])
LOWER = np.array([r[4] for r in _rows])
UPPER = np.array([r[5] for r in _rows])
DIST = tuple(r[6] for r in _rows)
PEAK = np.array([np.nan if r[7] is None else r[7] for r in _rows])
M_IN = len(INPUTS)
_I = {n: i for i, n in enumerate(INPUTS)}


def _unpack(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (M_IN,):
        raise ValueError(f'x must have shape ({M_IN},), got {x.shape}')
    return {n: x[i] for n, i in _I.items()}


def _parts(x):
    """Intermediate quantities of R_eff at one input vector."""
    p = _unpack(x)
    bg = p['b'] * p['gamma']
    D = (p['beta_max'] - p['beta_min']) * bg + p['beta_min']
    a = p['beta_max'] * p['beta_min'] / D                                        # Eq. (1)
    Kv = p['psi_v'] - p['mu_v'] - p['mu_v3'] * bg
    Kh = p['psi_h'] - p['mu_h']
    if not (Kv > 0 and Kh > 0 and p['beta_min'] >= 0 and D > 0 and p['mu_v2'] > 0 and p['mu_h2'] > 0):
        raise ValueError('R_eff is defined only for psi_v - mu_v - mu_v3 b gamma > 0, psi_h > mu_h, '
                         'mu_v2, mu_h2 > 0, beta_min >= 0 and (beta_max - beta_min) b gamma + beta_min > 0')
    kv = Kv / p['mu_v2']                                                         # p. 8
    kh = Kh / p['mu_h2']
    xi1 = a * p['b1'] * p['rho'] * kv / kh                                       # p. 10
    k1 = a * p['b2']
    p1 = p['r_s'] + p['delta_h'] + p['T_s'] + p['psi_h']
    p2 = p['r_r'] + p['delta_h'] + p['psi_h']
    p3 = p['T_s'] * (1.0 - p['e'])
    return dict(p=p, bg=bg, D=D, a=a, Kv=Kv, Kh=Kh, kv=kv, kh=kh, xi1=xi1, k1=k1, p1=p1, p2=p2, p3=p3)


def strain_parts(x):
    """(R_r, R_s) of Eq. (20), p. 11; R_eff = R_r + R_s."""
    q = _parts(x)
    beta, psi_v = q['p']['beta'], q['p']['psi_v']
    c = q['xi1'] * q['k1'] / psi_v
    return c * (1 - beta) / q['p2'], c * beta * (q['p2'] + q['p3']) / (q['p1'] * q['p2'])


def reff(x, grad=False):
    """R_eff (Eq. 19) at one input vector in physical units; with grad=True also dR_eff/dx (20,).

    The gradient is exact: R_eff = (a^2 b1 b2 rho kv / (kh psi_v)) H with
    H = (1 - beta)/p2 + beta (p2 + p3)/(p1 p2), so grad R_eff = R_eff grad ln R_eff, and
    grad ln R_eff is written out term by term below.
    """
    q = _parts(x)
    p = q['p']
    beta, p1, p2, p3 = p['beta'], q['p1'], q['p2'], q['p3']
    H = (1 - beta) / p2 + beta * (p2 + p3) / (p1 * p2)
    R = q['xi1'] * q['k1'] / p['psi_v'] * H
    if not grad:
        return R
    if p['beta_min'] == 0:
        raise ValueError('the gradient of R_eff needs beta_min > 0 (R_eff = 0 and its gradient is 0 '
                         'except along beta_min at beta_min = 0)')
    g = np.zeros(M_IN)                       # d ln R / dx
    D, bg, dmm = q['D'], q['bg'], p['beta_max'] - p['beta_min']
    # 2 ln a
    g[_I['beta_max']] += 2 * (1 / p['beta_max'] - bg / D)
    g[_I['beta_min']] += 2 * (1 / p['beta_min'] - (1 - bg) / D)
    g[_I['b']] += 2 * (-dmm * p['gamma'] / D)
    g[_I['gamma']] += 2 * (-dmm * p['b'] / D)
    # ln b1 + ln b2 + ln rho
    g[_I['b1']] += 1 / p['b1']
    g[_I['b2']] += 1 / p['b2']
    g[_I['rho']] += 1 / p['rho']
    # ln kv = ln(psi_v - mu_v - mu_v3 b gamma) - ln mu_v2
    Kv = q['Kv']
    g[_I['psi_v']] += 1 / Kv
    g[_I['mu_v']] += -1 / Kv
    g[_I['mu_v3']] += -bg / Kv
    g[_I['b']] += -p['mu_v3'] * p['gamma'] / Kv
    g[_I['gamma']] += -p['mu_v3'] * p['b'] / Kv
    g[_I['mu_v2']] += -1 / p['mu_v2']
    # -ln kh = -ln(psi_h - mu_h) + ln mu_h2
    Kh = q['Kh']
    g[_I['psi_h']] += -1 / Kh
    g[_I['mu_h']] += 1 / Kh
    g[_I['mu_h2']] += 1 / p['mu_h2']
    # -ln psi_v
    g[_I['psi_v']] += -1 / p['psi_v']
    # ln H
    dH_dbeta = -1 / p2 + (p2 + p3) / (p1 * p2)
    dH_dp1 = -beta * (p2 + p3) / (p1 ** 2 * p2)
    dH_dp2 = -(1 - beta) / p2 ** 2 - beta * p3 / (p1 * p2 ** 2)
    dH_dp3 = beta / (p1 * p2)
    g[_I['beta']] += dH_dbeta / H
    for n in ('r_s', 'delta_h', 'T_s', 'psi_h'):
        g[_I[n]] += dH_dp1 / H
    for n in ('r_r', 'delta_h', 'psi_h'):
        g[_I[n]] += dH_dp2 / H
    g[_I['T_s']] += dH_dp3 * (1 - p['e']) / H
    g[_I['e']] += dH_dp3 * (-p['T_s']) / H
    return R, R * g


def reffs(X, grad=False):
    """R_eff at each row of X (M, 20); with grad=True also the (M, 20) gradients."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if not grad:
        return np.array([reff(x) for x in X])
    out = [reff(x, grad=True) for x in X]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def ngm(x, alpha=0.3, alpha_s=0.0017, alpha_r=0.0017):
    """Next-generation matrix F V^{-1} with F and V of Eqs. (16)-(17), p. 10-11, in the order
    (v, w, x, y, g, h) of Eq. (15). alpha, alpha_s and alpha_r (Table 3 baselines by default) enter
    F V^{-1} but not its spectral radius."""
    q = _parts(x)
    p = q['p']
    xi1, k1, beta = q['xi1'], q['k1'], p['beta']
    F = np.zeros((6, 6))
    F[0, 4:] = beta * xi1
    F[1, 4:] = xi1 * (1 - beta)
    F[4, :2] = k1 * (1 - alpha)
    F[5, :2] = alpha * k1
    c1 = alpha_s + p['psi_h']
    c2 = alpha_r + p['psi_h']
    V = np.diag([q['p1'], q['p2'], c1, c2, p['psi_v'], p['psi_v']])
    V[1, 0] = -q['p3']
    V[2, 0] = -p['r_s']
    V[3, 1] = -p['r_r']
    return F @ np.linalg.inv(V)


def _tri_ppf(u, lo, c, hi):
    """Inverse CDF of the triangular distribution on [lo, hi] with mode c."""
    Fc = (c - lo) / (hi - lo)
    return np.where(u < Fc, lo + np.sqrt(u * (hi - lo) * (c - lo)),
                    hi - np.sqrt((1 - u) * (hi - lo) * (hi - c)))


def ppf(U):
    """Map an (M, 20) array of probabilities to the Table 3 distributions."""
    U = np.atleast_2d(U)
    X = LOWER + U * (UPPER - LOWER)
    for j, d in enumerate(DIST):
        if d == 'triangular':
            X[:, j] = _tri_ppf(U[:, j], LOWER[j], PEAK[j], UPPER[j])
    return X


def sample(M, rng, lhs=True):
    """M draws from the Table 3 distributions; Latin hypercube (one point per stratum per input,
    independent random permutations, a uniform point inside each stratum) if lhs, else Monte Carlo."""
    rng = np.random.default_rng(rng)
    if lhs:
        U = (np.argsort(rng.random((M, M_IN)), axis=0) + rng.random((M, M_IN))) / M
    else:
        U = rng.random((M, M_IN))
    return ppf(U)


def prcc(X, y):
    """Partial rank correlation coefficients of y with each column of X: Spearman ranks, the
    linear effect of the other ranked inputs (with an intercept) removed from both by least
    squares, then the Pearson correlation of the residuals."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    rank = lambda a: np.argsort(np.argsort(a, axis=0), axis=0).astype(float)  # noqa: E731
    RX, Ry = rank(X), rank(np.asarray(y, dtype=float))
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        Z = np.column_stack([np.ones(len(Ry)), np.delete(RX, j, axis=1)])
        ex = RX[:, j] - Z @ np.linalg.lstsq(Z, RX[:, j], rcond=None)[0]
        ey = Ry - Z @ np.linalg.lstsq(Z, Ry, rcond=None)[0]
        out[j] = ex @ ey / np.sqrt((ex @ ex) * (ey @ ey))
    return out
