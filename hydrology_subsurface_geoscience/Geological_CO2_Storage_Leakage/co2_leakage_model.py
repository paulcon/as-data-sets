"""One-dimensional Darcy leakage model of a geological CO2 storage site (Amrhein, Most, Walther 2026).

Reimplemented in numpy/scipy from S. K. T. Amrhein, T. Most, C. Walther, "An Uncertainty Quantification
Approach for Long-Term Risk Assessment of Geological CO2 Storage Sites Using a One-Dimensional Leakage
Model", Preprints.org (2026), doi:10.20944/preprints202609.2139.v1 (not peer reviewed): Eqs. (3)-(8) and
(11)-(14), Tables 3-4, Sec. 3.1-3.2, and the MATLAB listing of Table A1. No code was copied.

The site has n horizontal layers above the reservoir, numbered 1 (on top of the reservoir) to n (at the
ground surface). h_i is the depth of the lower edge of layer i, with h_1 > h_2 > ... > h_n > h_{n+1} = 0.
For a unit cross-section A = 1 m^2:

    L_i       = h_i - h_{i+1}                                   layer thickness       (Eq. 8)
    p_total,i = h_i * 0.01 MPa/m + p_pump * h_i / h_1           pressure at h_i       (Eqs. 11-12)
    dp_i      = p_total,i - p_total,i+1                         pressure drop         (Eq. 13)
    eta_i     = eta_L,i + (eta_U,i - eta_L,i) * u_eta           viscosity             (Eq. 14)
    V_i       = K_i A dp_i / (eta_i L_i)                        flow rate             (Eq. 3)
    t_i       = L_i A / V_i = eta_i L_i^2 / (K_i dp_i)          leakage time          (Eq. 5)
    V_min     = min_i V_i,  t_total = sum_i t_i                 outputs               (Eqs. 6-7)

with K_i = 10**logK_i in m^2, flows in m^3/year and times in years (365-day year, as in Table A1).
Because the pressure is linear in depth, p_total,i = c h_i with c = 1e4 Pa/m + 1e6 p_pump / h_1 (p_pump
in MPa), so dp_i = c L_i, and the thickness cancels from V_i = K_i c / eta_i, while t_i = eta_i L_i/(K_i c).

Inputs, in this order (m = 2n + 2): logK_1..logK_n (log10 m^2), h_1..h_n (m), p_pump (MPa), u_eta (1).
Distributions (Sec. 3.1, 3.2): logK_i symmetric triangular on the Table 3/4 bounds; h_i uniform on
[h_i - 5, h_i + 5] m; p_pump symmetric triangular, median +/- 4.5 MPa; u_eta uniform on [0, 1].

The minimum over layers in V_min makes it piecewise smooth (a kink where two layers tie). The smooth
replacement used for the active subspace is the p-norm soft minimum

    V_soft = (sum_i V_i**(-Q_SOFT))**(-1/Q_SOFT),   0 <= log10 V_min - log10 V_soft <= log10(n)/Q_SOFT.

Only numpy and the standard library are imported.
"""
import numpy as np

SEC_PER_YEAR = 3600.0 * 24.0 * 365.0     # Table A1
RHO_G = 1.0e4                             # hydrostatic gradient, Pa/m (0.01 MPa/m, Eq. 12)
H_SCATTER = 5.0                           # +/- m on each depth (Sec. 3.1, 3.2)
P_SCATTER = 4.5                           # +/- MPa on p_pump (Sec. 3.1, 3.2)
Q_SOFT = 10.0                             # exponent of the p-norm soft minimum (our choice)
LN10 = np.log(10.0)

# Table 3 (Ketzin) and Table 4 (Quest), layers listed from 1 (above the reservoir) to n (surface).
# logK bounds are log10 of the Table bounds in m^2; viscosity bounds are in Pa s (Table x 1e-3).
SITES = {
    'ketzin': dict(
        title='Ketzin pilot site, Germany (Table 3, Sec. 3.1)',
        depth=[629.0, 572.0, 557.0, 465.0, 381.0, 310.0, 264.0, 151.0, 55.0, 45.0],
        logK_lo=[-18.0, -14.0, -18.0, -15.0, -13.0, -14.0, -14.0, -17.0, -14.0, -11.0],
        logK_hi=[-16.0, -12.0, -16.0, -13.0, -11.0, -12.0, -12.0, -15.0, -12.0, -9.0],
        eta_lo=[0.014e-3] * 10,
        eta_hi=[0.070e-3] * 4 + [0.017e-3] * 6,
        p_pump=12.5,
        reservoir_depth=702.0,
        rock=['clay stone and clay marl stone', 'gypsum/anhydrite', 'clay stone',
              'clay marl stone, sandstone and clay stone', 'predominantly sandstone and siltstone',
              'clay stone, siltstone to sandstone', 'sandstone, siltstone and clay stone', 'clay', 'silt',
              'sand, gravel marl']),
    'quest': dict(
        title='Quest CCS project, Alberta, Canada (Table 4, Sec. 3.2)',
        depth=[2166.0, 2121.0, 1987.0, 1952.0, 1895.0, 1810.0, 1708.0, 1623.0, 1360.0, 1268.0, 1038.0,
               983.0, 683.0, 138.0],
        logK_lo=[-18.0, -14.0, -19.0, -14.0, -19.0, -14.0, -18.0] + [-10.0] * 7,
        logK_hi=[-16.0, -12.0, -17.0, -12.0, -17.0, -12.0, -16.0] + [-8.0] * 7,
        eta_lo=[0.040e-3] * 11 + [0.025e-3, 0.014e-3, 0.014e-3],
        eta_hi=[0.070e-3] * 13 + [0.017e-3],
        p_pump=8.34,
        reservoir_depth=2206.0,
        rock=['Middle Cambrian Shale (primary seal)', 'Basal Red Beds', 'Lower Lotsberg Salt (secondary seal)',
              'Devonian Mudstones', 'Upper Lotsberg Salt (ultimate seal)', 'Winnipegosis/Contact Rapids',
              'Prairie Evaporite (additional seal)'] + ['not available'] * 7),
}
for _s in SITES.values():
    for _k in ('depth', 'logK_lo', 'logK_hi', 'eta_lo', 'eta_hi'):
        _s[_k] = np.array(_s[_k], dtype=float)
    _s['n'] = len(_s['depth'])


def n_layers(site):
    return SITES[site]['n']


def names(site):
    n = n_layers(site)
    return ([f'logK{i}' for i in range(1, n + 1)] + [f'h{i}' for i in range(1, n + 1)]
            + ['p_pump', 'u_eta'])


def units(site):
    n = n_layers(site)
    return ['log10 m^2'] * n + ['m'] * n + ['MPa', '1']


def descriptions(site):
    s = SITES[site]
    n = s['n']
    return ([f'log10 permeability of layer {i} ({s["rock"][i - 1]})' for i in range(1, n + 1)]
            + [f'depth of the lower edge of layer {i}' for i in range(1, n + 1)]
            + ['injection overpressure at the reservoir (bottom of layer 1)',
               'normalized viscosity, one number for all layers (Eq. 14)'])


def distributions(site):
    n = n_layers(site)
    return ['symmetric triangular'] * n + ['uniform'] * n + ['symmetric triangular', 'uniform']


def bounds(site):
    """Lower and upper ends of the supports of the m inputs (physical units)."""
    s = SITES[site]
    lo = np.concatenate([s['logK_lo'], s['depth'] - H_SCATTER, [s['p_pump'] - P_SCATTER, 0.0]])
    hi = np.concatenate([s['logK_hi'], s['depth'] + H_SCATTER, [s['p_pump'] + P_SCATTER, 1.0]])
    return lo, hi


def nominal(site):
    """Medians of all inputs: the mean parameter set of Table A1 for Ketzin."""
    lo, hi = bounds(site)
    return 0.5 * (lo + hi)


def sample(M, rng, site):
    """M independent draws of the inputs by the inverse CDF of rng.random((M, m))."""
    lo, hi = bounds(site)
    n = n_layers(site)
    U = rng.random((M, 2 * n + 2))
    X = lo + (hi - lo) * U                                     # uniform inputs
    tri = np.r_[np.arange(n), 2 * n]                           # logK_i and p_pump
    u = U[:, tri]
    a, w = lo[tri], hi[tri] - lo[tri]
    X[:, tri] = np.where(u < 0.5, a + w * np.sqrt(u / 2.0), a + w - w * np.sqrt((1.0 - u) / 2.0))
    return X


def _split(X, site):
    s = SITES[site]
    n = s['n']
    X = np.asarray(X, dtype=float)
    one = X.ndim == 1
    X = np.atleast_2d(X)
    if X.shape[1] != 2 * n + 2:
        raise ValueError(f'{site}: expected {2 * n + 2} inputs, got {X.shape[1]}')
    if not np.all(np.isfinite(X)):
        raise ValueError('non-finite input')
    k, h, p, u = X[:, :n], X[:, n:2 * n], X[:, 2 * n], X[:, 2 * n + 1]
    L = h - np.column_stack([h[:, 1:], np.zeros(len(X))])
    if np.any(L <= 0):
        raise ValueError('layer depths must decrease strictly upward to the surface: '
                         'h_1 > h_2 > ... > h_n > 0')
    c = RHO_G + 1e6 * p / h[:, 0]                               # Pa/m
    if np.any(c <= 0):
        raise ValueError('total pressure gradient must be positive (p_pump > -0.01 MPa/m * h_1)')
    deta = s['eta_hi'] - s['eta_lo']
    eta = s['eta_lo'] + deta * u[:, None]
    if np.any(eta <= 0):
        raise ValueError('viscosity must be positive (u_eta too small)')
    return one, k, h, p, u, L, c, eta, deta


def layers(X, site):
    """Layer flow rates V_i (m^3/year, A = 1 m^2) and leakage times t_i (years), each (M, n).

    Computed exactly as Table A1 does (pressures at the layer edges, then differences), so that
    the thickness cancellation is numerical, not assumed."""
    one, k, h, p, u, L, c, eta, _ = _split(X, site)
    hz = np.column_stack([h, np.zeros(len(h))])
    ptot = hz * RHO_G + hz / h[:, :1] * p[:, None] * 1e6       # Eqs. (11)-(12), Pa
    dp = ptot[:, :-1] - ptot[:, 1:]                            # Eq. (13)
    V = 10.0 ** k * dp / eta / L * SEC_PER_YEAR                # Eq. (3), A = 1 m^2
    t = L / V                                                  # Eq. (5)
    return (V[0], t[0]) if one else (V, t)


def leakage_time(X, site):
    """Total leakage time t_total = sum_i t_i in years (Eq. 7)."""
    _, t = layers(X, site)
    return t.sum(axis=-1)


def min_flow(X, site):
    """Minimum layer flow rate V_min = min_i V_i in m^3/year (Eq. 6)."""
    V, _ = layers(X, site)
    return V.min(axis=-1)


def soft_min_flow(X, site, q=Q_SOFT):
    """p-norm soft minimum (sum_i V_i^-q)^(-1/q) in m^3/year; smooth, and V_min * n^(-1/q) <= it <= V_min."""
    V, _ = layers(X, site)
    lv = np.log(V)
    lmin = lv.min(axis=-1, keepdims=True)
    s = np.exp(-q * (lv - lmin)).sum(axis=-1)
    return np.exp(lmin[..., 0] - np.log(s) / q)


def _dlogc(h1, p, c):
    """d ln c / d p_pump and d ln c / d h_1."""
    return (1e6 / h1) / c, -(1e6 * p / h1 ** 2) / c


def grad_log10_leakage_time(X, site):
    """Exact gradient of log10 t_total with respect to the physical inputs, (M, m)."""
    one, k, h, p, u, L, c, eta, deta = _split(X, site)
    n = k.shape[1]
    t = eta * L / (10.0 ** k * c[:, None]) / SEC_PER_YEAR      # = t_i of layers()
    T = t.sum(axis=1)
    dlc_dp, dlc_dh1 = _dlogc(h[:, 0], p, c)
    G = np.empty((len(T), 2 * n + 2))
    G[:, :n] = -LN10 * t                                       # dT/dlogK_i
    dT_dL = t / L
    G[:, n:2 * n] = dT_dL                                      # L_j = h_j - h_{j+1}: +dT/dL_j ...
    G[:, n + 1:2 * n] -= dT_dL[:, :-1]                         # ... and -dT/dL_{j-1}
    G[:, n] += -T * dlc_dh1                                    # c depends on h_1
    G[:, 2 * n] = -T * dlc_dp
    G[:, 2 * n + 1] = (t * deta / eta).sum(axis=1)
    G /= (T * LN10)[:, None]
    return G[0] if one else G


def _grad_log10_flow_weighted(X, site, w):
    """Gradient of sum_i w_i log10 V_i (weights w, rows summing to 1) w.r.t. the physical inputs."""
    one, k, h, p, u, L, c, eta, deta = _split(X, site)
    n = k.shape[1]
    dlc_dp, dlc_dh1 = _dlogc(h[:, 0], p, c)
    G = np.zeros((len(k), 2 * n + 2))
    G[:, :n] = w
    G[:, n] = dlc_dh1 / LN10
    G[:, 2 * n] = dlc_dp / LN10
    G[:, 2 * n + 1] = -(w * deta / eta).sum(axis=1) / LN10
    return G[0] if one else G


def grad_log10_soft_min_flow(X, site, q=Q_SOFT):
    """Exact gradient of log10 V_soft with respect to the physical inputs, (M, m)."""
    V, _ = layers(np.atleast_2d(X), site)
    lv = np.log(V)
    w = np.exp(-q * (lv - lv.min(axis=1, keepdims=True)))
    w /= w.sum(axis=1, keepdims=True)
    return _grad_log10_flow_weighted(X, site, w)


def grad_log10_min_flow(X, site):
    """Gradient of log10 V_min where it exists (one layer attains the minimum), (M, m)."""
    V, _ = layers(np.atleast_2d(X), site)
    w = np.zeros_like(V)
    w[np.arange(len(V)), V.argmin(axis=1)] = 1.0
    return _grad_log10_flow_weighted(X, site, w)


def argmin_layer(X, site):
    """Layer number (1..n) that attains V_min."""
    V, _ = layers(np.atleast_2d(X), site)
    return V.argmin(axis=1) + 1
