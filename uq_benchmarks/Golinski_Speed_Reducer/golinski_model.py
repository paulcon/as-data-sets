"""Golinski's speed reducer (gearbox): volume and two shaft constraints, with exact gradients.

The volume is Golinski's own objective (J. Golinski, J. Mechanisms 5(3):287-309, 1970,
Sec. 3.1, p. 299, Fig. 11), written from the gear geometry with exact constants:

    gear k = 1, 2 (pinion z_1 = z teeth, wheel z_2 = i z teeth, shaft diameter d_k):
        d_st = m_t (z_k - 2.4)        root diameter of the rim
        d_w  = d_st - 4 m_t           inner diameter of the rim
        d_p  = d_k + 2 e_k = 2.4 d_k  hub diameter (e_k = 0.7 d_k)
        rim  (width b):     (pi b / 4)  (d_st^2 - d_w^2)
        web  (width b/3):   (pi b / 12) (d_w^2 - d_p^2)
        hub  (length 2 d_k): (pi / 4) (d_p^2 - d_k^2) 2 d_k
    shafts:  (pi / 4) (d_1^2 l_1 + d_2^2 l_2)

Two misprints in the printed formula are read from Fig. 11 (hub length 2 d_k, not "2b"; the
shaft term has d_k^2, not d_k). With z_2 = 3 z the sum expands to

    V = (pi/4) b m_t^2 (10/3 z^2 + 224/15 z - 646.4/15) - 0.48 pi b (d_1^2 + d_2^2)
        + 2.38 pi (d_1^3 + d_2^3) + (pi/4) (l_1 d_1^2 + l_2 d_2^2),

whose coefficients rounded to 4-5 digits (0.7854, 3.3333, 14.9334, 43.0934, 1.508, 7.477) are the
standard formulation of Ray (AIAA J. 41(3):556-558, 2003) and the NASA Langley MDO test suite.

The two constraint functions are those of Ray (2003) for shaft 1 (Golinski's conditions 7 and 9),
scaled so that they are satisfied when <= 0:

    g3 = 1.93 l_1^3 / (m_t z d_1^4) - 1                                  (shaft 1 deflection)
    g5 = sqrt((745 l_1 / (m_t z))^2 + 16.9e6) / (110 d_1^3) - 1           (shaft 1 stress)

Inputs are in cm (the paper's units); the volume is in cm^3; g3 and g5 are dimensionless.
The number of pinion teeth z is an integer and is held at 17; the transmission ratio i = 3.

Only numpy is imported. Written for as-data-sets from the papers' equations (MIT licence).
"""
import numpy as np

NAMES = ('b', 'm_t', 'l1', 'l2', 'd1', 'd2')
UNITS = ('cm', 'cm', 'cm', 'cm', 'cm', 'cm')
DESCRIPTIONS = ('face width', 'teeth module', 'shaft 1 length between bearings',
                'shaft 2 length between bearings', 'shaft 1 diameter', 'shaft 2 diameter')
# Study box: psdr build_golinski_design_domain (Hokanson), following Ray (2003)
LOWER = np.array([2.6, 0.7, 7.3, 7.3, 2.9, 5.0])
UPPER = np.array([3.6, 0.8, 8.3, 8.3, 3.9, 5.5])
NOMINAL = 0.5 * (LOWER + UPPER)

Z = 17          # number of pinion teeth (integer; held)
RATIO = 3       # transmission ratio i (paper p. 297)

QOI_NAMES = ('volume', 'g3', 'g5')
QOI_UNITS = ('cm^3', '1', '1')
QOI_DESCRIPTIONS = ('gearbox volume (Golinski objective, exact constants)',
                    'shaft 1 deflection constraint (Ray 2003), feasible if <= 0',
                    'shaft 1 stress constraint (Ray 2003), feasible if <= 0')


def _cols(X):
    X = np.asarray(X, dtype=float)
    one = X.ndim == 1
    X = np.atleast_2d(X)
    if X.shape[1] != 6:
        raise ValueError(f'expected 6 inputs (b, m_t, l1, l2, d1, d2), got {X.shape[1]}')
    if not np.all(np.isfinite(X)) or np.any(X <= 0):
        raise ValueError('every input must be finite and positive')
    return one, X.T


def _zpoly(z, ratio):
    """Sum over the two gears of rim + web coefficients of (pi/4) b m_t^2."""
    c = 0.0
    for zk in (z, ratio * z):
        dst, dw = zk - 2.4, zk - 6.4
        c += (dst**2 - dw**2) + dw**2 / 3
    return c


def volume_geometric(X, z=Z, ratio=RATIO):
    """Volume summed term by term from the gear geometry of Fig. 11 (an independent check)."""
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    V = 0.0
    for zk, d, l in ((z, d1, l1), (ratio * z, d2, l2)):
        dst = mt * zk - 2.4 * mt
        dw = dst - 4 * mt
        dp = d + 2 * 0.7 * d
        V = V + (dst**2 - dw**2) * np.pi * b / 4 + (dw**2 - dp**2) * np.pi * b / 12 \
            + (dp**2 - d**2) * np.pi / 4 * 2 * d + np.pi / 4 * d**2 * l
    return V[0] if one else V


def volume(X, z=Z, ratio=RATIO):
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    c = _zpoly(z, ratio)
    V = np.pi / 4 * b * mt**2 * c - 0.48 * np.pi * b * (d1**2 + d2**2) \
        + 2.38 * np.pi * (d1**3 + d2**3) + np.pi / 4 * (l1 * d1**2 + l2 * d2**2)
    return V[0] if one else V


def volume_grad(X, z=Z, ratio=RATIO):
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    c = _zpoly(z, ratio)
    G = np.stack([
        np.pi / 4 * mt**2 * c - 0.48 * np.pi * (d1**2 + d2**2),
        np.pi / 2 * b * mt * c,
        np.pi / 4 * d1**2,
        np.pi / 4 * d2**2,
        -0.96 * np.pi * b * d1 + 7.14 * np.pi * d1**2 + np.pi / 2 * l1 * d1,
        -0.96 * np.pi * b * d2 + 7.14 * np.pi * d2**2 + np.pi / 2 * l2 * d2,
    ], axis=1)
    return G[0] if one else G


def volume_rounded(X, z=Z):
    """The standard rounded-coefficient volume, with the coefficients printed in psdr (for comparison)."""
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    V = 0.7854 * b * mt**2 * (3.3333 * z**2 + 14.9334 * z - 43.0934) - 1.5079 * b * (d1**2 + d2**2) \
        + 7.477 * (d1**3 + d2**3) + 0.7854 * (l1 * d1**2 + l2 * d2**2)
    return V[0] if one else V


def g3(X, z=Z):
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    v = 1.93 * l1**3 / (mt * z * d1**4) - 1
    return v[0] if one else v


def g3_grad(X, z=Z):
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    r = 1.93 * l1**3 / (mt * z * d1**4)
    zero = np.zeros_like(b)
    G = np.stack([zero, -r / mt, 3 * r / l1, zero, -4 * r / d1, zero], axis=1)
    return G[0] if one else G


def g5(X, z=Z):
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    v = np.sqrt((745 * l1 / (mt * z))**2 + 16.9e6) / (110 * d1**3) - 1
    return v[0] if one else v


def g5_grad(X, z=Z):
    one, (b, mt, l1, l2, d1, d2) = _cols(X)
    t = 745 * l1 / (mt * z)
    s = np.sqrt(t**2 + 16.9e6)
    den = 110 * d1**3
    zero = np.zeros_like(b)
    ds_dt = t / s
    G = np.stack([zero, ds_dt * (-t / mt) / den, ds_dt * (t / l1) / den, zero,
                  -3 * s / (den * d1), zero], axis=1)
    return G[0] if one else G


def qoi(X, z=Z):
    """(N, 3) array (or (3,) for one point): volume (cm^3), g3, g5."""
    return np.stack([volume(X, z), g3(X, z), g5(X, z)], axis=-1)


def grad(X, z=Z):
    """(N, 3, 6) array (or (3, 6)): d(output i)/d(physical input j)."""
    return np.stack([volume_grad(X, z), g3_grad(X, z), g5_grad(X, z)], axis=-2)


def sample(n, seed, lower=LOWER, upper=UPPER):
    """n independent uniform samples on the box, numpy.random.default_rng(seed)."""
    rng = np.random.default_rng(seed)
    return lower + (upper - lower) * rng.random((n, len(lower)))
