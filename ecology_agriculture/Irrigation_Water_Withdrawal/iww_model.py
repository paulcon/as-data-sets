"""Irrigation water withdrawal (IWW) for wheat in one grid cell of Uvalde County, Texas (Puy et al. 2022).

The model is Eq. 16 of the Supplementary Materials of Puy et al., "The delusive accuracy of global
irrigation water withdrawal estimates", Nature Communications 13:3183 (2022),
doi:10.1038/s41467-022-30731-8; Supplementary Materials, Zenodo doi:10.5281/zenodo.5528844, v1.0.4
(section 2.5, pp. 17-19; inputs in Table S4, p. 18). The function `iww` evaluates it as the authors'
archived code evaluates it (function `full_model` in `code_ghm_auditing.R` in the same Zenodo record),
which produced the published results. That form differs from Eq. 16 as printed in two ways:

1. The reference evapotranspiration is computed as

       E = (0.408 Delta A + gamma 900/(T_a + 273) w v) / Delta + gamma (1 + 0.34 w),

   i.e. the numerator of the FAO-56 Penman-Monteith formula (Eq. 4, p. 5) is divided by Delta
   alone and gamma (1 + 0.34 w) is added, where Eq. 4 divides by Delta + gamma (1 + 0.34 w).
2. E is divided by 31 ("To get a daily value", in the code), which no printed equation does.

With I_a in ha, ET_0 and P in mm, and 1 mm over 1 ha = 10 m^3, the withdrawal for one day is

    y = 10 I_a (k_c E / 31 - P) / (E_a E_c M_f)   [m^3].

`iww(X, form='printed')` evaluates Eq. 16 with the Penman-Monteith denominator of Eq. 4 and without
the division by 31, for comparison only; `form='printed31'` adds the division by 31 to the printed
denominator. In both, P is subtracted inside the I_a factor, as in Eqs. 1 and 16 and the code.
Exact gradients are given for the archived form only.

This module was written from the equations; no code from the authors' archive was copied.
Only numpy is imported.
"""
import numpy as np

# Table S4 (Supplementary Materials, p. 18), in its order. Units from Eq. 4 (p. 5), Eq. 1 (p. 3) and the
# archived code's comments (I_a in ha, P and ET_0 in mm).
NAMES = ('Delta', 'gamma', 'A', 'T_a', 'w', 'v', 'k_c', 'I_a', 'E_a', 'E_c', 'M_f', 'P')
SYMBOLS = (r'$\Delta$', r'$\gamma$', r'$A$', r'$T_a$', r'$w$', r'$v$', r'$k_c$', r'$I_a$', r'$E_a$',
           r'$E_c$', r'$M_f$', r'$P$')
UNITS = ('kPa/degC', 'kPa/degC', 'MJ/m^2', 'degC', 'm/s', 'kPa', '-', 'ha', '-', '-', '-', 'mm')
DESCRIPTIONS = (
    'slope of the saturation vapour pressure curve ("vapour pressure" in Table S4)',
    'psychrometric constant',
    'net radiation minus soil heat flux',
    'mean air temperature',
    'wind speed',
    'vapour pressure deficit',
    'crop coefficient (wheat)',
    'irrigated area',
    'field application efficiency',
    'conveyance efficiency',
    'management factor',
    'precipitation',
)
LOWER = np.array([0.0796, 0.065, 297.550, 9.9, 2.67, 0.26, 0.45, 42.9, 0.49, 0.64, 0.50, 0.0])
UPPER = np.array([0.0804, 0.066, 402.448, 10.1, 2.95, 0.29, 1.14, 144.5, 0.88, 0.96, 0.97, 0.1])
NOMINAL = 0.5 * (LOWER + UPPER)   # the mean of every input: the point estimate of Fig. S13a (p. 21)
M = len(NAMES)

DAYS = 31.0          # the archived code's division "to get a daily value"
MM_HA_TO_M3 = 10.0   # 1 mm of water over 1 ha is 10 m^3
FORMS = ('archive', 'printed', 'printed31')


def _columns(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[-1] != M:
        raise ValueError(f'X must have {M} columns in the order of NAMES, got shape {X.shape}')
    return X, X.T


def et0(X, form='archive'):
    """Reference evapotranspiration for one day, in mm: E/31 (archive), Eq. 4 (printed) or Eq. 4 / 31 (printed31)."""
    if form not in FORMS:
        raise ValueError(f'form must be one of {FORMS}')
    _, (D, g, A, Ta, w, v, *_rest) = _columns(X)
    num = 0.408 * D * A + g * (900.0 / (Ta + 273.0)) * w * v
    if form == 'archive':
        return (num / D + g * (1.0 + 0.34 * w)) / DAYS
    E = num / (D + g * (1.0 + 0.34 * w))
    return E / DAYS if form == 'printed31' else E


def iww(X, form='archive'):
    """Irrigation water withdrawal y in m^3 for one day, shape (M,)."""
    X, cols = _columns(X)
    kc, Ia, Ea, Ec, Mf, P = cols[6:]
    return MM_HA_TO_M3 * Ia * (kc * et0(X, form) - P) / (Ea * Ec * Mf)


def grad_iww(X):
    """Exact gradient of iww(X, 'archive') with respect to the physical inputs, shape (M, 12)."""
    X, (D, g, A, Ta, w, v, kc, Ia, Ea, Ec, Mf, P) = _columns(X)
    c = 900.0 / (Ta + 273.0)
    ET = et0(X, 'archive')
    Ep = Ea * Ec * Mf
    y = MM_HA_TO_M3 * Ia * (kc * ET - P) / Ep
    dy_dET = MM_HA_TO_M3 * Ia * kc / Ep / DAYS        # derivative with respect to E (before / 31)
    G = np.empty_like(X)
    G[:, 0] = dy_dET * (-g * c * w * v / D**2)                   # d(num/D)/dD = -gamma c w v / D^2
    G[:, 1] = dy_dET * (c * w * v / D + 1.0 + 0.34 * w)
    G[:, 2] = dy_dET * 0.408
    G[:, 3] = dy_dET * (-g * w * v * 900.0 / (Ta + 273.0)**2 / D)
    G[:, 4] = dy_dET * (g * c * v / D + 0.34 * g)
    G[:, 5] = dy_dET * (g * c * w / D)
    G[:, 6] = MM_HA_TO_M3 * Ia * ET / Ep
    G[:, 7] = y / Ia
    G[:, 8] = -y / Ea
    G[:, 9] = -y / Ec
    G[:, 10] = -y / Mf
    G[:, 11] = -MM_HA_TO_M3 * Ia / Ep
    return G


def sample(n, rng):
    """n independent uniform draws from Table S4, shape (n, 12), by LOWER + (UPPER - LOWER) * rng.random((n, 12))."""
    return LOWER + (UPPER - LOWER) * rng.random((n, M))
