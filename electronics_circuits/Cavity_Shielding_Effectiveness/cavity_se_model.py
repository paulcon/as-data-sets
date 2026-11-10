"""Matched power-balance bound on the shielding effectiveness of a slotted cylindrical cavity.

Python (numpy only) implementation of Appendix A (Eqs. (5)-(6), pp. 20-21) of

    S. Campione, J. A. Stephens, N. Martin, A. Eckert, L. K. Warne, G. Huerta, R. A. Pfeiffer,
    A. Jones, "Developing uncertainty quantification strategies in electromagnetic problems
    involving highly resonant cavities", J. Verif. Valid. Uncert. Quantif. 6(4):041003, 2021,
    doi:10.1115/1.4051906.

An azimuthal slot of length l, width w and depth d (the wall thickness) in an aluminium-alloy
cylinder of interior height h, interior radius a and wall conductivity sigma is driven by a plane
wave. Below the slot resonance the received power of the slot with a matched backing cavity is

    l |H0|^2 (8/pi^2) / (R_int pi^2/(omega L l)^2 + G_rad) = P_rec = P_wall = 2 A R_S <|H_j|^2>_A   (5)

and the bound on the shielding effectiveness is SE_max = 4 <|H_j|^2>_A / |H0|^2 (6). In dB,
SE = 10 log10(SE_max) (a power ratio, i.e. 20 log10 of the H-field ratio; the paper's Eq. (1) is
written for the E-field ratio and does not print the conversion of Eq. (6)). Symbols, as printed on
pp. 20-21:

    A = 2 pi a^2 + 2 pi a h,  R_S = sqrt(omega mu0 / (2 sigma)),
    L = mu0 pi / Omega_e,  Omega_e = Omega + C_e,  Omega = 2 ln(l / a_eq),  C_e = 2 (ln 2 - 7/3),
    a_eq = 2 w / (pi exp(1 + pi d / (2 w))),
    R_int = Re(Z_int),  Z_int = (L / L_int^r)^2 Z_int^r,  Z_int^r = 2 Z_S / d,  Z_S = (1 + i) R_S,
    L_int^r = mu0 w / d,
    pi eta0 l G_rad = -[1 + cos(kl)] + (kl/(2 pi) + pi/(2 kl)) [Cin(kl + pi) - Cin(pi - kl)]
                      + (kl/2) [1 - (pi/kl)^2] [Si(kl + pi) - Si(pi - kl)].

Because L and L_int^r are real, R_int = (L / L_int^r)^2 2 R_S / d, which is what is coded (no Re()).
The slot length in the model is the arc length l = 2 a asin(l_p / (2 a)) computed from the
value l_p of Table 1 read as the projected length, by the formula on p. 4 (this module's reading;
reading the Table 1 values as the arc length itself is equally plausible, see DATA.md).

Inputs (m = 6, Table 1 p. 33 and Table 3 p. 35), in the paper's units:
    d (in), l_p (in), w (mil), h (in), a (in), sigma (1e7 S/m).

Gradients are complex-step derivatives, d f/d x_j = Im f(x + i t e_j) / t with t = 1e-30 (in the
input's units): no subtractive cancellation, so they are accurate to rounding. Every operation in
the model is analytic in the inputs (no abs, no real part), which the complex step needs.

Si and Cin are entire functions, evaluated by their power series (complex-safe). The series are
used for |z| <= ZMAX = 12; the model raises ValueError beyond, which covers k l + pi up to 12,
i.e. frequencies up to about 4.8 GHz for the longest slot of Table 1.

Only numpy is imported.
"""
import numpy as np

# --- constants ------------------------------------------------------------------------------
MU0 = 4e-7 * np.pi                  # H/m (classical value)
C0 = 299792458.0                    # m/s
EPS0 = 1.0 / (MU0 * C0 ** 2)        # F/m
ETA0 = MU0 * C0                     # ohm, free-space impedance
INCH = 0.0254                       # m
MIL = 1e-3 * INCH                   # m

# --- inputs (Table 1, p. 33; Table 3, p. 35) ------------------------------------------------
NAMES = ('d', 'l_p', 'w', 'h', 'a', 'sigma')
UNITS = ('in', 'in', 'mil', 'in', 'in', '1e7 S/m')
DESCRIPTIONS = ('slot depth (wall thickness)', 'projected slot length', 'slot width',
                'cavity interior height', 'cavity interior radius', 'wall conductivity')
LOWER = np.array([0.2, 1.5, 5.0, 21.6, 3.6, 2.2])
UPPER = np.array([0.3, 2.5, 25.0, 26.4, 4.4, 3.0])
NOMINAL = 0.5 * (LOWER + UPPER)             # Table 3 means: 0.25, 2.0, 15, 24, 4, 2.6
NORMAL_SD = (UPPER - LOWER) / 6.0           # Table 3 standard deviations (range / 6, p. 13)
_SCALE = np.array([INCH, INCH, MIL, INCH, INCH, 1e7])   # to SI

# --- quantities of interest -----------------------------------------------------------------
FREQS_GHZ = (1.0, 2.0, 3.0)                 # fixed frequencies of the shipped outputs
BAND_GHZ = np.linspace(1.0, 3.0, 51)        # the paper's 51 frequencies (p. 8)
QOI_NAMES = ('SE_1GHz', 'SE_2GHz', 'SE_3GHz', 'SE_band')
QOI_DESCRIPTIONS = ('SE bound at 1 GHz (dB)', 'SE bound at 2 GHz (dB)', 'SE bound at 3 GHz (dB)',
                    'mean of the SE bound (dB) over the 51 frequencies 1, 1.04, ..., 3 GHz')

ZMAX = 12.0
_NTERMS = 45
H_CS = 1e-30                                # complex step


def _si(z):
    """Sine integral Si(z) = int_0^z sin(u)/u du by its power series (complex-safe)."""
    z = np.asarray(z)
    if np.any(np.abs(z) > ZMAX):
        raise ValueError(f'Si argument beyond {ZMAX}: the series is not used there')
    z2 = z * z
    term = z.astype(complex) if np.iscomplexobj(z) else z.astype(float)   # z^(2n+1)/(2n+1)!
    s = term.copy()
    for n in range(1, _NTERMS):
        term = -term * z2 / ((2 * n) * (2 * n + 1))
        s = s + term / (2 * n + 1)
    return s


def _cin(z):
    """Cin(z) = int_0^z (1 - cos u)/u du = -Ci(z) + ln z + gamma, by its power series (even in z)."""
    z = np.asarray(z)
    if np.any(np.abs(z) > ZMAX):
        raise ValueError(f'Cin argument beyond {ZMAX}: the series is not used there')
    z2 = z * z
    term = z2 / 2.0                         # z^(2n)/(2n)!, n = 1
    s = term / 2.0
    for n in range(2, _NTERMS):
        term = -term * z2 / ((2 * n - 1) * (2 * n))
        s = s + term / (2 * n)
    return s


def _check(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[-1] != 6:
        raise ValueError(f'expected 6 inputs (d, l_p, w, h, a, sigma), got shape {X.shape}')
    xr = X.real
    if not np.all(np.isfinite(xr)) or np.any(xr <= 0):
        raise ValueError('inputs must be finite and positive')
    lp, a = xr[:, 1], xr[:, 4]
    if np.any(lp >= 2 * a):
        raise ValueError('projected slot length l_p must be below the cavity diameter 2a')
    return X


def parts(X, f_hz):
    """Intermediate quantities, vectorized over rows of X (paper units) at frequency f_hz.

    Returns a dict of arrays: l (m, arc length), kl, a_eq (m), Omega_e, L (H/m), R_S (ohm),
    R_int (ohm/m), T_int = R_int pi^2/(omega L l)^2 and G_rad (both 1/(ohm m)), A (m^2),
    ratio = SE_max (dimensionless) and SE_dB.
    """
    X = _check(X)
    d, lp, w, h, a, sig = (X * _SCALE).T
    l = 2.0 * a * np.arcsin(lp / (2.0 * a))
    om = 2.0 * np.pi * f_hz
    k = om / C0
    kl = k * l
    A = 2.0 * np.pi * a ** 2 + 2.0 * np.pi * a * h
    Rs = np.sqrt(om * MU0 / (2.0 * sig))
    aeq = 2.0 * w / (np.pi * np.exp(1.0 + np.pi * d / (2.0 * w)))
    Ome = 2.0 * np.log(l / aeq) + 2.0 * (np.log(2.0) - 7.0 / 3.0)
    L = MU0 * np.pi / Ome
    Lr = MU0 * w / d
    Rint = (L / Lr) ** 2 * 2.0 * Rs / d
    T = Rint * np.pi ** 2 / (om * L * l) ** 2
    G = (-(1.0 + np.cos(kl))
         + (kl / (2 * np.pi) + np.pi / (2 * kl)) * (_cin(kl + np.pi) - _cin(np.pi - kl))
         + 0.5 * kl * (1.0 - (np.pi / kl) ** 2) * (_si(kl + np.pi) - _si(np.pi - kl))) / (np.pi * ETA0 * l)
    Prec = l * (8.0 / np.pi ** 2) / (T + G)        # per |H0|^2
    ratio = 4.0 * Prec / (2.0 * A * Rs)            # 4 <|H_j|^2>_A / |H0|^2
    return {'l': l, 'kl': kl, 'a_eq': aeq, 'Omega_e': Ome, 'L': L, 'R_S': Rs, 'R_int': Rint,
            'T_int': T, 'G_rad': G, 'A': A, 'ratio': ratio, 'SE_dB': 10.0 * np.log10(ratio)}


def se_db(X, f_hz, grad=False):
    """SE bound in dB at frequency f_hz (Hz) for inputs X, shape (6,) or (M, 6), paper units.

    Returns f of shape (M,), and with grad=True also the gradient, shape (M, 6), with respect
    to the inputs in their paper units (dB/in, dB/in, dB/mil, dB/in, dB/in, dB/(1e7 S/m)).
    """
    X = _check(np.asarray(X, dtype=float))
    f = parts(X, f_hz)['SE_dB']
    if not np.all(np.isfinite(f)):
        raise ValueError('non-finite SE')
    if not grad:
        return f
    G = np.empty(X.shape)
    for j in range(6):
        Xc = X.astype(complex)
        Xc[:, j] += 1j * H_CS
        G[:, j] = parts(Xc, f_hz)['SE_dB'].imag / H_CS
    return f, G


def band_mean(X, grad=False, freqs_ghz=BAND_GHZ):
    """Mean of the SE bound in dB over the frequencies freqs_ghz (default: the paper's 51)."""
    out = [se_db(X, fg * 1e9, grad=grad) for fg in freqs_ghz]
    if not grad:
        return np.mean(out, axis=0)
    return np.mean([o[0] for o in out], axis=0), np.mean([o[1] for o in out], axis=0)


def qois(X, grad=False):
    """The four shipped outputs (QOI_NAMES): shape (M, 4); with grad=True also (M, 4, 6)."""
    res = [se_db(X, fg * 1e9, grad=grad) for fg in FREQS_GHZ] + [band_mean(X, grad=grad)]
    if not grad:
        return np.column_stack(res)
    return np.column_stack([r[0] for r in res]), np.stack([r[1] for r in res], axis=1)


def kl_over_pi(X, f_hz):
    """k l / pi for each row: values above 1 are above the half-wave slot resonance."""
    return parts(np.asarray(X, dtype=float), f_hz)['kl'] / np.pi
