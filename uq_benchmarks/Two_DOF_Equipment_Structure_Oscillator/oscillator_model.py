"""Two-degree-of-freedom equipment-structure oscillator (Igusa and Der Kiureghian 1985, Eq. 37).

A light secondary oscillator (the equipment: mass ms, stiffness ks, damping ratio zeta_s) is
mounted on a primary oscillator (the structure: mp, kp, zeta_p), and the base is shaken by
stationary white-noise acceleration with one-sided power spectral density S0 (the paper's
G0). Igusa and Der Kiureghian (J. Eng. Mech. 111(1):1-19, 1985, p. 12) derive by
perturbation theory a closed form for the mean-square response of the equipment, valid for
all values of the tuning parameter beta:

    E[x_s^2] = pi S0 / (4 zeta_s ws^3)
               * zeta_a zeta_s / (zeta_p zeta_s (4 zeta_a^2 + beta^2) + gamma zeta_a^2)
               * (zeta_p wp^3 + zeta_s ws^3) wp / (4 zeta_a wa^4),            (Eq. 37)

with wp = sqrt(kp/mp), ws = sqrt(ks/ms), wa = (wp + ws)/2 (Eq. 2d), beta = (wp - ws)/wa
(Eq. 2a), gamma = ms/mp (Eq. 2b) and zeta_a = (zeta_p + zeta_s)/2 (Eq. 3a). The common factor
zeta_a cancels, so E[x_s^2] = pi S0 N wp / (16 ws^3 wa^4 D) with N = zeta_p wp^3 +
zeta_s ws^3 and D = zeta_p zeta_s (4 zeta_a^2 + beta^2) + gamma zeta_a^2; this module uses
that form. x_s is the displacement of the equipment relative to the primary mass (the
deformation of the secondary spring); see `exact_mean_squares` and DATA.md.

Quantities of interest:
- `rms`: sqrt(E[x_s^2]), the root-mean-square deformation of the secondary spring
  (7 inputs: mp, ms, kp, ks, zeta_p, zeta_s, S0);
- `limit_state`: G = Fs - 3 ks rms, the reliability benchmark of Der Kiureghian and
  De Stefano (1991) as given by OpenTURNS (8 inputs, adds the spring force capacity Fs);
  failure is G <= 0.

Inputs (benchmark of Der Kiureghian and De Stefano 1991, via OpenTURNS / UQTestFuns): all
independent lognormal, given by mean and coefficient of variation; ln x ~ N(MU, SIGMA^2) with
SIGMA = sqrt(ln(1 + cov^2)) and MU = ln(mean) - SIGMA^2/2. Units are not given by the
benchmark; any consistent set (the nominal ws is 1 rad per unit time).

Gradients are exact (hand-coded forward-mode chain rule) with respect to the log inputs
z = ln x (`grad_log_rms`, `grad_log_limit_state`); divide by x for physical gradients.
`exact_mean_squares` solves the Lyapunov equation of the 2-DOF system of the paper's Eq. 1
(no perturbation approximation) for the accuracy check.

Only numpy and scipy are imported.
"""
import numpy as np

#: Input names, in column order. The first seven are the inputs of `rms`.
NAMES = ('mp', 'ms', 'kp', 'ks', 'zeta_p', 'zeta_s', 'S0', 'Fs')

#: Short descriptions, in column order.
DESCRIPTIONS = ('primary (structure) mass', 'secondary (equipment) mass',
                'primary spring stiffness', 'secondary spring stiffness',
                'primary damping ratio', 'secondary damping ratio',
                'white-noise base acceleration intensity (one-sided PSD, the paper\'s G0)',
                'force capacity of the secondary spring')

#: Lognormal means (OpenTURNS oscillator use case; Der Kiureghian and De Stefano 1991).
MEAN = np.array([1.5, 0.01, 1.0, 0.01, 0.05, 0.02, 100.0, 27.5])

#: Coefficients of variation (same source).
COV = np.array([0.1, 0.1, 0.2, 0.2, 0.4, 0.5, 0.1, 0.1])

#: Parameters of the log inputs: ln x ~ N(MU, SIGMA^2).
SIGMA = np.sqrt(np.log1p(COV**2))
MU = np.log(MEAN) - SIGMA**2/2

#: Number of inputs of `rms` (all but Fs).
M_RMS = 7


def _cols(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if np.any(~np.isfinite(x)) or np.any(x[:, :M_RMS] <= 0):
        raise ValueError('oscillator model needs finite, positive mp, ms, kp, ks, zeta_p, '
                         'zeta_s, S0')
    return x


def mean_square(x):
    """E[x_s^2] of Eq. 37 for each row of x (columns in NAMES order; Fs, if present, unused)."""
    x = _cols(x)
    mp, ms, kp, ks, zp, zs, S0 = x[:, :M_RMS].T
    wp, ws = np.sqrt(kp/mp), np.sqrt(ks/ms)
    wa, za, gam = (wp + ws)/2, (zp + zs)/2, ms/mp
    beta = (wp - ws)/wa
    N = zp*wp**3 + zs*ws**3
    D = zp*zs*(4*za**2 + beta**2) + gam*za**2
    return np.pi*S0*N*wp/(16*ws**3*wa**4*D)


def mean_square_as_printed(x):
    """Eq. 37 term by term as printed (zeta_a not cancelled); for checking `mean_square`."""
    x = _cols(x)
    mp, ms, kp, ks, zp, zs, S0 = x[:, :M_RMS].T
    wp, ws = np.sqrt(kp/mp), np.sqrt(ks/ms)
    wa, za, gam = (wp + ws)/2, (zp + zs)/2, ms/mp
    beta = (wp - ws)/wa
    return (np.pi*S0/(4*zs*ws**3) * za*zs/(zp*zs*(4*za**2 + beta**2) + gam*za**2)
            * (zp*wp**3 + zs*ws**3)*wp/(4*za*wa**4))


def rms(x):
    """Root-mean-square deformation of the secondary spring, sqrt(E[x_s^2])."""
    return np.sqrt(mean_square(x))


def limit_state(x):
    """G = Fs - 3 ks rms (failure if G <= 0); x has all eight columns."""
    x = _cols(x)
    return x[:, 7] - 3*x[:, 3]*rms(x)


def grad_log_mean_square(x):
    """d ln E[x_s^2] / d ln x_j, j = 0..6 (exact), shape (M, 7)."""
    x = _cols(x)
    M = len(x)
    mp, ms, kp, ks, zp, zs, S0 = x[:, :M_RMS].T
    E = np.eye(M_RMS)
    e = {n: np.broadcast_to(E[j], (M, M_RMS)) for j, n in enumerate(NAMES[:M_RMS])}
    wp, ws = np.sqrt(kp/mp), np.sqrt(ks/ms)
    wa, za, gam = (wp + ws)/2, (zp + zs)/2, ms/mp
    beta = (wp - ws)/wa
    c = lambda v: v[:, None]                                   # noqa: E731
    dlwp = (e['kp'] - e['mp'])/2                               # d ln wp
    dlws = (e['ks'] - e['ms'])/2                               # d ln ws
    dwp, dws = c(wp)*dlwp, c(ws)*dlws
    dwa = (dwp + dws)/2
    dzp, dzs = c(zp)*e['zeta_p'], c(zs)*e['zeta_s']
    dza = (dzp + dzs)/2
    dgam = c(gam)*(e['ms'] - e['mp'])
    dbeta = (dwp - dws)/c(wa) - c(beta)*dwa/c(wa)
    N = zp*wp**3 + zs*ws**3
    dN = dzp*c(wp**3) + 3*c(zp*wp**2)*dwp + dzs*c(ws**3) + 3*c(zs*ws**2)*dws
    D = zp*zs*(4*za**2 + beta**2) + gam*za**2
    dD = ((dzp*c(zs) + c(zp)*dzs)*c(4*za**2 + beta**2) + c(zp*zs)*(8*c(za)*dza + 2*c(beta)*dbeta)
          + dgam*c(za**2) + 2*c(gam*za)*dza)
    return e['S0'] + dN/c(N) + dlwp - 3*dlws - dD/c(D) - 4*dwa/c(wa)


def grad_log_rms(x):
    """d rms / d ln x_j, j = 0..6 (exact), shape (M, 7)."""
    return 0.5*rms(x)[:, None]*grad_log_mean_square(x)


def grad_log_limit_state(x):
    """dG / d ln x_j, j = 0..7 (exact), shape (M, 8); x has all eight columns."""
    x = _cols(x)
    r = rms(x)
    g = np.zeros((len(x), 8))
    g[:, :M_RMS] = -3*x[:, [3]]*grad_log_rms(x)
    g[:, 3] -= 3*x[:, 3]*r                                     # ks also multiplies rms
    g[:, 7] = x[:, 7]                                          # dFs/dln Fs
    return g


def exact_mean_squares(x):
    """Exact stationary mean squares of the 2-DOF system of the paper's Eq. 1.

    Solves A P + P A^T + pi S0 b b^T = 0 for the state covariance P of (x_p, x_s, x_p', x_s')
    (displacements relative to the ground, M x'' + C x' + K x = -M r a_g, r = [1, 1]^T),
    where white noise with one-sided PSD S0 has intensity pi S0. Returns, per row, the mean
    squares of x_s - x_p (deformation of the secondary spring), of x_s (equipment relative to
    the ground) and of x_p, shape (M, 3).
    """
    from scipy.linalg import solve_continuous_lyapunov
    x = _cols(x)
    out = np.empty((len(x), 3))
    for i, (mp, ms, kp, ks, zp, zs, S0) in enumerate(x[:, :M_RMS]):
        cp, cs = 2*zp*np.sqrt(kp*mp), 2*zs*np.sqrt(ks*ms)
        Minv = np.diag([1/mp, 1/ms])
        K = np.array([[kp + ks, -ks], [-ks, ks]])
        C = np.array([[cp + cs, -cs], [-cs, cs]])
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Minv @ K, -Minv @ C]])
        b = np.array([0., 0., -1., -1.])
        P = solve_continuous_lyapunov(A, -np.pi*S0*np.outer(b, b))
        out[i] = P[0, 0] + P[1, 1] - 2*P[0, 1], P[1, 1], P[0, 0]
    return out


def sample(n, seed):
    """n independent draws of the eight lognormal inputs, x = exp(MU + SIGMA xi)."""
    xi = np.random.default_rng(seed).standard_normal((n, len(NAMES)))
    return np.exp(MU + SIGMA*xi)
