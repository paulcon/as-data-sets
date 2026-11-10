"""Steel column limit state of Kuschel & Rackwitz (1997), Sec. 6, second example (pp. 326-327).

    G(x, p) = Fs - P (1/As + F0/Ms * Eb/(Eb - P))                       (MPa)

with P = P1 + P2 + P3, As = 2 B D, Ms = B D H, Mi = B D H^2 / 2 and Eb = pi^2 E Mi / s^2, s = 7500 mm
(the column length). Nine independent random inputs X = (Fs, P1, P2, P3, B, D, H, F0, E) (table,
p. 326); the design parameters p = (b, d, h) are the means of B, D and H. G <= 0 is failure.

Inputs are scaled through their CDFs: u_i = Phi^{-1}(F_i(x_i)), so u ~ N(0, I_9). The functions
here give the limit state and its exact gradient with respect to the physical inputs, the maps
x(u) and u(x), and dx/du (the Jacobian is diagonal because the inputs are independent).

Two configurations are defined in CONFIGS (see DATA.md):
  'benchmark' : (b, d, h) = (300, 20, 300) mm, mean of E = 210000 MPa (the design and E used by the
                steel-column benchmark page of Surjanovic & Bingham, which cites Eldred, Webster &
                Constantine 2008 and Kuschel & Rackwitz 1997);
  'paper'     : (b, d, h) = (200, 17.5, 100) mm, mean of E = 21000 MPa, the paper's optimum for
                maximal cost 4000 (table, p. 327) with E as printed in the table on p. 326.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy import optimize, special

L_COLUMN = 7500.0  # mm, "constant length of 7500 mm" (p. 326)
NAMES = ('Fs', 'P1', 'P2', 'P3', 'B', 'D', 'H', 'F0', 'E')
UNITS = ('MPa', 'N', 'N', 'N', 'mm', 'mm', 'mm', 'mm', 'MPa')
DESCRIPTIONS = ('yield stress', 'dead weight load', 'variable load', 'variable load',
                'flange breadth', 'flange thickness', 'height of profile', 'initial deflection',
                "Young's modulus")
DISTRIBUTIONS = ('lognormal', 'normal', 'gumbel', 'gumbel', 'lognormal', 'lognormal', 'lognormal',
                 'normal', 'weibull')
SD = np.array([35.0, 50000.0, 90000.0, 90000.0, 3.0, 2.0, 5.0, 10.0, 4200.0])
CONFIGS = {
    'benchmark': {'b': 300.0, 'd': 20.0, 'h': 300.0, 'E_mean': 210000.0},
    'paper': {'b': 200.0, 'd': 17.5, 'h': 100.0, 'E_mean': 21000.0},
}
_C = np.pi ** 2 / (2.0 * L_COLUMN ** 2)  # Eb = _C * E * B * D * H^2


def means(config):
    c = CONFIGS[config] if isinstance(config, str) else config
    return np.array([400.0, 500000.0, 600000.0, 600000.0, c['b'], c['d'], c['h'], 30.0, c['E_mean']])


def _weibull_params(mean, sd):
    """Shape k and scale lam of the two-parameter Weibull with the given mean and sd."""
    cv = sd / mean

    def r(k):
        return np.sqrt(np.exp(special.gammaln(1 + 2 / k) - 2 * special.gammaln(1 + 1 / k)) - 1) - cv
    k = optimize.brentq(r, 0.1, 1e4, xtol=1e-14, rtol=1e-15)
    return k, mean / special.gamma(1 + 1 / k)


class Marginals:
    """The nine independent marginals of one configuration, mean/sd as in the table on p. 326."""

    def __init__(self, config='benchmark'):
        self.mean = means(config)
        self.sd = SD.copy()
        m, s = self.mean, self.sd
        self.ln_s = {}
        self.ln_mu = {}
        for i in (0, 4, 5, 6):
            s2 = np.log1p((s[i] / m[i]) ** 2)
            self.ln_s[i] = np.sqrt(s2)
            self.ln_mu[i] = np.log(m[i]) - s2 / 2
        self.gum_scale = s[2] * np.sqrt(6.0) / np.pi
        self.gum_loc = m[2] - np.euler_gamma * self.gum_scale
        self.wb_k, self.wb_lam = _weibull_params(m[8], s[8])

    def x_from_u(self, U):
        U = np.atleast_2d(np.asarray(U, float))
        X = np.empty_like(U)
        for i in (0, 4, 5, 6):
            X[:, i] = np.exp(self.ln_mu[i] + self.ln_s[i] * U[:, i])
        for i in (1, 7):
            X[:, i] = self.mean[i] + self.sd[i] * U[:, i]
        for i in (2, 3):
            X[:, i] = self.gum_loc - self.gum_scale * np.log(-special.log_ndtr(U[:, i]))
        X[:, 8] = self.wb_lam * (-special.log_ndtr(-U[:, 8])) ** (1 / self.wb_k)
        return X

    def u_from_x(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        U = np.empty_like(X)
        for i in (0, 4, 5, 6):
            U[:, i] = (np.log(X[:, i]) - self.ln_mu[i]) / self.ln_s[i]
        for i in (1, 7):
            U[:, i] = (X[:, i] - self.mean[i]) / self.sd[i]
        for i in (2, 3):
            z = np.exp(-(X[:, i] - self.gum_loc) / self.gum_scale)  # -ln F(x)
            # F = exp(-z); use the upper tail 1 - F = -expm1(-z) where F > 1/2
            U[:, i] = np.where(z < np.log(2), -special.ndtri(-np.expm1(-z)), special.ndtri(np.exp(-z)))
        t = (X[:, 8] / self.wb_lam) ** self.wb_k  # -ln(1 - F)
        U[:, 8] = np.where(t < np.log(2), special.ndtri(-np.expm1(-t)), -special.ndtri(np.exp(-t)))
        return U

    def dx_du(self, U):
        """Diagonal of the Jacobian dx/du at U, shape (M, 9)."""
        U = np.atleast_2d(np.asarray(U, float))
        X = self.x_from_u(U)
        J = np.empty_like(U)
        for i in (0, 4, 5, 6):
            J[:, i] = self.ln_s[i] * X[:, i]
        for i in (1, 7):
            J[:, i] = self.sd[i]
        logphi = -0.5 * U ** 2 - 0.5 * np.log(2 * np.pi)
        for i in (2, 3):
            lF = special.log_ndtr(U[:, i])
            J[:, i] = self.gum_scale * np.exp(logphi[:, i] - lF) / (-lF)
        lS = special.log_ndtr(-U[:, 8])
        t = -lS
        J[:, 8] = self.wb_lam / self.wb_k * t ** (1 / self.wb_k - 1) * np.exp(logphi[:, 8] - lS)
        return J

    def sample(self, M, rng):
        """M Monte Carlo draws: U ~ N(0, I_9) from `rng`, returns (U, X)."""
        U = rng.standard_normal((M, 9))
        return U, self.x_from_u(U)


def euler_load(X):
    """Eb = pi^2 E Mi / s^2 (N) with Mi = B D H^2 / 2."""
    X = np.atleast_2d(np.asarray(X, float))
    return _C * X[:, 8] * X[:, 4] * X[:, 5] * X[:, 6] ** 2


def limit_state(X, grad=False):
    """G (MPa) at the rows of X (physical units), and optionally dG/dx, shape (M, 9).

    Complex X is accepted (for complex-step checks). Raises ValueError if Eb = P at some row (the
    pole of the amplification factor)."""
    X = np.atleast_2d(np.asarray(X))
    if not np.iscomplexobj(X):
        X = X.astype(float)
    Fs, P1, P2, P3, B, D, H, F0, E = X.T
    P = P1 + P2 + P3
    A = 2 * B * D
    Ms = B * D * H
    Eb = _C * E * B * D * H ** 2
    den = Eb - P
    if np.any(den == 0):
        raise ValueError('Euler load equals the axial load (Eb = P): G is undefined')
    R = Eb / den
    T = P * F0 * R / Ms
    G = Fs - P / A - T
    if not grad:
        return G
    K = P * F0 / Ms * P * Eb / den ** 2      # (P F0/Ms) * (-dR/dEb) * Eb
    dG = np.empty_like(X)
    dG[:, 0] = 1.0
    dPi = -1.0 / A - F0 / Ms * (R + P * Eb / den ** 2)  # R + P dR/dP, dR/dP = Eb/den^2
    dG[:, 1] = dG[:, 2] = dG[:, 3] = dPi
    dG[:, 4] = (P / A + T + K) / B
    dG[:, 5] = (P / A + T + K) / D
    dG[:, 6] = (T + 2 * K) / H
    dG[:, 7] = -P * R / Ms
    dG[:, 8] = K / E
    return G, dG


def limit_state_u(U, config='benchmark', grad=False):
    """G and dG/du at the rows of U (standard normal coordinates) for one configuration."""
    mg = Marginals(config)
    X = mg.x_from_u(U)
    if not grad:
        return limit_state(X)
    G, dG = limit_state(X, grad=True)
    return G, dG * mg.dx_du(U)
