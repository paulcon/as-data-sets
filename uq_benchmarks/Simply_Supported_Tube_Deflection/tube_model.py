"""Simply supported hollow tube under an off-centre point load: closed-form deflection and end slopes.

Euler-Bernoulli beam of length L on simple supports at x = 0 and x = L, loaded by a downward force F
at x = a, with b = L - a. Standard results from beam-deflection tables (e.g. Gere and Goodno,
Mechanics of Materials, 7th ed., 2009; Budynas and Nisbett, Shigley's Mechanical Engineering
Design, 9th ed., 2011), with v positive upward, v' = dv/dx:

    deflection at the load       y       = v(a)  = -F a^2 b^2 / (3 E L I)
    slope at the left support    theta_L = v'(0) = -F b (L^2 - b^2) / (6 E L I)
    slope at the right support   theta_R = v'(L) = +F a (L^2 - a^2) / (6 E L I)

with the second moment of area of a circular tube of outer diameter De and inner diameter di,
I = pi (De^4 - di^4) / 64. These are the three outputs of the OpenTURNS use case "Vertical
deflection of a tube" (class DeflectionTube), except that OpenTURNS writes I = pi (De^4 - di^4)/32,
which is the polar moment of the section, so its outputs are exactly half of these
(`openturns_convention=True` reproduces them). Written from the equations; no OpenTURNS code is used.

Inputs, in the order of the columns of X: F, L, a, De, di, E (consistent units; the benchmark gives
none). Input distributions are those of the OpenTURNS use-case page (independent):
F ~ N(1, 0.1), L ~ N(1.5, 0.01), a ~ U(0.7, 1.2), De ~ Tri(0.75, 0.8, 0.85), di ~ Tri(0.09, 0.1, 0.11),
E ~ N(200000, 2000).

Only numpy is imported (scipy.integrate inside `virtual_work`, used for checking).
"""
import numpy as np

NAMES = ('F', 'L', 'a', 'De', 'di', 'E')
DESCRIPTIONS = ('point force (downward)', 'tube length (span between supports)',
                'distance of the force from the left support', 'outer diameter of the tube',
                'inner diameter of the tube', "Young's modulus")
DISTRIBUTIONS = ('Normal(1, 0.1)', 'Normal(1.5, 0.01)', 'Uniform(0.7, 1.2)', 'Triangular(0.75, 0.8, 0.85)',
                 'Triangular(0.09, 0.1, 0.11)', 'Normal(200000, 2000)')
QOI_NAMES = ('deflection', 'theta_left', 'theta_right')

#: Means and standard deviations of the six distributions (exact).
MEAN = np.array([1.0, 1.5, 0.95, 0.8, 0.1, 2.0e5])
SD = np.array([0.1, 0.01, 0.5 / np.sqrt(12.0), 0.05 / np.sqrt(6.0), 0.01 / np.sqrt(6.0), 2000.0])


def sample(n, seed):
    """n independent Monte Carlo draws of the six inputs, shape (n, 6), from default_rng(seed).

    Columns are drawn one after another in the order F, L, a, De, di, E.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(1.0, 0.1, n)
    L = rng.normal(1.5, 0.01, n)
    a = rng.uniform(0.7, 1.2, n)
    De = rng.triangular(0.75, 0.8, 0.85, n)
    di = rng.triangular(0.09, 0.1, 0.11, n)
    E = rng.normal(2.0e5, 2000.0, n)
    return np.column_stack([F, L, a, De, di, E])


def _split(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] != 6:
        raise ValueError(f'expected 6 inputs (F, L, a, De, di, E), got {X.shape[1]}')
    F, L, a, De, di, E = X.T
    bad = ~(np.isfinite(X).all(axis=1) & (F > 0) & (L > 0) & (a > 0) & (a < L) & (di >= 0) & (De > di) & (E > 0))
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f'input row {i} outside the model domain (need F, L, E > 0, 0 < a < L, 0 <= di < De): '
                         f'{ {n: float(v) for n, v in zip(NAMES, X[i])} }')
    return F, L, a, De, di, E


def second_moment(De, di, openturns_convention=False):
    """Second moment of area of the tube section about a diameter, pi (De^4 - di^4)/64.

    With openturns_convention=True: pi (De^4 - di^4)/32 (the polar moment), as in OpenTURNS.
    """
    return np.pi * (De**4 - di**4) / (32.0 if openturns_convention else 64.0)


def outputs(X, openturns_convention=False):
    """The three QoIs, shape (n, 3): deflection v(a), slope v'(0), slope v'(L)."""
    F, L, a, De, di, E = _split(X)
    b = L - a
    EIL = E * second_moment(De, di, openturns_convention) * L
    y = -F * a**2 * b**2 / (3.0 * EIL)
    tL = -F * b * (L**2 - b**2) / (6.0 * EIL)
    tR = F * a * (L**2 - a**2) / (6.0 * EIL)
    return np.column_stack([y, tL, tR])


def gradients(X, openturns_convention=False):
    """Exact gradients with respect to the physical inputs, shape (n, 3, 6).

    Each QoI g is a product of powers, so dg/dx_j = g * d ln|g| / dx_j, written out by hand:
      ln|y|  = ln F + 2 ln a + 2 ln(L-a) - ln E - ln L - ln I + const
      ln|tL| = ln F + ln a + ln(L-a) + ln(2L-a) - ln E - ln L - ln I + const   [b(L^2-b^2) = a b (2L-a)]
      ln|tR| = ln F + ln a + ln(L-a) + ln(L+a) - ln E - ln L - ln I + const    [a(L^2-a^2) = a b (L+a)]
      d ln I/d De = 4 De^3/(De^4 - di^4),  d ln I/d di = -4 di^3/(De^4 - di^4)
    """
    F, L, a, De, di, E = _split(X)
    g = outputs(X, openturns_convention)
    b = L - a
    q = De**4 - di**4
    dlnI_dDe, dlnI_ddi = 4 * De**3 / q, -4 * di**3 / q
    common = {0: 1 / F, 3: -dlnI_dDe, 4: -dlnI_ddi, 5: -1 / E}
    dlog = np.zeros((len(F), 3, 6))
    for j, v in common.items():
        dlog[:, :, j] = v[:, None]
    # y
    dlog[:, 0, 1] = 2 / b - 1 / L
    dlog[:, 0, 2] = 2 / a - 2 / b
    # theta_L
    dlog[:, 1, 1] = 1 / b + 2 / (2 * L - a) - 1 / L
    dlog[:, 1, 2] = 1 / a - 1 / b - 1 / (2 * L - a)
    # theta_R
    dlog[:, 2, 1] = 1 / b + 1 / (L + a) - 1 / L
    dlog[:, 2, 2] = 1 / a - 1 / b + 1 / (L + a)
    return g[:, :, None] * dlog


def virtual_work(x, openturns_convention=False):
    """Independent check by the unit-load method (scipy.integrate.quad), one input point x (6,).

    v(a) = -int M(s) m_a(s) ds / EI with m_a the moment of a unit downward load at a;
    v'(0) = -int M(s) (1 - s/L) ds / EI and v'(L) = +int M(s) (s/L) ds / EI (unit end couples).
    M(s) = F b s / L for s <= a, F a (L - s)/L for s >= a (sagging positive).
    """
    from scipy.integrate import quad
    F, L, a, De, di, E = (float(v) for v in x)
    b = L - a
    EI = E * float(second_moment(De, di, openturns_convention))
    M = lambda s, P=F: P * b * s / L if s <= a else P * a * (L - s) / L
    m = lambda s: M(s, 1.0)
    kw = dict(points=[a], epsabs=0, epsrel=1e-13, limit=200)
    y = -quad(lambda s: M(s) * m(s), 0, L, **kw)[0] / EI
    tL = -quad(lambda s: M(s) * (1 - s / L), 0, L, **kw)[0] / EI
    tR = quad(lambda s: M(s) * s / L, 0, L, **kw)[0] / EI
    return np.array([y, tL, tR])
