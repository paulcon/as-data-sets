"""Two-population competitive Lotka-Volterra tumour-spheroid model (Zou and Lewis 2026).

Equations (3.1)-(3.2) of H. Zou, A. L. Lewis, Math. Biosci. Eng. 23(7):2178-2207, 2026,
doi:10.3934/mbe.2026079 (p. 2186):

    dS/dt = r_S S (1 - S/K_S - gamma_R R/K_S),
    dR/dt = r_R R (1 - R/K_R - gamma_S S/K_R),

with S(0) = 0.018 mm^3 and R(0) = 0.002 mm^3 (total 0.02 mm^3, ratio 9:1, p. 2187).

Quantities of interest (p. 2187 and p. 2198):

    q_pri = trapezoidal rule of S(t) + R(t) on t = 0, 1, ..., 56 days     (Eq. 3.3, mm^3 day)
    q_sec = S(56) / (S(56) + R(56))                                         (Eq. 3.5, -)

Inputs x = (r_S, r_R, K_S, K_R, gamma_S, gamma_R), each on [0, 1] (p. 2187). Gradients are exact
derivatives of the discretised QoIs up to the ODE tolerance. The system is solved in the
logarithmic states u = ln S, v = ln R (u' = r_S (1 - (S + gamma_R R)/K_S), and similarly for v),
which keeps both populations positive and resolves them to a relative accuracy even when one of
them dies out. The forward sensitivities du/dp, dv/dp solve s' = J s + df/dp with the states
(2 + 12 equations); dS/dp = S du/dp, and the trapezoidal rule is applied on the same day grid.

K_S = 0 or K_R = 0 is singular (the right-hand side divides by K). `qoi` raises ValueError for
K <= 0; any K > 0 is accepted. For K much smaller than the initial volume the population first
collapses towards K on a time scale K/(r S0), which makes the system stiff; LSODA (scipy) switches
to its stiff method there.

Only numpy and scipy are imported.
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

NAMES = ['r_S', 'r_R', 'K_S', 'K_R', 'gamma_S', 'gamma_R']
UNITS = ['1/day', '1/day', 'mm^3', 'mm^3', '-', '-']
DESCRIPTIONS = ['growth rate of Type S cells', 'growth rate of Type R cells',
                'carrying capacity of Type S cells', 'carrying capacity of Type R cells',
                'competitive effect of Type S on Type R', 'competitive effect of Type R on Type S']
LOWER = np.zeros(6)
UPPER = np.ones(6)
S0, R0 = 0.018, 0.002          # mm^3 (0.02 mm^3 in the ratio 9:1)
T_END = 56.0                   # days
T_GRID = np.arange(0.0, 57.0)  # trapezoidal-rule nodes, 1-day bins
RTOL, ATOL = 1e-10, 1e-10   # on ln S, ln R and their sensitivities
QOI_NAMES = ['q_pri', 'q_sec']
QOI_UNITS = ['mm^3 day', '-']


def _check(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (6,):
        raise ValueError(f'x must have 6 entries (r_S, r_R, K_S, K_R, gamma_S, gamma_R), got shape {x.shape}')
    if not np.all(np.isfinite(x)):
        raise ValueError('x must be finite')
    if x[2] <= 0 or x[3] <= 0:
        raise ValueError('K_S and K_R must be > 0: the model divides by the carrying capacities')
    return x


def _rhs(t, z, p):
    """Right-hand side in logarithmic states u = ln S, v = ln R."""
    rS, rR, KS, KR, gS, gR = p
    S, R = np.exp(z[0]), np.exp(z[1])
    return [rS * (1 - (S + gR * R) / KS), rR * (1 - (R + gS * S) / KR)]


def _rhs_sens(t, z, p):
    """Log states and their 2 x 6 forward sensitivities, flattened as [u, v, du/dp (6), dv/dp (6)]."""
    rS, rR, KS, KR, gS, gR = p
    S, R = np.exp(z[0]), np.exp(z[1])
    su, sv = z[2:8], z[8:14]
    a = (S + gR * R) / KS
    b = (R + gS * S) / KR
    J11 = -rS * S / KS
    J12 = -rS * gR * R / KS
    J21 = -rR * gS * S / KR
    J22 = -rR * R / KR
    out = np.empty(14)
    out[0] = rS * (1 - a)
    out[1] = rR * (1 - b)
    out[2:8] = J11 * su + J12 * sv
    out[8:14] = J21 * su + J22 * sv
    out[2] += 1 - a                # d/dr_S
    out[4] += rS * a / KS          # d/dK_S
    out[7] += -rS * R / KS         # d/dgamma_R
    out[9] += 1 - b                # d/dr_R
    out[11] += rR * b / KR         # d/dK_R
    out[12] += -rR * S / KR        # d/dgamma_S
    return out


def _solve(fun, z0, x, rtol, atol, t=T_GRID):
    sol = solve_ivp(fun, (0.0, float(t[-1])), z0, method='LSODA', t_eval=t, args=(x,), rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != len(t):
        raise RuntimeError(f'ODE solve failed at x = {x.tolist()}: {sol.message}')
    return sol.y


def trajectory(x, t=None, rtol=RTOL, atol=ATOL):
    """S and R (mm^3) at the times t (increasing, starting at 0; default the days 0, 1, ..., 56),
    shape (len(t), 2)."""
    x = _check(x)
    t = T_GRID if t is None else np.asarray(t, dtype=float)
    if t[0] != 0:
        raise ValueError('t must start at 0')
    return np.exp(_solve(_rhs, [np.log(S0), np.log(R0)], x, rtol, atol, t)).T


def qoi(x, grad=False, rtol=RTOL, atol=ATOL):
    """(q_pri, q_sec) at x; with grad=True also the 2 x 6 gradient with respect to x.

    Raises ValueError for K_S <= 0 or K_R <= 0 and RuntimeError if the solver fails.
    """
    x = _check(x)
    if not grad:
        z = _solve(_rhs, [np.log(S0), np.log(R0)], x, rtol, atol)
    else:
        z0 = np.zeros(14)
        z0[0], z0[1] = np.log(S0), np.log(R0)
        z = _solve(_rhs_sens, z0, x, rtol, atol)
    S, R = np.exp(z[0]), np.exp(z[1])
    qsec = expit(z[0, -1] - z[1, -1])
    q = np.array([np.trapezoid(S + R, T_GRID), qsec])
    if not grad:
        return q
    su, sv = z[2:8], z[8:14]
    g = np.empty((2, 6))
    g[0] = np.trapezoid(S * su + R * sv, T_GRID, axis=1)
    g[1] = qsec * (1 - qsec) * (su[:, -1] - sv[:, -1])
    return q, g


def qois(X, grad=False, **kw):
    """qoi over the rows of X: (M, 2) values and, with grad=True, (M, 2, 6) gradients."""
    X = np.atleast_2d(X)
    if not grad:
        return np.array([qoi(x, **kw) for x in X])
    out = [qoi(x, grad=True, **kw) for x in X]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def equilibrium(x):
    """Index of the asymptotically stable equilibrium (Table 3, p. 2196) for gamma_S gamma_R < 1:
    0 = (0, 0), 1 = (0, K_R), 2 = (K_S, 0), 3 = coexistence.

    The competitive-exclusion state (K_S, 0) is stable when gamma_S K_S > K_R (the eigenvalue
    r_R (1 - gamma_S K_S / K_R) of the Jacobian is negative), (0, K_R) when gamma_R K_R > K_S, and
    the coexistence state when neither holds. Both cannot hold when gamma_S gamma_R < 1.
    Requires r_S, r_R > 0.
    """
    x = _check(x)
    rS, rR, KS, KR, gS, gR = x
    if gS * KS > KR and gR * KR > KS:
        raise ValueError('bistable case (gamma_S K_S > K_R and gamma_R K_R > K_S): not possible for gammas < 1')
    if gS * KS > KR:
        return 2
    if gR * KR > KS:
        return 1
    return 3


# The shipped samples. generate_data.py writes them as four files, each under 10 MB; together they
# hold the arrays listed in DATA.md. The full set is also archived as one file, lv_samples.npz, on
# Zenodo (doi:10.5281/zenodo.24174099).
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILES = {
    'lv_samples_grid4.npz': ('grid4_region', 'grid4_X', 'grid4_f',
                             'names', 'units', 'qoi_names', 'qoi_units', 'description'),
    'lv_samples_grid4_grad.npz': ('grid4_grad',),
    'lv_samples_sub8.npz': ('sub8_region', 'sub8_X', 'sub8_f'),
    'lv_samples_sub8_grad.npz': ('sub8_grad',),
}


def load_samples(directory=HERE):
    """All arrays of the shipped samples, as one dict (load with numpy, no pickle)."""
    D = {}
    for name in SAMPLE_FILES:
        with np.load(os.path.join(directory, name)) as z:
            D.update({k: z[k] for k in z.files})
    return D


def save_samples(data, directory):
    """Write the arrays of `data` to the files of SAMPLE_FILES in `directory`; return their paths."""
    keys = [k for ks in SAMPLE_FILES.values() for k in ks]
    assert sorted(keys) == sorted(data), 'data must hold exactly the arrays of SAMPLE_FILES'
    paths = []
    for name, ks in SAMPLE_FILES.items():
        paths.append(os.path.join(directory, name))
        np.savez_compressed(paths[-1], **{k: data[k] for k in ks})
    return paths
