"""SEIR-SEI model of the 2016 Zika outbreak in Brazil (Dantas, Tosin and Cunha Jr, 2018).

Dantas E., Tosin M., Cunha Jr A., "Calibration of a SEIR-SEI epidemic model to
describe the Zika virus outbreak in Brazil", Appl. Math. Comput. 338:249-259
(2018), doi:10.1016/j.amc.2018.06.024, Eq. (1), Eq. (3) and Eq. (7).

Humans (S_h, E_h, I_h, R_h, in people) and vectors (S_v, E_v, I_v, proportions
of a constant vector population N_v = 1), plus the cumulative number C of
humans who have become infectious:

    dS_h/dt = -beta_h S_h I_v / N_v          dS_v/dt = delta N_v - beta_v S_v I_h / N_h - delta S_v
    dE_h/dt =  beta_h S_h I_v / N_v - alpha_h E_h
    dI_h/dt =  alpha_h E_h - gamma I_h        dE_v/dt = beta_v S_v I_h / N_h - (alpha_v + delta) E_v
    dR_h/dt =  gamma I_h                      dI_v/dt = alpha_v E_v - delta I_v
    dC/dt   =  alpha_h E_h

Time is in days. Epidemiological week (EW) w of 2016 ends at t = 7 w, the run
starts at t0 = 7 (EW 1) and ends at t = 364 (EW 52), and C(7) = C_i = 8201
(the paper's convention, Sec. 3.1). New cases in EW w are N_1 = C(7) and
N_w = C(7 w) - C(7 (w - 1)) (Eq. 3).

The ten inputs, in column order (`NAMES`), are the six rates and four initial
conditions; the two susceptible initial conditions are derived so that the
compartments add up (paper Sec. 3.1):
    S_hi = N_h - R_hi - E_hi - I_hi,   S_vi = N_v - E_vi - I_vi.
N_h = 206e6, R_hi = 29,639, C_i = 8,201 and N_v = 1 are held fixed, as in the
paper's calibration (Sec. 3.2).

Quantities of interest:
- `cumulative_cases`: C(364), people (the end point of the paper's Fig. 5a/6a curve);
- `misfit`: J = sum_w (y_w - N_w)^2 over the 52 weeks of `data_cases.csv`
  (Eq. 7), people^2.

Gradients come from forward sensitivity equations solved together with the
state (7 states + 7 x 10 sensitivities). Internally the human compartments are
scaled by N_h (fractions), so all 77 unknowns are O(1) or smaller.

Only numpy and scipy are imported.
"""
import os

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))

#: Input names, in column order.
NAMES = ('alpha_h', 'alpha_v', 'gamma', 'delta', 'beta_h', 'beta_v',
         'E_hi', 'I_hi', 'E_vi', 'I_vi')
UNITS = ('1/day',) * 6 + ('people', 'people', '-', '-')

#: Nominal values: the paper's "TRR input" column, Tables 1-2 (pp. 255-256 of the AMC version).
NOMINAL = np.array([1/5.9, 1/9.1, 1/7.9, 1/11, 1/11.3, 1/8.6, 8201., 8201., 2.2e-4, 2.2e-4])

#: Study ranges (uniform). Rates: lb/ub of Tables 1-2. E_hi, I_hi: Table 2 (0 to 10,000).
#: E_vi, I_vi: [0, 1e-3] each, chosen here (the paper's [0, 1] is a placeholder; see DATA.md).
LOWER = np.array([1/12, 1/10, 1/8.8, 1/21, 1/16.3, 1/11.6, 0., 0., 0., 0.])
UPPER = np.array([1/3, 1/5, 1/3, 1/11, 1/8, 1/6.2, 1e4, 1e4, 1e-3, 1e-3])

#: Held constants (paper Secs. 2.3 and 3.1).
N_H = 206e6       # Brazilian population, people
N_V = 1.0         # vector population (proportions)
R_HI = 29639.     # initial recovered humans (suspected 2015 cases), people
C_I = 8201.       # cumulative cases at t0 = EW 1, people

T0, T1 = 7., 364.                 # days: EW 1 and EW 52
WEEKS = np.arange(1, 53)          # EW 1..52
TIMES = 7.*WEEKS                  # days

RTOL = 1e-11
ATOL = 1e-15

_N_STATE = 7
_N_PAR = len(NAMES)


def load_cases(path=None):
    """Probable Zika cases per EW in Brazil, 2016 (52 values, people/week)."""
    path = path or os.path.join(HERE, 'data_cases.csv')
    d = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    return d[:, 1]


def _check(p, n_h=N_H, r_hi=R_HI, c_i=C_I):
    p = np.asarray(p, dtype=float)
    if not (n_h > 0 and 0 <= r_hi < n_h and 0 <= c_i <= n_h):
        raise ValueError('need N_h > 0, 0 <= R_hi < N_h and 0 <= C_i <= N_h')
    if p.shape != (_N_PAR,):
        raise ValueError(f'expected {_N_PAR} inputs {NAMES}, got shape {p.shape}')
    if not np.all(np.isfinite(p)):
        raise ValueError('non-finite input')
    if np.any(p[:6] <= 0):
        raise ValueError('rates must be > 0')
    if np.any(p[6:] < 0):
        raise ValueError('initial conditions must be >= 0')
    if p[6] + p[7] > n_h - r_hi:
        raise ValueError('E_hi + I_hi exceeds N_h - R_hi (S_hi would be negative)')
    if p[8] + p[9] > N_V:
        raise ValueError('E_vi + I_vi exceeds N_v = 1 (S_vi would be negative)')
    return p


def initial_state(p, n_h=N_H, r_hi=R_HI, c_i=C_I):
    """Scaled initial state (s_h, e_h, i_h, s_v, e_v, i_v, c); human entries are fractions of N_h."""
    E_hi, I_hi, E_vi, I_vi = p[6:]
    S_hi = n_h - r_hi - E_hi - I_hi
    S_vi = N_V - E_vi - I_vi
    return np.array([S_hi/n_h, E_hi/n_h, I_hi/n_h, S_vi, E_vi, I_vi, c_i/n_h])


def _rhs_state(y, r):
    a_h, a_v, g, d, b_h, b_v = r
    s_h, e_h, i_h, s_v, e_v, i_v, _ = y
    inf_h = b_h*s_h*i_v/N_V
    inf_v = b_v*s_v*i_h
    return np.array([-inf_h,
                     inf_h - a_h*e_h,
                     a_h*e_h - g*i_h,
                     d*N_V - inf_v - d*s_v,
                     inf_v - (a_v + d)*e_v,
                     a_v*e_v - d*i_v,
                     a_h*e_h])


def _jac_state(y, r):
    a_h, a_v, g, d, b_h, b_v = r
    s_h, e_h, i_h, s_v, e_v, i_v, _ = y
    A = np.zeros((7, 7))
    A[0, 0], A[0, 5] = -b_h*i_v, -b_h*s_h
    A[1, 0], A[1, 1], A[1, 5] = b_h*i_v, -a_h, b_h*s_h
    A[2, 1], A[2, 2] = a_h, -g
    A[3, 2], A[3, 3] = -b_v*s_v, -b_v*i_h - d
    A[4, 2], A[4, 3], A[4, 4] = b_v*s_v, b_v*i_h, -(a_v + d)
    A[5, 4], A[5, 5] = a_v, -d
    A[6, 1] = a_h
    return A


def _dfdr(y, r):
    """d f / d (alpha_h, alpha_v, gamma, delta, beta_h, beta_v), 7 x 6."""
    s_h, e_h, i_h, s_v, e_v, i_v, _ = y
    B = np.zeros((7, 6))
    B[1, 0], B[2, 0], B[6, 0] = -e_h, e_h, e_h
    B[4, 1], B[5, 1] = -e_v, e_v
    B[2, 2] = -i_h
    B[3, 3], B[4, 3], B[5, 3] = N_V - s_v, -e_v, -i_v
    B[0, 4], B[1, 4] = -s_h*i_v, s_h*i_v
    B[3, 5], B[4, 5] = -s_v*i_h, s_v*i_h
    return B


def _rhs_sens(t, z, r):
    y = z[:_N_STATE]
    S = z[_N_STATE:].reshape(_N_STATE, _N_PAR)
    dS = _jac_state(y, r) @ S
    dS[:, :6] += _dfdr(y, r)
    return np.concatenate([_rhs_state(y, r), dS.ravel()])


def _initial_sens():
    """d y(t0) / d (inputs), with E_hi and I_hi scaled by 1/N_h (so the columns are O(1))."""
    S0 = np.zeros((_N_STATE, _N_PAR))
    S0[0, 6], S0[1, 6] = -1., 1.      # E_hi (in units of N_h)
    S0[0, 7], S0[2, 7] = -1., 1.      # I_hi
    S0[3, 8], S0[4, 8] = -1., 1.      # E_vi
    S0[3, 9], S0[5, 9] = -1., 1.      # I_vi
    return S0


def trajectory(p, times=TIMES, rtol=RTOL, atol=ATOL, method='DOP853', n_h=N_H, r_hi=R_HI, c_i=C_I):
    """States in physical units at `times` (days, >= 7), shape (len(times), 8).

    Columns: S_h, E_h, I_h, R_h (people), S_v, E_v, I_v (proportions), C (people).
    The held constants N_h, R_hi and C_i can be changed with `n_h`, `r_hi`, `c_i`.
    """
    p = _check(p, n_h, r_hi, c_i)
    kw = {'jac': lambda t, y: _jac_state(y, p[:6])} if method in ('Radau', 'BDF', 'LSODA') else {}
    sol = solve_ivp(lambda t, y: _rhs_state(y, p[:6]), (T0, float(np.max(times))),
                    initial_state(p, n_h, r_hi, c_i), method=method, t_eval=times, rtol=rtol,
                    atol=atol, **kw)
    if not sol.success:
        raise RuntimeError(f'solve_ivp failed: {sol.message}')
    y = sol.y.T
    out = np.empty((len(times), 8))
    out[:, 0:3] = n_h*y[:, 0:3]
    out[:, 3] = n_h - out[:, 0:3].sum(axis=1)       # R_h from conservation
    out[:, 4:7] = y[:, 3:6]
    out[:, 7] = n_h*y[:, 6]
    if not np.all(np.isfinite(out)):
        raise RuntimeError('non-finite solution')
    return out


def cumulative_with_sens(p, times=TIMES, rtol=RTOL, atol=ATOL):
    """C(t) in people at `times`, and dC(t)/dp (physical inputs), shapes (n,), (n, 10)."""
    p = _check(p)
    z0 = np.concatenate([initial_state(p), _initial_sens().ravel()])
    sol = solve_ivp(_rhs_sens, (T0, float(np.max(times))), z0, method='DOP853', t_eval=times,
                    args=(p[:6],), rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'solve_ivp failed: {sol.message}')
    c = sol.y[6]
    S = sol.y[_N_STATE:].reshape(_N_STATE, _N_PAR, -1)[6].T   # dc/dp, (n, 10)
    C = N_H*c
    dC = N_H*S
    dC[:, 6:8] /= N_H                                          # E_hi, I_hi are in people
    if not (np.all(np.isfinite(C)) and np.all(np.isfinite(dC))):
        raise RuntimeError('non-finite solution')
    return C, dC


def new_cases(C):
    """Weekly new cases N_w from C at EW 1..52 (Eq. 3)."""
    return np.concatenate([C[..., :1], np.diff(C, axis=-1)], axis=-1)


def qoi_and_grad(p, y=None):
    """(C(364), dC(364)/dp, J, dJ/dp) at physical inputs p. y: weekly case data (default: data_cases.csv)."""
    y = load_cases() if y is None else y
    C, dC = cumulative_with_sens(p)
    Nw = new_cases(C)
    dNw = np.concatenate([dC[:1], np.diff(dC, axis=0)], axis=0)
    res = y - Nw
    J = float(res @ res)
    dJ = -2.*res @ dNw
    return float(C[-1]), dC[-1], J, dJ


def qoi(p, y=None, rtol=RTOL, atol=ATOL, n_h=N_H, r_hi=R_HI, c_i=C_I):
    """(C(364), J) without sensitivities."""
    y = load_cases() if y is None else y
    C = trajectory(p, rtol=rtol, atol=atol, n_h=n_h, r_hi=r_hi, c_i=c_i)[:, 7]
    res = y - new_cases(C)
    return float(C[-1]), float(res @ res)
