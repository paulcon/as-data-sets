"""Long-term HIV model of Loudon and Pankavich (Math. Biosci. Eng. 14(3), 2017, Eq. 1).

Seven ODEs for the CD4+ T cells T, actively infected T cells T_I, latently
infected T cells T_L, macrophages M, infected macrophages M_I, cytotoxic
lymphocytes CTL and virions V, with 27 parameters (Table 1 of the paper).

`Tcells(p, times)` returns T(t) at the given times for each row of physical
parameters `p`. Settings, kept from the 2016 code that produced the shipped data:

- Initial conditions are the same for every sample, by design: the uninfected
  equilibrium at the nominal parameters plus one virion,
  (T, T_I, T_L, M, M_I, CTL, V)(0) = (1000, 0, 0, 0.15/0.005, 0, 5/0.015, 1).
- T is floored at 0.1: once T falls to 0.1 the state is frozen, and T stays 0.1
  for the rest of the run. The paper does not have this floor. Here it is an
  event (T = 0.1, terminal) of `solve_ivp`; the 2016 code returned a zero
  right-hand side below 0.1, which has the same frozen values.
- Solver: `scipy.integrate.solve_ivp`, LSODA, rtol = atol = 1e-10, with the
  analytic Jacobian.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy.integrate import solve_ivp

#: Parameter names, in column order (Table 1 of the paper).
NAMES = ('s1', 's2', 's3', 'p1', 'C1', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8',
         'K9', 'K10', 'K11', 'K12', 'K13', 'delta1', 'delta2', 'delta3', 'delta4',
         'delta5', 'delta6', 'delta7', 'alpha1', 'psi')

#: Nominal parameter values (Table 1 of the paper).
NOMINAL = np.array([10, .15, 5, .2, 55.6, 3.87e-3, 1e-6, 4.5e-4, 7.45e-4, 5.22e-4, 3e-6,
                    3.3e-4, 6e-9, .537, .285, 7.79e-6, 1e-6, 4e-5, .01, .28, .05, .005,
                    .005, .015, 2.39, 3e-4, .97])

#: Uniform input ranges: 2.5 % below and above nominal (p. 718 of the paper).
LOWER = .975*NOMINAL
UPPER = 1.025*NOMINAL

#: Initial state (T, T_I, T_L, M, M_I, CTL, V), the same for every sample.
Y0 = np.array([1000., 0., 0., .15/.005, 0., 5/.015, 1.])

#: The floor on T.
T_FLOOR = .1

RTOL = ATOL = 1e-10


def _rhs(t, x, q):
    s1, s2, s3, p1, C1, K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, \
        d1, d2, d3, d4, d5, d6, d7, a1, psi = q
    T, TI, TL, M, MI, CTL, V = x
    inf = (K1*V + K2*MI)*T
    return np.array([
        s1 + p1*T*V/(C1 + V) - d1*T - inf,
        psi*inf + a1*TL - d2*TI - K3*TI*CTL,
        (1 - psi)*inf - a1*TL - d3*TL,
        s2 + K4*M*V - K5*M*V - d4*M,
        K5*M*V - d5*MI - K6*MI*CTL,
        s3 + (K7*TI + K8*MI)*CTL - d6*CTL,
        K9*TI + K10*MI - K11*T*V - (K12 + K13)*M*V - d7*V])


def _jac(t, x, q):
    s1, s2, s3, p1, C1, K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, \
        d1, d2, d3, d4, d5, d6, d7, a1, psi = q
    T, TI, TL, M, MI, CTL, V = x
    r = K1*V + K2*MI
    J = np.zeros((7, 7))
    J[0] = [p1*V/(C1 + V) - d1 - r, 0, 0, 0, -K2*T, 0, p1*T*C1/(C1 + V)**2 - K1*T]
    J[1] = [psi*r, -d2 - K3*CTL, a1, 0, psi*K2*T, -K3*TI, psi*K1*T]
    J[2] = [(1 - psi)*r, 0, -a1 - d3, 0, (1 - psi)*K2*T, 0, (1 - psi)*K1*T]
    J[3] = [0, 0, 0, (K4 - K5)*V - d4, 0, 0, (K4 - K5)*M]
    J[4] = [0, 0, 0, K5*V, -d5 - K6*CTL, -K6*MI, K5*M]
    J[5] = [0, K7*CTL, 0, 0, K8*CTL, K7*TI + K8*MI - d6, 0]
    J[6] = [-K11*V, K9, 0, -(K12 + K13)*V, K10, 0, -K11*T - (K12 + K13)*M - d7]
    return J


def _floor(t, x, q):
    return x[0] - T_FLOOR


_floor.terminal = True
_floor.direction = -1


def trajectory(q, times):
    """T at `times` (increasing, > 0) for one parameter vector `q`, and the floor time.

    Returns ``(T, t_floor)``; ``t_floor`` is ``inf`` if T stays above 0.1 up to
    ``times[-1]``, and ``T`` is 0.1 at every time after ``t_floor``.
    """
    times = np.asarray(times, dtype=float)
    q = tuple(float(v) for v in q)
    sol = solve_ivp(_rhs, (0., times[-1]), Y0, method='LSODA', t_eval=times, events=_floor,
                    args=(q,), jac=_jac, rtol=RTOL, atol=ATOL)
    if sol.status < 0:
        raise RuntimeError(f'solve_ivp failed: {sol.message}')
    T = np.full(len(times), T_FLOOR)
    T[:len(sol.t)] = sol.y[0]
    t_floor = sol.t_events[0][0] if len(sol.t_events[0]) else np.inf
    return T, t_floor


def Tcells(p, times):
    """T-cell count T(t) at `times` for each row of physical parameters `p`.

    Parameters
    ----------
    p : array_like, shape (N, 27)
        Physical parameters, columns in the order of `NAMES`.
    times : array_like, shape (K,)
        Increasing output times in days, all > 0.

    Returns
    -------
    numpy.ndarray, shape (N, K)
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    return np.array([trajectory(q, times)[0] for q in p])
