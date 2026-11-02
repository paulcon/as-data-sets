"""SEIR epidemic-peak model of Jensen, Engsig-Karup and Knudsen (Math. Model. Nat. Phenom. 17:8, 2022), Case 1.

The ODEs are Eq. (4.1) of the paper (Sec. 4.1, p. 7):

    S' = -beta I S / N,   E' = beta I S / N - sigma E,   I' = sigma E - gamma I,   R' = gamma I,

with N = 5.8e6 (the population of Denmark, p. 7). The three uncertain inputs (p. 10) are the
basic reproduction number R0 = beta / gamma, the incubation period tau_inc = 1/sigma (days) and the
infectious period tau_inf = 1/gamma (days), so beta = R0 / tau_inf, sigma = 1 / tau_inc and
gamma = 1 / tau_inf. The outputs (Eq. (4.2)) are the time t_peak (days) at which I is largest and
the peak number of infectious individuals I_peak = I(t_peak).

Choices not stated in the paper (see DATA.md, which shows that they reproduce the paper's
published moments to 0.1 day and 0.01 %):

- Initial state (S, E, I, R)(0) = (N - 1, 0, 1, 0): one infectious individual at t = 0.
- The table on p. 10 gives "mean" and "variance" (0.025^2, 0.25^2, 0.25^2); we read them as the
  mean and variance of the lognormal variable itself, so ln x ~ N(mu, s^2) with
  s^2 = ln(1 + sd^2/mean^2) and mu = ln(mean) - s^2/2 (`MU`, `SIGMA`).
- Solver: scipy `solve_ivp`, LSODA, rtol = 1e-11, atol = 1e-9 on the states and sensitivities,
  up to T_END = 2000 days. The peak is located as a root of g = sigma E - gamma I (= I')
  crossing from positive to negative (solve_ivp event); with I(0) = 1 and E(0) = 0, I first falls
  (g < 0), so the first downward crossing is the epidemic peak. A sample without a downward
  crossing before T_END, or whose crossing has I <= I(0) (no outbreak: with R0 well below 1 the
  solver can report a spurious crossing at I ~ 1e-12), raises RuntimeError. Near R0 = 1 the peak
  moves past T_END (R0 = 1.05 peaks at day 1548; R0 = 1.02 has no peak before day 2000).

Gradients are exact up to the ODE tolerance: forward sensitivities of (S, E, I) with respect to
(beta, sigma, gamma) are integrated with the states (12 equations), and at the peak

    d t_peak / d theta = -(dg/dtheta + grad_y g . s_theta) / (dg/dt),   d I_peak / d theta = s_I,

(the second because I' = 0 at the peak), then mapped to the inputs by the chain rule.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy.integrate import solve_ivp

#: Input names, in column order (p. 10 of the paper).
NAMES = ('R0', 'tau_inc', 'tau_inf')
UNITS = ('-', 'days', 'days')
DESCRIPTIONS = ('basic reproduction number beta/gamma',
                'mean incubation period 1/sigma (time in E)',
                'mean infectious period 1/gamma (time in I)')

#: Mean and standard deviation of each lognormal input (table on p. 10: variance 0.025^2, 0.25^2, 0.25^2).
MEAN = np.array([1.4, 4.2, 3.3])
SD = np.array([0.025, 0.25, 0.25])

#: Parameters of ln x ~ N(MU, SIGMA^2), matching MEAN and SD of x.
SIGMA = np.sqrt(np.log1p((SD / MEAN)**2))
MU = np.log(MEAN) - SIGMA**2 / 2

#: Population (p. 7) and initial state (S, E, I) (not stated in the paper; one infectious person).
N_POP = 5.8e6
Y0 = np.array([N_POP - 1.0, 0.0, 1.0])

T_END = 2000.0
RTOL = 1e-11
ATOL = 1e-9

QOI_NAMES = ('t_peak', 'I_peak')
QOI_UNITS = ('days', 'individuals')


def rates(x):
    """(beta, sigma, gamma) in 1/day from x = (R0, tau_inc, tau_inf)."""
    R0, tinc, tinf = x
    return R0 / tinf, 1.0 / tinc, 1.0 / tinf


def _rhs(t, y, beta, sigma, gamma, n):
    S, E, I = y[:3]
    inf = beta * I * S / n
    dy = np.empty(12)
    dy[0] = -inf
    dy[1] = inf - sigma * E
    dy[2] = sigma * E - gamma * I
    # forward sensitivities: s' = J s + df/dtheta, theta = (beta, sigma, gamma); y[3 + 3k: 6 + 3k] = dy/dtheta_k
    J = np.array([[-beta * I / n, 0.0, -beta * S / n],
                  [beta * I / n, -sigma, beta * S / n],
                  [0.0, sigma, -gamma]])
    dfdth = np.array([[-I * S / n, 0.0, 0.0],
                      [I * S / n, -E, 0.0],
                      [0.0, E, -I]])
    s = y[3:].reshape(3, 3).T          # column k = dy/dtheta_k
    dy[3:] = (J @ s + dfdth).T.ravel()
    return dy


def _rhs_states(t, y, beta, sigma, gamma, n):
    S, E, I = y
    inf = beta * I * S / n
    return [-inf, inf - sigma * E, sigma * E - gamma * I]


def _peak_event(t, y, beta, sigma, gamma, n):
    return sigma * y[1] - gamma * y[2]


_peak_event.terminal = True
_peak_event.direction = -1


def _check(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (3,) or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError(f'SEIR inputs (R0, tau_inc, tau_inf) must be three finite positive numbers; got {x!r}')
    return x


def peak(x, grad=False, y0=Y0, n=N_POP, rtol=RTOL, atol=ATOL, t_end=T_END):
    """(t_peak, I_peak) for one input x = (R0, tau_inc, tau_inf).

    With grad=True also returns the 2 x 3 Jacobian d(t_peak, I_peak)/dx with respect to the
    physical inputs. Raises RuntimeError if I has no maximum above I(0) before t_end.
    """
    x = _check(x)
    beta, sigma, gamma = rates(x)
    args = (beta, sigma, gamma, n)
    if grad:
        y_init = np.concatenate([y0, np.zeros(9)])
        sol = solve_ivp(_rhs, (0.0, t_end), y_init, method='LSODA', rtol=rtol, atol=atol,
                        events=_peak_event, args=args)
    else:
        sol = solve_ivp(_rhs_states, (0.0, t_end), y0, method='LSODA', rtol=rtol, atol=atol,
                        events=_peak_event, args=args)
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise RuntimeError(f'no epidemic peak before t = {t_end} days for x = {x} ({sol.message})')
    tp = sol.t_events[0][0]
    yp = sol.y_events[0][0]
    if not yp[2] > y0[2]:
        raise RuntimeError(f'no outbreak for x = {x}: I at the first maximum ({yp[2]:.3g}) does not exceed I(0) = {y0[2]:g}')
    if not grad:
        return tp, yp[2]
    S, E, I = yp[:3]
    s = yp[3:].reshape(3, 3).T          # s[:, k] = d(S, E, I)/dtheta_k
    dEdt = beta * I * S / n - sigma * E
    dgdt = sigma * dEdt                 # gamma I' = 0 at the peak
    dg_direct = np.array([0.0, E, -I])  # dg/d(beta, sigma, gamma) at fixed state
    dg_state = np.array([0.0, sigma, -gamma]) @ s
    dt_dth = -(dg_direct + dg_state) / dgdt
    dI_dth = s[2]
    # chain rule: theta = (R0/tau_inf, 1/tau_inc, 1/tau_inf)
    R0, tinc, tinf = x
    dth_dx = np.array([[1.0 / tinf, 0.0, -R0 / tinf**2],
                       [0.0, -1.0 / tinc**2, 0.0],
                       [0.0, 0.0, -1.0 / tinf**2]])
    Jac = np.vstack([dt_dth @ dth_dx, dI_dth @ dth_dx])
    return tp, yp[2], Jac


def peaks(X, grad=False, **kw):
    """Row-wise `peak` for X of shape (M, 3): returns f (M, 2) [and G (M, 2, 3)]."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f = np.empty((len(X), 2))
    G = np.empty((len(X), 2, 3)) if grad else None
    for i, x in enumerate(X):
        r = peak(x, grad=grad, **kw)
        f[i] = r[:2]
        if grad:
            G[i] = r[2]
    return (f, G) if grad else f


def trajectory(x, t, y0=Y0, n=N_POP, rtol=RTOL, atol=ATOL):
    """(S, E, I) at the times t (days), shape (3, len(t)), for one input x."""
    x = _check(x)
    t = np.asarray(t, dtype=float)
    sol = solve_ivp(_rhs_states, (0.0, t[-1]), y0, method='LSODA', rtol=rtol, atol=atol,
                    t_eval=t, args=(*rates(x), n))
    if sol.status != 0:
        raise RuntimeError(sol.message)
    return sol.y


def gauss_hermite_moments(npts, qoi=None):
    """Mean, standard deviation and first-order/total Sobol indices of (t_peak, I_peak) by a
    tensor Gauss-Hermite rule with npts points per input in xi = (ln x - MU)/SIGMA (the paper's
    gPC design: npts = 3 gives 27 runs, npts = 10 gives 1000), and a tensor Hermite expansion of
    degree npts - 1 per input. Returns a dict of arrays indexed [qoi]."""
    from itertools import product
    from math import factorial
    xi, w = np.polynomial.hermite_e.hermegauss(npts)
    w = w / w.sum()
    idx = list(product(range(npts), repeat=3))
    X = np.array([np.exp(MU + SIGMA * xi[list(k)]) for k in idx])
    W = np.array([np.prod(w[list(k)]) for k in idx])
    f = peaks(X)
    # orthonormal probabilists' Hermite polynomials at the nodes: P[d, q] = He_d(xi_q)/sqrt(d!)
    P = np.array([np.polynomial.hermite_e.hermeval(xi, [0] * d + [1]) / np.sqrt(factorial(d))
                  for d in range(npts)])
    alphas = idx
    Phi = np.array([[np.prod([P[a[j], k[j]] for j in range(3)]) for k in idx] for a in alphas])
    c = Phi @ (W[:, None] * f)                       # (n_alpha, 2) expansion coefficients
    A = np.array(alphas)
    var = (c[1:]**2).sum(axis=0)
    S1 = np.array([[(c[(A[:, j] > 0) & (A.sum(1) == A[:, j])]**2).sum(0)[q] / var[q]
                    for j in range(3)] for q in range(2)])
    ST = np.array([[(c[A[:, j] > 0]**2).sum(0)[q] / var[q] for j in range(3)] for q in range(2)])
    return {'mean': c[0], 'std': np.sqrt(var), 'S1': S1, 'ST': ST, 'runs': len(X)}
