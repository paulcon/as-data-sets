"""Hodgkin-Huxley point neuron with eleven uncertain parameters, after Tennøe, Halnes and Einevoll (2018).

The space-clamped Hodgkin-Huxley (1952) model, with the membrane potential v (mV) measured from
rest (0 mV) and depolarization positive:

    Cm dv/dt = I - gNa m^3 h (v - ENa) - gK n^4 (v - EK) - gL (v - EL)
    dm/dt = alpha_m(v) (1 - m) - beta_m(v) m
    dn/dt = alpha_n(v) (1 - n) - beta_n(v) n
    dh/dt = alpha_h(v) (1 - h) - beta_h(v) h

    alpha_n = 0.01 (10 - v) / (exp((10 - v)/10) - 1)    beta_n = 0.125 exp(-v/80)
    alpha_m = 0.1 (25 - v) / (exp((25 - v)/10) - 1)     beta_m = 4 exp(-v/18)
    alpha_h = 0.07 exp(-v/20)                           beta_h = 1 / (exp((30 - v)/10) + 1)

with t in ms, conductances in mS/cm^2, Cm in uF/cm^2 and the stimulus I = 140 uA/cm^2 constant
from t = 0. The rate functions are those of Eq. (1) of Torres Valderrama et al. (2015); the
parameter values are Table 2 of Tennøe et al. (2018). The removable singularities of alpha_n and
alpha_m (v = 10 and v = 25 mV) are evaluated through u / expm1(u) with a series near u = 0.

Inputs (order of NAMES): V0, Cm, gNa, gK, gL, ENa, EK, EL, n0, m0, h0, each uniform on its
nominal value +/- 10 %.

Outputs (order of QOI_NAMES), from one solve on [0, T_END] ms:
    vpeak, tpeak   the first local maximum of v after T_WINDOW[0] = 5 ms (mV, ms)
    vmin, tmin     the first local minimum of v after tpeak (mV, ms)
    v(t)           v at t = 5, 5.5, ..., 15 ms (mV)

Gradients with respect to the physical inputs come from the forward sensitivity equations,
solved with the state. At an extremum t* of v, dv/dt = 0, so d v(t*)/dp = s_v(t*) and
d t*/dp = -(grad_x F_v . s(t*) + dF_v/dp) / (grad_x F_v . F) at t*, where F is the right-hand
side and F_v its v component.

Functions: qoi_and_grad(p), qoi(p), trajectory(p, times), extrema(p), sample(M, rng), and the
right-hand side and Jacobians rhs, jacobians, rates. Constants: NAMES, UNITS, NOMINAL, LOWER,
UPPER, I_STIM, T_END, T_WINDOW, TIMES, QOI_NAMES, QOI_UNITS.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy.integrate import solve_ivp

NAMES = ('V0', 'Cm', 'gNa', 'gK', 'gL', 'ENa', 'EK', 'EL', 'n0', 'm0', 'h0')
UNITS = ('mV', 'uF/cm^2', 'mS/cm^2', 'mS/cm^2', 'mS/cm^2', 'mV', 'mV', 'mV', '-', '-', '-')
NOMINAL = np.array([-10.0, 1.0, 120.0, 36.0, 0.3, 112.0, -12.0, 10.613, 0.0011, 0.0003, 0.9998])
REL_HALF_WIDTH = 0.10
LOWER = NOMINAL - REL_HALF_WIDTH * np.abs(NOMINAL)
UPPER = NOMINAL + REL_HALF_WIDTH * np.abs(NOMINAL)
M_INPUTS = len(NAMES)

I_STIM = 140.0                     # uA/cm^2, constant from t = 0
T_END = 15.0                       # ms
T_WINDOW = (5.0, 15.0)             # ms, the analysis window
TIMES = np.linspace(5.0, 15.0, 21)  # ms, 0.5 ms apart
QOI_NAMES = ('vpeak', 'tpeak', 'vmin', 'tmin') + tuple(f'v({t:g})' for t in TIMES)
QOI_UNITS = ('mV', 'ms', 'mV', 'ms') + ('mV',) * len(TIMES)
N_QOI = len(QOI_NAMES)

RTOL = 1e-10
ATOL = 1e-12

_IV0, _ICM, _IGNA, _IGK, _IGL, _IENA, _IEK, _IEL, _IN0, _IM0, _IH0 = range(M_INPUTS)


# ---------------------------------------------------------------- rate functions and derivatives
def _xexpm1(u):
    """g(u) = u / (exp(u) - 1) and g'(u), with g(0) = 1, g'(0) = -1/2."""
    u = np.asarray(u, dtype=float)
    small = np.abs(u) < 1e-4
    us = np.where(small, 1.0, u)
    e = np.expm1(us)
    g = np.where(small, 1.0 - u / 2 + u * u / 12, us / e)
    dg = np.where(small, -0.5 + u / 6 - u ** 3 / 180, (e - us * (e + 1.0)) / (e * e))
    return g, dg


def rates(v):
    """(alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h) and their derivatives with respect to v."""
    gm, dgm = _xexpm1((25.0 - v) / 10.0)
    gn, dgn = _xexpm1((10.0 - v) / 10.0)
    am, dam = gm, -dgm / 10.0                   # 0.1 (25 - v) / expm1((25 - v)/10) = g(u)
    an, dan = 0.1 * gn, -0.01 * dgn             # 0.01 (10 - v) / expm1((10 - v)/10) = 0.1 g(u)
    bm = 4.0 * np.exp(-v / 18.0)
    bn = 0.125 * np.exp(-v / 80.0)
    ah = 0.07 * np.exp(-v / 20.0)
    eh = np.exp((30.0 - v) / 10.0)
    bh = 1.0 / (eh + 1.0)
    return ((am, bm, an, bn, ah, bh),
            (dam, -bm / 18.0, dan, -bn / 80.0, -ah / 20.0, eh / (10.0 * (eh + 1.0) ** 2)))


# ---------------------------------------------------------------- right-hand side and Jacobians
def rhs(x, p, I=I_STIM):
    """F(x; p) for the state x = (v, m, n, h)."""
    v, m, n, h = x
    (am, bm, an, bn, ah, bh), _ = rates(v)
    _, Cm, gNa, gK, gL, ENa, EK, EL = p[:8]
    iion = gNa * m ** 3 * h * (v - ENa) + gK * n ** 4 * (v - EK) + gL * (v - EL)
    return np.array([(I - iion) / Cm,
                     am * (1 - m) - bm * m,
                     an * (1 - n) - bn * n,
                     ah * (1 - h) - bh * h])


def jacobians(x, p, I=I_STIM):
    """F, dF/dx (4 x 4) and dF/dp (4 x 11)."""
    v, m, n, h = x
    (am, bm, an, bn, ah, bh), (dam, dbm, dan, dbn, dah, dbh) = rates(v)
    _, Cm, gNa, gK, gL, ENa, EK, EL = p[:8]
    m3h, n4 = m ** 3 * h, n ** 4
    iion = gNa * m3h * (v - ENa) + gK * n4 * (v - EK) + gL * (v - EL)
    F = np.array([(I - iion) / Cm,
                  am * (1 - m) - bm * m,
                  an * (1 - n) - bn * n,
                  ah * (1 - h) - bh * h])
    Jx = np.zeros((4, 4))
    Jx[0, 0] = -(gNa * m3h + gK * n4 + gL) / Cm
    Jx[0, 1] = -gNa * 3 * m ** 2 * h * (v - ENa) / Cm
    Jx[0, 2] = -gK * 4 * n ** 3 * (v - EK) / Cm
    Jx[0, 3] = -gNa * m ** 3 * (v - ENa) / Cm
    Jx[1, 0] = dam * (1 - m) - dbm * m
    Jx[1, 1] = -(am + bm)
    Jx[2, 0] = dan * (1 - n) - dbn * n
    Jx[2, 2] = -(an + bn)
    Jx[3, 0] = dah * (1 - h) - dbh * h
    Jx[3, 3] = -(ah + bh)
    Jp = np.zeros((4, M_INPUTS))
    Jp[0, _ICM] = -(I - iion) / Cm ** 2
    Jp[0, _IGNA] = -m3h * (v - ENa) / Cm
    Jp[0, _IGK] = -n4 * (v - EK) / Cm
    Jp[0, _IGL] = -(v - EL) / Cm
    Jp[0, _IENA] = gNa * m3h / Cm
    Jp[0, _IEK] = gK * n4 / Cm
    Jp[0, _IEL] = gL / Cm
    return F, Jx, Jp


def initial_state(p):
    """(v, m, n, h) at t = 0 and its derivative with respect to p (4 x 11)."""
    x0 = np.array([p[_IV0], p[_IM0], p[_IN0], p[_IH0]])
    S0 = np.zeros((4, M_INPUTS))
    S0[0, _IV0] = S0[1, _IM0] = S0[2, _IN0] = S0[3, _IH0] = 1.0
    return x0, S0


# ---------------------------------------------------------------- solves
def _augmented(t, y, p, I):
    x = y[:4]
    S = y[4:].reshape(4, M_INPUTS)
    F, Jx, Jp = jacobians(x, p, I)
    return np.concatenate([F, (Jx @ S + Jp).ravel()])


def trajectory(p, times, I=I_STIM, rtol=RTOL, atol=ATOL):
    """v(t) at the given times (state only, no sensitivities)."""
    p = np.asarray(p, dtype=float)
    x0, _ = initial_state(p)
    sol = solve_ivp(lambda t, x: rhs(x, p, I), (0.0, float(np.max(times))), x0, method='DOP853',
                    t_eval=times, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0]


def _dvdt_event(t, y, p, I):
    return rhs(y[:4], p, I)[0]


def _solve(p, I, rtol, atol, grad):
    x0, S0 = initial_state(p)
    event = lambda t, y, p=p, I=I: _dvdt_event(t, y, p, I)  # noqa: E731
    if grad:
        fun, y0 = (lambda t, y: _augmented(t, y, p, I)), np.concatenate([x0, S0.ravel()])
    else:
        fun, y0 = (lambda t, y: rhs(y, p, I)), x0
    sol = solve_ivp(fun, (0.0, T_END), y0, method='DOP853', t_eval=TIMES, events=event,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def extrema(p, I=I_STIM, rtol=RTOL, atol=ATOL):
    """Times and values of every local maximum and minimum of v on (0, T_END]:
    (t_max, v_max, t_min, v_min)."""
    p = np.asarray(p, dtype=float)
    sol = _solve(p, I, rtol, atol, grad=False)
    te, ye = sol.t_events[0], sol.y_events[0]
    d2 = np.array([jacobians(y[:4], p, I)[1][0] @ rhs(y[:4], p, I) for y in ye])
    return te[d2 < 0], ye[d2 < 0, 0], te[d2 > 0], ye[d2 > 0, 0]


def qoi_and_grad(p, I=I_STIM, rtol=RTOL, atol=ATOL, grad=True):
    """The N_QOI outputs and their gradient (N_QOI x 11) with respect to the physical inputs
    (the gradient is None when grad is False)."""
    p = np.asarray(p, dtype=float)
    sol = _solve(p, I, rtol, atol, grad)
    te, ye = sol.t_events[0], sol.y_events[0]
    # classify each zero of dv/dt by the sign of d2v/dt2 = grad_x F_v . F
    kinds = []
    for y in ye:
        F, Jx, _ = jacobians(y[:4], p, I)
        kinds.append(Jx[0] @ F)
    kinds = np.array(kinds)
    after = te > T_WINDOW[0]
    imax = np.flatnonzero(after & (kinds < 0))
    if imax.size == 0:
        raise RuntimeError('no maximum of v after 5 ms')
    ipk = imax[0]
    imin = np.flatnonzero((te > te[ipk]) & (kinds > 0))
    if imin.size == 0:
        raise RuntimeError('no minimum of v after the peak before T_END')
    imn = imin[0]

    f = np.empty(N_QOI)
    f[0], f[1], f[2], f[3] = ye[ipk][0], te[ipk], ye[imn][0], te[imn]
    f[4:] = sol.y[0]
    if not grad:
        return f, None
    G = np.empty((N_QOI, M_INPUTS))
    for k, i in enumerate((ipk, imn)):
        y = ye[i]
        S = y[4:].reshape(4, M_INPUTS)
        F, Jx, Jp = jacobians(y[:4], p, I)
        G[2 * k] = S[0]
        G[2 * k + 1] = -(Jx[0] @ S + Jp[0]) / (Jx[0] @ F)
    G[4:] = sol.y[4:4 + M_INPUTS].T
    return f, G


def qoi(p, I=I_STIM, rtol=RTOL, atol=ATOL):
    """The N_QOI outputs, from a solve of the state alone."""
    return qoi_and_grad(p, I, rtol, atol, grad=False)[0]


def sample(M, rng):
    """M independent uniform samples of the physical inputs, shape (M, 11)."""
    return LOWER + (UPPER - LOWER) * rng.random((M, M_INPUTS))
