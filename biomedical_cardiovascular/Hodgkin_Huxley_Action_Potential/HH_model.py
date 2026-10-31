"""Hodgkin-Huxley membrane model with 11 uncertain inputs (Torres Valderrama et al. 2015).

The model is Eq. (1) of A. Torres Valderrama, J. Witteveen, M. Navarro and J. Blom,
"Uncertainty propagation in nerve impulses through the action potential mechanism",
J. Math. Neurosci. 5:3 (2015), doi:10.1186/2190-8567-5-3: the Hodgkin-Huxley (1952)
equations for the membrane potential v (mV, measured from rest, depolarization
positive) and the gating variables m, n, h,

    C dv/dt = -gNa m^3 h (v - ENa) - gK n^4 (v - EK) - gL (v - EL) + I,
    dm/dt = alpha_m(v) (1 - m) - beta_m(v) m,   (and the same for n and h),

with the rate functions alpha, beta as printed in Eq. (1). Eq. (1) as printed adds I
to dv/dt without dividing it by C; `form='printed'` gives that version. The default,
`form='standard'`, is the Hodgkin-Huxley form above, which reproduces the paper's
Sobol ranking (C minor); the printed form makes C the most important input (DATA.md).

The 11 inputs are the vector xi of the paper's Eq. (2), here in the order of its
Table 2: (n0, m0, h0, v0, gK, gNa, gL, EK, ENa, EL, C), uniform on nominal +- 20 %.

Settings that the paper does not state (see DATA.md):

- Injected current I = 150 uA/cm^2 (constant, from t = 0). With this value the
  nominal trajectory matches Fig. 1(a) (v(5 ms) ~ 5 mV, peak ~ 46 mV at 8.5 ms,
  trough ~ 9 mV at 10.7 ms); at C = 1 the two forms are identical.
- Time window 0 to 14 ms; Fig. 1(a) shows 5 to 14 ms.

Quantities of interest (`qoi`): the second local maximum of v(t) (the "acme" of
Fig. 1(a), mV), its time (ms), and v at the 19 times 5, 5.5, ..., 14 ms. The first
maximum is the spike at t < 1 ms; some samples have a third maximum after 13.2 ms,
which is not used.
`qoi_and_grad` also returns their gradients with respect to the physical inputs,
from the forward sensitivity equations solved together with the model.

Solver: `scipy.integrate.solve_ivp`, DOP853, rtol = 1e-10, atol = 1e-12; the
maxima are located by a `solve_ivp` event on dv/dt = 0 (decreasing).

Only numpy and scipy are imported.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp

#: Input names, in column order (Table 2 of the paper).
NAMES = ('n0', 'm0', 'h0', 'v0', 'gK', 'gNa', 'gL', 'EK', 'ENa', 'EL', 'C')

#: Units of the inputs. Table 2 prints the conductances in uS/cm^2; the
#: Hodgkin-Huxley values 36, 120, 0.3 are in mS/cm^2 with C in uF/cm^2 and t in ms.
UNITS = ('-', '-', '-', 'mV', 'mS/cm^2', 'mS/cm^2', 'mS/cm^2', 'mV', 'mV', 'mV', 'uF/cm^2')

#: Nominal values (Table 2 of the paper, first values; Fig. 1 setting).
NOMINAL = np.array([0.0003, 0.0011, 0.9998, -10., 36., 120., 0.3, -12., 115., 10.613, 1.])

#: Uniform ranges: nominal +- 20 % (paper p. 3, "20 % of variability"). For the two
#: negative nominal values (v0, EK) the bounds are 1.2 and 0.8 times nominal.
LOWER = np.minimum(0.8*NOMINAL, 1.2*NOMINAL)
UPPER = np.maximum(0.8*NOMINAL, 1.2*NOMINAL)

#: Injected current, constant from t = 0 (not stated in the paper; see DATA.md).
I_INJ = 150.

#: 'standard' (C dv/dt = I - ionic) or 'printed' (dv/dt = I - ionic/C, Eq. (1) as printed).
FORM = 'standard'

#: End of the simulation (ms) and the 19 output times of the window of Fig. 1(a).
T_END = 14.
TIMES = np.linspace(5., 14., 19)

#: Names of the 21 quantities of interest, in column order.
QOI_NAMES = ('vpeak', 'tpeak') + tuple(f'v({t:g})' for t in TIMES)

RTOL, ATOL = 1e-10, 1e-12


def _xr(x):
    """x / (exp(x) - 1), continuous at x = 0."""
    if abs(x) < 1e-6:
        return 1. - x/2 + x*x/12
    return x/math.expm1(x)


def _dxr(x):
    """Derivative of x / (exp(x) - 1)."""
    if abs(x) < 1e-6:
        return -0.5 + x/6
    e = math.expm1(x)
    return (e - x*(e + 1.))/(e*e)


def _rates(v):
    """alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h and their v-derivatives."""
    un, um, w = (10. - v)/10., (25. - v)/10., (30. - v)/10.
    an, dan = 0.1*_xr(un), -0.01*_dxr(un)
    bn = 0.125*math.exp(-v/80.)
    am, dam = _xr(um), -0.1*_dxr(um)
    bm = 4.*math.exp(-v/18.)
    ah = 0.07*math.exp(-v/20.)
    ew = math.exp(w)
    bh = 1./(ew + 1.)
    dbh = 0.1*ew/(ew + 1.)**2
    return (an, bn, am, bm, ah, bh), (dan, -bn/80., dam, -bm/18., -ah/20., dbh)


def _rhs(t, y, p, I, form):
    v, m, n, h = y[0], y[1], y[2], y[3]
    gK, gNa, gL, EK, ENa, EL, C = p[4:]
    (an, bn, am, bm, ah, bh), _ = _rates(v)
    Iv = I/C if form == 'standard' else I
    return [(-gNa*m**3*h*(v - ENa) - gK*n**4*(v - EK) - gL*(v - EL))/C + Iv,
            am*(1 - m) - bm*m, an*(1 - n) - bn*n, ah*(1 - h) - bh*h]


def _jacobians(y, p, I, form):
    """d(rhs)/d(state) (4 x 4) and d(rhs)/d(inputs) (4 x 11)."""
    v, m, n, h = y[0], y[1], y[2], y[3]
    gK, gNa, gL, EK, ENa, EL, C = p[4:]
    (an, bn, am, bm, ah, bh), (dan, dbn, dam, dbm, dah, dbh) = _rates(v)
    iNa, iK, iL = gNa*m**3*h*(v - ENa), gK*n**4*(v - EK), gL*(v - EL)
    Jy = np.array([
        [-(gNa*m**3*h + gK*n**4 + gL)/C, -3*gNa*m*m*h*(v - ENa)/C,
         -4*gK*n**3*(v - EK)/C, -gNa*m**3*(v - ENa)/C],
        [dam*(1 - m) - dbm*m, -(am + bm), 0., 0.],
        [dan*(1 - n) - dbn*n, 0., -(an + bn), 0.],
        [dah*(1 - h) - dbh*h, 0., 0., -(ah + bh)]])
    Jp = np.zeros((4, 11))
    Jp[0, 4:] = [-n**4*(v - EK)/C, -m**3*h*(v - ENa)/C, -(v - EL)/C,
                 gK*n**4/C, gNa*m**3*h/C, gL/C,
                 (iNa + iK + iL - (I if form == 'standard' else 0.))/C**2]
    return Jy, Jp


def _rhs_sens(t, z, p, I, form):
    """Model plus forward sensitivities S = dy/dp (4 x 11), flattened after y."""
    y = z[:4]
    Jy, Jp = _jacobians(y, p, I, form)
    S = z[4:].reshape(4, 11)
    return np.concatenate([_rhs(t, y, p, I, form), (Jy @ S + Jp).ravel()])


def _vmax(t, z, p, I, form):
    """Event: dv/dt = 0 with dv/dt decreasing (a local maximum of v)."""
    return _rhs(t, z, p, I, form)[0]


_vmax.direction = -1


def _y0(p):
    # state order (v, m, n, h); input order (n0, m0, h0, v0, ...)
    return [p[3], p[1], p[0], p[2]]


def trajectory(p, times, I=I_INJ, t_end=T_END, form=FORM, rtol=RTOL, atol=ATOL):
    """Solve the model for one input vector; returns the solve_ivp result (y = v, m, n, h)."""
    p = tuple(float(x) for x in p)
    return solve_ivp(_rhs, (0., t_end), _y0(p), method='DOP853', t_eval=times,
                     args=(p, I, form), events=_vmax, rtol=rtol, atol=atol)


def _peak(sol):
    """Index of the second local maximum of v; raises if the regime is not the expected one."""
    te = sol.t_events[0]
    if len(te) < 2 or not (te[0] < 2. < TIMES[0] < te[1] < TIMES[-1] - 2.):
        raise RuntimeError(f'expected the first maximum of v before 2 ms and the second in '
                           f'({TIMES[0]}, {TIMES[-1] - 2}) ms; got maxima at t = {te}')
    return 1


def qoi(p, I=I_INJ, form=FORM, rtol=RTOL, atol=ATOL):
    """The 21 quantities of interest (QOI_NAMES) for one physical input vector."""
    sol = trajectory(p, TIMES, I, form=form, rtol=rtol, atol=atol)
    k = _peak(sol)
    return np.concatenate([[sol.y_events[0][k, 0], sol.t_events[0][k]], sol.y[0]])


def qoi_and_grad(p, I=I_INJ, form=FORM, rtol=RTOL, atol=ATOL):
    """The 21 QoIs and their gradients w.r.t. the physical inputs, shapes (21,) and (21, 11).

    v(t_i) and the peak value: the v-row of the forward sensitivity S(t) (at the
    peak dv/dt = 0, so d vpeak/dp = S_v(tpeak)). Peak time, by the implicit function
    theorem on dv/dt(tpeak) = 0: d tpeak/dp = -(dS_v/dt)(tpeak) / v''(tpeak).
    """
    p = tuple(float(x) for x in p)
    S0 = np.zeros((4, 11))
    S0[0, 3] = S0[1, 1] = S0[2, 0] = S0[3, 2] = 1.
    sol = solve_ivp(_rhs_sens, (0., T_END), np.concatenate([_y0(p), S0.ravel()]),
                    method='DOP853', t_eval=TIMES, args=(p, I, form), events=_vmax,
                    rtol=rtol, atol=atol)
    k = _peak(sol)
    zp = sol.y_events[0][k]
    yp, Sp = zp[:4], zp[4:].reshape(4, 11)
    Jy, Jp = _jacobians(yp, p, I, form)
    dSv = Jy[0] @ Sp + Jp[0]                   # d/dt of dv/dp at the peak
    vdd = Jy[0] @ np.array(_rhs(0., yp, p, I, form))  # v'' at the peak
    f = np.concatenate([[yp[0], sol.t_events[0][k]], sol.y[0]])
    G = np.vstack([Sp[0], -dSv/vdd, sol.y[4:4 + 11].T])
    return f, G
