"""Simplified MAPK cascade of Nguyen et al. (2015) as used by Linden, Kramer and Rangamani
(PLOS Comput. Biol. 18(10):e1010651, 2022, Eqs. 32a-c, p. 28).

Three states, the phosphorylated RAF, MEK and MAPK/ERK concentrations x1, x2, x3:

    dx1/dt = k1 (S1t - x1) K1^n1 / (K1^n1 + x3^n1) - k2 x1
    dx2/dt = k3 (S2t - x2) x1 (1 + alpha x3^n2 / (K2^n2 + x3^n2)) - k4 x2
    dx3/dt = k5 (S3t - x3) x2 - k6 x3

with 14 parameters. Two nominal parameter sets give the two regimes the paper studies
(bistable steady states, limit-cycle oscillations). The seven parameters varied in the
paper's Sobol' study are the rate constants k1-k6 and the feedback strength alpha; the
other seven (S1t, S2t, S3t, n1, K1, n2, K2) are held at their nominal values.

Nominal values, initial conditions, ranges, time windows and the limit-cycle thresholds
are those of the authors' code (github.com/RangamaniLabUCSD/CIUKF-MCMC, commit 11a4f1e,
`MAPK/sensitivity_analysis_RUN.m`, `utils/limitCycleFeatures.m`); the code itself is not
copied (the repository states no licence). Time is in seconds, as in that code
(t = 1800 s is the paper's 30 min); concentrations are in nM, and the units of the rate
constants (1/s and 1/(nM s)) are inferred from the equations (the paper's S2 Table prints
nM/s for all of them; see DATA.md). The range of k1 is the code's [0, 0.1]; the S2 Table
gives [0, 0.05].

Public functions:

- `bistable(theta, x0)`: x2 and x3 at t = 1800 s and their gradients with respect to
  the seven free parameters (forward sensitivities), bistable nominal set.
- `oscillatory(theta)`: time-mean of x3 over 980 <= t <= 5400 s with its gradient, and
  the limit-cycle amplitude and period of x3 (no gradients), oscillatory nominal set.
- `limit_cycle_features(y, dt, late_minima=False)`: the amplitude/period rule of the paper's
  Section 1.13 (late_minima=True: the authors' one-sample-late reading of the minima).

Only numpy and scipy are imported.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

#: All 14 parameters, in the order of the authors' sensitivity script.
ALL_NAMES = ('S1t', 'S2t', 'S3t', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',
             'n1', 'K1', 'n2', 'K2', 'alpha')

#: Nominal values, bistable regime (`ptrueBistable` in the authors' code; S2 Table).
NOMINAL_BISTABLE = np.array([0.22, 10, 53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066,
                             5, 9.5, 10, 15, 95.])
#: Nominal values, oscillatory regime (`ptrueOSC` in the authors' code; S2 Table).
NOMINAL_OSCILLATORY = np.array([100, 100, 100, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01,
                                10, 1, 15, 8, 10.])

#: The seven parameters varied in the Sobol' study (Fig. 4 caption), in column order.
NAMES = ('k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'alpha')
FREE = np.array([ALL_NAMES.index(n) for n in NAMES])

#: Uniform ranges of the free parameters (`paramBounds` in the authors' code).
LOWER_BISTABLE = np.array([0, 0, 0, 0, 0, 0, 0.])
UPPER_BISTABLE = np.array([0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 100.])
#: Oscillatory regime: the same, except k5 >= 1e-5 (S2 Table note).
LOWER_OSCILLATORY = np.array([0, 0, 0, 0, 1e-5, 0, 0.])
UPPER_OSCILLATORY = UPPER_BISTABLE.copy()

#: Initial conditions (Fig. 4 caption; `BISTABLE_*SS_RUN.m`; `x0` of the oscillatory run).
X0_HIGH = np.array([0.1245, 2.4870, 31.2623])
X0_LOW = np.array([0.0015, 3.6678, 28.7307])
X0_OSC = np.array([10., 80., 80.])

#: Held settings.
T_END_BISTABLE = 1800.0      # s (30 min); the QoI is the state at this time
T_END_OSC = 5400.0           # s
DT_OSC = 20.0                # s, output grid of the oscillatory run
START_OSC = 49               # 0-based index of t = 980 s (MATLAB startIdx = 50)
PEAK_THRESHOLD = 17.0        # max - min of the peak heights must be below this (the code's full
                             # spread, limitCycleFeatures.m; the paper's text, p. 23, says one half)
LCA_THRESHOLD = 1.0          # amplitude must exceed this
DECAY_THRESHOLD = 5.0        # range of the cycle amplitudes must be below this
RTOL = ATOL = 1e-10          # bistable runs
RTOL_OSC = ATOL_OSC = 1e-8   # oscillatory runs (5400 s of spiking oscillation)

_NF = len(NAMES)


def _full(theta, nominal):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (_NF,):
        raise ValueError(f'theta must have shape ({_NF},), got {theta.shape}')
    if not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise ValueError('the free parameters must be finite and nonnegative')
    p = np.array(nominal, dtype=float)
    p[FREE] = theta
    return p


def rhs(x, p):
    """Right-hand side of Eqs. 32a-c for the full 14-parameter vector p."""
    S1t, S2t, S3t, k1, k2, k3, k4, k5, k6, n1, K1, n2, K2, a = p
    x1, x2, x3 = x
    h1 = K1**n1/(K1**n1 + x3**n1)
    g = 1 + a*x3**n2/(K2**n2 + x3**n2)
    return np.array([k1*(S1t - x1)*h1 - k2*x1,
                     k3*(S2t - x2)*x1*g - k4*x2,
                     k5*(S3t - x3)*x2 - k6*x3])


def _coef(x, p):
    """f, the 7 nonzero entries of df/dx and the 7 nonzero entries of df/dtheta at x."""
    S1t, S2t, S3t, k1, k2, k3, k4, k5, k6, n1, K1, n2, K2, a = p
    x1, x2, x3 = x[0], x[1], x[2]
    A = K1**n1
    u = x3**n1
    h1 = A/(A + u)
    dh1 = -A*n1*x3**(n1 - 1)/(A + u)**2 if x3 > 0 else 0.0
    B = K2**n2
    v = x3**n2
    r = v/(B + v)
    dr = n2*x3**(n2 - 1)*B/(B + v)**2 if x3 > 0 else 0.0
    g = 1 + a*r
    s1, s2, s3 = S1t - x1, S2t - x2, S3t - x3
    f = (k1*s1*h1 - k2*x1, k3*s2*x1*g - k4*x2, k5*s3*x2 - k6*x3)
    J = (-k1*h1 - k2, k1*s1*dh1,                          # J11, J13
         k3*s2*g, -k3*x1*g - k4, k3*s2*x1*a*dr,           # J21, J22, J23
         k5*s3, -k5*x2 - k6)                              # J32, J33
    # df/dk1 (row 1), df/dk2 (1), df/dk3 (2), df/dk4 (2), df/dk5 (3), df/dk6 (3), df/dalpha (2)
    Fp = (s1*h1, -x1, s2*x1*g, -x2, s3*x2, -x3, k3*s2*x1*r)
    return f, J, Fp


def jacobian(x, p):
    """df/dx, 3x3, for the full 14-parameter vector p."""
    J11, J13, J21, J22, J23, J32, J33 = _coef(x, p)[1]
    return np.array([[J11, 0.0, J13], [J21, J22, J23], [0.0, J32, J33]])


def _aug(t, z, p):
    """Augmented system z = (x, s_1, ..., s_7), s_j = dx/dtheta_j: x' = f, s_j' = J s_j + f_j."""
    f, (J11, J13, J21, J22, J23, J32, J33), Fp = _coef(z, p)
    S = z[3:].reshape(_NF, 3)
    a, b, c = S[:, 0], S[:, 1], S[:, 2]
    d1 = J11*a + J13*c
    d2 = J21*a + J22*b + J23*c
    d3 = J32*b + J33*c
    d1[0] += Fp[0]
    d1[1] += Fp[1]
    d2[2] += Fp[2]
    d2[3] += Fp[3]
    d3[4] += Fp[4]
    d3[5] += Fp[5]
    d2[6] += Fp[6]
    return np.concatenate((f, np.stack((d1, d2, d3), 1).ravel()))


_EYE = np.eye(_NF + 1)


def _aug_jac(t, z, p):
    """Block-diagonal Newton matrix kron(I_8, df/dx) for the augmented system.

    It omits the coupling d(J s_j)/dx; LSODA uses it only in its Newton iteration, so
    it changes the cost, not the solution (error control is unchanged)."""
    return np.kron(_EYE, jacobian(z, p))


def solve(p, x0, times, sensitivities=True, rtol=RTOL, atol=ATOL):
    """Solve Eqs. 32 from x0 and return states (3, nt) and, optionally, the forward
    sensitivities d x / d theta (3, 7, nt) at `times` (times[0] must be 0).

    LSODA (scipy.integrate.solve_ivp) with an analytic (block) Jacobian. Raises
    RuntimeError on solver failure or a non-finite or negative state (the model's
    states stay in [0, S_it])."""
    times = np.asarray(times, dtype=float)
    if sensitivities:
        z0 = np.concatenate([x0, np.zeros(3*_NF)])
        fun, jac = _aug, _aug_jac
    else:
        z0 = np.asarray(x0, dtype=float)
        fun = lambda t, x, p: rhs(x, p)  # noqa: E731
        jac = lambda t, x, p: jacobian(x, p)  # noqa: E731
    sol = solve_ivp(fun, (times[0], times[-1]), z0, t_eval=times, args=(p,),
                    method='LSODA', jac=jac, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'ODE solver failed: {sol.message}')
    y = sol.y
    if not np.all(np.isfinite(y)):
        raise RuntimeError('non-finite state or sensitivity')
    if y[:3].min() < -1e-6:
        raise RuntimeError(f'negative state {y[:3].min():.3g}')
    if sensitivities:
        return y[:3], np.transpose(y[3:].reshape(_NF, 3, -1), (1, 0, 2))
    return y[:3], None


def bistable(theta, x0, gradients=True, rtol=RTOL, atol=ATOL):
    """x2(1800 s), x3(1800 s) and their gradients (2, 7) w.r.t. (k1..k6, alpha).

    theta: the seven free parameters (physical units); x0: X0_HIGH or X0_LOW.
    The other seven parameters are at NOMINAL_BISTABLE."""
    p = _full(theta, NOMINAL_BISTABLE)
    y, S = solve(p, x0, [0.0, T_END_BISTABLE], gradients, rtol, atol)
    q = y[1:, -1]
    return (q, S[1:, :, -1]) if gradients else (q, None)


def osc_times():
    return np.arange(0.0, T_END_OSC + DT_OSC/2, DT_OSC)


def oscillatory(theta, gradients=True, rtol=RTOL_OSC, atol=ATOL_OSC):
    """Oscillatory regime (other parameters at NOMINAL_OSCILLATORY, x0 = X0_OSC).

    Returns dict: mean_x3 (mean of x3 on the 20 s grid, 980 <= t <= 5400 s),
    grad_mean_x3 (7,), amplitude, period (min) and oscillates (bool) from
    `limit_cycle_features` on the same window."""
    p = _full(theta, NOMINAL_OSCILLATORY)
    y, S = solve(p, X0_OSC, osc_times(), gradients, rtol, atol)
    x3 = y[2, START_OSC:]
    lca, per = limit_cycle_features(x3, DT_OSC)
    out = {'mean_x3': float(x3.mean()), 'amplitude': lca, 'period': per,
           'oscillates': lca > 0}
    if gradients:
        out['grad_mean_x3'] = S[2, :, START_OSC:].mean(axis=-1)
    return out


def limit_cycle_features(y, dt, late_minima=False):
    """Limit-cycle amplitude and period (minutes) of a sampled trajectory y (step dt, s).

    The rule of the paper's Section 1.13 with the thresholds of the authors' code: local
    maxima and minima (scipy.signal.find_peaks, MATLAB findpeaks semantics); no limit
    cycle (0, 0) if either set is empty, if the peak heights span PEAK_THRESHOLD or more,
    if the mean amplitude (paired maxima minus minima) is <= LCA_THRESHOLD, if the range of
    the amplitudes is >= DECAY_THRESHOLD, or if neither set has two members. Otherwise
    amplitude = mean(max_i - min_i) over the first min(#max, #min) pairs and period = mean
    of all max-to-max and min-to-min intervals. 0 marks 'no limit cycle', as in the authors' code.

    late_minima=False (the default, used for the data) takes the minima at the local-minimum
    samples. late_minima=True reproduces the authors' `limitCycleFeatures.m`, which reads each
    minimum one grid point late (`lMin = trajectory(delay+minLocs)` with 1-based `minLocs`
    into `trajectory(delay:end)`); the maxima and the period are the same either way.

    On a flat-topped peak (equal neighbouring samples) scipy's find_peaks returns the middle
    sample of the plateau, MATLAB's findpeaks the first. The heights are equal, so only the
    period can differ, by at most half a plateau width; this needs exactly equal samples,
    which do not occur at the nominal point (period 13.04 min, as in the paper).
    """
    y = np.asarray(y, dtype=float)
    imax, _ = find_peaks(y)
    imin, _ = find_peaks(-y)
    if len(imax) == 0 or len(imin) == 0:
        return 0.0, 0.0
    lmax = y[imax]
    lmin = y[np.minimum(imin + 1, len(y) - 1)] if late_minima else y[imin]
    if np.ptp(lmax) >= PEAK_THRESHOLD:
        return 0.0, 0.0
    k = min(len(lmax), len(lmin))
    amps = lmax[:k] - lmin[:k]
    lca = float(amps.mean())
    if lca <= LCA_THRESHOLD or np.ptp(amps) >= DECAY_THRESHOLD:
        return 0.0, 0.0
    gaps = np.concatenate([np.diff(imax), np.diff(imin)])*dt/60.0
    if len(gaps) == 0:
        return 0.0, 0.0
    return lca, float(gaps.mean())
