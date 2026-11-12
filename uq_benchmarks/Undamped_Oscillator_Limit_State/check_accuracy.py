"""Recompute the numbers of DATA.md (Undamped_Oscillator_Limit_State) that neither the notebook
nor generate_data.py prints.

    python check_accuracy.py                    all checks (a few minutes; see DATA.md)
    python check_accuracy.py --skip-slow        without the two large Monte Carlo runs (checks 6, 7)
    python check_accuracy.py --skip-uqtestfuns  without check 5 (needs UQTestFuns)

Yielding oscillator (the main model):
 1. Without yielding (r = 10^6) the event-by-event solution against the linear closed form at the
    1000 shipped inputs.
 2. Against an independent time-stepping solution (velocity Verlet with a return map for spring 2,
    the pulse end on the time grid, written here for this check) at rows 0-199 of the shipped
    design, with 100 to 3200 steps per pulse duration: the largest relative difference in z_max
    and the observed order of convergence.
 3. Gradients: exact (forward-mode) against central differences in the standardized log inputs
    with steps 1e-3 to 1e-7 at all 1000 shipped samples, and the number of samples whose
    stencil reaches a switching set (pulse end at a yield event; yield onset) for each step.
 4. Peaks: in 10^6 lognormal draws (seed 21), how often the largest |z| is not the first peak
    or is negative; and, on 10^4 of them, that the energy-bound stop leaves z_max unchanged.
Linear closed form (the secondary study):
 5. Implementation: limit_state_linear against UQTestFuns 0.7.0 (uqtestfuns.UndampedOscillator)
    at the 1000 shipped inputs and at 10^5 draws of its own default input; and the input model
    printed by UQTestFuns. Needs `pip install uqtestfuns==0.7.0`.
 6. Failure probabilities for the reduced pulse loads of Echard et al. (2013), the second and
    third input models of UQTestFuns (normal inputs, F1 ~ N(0.6, 0.1) and N(0.45, 0.075)):
    FORM design point by the HL-RF iteration with the exact gradients, then importance sampling
    (10^6 draws from N(u*, I) in standard normal coordinates, seed 11) with a 95 % normal-theory
    interval; and, for N(0.6, 0.1) only, plain Monte Carlo with 10^8 draws (seed 12; slow) with a
    95 % Wilson interval.
Both models:
 7. Failure probability with the paper's lognormal inputs by plain Monte Carlo, 10^7 draws (seed
    31), 95 % Wilson intervals (slow: about 70 s for the yielding model).

Deterministic. Imports numpy, the standard library and, for check 5 only, uqtestfuns.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oscillator_model as om  # noqa: E402

Z95 = 1.959963984540054


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def zmax_timestep(X, n_pulse):
    """Velocity Verlet with a return map: c1 linear, c2 elastic-perfectly-plastic (|s| <= r)."""
    m, c1, c2, r, F1, t1 = X.T
    dt = t1 / n_pulse
    t_end = t1 + 2 * np.pi * np.sqrt(m / c1)          # the first peak comes well before this
    n_steps = int(np.ceil(np.max(t_end / dt)))
    z = np.zeros(len(X))
    v = np.zeros(len(X))
    s = np.zeros(len(X))
    zm = np.zeros(len(X))
    a = F1 / m
    for k in range(n_steps):
        vh = v + 0.5 * dt * a
        dz = dt * vh
        z = z + dz
        s = np.clip(s + dz, -r, r)
        # the force at the end of the step: left limit F1 up to and including t1, zero after
        F = np.where(k + 1 <= n_pulse, F1, 0.0)
        a = (F - c1 * z - c2 * s) / m
        v = vh + 0.5 * dt * a
        if k + 1 == n_pulse:
            a = (0.0 - c1 * z - c2 * s) / m          # the pulse ends exactly on the grid
        inside = (k + 1) * dt <= t_end
        zm = np.where(inside, np.maximum(zm, np.abs(z)), zm)
    return zm


def check_uqtestfuns(X):
    try:
        import uqtestfuns as uqtf
    except ImportError:
        print('check 5 needs UQTestFuns: install it with `pip install uqtestfuns==0.7.0` and rerun '
              '(or rerun with --skip-uqtestfuns)', file=sys.stderr)
        sys.exit(3)
    fun = uqtf.UndampedOscillator()
    print(f'5. UQTestFuns {uqtf.__version__}: default input model of UndampedOscillator')
    for mg in fun.prob_input.marginals:
        print(f'   {mg.name:3s} {mg.distribution:7s} parameters {np.asarray(mg.parameters).tolist()}')
    d1 = np.max(np.abs(fun(X) - om.limit_state_linear(X)))
    Xu = fun.prob_input.get_sample(100000)
    d2 = np.max(np.abs(fun(Xu) - om.limit_state_linear(Xu)))
    print(f'   max |g_lin - g_UQTestFuns|: shipped inputs {d1:.1e}; 10^5 draws of the UQTestFuns input {d2:.1e}')


def form_is(mean, sd, N, seed):
    """HL-RF design point in u = (x - mean)/sd, then importance sampling from N(u*, I) (linear model)."""
    u = np.zeros(6)
    for it in range(100):
        g, G = om.limit_state_linear(mean + sd * u, grad=True)
        gu = G * sd
        u_new = (gu @ u - g) / (gu @ gu) * gu
        if np.linalg.norm(u_new - u) < 1e-12:
            u = u_new
            break
        u = u_new
    beta = np.linalg.norm(u)
    rng = np.random.default_rng(seed)
    V = u + rng.standard_normal((N, 6))
    fail = om.limit_state_linear(mean + sd * V) <= 0
    w = np.exp(-V @ u + 0.5 * beta ** 2) * fail       # phi(v) / phi(v - u*)
    pf = w.mean()
    hw = Z95 * w.std(ddof=1) / np.sqrt(N)
    from math import erfc, sqrt
    return u, beta, 0.5 * erfc(beta / sqrt(2)), it + 1, pf, hw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-slow', action='store_true', help='skip the large Monte Carlo runs (checks 6 MC and 7)')
    ap.add_argument('--skip-uqtestfuns', action='store_true', help='skip check 5')
    args = ap.parse_args()
    t0 = time.perf_counter()
    X, G = load('inputs.csv'), load('gradients.csv')

    Xn = X.copy()
    Xn[:, 3] = 1e6
    zn = om.response(Xn)['z_max']
    zl, _ = om.parts_linear(X)
    print(f'1. yielding switched off (r = 1e6): max relative difference from the linear closed form {np.max(np.abs(zn - zl) / zl):.1e}')

    ze = om.response(X[:200])['z_max']
    print('2. time stepping (velocity Verlet with return map) against the event-by-event solution, rows 0-199:')
    prev = None
    for n in (100, 200, 400, 800, 1600, 3200):
        e = np.max(np.abs(zmax_timestep(X[:200], n) - ze) / ze)
        rate = '' if prev is None else f', observed order {np.log2(prev / e):.2f}'
        print(f'   {n:5d} steps per pulse: max relative difference {e:.1e}{rate}')
        prev = e

    print('3. gradients vs central differences in the standardized log inputs (relative error, row 2-norm):')
    res = om.response(X, grad=True)
    sy = om.SIGMA * X
    Gy = res['grad'] * sy
    print(f'   shipped gradients vs the model: max |difference| {np.max(np.abs(res["grad"] - G)):.1e}')
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        fd = np.empty_like(Gy)
        for j in range(6):
            e = np.zeros(6)
            e[j] = h * om.SIGMA[j]
            fd[:, j] = (om.limit_state(X * np.exp(e)) - om.limit_state(X * np.exp(-e))) / (2 * h)
        err = np.linalg.norm(fd - Gy, axis=1) / np.linalg.norm(Gy, axis=1)
        near = {k: int((v < h).sum()) for k, v in res['margins'].items()}
        print(f'   h = {h:g}: max {err.max():.1e}, median {np.median(err):.1e}; samples within h of a '
              f'pulse-end / yield-onset switching set: {near["pulse_end"]} / {near["yield_onset"]}')

    rng = np.random.default_rng(21)
    nn, n_not1, n_neg = 0, 0, 0
    for _ in range(10):
        r_ = om.response(om.sample_lognormal(10 ** 5, rng))
        nn += len(r_['z_max'])
        n_not1 += int((r_['peak_index'] != 1).sum())
        n_neg += int((r_['peak_sign'] < 0).sum())
    Xs = om.sample_lognormal(10 ** 4, np.random.default_rng(22))
    a_, b_ = om.response(Xs, full=True), om.response(Xs, full=False)
    print(f'4. {nn:.0e} lognormal draws: largest |z| not at the first peak in {n_not1}, negative in {n_neg}; '
          f'10^4 draws, energy-bound stop vs full solution: max |difference in z_max| {np.max(np.abs(a_["z_max"] - b_["z_max"])):.1e}, '
          f'smallest relative gap to the next peak (full) {np.min(a_["peak_gap"]):.1e}')

    if not args.skip_uqtestfuns:
        check_uqtestfuns(X)

    print('6. linear closed form, reduced pulse loads (normal inputs, Echard et al. 2013 Table 4 as listed by UQTestFuns):')
    for F1m, F1s, ref in ((0.60, 0.100, '9.09e-6 (MCS, 1.8e8 draws, CoV 2.47 %)'),
                          (0.45, 0.075, '1.55e-8 (MCS, 9e8 draws, CoV 2.68 %)')):
        mean = om.MEAN.copy()
        sd = om.SD.copy()
        mean[4], sd[4] = F1m, F1s
        u, beta, pform, its, pf, hw = form_is(mean, sd, 10 ** 6, 11)
        print(f'   F1 ~ N({F1m}, {F1s}): HL-RF {its} iterations, beta = {beta:.4f}, FORM Phi(-beta) = {pform:.3e}; '
              f'IS (10^6) Pf = {pf:.3e} +- {hw:.1e} (95 %); UQTestFuns reference {ref}')
    if not args.skip_slow:
        mean = om.MEAN.copy()
        sd = om.SD.copy()
        mean[4], sd[4] = 0.60, 0.100
        t1 = time.perf_counter()
        pf, nf, N, (lo, hi), _ = om.failure_probability(None, 10 ** 8, np.random.default_rng(12), chunk=10 ** 6,
                                                        mean=mean, sd=sd, model=om.limit_state_linear)
        print(f'   F1 ~ N(0.6, 0.1), plain Monte Carlo: {nf} failures in {N:.0e} draws, Pf = {pf:.3e}, '
              f'95 % interval [{lo:.3e}, {hi:.3e}] ({time.perf_counter() - t1:.0f} s)')

        print('7. lognormal inputs (Table 7), plain Monte Carlo 10^7 draws (seed 31), 95 % Wilson intervals:')
        for name, model in (('yielding oscillator', om.limit_state), ('linear closed form', om.limit_state_linear)):
            t1 = time.perf_counter()
            pf, nf, N, (lo, hi), _ = om.failure_probability(om.sample_lognormal, 10 ** 7, np.random.default_rng(31),
                                                            model=model)
            print(f'   {name}: {nf} failures, Pf = {pf:.4e} [{lo:.4e}, {hi:.4e}] ({time.perf_counter() - t1:.0f} s)')
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
