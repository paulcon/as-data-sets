"""Recompute the numbers of DATA.md (Lorenz96_Initial_State_Inference) that the notebook and
generate_data.py do not print.

    python check_accuracy.py        all checks (a few seconds)

Run it from any directory; it reads the shipped files next to this file and evaluates
lorenz96_model.py. Each line prints the recomputed value; DATA.md quotes these values.

Checks:
 1. The misfit at the truth, Phi_1 and Phi_10, against the noise-only expectation, half the number
    of observations in the window (36 x 101 / 2 and 36 x 1001 / 2).
 2. Time stepping: the RK4 trajectory of lorenz96_model (dt = 0.01) from the truth against
    scipy.integrate.solve_ivp (DOP853, rtol = atol = 1e-13) at the same times: the largest
    difference on [0, 1] and [0, 2], the first times at which it exceeds 1e-3 and 1, and the
    standard deviation of the truth state over t in [0, 10] and the 36 components.
 3. Failure: initial values alternating +-100 (F = 8) make RK4 at dt = 0.01 unstable, and the model
    raises FloatingPointError instead of returning a value; alternating +-50 still runs.
 4. The shipped inputs: the range of F, the largest |input|, the number of distinct rows, whether any
    row equals the prior mean, and the total size of the five CSVs.

Deterministic. Imports numpy, scipy and the standard library.
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lorenz96_model as lm  # noqa: E402

FILES = ('truth.csv', 'observations.csv', 'inputs.csv', 'outputs.csv', 'gradients.csv')


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def check_truth_misfit(x_true, d):
    phi = lm.misfit(x_true[None], d, [1.0, 10.0])[0]
    n1, n10 = lm.K * (int(round(1.0 / lm.DT)) + 1), lm.K * (lm.N_STEPS + 1)
    print(f'1. misfit at the truth: Phi_1 = {phi[0]:.2f} (noise-only expectation {n1 / 2:g}), '
          f'Phi_10 = {phi[1]:.2f} (noise-only expectation {n10 / 2:g})')


def check_time_stepping(x_true):
    t = np.arange(lm.N_STEPS + 1) * lm.DT
    rk4 = lm.trajectory(x_true)[0]
    F = x_true[lm.K]
    ref = solve_ivp(lambda _, X: lm.rhs(X, F), (0.0, lm.T_END), x_true[:lm.K], method='DOP853',
                    t_eval=t, rtol=1e-13, atol=1e-13).y.T
    diff = np.max(np.abs(rk4 - ref), axis=1)
    first = lambda tol: t[np.argmax(diff > tol)] if np.any(diff > tol) else float('nan')  # noqa: E731
    print(f'2. RK4 (dt = 0.01) vs DOP853 (rtol = atol = 1e-13) from the truth: largest difference '
          f'{diff[t <= 1 + 1e-9].max():.2e} on [0, 1], {diff[t <= 2 + 1e-9].max():.2e} on [0, 2]; '
          f'exceeds 1e-3 first at t = {first(1e-3):.2f}, 1 at t = {first(1.0):.2f}; '
          f'standard deviation of the truth state over [0, 10] {rk4.std():.2f}')


def check_failure(d):
    alt = np.where(np.arange(lm.K) % 2 == 0, 1.0, -1.0)
    for a in (100.0, 50.0):
        x = np.r_[a * alt, lm.F_TRUE][None]
        with np.errstate(all='ignore'):
            try:
                phi = lm.misfit(x, d, [lm.T_END])[0, 0]
                out = f'runs (Phi_10 = {phi:.3e})'
            except FloatingPointError as e:
                out = f'raises FloatingPointError ({e})'
        print(f'3. initial values alternating +-{a:g}, F = 8: {out}')


def check_inputs(X):
    size = sum(os.path.getsize(os.path.join(HERE, n)) for n in FILES)
    print(f'4. inputs.csv: F from {X[:, -1].min():.2f} to {X[:, -1].max():.2f}; largest |input| '
          f'{np.abs(X).max():.2f}; {len(np.unique(X, axis=0))} distinct rows of {len(X)}; rows equal to the '
          f'prior mean: {int(np.sum(np.all(X == 0, axis=1)))}; the five CSVs: {size / 1e6:.2f} MB '
          f'({size / 2**20:.2f} MiB)')


def main():
    t0 = time.perf_counter()
    x_true = load('truth.csv')[0]
    d = load('observations.csv')[:, 1:]
    X = load('inputs.csv')
    check_truth_misfit(x_true, d)
    check_time_stepping(x_true)
    check_failure(d)
    check_inputs(X)
    print(f'run time {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
