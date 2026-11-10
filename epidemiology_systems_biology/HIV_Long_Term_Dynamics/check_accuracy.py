"""Recompute the accuracy numbers of DATA.md for HIV_Long_Term_Dynamics.

    python check_accuracy.py                  # both checks, 4 worker processes
    python check_accuracy.py --workers 2

1. Outputs against tighter solves: Radau at rtol = atol = 1e-13, with the same floor event,
   against outputs.csv (LSODA at 1e-10) at rows 0-4 and at rows 714, 51 and 175, three samples
   whose T comes close to the floor. Largest relative difference over the times above the floor
   (DATA.md: 3e-9 to 3.5e-7 at rows 0-4, 4e-9 at rows 0, 1, 4; at most 1.1e-5 at rows 714, 51, 175).
2. Gradients against tighter solves: central differences with h = 3e-4 in x, with LSODA at
   rtol = atol = 1e-12, against gradients.csv (h = 3e-4 at 1e-10), at rows 0, 1, 2, 714 and at
   rows 482, 197, 469; relative 2-norm error of the 27-vector, worst time above the floor (DATA.md:
   1e-5, 9.0e-6 to 1.6e-5). At rows 0, 1, 2 and 714 also the steps h = 1e-4 and h = 1e-3 at
   1e-10 against the same reference (DATA.md: rows 0-2, 2-3e-5 and 3e-6; row 714, 2e-2 and 4e-2
   at t = 3400).

The check of HIV_model.py against outputs.csv (1.8e-13) is printed by the notebook, and
`generate_data.py --check` compares a rerun with the files. Imports numpy, scipy, the standard
library and the modules of this directory.
Deterministic (fixed rows); it reads the files here and writes nothing.
"""
import argparse
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import HIV_model as hm  # noqa: E402
from generate_data import TIMES, read_csv  # noqa: E402

TYPICAL = [0, 1, 2, 3, 4]               # check 1
NEAR_FLOOR = [714, 51, 175]             # check 1
STEP_ROWS = [0, 1, 2, 714]              # check 2, with the step study
MORE_GRAD_ROWS = [482, 197, 469]        # check 2
H = 3e-4
HALF = (hm.UPPER - hm.LOWER)/2


def traj(q, method='LSODA', tol=hm.RTOL):
    """T at TIMES and the floor time, as HIV_model.trajectory but with a chosen solver and tolerance."""
    q = tuple(float(v) for v in q)
    sol = solve_ivp(hm._rhs, (0., TIMES[-1]), hm.Y0, method=method, t_eval=TIMES, events=hm._floor,
                    args=(q,), jac=hm._jac, rtol=tol, atol=tol)
    if sol.status < 0:
        raise RuntimeError(f'solve_ivp failed: {sol.message}')
    T = np.full(len(TIMES), hm.T_FLOOR)
    T[:len(sol.t)] = sol.y[0]
    return T, (sol.t_events[0][0] if len(sol.t_events[0]) else np.inf)


def _radau(p):
    return traj(p, 'Radau', 1e-13)[0]


def _cd(args):
    """Central differences in x with step h at tolerance tol (as generate_data.sample), shape (19, 27)."""
    p, h, tol = args
    _, t_floor = traj(p, tol=tol)
    G = np.empty((len(TIMES), len(p)))
    for j in range(len(p)):
        dp = np.zeros(len(p))
        dp[j] = h*HALF[j]
        G[:, j] = (traj(p + dp, tol=tol)[0] - traj(p - dp, tol=tol)[0])/(2*h)
    G[TIMES > t_floor] = 0.
    return G


def grad_err(G, Gref):
    """Relative 2-norm error per time where the reference is nonzero (NaN elsewhere)."""
    n = np.linalg.norm(Gref, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, np.linalg.norm(G - Gref, axis=-1)/n, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    t_start = time.perf_counter()

    X = read_csv(os.path.join(HERE, 'inputs.csv'))
    f = read_csv(os.path.join(HERE, 'outputs.csv'))
    G = read_csv(os.path.join(HERE, 'gradients.csv')).reshape(len(X), len(TIMES), 27)
    print(f'scipy {scipy.__version__}, numpy {np.__version__}')

    with Pool(a.workers) as pool:
        print('\n1. Radau at rtol = atol = 1e-13 against outputs.csv (largest relative difference over the '
              'times above the floor)')
        rows1 = TYPICAL + NEAR_FLOOR
        R = dict(zip(rows1, pool.map(_radau, X[rows1])))
        for r in rows1:
            k = f[r] > hm.T_FLOOR
            print(f'   row {r:3d} (min T {f[r].min():8.4g}): {np.max(np.abs(R[r] - f[r])[k]/f[r][k]):.1e}')
        print('   (DATA.md: 3e-9 to 3.5e-7 at rows 0-4, 4e-9 at rows 0, 1, 4; at most 1.1e-5 at rows 714, 51, 175)')

        print('\n2. central differences, h = 3e-4 at rtol = atol = 1e-12 (reference), against gradients.csv '
              '(relative 2-norm error, worst time above the floor)')
        rows2 = STEP_ROWS + MORE_GRAD_ROWS
        tasks = [(X[r], H, 1e-12) for r in rows2] + [(X[r], h, hm.RTOL) for r in STEP_ROWS for h in (1e-4, 1e-3)]
        out = pool.map(_cd, tasks)
        ref = dict(zip(rows2, out[:len(rows2)]))
        step = dict(zip([(r, h) for r in STEP_ROWS for h in (1e-4, 1e-3)], out[len(rows2):]))
        for r in rows2:
            e = grad_err(G[r], ref[r])
            line = f'   row {r:3d}: h = 3e-4 (shipped) {np.nanmax(e):.1e}'
            if r in STEP_ROWS:
                for h in (1e-4, 1e-3):
                    eh = grad_err(step[r, h], ref[r])
                    line += f'; h = {h:.0e} {np.nanmax(eh):.1e} (at t = {TIMES[np.nanargmax(eh)]})'
            print(line)
        print('   (DATA.md: 9.0e-6 to 1.6e-5; rows 0-2: h = 1e-4 2-3e-5, h = 1e-3 3e-6; row 714: 2e-2 and 4e-2 '
              'at t = 3400)')

    print(f'\nrun time {time.perf_counter() - t_start:.0f} s on {a.workers} workers')


if __name__ == '__main__':
    main()
