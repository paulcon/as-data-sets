"""Recompute the accuracy numbers of DATA.md for HIV_Long_Term_Dynamics, and its comparison with the 2016 files.

    python check_accuracy.py                  # all checks, 4 worker processes, 6-13 min
    python check_accuracy.py --skip-slow      # without check 5 (the 2016 code on all 1000 samples), 34 s
    python check_accuracy.py --workers 2

Checks 3-6 read the 2016 files and the 2016 HIV_model.py at the tag legacy-master (the directory
was then HIV/) with `git show`. They need a git checkout of this repository that has the tag;
without one the script stops with an error before any check runs.

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
3. The 2016 outputs against outputs.csv, on the 18 times both files have: the samples at the
   floor (DATA.md: the same at every time; 6, 24, 61, 130, 283, 474 at t = 1800 ... 3400, and 2
   at t = 1700, which the 2016 files lack); relative differences above the floor (DATA.md: median
   1.0e-7, 99th percentile 1.2e-5, maximum 5.5e-4; the ten above 1e-4 at T < 3; largest at rows
   714, 51, 175); at those three rows the Radau solve of check 1 against both files (DATA.md:
   within 1.1e-5 of the new values, up to 5.6e-4 from the 2016 ones); and the 2016 values at
   the floor (DATA.md: 0.09999512 to 0.09999999; the new ones exactly 0.1).
4. The 2016 gradients against gradients.csv, same times: relative difference of the 27-vector
   above the floor (DATA.md: median 2.9e-5, 99th percentile 2.6e-3, maximum 0.16 at t = 38;
   above 1e-2 at t = 24-140, rows 197, 469, 482, 952, and at t = 3400, row 714); at rows 482,
   197, 469 the 2016 gradient against the 1e-12 reference of check 2 (DATA.md: 0.16, 0.05, 0.04);
   and the norms of the 2016 gradients at the floor (DATA.md: about 0.1 to 230; the new ones are 0).
5. The 2016 code rerun: the 2016 `Tcells` (scipy `ode`, vode), read from the tag and run with
   the installed scipy on the 1000 inputs, against the 2016 outputs.csv: the largest relative
   difference, the samples at the floor in one run and not in the other, and the largest
   difference where both are above the floor (DATA.md, scipy 1.17.1: 0.45; 20 samples; 0.10 at row
   714, t = 3400, and above 1e-3 at 176 samples). About 0.5 s per sample; `--skip-slow` leaves it out.
6. The analysis on the 2016 files: eigenvalues of the mean outer product of the gradients (all
   1000 samples; the matrix that v1's active_subspace estimates from these gradients) at t = 24 and t = 3400 (DATA.md: 2016 files lambda_1 = 1.0461e4, lambda_1/lambda_2
   = 3957 at t = 24 and 4.5356e5, 853 at t = 3400; new files 1.046e4, 3954 and 4.535e5, 874), and
   the largest difference between the first eigenvectors of the two files, up to sign, over the
   18 common times (DATA.md: 9e-5).

The check of HIV_model.py against outputs.csv (1.8e-13) is printed by the notebook, and
`generate_data.py --check` compares a rerun with the files. Imports numpy, scipy, the standard
library and the modules of this directory; check 5 also runs the 2016 module read from git.
Deterministic (fixed rows); it reads the files here and writes nothing.
"""
import argparse
import io
import os
import subprocess
import sys
import time
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import HIV_model as hm  # noqa: E402
from generate_data import TIMES, read_csv  # noqa: E402

TAG, OLD_DIR = 'legacy-master', 'HIV'
#: Output times of the 2016 files (days).
TIMES16 = np.array([5, 15, 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 750, 1000, 1600, 1800, 2000,
                    2200, 2400, 2800, 3400])
TYPICAL = [0, 1, 2, 3, 4]               # check 1
NEAR_FLOOR = [714, 51, 175]             # checks 1 and 3
STEP_ROWS = [0, 1, 2, 714]              # check 2, with the step study
OLD_GRAD_ROWS = [482, 197, 469]         # checks 2 and 4
H = 3e-4
HALF = (hm.UPPER - hm.LOWER)/2


def git_show(name):
    """Text of HIV/<name> at the tag legacy-master; exits if git or the tag is missing."""
    try:
        out = subprocess.run(['git', 'show', f'{TAG}:{OLD_DIR}/{name}'], cwd=HERE, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f'error: cannot read {OLD_DIR}/{name} at the tag {TAG} with git ({e}).\n'
                 f'Checks 3-6 need a git checkout of github.com/paulcon/as-data-sets with the tag {TAG} '
                 f'(git fetch --tags). Run this script inside it and rerun.')
    return out.stdout


def read_2016(name):
    """A 2016 pandas CSV (integer header row, index column), as an array without the index."""
    return np.loadtxt(io.StringIO(git_show(name)), delimiter=',', skiprows=1, ndmin=2)[:, 1:]


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


_LEGACY = {}


def _legacy_init(src):
    exec(compile(src, f'{TAG}:{OLD_DIR}/HIV_model.py', 'exec'), _LEGACY)


def _legacy(p):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # vode warns "excess work done"
        return _LEGACY['Tcells'](p[None], np.linspace(1, 3400, 3400))[0, TIMES16 - 1]


def grad_err(G, Gref):
    """Relative 2-norm error per time where the reference is nonzero (NaN elsewhere)."""
    n = np.linalg.norm(Gref, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, np.linalg.norm(G - Gref, axis=-1)/n, np.nan)


def eig(G):
    lam, W = np.linalg.eigh(G.T @ G/len(G))
    return lam[::-1], W[:, ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--skip-slow', action='store_true', help='leave out check 5')
    a = ap.parse_args()
    t_start = time.perf_counter()

    legacy_src = git_show('HIV_model.py')             # exits here without git or the tag
    X = read_csv(os.path.join(HERE, 'inputs.csv'))
    f = read_csv(os.path.join(HERE, 'outputs.csv'))
    G = read_csv(os.path.join(HERE, 'gradients.csv')).reshape(len(X), len(TIMES), 27)
    X16, f16 = read_2016('inputs.csv'), read_2016('outputs.csv')
    G16 = read_2016('gradients.csv').reshape(len(X), len(TIMES16), 27)
    print(f'scipy {scipy.__version__}, numpy {np.__version__}; 2016 files read from {TAG}:{OLD_DIR}/ '
          f'(inputs identical to inputs.csv: {np.array_equal(X16, X)})')

    with Pool(a.workers, initializer=_legacy_init, initargs=(legacy_src,)) as pool:
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
        rows2 = STEP_ROWS + OLD_GRAD_ROWS
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

        # the 18 common times
        c_new = np.array([k for k, t in enumerate(TIMES) if t in TIMES16])
        c_old = np.array([list(TIMES16).index(TIMES[k]) for k in c_new])
        tc = TIMES[c_new]
        fn, fo = f[:, c_new], f16[:, c_old]
        floor_n, floor_o = fn <= hm.T_FLOOR, fo <= hm.T_FLOOR
        print(f'\n3. 2016 outputs against outputs.csv on the {len(tc)} common times')
        print(f'   same samples at the floor at every time: {np.array_equal(floor_n, floor_o)}; at the floor at '
              + ', '.join(f't = {t}: {n}' for t, n in zip(tc, floor_n.sum(0)) if n)
              + f'; at t = 1700 (new files only): {(f[:, list(TIMES).index(1700)] <= hm.T_FLOOR).sum()}')
        above = ~floor_n
        rel = np.abs(fo - fn)/fn
        ra = rel[above]
        print(f'   above the floor: median {np.median(ra):.1e}, 99th percentile {np.percentile(ra, 99):.1e}, '
              f'maximum {ra.max():.1e}')
        big = above & (rel > 1e-4)
        print(f'   {big.sum()} differences above 1e-4, all at T < 3: {bool(np.all(fn[big] < 3))} '
              f'(largest new T among them {fn[big].max():.2f})')
        worst = np.argsort(-np.where(above, rel, 0).max(1))[:3]
        print(f'   rows with the three largest differences: {", ".join(map(str, worst))}')
        for r in NEAR_FLOOR:
            kn = f[r] > hm.T_FLOOR
            ko = above[r]
            print(f'   row {r}: Radau 1e-13 against the new values {np.max(np.abs(R[r] - f[r])[kn]/f[r][kn]):.1e}, '
                  f'against the 2016 values {np.max(np.abs(R[r][c_new] - fo[r])[ko]/fo[r][ko]):.1e}')
        print(f'   at the floor: 2016 values {fo[floor_o].min():.10f} to {fo[floor_o].max():.10f}; new values all '
              f'exactly 0.1: {bool(np.all(fn[floor_n] == hm.T_FLOOR))}')
        print('   (DATA.md: the same samples; 6, 24, 61, 130, 283, 474 at t = 1800 ... 3400, 2 at t = 1700; median '
              '1.0e-7, 99th percentile 1.2e-5, maximum 5.5e-4; ten above 1e-4, at T < 3; rows 714, 51, 175; '
              'Radau within 1.1e-5 of the new values, up to 5.6e-4 from the 2016 ones; 0.09999512 to 0.09999999)')

        print('\n4. 2016 gradients against gradients.csv on the common times (relative difference of the 27-vector)')
        Gn, Go = G[:, c_new], G16[:, c_old]
        rg = grad_err(Go, Gn)
        rga = rg[above]
        print(f'   above the floor: median {np.median(rga):.1e}, 99th percentile {np.percentile(rga, 99):.1e}, '
              f'maximum {rga.max():.2f} (at t = {tc[np.nanargmax(np.where(above, rg, np.nan)) % len(tc)]})')
        r_big, k_big = np.nonzero(above & (rg > 1e-2))
        print(f'   above 1e-2 at rows {", ".join(map(str, np.unique(r_big)))}; (row, t): '
              + ', '.join(f'({r}, {tc[k]})' for r, k in sorted(zip(r_big, k_big))))
        for r in OLD_GRAD_ROWS:
            print(f'   row {r}: 2016 gradient against the 1e-12 reference of check 2: '
                  f'{np.nanmax(grad_err(Go[r], ref[r][c_new])):.2f}; new gradient {np.nanmax(grad_err(G[r], ref[r])):.1e}')
        no = np.linalg.norm(Go, axis=2)[floor_n]
        print(f'   at the floor: 2016 gradient norms {no.min():.2g} to {no.max():.3g}; new gradients all 0: '
              f'{bool(np.all(np.linalg.norm(Gn, axis=2)[floor_n] == 0))}')
        print('   (DATA.md: median 2.9e-5, 99th percentile 2.6e-3, maximum 0.16 at t = 38; above 1e-2 at t = 24-140, rows '
              '197, 469, 482, 952, and at t = 3400, row 714; rows 482, 197, 469: 0.16, 0.05, 0.04; at the floor about '
              '0.1 to 230)')

        if a.skip_slow:
            print('\n5. skipped (--skip-slow)')
        else:
            t0 = time.perf_counter()
            T16 = np.array(pool.map(_legacy, X, chunksize=10))
            rl = np.abs(T16 - f16)/np.abs(f16)
            both = (T16 > hm.T_FLOOR) & (f16 > hm.T_FLOOR)         # both runs above the floor
            one = (T16 > hm.T_FLOOR) != (f16 > hm.T_FLOOR)         # one run at the floor, the other not
            i, k = np.unravel_index(np.argmax(rl), rl.shape)
            ib, kb = np.unravel_index(np.argmax(np.where(both, rl, 0)), rl.shape)
            print(f'\n5. the 2016 Tcells (vode) rerun with scipy {scipy.__version__} on the 1000 inputs, against the '
                  f'2016 outputs.csv at its 21 times ({time.perf_counter() - t0:.0f} s)')
            print(f'   largest relative difference {rl[i, k]:.1e} (row {i}, t = {TIMES16[k]}); '
                  f'{one.sum()} values at {len(np.unique(np.nonzero(one)[0]))} samples are at the floor in one run '
                  f'and not in the other')
            print(f'   where both are above the floor: largest {rl[ib, kb]:.1e} (row {ib}, t = {TIMES16[kb]}), '
                  f'{(np.where(both, rl, 0).max(1) > 1e-3).sum()} samples above 1e-3 (DATA.md: 0.45; 20 samples; 0.10, 176 samples)')

    print('\n6. eigenvalues of the mean outer product of the gradients, all 1000 samples')
    for t in (24, 3400):
        lo, _ = eig(G16[:, list(TIMES16).index(t)])
        ln, _ = eig(G[:, list(TIMES).index(t)])
        print(f'   t = {t}: 2016 files lambda_1 = {lo[0]:.4e}, lambda_1/lambda_2 = {lo[0]/lo[1]:.0f}; '
              f'new files {ln[0]:.4e}, {ln[0]/ln[1]:.0f}')
    dmax = 0.
    for kn, ko in zip(c_new, c_old):
        wn, wo = eig(G[:, kn])[1][:, 0], eig(G16[:, ko])[1][:, 0]
        dmax = max(dmax, np.abs(wo*np.sign(wo @ wn) - wn).max())
    print(f'   first eigenvectors, largest difference in any component, up to sign, over the {len(tc)} times: {dmax:.1e}')
    print('   (DATA.md: 2016 1.0461e4, 3957 and 4.5356e5, 853; new 1.046e4, 3954 and 4.535e5, 874; eigenvectors 9e-5)')
    print(f'\nrun time {time.perf_counter() - t_start:.0f} s on {a.workers} workers')


if __name__ == '__main__':
    main()
