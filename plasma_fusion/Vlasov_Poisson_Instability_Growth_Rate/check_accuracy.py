"""Recompute the accuracy numbers of DATA.md that the notebook and generate_data.py do not print
(Vlasov_Poisson_Instability_Growth_Rate).

    python check_accuracy.py                      # all three checks, 2 to 4 min in one process
    python check_accuracy.py --checks 1 3         # a subset

1. Stress test on wide boxes ("Failure rates"). 1500 Bi-Maxwellian samples, uniform on
   k in [0.1, 1.5], mu1 in [-1, 1], mu2 in [1, 8], s1, s2 in [0.05, 2], beta in [0.01, 0.99], and
   1500 Two-Stream samples, uniform on k in [0.1, 1.5], mu in [-1, 1], s in [0.1, 3]
   (numpy default_rng([2024, 1]) and default_rng([2024, 2])), each solved once with
   `most_unstable_root`. Printed: how many raised RootError, how many needed the finer rescan,
   the range of gamma (Bi-Maxwellian), and the median and largest time a sample.
   DATA.md: none failed and none needed the rescan; gamma from -3.21 to 0.34; 10-40 ms a sample
   (median), about 0.1 s at most. The times depend on the machine and its load.

2. Gradients against finite differences ("Accuracy"). At 60 rows of each of the five files
   (default_rng(2024).choice, without replacement), central differences of `growth_rate` in the
   normalized inputs of the row's own box (step 1e-5; the held mu1 of bump_on_tail_s025 has no
   normalized direction and is left out) against the stored gradient times (upper - lower)/2.
   Printed: the median and largest relative error (2-norm of the gradient vector) per file.
   DATA.md: medians from 8.4e-10 to 8.5e-9; maxima 6.6e-9 (global), 6.0e-8 (Double Beam),
   5.2e-8 (Bump-on-Tail), 9.2e-8 (Bump-on-Tail sigma^2 = 0.25), 3.5e-8 (Two-Stream).

3. Continuation of the unstable root ("Stable samples"). For each of the 512 samples at 10 % in
   bump_on_tail_s025.csv, the baseline root (sigma1^2 = sigma2^2 = 0.25, mu1 = 0, from
   `most_unstable_root`, gamma > 0) is continued along the straight line in input space from the
   baseline to the sample, in 100 equal steps, with Newton's method (`dispersion`,
   `dispersion_du`) started at each step from the root of the step before. Printed: at how many
   samples the continued root is the stored one (|omega_continued - omega_file| < 1e-8, with
   omega = gamma i + omega_r), and the largest differences in gamma and omega_r.
   DATA.md: the same root at all 512 samples.

The other numbers of DATA.md are printed by generate_data.py --check (outputs against the shipped
files, rescans and failures in the shipped run) and by the notebook (5 rows against central
differences, the gap counts, the comparison with the paper).

Deterministic apart from the run times; imports numpy, scipy, plasma_dispersion_model.py and
generate_data.py (for the study boxes); reads the files next to it and writes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True                  # write nothing into the directory
sys.path.insert(0, HERE)
import plasma_dispersion_model as pm  # noqa: E402
import generate_data as gd  # noqa: E402

WIDE = {'bm': (np.array([0.1, -1.0, 1.0, 0.05, 0.05, 0.01]), np.array([1.5, 1.0, 8.0, 2.0, 2.0, 0.99])),
        'ts': (np.array([0.1, -1.0, 0.1]), np.array([1.5, 1.0, 3.0]))}
N_WIDE = 1500
N_FD, H_FD = 60, 1e-5
N_STEPS = 100


def load(study):
    """inputs, gamma, omega_r, gradients and the (lower, upper) box of each row of a study file."""
    fn, kind, boxes = gd.STUDIES[study]
    A = np.loadtxt(os.path.join(HERE, fn), delimiter=',')
    m, o = len(pm.NAMES[kind]), int(study != 'global')
    X, f, wr, G = A[:, o:o + m], A[:, o + m], A[:, o + m + 1], A[:, o + m + 2:o + 2*m + 2]
    lo, hi = np.empty_like(X), np.empty_like(X)
    for pct, lo_, hi_ in boxes:
        sel = A[:, 0] == pct if o else slice(None)
        lo[sel], hi[sel] = lo_, hi_
    return kind, X, f, wr, G, lo, hi


def check_wide():
    print(f'1. stress test on wide boxes, {N_WIDE} samples each')
    for i, (kind, label) in enumerate([('bm', 'Bi-Maxwellian'), ('ts', 'Two-Stream')], start=1):
        lo, hi = WIDE[kind]
        X = lo + (hi - lo)*np.random.default_rng([2024, i]).random((N_WIDE, len(lo)))
        fails, refined, gam, times = 0, 0, [], []
        for x in X:
            t = time.perf_counter()
            try:
                u, info = pm.most_unstable_root(x, kind, return_info=True)
            except pm.RootError:
                fails += 1
                continue
            finally:
                times.append(time.perf_counter() - t)
            refined += info['refined']
            gam.append(x[0]*u.imag)
        print(f'   {label}: {fails} failed, {refined} needed the finer rescan; gamma from {min(gam):.2f} to '
              f'{max(gam):.2f}; {1e3*np.median(times):.0f} ms a sample (median), {max(times):.2f} s at most')
    print('   (DATA.md: none failed, none rescanned; Bi-Maxwellian gamma from -3.21 to 0.34; '
          '10-40 ms a sample (median), about 0.1 s at most)')


def check_fd():
    print(f'2. stored gradients against central differences (normalized inputs, step {H_FD:g}), {N_FD} rows '
          'per file; relative 2-norm error')
    for study in gd.STUDIES:
        kind, X, f, wr, G, lo, hi = load(study)
        rows = np.random.default_rng(2024).choice(len(X), N_FD, replace=False)
        err = []
        for i in rows:
            half = (hi[i] - lo[i])/2
            g_fd = np.zeros(len(half))
            for j in np.flatnonzero(half > 0):
                e = np.zeros(len(half))
                e[j] = H_FD*half[j]
                g_fd[j] = (pm.growth_rate(X[i] + e, kind) - pm.growth_rate(X[i] - e, kind))/(2*H_FD)
            g = G[i]*half
            err.append(np.linalg.norm(g_fd - g)/np.linalg.norm(g))
        print(f'   {gd.STUDIES[study][0]:24s} median {np.median(err):.1e}, max {max(err):.1e}')
    print('   (DATA.md: medians 8.4e-10 to 8.5e-9; maxima 6.6e-9 global, 6.0e-8 Double Beam, 5.2e-8 Bump-on-Tail, '
          '9.2e-8 Bump-on-Tail sigma^2 = 0.25, 3.5e-8 Two-Stream)')


def check_continuation():
    kind, X, f, wr, G, lo, hi = load('bump_on_tail_s025')
    A = np.loadtxt(os.path.join(HERE, gd.STUDIES['bump_on_tail_s025'][0]), delimiter=',')
    sel = A[:, 0] == 10
    X, f, wr = X[sel], f[sel], wr[sel]
    x0 = gd.NOMINAL_BOT_S025
    u0 = pm.most_unstable_root(x0, kind)
    u = np.full(len(X), u0, complex)
    for s in np.arange(1, N_STEPS + 1)/N_STEPS:
        xs = (x0[:, None] + s*(X - x0).T)                     # (6, n): inputs at this step, one column a sample
        done = np.zeros(len(X), bool)
        for _ in range(50):
            step = pm.dispersion(u, xs, kind)/pm.dispersion_du(u, xs, kind)
            step[done] = 0
            u = u - step
            done |= np.abs(step) < 1e-13*(1 + np.abs(u))
            if done.all():
                break
        if not done.all() or not np.all(np.isfinite(u)):
            raise RuntimeError(f'Newton did not converge at step {s:.2f} for {np.sum(~done)} samples')
    k = X[:, 0]
    dg, dw = np.abs(k*u.imag - f), np.abs(k*u.real - wr)
    same = np.abs(k*u - (wr + 1j*f)) < 1e-8
    print(f'3. Bump-on-Tail sigma^2 = 0.25, 10 %: baseline root omega = {x0[0]*u0.real:.4f} + {x0[0]*u0.imag:.4f}i '
          f'continued in {N_STEPS} steps to each of the {len(X)} samples ({np.sum(f < 0)} stable)')
    print(f'   the continued root is the stored one at {same.sum()} of {len(X)} samples; max |gamma difference| '
          f'{dg.max():.1e}, max |omega_r difference| {dw.max():.1e} (DATA.md: all 512)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--checks', type=int, nargs='*', default=[1, 2, 3], choices=[1, 2, 3])
    a = ap.parse_args()
    t0 = time.perf_counter()
    for c in a.checks:
        t = time.perf_counter()
        {1: check_wide, 2: check_fd, 3: check_continuation}[c]()
        print(f'   ({time.perf_counter() - t:.0f} s)')
    print(f'run time {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
