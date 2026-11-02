"""Recompute the numbers of DATA.md (MAPK_Cascade_Bistable_Oscillatory) that neither the notebook
nor generate_data.py prints.

    python check_accuracy.py               all checks (several minutes; see DATA.md for the time)
    python check_accuracy.py --skip-slow   without check 6 (about 1-2 minutes)

Run it from any directory; it reads the shipped files next to this file and evaluates
MAPK_model.py, with the finite-difference helper of generate_data.py. Each line prints the
recomputed value; DATA.md quotes these values.

Checks:
 1. Study 2 (oscillatory): the stored mean_x3 and amplitude at the notebook's four rows
    (0, 128, 256, 384) against state-only solves at rtol = atol = 1e-8, the tolerance of the
    stored coupled solve.
 2. Study 3 (log design): the stored gradients of x3_high and x3_low at rows 0 and 306 against
    central differences of state-only solves at rtol = atol = 1e-12, relative steps 1e-5, 1e-6, 1e-7.
 3. Study 2 row 312 (near the edge of the oscillating region): the stored gradient of mean_x3
    against central differences (rtol = atol = 1e-12) with relative steps 1e-4, 1e-5, 1e-6.
 4. Study 2 rows 0-15: the stored gradients of mean_x3 and the amplitudes against coupled solves
    at rtol = atol = 1e-9 (the files use 1e-8).
 5. Study 1 (paper's design): samples with k1 in the S2 Table's range [0, 0.05] (the files use the
    code's [0, 0.1]).
 6. Study 2, all 512 samples: amplitude, period and the oscillating flag read on the 20 s grid and
    on a 1 s grid of the same state-only solve (rtol = atol = 1e-10): changes of class, and the
    amplitude change for the oscillating samples with periods under and over 2 minutes (slow).

Relative errors of gradients are 2-norms of the 7-vector. Deterministic. Imports numpy, scipy
(through MAPK_model) and the standard library.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MAPK_model as mm  # noqa: E402
import generate_data as gd  # noqa: E402

TIGHT = 1e-12


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def q_bistable(x0, tol=TIGHT):
    return lambda th: mm.solve(mm._full(th, mm.NOMINAL_BISTABLE), x0, [0.0, mm.T_END_BISTABLE], False, tol, tol)[0][2, -1]


def q_mean(tol=TIGHT):
    t = mm.osc_times()
    return lambda th: mm.solve(mm._full(th, mm.NOMINAL_OSCILLATORY), mm.X0_OSC, t, False, tol, tol)[0][2, mm.START_OSC:].mean()


def rel(g, ref):
    return np.linalg.norm(g - ref) / np.linalg.norm(ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-slow', action='store_true', help='skip check 6')
    args = ap.parse_args()
    t0 = time.perf_counter()
    Xb = load('bistable_inputs.csv')
    Xo, Fo, Go = load('oscillatory_inputs.csv'), load('oscillatory_outputs.csv'), load('oscillatory_gradients.csv')
    Xl, Gl = load('bistable_log_inputs.csv'), load('bistable_log_gradients.csv').reshape(-1, 4, 7)

    # 1. state-only solves at 1e-8 vs the stored coupled solve
    dm, da = [], []
    for i in (0, 128, 256, 384):
        o = mm.oscillatory(Xo[i], gradients=False)
        dm.append(abs(o['mean_x3'] - Fo[i, 0]) / abs(Fo[i, 0]))
        da.append(abs(o['amplitude'] - Fo[i, 1]) / max(abs(Fo[i, 1]), 1e-300) if Fo[i, 1] > 0 else abs(o['amplitude']))
    print(f'1. Study 2 rows 0, 128, 256, 384, state-only solve at 1e-8 vs files: max relative difference '
          f'mean_x3 {max(dm):.1e}, amplitude {max(da):.1e}')

    # 2. Study 3 rows 0 and 306, gradients vs central differences
    for i in (0, 306):
        out = []
        for h in (1e-5, 1e-6, 1e-7):
            e = max(rel(gd._fd(q_bistable(x0), Xl[i], h), Gl[i, k]) for k, x0 in ((1, mm.X0_HIGH), (3, mm.X0_LOW)))
            out.append(f'step {h:.0e}: {e:.1e}')
        share = np.sum((Gl[i, 1] * Xl[i] * np.log(100) / 2) ** 2) / np.sum((Gl[:, 1] * Xl * np.log(100) / 2) ** 2)
        print(f'2. Study 3 row {i}, x3_high and x3_low gradients vs central differences (1e-12 solves), max relative error: '
              + '; '.join(out) + f' (row {i} carries {share:.2f} of tr C for x3_high)')

    # 3. Study 2 row 312, mean_x3 gradient vs central differences
    out = [f'step {h:.0e}: {rel(gd._fd(q_mean(), Xo[312], h), Go[312]):.2g}' for h in (1e-4, 1e-5, 1e-6)]
    print(f'3. Study 2 row 312 (amplitude {Fo[312, 1]:.2f} nM), mean_x3 gradient vs central differences: ' + '; '.join(out))

    # 4. coupled solves at 1e-9 vs the files (1e-8), rows 0-15
    eg, ea = [], []
    for i in range(16):
        o = mm.oscillatory(Xo[i], rtol=1e-9, atol=1e-9)
        eg.append(rel(Go[i], o['grad_mean_x3']))
        ea.append(abs(o['amplitude'] - Fo[i, 1]))
    print(f'4. Study 2 rows 0-15, coupled solve at 1e-9 vs files (1e-8): max relative gradient difference '
          f'{max(eg):.1e}, max amplitude difference {max(ea):.1e} nM')

    # 5. k1 and the S2 Table's range
    n = int(np.sum(Xb[:, 0] > 0.05))
    print(f'5. Study 1: {n} of {len(Xb)} samples have k1 > 0.05, outside the S2 Table range [0, 0.05]')

    # 6. 20 s grid vs 1 s grid
    if not args.skip_slow:
        t1 = time.perf_counter()
        t = np.arange(0.0, mm.T_END_OSC + 0.5, 1.0)
        start1 = int(round(mm.START_OSC * mm.DT_OSC))           # t = 980 s on the 1 s grid
        A = np.zeros((len(Xo), 2))
        P = np.zeros((len(Xo), 2))
        for i, x in enumerate(Xo):
            y = mm.solve(mm._full(x, mm.NOMINAL_OSCILLATORY), mm.X0_OSC, t, False, 1e-10, 1e-10)[0][2]
            A[i, 0], P[i, 0] = mm.limit_cycle_features(y[::int(mm.DT_OSC)][mm.START_OSC:], mm.DT_OSC)
            A[i, 1], P[i, 1] = mm.limit_cycle_features(y[start1:], 1.0)
        osc20, osc1 = A[:, 0] > 0, A[:, 1] > 0
        print(f'6. Study 2, 1e-10 solves: oscillating on the 20 s grid {osc20.sum()} (files: {int(Fo[:, 3].sum())}), '
              f'on the 1 s grid {osc1.sum()}; 20 s only: rows {np.flatnonzero(osc20 & ~osc1).tolist()}, '
              f'1 s only: rows {np.flatnonzero(~osc20 & osc1).tolist()}')
        only1 = ~osc20 & osc1
        if only1.any():
            print(f'   oscillating on the 1 s grid only: period {60 * P[only1, 1].min():.0f}-{60 * P[only1, 1].max():.0f} s, '
                  f'amplitude {A[only1, 1].min():.2f}-{A[only1, 1].max():.2f} nM (1 s grid)')
        both = osc20 & osc1
        short = both & (P[:, 0] < 2.0)
        r = A[:, 1] / np.where(A[:, 0] > 0, A[:, 0], 1.0) - 1
        print(f'   oscillating on both grids with period < 2 min (20 s grid): {short.sum()} samples (of '
              f'{int(np.sum(osc20 & (P[:, 0] < 2.0)))} on the 20 s grid); amplitude on the 1 s grid larger by '
              f'{100 * r[short].min():.0f} % to {100 * r[short].max():.0f} % (median {100 * np.median(r[short]):.0f} %)')
        lng = both & (P[:, 0] >= 2.0)
        print(f'   period >= 2 min: {lng.sum()} samples; amplitude change median {100 * np.median(np.abs(r[lng])):.2f} %, '
              f'max {100 * np.abs(r[lng]).max():.1f} %; period change max {np.abs(P[lng, 1] - P[lng, 0]).max():.2f} min '
              f'({time.perf_counter() - t1:.0f} s)')
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
