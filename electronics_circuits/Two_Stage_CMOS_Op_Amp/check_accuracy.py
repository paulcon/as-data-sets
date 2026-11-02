"""Recompute the accuracy numbers quoted in DATA.md for Two_Stage_CMOS_Op_Amp (needs ngspice).

    python check_accuracy.py                 # all checks, about 470 ngspice runs, 2 worker processes
    python check_accuracy.py --workers 4

Prints, beside the values DATA.md quotes:

1. Held settings, on gain200 rows 0-9: the change of the four outputs when ngspice's reltol is
   AutoCkt's 1e-3 instead of 1e-9, and when the AC sweep has 10 points per decade instead of 100
   (relative change of gain, ugbw and i_supply; absolute change of pm in degrees).
2. Sweep density, same rows: 1000 points per decade instead of 100 (ugbw relative, pm in degrees).
3. Gradients by an independent formula: central differences in the physical inputs,
   x_j (1 +- d) with d = 1e-3 and 3e-4, at gain200 rows 59, 634, 785 and box rows 862, 250, 336,
   against the stored gradients (relative 2-norm error of each output's 7-vector, scaled by x_j).
4. Smoothness of log10 f_u across the sweep grid at gain200 row 0: 121 designs along Cc, and 121
   along mn3, spanning about two grid intervals of the AC sweep (0.02 decades of f_u, from the
   stored gradient); standard deviation of the second differences of log10 f_u against the mean
   absolute first difference (a kink at a grid point would show as a second-difference spike).

The gradient step study (steps 1e-3 and 1e-5 against 1e-4) is `generate_data.py --fd-check`.
Needs numpy, scipy and the ngspice program on PATH (or NGSPICE=/path/to/ngspice); the shipped
files were written with ngspice-42. Without ngspice the script stops with an error. Deterministic
(fixed rows); it reads the files next to it and writes nothing.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opamp_spice as op  # noqa: E402
import generate_data as gd  # noqa: E402

HELD_ROWS = list(range(10))                              # gain200 rows for checks 1 and 2
GRAD_ROWS = [('gain200_', 59), ('gain200_', 634), ('gain200_', 785), ('', 862), ('', 250), ('', 336)]
SWEEP_ROW = 0                                            # gain200 row for check 4
N_SWEEP = 121


def load(prefix):
    ld = lambda n: np.loadtxt(os.path.join(HERE, prefix + n), delimiter=',', ndmin=2)  # noqa: E731
    return ld('inputs.csv'), ld('outputs.csv'), ld('gradients.csv').reshape(-1, 4, 7)


def _sim(args):
    x, held = args
    return op.qoi_vector(x, strict=False, **held)


def changes(F1, F0):
    """max relative change of gain, ugbw, i_supply and max |change| of pm (deg) over rows."""
    r = np.abs(F1 - F0) / np.abs(F0)
    return (np.nanmax(r[:, [0, 1, 3]]), np.nanmax(np.abs(F1[:, 2] - F0[:, 2])),
            np.nanmax(r[:, 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    gd.require_ngspice()                              # exits with a message if ngspice is missing
    t0 = time.perf_counter()
    print(op.ngspice_version())
    Xa, Fa, Ga = load('gain200_')
    Xb, Fb, Gb = load('')
    with Pool(a.workers) as pool:
        run = lambda X, **held: np.array(pool.map(_sim, [(x, held) for x in X]))  # noqa: E731

        X10 = Xa[HELD_ROWS]
        F0 = run(X10)
        print(f'\n0. gain200 rows 0-9 re-simulated with the held settings: max relative difference from '
              f'the stored outputs {np.nanmax(np.abs(F0 - Fa[HELD_ROWS]) / np.abs(Fa[HELD_ROWS])):.1e}')
        r_all, d_pm, _ = changes(run(X10, reltol=1e-3), F0)
        print(f'1. reltol 1e-3 instead of 1e-9: outputs change by at most {r_all:.1e} relative, pm by {d_pm:.1e} deg '
              f'(DATA.md: 1.2e-5 relative, 1.9e-4 deg)')
        _, d_pm, r_u = changes(run(X10, points_per_decade=10), F0)
        print(f'   10 points per decade instead of 100: ugbw changes by {r_u:.1e} relative, pm by {d_pm:.1e} deg '
              f'(DATA.md: 3.6e-6, 4.6e-4 deg)')
        _, d_pm, r_u = changes(run(X10, points_per_decade=1000), F0)
        print(f'2. 1000 points per decade instead of 100: ugbw changes by {r_u:.1e} relative, pm by {d_pm:.1e} deg '
              f'(DATA.md: 5e-10, 3e-8 deg)')

        print('\n3. central differences in physical units against the stored gradients '
              '(relative 2-norm error per output; gain, ugbw, pm, i_supply)')
        errs = []
        for prefix, i in GRAD_ROWS:
            X, G = (Xa, Ga) if prefix else (Xb, Gb)
            x = X[i]
            for d in (1e-3, 3e-4):
                pts = []
                for j in range(7):
                    for s in (1, -1):
                        xp = x.copy(); xp[j] *= 1 + s * d
                        pts.append(xp)
                F = run(pts).reshape(7, 2, 4)
                Gn = ((F[:, 0] - F[:, 1]) / (2 * d * x[:, None])).T          # (4, 7) physical
                e = np.linalg.norm((Gn - G[i]) * x, axis=1) / np.linalg.norm(G[i] * x, axis=1)
                errs += list(e[np.isfinite(e)])
                print(f'   {prefix or "box_"}row {i:4d}, d = {d:.0e}: ' + ', '.join(f'{v:.1e}' for v in e))
        print(f'   range over rows, steps and outputs: {min(errs):.1e} to {max(errs):.1e} '
              f'(DATA.md: 4.6e-7 to 6.9e-5)')

        print(f'\n4. log10 f_u along {N_SWEEP} designs at gain200 row {SWEEP_ROW}, spanning 0.02 decades of f_u')
        x = Xa[SWEEP_ROW]
        for name in ('cc_pF', 'mn3'):
            j = op.NAMES.index(name)
            slope = Ga[SWEEP_ROW, 1, j] * x[j] / Fa[SWEEP_ROW, 1]           # d ln f_u / d ln x_j
            span = 0.02 * np.log(10) / abs(slope)                          # in ln x_j
            pts = np.repeat(x[None], N_SWEEP, axis=0)
            pts[:, j] = x[j] * np.exp(np.linspace(-span / 2, span / 2, N_SWEEP))
            lf = np.log10(run(pts)[:, 1])
            d1, d2 = np.diff(lf), np.diff(lf, 2)
            print(f'   {name}: x from {pts[0, j]:.4g} to {pts[-1, j]:.4g}; second differences sd {d2.std():.1e}, '
                  f'max |.| {np.abs(d2).max():.1e}; mean |first difference| {np.abs(d1).mean():.1e} '
                  f'(DATA.md: Cc 2e-11 against 4e-5, mn3 7e-10 against 1.6e-4)')
    print(f'\nrun time {time.perf_counter() - t0:.0f} s on {a.workers} workers')


if __name__ == '__main__':
    main()
