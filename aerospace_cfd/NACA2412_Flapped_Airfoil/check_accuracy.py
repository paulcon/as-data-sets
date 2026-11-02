"""Recompute the numbers of DATA.md (NACA2412_Flapped_Airfoil) that the notebook does not print.

    python check_accuracy.py                         checks 1-4, from the shipped files (under 1 s)
    python check_accuracy.py --davis-src FILE        also check 5: compare davis_training_98.csv with
                                                     the authors' training/training_100.csv

Run it from any directory; it reads the shipped CSVs next to this file. It does not run XFOIL.
Each line prints the recomputed value; DATA.md quotes these values.

Checks:
 1. Rows per file that converged in attempts 1-3 of the protocol (status 0, 1 or 2), rows that
    needed the retry ladder (3, 4) or failed (-1), and agreement of convergence_ladder.csv with the
    status columns (one row, with the same status, for every status 3, 4 or -1 point).
 2. Status -1 rows: nan in all six output columns, in every file.
 3. Row 15 of outputs.csv, the high-lift point (7.286 deg, 10.573 deg) that converges directly with
    ITER 100 (status 3), against the largest CL of the file.
 4. davis_training_98.csv: number of rows, significant digits of its values, and the span of its
    Reynolds numbers.
 5. (--davis-src FILE) the data lines of davis_training_98.csv against those of the authors' file
    (github.com/csdlpsu/uqchallenge2, commit 73eaec9, training/training_100.csv): identical text?

Deterministic. Imports numpy and the standard library.
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["outputs.csv", "extension_outputs.csv", "reference_points.csv", "davis_rerun_outputs.csv",
         "re_sensitivity.csv"]          # in the order of the file column of convergence_ladder.csv


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def data_lines(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--davis-src', help="the authors' training_100.csv (commit 73eaec9)")
    a = ap.parse_args()

    L = load('convergence_ladder.csv')
    print('1. rows converged in attempts 1-3 (status 0-2) | retry ladder (3, 4) | failed (-1) | '
          'convergence_ladder.csv rows:')
    all_ok = True
    for fid, fn in enumerate(FILES):
        st = load(fn)[:, -1].astype(int)
        lad = L[L[:, 0] == fid]
        want = {(i, s) for i, s in enumerate(st) if s >= 3 or s < 0}
        have = {(int(r[1]), int(r[2])) for r in lad}
        all_ok &= want == have
        print(f'   {fn:24s} {np.sum((st >= 0) & (st <= 2)):4d} | {np.sum(st >= 3):3d} | {np.sum(st < 0):2d} | '
              f'{len(lad):2d} (match: {want == have})')
    print(f'   convergence_ladder.csv lists exactly the status 3, 4 and -1 points of every file: {all_ok}')

    nan_ok = True
    for fn in FILES:
        A = load(fn)
        Y = A[:, -7:-1]
        f = A[:, -1] < 0
        nan_ok &= bool(np.all(np.isnan(Y[f]))) and bool(np.all(np.isfinite(Y[~f])))
    print(f'2. status -1 rows are nan in all six outputs, and every other row is finite, in every file: {nan_ok}')

    X, Y = load('inputs.csv'), load('outputs.csv')
    print(f'3. outputs.csv row 15: alpha {X[15, 0]:.3f} deg, flap {X[15, 1]:.3f} deg, status {int(Y[15, 6])}; '
          f'CL {Y[15, 0]:.4f}, CD {Y[15, 1]:.5f}, CM {Y[15, 3]:.4f}; largest CL in the file {np.nanmax(Y[:, 0]):.4f}')

    lines = data_lines(os.path.join(HERE, 'davis_training_98.csv'))
    D = load('davis_training_98.csv')
    sig = [len(v.lstrip('-').replace('.', '').lstrip('0')) for ln in lines for v in ln.split(',')[3:]]
    print(f"4. davis_training_98.csv: {len(lines)} rows; significant digits of CL, CD, CM: {min(sig)} to {max(sig)} "
          f"(XFOIL's polar file prints 4 or 5 decimals); Re from {D[:, 0].min():.0f} to {D[:, 0].max():.0f}")

    if a.davis_src:
        src = data_lines(a.davis_src)
        if src and src[0].startswith('re,'):
            src = src[1:]
        print(f"5. data lines identical to the authors' file ({len(src)} lines): {src == lines}")
    else:
        print("5. skipped (needs --davis-src, the authors' training/training_100.csv at commit 73eaec9)")


if __name__ == '__main__':
    main()
