"""Write inputs.csv, outputs.csv and gradients.csv for the fire detection satellite.

    python generate_data.py                       # writes the three CSVs here
    python generate_data.py --check               # writes to a temporary directory and compares with the files here

Draws N = 2000 samples of the nine inputs from independent truncated normals
TN(MEAN, SD, LOWER, UPPER) (see fire_satellite_model.py) with
numpy.random.default_rng(SEED), SEED = 2012, solves the coupled system to a
relative fixed-point tolerance of 1e-13, and computes the gradients of the three
outputs with respect to the physical inputs by the complex step through the
fixed-point iteration. The whole run takes well under a second.

`--check` prints the largest differences from the shipped files and fails only on structural
mismatches (shapes, NaN pattern, inputs).
Only numpy, scipy and the standard library are imported.
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fire_satellite_model as fm  # noqa: E402

SEED = 2012
N = 2000
INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only

PROVENANCE = ('fire_satellite_model.py (Sankararaman & Mahadevan 2012, Sec. 6; fixed point on P_ACS '
              'to relative tolerance 1e-13); written by generate_data.py, seed 2012')


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def generate(n=N, seed=SEED):
    X = fm.sample(n, seed)
    F = fm.qoi(X)
    G = fm.grad(X)
    return X, F, G


def write(out, X, F, G):
    names = [f'{n} [{u}]' for n, u in zip(fm.NAMES, fm.UNITS)]
    sd = ', '.join(f'{n} TN({m:g}, {s:g}, {lo:g}, {hi:g})' for n, m, s, lo, hi in
                   zip(fm.NAMES, fm.MEAN, fm.SD, fm.LOWER, fm.UPPER))
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the 9 physical inputs, independent truncated normals '
                      f'TN(mean, sd, lower, upper): {sd}. Seed {SEED}.\n' + ','.join(names))
    np.savetxt(os.path.join(out, 'outputs.csv'), F, fmt='%.17g', delimiter=',',
               header=f'Outputs: total torque, total power, solar array area. {PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(fm.QOI_NAMES, fm.QOI_UNITS)))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), -1), fmt='%.17g',
               delimiter=',',
               header=f'Gradients w.r.t. the PHYSICAL inputs, column 9*i + j is d(output i)/d(input j); '
                      f'complex step through the fixed-point iteration. {PROVENANCE}\n'
                      + ','.join(f'd{q}/d{n}' for q in fm.QOI_NAMES for n in fm.NAMES))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            args.out = HERE
    if os.path.abspath(args.out) == HERE and args.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X, F, G = generate()
    write(args.out, X, F, G)
    print(f'wrote {args.out}: {len(X)} samples, {time.perf_counter() - t0:.2f} s')

    if args.check:
        new = {'inputs.csv': X, 'outputs.csv': F, 'gradients.csv': G.reshape(len(G), -1)}
        bad, worst = [], (-1.0, '')
        for n, a in new.items():
            b = read_csv(os.path.join(HERE, n))
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                bad.append(f'FAIL: {n}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
                continue
            if n == 'gradients.csv':   # per output: 2-norm of the gradient vector
                a, b = a.reshape(G.shape), b.reshape(G.shape)
                d, measure = np.nanmax(np.linalg.norm(a - b, axis=2) / np.linalg.norm(b, axis=2)), 'relative, 2-norm'
            else:
                d, measure = np.nanmax(np.abs(a - b) / np.abs(b)), 'relative'
            print(f'{n}: largest difference {d:.1e} ({measure})')
            if n == 'inputs.csv' and d > INPUT_RTOL:
                bad.append(f'FAIL: {n} differs from the shipped inputs by {d:.1e} > {INPUT_RTOL:g}')
            worst = max(worst, (d, f'{measure}, {n}'))
        print(f'largest difference from the shipped files: {worst[0]:.1e} ({worst[1]})')
        for line in bad:
            print(line)
        if bad:
            sys.exit(1)


if __name__ == '__main__':
    main()
