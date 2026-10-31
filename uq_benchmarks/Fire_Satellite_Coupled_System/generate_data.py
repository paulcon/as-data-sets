"""Write inputs.csv, outputs.csv and gradients.csv for the fire detection satellite.

    python generate_data.py                       # writes the three CSVs here
    python generate_data.py --out /tmp/fs --check # writes elsewhere and compares with the files here

Draws N = 2000 samples of the nine inputs from independent truncated normals
TN(MEAN, SD, LOWER, UPPER) (see fire_satellite_model.py) with
numpy.random.default_rng(SEED), SEED = 2012, solves the coupled system to a
relative fixed-point tolerance of 1e-13, and computes the gradients of the three
outputs with respect to the physical inputs by the complex step through the
fixed-point iteration. The whole run takes well under a second.

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fire_satellite_model as fm  # noqa: E402

SEED = 2012
N = 2000

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
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X, F, G = generate()
    write(args.out, X, F, G)
    print(f'wrote {args.out}: {len(X)} samples, {time.perf_counter() - t0:.2f} s')

    if args.check:
        X0 = read_csv(os.path.join(HERE, 'inputs.csv'))
        F0 = read_csv(os.path.join(HERE, 'outputs.csv'))
        G0 = read_csv(os.path.join(HERE, 'gradients.csv')).reshape(G.shape)
        print('max relative difference from the shipped files: inputs '
              f'{np.max(np.abs(X - X0) / np.abs(X0)):.1e}, outputs {np.max(np.abs(F - F0) / np.abs(F0)):.1e}, '
              f'gradients {np.max(np.linalg.norm(G - G0, axis=2) / np.linalg.norm(G0, axis=2)):.1e}')


if __name__ == '__main__':
    main()
