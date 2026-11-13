"""Draw the input samples and write inputs.csv, outputs.csv and gradients.csv.

    python generate_data.py                       # 500 samples, 2 processes, writes here
    python generate_data.py --check               # all 500 samples vs the shipped files
    python generate_data.py --rows 0:10 --out DIR --check

The six FLORIS wake parameters (alpha, beta, ka, kb, ad, bd) are drawn independently and
uniformly on the bounds of the paper's Table 2 with numpy.random.default_rng(SEED). For each
sample, `lillgrund_model.farm_power_and_grad` gives the total power of the 48 Lillgrund turbines
at 8 m/s for the six alignment directions of the paper's Table 1 (one per turbine spacing), and
its gradient with respect to the physical parameters by central differences with step
1e-4*(upper - lower)/2, with FLORIS's discrete choices frozen at the sample (13 farm solves per
sample and direction).

The inputs come from the seed and the outputs from the model, so a rerun reproduces all three
files; `--workers` changes only the run time. `--check` prints the largest differences of the rows
written from the shipped files (relative: outputs elementwise, gradients 2-norm of each row) and fails
only on structural mismatches (shapes, NaN pattern, inputs differing by more than INPUT_RTOL); without
--out it writes into a temporary directory and removes it.

Only numpy and the standard library are imported (plus lillgrund_model.py).
"""
import argparse
import atexit
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lillgrund_model as lm  # noqa: E402

SEED = 2021
M = 500
WS = 8.0
H = 1e-4      # central-difference step, as a fraction of the half-range of each parameter
INPUT_RTOL = 1e-8  # --check fails if the regenerated inputs differ by more than this (relative)

PROVENANCE = ('lillgrund_model.py (numpy port of FLORIS 2.0.1 GCH, zero yaw, 48 turbines, '
              'U = 8 m/s, TI = 0.06, shear 0.12); written by generate_data.py')


def draw():
    rng = np.random.default_rng(SEED)
    return rng.uniform(lm.LOWER, lm.UPPER, size=(M, len(lm.NAMES)))


def sample(x):
    f = np.empty(len(lm.DIRECTIONS))
    G = np.empty((len(lm.DIRECTIONS), len(lm.NAMES)))
    for i, wd in enumerate(lm.DIRECTIONS):
        p, g = lm.farm_power_and_grad(x, wd, WS, h=H)
        f[i], G[i] = p/1e6, g/1e6
    return f, G


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, f, G):
    cols = [f'P({wd:g} deg; {s:g}D)' for wd, s in zip(lm.DIRECTIONS, lm.SPACINGS)]
    units = [f'{n} ({u})' for n, u in zip(lm.NAMES, lm.UNITS)]
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the six FLORIS wake parameters, independent uniform on '
                      f'the bounds of Table 2 of van Beek et al. 2021 (numpy default_rng({SEED}))\n'
                      + ','.join(units))
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.15e', delimiter=',',
               header=f'farm power in MW at 8 m/s for wind from the six alignment directions of '
                      f'Table 1. {PROVENANCE}\n' + ','.join(c + ' MW' for c in cols))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), -1), fmt='%.12e',
               delimiter=',',
               header=f'dP/dx_j in MW per unit of x_j (physical inputs), column 6*i + j for '
                      f'direction i and input j; central differences, step 1e-4 of the half-range, '
                      f'discrete FLORIS choices frozen. {PROVENANCE}\n'
                      + ','.join(f'dP({wd:g})/d{n}' for wd in lm.DIRECTIONS for n in lm.NAMES))


def compare(X, f, G, X0, f0, G0):
    """Print the largest differences of the new rows from the shipped ones; return False on a structural mismatch."""
    G = G.reshape(len(G), -1)
    if X.shape != X0.shape or f.shape != f0.shape or G.shape != G0.shape:
        print(f'shapes {X.shape}, {f.shape}, {G.shape} differ from the shipped {X0.shape}, {f0.shape}, '
              f'{G0.shape}: FAIL')
        return False
    ok = True
    if not (np.array_equal(np.isnan(f), np.isnan(f0)) and np.array_equal(np.isnan(G), np.isnan(G0))):
        print('NaN pattern differs from the shipped files: FAIL')
        ok = False
    rel = {'inputs.csv': np.max(np.abs(X - X0)/np.abs(X0)),
           'outputs.csv': np.nanmax(np.abs(f - f0)/np.abs(f0)),
           'gradients.csv': np.nanmax(np.linalg.norm(G - G0, axis=1)/np.linalg.norm(G0, axis=1))}
    print(f'largest relative difference from the shipped files: inputs {rel["inputs.csv"]:.1e} (elementwise), '
          f'outputs {rel["outputs.csv"]:.1e} (elementwise), gradients {rel["gradients.csv"]:.1e} '
          '(2-norm of each row)')
    if rel['inputs.csv'] > INPUT_RTOL:
        print(f'inputs differ by more than {INPUT_RTOL:g}: FAIL')
        ok = False
    n = max(rel, key=rel.get)
    print(f'largest difference from the shipped files: {rel[n]:.1e} (relative, {n})')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', default=':', help='row slice of the samples, e.g. 0:10 (default all)')
    ap.add_argument('--workers', type=int, default=2)
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
    if os.path.abspath(args.out) == HERE and (args.check or args.rows != ':'):
        ap.error('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')

    a, b = (int(s) if s else None for s in args.rows.split(':'))
    X = draw()[a:b]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    f = np.empty((len(X), len(lm.DIRECTIONS)))
    G = np.empty((len(X), len(lm.DIRECTIONS), len(lm.NAMES)))
    with mp.Pool(args.workers) as pool:
        for i, (fi, gi) in enumerate(pool.imap(sample, X, chunksize=5)):
            f[i], G[i] = fi, gi
            if (i + 1) % 50 == 0 or i + 1 == len(X):
                print(f'{i + 1}/{len(X)} samples, {time.perf_counter() - t0:.0f} s', flush=True)
    write(args.out, X, f, G)
    print(f'wrote {args.out} (rows {args.rows}): {time.perf_counter() - t0:.0f} s', flush=True)

    if args.check:
        X0 = read_csv(os.path.join(HERE, 'inputs.csv'))[a:b]
        f0 = read_csv(os.path.join(HERE, 'outputs.csv'))[a:b]
        G0 = read_csv(os.path.join(HERE, 'gradients.csv'))[a:b]
        if not compare(X, f, G, X0, f0, G0):
            sys.exit(1)


if __name__ == '__main__':
    main()
