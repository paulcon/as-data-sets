"""Regenerate outputs.csv and gradients.csv from the 1000 stored inputs in inputs.csv.

    python generate_data.py                      # all 1000 samples, 3 processes, writes here
    python generate_data.py --check              # samples 0-49 vs the shipped files
    python generate_data.py --check --full       # all 1000 samples vs the shipped files
    python generate_data.py --rows 0:20 --out DIR --check

For each row of inputs.csv (physical parameters) this computes T(t) at the 19
times in TIMES with `HIV_model.trajectory`, and the gradient of each T(t_i)
with respect to the normalized inputs x in [-1, 1]^27 by central finite
differences with step h = 3e-4 in x (55 trajectories per sample, each giving
all 19 times). At a time when the sample's T has reached the 0.1 floor, the
model output does not depend on the parameters, and the gradient is set to 0.

There is no randomness: the 1000 inputs were drawn without a seed in 2016 and
are stored data. Samples are independent, so `--workers` only changes the run
time, never the numbers. The 2026 run took 33 min on three processes (about
4 s of CPU per sample).

`--check` prints the largest differences from the files next to this script (relative: outputs
elementwise, gradients in the 2-norm of each time's row) and fails only on structural mismatches
(shapes, NaN pattern, a different set of zero-gradient rows).
Without --rows it checks the first 50 samples, or all of them with --full. `--check`
and `--rows` need `--out` outside this directory, so that the shipped files are not
overwritten; `--check` without `--out` writes into a temporary directory and removes it. inputs.csv is never
rewritten here (it cannot be redrawn); with `--out` elsewhere a copy is written there.

Only numpy, scipy and the standard library are imported.
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
import HIV_model as hm  # noqa: E402

#: Output times in days: the paper's 18 (Appendix B) plus 1700 (its worked example).
TIMES = np.array([24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 1000, 1600, 1700, 1800, 2000,
                  2200, 2400, 2800, 3400])

H = 3e-4  # central finite-difference step in normalized coordinates

QUICK_ROWS = 50      # samples checked by --check without --rows or --full

PROVENANCE = ('HIV_model.py (solve_ivp LSODA, rtol = atol = 1e-10, T floor 0.1); '
              'gradients w.r.t. normalized inputs x = (2p - (upper + lower))/(upper - lower), '
              'central differences with h = 3e-4 in x; written by generate_data.py')


def read_csv(path):
    """Read one of the three CSVs: '#' comment lines, then comma-separated numbers.

    The 2016 files were pandas CSVs with an integer header row and an index
    column; those are read too (header and index dropped), so that running this
    script on them converts inputs.csv losslessly (checked in `write`).
    """
    with open(path) as fh:
        legacy = fh.readline().startswith(',')
    if legacy:
        return np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)[:, 1:]
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def sample(p):
    """T at TIMES and its gradient w.r.t. normalized inputs, shape (19,) and (19, 27)."""
    half = (hm.UPPER - hm.LOWER)/2
    T, t_floor = hm.trajectory(p, TIMES)
    G = np.empty((len(TIMES), len(p)))
    for j in range(len(p)):
        dp = np.zeros(len(p))
        dp[j] = H*half[j]
        G[:, j] = (hm.trajectory(p + dp, TIMES)[0] - hm.trajectory(p - dp, TIMES)[0])/(2*H)
    G[TIMES > t_floor] = 0.
    return T, G


def write_inputs(out, X):
    names = list(hm.NAMES)
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.12g', delimiter=',',
               header='1000 samples of the 27 physical parameters, uniform on nominal +- 2.5 % '
                      '(drawn without a seed in 2016; stored data)\n' + ','.join(names))
    if not np.array_equal(read_csv(os.path.join(out, 'inputs.csv')), X):
        raise RuntimeError('inputs.csv did not round-trip exactly')


def write(out, X, f, G):
    names = list(hm.NAMES)
    t = [str(v) for v in TIMES]
    if os.path.abspath(out) == HERE:       # never rewrite the 2016 inputs: no seed, cannot be redrawn
        X0 = read_csv(os.path.join(out, 'inputs.csv'))
        if not np.array_equal(X0[:len(X)], X):
            raise RuntimeError('inputs differ from inputs.csv')
    else:
        write_inputs(out, X)
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.12e', delimiter=',',
               header=f'T(t) in cells/mm^3 at t = {", ".join(t)} days. {PROVENANCE}\n'
                      + ','.join(f'T({s})' for s in t))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), -1), fmt='%.10e',
               delimiter=',',
               header=f'dT(t)/dx_j, column 27*i + j for time i and input j. {PROVENANCE}\n'
                      + ','.join(f'dT({s})/d{n}' for s in t for n in names))


def compare(f, G, f0, G0):
    """Print the largest differences of f, G from the shipped f0, G0; return False on a structural mismatch."""
    if f.shape != f0.shape or G.shape != G0.shape:
        print(f'shapes {f.shape}, {G.shape} differ from the shipped {f0.shape}, {G0.shape}: FAIL')
        return False
    ok = True
    if not (np.array_equal(np.isnan(f), np.isnan(f0)) and np.array_equal(np.isnan(G), np.isnan(G0))):
        print('NaN pattern differs from the shipped files: FAIL')
        ok = False
    nz = np.linalg.norm(G0, axis=2) > 0
    if not np.array_equal(np.linalg.norm(G, axis=2) == 0, ~nz):
        print('zero-gradient rows differ from the shipped file: FAIL')
        ok = False
    rf = np.nanmax(np.abs(f - f0)/np.abs(f0), initial=0.)
    rg = np.nanmax(np.linalg.norm(G - G0, axis=2)[nz]/np.linalg.norm(G0, axis=2)[nz], initial=0.)
    print(f'largest relative difference from the shipped files: outputs {rf:.1e} (elementwise), '
          f'gradients {rg:.1e} (2-norm of each nonzero row)')
    r, where = max((rf, 'outputs.csv'), (rg, 'gradients.csv'))
    print(f'largest difference from the shipped files: {r:.1e} (relative, {where})')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', default=None,
                    help=f'row slice of inputs.csv, e.g. 0:20 (default all; with --check, 0:{QUICK_ROWS})')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    ap.add_argument('--check', action='store_true',
                    help='compare with the CSVs in this directory')
    ap.add_argument('--full', action='store_true', help='with --check and no --rows: check all samples')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            args.out = HERE
    if args.rows is None:
        args.rows = f'0:{QUICK_ROWS}' if args.check and not args.full else ':'
    if os.path.abspath(args.out) == HERE and (args.check or args.rows != ':'):
        ap.error('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')

    X = read_csv(os.path.join(HERE, 'inputs.csv'))
    a, b = (int(s) if s else None for s in args.rows.split(':'))
    X = X[a:b]
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    f = np.empty((len(X), len(TIMES)))
    G = np.empty((len(X), len(TIMES), X.shape[1]))
    with mp.Pool(args.workers) as pool:
        for i, (T, g) in enumerate(pool.imap(sample, X, chunksize=4)):
            f[i], G[i] = T, g
            if (i + 1) % 50 == 0 or i + 1 == len(X):
                print(f'{i + 1}/{len(X)} samples, {time.perf_counter() - t0:.0f} s', flush=True)
    write(args.out, X, f, G)
    print(f'wrote {args.out} (rows {args.rows}): {time.perf_counter() - t0:.0f} s, '
          f'{(f[:, -1] <= hm.T_FLOOR).sum()} samples at the floor at t = {TIMES[-1]}', flush=True)

    if args.check:
        f0 = read_csv(os.path.join(HERE, 'outputs.csv'))[a:b]
        G0 = read_csv(os.path.join(HERE, 'gradients.csv'))[a:b]
        G0 = G0.reshape(len(G0), *G.shape[1:])
        if not compare(f, G, f0, G0):
            sys.exit(1)


if __name__ == '__main__':
    main()
