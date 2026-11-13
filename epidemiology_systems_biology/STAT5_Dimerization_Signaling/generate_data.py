"""Write inputs.csv, outputs.csv and gradients.csv for the STAT5 dimerization data set.

    python generate_data.py                          # 1000 samples, 2 processes, writes here
    python generate_data.py --check                  # all 1000 rows, compared with the files here
    python generate_data.py --n 20 --out DIR --check

Draws N samples of the 6 inputs xi = log10(theta), independent uniform on the study box
(STAT5_model.LOWER, STAT5_model.UPPER: the PEtab best fit +- 1 decade, clipped to the
PEtab bounds), with numpy's default_rng(SEED). For each sample it solves the model with
its forward sensitivity equations (`STAT5_model.qoi_and_grad`) and writes the 48
quantities of interest (time averages of the 3 observables over [0, 240] min, and the
3 observables at 15 measurement times) and their gradients with respect to xi.

The samples are the first N rows of one fixed stream, so `--n 20` gives the first 20
rows of the shipped files; `--workers` changes only the run time. `--check` prints the
largest differences from the files next to this script (relative, each difference scaled by the
largest shipped value of the same output, or gradient norm) and fails only on structural mismatches
(shapes, NaN pattern, inputs differing by more than INPUT_RTOL). Without --out it writes into a
temporary directory and removes it.

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
import STAT5_model as sm  # noqa: E402

SEED = 2019
N = 1000
INPUT_RTOL = 1e-8  # --check fails if the regenerated inputs differ by more than this (relative)

PROVENANCE = ('STAT5_model.py (Boehm_JProteomeRes2014 benchmark of Hass et al. 2019; '
              'solve_ivp Radau, rtol = atol = 1e-10, forward sensitivities); '
              'written by generate_data.py, seed 2019')


def read_csv(path):
    """Read one of the CSVs: two '#' lines, then comma-separated numbers."""
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def draw(n, seed=SEED):
    """The first n input samples, log10(theta), uniform on [LOWER, UPPER]."""
    u = np.random.default_rng(seed).random((n, len(sm.NAMES)))
    return sm.LOWER + (sm.UPPER - sm.LOWER)*u


def write(out, X, f, G):
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the 6 inputs log10(theta), independent uniform on the PEtab '
                      f'best fit +- 1 decade clipped to [-5, 5] (numpy default_rng({SEED}))\n'
                      + ','.join(f'log10_{n} [log10({u})]' for n, u in zip(sm.NAMES, sm.UNITS)))
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.12e', delimiter=',',
               header='time averages over [0, 240] min of pSTAT5A_rel, pSTAT5B_rel, rSTAT5A_rel (%), then '
                      f'each observable (%) at t = 2.5, 5, ..., 240 min. {PROVENANCE}\n'
                      + ','.join(f'{q} [%]' for q in sm.QOI_NAMES))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), -1), fmt='%.10e',
               delimiter=',',
               header='gradients w.r.t. the inputs log10(theta): column 6*i + j is d(QoI i)/d(log10 theta_j), '
                      f'QoIs in the order of outputs.csv. {PROVENANCE}\n'
                      + ','.join(f'd{q}/dlog10_{n}' for q in sm.QOI_NAMES for n in sm.NAMES))


def compare(X, f, G, X0, f0, G0):
    """Print the largest differences of the new rows from the shipped ones; return False on a structural mismatch.

    G is (rows, QoIs, inputs); G0 is the shipped gradients.csv rows, one row per sample."""
    if X.shape != X0.shape or f.shape != f0.shape or G.reshape(len(G), -1).shape != G0.shape:
        print(f'shapes {X.shape}, {f.shape}, {G.reshape(len(G), -1).shape} differ from the shipped {X0.shape}, '
              f'{f0.shape}, {G0.shape}: FAIL')
        return False
    G0 = G0.reshape(G.shape)
    ok = True
    if not (np.array_equal(np.isnan(f), np.isnan(f0)) and np.array_equal(np.isnan(G), np.isnan(G0))):
        print('NaN pattern differs from the shipped files: FAIL')
        ok = False
    rx = np.max(np.abs(X - X0)/np.maximum(np.abs(X0), 1e-300))
    print(f'largest relative difference (elementwise): inputs {rx:.1e}, '
          f'outputs {np.nanmax(np.abs(f - f0)/np.maximum(np.abs(f0), 1e-12)):.1e}, '
          f'gradients {np.nanmax(np.linalg.norm(G - G0, axis=2)/np.linalg.norm(G0, axis=2)):.1e} (2-norm of each row)')
    rel = {'outputs.csv': np.nanmax(np.abs(f - f0)/np.nanmax(np.abs(f0), axis=0)),
           'gradients.csv': np.nanmax(np.linalg.norm(G - G0, axis=2)/np.nanmax(np.linalg.norm(G0, axis=2), axis=0))}
    print(f'largest difference scaled by the largest value of each output: outputs {rel["outputs.csv"]:.1e}, '
          f'gradient norms {rel["gradients.csv"]:.1e}')
    if rx > INPUT_RTOL:
        print(f'inputs differ by more than {INPUT_RTOL:g}: FAIL')
        ok = False
    rel['inputs.csv'] = rx
    n = max(rel, key=rel.get)
    print(f'largest difference from the shipped files: {rel[n]:.1e} (relative, {n})')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=N, help='number of samples (default 1000)')
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
    if os.path.abspath(args.out) == HERE and (args.check or args.n != N):
        ap.error('--check and --n need --out DIR outside this directory, so that the shipped files are not overwritten')

    X = draw(args.n)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    f = np.empty((len(X), len(sm.QOI_NAMES)))
    G = np.empty((len(X), len(sm.QOI_NAMES), len(sm.NAMES)))
    with mp.Pool(args.workers) as pool:
        for i, (fi, gi) in enumerate(pool.imap(sm.qoi_and_grad, X, chunksize=5)):
            f[i], G[i] = fi, gi
            if (i + 1) % 100 == 0 or i + 1 == len(X):
                print(f'{i + 1}/{len(X)} samples, {time.perf_counter() - t0:.0f} s', flush=True)
    write(args.out, X, f, G)
    print(f'wrote {args.out}: {time.perf_counter() - t0:.0f} s', flush=True)

    if args.check:
        n = len(X)
        X0 = read_csv(os.path.join(HERE, 'inputs.csv'))[:n]
        f0 = read_csv(os.path.join(HERE, 'outputs.csv'))[:n]
        G0 = read_csv(os.path.join(HERE, 'gradients.csv'))[:n]
        if not compare(read_csv(os.path.join(args.out, 'inputs.csv')), f, G, X0, f0, G0):
            sys.exit(1)


if __name__ == '__main__':
    main()
