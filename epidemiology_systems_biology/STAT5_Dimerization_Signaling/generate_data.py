"""Write inputs.csv, outputs.csv and gradients.csv for the STAT5 dimerization data set.

    python generate_data.py                          # 1000 samples, 2 processes, writes here
    python generate_data.py --n 20 --out /tmp/stat5 --check

Draws N samples of the 6 inputs xi = log10(theta), independent uniform on the study box
(STAT5_model.LOWER, STAT5_model.UPPER: the PEtab best fit +- 1 decade, clipped to the
PEtab bounds), with numpy's default_rng(SEED). For each sample it solves the model with
its forward sensitivity equations (`STAT5_model.qoi_and_grad`) and writes the 48
quantities of interest (time averages of the 3 observables over [0, 240] min, and the
3 observables at 15 measurement times) and their gradients with respect to xi.

The samples are the first N rows of one fixed stream, so `--n 20` gives the first 20
rows of the shipped files; `--workers` changes only the run time. `--check` compares
the new files with the ones next to this script.

Only numpy, scipy and the standard library are imported.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import STAT5_model as sm  # noqa: E402

SEED = 2019
N = 1000

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=N, help='number of samples (default 1000)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()

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
        G0 = read_csv(os.path.join(HERE, 'gradients.csv'))[:n].reshape(G.shape)
        print(f'inputs identical: {np.array_equal(X0, read_csv(os.path.join(args.out, "inputs.csv")))}; '
              f'max relative difference: outputs {np.max(np.abs(f - f0)/np.maximum(np.abs(f0), 1e-12)):.1e}, '
              f'gradients {np.max(np.linalg.norm(G - G0, axis=2)/np.linalg.norm(G0, axis=2)):.1e}')


if __name__ == '__main__':
    main()
