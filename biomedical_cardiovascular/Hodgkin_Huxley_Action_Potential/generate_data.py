"""Write inputs.csv, outputs.csv and gradients.csv for the Hodgkin-Huxley data set.

    python generate_data.py                       # 1000 samples, 2 processes, writes here
    python generate_data.py --n 20 --out /tmp/hh --check

Draws N samples of the 11 physical inputs, independent uniform on nominal +- 20 %
(HH_model.LOWER, HH_model.UPPER), with numpy's default_rng(SEED). For each sample it
solves the model together with its forward sensitivity equations
(`HH_model.qoi_and_grad`) and writes the 21 quantities of interest (peak potential,
peak time, v at 19 times) and their gradients with respect to the physical inputs.

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
import HH_model as hm  # noqa: E402

SEED = 2015
N = 1000

PROVENANCE = ('HH_model.py (Eq. (1) of Torres Valderrama et al. 2015; I = 150, t in [0, 14] ms; '
              'solve_ivp DOP853, rtol = 1e-10, atol = 1e-12, forward sensitivities); '
              'written by generate_data.py, seed 2015')


def read_csv(path):
    """Read one of the CSVs: two '#' lines, then comma-separated numbers."""
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def draw(n, seed=SEED):
    """The first n input samples (physical units), uniform on [LOWER, UPPER]."""
    u = np.random.default_rng(seed).random((n, len(hm.NAMES)))
    return hm.LOWER + (hm.UPPER - hm.LOWER)*u


def write(out, X, f, G):
    names = list(hm.NAMES)
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the 11 physical inputs, independent uniform on '
                      f'nominal +- 20 % (numpy default_rng({SEED}))\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(names, hm.UNITS)))
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.12e', delimiter=',',
               header=f'peak membrane potential (second local maximum of v, mV), its time (ms), '
                      f'and v (mV) at t = 5, 5.5, ..., 14 ms. {PROVENANCE}\n'
                      'vpeak [mV],tpeak [ms],' + ','.join(f'{q} [mV]' for q in hm.QOI_NAMES[2:]))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), -1), fmt='%.10e',
               delimiter=',',
               header=f'gradients w.r.t. the PHYSICAL inputs: column 11*i + j is d(QoI i)/d(input j), '
                      f'QoIs in the order of outputs.csv. {PROVENANCE}\n'
                      + ','.join(f'd{q}/d{n}' for q in hm.QOI_NAMES for n in names))


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
    f = np.empty((len(X), len(hm.QOI_NAMES)))
    G = np.empty((len(X), len(hm.QOI_NAMES), len(hm.NAMES)))
    with mp.Pool(args.workers) as pool:
        for i, (fi, gi) in enumerate(pool.imap(hm.qoi_and_grad, X, chunksize=10)):
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
              f'max relative difference: outputs {np.max(np.abs(f - f0)/np.abs(f0)):.1e}, '
              f'gradients {np.max(np.linalg.norm(G - G0, axis=2)/np.linalg.norm(G0, axis=2)):.1e}')


if __name__ == '__main__':
    main()
