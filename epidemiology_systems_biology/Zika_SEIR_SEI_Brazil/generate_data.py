"""Regenerate inputs.csv, outputs.csv and gradients.csv for Zika_SEIR_SEI_Brazil.

    python generate_data.py                       # 1000 samples, seed 2018, writes here
    python generate_data.py --n 50 --out /tmp/zika --check
    python generate_data.py --fd-check 20         # gradients vs. independent central differences

Draws M = 1000 points uniformly on the box zika_model.LOWER..UPPER (10 inputs,
physical units) with numpy's default_rng(2018), and for each point solves the
SEIR-SEI model with its forward sensitivities (zika_model.qoi_and_grad):

- outputs.csv: C(364) in people and the misfit J (Eq. 7) in people^2;
- gradients.csv: dC(364)/dp and dJ/dp with respect to the physical inputs p.

`--check` compares newly written files with the ones next to this script.
`--fd-check K` compares the sensitivity gradients at the first K stored rows
with central differences of an independent solver (Radau, rtol = 1e-12,
atol = 1e-16, step 1e-4 of each input's half-range), and prints the largest
relative error (2-norm of the 10-vector).

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zika_model as zm  # noqa: E402

SEED = 2018
M = 1000
FD_STEP = 1e-4   # central-difference step, as a fraction of each input's half-range

PROVENANCE = ('zika_model.py (scipy solve_ivp DOP853, rtol 1e-11, atol 1e-15, forward sensitivities); '
              'written by generate_data.py, seed 2018')


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def draw(n, seed=SEED):
    rng = np.random.default_rng(seed)
    return zm.LOWER + rng.random((n, len(zm.NAMES)))*(zm.UPPER - zm.LOWER)


def compute(X):
    f = np.empty((len(X), 2))
    G = np.empty((len(X), 2*len(zm.NAMES)))
    for i, x in enumerate(X):
        C, dC, J, dJ = zm.qoi_and_grad(x)
        f[i] = C, J
        G[i] = np.concatenate([dC, dJ])
    return f, G


def write(out, X, f, G):
    names = list(zm.NAMES)
    units = [f'{n} ({u})' for n, u in zip(zm.NAMES, zm.UNITS)]
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} points, independent uniform on the box in zika_model.py '
                      f'(LOWER, UPPER), numpy default_rng({SEED})\n' + ','.join(units))
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.15e', delimiter=',',
               header=f'Cumulative cases C at EW 52 (t = 364 d) and misfit J to the 2016 weekly '
                      f'case data (Eq. 7). {PROVENANCE}\nC52 (people),J (people^2)')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.15e', delimiter=',',
               header=f'Gradients w.r.t. the physical inputs: columns 0-9 dC52/dp, columns 10-19 '
                      f'dJ/dp (units: output units per input unit). {PROVENANCE}\n'
                      + ','.join([f'dC52/d{n}' for n in names] + [f'dJ/d{n}' for n in names]))


def fd_gradients(x, y):
    """Central differences of (C52, J) with an independent Radau solve, physical inputs."""
    h = FD_STEP*(zm.UPPER - zm.LOWER)/2
    g = np.empty((2, len(x)))
    for j in range(len(x)):
        e = np.zeros(len(x))
        e[j] = h[j]
        if x[j] - h[j] < 0:   # an initial condition within one step of 0: second-order one-sided
            q0, q1, q2 = (np.array(_qoi_radau(x + k*e, y)) for k in range(3))
            g[:, j] = (-3*q0 + 4*q1 - q2)/(2*h[j])
        else:
            g[:, j] = (np.array(_qoi_radau(x + e, y)) - np.array(_qoi_radau(x - e, y)))/(2*h[j])
    return g


def _qoi_radau(x, y):
    C = zm.trajectory(x, rtol=1e-12, atol=1e-16, method='Radau')[:, 7]
    res = y - zm.new_cases(C)
    return C[-1], res @ res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=M, help='number of points (first n of the seed-2018 draw)')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--fd-check', type=int, default=0, metavar='K',
                    help='check gradients at the first K stored rows against central differences')
    args = ap.parse_args()

    if args.fd_check:
        X = read_csv(os.path.join(HERE, 'inputs.csv'))[:args.fd_check]
        G = read_csv(os.path.join(HERE, 'gradients.csv'))[:args.fd_check]
        y = zm.load_cases()
        t0 = time.perf_counter()
        err = np.empty((len(X), 2))
        for i, x in enumerate(X):
            g = fd_gradients(x, y)
            for k in range(2):
                gs = G[i, 10*k:10*(k + 1)]
                err[i, k] = np.linalg.norm(gs - g[k])/np.linalg.norm(g[k])
        print(f'{len(X)} rows, {time.perf_counter() - t0:.0f} s: max relative gradient error '
              f'C52 {err[:, 0].max():.1e} (median {np.median(err[:, 0]):.1e}), '
              f'J {err[:, 1].max():.1e} (median {np.median(err[:, 1]):.1e})')
        return

    t0 = time.perf_counter()
    X = draw(M)[:args.n]
    f, G = compute(X)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, f, G)
    print(f'wrote {len(X)} points to {args.out} in {time.perf_counter() - t0:.1f} s')

    if args.check:
        n = len(X)
        X0 = read_csv(os.path.join(HERE, 'inputs.csv'))[:n]
        f0 = read_csv(os.path.join(HERE, 'outputs.csv'))[:n]
        G0 = read_csv(os.path.join(HERE, 'gradients.csv'))[:n]
        print(f'inputs equal: {np.array_equal(X, X0)}; max relative difference: outputs '
              f'{np.max(np.abs(f - f0)/np.abs(f0)):.1e}, gradients '
              f'{np.max(np.linalg.norm(G - G0, axis=1)/np.linalg.norm(G0, axis=1)):.1e}')


if __name__ == '__main__':
    main()
