"""Write inputs.csv, outputs.csv and gradients.csv for the SEIR epidemic-peak model.

    python generate_data.py                        # writes the three CSVs here
    python generate_data.py --out DIR --check       # writes to DIR, compares with the files here

Draws M = 1000 independent samples of the three lognormal inputs of `seir_model`
(R0, tau_inc, tau_inf; ln x ~ N(MU, diag(SIGMA^2))) with numpy's default generator seeded with
SEED = 2022, and computes the peak time t_peak (days), the peak number of infectious
individuals I_peak, and their exact (forward-sensitivity) gradients with respect to the
physical inputs.

It also checks the gradients against central finite differences in z = ln x (step 1e-5,
solves without sensitivities) at the first FD_ROWS samples and prints the largest relative
error (row 2-norm).

Only numpy, scipy and the standard library are imported. One process.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seir_model as sm  # noqa: E402

SEED = 2022
M = 1000
H_FD = 1e-5     # central-difference step in z = ln x, for the check only
FD_ROWS = 100

PROVENANCE = ('seir_model.py (Jensen et al. 2022 Eq. 4.1, N = 5.8e6, (S,E,I,R)(0) = (N-1,0,1,0); '
              'LSODA rtol 1e-11 atol 1e-9; lognormal inputs with the mean and sd of p. 10); '
              f'numpy default_rng({SEED}); written by generate_data.py')


def draw(seed=SEED, M=M):
    """M samples of the physical inputs, shape (M, 3): x = exp(MU + SIGMA * N(0, I))."""
    rng = np.random.default_rng(seed)
    return np.exp(sm.MU + sm.SIGMA * rng.standard_normal((M, 3)))


def fd_check(X, G):
    """Largest relative error (row 2-norm, per QoI) of the gradients w.r.t. z = ln x against
    central differences in z."""
    err = np.zeros((len(X), 2))
    for i, x in enumerate(X):
        z = np.log(x)
        Gz = G[i] * x
        Gfd = np.empty((2, 3))
        for j in range(3):
            dz = np.zeros(3)
            dz[j] = H_FD
            Gfd[:, j] = (np.array(sm.peak(np.exp(z + dz))) - np.array(sm.peak(np.exp(z - dz)))) / (2 * H_FD)
        err[i] = np.linalg.norm(Gfd - Gz, axis=1) / np.linalg.norm(Gz, axis=1)
    return err.max(axis=0)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, f, G):
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the physical inputs, lognormal with mean {sm.MEAN.tolist()} '
                      f'and sd {sm.SD.tolist()} (Jensen et al. 2022, p. 10). {PROVENANCE}\n'
                      'R0 (-),tau_inc (days),tau_inf (days)')
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.17g', delimiter=',',
               header=f'Epidemic peak time and size. {PROVENANCE}\n'
                      't_peak (days),I_peak (individuals)')
    cols = [f'd{q}/d{n} ({u})' for q, qu in zip(sm.QOI_NAMES, ('days', 'individuals'))
            for n, u in zip(sm.NAMES, (qu, f'{qu}/day', f'{qu}/day'))]
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), 6), fmt='%.17g', delimiter=',',
               header='Exact gradients with respect to the PHYSICAL inputs (forward sensitivities); '
                      'columns 0-2 for t_peak, 3-5 for I_peak. ' + PROVENANCE + '\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--fd-rows', type=int, default=FD_ROWS,
                    help=f'samples for the finite-difference check (default {FD_ROWS})')
    args = ap.parse_args()
    if os.path.abspath(args.out) == HERE and args.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X = draw()
    f, G = sm.peaks(X, grad=True)
    t1 = time.perf_counter()
    write(args.out, X, f, G)
    print(f'wrote {args.out}: M = {len(X)}, {t1 - t0:.1f} s for the model runs '
          f'({(t1 - t0) / len(X) * 1e3:.0f} ms per sample with sensitivities)')
    print(f'R0 range [{X[:, 0].min():.4f}, {X[:, 0].max():.4f}] (all > 1: {bool((X[:, 0] > 1).all())}); '
          f't_peak range [{f[:, 0].min():.2f}, {f[:, 0].max():.2f}] days; '
          f'I_peak range [{f[:, 1].min():.0f}, {f[:, 1].max():.0f}]')
    if args.fd_rows > 0:
        e = fd_check(X[:args.fd_rows], G[:args.fd_rows])
        print(f'gradient check vs central differences in ln x (h = {H_FD:g}), first {args.fd_rows} samples: '
              f'max relative error t_peak {e[0]:.1e}, I_peak {e[1]:.1e} ({time.perf_counter() - t1:.0f} s)')
    if args.check:
        rel = [np.max(np.abs(read_csv(os.path.join(args.out, n)) - read_csv(os.path.join(HERE, n)))
                      / np.abs(read_csv(os.path.join(HERE, n))))
               for n in ('inputs.csv', 'outputs.csv', 'gradients.csv')]
        print('max relative differences from the shipped files (inputs, outputs, gradients):',
              [f'{r:.1e}' for r in rel])
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
