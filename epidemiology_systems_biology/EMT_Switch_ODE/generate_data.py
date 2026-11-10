"""Write the samples, outputs and gradients of the E-cadherin / Slug EMT model (emt_model.py) for
the treatment group C = 2 cells, T = 3.64e-6 ng/mL, over the +/-10 % and +/-25 % boxes of
Supplementary Table S1 of Gasior (2026).

    python generate_data.py                        # writes the six CSVs here
    python generate_data.py --out DIR --check      # writes to DIR, compares with the files here

For each box, M = 2000 independent uniform (Monte Carlo) draws of the 11 inputs with numpy's
default generator, seeded with SEED_10 = 575 (+/-10 %) and SEED_25 = 576 (+/-25 %). At each draw
the model is integrated from the initial condition of Supplementary Table S2 for C = 2 to
t = 10,000 min; the end state is polished to its equilibrium, the gradients come from the implicit
function theorem, and the number of equilibria of the group at that draw is recorded.

It also checks the gradients against central finite differences in the normalized inputs
(step 1e-6 of the half-width) at the first FD_ROWS draws of each box and prints the largest
relative error (row 2-norm).

Only numpy, scipy and the standard library are imported. One process.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emt_model as em  # noqa: E402

C = 2
T = em.T_ON
SEEDS = {10: 575, 25: 576}
M = 2000
H_FD = 1e-6
FD_ROWS = 50
TOL = {'inputs': 1e-15, 'outputs': 1e-9, 'gradients': 1e-7}


def draw(box, M=M):
    rng = np.random.default_rng(SEEDS[box])
    return em.LOWER[box] + (em.UPPER[box] - em.LOWER[box]) * rng.random((M, 11))


def run(X):
    f = np.empty((len(X), 3))
    G = np.empty((len(X), 2, 11))
    gap = 0.0
    slow = -np.inf
    for i, x in enumerate(X):
        y, g, info = em.qoi(x, C, T, grad=True, return_info=True)
        f[i, :2], f[i, 2], G[i] = y, info['n_equilibria'], g
        gap = max(gap, info['polish'].max())
        slow = max(slow, info['eig'].real.max())
    return f, G, gap, slow


def fd_check(X, G, box):
    hw = (em.UPPER[box] - em.LOWER[box]) / 2
    err = np.zeros((len(X), 2))
    for i, x in enumerate(X):
        Gfd = np.empty((2, 11))
        for j in range(11):
            d = np.zeros(11)
            d[j] = H_FD * hw[j]
            Gfd[:, j] = (em.qoi(x + d, C, T) - em.qoi(x - d, C, T)) / (2 * H_FD)
        Gn = G[i] * hw
        err[i] = np.linalg.norm(Gfd - Gn, axis=1) / np.linalg.norm(Gn, axis=1)
    return err.max(axis=0)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, box, X, f, G):
    prov = (f'emt_model.py (Gasior 2026 Eqs. 1-2, Table 1 nominals, Table S1 +/-{box}% box, '
            f'Table S2 initial condition for C = 2; C = 2 cells, T = 3.64e-6 ng/mL; LSODA rtol {em.RTOL:g} '
            f'atol {em.ATOL:g} to t = 10000 min, Newton polish); numpy default_rng({SEEDS[box]}); '
            'written by generate_data.py')
    np.savetxt(os.path.join(out, f'inputs_pm{box}.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} independent uniform draws of the 11 inputs on the +/-{box}% box of '
                      f'Table S1. {prov}\n' + ','.join(f'{n} ({u})' for n, u in zip(em.NAMES, em.UNITS)))
    np.savetxt(os.path.join(out, f'outputs_pm{box}.csv'), f, fmt='%.17g', delimiter=',',
               header=f'E-cadherin and Slug at t = 10000 min, and the number of equilibria of the group '
                      f'at the draw (1 = monostable). {prov}\nE (ng/mL),S (ng/mL),n_equilibria (-)')
    cols = [f'd{q}/d{n} ((ng/mL)/({u}))' for q in em.QOI_NAMES for n, u in zip(em.NAMES, em.UNITS)]
    np.savetxt(os.path.join(out, f'gradients_pm{box}.csv'), G.reshape(len(G), 22), fmt='%.17g',
               delimiter=',',
               header='Gradients with respect to the PHYSICAL inputs (implicit function theorem at the '
                      'equilibrium); columns 0-10 for E, 11-21 for S. ' + prov + '\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--fd-rows', type=int, default=FD_ROWS,
                    help=f'draws per box for the finite-difference check (default {FD_ROWS})')
    args = ap.parse_args()
    if os.path.abspath(args.out) == HERE and args.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    for box in (10, 25):
        t1 = time.perf_counter()
        X = draw(box)
        f, G, gap, slow = run(X)
        write(args.out, box, X, f, G)
        print(f'+/-{box}%: M = {len(X)}, {time.perf_counter() - t1:.1f} s; equilibria per draw '
              f'{sorted(set(f[:, 2].astype(int).tolist()))}; E in [{f[:, 0].min():.5f}, {f[:, 0].max():.5f}], '
              f'S in [{f[:, 1].min():.5f}, {f[:, 1].max():.5f}] ng/mL; largest Newton correction of the '
              f'state at t = 10000 min {gap:.1e} (relative); slowest decay rate {-slow:.3f} per min')
        if args.fd_rows > 0:
            e = fd_check(X[:args.fd_rows], G[:args.fd_rows], box)
            print(f'   gradients vs central differences (step {H_FD:g} of the half-width), first '
                  f'{args.fd_rows} draws: max relative error E {e[0]:.1e}, S {e[1]:.1e}')
        if args.check:
            for kind in ('inputs', 'outputs', 'gradients'):
                n = f'{kind}_pm{box}.csv'
                A, B = read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n))
                rel = np.max(np.abs(A - B) / np.maximum(np.abs(B), 1e-300))
                print(f'   {n}: max relative difference from the shipped file {rel:.1e} '
                      f'(tolerance {TOL[kind]:g}) {"OK" if rel <= TOL[kind] else "FAIL"}')
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
