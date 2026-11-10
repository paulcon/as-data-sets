"""Write inputs.csv, outputs.csv and gradients.csv for the speed-reducer shaft limit state.

    python generate_data.py --out DIR              # writes the three CSVs to DIR
    python generate_data.py --out DIR --check      # writes to DIR (not this directory), compares
                                                   # with the files here
    python generate_data.py --out .                # (from this directory) rewrites the shipped files

--out is required: the script never writes into this directory unless it is named explicitly.

Draws M = 1000 independent Monte Carlo samples of the five inputs of `shaft_model` (xi ~ N(0, I_5)
from numpy's default generator seeded with SEED = 2004, mapped to the physical marginals by
x_i = F_i^{-1}(Phi(xi_i))), and computes the limit state g (Pa) and its exact gradient with respect
to the physical inputs. It checks the gradients against central finite differences at every
sample and prints the largest relative error (row 2-norm, each component scaled by the input's sd).

Only numpy, scipy and the standard library are imported. One process; under a second.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shaft_model as sm  # noqa: E402

SEED = 2004
M = 1000
H_REL = 1e-6        # central-difference step, relative to each input's mean (check only)
TOL = 1e-12         # --check: max |difference| / max |shipped value|, per column

PROVENANCE = ('shaft_model.py (limit state of Du and Sudjianto 2004 as defined in uqtestfuns 0.7.0 '
              'RSSpeedReducerShaft); Monte Carlo, xi = numpy default_rng(%d).standard_normal((%d, 5)), '
              'x = F^-1(Phi(xi)); written by generate_data.py' % (SEED, M))


def draw(seed=SEED, M=M):
    return sm.sample(M, np.random.default_rng(seed))[0]


def fd_check(X, G):
    h = H_REL * np.abs(sm.MEAN)
    Gfd = np.empty_like(G)
    for j in range(5):
        e = np.zeros(5)
        e[j] = h[j]
        Gfd[:, j] = (sm.limit_state(X + e) - sm.limit_state(X - e)) / (2 * h[j])
    return np.max(np.linalg.norm((Gfd - G) * sm.SD, axis=1) / np.linalg.norm(G * sm.SD, axis=1))


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, g, G):
    marg = ('D ~ N(39, 0.1) mm, L ~ N(400, 0.1) mm, F ~ Gumbel max (mean 1500, sd 350) N, '
            'T ~ N(250, 35) N m, S ~ U[70, 80] MPa')
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Monte Carlo samples of the physical inputs, {marg}. {PROVENANCE}\n'
                      'D (mm),L (mm),F (N),T (N m),S (MPa)')
    np.savetxt(os.path.join(out, 'outputs.csv'), g.reshape(-1, 1), fmt='%.17g', delimiter=',',
               header=f'Limit state g = strength - stress (g <= 0 is failure). {PROVENANCE}\n'
                      'g (Pa)')
    cols = [f'dg/d{n} (Pa/{u})' for n, u in zip(sm.NAMES, sm.UNITS)]
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header='Exact gradients of g with respect to the PHYSICAL inputs (closed form). '
                      + PROVENANCE + '\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory (required; name this directory explicitly to rewrite the shipped files)')
    ap.add_argument('--check', action='store_true',
                    help='compare with the CSVs in this directory (needs --out DIR elsewhere)')
    args = ap.parse_args()
    here = os.path.realpath(HERE)
    out_real = os.path.realpath(args.out)
    inside = os.path.commonpath([out_real, here]) == here
    if args.check and inside:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X = draw()
    g, G = sm.limit_state(X, grad=True)
    write(args.out, X, g, G)
    print(f'wrote {len(X)} samples to {"this directory" if out_real == here else "--out"}')
    print('input ranges: ' + '; '.join(f'{n} [{X[:, j].min():.4g}, {X[:, j].max():.4g}] {u}'
                                        for j, (n, u) in enumerate(zip(sm.NAMES, sm.UNITS))))
    print(f'g range [{g.min():.4e}, {g.max():.4e}] Pa; samples with g <= 0: {int((g <= 0).sum())}; '
          f'F < 0: {int((X[:, 2] < 0).sum())}')
    print(f'gradient check vs central differences (step {H_REL:g} x mean), all {len(X)} samples: '
          f'max relative error {fd_check(X, G):.1e}')
    if args.check:
        worst = []
        for n in ('inputs.csv', 'outputs.csv', 'gradients.csv'):
            a, b = read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n))
            worst.append(np.max(np.abs(a - b).max(axis=0) / np.abs(b).max(axis=0)))
        print('max |difference| / max |shipped value| per column (inputs, outputs, gradients): '
              + ', '.join(f'{w:.1e}' for w in worst) + f'; tolerance {TOL:g}: '
              + ('PASS' if max(worst) <= TOL else 'FAIL'))
        if max(worst) > TOL:
            sys.exit(1)
    print(f'total {time.perf_counter() - t0:.2f} s')


if __name__ == '__main__':
    main()
