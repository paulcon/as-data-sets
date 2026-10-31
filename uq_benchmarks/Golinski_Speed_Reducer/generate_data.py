"""Write inputs.csv, outputs.csv and gradients.csv for Golinski's speed reducer.

    python generate_data.py                       # writes the three CSVs here
    python generate_data.py --out /tmp/gs --check # writes elsewhere and compares with the files here

Draws N = 1000 samples of the six continuous inputs (b, m_t, l1, l2, d1, d2), independent and
uniform on the box of golinski_model.py (LOWER, UPPER; the psdr box after Ray 2003), with
numpy.random.default_rng(SEED), SEED = 1970, holds the number of pinion teeth at z = 17, and
evaluates the volume and the two shaft-1 constraints with their exact gradients with respect
to the physical inputs. The run takes a few milliseconds.

Only numpy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golinski_model as gm  # noqa: E402

SEED = 1970
N = 1000

PROVENANCE = ('golinski_model.py (Golinski 1970 volume with exact constants, z = 17, i = 3; '
              'Ray 2003 constraints g3, g5); written by generate_data.py, seed 1970')


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def generate(n=N, seed=SEED):
    X = gm.sample(n, seed)
    return X, gm.qoi(X), gm.grad(X)


def write(out, X, F, G):
    box = ', '.join(f'{n} U[{lo:g}, {hi:g}]' for n, lo, hi in zip(gm.NAMES, gm.LOWER, gm.UPPER))
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the 6 physical inputs (cm), independent uniform: {box}; '
                      f'z = {gm.Z} held. Seed {SEED}.\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(gm.NAMES, gm.UNITS)))
    np.savetxt(os.path.join(out, 'outputs.csv'), F, fmt='%.17g', delimiter=',',
               header=f'Outputs: gearbox volume, shaft 1 deflection constraint g3, shaft 1 stress '
                      f'constraint g5 (feasible if <= 0). {PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(gm.QOI_NAMES, gm.QOI_UNITS)))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), -1), fmt='%.17g',
               delimiter=',',
               header='Exact gradients w.r.t. the PHYSICAL inputs, column 6*i + j is '
                      f'd(output i)/d(input j). {PROVENANCE}\n'
                      + ','.join(f'd{q}/d{n}' for q in gm.QOI_NAMES for n in gm.NAMES))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X, F, G = generate()
    write(args.out, X, F, G)
    print(f'wrote {args.out}: {len(X)} samples, {time.perf_counter() - t0:.3f} s')

    if args.check:
        X0 = read_csv(os.path.join(HERE, 'inputs.csv'))
        F0 = read_csv(os.path.join(HERE, 'outputs.csv'))
        G0 = read_csv(os.path.join(HERE, 'gradients.csv')).reshape(G.shape)
        print('max abs difference from the shipped files: inputs '
              f'{np.max(np.abs(X - X0)):.1e}, outputs {np.max(np.abs(F - F0)):.1e}, '
              f'gradients {np.max(np.abs(G - G0)):.1e}')


if __name__ == '__main__':
    main()
