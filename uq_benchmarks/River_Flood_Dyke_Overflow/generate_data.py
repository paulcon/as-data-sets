"""Write inputs.csv, outputs.csv and gradients.csv for the simplified river flood model.

    python generate_data.py                       # writes the three CSVs here
    python generate_data.py --out /tmp/f --check  # writes elsewhere, compares with the files here

Draws M = 10000 independent samples of the eight inputs of `flood_model`
(Table 4 of Lamboni et al. 2013) by the inverse CDF of
numpy.random.default_rng(SEED).random((M, 8)), with SEED = 2013, and evaluates
the overflow S (m, Eq. 4.20), the cost Cp (million euros, Eq. 4.21) and their
exact gradients with respect to the physical inputs.

It also checks the exact gradients against central finite differences
(step 1e-6 times each input's standard deviation) at every sample and prints
the largest relative error (row 2-norm). For Cp the samples within 1e-5 m of
the kink at Hd = 8 m are left out of that check.

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flood_model as fm  # noqa: E402

SEED = 2013
M = 10000
H_FD = 1e-6  # central-difference step in units of each input's standard deviation (check only)

PROVENANCE = ('flood_model.py (Lamboni, Iooss, Popelin, Gamboa 2013, Eqs. 4.20-4.21, Table 4); '
              f'inverse-CDF draws from numpy default_rng({SEED}).random((M, 8)); '
              'written by generate_data.py')


def draw(seed=SEED, M=M):
    """M samples of the physical inputs, shape (M, 8)."""
    return fm.sample(M, np.random.default_rng(seed))


def fd_check(X):
    """Largest relative errors (row 2-norm) of the exact gradients of S and Cp vs central differences."""
    _, sd = fm.mean_and_sd()
    GS, GC = fm.grad_overflow(X), fm.grad_cost(X)
    FS, FC = np.empty_like(GS), np.empty_like(GC)
    for j in range(8):
        dx = np.zeros(8)
        dx[j] = H_FD * sd[j]
        FS[:, j] = (fm.overflow(X + dx) - fm.overflow(X - dx)) / (2 * dx[j])
        FC[:, j] = (fm.cost(X + dx) - fm.cost(X - dx)) / (2 * dx[j])
    eS = np.linalg.norm(FS - GS, axis=1) / np.linalg.norm(GS, axis=1)
    away = np.abs(X[:, 4] - 8.0) > 1e-5
    eC = np.linalg.norm(FC - GC, axis=1)[away] / np.linalg.norm(GC, axis=1)[away]
    return eS.max(), eC.max(), int((~away).sum())


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, S, Cp, GS, GC):
    names = list(fm.NAMES)
    units = list(fm.UNITS)
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the 8 physical inputs, independent, distributions of '
                      f'Table 4. {PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(names, units)))
    np.savetxt(os.path.join(out, 'outputs.csv'), np.column_stack([S, Cp]), fmt='%.17g',
               delimiter=',',
               header=f'maximal annual overflow S (Eq. 4.20) and cost Cp (Eq. 4.21). {PROVENANCE}\n'
                      'S [m],Cp [million euros]')
    gS = [f'dS/d{n} [m per {u}]' for n, u in zip(names, units)]
    gC = [f'dCp/d{n} [million euros per {u}]' for n, u in zip(names, units)]
    np.savetxt(os.path.join(out, 'gradients.csv'), np.column_stack([GS, GC]), fmt='%.17g',
               delimiter=',',
               header='exact gradients with respect to the PHYSICAL inputs: columns 0-7 dS/dx, '
                      'columns 8-15 dCp/dx (left derivative in Hd at the kink Hd = 8). '
                      f'{PROVENANCE}\n' + ','.join(gS + gC))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()

    t0 = time.perf_counter()
    X = draw()
    S, Cp = fm.overflow(X), fm.cost(X)
    GS, GC = fm.grad_overflow(X), fm.grad_cost(X)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, S, Cp, GS, GC)
    t1 = time.perf_counter()
    print(f'wrote {args.out}: {len(X)} samples in {t1 - t0:.2f} s; S in [{S.min():.3f}, {S.max():.3f}] m '
          f'(no sample with S > 0); Cp in [{Cp.min():.4f}, {Cp.max():.4f}]')
    eS, eC, nk = fd_check(X)
    print(f'exact gradients vs central differences (step 1e-6 sd): max relative error '
          f'S {eS:.1e}, Cp {eC:.1e} ({nk} samples within 1e-5 m of Hd = 8 left out)')
    if args.check:
        same = [np.array_equal(read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n)))
                for n in ('inputs.csv', 'outputs.csv', 'gradients.csv')]
        print('files identical to the shipped ones (inputs, outputs, gradients):', same)


if __name__ == '__main__':
    main()
