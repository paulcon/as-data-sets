"""Write inputs.csv, outputs.csv, gradients_fixed.csv and gradients_pinned.csv.

    python generate_data.py                 # writes the four CSVs next to this script
    python generate_data.py --out /tmp/arch --check

Draws M = 1000 independent uniform samples of the seven inputs
(S, Phi, E, D, B, tw, tf) on the bounds in `arch_buckling_model` with
numpy's default_rng(SEED), SEED = 2020, and evaluates the in-plane elastic
buckling load N_acr (kN) of the fixed arch (Eq. 3 of Nguyen 2020) and of the
pin-ended arch (Eq. 2), with their exact gradients with respect to the physical
inputs, and the slenderness lambda (Eq. 1).

`--check` compares the new files with the ones next to this script and prints
the largest relative differences. Only numpy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arch_buckling_model as am  # noqa: E402

SEED = 2020
M = 1000
PROV = 'arch_buckling_model.py (Nguyen 2020, Eqs. 1-3, 6); written by generate_data.py, seed 2020'


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    t0 = time.perf_counter()

    X = am.sample(M, SEED)
    Nf, Gf = am.buckling_load(X, 'fixed', grad=True)
    Np, Gp = am.buckling_load(X, 'pinned', grad=True)
    lam = am.slenderness(X)

    os.makedirs(args.out, exist_ok=True)
    names = ','.join(f'{n} ({u})' for n, u in zip(am.NAMES, am.UNITS))
    np.savetxt(os.path.join(args.out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{M} independent uniform samples of the 7 physical inputs on the Table II '
                      f'bounds (E in MPa, corrected by 1/10), numpy default_rng({SEED})\n' + names)
    np.savetxt(os.path.join(args.out, 'outputs.csv'), np.column_stack([Nf, Np, lam]), fmt='%.17g',
               delimiter=',',
               header=f'In-plane elastic buckling load N_acr of the fixed (Eq. 3) and pin-ended '
                      f'(Eq. 2) arch, and the slenderness lambda (Eq. 1). {PROV}\n'
                      'N_acr fixed (kN),N_acr pinned (kN),lambda (-)')
    for case, G in (('fixed', Gf), ('pinned', Gp)):
        np.savetxt(os.path.join(args.out, f'gradients_{case}.csv'), G, fmt='%.17g', delimiter=',',
                   header=f'Exact gradient of N_acr ({case}) with respect to the physical inputs. '
                          f'{PROV}\n' + ','.join(f'dN/d{n} (kN/{u})' for n, u in
                                                 zip(am.NAMES, am.UNITS)))
    print(f'wrote {args.out}: {M} samples in {time.perf_counter() - t0:.2f} s')

    if args.check:
        def rel(a, b):
            return np.max(np.abs(a - b)/np.abs(b))
        print('max relative difference from the shipped files: '
              f"inputs {rel(X, read_csv(os.path.join(HERE, 'inputs.csv'))):.1e}, "
              f"outputs {rel(np.column_stack([Nf, Np, lam]), read_csv(os.path.join(HERE, 'outputs.csv'))):.1e}, "
              f"gradients_fixed {rel(Gf, read_csv(os.path.join(HERE, 'gradients_fixed.csv'))):.1e}, "
              f"gradients_pinned {rel(Gp, read_csv(os.path.join(HERE, 'gradients_pinned.csv'))):.1e}")


if __name__ == '__main__':
    main()
