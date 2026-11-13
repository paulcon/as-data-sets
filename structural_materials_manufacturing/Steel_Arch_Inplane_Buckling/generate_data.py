"""Write inputs.csv, outputs.csv, gradients_fixed.csv and gradients_pinned.csv.

    python generate_data.py                 # writes the four CSVs next to this script
    python generate_data.py --check         # regenerates into a temporary directory and compares

Draws M = 1000 independent uniform samples of the seven inputs
(S, Phi, E, D, B, tw, tf) on the bounds in `arch_buckling_model` with
numpy's default_rng(SEED), SEED = 2020, and evaluates the in-plane elastic
buckling load N_acr (kN) of the fixed arch (Eq. 3 of Nguyen 2020) and of the
pin-ended arch (Eq. 2), with their exact gradients with respect to the physical
inputs, and the slenderness lambda (Eq. 1).

`--check` prints the largest differences from the shipped files and fails only on structural
mismatches (shapes, NaN pattern, inputs). Only numpy and the standard library are imported.
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arch_buckling_model as am  # noqa: E402

SEED = 2020
M = 1000
INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only
PROV = 'arch_buckling_model.py (Nguyen 2020, Eqs. 1-3, 6); written by generate_data.py, seed 2020'


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
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
    if os.path.abspath(args.out) == HERE and args.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
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
        new = {'inputs.csv': X, 'outputs.csv': np.column_stack([Nf, Np, lam]),
               'gradients_fixed.csv': Gf, 'gradients_pinned.csv': Gp}
        bad, worst = [], (-1.0, '')
        for n, a in new.items():
            b = read_csv(os.path.join(HERE, n))
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                bad.append(f'FAIL: {n}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
                continue
            d = np.nanmax(np.abs(a - b)/np.abs(b))
            print(f'{n}: largest relative difference (elementwise) {d:.1e}')
            if n == 'inputs.csv' and d > INPUT_RTOL:
                bad.append(f'FAIL: {n} differs from the shipped inputs by {d:.1e} > {INPUT_RTOL:g}')
            worst = max(worst, (d, n))
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
        for line in bad:
            print(line)
        if bad:
            sys.exit(1)

if __name__ == '__main__':
    main()
