"""Write inputs.csv, outputs.csv and gradients.csv for the malaria R_eff model.

    python generate_data.py --out DIR                  # writes the three CSVs to DIR (outside this directory)
    python generate_data.py --check                    # writes to a temporary directory and compares with the files here
    python generate_data.py --out DIR --check          # writes to DIR and compares with the files here
    python generate_data.py --here                     # rewrites the shipped files in this directory

Draws a Latin hypercube sample of M = 1000 points from the Table 3 distributions of the 20
parameters that enter R_eff (uniform, and triangular for b and r_r) with numpy's default
generator seeded with SEED = 2019, and computes R_eff (Eq. 19 of Bala and Gimba 2019), its two
parts R_r and R_s (Eq. 20), and the exact gradient of R_eff with respect to the physical inputs.

It also checks the gradients against central differences (relative step 1e-6 in each input) at
the first FD_ROWS samples and the identity R_eff = rho(F V^{-1})^2 (Eqs. 16-19) at every sample.

--check prints the largest differences from the shipped files (relative, elementwise) and fails only on
structural mismatches (shapes, NaN pattern, inputs differing by more than INPUT_RTOL).

Only numpy and the standard library are imported. One process; about a second.
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
import malaria_model as mm  # noqa: E402

SEED = 2019
M = 1000
H_FD = 1e-6
FD_ROWS = 100
INPUT_RTOL = 1e-8  # --check fails if the regenerated inputs differ by more than this (relative)
FILES = ('inputs.csv', 'outputs.csv', 'gradients.csv')

PROVENANCE = ('malaria_model.py (Bala and Gimba 2019, Eqs. 1 and 19; xi1 = a b1 rho kv/kh, p. 10); '
              f'Latin hypercube, numpy default_rng({SEED}); written by generate_data.py')


def _inside_here(path):
    """True if path is this directory or any directory below it (symbolic links resolved)."""
    here, p = os.path.realpath(HERE), os.path.realpath(path)
    return os.path.commonpath([here, p]) == here


def draw(seed=SEED, M=M):
    return mm.sample(M, seed, lhs=True)


def fd_check(X, G):
    """Largest relative error (2-norm per row) of G against central differences."""
    err = []
    for x, g in zip(X, G):
        gfd = np.empty(mm.M_IN)
        for j in range(mm.M_IN):
            dx = np.zeros(mm.M_IN)
            dx[j] = H_FD * x[j]
            gfd[j] = (mm.reff(x + dx) - mm.reff(x - dx)) / (2 * dx[j])
        err.append(np.linalg.norm(gfd - g) / np.linalg.norm(g))
    return max(err)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, F, G):
    cols = ','.join(f'{n} ({u})' for n, u in zip(mm.NAMES, mm.UNITS))
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Latin hypercube samples of the 20 parameters of R_eff, Table 3 '
                      f'distributions (Bala and Gimba 2019, p. 19). {PROVENANCE}\n{cols}')
    np.savetxt(os.path.join(out, 'outputs.csv'), F, fmt='%.17g', delimiter=',',
               header=f'Effective reproduction number R_eff (Eq. 19) and its parts R_r, R_s (Eq. 20), '
                      f'R_eff = R_r + R_s. {PROVENANCE}\nR_eff (-),R_r (-),R_s (-)')
    inv = {'-': '-', '1/day': 'day', '1/(human day)': 'human day', '1/(mosquito day)': 'mosquito day'}
    gcols = ','.join(f'dR_eff/d{n} ({inv[u]})' for n, u in zip(mm.NAMES, mm.UNITS))
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header='Exact gradient of R_eff with respect to the PHYSICAL inputs. '
                      f'{PROVENANCE}\n{gcols}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory, outside this directory and its subdirectories (with --check, '
                         'default a temporary directory)')
    ap.add_argument('--here', action='store_true', help='rewrite the shipped files in this directory')
    ap.add_argument('--check', action='store_true', help='compare the new files with the shipped ones')
    ap.add_argument('--fd-rows', type=int, default=FD_ROWS)
    args = ap.parse_args()
    if args.out is None and args.check and not args.here:
        args.out = tempfile.mkdtemp(prefix='check_')
        atexit.register(shutil.rmtree, args.out, ignore_errors=True)
    if args.here == bool(args.out):
        ap.error('give exactly one of --out DIR (outside this directory) or --here')
    if args.out and _inside_here(args.out):
        ap.error('--out must be outside this directory and its subdirectories; '
                 'use --here to rewrite the shipped files')
    if args.here and args.check:
        ap.error('--check needs --out DIR')
    out = HERE if args.here else args.out
    os.makedirs(out, exist_ok=True)

    t0 = time.perf_counter()
    X = draw()
    R, G = mm.reffs(X, grad=True)
    parts = np.array([mm.strain_parts(x) for x in X])
    F = np.column_stack([R, parts])
    write(out, X, F, G)
    print(f'wrote {FILES} to {"this directory" if args.here else "--out"}: M = {len(X)}, '
          f'{time.perf_counter() - t0:.2f} s')
    print(f'R_eff in [{R.min():.3g}, {R.max():.4g}], median {np.median(R):.4g}; '
          f'{int(np.sum(R < 1))} of {len(R)} samples have R_eff < 1')
    print(f'max |R_r + R_s - R_eff| / R_eff = {np.max(np.abs(parts.sum(axis=1) - R) / R):.1e}')
    ngm = np.array([np.max(np.abs(np.linalg.eigvals(mm.ngm(x)))) ** 2 for x in X])
    print(f'max |rho(F V^-1)^2 - R_eff| / R_eff = {np.max(np.abs(ngm - R) / R):.1e}')
    if args.fd_rows > 0:
        print(f'gradient vs central differences (relative step {H_FD:g}), first {args.fd_rows} samples: '
              f'max relative error {fd_check(X[:args.fd_rows], G[:args.fd_rows]):.1e}')
    if args.check:
        bad, worst = False, (0.0, FILES[0])
        for n in FILES:
            a, b = read_csv(os.path.join(out, n)), read_csv(os.path.join(HERE, n))
            if a.shape != b.shape:
                print(f'{n}: shape {a.shape} differs from the shipped {b.shape}: FAIL')
                bad = True
                continue
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                print(f'{n}: NaN pattern differs from the shipped file: FAIL')
                bad = True
            rel = float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300), initial=0.0))
            print(f'{n}: largest relative difference (elementwise) from the shipped file {rel:.1e}')
            if n == 'inputs.csv' and rel > INPUT_RTOL:
                print(f'{n}: inputs differ by more than {INPUT_RTOL:g}: FAIL')
                bad = True
            worst = max(worst, (rel, n))
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
        if bad:
            sys.exit(1)
    print(f'total {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
