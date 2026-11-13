"""Write inputs.csv, outputs.csv and gradients.csv for the sulfate aerosol forcing model.

    python generate_data.py                       # writes the three CSVs here
    python generate_data.py --check               # writes to a temporary directory, compares with the files here

Draws M = 1000 independent samples of the nine lognormal inputs of
`sulfur_model` (ln x ~ N(MU, diag(SIGMA^2))) with numpy's default generator
seeded with SEED = 1992, and evaluates the direct radiative forcing dF_R
(W/m^2) and its exact gradient with respect to the log inputs z = ln x.

It also checks the exact gradients against central finite differences in z
(step 1e-6) at every sample and prints the largest relative error. `--check` prints the
largest differences from the shipped files and fails only on structural mismatches (shapes, NaN
pattern, inputs).

Only numpy and the standard library are imported.
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
import sulfur_model as sm  # noqa: E402

SEED = 1992
M = 1000
INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only
H_FD = 1e-6  # central-difference step in z = ln x, for the check only

PROVENANCE = ('sulfur_model.py (Charlson et al. 1992 Eqs. 1-5; lognormal inputs of '
              'Penner et al. 1994 via UQTestFuns; S0 = 1361 W/m^2, A = 5.1e14 m^2); '
              f'numpy default_rng({SEED}); written by generate_data.py')


def draw(seed=SEED, M=M):
    """M samples of the physical inputs, shape (M, 9): x = exp(MU + SIGMA * N(0, I))."""
    rng = np.random.default_rng(seed)
    Z = sm.MU + sm.SIGMA * rng.standard_normal((M, len(sm.NAMES)))
    return np.exp(Z)


def fd_check(X):
    """Largest relative error (row 2-norm) of grad_log against central differences in z."""
    Z = np.log(X)
    G = sm.grad_log(X)
    G_fd = np.empty_like(G)
    for j in range(Z.shape[1]):
        dz = np.zeros(Z.shape[1])
        dz[j] = H_FD
        G_fd[:, j] = (sm.forcing(np.exp(Z + dz)) - sm.forcing(np.exp(Z - dz))) / (2 * H_FD)
    return np.max(np.linalg.norm(G_fd - G, axis=1) / np.linalg.norm(G, axis=1))


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, f, G):
    names = list(sm.NAMES)
    units = [u or '-' for u in sm.UNITS]
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the 9 physical inputs, independent lognormal, '
                      f'ln x ~ N(MU, SIGMA^2). {PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(names, units)))
    np.savetxt(os.path.join(out, 'outputs.csv'), f[:, None], fmt='%.17g', delimiter=',',
               header=f'direct radiative forcing by anthropogenic sulfate. {PROVENANCE}\n'
                      'dF_R [W/m^2]')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header='exact gradient of dF_R with respect to the LOG inputs z = ln x '
                      f'(W/m^2 per unit of ln x; equals dF_R in every column). {PROVENANCE}\n'
                      + ','.join(f'd(dF_R)/d(ln {n}) [W/m^2]' for n in names))


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
    X = draw()
    f = sm.forcing(X)
    G = sm.grad_log(X)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, f, G)
    print(f'wrote {args.out}: {len(X)} samples in {time.perf_counter() - t0:.2f} s; '
          f'sample mean {f.mean():.4f}, variance {f.var(ddof=1):.4f} W/m^2 '
          f'(exact {sm.moments()[0]:.4f}, {sm.moments()[1]:.4f})')
    print(f'exact gradients vs central differences in ln x (h = {H_FD:g}): '
          f'max relative error {fd_check(X):.1e}')
    if args.check:
        bad, worst = [], (-1.0, '')
        for n in ('inputs.csv', 'outputs.csv', 'gradients.csv'):
            a, b = read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n))
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                bad.append(f'FAIL: {n}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
                continue
            rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
            print(f'{n}: largest difference from the shipped file {rel:.1e} (relative, elementwise)')
            if n == 'inputs.csv' and rel > INPUT_RTOL:
                bad.append(f'FAIL: {n} differs from the shipped inputs by {rel:.1e} > {INPUT_RTOL:g}')
            worst = max(worst, (rel, n))
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
        for line in bad:
            print(line)
        if bad:
            sys.exit(1)

if __name__ == '__main__':
    main()
