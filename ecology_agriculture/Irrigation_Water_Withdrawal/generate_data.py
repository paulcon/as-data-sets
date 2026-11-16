"""Write inputs.csv, outputs.csv and gradients.csv for the irrigation water withdrawal model (iww_model.py).

    python generate_data.py             # writes the three CSVs here
    python generate_data.py --check     # writes to a temporary directory, compares with the files here

Draws M = 2000 Monte Carlo samples of the twelve independent uniform inputs of Table S4 (Puy et al.
2022, Supplementary Materials, p. 18) as LOWER + (UPPER - LOWER) * numpy.random.default_rng(SEED).random((M, 12)),
SEED = 2022, and evaluates the irrigation water withdrawal y (m^3, one day, archived-code form of Eq. 16)
and its exact gradient with respect to the physical inputs. It also checks the exact gradients at every
sample against central finite differences (step 1e-6 times each input's range) and prints the largest
relative error (row 2-norm), and counts samples with y <= 0 or k_c ET_0 <= P.

`--check` prints the largest differences from the shipped files and exits with code 1 only on a structural
mismatch (a missing file, a different shape or NaN pattern, or inputs that differ by more than 1e-8 relative).
Imports numpy and the standard library. Runs in about a second.
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
import iww_model as im  # noqa: E402

SEED = 2022
M = 2000
INPUT_RTOL = 1e-8
H_FD = 1e-6   # central-difference step, as a fraction of each input's range (check only)
FILES = ('inputs.csv', 'outputs.csv', 'gradients.csv')
PROVENANCE = (f'iww_model.py (Puy et al. 2022, Supplementary Materials Eq. 16 as evaluated by the archived code, '
              f'Table S4 inputs); Monte Carlo, numpy default_rng({SEED}).random((M, 12)); written by generate_data.py')


def gradient_check(X):
    """Largest relative error (row 2-norm) of the exact gradient against central differences."""
    h = H_FD * (im.UPPER - im.LOWER)
    G = im.grad_iww(X)
    F = np.column_stack([(im.iww(X + h[j] * e) - im.iww(X - h[j] * e)) / (2 * h[j])
                         for j, e in enumerate(np.eye(im.M))])
    return np.max(np.linalg.norm(F - G, axis=1) / np.linalg.norm(G, axis=1))


def write(out, X, f, G):
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Monte Carlo samples of the 12 independent uniform inputs of Table S4. {PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(im.NAMES, im.UNITS)))
    np.savetxt(os.path.join(out, 'outputs.csv'), f[:, None], fmt='%.17g', delimiter=',',
               header=f'irrigation water withdrawal for one day in the grid cell. {PROVENANCE}\ny [m^3]')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header=f'exact gradient of y with respect to the PHYSICAL inputs. {PROVENANCE}\n'
                      + ','.join(f'dy/d{n} [m^3 per {u}]' if u != '-' else f'dy/d{n} [m^3]'
                                 for n, u in zip(im.NAMES, im.UNITS)))


def compare(out):
    """Print the largest relative difference of each file from the shipped one; False on a structural mismatch."""
    bad, worst = [], (-1.0, '')
    for name in FILES:
        shipped = os.path.join(HERE, name)
        if not os.path.exists(shipped):
            bad.append(f'FAIL: {name}: no shipped file')
            continue
        a = np.loadtxt(os.path.join(out, name), delimiter=',', ndmin=2)
        b = np.loadtxt(shipped, delimiter=',', ndmin=2)
        if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
            bad.append(f'FAIL: {name}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
            continue
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
        print(f'{name}: largest relative difference (elementwise) from the shipped file {rel:.1e}')
        if name == 'inputs.csv':
            scale = np.max(np.abs(b), axis=0)
            if np.max(np.abs(a - b) / scale) > INPUT_RTOL:
                bad.append(f'FAIL: {name} differs from the shipped inputs by more than {INPUT_RTOL:g} of each '
                           f'input\'s largest magnitude')
        worst = max(worst, (rel, name))
    if worst[1]:
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
    for line in bad:
        print(line)
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    ap.add_argument('--check', action='store_true', help='compare a rerun with the CSVs in this directory')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            args.out = HERE
    if args.check and os.path.abspath(args.out) == HERE:
        ap.error('--check needs --out outside this directory, so that the shipped files are not overwritten')

    t0 = time.perf_counter()
    X = im.sample(M, np.random.default_rng(SEED))
    f = im.iww(X)
    G = im.grad_iww(X)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, f, G)
    print(f'wrote {len(X)} samples in {time.perf_counter() - t0:.2f} s; y in [{f.min():.1f}, {f.max():.1f}] m^3')
    print(f'exact gradients: max relative error vs central differences (step {H_FD:g} x range) {gradient_check(X):.1e}')
    kcet = X[:, 6] * im.et0(X)
    print(f'samples with y <= 0: {int((f <= 0).sum())}; with k_c ET_0 <= P: {int((kcet <= X[:, 11]).sum())}')
    if args.check and not compare(args.out):
        sys.exit(1)


if __name__ == '__main__':
    main()
