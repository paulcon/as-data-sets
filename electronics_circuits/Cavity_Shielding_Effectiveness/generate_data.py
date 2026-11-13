"""Write inputs.csv, outputs.csv and gradients.csv for the slotted-cavity shielding-effectiveness bound.

    python generate_data.py --out DIR             # writes the three CSVs to DIR
    python generate_data.py --check               # writes to a temporary directory, compares with the files here
    python generate_data.py --out DIR --check     # writes to DIR, compares with the files here
    python generate_data.py --here                # maintainers only: rewrites the shipped files here

Draws M = 1000 independent Monte Carlo samples of the six inputs of `cavity_se_model`, uniform on
the ranges of Table 1 (p. 33) of Campione et al. 2021, with numpy's default generator seeded with
SEED = 2021, and computes four outputs: the matched power-balance SE bound (dB) at 1, 2 and 3 GHz
and its mean over the paper's 51 frequencies 1-3 GHz, with complex-step gradients with respect to
the physical inputs (paper units).

It also checks the gradients against central finite differences (step 1e-6 of each input's range)
at the first FD_ROWS samples and prints the largest relative error (row 2-norm).

--check prints the largest differences from the shipped files (relative to each column's largest value)
and fails only on structural mismatches (shapes, NaN pattern, inputs).

Only numpy and the standard library are imported. One process; a few seconds.
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import cavity_se_model as cm  # noqa: E402

SEED = 2021
M = 1000
FD_REL = 1e-6
FD_ROWS = 100
INPUT_RTOL = 1e-8   # --check fails if a regenerated input (a seeded draw) differs by more (relative)

PROVENANCE = ('cavity_se_model.py (Campione et al. 2021, Appendix A, Eqs. 5-6; slot arc length '
              'l = 2a asin(l_p/(2a)), p. 4); uniform inputs on Table 1 ranges; '
              f'numpy default_rng({SEED}); written by generate_data.py')


def draw(seed=SEED, M=M):
    """M Monte Carlo samples of the physical inputs, uniform on [LOWER, UPPER], shape (M, 6)."""
    rng = np.random.default_rng(seed)
    return cm.LOWER + (cm.UPPER - cm.LOWER) * rng.random((M, 6))


def fd_check(X, G):
    """Largest relative error (row 2-norm, per output) of G against central differences."""
    h = FD_REL * (cm.UPPER - cm.LOWER)
    Gfd = np.empty_like(G)
    for j in range(6):
        e = np.zeros(6)
        e[j] = h[j]
        Gfd[:, :, j] = (cm.qois(X + e) - cm.qois(X - e)) / (2 * h[j])
    return (np.linalg.norm(Gfd - G, axis=2) / np.linalg.norm(G, axis=2)).max(axis=0)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, f, G):
    names = [f'{n} ({u})' for n, u in zip(cm.NAMES, cm.UNITS)]
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Monte Carlo samples of the physical inputs, uniform on the ranges of Table 1 '
                      f'(Campione et al. 2021, p. 33). {PROVENANCE}\n' + ','.join(names))
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.17g', delimiter=',',
               header='Matched power-balance bound on the shielding effectiveness, 10 log10(4 <|H_j|^2>_A/|H0|^2): '
                      'at 1, 2, 3 GHz and the mean over 51 frequencies 1-3 GHz. ' + PROVENANCE + '\n'
                      + ','.join(f'{q} (dB)' for q in cm.QOI_NAMES))
    cols = [f'd{q}/d{n} (dB/{u})' for q in cm.QOI_NAMES for n, u in zip(cm.NAMES, cm.UNITS)]
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), 24), fmt='%.17g', delimiter=',',
               header='Complex-step gradients with respect to the PHYSICAL inputs (paper units); columns 0-5 '
                      'SE_1GHz, 6-11 SE_2GHz, 12-17 SE_3GHz, 18-23 SE_band. ' + PROVENANCE + '\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory, outside this directory (with --check, default a temporary directory)')
    ap.add_argument('--here', action='store_true', help='rewrite the shipped files in this directory')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--fd-rows', type=int, default=FD_ROWS,
                    help=f'samples for the finite-difference check (default {FD_ROWS})')
    args = ap.parse_args()
    if args.here:
        if args.out or args.check:
            ap.error('--here takes no --out or --check')
        out = HERE
    else:
        if not args.out and args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        if not args.out:
            ap.error('give --out DIR (outside this directory), or --here to rewrite the shipped files')
        out = os.path.realpath(args.out)                  # symbolic links resolved
        if os.path.commonpath([out, HERE]) == HERE:
            ap.error('--out must be outside this directory, so that the shipped files are not overwritten')
    os.makedirs(out, exist_ok=True)

    t0 = time.perf_counter()
    X = draw()
    f, G = cm.qois(X, grad=True)
    t1 = time.perf_counter()
    write(out, X, f, G)
    klp = cm.kl_over_pi(X, 3e9)
    print(f'wrote {out}: M = {len(X)}, {t1 - t0:.1f} s for the model runs (outputs and complex-step gradients)')
    print('output ranges (dB): ' + ', '.join(f'{q} [{f[:, i].min():.2f}, {f[:, i].max():.2f}]'
                                           for i, q in enumerate(cm.QOI_NAMES)))
    print(f'max k l / pi: {cm.kl_over_pi(X, 1e9).max():.3f} at 1 GHz, {cm.kl_over_pi(X, 2e9).max():.3f} at 2 GHz, '
          f'{klp.max():.3f} at 3 GHz ({int((klp > 1).sum())} of {len(X)} samples above the slot resonance at 3 GHz)')
    if args.fd_rows > 0:
        e = fd_check(X[:args.fd_rows], G[:args.fd_rows])
        print(f'gradient check vs central differences (step {FD_REL:g} x range), first {args.fd_rows} samples: '
              'max relative error ' + ', '.join(f'{q} {v:.1e}' for q, v in zip(cm.QOI_NAMES, e)))
    print(f'total {time.perf_counter() - t0:.1f} s')
    if args.check:
        ok, worst = True, (0.0, 'outputs.csv')
        for n in ('inputs.csv', 'outputs.csv', 'gradients.csv'):
            a, b = read_csv(os.path.join(out, n)), read_csv(os.path.join(HERE, n))
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                print(f'FAIL: {n}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
                ok = False
                continue
            rel = np.nanmax(np.abs(a - b) / np.nanmax(np.abs(b), axis=0))   # relative to each column's largest value
            print(f'{n}: max relative difference from the shipped file {rel:.1e}')
            if n == 'inputs.csv':
                if rel > INPUT_RTOL:
                    print(f'FAIL: {n} differs from the shipped inputs by {rel:.1e} > {INPUT_RTOL:g}')
                    ok = False
            elif rel >= worst[0]:
                worst = (rel, n)
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative to the column max, {worst[1]})')
        if not ok:
            sys.exit(1)

if __name__ == '__main__':
    main()
