"""Write inputs.csv, outputs.csv and gradients.csv for Golinski's speed reducer.

    python generate_data.py                       # writes the three CSVs here
    python generate_data.py --check               # writes to a temporary directory and compares with the files here

Draws N = 1000 samples of the six continuous inputs (b, m_t, l1, l2, d1, d2), independent and
uniform on the box of golinski_model.py (LOWER, UPPER; the psdr box after Ray 2003), with
numpy.random.default_rng(SEED), SEED = 1970, holds the number of pinion teeth at z = 17, and
evaluates the volume and the two shaft-1 constraints with their exact gradients with respect
to the physical inputs. The run takes a few milliseconds. `--check` prints the largest
differences from the shipped files and fails only on structural mismatches (shapes, NaN pattern,
inputs).

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
import golinski_model as gm  # noqa: E402

SEED = 1970
N = 1000
INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only

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
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X, F, G = generate()
    write(args.out, X, F, G)
    print(f'wrote {args.out}: {len(X)} samples, {time.perf_counter() - t0:.3f} s')

    if args.check:
        new = {'inputs.csv': X, 'outputs.csv': F, 'gradients.csv': G.reshape(len(G), -1)}
        bad, worst = [], (-1.0, '')
        for n, a in new.items():
            b = read_csv(os.path.join(HERE, n))
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                bad.append(f'FAIL: {n}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
                continue
            d = np.nanmax(np.abs(a - b))
            r = np.nanmax(np.nanmax(np.abs(a - b), axis=0) / np.maximum(np.nanmax(np.abs(b), axis=0), 1e-300))
            print(f'{n}: largest difference {d:.1e} (absolute), {r:.1e} (relative to the largest shipped value per column)')
            if n == 'inputs.csv' and r > INPUT_RTOL:
                bad.append(f'FAIL: {n} differs from the shipped inputs by {r:.1e} > {INPUT_RTOL:g}')
            worst = max(worst, (r, n))
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative to the largest value per column, {worst[1]})')
        for line in bad:
            print(line)
        if bad:
            sys.exit(1)

if __name__ == '__main__':
    main()
