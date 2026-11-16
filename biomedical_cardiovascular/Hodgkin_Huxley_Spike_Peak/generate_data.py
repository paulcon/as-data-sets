"""Write inputs.csv, outputs.csv and gradients.csv for the Hodgkin-Huxley spike-peak model.

    python generate_data.py              # writes the three CSVs here
    python generate_data.py --check      # writes to a temporary directory and compares with the files here
    python generate_data.py --workers 1  # one process (default: up to 4)

Draws M = 1000 independent uniform samples of the eleven inputs of `hh_model` (Table 2 of
Tennøe et al. 2018, each on its nominal value +/- 10 %) as
LOWER + (UPPER - LOWER) * numpy.random.default_rng(SEED).random((M, 11)), SEED = 2018, and
solves the model with its forward sensitivity equations (DOP853, rtol 1e-10, atol 1e-12) for
the 25 outputs and their gradients with respect to the physical inputs.

It also prints the regime of every sample (the number of maxima of v before 5 ms and in the
5-15 ms window, the ranges of the peak and trough times) and the number of samples with
h0 > 1. `--check` prints the largest differences from the shipped files and fails only on
structural mismatches (shapes, NaN pattern, inputs).

Only numpy, scipy and the standard library are imported.
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse  # noqa: E402
import atexit  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hh_model as hm  # noqa: E402

SEED = 2018
M = 1000
INPUT_RTOL = 1e-8  # --check: regenerated inputs may differ from the shipped ones by round-off only
FILES = ('inputs.csv', 'outputs.csv', 'gradients.csv')

PROVENANCE = ('hh_model.py (Hodgkin-Huxley 1952 model; rate functions of Torres Valderrama et al. 2015, Eq. 1; '
              'parameters of Tennøe et al. 2018, Table 2; I = 140 uA/cm^2 from t = 0); '
              f'independent uniform draws from numpy default_rng({SEED}).random((M, 11)); '
              'DOP853 with forward sensitivities, rtol 1e-10, atol 1e-12; written by generate_data.py')


def draw(seed=SEED, M=M):
    """M samples of the physical inputs, shape (M, 11)."""
    return hm.sample(M, np.random.default_rng(seed))


def _one(p):
    f, G = hm.qoi_and_grad(p)
    tmax, _, _, _ = hm.extrema(p)
    return f, G, int(np.sum(tmax <= hm.T_WINDOW[0])), int(np.sum(tmax > hm.T_WINDOW[0]))


def evaluate(X, workers):
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_one, X, chunksize=10))
    else:
        res = [_one(p) for p in X]
    f = np.array([r[0] for r in res])
    G = np.array([r[1] for r in res])
    n_before = np.array([r[2] for r in res])
    n_window = np.array([r[3] for r in res])
    return f, G, n_before, n_window


def write(out, X, f, G):
    names, units = hm.NAMES, hm.UNITS
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} independent uniform samples of the 11 physical inputs, each on its nominal '
                      f'value +/- 10 %. {PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(names, units)))
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.13g', delimiter=',',
               header='vpeak, tpeak: first local maximum of v after 5 ms; vmin, tmin: first local minimum '
                      f'after it; v(t) at t = 5, 5.5, ..., 15 ms. {PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(hm.QOI_NAMES, hm.QOI_UNITS)))
    cols = []
    for q, qu in zip(hm.QOI_NAMES, hm.QOI_UNITS):
        for n, u in zip(names, units):
            cols.append(f'd{q}/d{n} [{qu}]' if u == '-' else f'd{q}/d{n} [{qu} per {u}]')
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(X), -1), fmt='%.11g', delimiter=',',
               header='gradients with respect to the PHYSICAL inputs from the forward sensitivity equations; '
                      f'column {hm.M_INPUTS} i + j is d(output i)/d(input j). {PROVENANCE}\n'
                      + ','.join(cols))


def compare(out, here, files):
    """Print the largest relative difference (per element) between the files in out and here.

    Returns False on a structural mismatch: a different shape or NaN/inf pattern, or inputs.csv
    beyond INPUT_RTOL of each input's largest magnitude."""
    bad, worst = [], (-1.0, '')
    for name in files:
        a = np.loadtxt(os.path.join(out, name), delimiter=',', comments='#', ndmin=2)
        b = np.loadtxt(os.path.join(here, name), delimiter=',', comments='#', ndmin=2)
        if a.shape != b.shape or not (np.array_equal(np.isnan(a), np.isnan(b))
                                      and np.array_equal(np.isinf(a), np.isinf(b))):
            bad.append(f'FAIL: {name}: shape {a.shape} or NaN/inf pattern differs from the shipped {b.shape}')
            continue
        ok = np.isfinite(b)
        rel = float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300))) if ok.any() else 0.0
        print(f'{name}: largest relative difference (elementwise) from the shipped file {rel:.1e}')
        if name == 'inputs.csv':
            col = np.max(np.abs(a - b), axis=0) / np.max(np.abs(b), axis=0)
            if np.max(col) > INPUT_RTOL:
                bad.append(f'FAIL: {name} differs from the shipped inputs by {np.max(col):.1e} > {INPUT_RTOL:g}')
        worst = max(worst, (rel, name))
    print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
    for line in bad:
        print(line)
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                    help='worker processes (default: min(4, CPUs))')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            args.out = HERE
    if args.check and os.path.abspath(args.out) == HERE:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')

    t0 = time.perf_counter()
    X = draw()
    f, G, n_before, n_window = evaluate(X, args.workers)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, f, G)
    t1 = time.perf_counter()
    print(f'wrote {len(X)} samples ({args.workers} worker processes) in {t1 - t0:.1f} s')
    for i, (n, u) in enumerate(zip(hm.QOI_NAMES[:4], hm.QOI_UNITS[:4])):
        print(f'  {n}: [{f[:, i].min():.4f}, {f[:, i].max():.4f}] {u}')
    print(f'maxima of v in (0, 5] ms per sample: {sorted(set(n_before.tolist()))}; '
          f'in (5, 15] ms: {sorted(set(n_window.tolist()))} '
          f'(samples with 2: {int(np.sum(n_window == 2))})')
    print(f'samples with h0 > 1 (outside the physical range [0, 1] of a gating variable): '
          f'{int(np.sum(X[:, hm.NAMES.index("h0")] > 1))} of {len(X)}')
    if args.check and not compare(args.out, HERE, FILES):
        sys.exit(1)


if __name__ == '__main__':
    main()
