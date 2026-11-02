"""Recompute the accuracy numbers quoted in DATA.md for SIPNET_Ecosystem_Carbon_Flux.

    python check_accuracy.py --files-only               # part 1 only: shipped files, numpy, about 1 s
    python check_accuracy.py --sipnet /path/to/sipnet   # parts 1-4 (about 6000 SIPNET runs)

Part 1 needs only the shipped files. Parts 2-4 run SIPNET (the binary that generate_data.py
uses: --sipnet, else $SIPNET, else `sipnet` on PATH); without it the script prints part 1 and
stops with an error that says how to build SIPNET.

1. Small soilWHC: at the 100 samples of fd_check_soilWHC.csv (normalized soilWHC < -0.8), the
   relative 2-norm error of the stored normalized gradient when its soilWHC component is replaced
   by the h = 3e-5 value, and the difference between the h = 1e-5 and h = 3e-5 values (same norm).
2. Smoothness: scans of 401 equally spaced points along each input over its prior range at
   sample 0, all other inputs fixed; for each output the largest step between neighbouring scan
   points as a fraction of the output's range along the scan (a jump would give a value near 1;
   a smooth response about 1/400 to a few hundredths where it is steep).
3. Step study at samples 0-5: relative 2-norm difference of the normalized gradient with steps
   h = 1e-1 ... 1e-4 from the h = 1e-3 gradient (the shipped step), per output.
4. Independent check at 40 samples drawn with numpy default_rng(2018) (without replacement): the
   stored gradient against fourth-order central differences with h = 2e-3 and against central
   differences with h = 3e-4; median, 90th percentile and maximum of the relative 2-norm error.

Deterministic (fixed rows and seed). Run from any directory; it reads the files next to it and
writes nothing.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_data as gd  # noqa: E402

HALF = (gd.UPPER - gd.LOWER) / 2
STEPS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
N_INDEP, SEED_INDEP = 40, 2018


def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def load():
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=',', ndmin=2)  # noqa: E731
    X, G = ld('inputs.csv'), ld('gradients.csv').reshape(-1, 3, 8)
    return X, G * HALF, ld('fd_check_soilWHC.csv')        # normalized gradients (M, 3, 8)


def part1(Gn, D):
    rows, whc = D[:, 0].astype(int), D[:, 1]
    j = gd.NAMES.index('soilWHC')
    thr = gd.LOWER[j] + (gd.FD_CHECK_Z + 1) * HALF[j]
    print(f'1. small soilWHC: {len(rows)} samples with soilWHC < {thr:.2f} cm, '
          f'{np.sum(whc < 0.6)} below 0.6 cm')
    for q, name in enumerate(gd.QOIS):
        g = Gn[rows, q]
        g3 = g.copy(); g3[:, j] = D[:, 2 + q] * HALF[j]
        g1 = g3.copy(); g1[:, j] = D[:, 5 + q] * HALF[j]
        e, e1 = rel(g, g3), rel(g1, g3)
        print(f'   {name}: stored vs h = 3e-5: median {np.median(e):.1e}, above 1 % at {np.sum(e > 0.01)}, '
              f'maximum {e.max():.3g} (soilWHC {whc[e.argmax()]:.2f} cm), maximum for soilWHC >= 0.6 cm '
              f'{e[whc >= 0.6].max():.1e}; h = 1e-5 vs h = 3e-5: maximum {e1.max():.1e}')


def _run(args):
    x, binary = args
    return gd.run_sipnet(x, binary)


def _grad(args):
    x, binary, h = args
    return gd.value_and_gradient((x, binary, h))[1] * HALF


def _grad4(args):
    """Fourth-order central differences (normalized step h; points stay in the box or the row is skipped)."""
    x, binary, h = args
    G = np.empty((3, 8))
    for j in range(8):
        e = np.zeros(8); e[j] = h * HALF[j]
        if np.any(x[j] + 2 * e[j] > gd.UPPER[j]) or np.any(x[j] - 2 * e[j] < gd.LOWER[j]):
            return None
        f = [gd.run_sipnet(x + k * e, binary) for k in (-2, -1, 1, 2)]
        G[:, j] = (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h)
    return G


def part2(X, binary, workers):
    x0 = X[0]
    print('\n2. scans of 401 points along each input at sample 0: largest neighbour step / range of the output')
    with ProcessPoolExecutor(workers) as ex:
        for j, name in enumerate(gd.NAMES):
            pts = np.repeat(x0[None], 401, axis=0)
            pts[:, j] = np.linspace(gd.LOWER[j], gd.UPPER[j], 401)
            F = np.array(list(ex.map(_run, [(p, binary) for p in pts], chunksize=8)))
            step = np.abs(np.diff(F, axis=0)).max(axis=0) / np.maximum(np.ptp(F, axis=0), 1e-300)
            print(f'   {name:16s} ' + ', '.join(f'{q} {s:.3f}' for q, s in zip(gd.QOIS, step)))


def part3(X, Gn, binary, workers):
    print('\n3. step study, samples 0-5: relative difference from the h = 1e-3 gradient')
    with ProcessPoolExecutor(workers) as ex:
        res = {h: np.array(list(ex.map(_grad, [(X[i], binary, h) for i in range(6)])))
               for h in STEPS}
    ref = res[1e-3]
    print(f'   h = 1e-3 recomputed vs stored: max {rel(ref, Gn[:6]).max():.1e}')
    for q, name in enumerate(gd.QOIS):
        for h in STEPS:
            if h != 1e-3:
                print(f'   {name} h = {h:.0e}: ' + ', '.join(f'{v:.1e}' for v in rel(res[h][:, q], ref[:, q])))


def part4(X, Gn, binary, workers):
    rows = np.sort(np.random.default_rng(SEED_INDEP).choice(len(X), N_INDEP, replace=False))
    print(f'\n4. independent check at {N_INDEP} samples (default_rng({SEED_INDEP})): rows {rows.tolist()}')
    with ProcessPoolExecutor(workers) as ex:
        G4 = list(ex.map(_grad4, [(X[i], binary, 2e-3) for i in rows]))
        Gc = np.array(list(ex.map(_grad, [(X[i], binary, 3e-4) for i in rows])))
    keep = np.array([g is not None for g in G4])
    G4 = np.array([g for g in G4 if g is not None])
    print(f'   fourth-order stencil fits in the box at {keep.sum()} of {N_INDEP} samples')
    for q, name in enumerate(gd.QOIS):
        e4 = rel(Gn[rows[keep], q], G4[:, q])
        ec = rel(Gn[rows, q], Gc[:, q])
        e = np.concatenate([e4, ec])
        print(f'   {name}: against fourth order median {np.median(e4):.1e}, 90th percentile '
              f'{np.percentile(e4, 90):.1e}, maximum {e4.max():.1e}; against h = 3e-4 median {np.median(ec):.1e}, '
              f'90th percentile {np.percentile(ec, 90):.1e}, maximum {ec.max():.1e}; both: maximum {e.max():.1e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sipnet', default=None, help='SIPNET binary (default: $SIPNET, else sipnet on PATH)')
    ap.add_argument('--files-only', action='store_true', help='part 1 only (no SIPNET needed)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    t0 = time.perf_counter()
    X, Gn, D = load()
    part1(Gn, D)
    if a.files_only:
        print(f'\nrun time {time.perf_counter() - t0:.1f} s')
        return
    binary = gd.find_sipnet(a.sipnet)          # exits with a message if SIPNET is missing
    part2(X, binary, a.workers)
    part3(X, Gn, binary, a.workers)
    part4(X, Gn, binary, a.workers)
    print(f'\nrun time {time.perf_counter() - t0:.0f} s on {a.workers} workers')


if __name__ == '__main__':
    main()
