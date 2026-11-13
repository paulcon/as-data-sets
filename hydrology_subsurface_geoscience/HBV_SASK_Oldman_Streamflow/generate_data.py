"""Write inputs.csv, outputs.csv, gradients.csv and tt_tangent.csv for the HBV-SASK Oldman data set.

    python generate_data.py --out DIR            # write the four CSVs to DIR
    python generate_data.py --check              # write them to a temporary directory and compare
    python generate_data.py --out DIR --check    # also compare them with the files shipped here

Draws M = 2000 independent uniform (Monte Carlo) samples of the 12 HBV-SASK parameters on the
box of Table 2 (Razavi et al. 2019, p. 105) with numpy's default generator seeded with
SEED = 689, runs `hbv_sask_model` from 1979-01-01 to 2008-12-31, and writes three quantities
of interest over 1982-01-01 to 1996-12-31 (NSE, Qmean, Qpeak) with their gradients with
respect to the physical parameters.

Gradient columns: the 11 parameters other than TT carry the exact forward (tangent)
sensitivity. The TT column is the central difference (f(TT + H_TT) - f(TT - H_TT)) / (2 H_TT) with
H_TT = 0.1 degC (TT +- 0.1 degC), because the
rain/snow switch makes the outputs jump whenever TT crosses the temperature of a wet day
(temperatures are recorded to 0.01 degC), and the tangent does not see those jumps. The
tangent derivative with respect to TT is written separately to tt_tangent.csv.

The script also checks the tangent against central differences (step 1e-7 of each range) at
the first FD_ROWS samples and prints the largest relative error (row 2-norm, coordinates scaled
to [-1, 1]).

--out is required without --check: the shipped files are never overwritten by default. --check
without --out uses a temporary directory and removes it. With --check, --out must be outside this directory (neither this directory nor any directory inside it). --check prints the largest
differences from the shipped files (each scaled by the largest absolute value of its column) and fails
(exit status 1) only on structural mismatches (shapes, NaN pattern, inputs differing by more than
INPUT_RTOL). Only numpy and the standard library are imported.
One process.
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
import hbv_sask_model as hm  # noqa: E402

SEED = 689
M = 2000
H_TT = 0.1        # degC, central-difference step for the TT column
H_FD = 1e-7       # fraction of each range, for the tangent check only
FD_ROWS = 20
CHUNK = 250
INPUT_RTOL = 1e-8  # inputs are deterministic draws: a larger difference means a different seed or range
FILES = ('inputs.csv', 'outputs.csv', 'gradients.csv', 'tt_tangent.csv')

PROVENANCE = (f'hbv_sask_model.py (HBV-SASK, Razavi et al. 2019 Fig. 9; Oldman River forcing 1979-2008 '
              f'from VARS-TOOL); uniform Monte Carlo on the Table 2 box, numpy default_rng({SEED}); '
              'written by generate_data.py')


def draw(seed=SEED, M=M):
    rng = np.random.default_rng(seed)
    return hm.LB + (hm.UB - hm.LB) * rng.random((M, 12))


def evaluate(X):
    """Outputs (M, 3), mixed gradients (M, 3, 12), tangent dQ/dTT (M, 3)."""
    F, G, T = [], [], []
    for a in range(0, len(X), CHUNK):
        x = X[a:a + CHUNK]
        f, g = hm.qoi(x, grad=True)
        xp, xm = x.copy(), x.copy()
        xp[:, 0] += H_TT
        xm[:, 0] -= H_TT
        T.append(g[:, :, 0].copy())
        g[:, :, 0] = (hm.qoi(xp) - hm.qoi(xm)) / (2 * H_TT)
        F.append(f)
        G.append(g)
    return np.vstack(F), np.vstack(G), np.vstack(T)


def fd_check(X):
    """Largest row relative error of the tangent gradient against central differences."""
    f, g = hm.qoi(X, grad=True)
    w = hm.UB - hm.LB
    n = len(X)
    Xp = np.repeat(X, 12, axis=0)
    Xm = Xp.copy()
    for j in range(12):
        Xp[j::12, j] += H_FD * w[j]
        Xm[j::12, j] -= H_FD * w[j]
    d = (hm.qoi(Xp) - hm.qoi(Xm)) / (2 * H_FD * np.tile(w, n))[:, None]
    gfd = d.reshape(n, 12, 3).transpose(0, 2, 1)
    s = w / 2
    return np.max(np.linalg.norm((g - gfd) * s, axis=2) / np.linalg.norm(g * s, axis=2), axis=0)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, F, G, T):
    bounds = '; '.join(f'{n} [{lo:g}, {hi:g}]' for n, lo, hi in zip(hm.NAMES, hm.LB, hm.UB))
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} uniform Monte Carlo samples of the HBV-SASK parameters (Table 2, p. 105): '
                      f'{bounds}. {PROVENANCE}\n'
                      + ','.join(f'{n} ({u})' for n, u in zip(hm.NAMES, hm.UNITS)))
    np.savetxt(os.path.join(out, 'outputs.csv'), F, fmt='%.17g', delimiter=',',
               header='Oldman River at Waldron\'s Corner, 1982-01-01 to 1996-12-31: Nash-Sutcliffe efficiency of '
                      'daily streamflow, mean flow, mean annual maximum daily flow. ' + PROVENANCE + '\n'
                      'NSE (-),Qmean (m3/s),Qpeak (m3/s)')
    cols = []
    for q, qu in zip(hm.QOI_NAMES, hm.QOI_UNITS):
        for n, u in zip(hm.NAMES, hm.UNITS):
            cols.append(f'd{q}/d{n} ({qu} per {u})')
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), 36), fmt='%.17g', delimiter=',',
               header='Gradients with respect to the PHYSICAL parameters; columns 0-11 NSE, 12-23 Qmean, 24-35 Qpeak. '
                      f'TT columns: central difference with step {H_TT} degC; other columns: exact forward '
                      'sensitivities. ' + PROVENANCE + '\n' + ','.join(cols))
    np.savetxt(os.path.join(out, 'tt_tangent.csv'), T, fmt='%.17g', delimiter=',',
               header='Forward (tangent) derivative with respect to TT, which omits the jumps at the rain/snow '
                      'switch. ' + PROVENANCE + '\n'
                      'dNSE/dTT (1/degC),dQmean/dTT (m3/s/degC),dQpeak/dTT (m3/s/degC)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (required without --check; with --check, default: a temporary directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs shipped in this directory')
    ap.add_argument('--fd-rows', type=int, default=FD_ROWS,
                    help=f'samples for the tangent check (default {FD_ROWS}; 0 to skip)')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            ap.error('--out DIR is required without --check')
    here, out = os.path.realpath(HERE), os.path.realpath(args.out)
    if args.check and os.path.commonpath([here, out]) == here:
        ap.error('--check needs --out DIR outside this directory (not this directory or a subdirectory of it)')
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    X = draw()
    F, G, T = evaluate(X)
    t1 = time.perf_counter()
    write(args.out, X, F, G, T)
    print(f'wrote {args.out}: M = {len(X)}, {t1 - t0:.1f} s for the model runs')
    print('NSE range [{:.3f}, {:.4f}], Qmean range [{:.3f}, {:.3f}] m3/s, Qpeak range [{:.2f}, {:.2f}] m3/s'.format(
        F[:, 0].min(), F[:, 0].max(), F[:, 1].min(), F[:, 1].max(), F[:, 2].min(), F[:, 2].max()))
    if args.fd_rows > 0:
        e = fd_check(X[:args.fd_rows])
        print(f'tangent vs central differences (step {H_FD:g} of each range), first {args.fd_rows} samples: '
              f'max row relative error NSE {e[0]:.1e}, Qmean {e[1]:.1e}, Qpeak {e[2]:.1e}')
    if args.check:
        ok, diffs = True, []
        for n in FILES:
            a, b = read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n))
            if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                print(f'FAIL: {n}: shape {a.shape} or NaN pattern differs from the shipped file {b.shape}')
                ok = False
                continue
            scale = np.maximum(np.nanmax(np.abs(b), axis=0), 1e-300)
            r = float(np.nanmax(np.abs(a - b) / scale))
            print(f'{n}: largest difference / column maximum {r:.1e}')
            if n == 'inputs.csv' and r > INPUT_RTOL:
                print(f'FAIL: {n}: the inputs differ from the shipped ones by more than {INPUT_RTOL:g} (relative)')
                ok = False
            diffs.append((r, n))
        if diffs:
            r, n = max(diffs, key=lambda t: t[0])
            print(f'largest difference from the shipped files: {r:.1e} (relative, {n})')
        if not ok:
            sys.exit(1)
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
