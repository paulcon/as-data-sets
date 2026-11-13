"""Write the samples (lv_samples_*.npz) for the competitive Lotka-Volterra tumour model (Zou and Lewis 2026).

    python generate_data.py                                   # writes lv_samples_*.npz here (22 to 60 min, DATA.md)
    python generate_data.py --check                           # regenerates the first block of each
                                                              # design in a temporary directory and
                                                              # compares (about 1.5 min)
    python generate_data.py --check --full                    # all blocks (about 30 min)
    python generate_data.py --out DIR --check --blocks 4      # the first 4 blocks, kept in DIR
    python generate_data.py --out DIR --cache CDIR --max-minutes 9   # resumable chunks

Two designs, each 4096 regions x 25 points = 102,400 points of x = (r_S, r_R, K_S, K_R, gamma_S,
gamma_R) in [0, 1]^6:

  grid4  every cell of a 4^6 grid of the unit cube (cells of width 0.25). A coarser grid than the
         paper's 8^6 cells of width 0.125 (p. 2187), chosen so that both designs together take about an hour (3589 s of model runs measured, DATA.md).
  sub8   4096 of the paper's 8^6 = 262,144 cells of width 0.125: the two cells named on p. 2201
         and 4094 more drawn uniformly without replacement.

Inside every cell, M = 25 points form a Latin hypercube (one point in each of 25 equal slices of
each coordinate, uniform within the slice), as in the paper (p. 2188). The generator for cell c
of design G (G = 4 or 8) is numpy.random.default_rng([SEED, G, c]), with c the C-order flat
index of the cell's multi-index (r_S slowest). Pooled over the cells, each design is a uniform
stratified sample of the unit cube.

For every point it stores q_pri (Eq. 3.3), q_sec (Eq. 3.5) and their exact forward-sensitivity
gradients (tumour_model.qoi). Inputs and values are stored as float64; the gradients are computed
in float64 and stored rounded to float32 (largest relative rounding per row about 6e-8, far below
the error of the paper's forward differences);
reduce_gradients.py applies the same rounding to a float64 file. It also checks the gradients of
the first FD_ROWS grid4 points against central differences (step 1e-6, solves at rtol = atol = 1e-12) and prints the largest
relative error; for q_sec only at points with q_sec (1 - q_sec) > 1e-6, because where q_sec is
within 1e-6 of 0 or 1 its gradient is below about 1e-6 and the differences of q_sec are round-off.

Blocks of 256 cells are cached as .npz files when --cache is given, so an interrupted run
continues where it stopped; --max-minutes stops after the block that passes the limit.
--check prints the largest differences from the shipped files and fails only on structural
mismatches (shapes, NaN pattern, cells, inputs), or when --max-minutes stops it before the end.
Only numpy, scipy and the standard library are imported. One process.
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
import tumour_model as tm  # noqa: E402

SEED = 246
M_LOCAL = 25
N_REGIONS = 4096
BLOCK = 256
NAMED_CELLS_8 = [(1, 2, 3, 4, 1, 4),    # r_S, gamma_S in [0.125, 0.25]; r_R in [0.25, 0.375];
                                        # K_S in [0.375, 0.5]; K_R, gamma_R in [0.5, 0.625] (p. 2201)
                 (7, 7, 7, 7, 7, 7)]    # the top cell [0.875, 1]^6 (p. 2201)
FD_ROWS = 20
H_FD = 1e-6
# --check measures: q_pri relative, q_sec absolute; gradients relative row 2-norm (floor 1e-8 on the
# norm), after rounding the new gradients to float32
INPUT_RTOL = 1e-8           # --check fails if a regenerated input (a seeded draw) differs by more (relative)
GRAD_DTYPE = np.float32     # storage precision of *_grad
DESCRIPTION = (
    'Competitive Lotka-Volterra tumour model of Zou and Lewis 2026 (doi:10.3934/mbe.2026079), '
    'Eqs. 3.1-3.2, S(0) = 0.018, R(0) = 0.002 mm^3; q_pri = trapezoidal integral of S + R over '
    'days 0..56 (mm^3 day), q_sec = S(56)/(S(56) + R(56)); exact forward-sensitivity gradients '
    'with respect to the physical inputs, *_grad[i, k, j] = d q_k / d x_j, computed in float64 and stored '
    'as float32 (X and f are float64). grid4: 25 LHS points in '
    'each of the 4^6 cells of width 0.25; sub8: 25 LHS points in each of 4096 of the 8^6 cells of '
    'width 0.125; *_region = C-order flat cell index. tumour_model.py (LSODA in ln S, ln R, '
    f'rtol = atol = 1e-10); seed {SEED}; written by generate_data.py')


def cells(G):
    """Flat C-order indices of the cells of design G (4: all 4^6; 8: the subset)."""
    if G == 4:
        return np.arange(4 ** 6)
    named = [np.ravel_multi_index(c, (8,) * 6) for c in NAMED_CELLS_8]
    rng = np.random.default_rng([SEED, 8])
    rest = np.setdiff1d(np.arange(8 ** 6), named)
    pick = np.sort(rng.choice(rest, N_REGIONS - len(named), replace=False))
    return np.concatenate([named, pick])


def lhs_cell(G, c):
    """25 Latin-hypercube points in cell c of the G^6 grid, shape (25, 6)."""
    rng = np.random.default_rng([SEED, G, int(c)])
    u = (np.column_stack([rng.permutation(M_LOCAL) for _ in range(6)]) + rng.random((M_LOCAL, 6))) / M_LOCAL
    lower = np.array(np.unravel_index(int(c), (G,) * 6)) / G
    X = lower + u / G
    if np.any(X[:, 2:4] <= 0):
        raise RuntimeError('a carrying capacity of exactly 0 was drawn (probability 2^-53 per draw)')
    return X


def run_block(G, cs):
    X = np.concatenate([lhs_cell(G, c) for c in cs])
    f, Gr = tm.qois(X, grad=True)
    return X, f, Gr


def design(G, nblocks=None, cache=None, deadline=None):
    cs = cells(G)
    blocks = [cs[i:i + BLOCK] for i in range(0, len(cs), BLOCK)]
    if nblocks is not None:
        blocks = blocks[:nblocks]
    out = []
    for b, cb in enumerate(blocks):
        path = None if cache is None else os.path.join(cache, f'G{G}_block{b:03d}.npz')
        if path is not None and os.path.exists(path):
            d = np.load(path)
            out.append((d['X'], d['f'], d['G']))
            continue
        if deadline is not None and time.perf_counter() > deadline:
            return None
        t0 = time.perf_counter()
        X, f, Gr = run_block(G, cb)
        if path is not None:
            np.savez(path, X=X, f=f, G=Gr)
        out.append((X, f, Gr))
        print(f'  design {G}: block {b + 1}/{len(blocks)} done in {time.perf_counter() - t0:.0f} s', flush=True)
    region = np.repeat(np.concatenate(blocks), M_LOCAL)
    return region, np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]), \
        np.concatenate([o[2] for o in out])


def fd_check(X, Gr):
    err = np.zeros((len(X), 2))
    for i, x in enumerate(X):
        g = np.empty((2, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = H_FD
            g[:, j] = (tm.qoi(x + e, rtol=1e-12, atol=1e-12) - tm.qoi(x - e, rtol=1e-12, atol=1e-12)) / (2 * H_FD)
        err[i] = np.linalg.norm(g - Gr[i], axis=1) / np.linalg.norm(Gr[i], axis=1)
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    ap.add_argument('--check', action='store_true',
                    help='compare with the lv_samples_*.npz files in this directory (the first block of each '
                         'design unless --full or --blocks is given)')
    ap.add_argument('--full', action='store_true', help='with --check, regenerate and compare every block')
    ap.add_argument('--blocks', type=int, default=None,
                    help=f'compute only the first BLOCKS blocks of {BLOCK} cells of each design')
    ap.add_argument('--cache', default=None, help='directory for resumable block files')
    ap.add_argument('--max-minutes', type=float, default=None, help='stop after the block that passes this time')
    args = ap.parse_args()
    if args.full and not args.check:
        ap.error('--full is an option of --check')
    if args.full and args.blocks is not None:
        ap.error('--full and --blocks exclude each other')
    if args.check and not args.full and args.blocks is None:
        args.blocks = 1
        print(f'--check: the first block ({BLOCK} cells, {BLOCK * M_LOCAL} points) of each design; '
              '--check --full for all blocks', flush=True)
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            args.out = HERE
    here, out = os.path.realpath(HERE), os.path.realpath(args.out)
    if os.path.commonpath([here, out]) == here and (args.check or args.blocks is not None):
        ap.error('--check and --blocks need --out DIR outside this directory (not this directory or a subdirectory '
                 'of it), so that the shipped file is not overwritten')
    if args.cache is not None and os.path.commonpath([here, os.path.realpath(args.cache)]) == here:
        ap.error('--cache must be outside this directory (not this directory or a subdirectory of it)')
    os.makedirs(out, exist_ok=True)
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    t0 = time.perf_counter()
    deadline = None if args.max_minutes is None else t0 + 60 * args.max_minutes
    res = {}
    for G in (4, 8):
        r = design(G, args.blocks, args.cache, deadline)
        if r is None:
            print(f'stopped after {time.perf_counter() - t0:.0f} s (--max-minutes); rerun with the same --cache to continue')
            if args.check:
                print('FAIL: --check did not finish what it was asked to check')
                sys.exit(1)
            return
        res[G] = r
    t1 = time.perf_counter()
    n_pts = sum(len(res[G][1]) for G in res)
    data = {}
    for G, key in ((4, 'grid4'), (8, 'sub8')):
        region, X, f, Gr = res[G]
        data.update({f'{key}_region': region, f'{key}_X': X, f'{key}_f': f, f'{key}_grad': Gr.astype(GRAD_DTYPE)})
    data.update({
        'names': np.array(tm.NAMES), 'units': np.array(tm.UNITS),
        'qoi_names': np.array(tm.QOI_NAMES), 'qoi_units': np.array(tm.QOI_UNITS),
        'description': np.array(DESCRIPTION),
    })
    paths = tm.save_samples(data, out)
    print(f'wrote {", ".join(os.path.basename(p) for p in paths)} in {out}: {n_pts} points, {t1 - t0:.0f} s for the model runs '
          f'({(t1 - t0) / max(n_pts, 1) * 1e3:.1f} ms per point with sensitivities, including cached blocks)')
    X4 = res[4][1]
    print(f'smallest K_S, K_R drawn: {X4[:, 2].min():.3g}, {X4[:, 3].min():.3g} (grid4); '
          f'{res[8][1][:, 2].min():.3g}, {res[8][1][:, 3].min():.3g} (sub8)')
    e = fd_check(X4[:FD_ROWS], res[4][3][:FD_ROWS])
    qs = res[4][2][:FD_ROWS, 1]
    ok = qs * (1 - qs) > 1e-6   # elsewhere q_sec is within 1e-6 of 0 or 1 and its differences are round-off
    print(f'gradient check vs central differences (h = {H_FD:g}, rtol = atol = 1e-12), first {FD_ROWS} grid4 points: '
          f'max relative error q_pri {e[:, 0].max():.1e}; q_sec {e[ok, 1].max():.1e} ({ok.sum()} points with '
          f'q_sec (1 - q_sec) > 1e-6)')
    failed = 0
    if args.check:
        ref = tm.load_samples()
        largest = (0.0, 'relative, q_pri of grid4')
        for key in ('grid4', 'sub8'):
            n = len(data[f'{key}_X'])
            rows = {len(ref[f'{key}_{a}']) for a in ('region', 'X', 'f', 'grad')}
            shapes = [(ref[f'{key}_{a}'][:n].shape, data[f'{key}_{a}'].shape) for a in ('region', 'X', 'f', 'grad')]
            if rows != {N_REGIONS * M_LOCAL} or any(r != d for r, d in shapes):
                failed += 1
                print(f'FAIL: {key}: shipped rows {sorted(rows)}; shipped vs new shapes {shapes}')
                continue
            fr, fn = ref[f'{key}_f'][:n], data[f'{key}_f']
            gr = ref[f'{key}_grad'][:n].astype(np.float64)            # stored float32
            gn = data[f'{key}_grad'].astype(np.float64)               # new, rounded to float32 above
            Xr, Xn = ref[f'{key}_X'][:n], data[f'{key}_X']
            bad = []
            if not np.array_equal(ref[f'{key}_region'][:n], data[f'{key}_region']):
                bad.append('cells differ')
            if not (np.array_equal(np.isnan(fr), np.isnan(fn)) and np.array_equal(np.isnan(gr), np.isnan(gn))
                    and np.array_equal(np.isnan(Xr), np.isnan(Xn))):
                bad.append('NaN pattern differs')
            dx = np.nanmax(np.abs(Xr - Xn) / np.abs(Xr))
            if dx > INPUT_RTOL:
                bad.append(f'inputs differ by more than {INPUT_RTOL:g} (relative)')
            dg = np.nanmax(np.linalg.norm(gr - gn, axis=2) / np.maximum(np.linalg.norm(gr, axis=2), 1e-8), axis=0)
            d = {'relative, q_pri': np.nanmax(np.abs(fr[:, 0] - fn[:, 0]) / np.abs(fr[:, 0])),
                 'absolute, q_sec': np.nanmax(np.abs(fr[:, 1] - fn[:, 1])),           # in [0, 1]
                 'relative, gradients': dg.max()}
            print(f'{key}: {n} points compared; largest differences from the shipped file: inputs {dx:.1e} '
                  f'(relative), q_pri {d["relative, q_pri"]:.1e} (relative), q_sec {d["absolute, q_sec"]:.1e} '
                  f'(absolute), gradients of q_pri, q_sec {dg[0]:.1e}, {dg[1]:.1e} (relative, row 2-norm)')
            for b in bad:
                print(f'FAIL: {key}: {b}')
            failed += bool(bad)
            for k in d:
                if d[k] > largest[0]:
                    largest = (d[k], f'{k} of {key}')
    print(f'total {time.perf_counter() - t0:.0f} s')
    if args.check:
        print(f'largest difference from the shipped files: {largest[0]:.1e} ({largest[1]})')
    if failed:
        sys.exit(1)

if __name__ == '__main__':
    main()
