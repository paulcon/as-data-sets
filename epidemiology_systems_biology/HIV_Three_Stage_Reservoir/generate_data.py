"""Write inputs.csv, outputs.csv and gradients.csv for the three-stage HIV model (hiv3_model.py).

    python generate_data.py --out DIR                  # all samples, written to DIR
    python generate_data.py --out DIR --check          # the same, then compared with the files here
    python generate_data.py --out DIR --rows 0:20 --check   # a subset, compared with the same rows here
    python generate_data.py --out . --here             # rewrite the shipped files (deliberate)

Draws M = 1000 independent Monte Carlo samples, uniform on the 10 % ranges of Table 1 (p. 6) of
Clarke and Pankavich (2024), with numpy's default generator seeded with SEED = 2024, and computes
T(5 years), T(10 years) and the time to AIDS F with exact (forward-sensitivity) gradients with
respect to the physical parameters.

The samples are computed in blocks of BLOCK rows; each finished block is saved in DIR/_blocks/ and
reused on a rerun (with --here they go to a temporary directory instead), so an interrupted run resumes where it stopped (--max-blocks limits the number of
new blocks per call). The CSVs are written when every block requested is present.

It also checks the gradients against central differences in ln p (step 1e-5, solves at rtol 1e-12)
at the first --fd-rows samples and prints the largest relative error (row 2-norm, per output).

The script never writes into its own directory unless --here is given; --check and --rows need an
--out directory outside it. Only numpy, scipy and the standard library are imported. One process.
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hiv3_model as hm  # noqa: E402

SEED = 2024
M = 1000
BLOCK = 50
H_FD = 1e-5
FD_ROWS = 10
TOL_OUT = 1e-6     # relative, elementwise, for --check
TOL_GRAD = 1e-5    # relative, row 2-norm of each 15-vector, for --check

PROVENANCE = (f'hiv3_model.py (Clarke & Pankavich 2024 Eq. 1, Table 1; y(0) = (1000, 0, 150, 0, 0.01); '
              f'LSODA rtol {hm.RTOL:g} atol {hm.ATOL:g}; 1 year = 365 days); numpy default_rng({SEED}) '
              f'uniform on the Table 1 ranges; written by generate_data.py')


def draw(seed=SEED, M=M):
    """M independent uniform samples of the 15 physical parameters, shape (M, 15)."""
    rng = np.random.default_rng(seed)
    return hm.LOWER + (hm.UPPER - hm.LOWER) * rng.random((M, 15))


def fd_check(X, G):
    err = np.zeros((len(X), 3))
    for i, p in enumerate(X):
        Gl = G[i] * p
        Gfd = np.empty((3, 15))
        for j in range(15):
            e = np.zeros(15)
            e[j] = H_FD
            Gfd[:, j] = (hm.qoi(p * np.exp(e), rtol=1e-12, atol=1e-10)
                         - hm.qoi(p * np.exp(-e), rtol=1e-12, atol=1e-10)) / (2 * H_FD)
        err[i] = np.linalg.norm(Gfd - Gl, axis=1) / np.linalg.norm(Gl, axis=1)
    return err.max(axis=0)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, f, G):
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Monte Carlo samples of the 15 physical parameters, independent uniform on the '
                      f'ranges of Table 1 (Clarke & Pankavich 2024, p. 6). {PROVENANCE}\n'
                      + ','.join(f'{n} ({u})' for n, u in zip(hm.NAMES, hm.UNITS)))
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.17g', delimiter=',',
               header=f'T-cell count at 5 and 10 years and time to AIDS. {PROVENANCE}\n'
                      'T5 (cells/mm^3),T10 (cells/mm^3),F (years)')
    cols = [f'd{q}/d{n} ({qu} per {u})' for q, qu in zip(hm.QOI_NAMES, hm.QOI_UNITS)
            for n, u in zip(hm.NAMES, hm.UNITS)]
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), 45), fmt='%.17g', delimiter=',',
               header='Exact gradients with respect to the PHYSICAL parameters (forward sensitivities); '
                      'column 15 q + j is d(output q)/d(parameter j), q = T5, T10, F. ' + PROVENANCE
                      + '\n' + ','.join(cols))


def parse_rows(s):
    a, b = s.split(':')
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory (this directory only with --here)')
    ap.add_argument('--here', action='store_true', help='allow --out to be this directory (rewrites the shipped files)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--rows', type=parse_rows, help='compute only rows a:b of the design')
    ap.add_argument('--fd-rows', type=int, default=FD_ROWS, help=f'samples for the finite-difference check (default {FD_ROWS})')
    ap.add_argument('--max-blocks', type=int, default=None, help='compute at most this many new blocks, then stop')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if out == HERE and not args.here:
        ap.error('--out is this directory: add --here to rewrite the shipped files')
    if out == HERE and (args.check or args.rows):
        ap.error('--check and --rows need --out DIR outside this directory')
    os.makedirs(out, exist_ok=True)
    # finished blocks are kept under DIR/_blocks (a temporary directory with --here)
    blocks = tempfile.mkdtemp(prefix='hiv3_blocks_') if args.here else os.path.join(out, '_blocks')
    os.makedirs(blocks, exist_ok=True)

    t0 = time.perf_counter()
    X = draw()
    a, b = args.rows if args.rows else (0, M)
    new = 0
    fs, Gs = [], []
    for start in range(a, b, BLOCK):
        stop = min(start + BLOCK, b)
        path = os.path.join(blocks, f'rows_{start:04d}_{stop:04d}.npz')
        if os.path.exists(path):
            d = np.load(path)
            if not np.array_equal(d['X'], X[start:stop]):
                raise RuntimeError(f'{path} was computed from different inputs; delete it')
            fs.append(d['f'])
            Gs.append(d['G'])
            continue
        if args.max_blocks is not None and new >= args.max_blocks:
            print(f'stopped after {new} new blocks ({time.perf_counter() - t0:.0f} s); rerun to continue')
            return
        tb = time.perf_counter()
        f, G = hm.qois(X[start:stop], grad=True)
        np.savez(path, X=X[start:stop], f=f, G=G)
        new += 1
        print(f'rows {start}:{stop} done in {time.perf_counter() - tb:.0f} s', flush=True)
        fs.append(f)
        Gs.append(G)
    f = np.concatenate(fs)
    G = np.concatenate(Gs)
    Xr = X[a:b]
    t1 = time.perf_counter()
    write(out, Xr, f, G)
    print(f'wrote {out}: rows {a}:{b} ({len(Xr)} samples), {t1 - t0:.0f} s for the model runs in this call')
    print(f'T5 in [{f[:, 0].min():.2f}, {f[:, 0].max():.2f}], T10 in [{f[:, 1].min():.3f}, {f[:, 1].max():.2f}] '
          f'cells/mm^3; F in [{f[:, 2].min():.3f}, {f[:, 2].max():.3f}] years')
    if args.fd_rows > 0:
        e = fd_check(Xr[:args.fd_rows], G[:args.fd_rows])
        print(f'gradient check vs central differences in ln p (h = {H_FD:g}, rtol 1e-12), first {args.fd_rows} '
              f'samples: max relative error T5 {e[0]:.1e}, T10 {e[1]:.1e}, F {e[2]:.1e} '
              f'({time.perf_counter() - t1:.0f} s)')
    if args.check:
        Xs, fsh, Gsh = (read_csv(os.path.join(HERE, n))[a:b] for n in ('inputs.csv', 'outputs.csv', 'gradients.csv'))
        rx = np.max(np.abs(Xr - Xs) / np.abs(Xs))
        rf = np.max(np.abs(f - fsh) / np.abs(fsh))
        Gsh = Gsh.reshape(-1, 3, 15)
        rg = np.max(np.linalg.norm(G - Gsh, axis=2) / np.linalg.norm(Gsh, axis=2))
        ok = rx < 1e-15 and rf < TOL_OUT and rg < TOL_GRAD
        print(f'max relative differences from the shipped files: inputs {rx:.1e}, outputs {rf:.1e} '
              f'(tolerance {TOL_OUT:g}), gradients {rg:.1e} (row norm; tolerance {TOL_GRAD:g}): '
              + ('PASS' if ok else 'FAIL'))
        if not ok:
            sys.exit(1)
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
