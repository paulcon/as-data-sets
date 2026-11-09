"""Write the input, output and gradient CSVs of the two parameter studies of `pygem_mb_model`.

    python generate_data.py --out DIR                 # writes the six CSVs to DIR
    python generate_data.py --out DIR --check         # DIR outside this directory; compares with the files here (the safe mode)

--out may not be a folder inside this directory; it may be this directory itself (`--out .`, which
rewrites the shipped files) only without --check.

Study m3: inputs (k_p, T_bias, f_snow), the three parameters calibrated by Rounce et al. (2020).
Study m5: (k_p, T_bias, f_snow, d_prec, T_snow); the paper fixes the precipitation gradient d_prec and
the snow/rain threshold T_snow (p. 177), and freeing them is this data set's choice.

Each study is M = 1000 independent Monte Carlo draws, uniform on the box in `BOX`
(numpy default_rng, seeds 601 and 602). For each draw: B (m w.e. a-1, 2000-2017), the glacier area
(km2) and ice volume (km3) at the end of 2017, and dB/dx with respect to the physical inputs by the
complex step. The gradients are checked against central differences at the first FD_ROWS draws.

Only numpy and the standard library are imported. One process.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pygem_mb_model as pm  # noqa: E402

M = 1000
SEEDS = {'m3': 601, 'm5': 602}
# Box (this data set's choice; see DATA.md): k_p (-), T_bias (degC), f_snow (mm w.e. d-1 degC-1),
# d_prec (m-1), T_snow (degC)
BOX = {
    'k_p': (0.5, 3.5),
    'T_bias': (-2.0, 2.0),
    'f_snow': (1.1, 7.1),
    'd_prec': (0.0, 2e-4),
    'T_snow': (0.0, 2.0),
}
STUDIES = {'m3': pm.NAMES3, 'm5': pm.NAMES5}
FD_ROWS = 50
FD_REL_STEP = 1e-7       # central-difference step, relative to the box width
UNIT_TXT = {'k_p': '-', 'T_bias': 'degC', 'f_snow': 'mm w.e. d-1 degC-1', 'd_prec': 'm-1', 'T_snow': 'degC'}

PROVENANCE = ('pygem_mb_model.py (numpy port of PyGEM v0.1 runmassbalance; Hintereisferner RGI60-11.00897, '
              'ERA5 2000-2017 from OGGM sample data); written by generate_data.py')


def bounds(names):
    lo = np.array([BOX[n][0] for n in names])
    hi = np.array([BOX[n][1] for n in names])
    return lo, hi


def draw(study):
    names = STUDIES[study]
    lo, hi = bounds(names)
    rng = np.random.default_rng(SEEDS[study])
    return lo + (hi - lo) * rng.random((M, len(names)))


def run(X):
    f = np.empty((len(X), 3))
    G = np.empty(X.shape)
    for i, x in enumerate(X):
        r = pm.simulate(*x)
        f[i] = (np.real(r['B']), r['area_annual'][:, -1].sum().real,
                (r['area_annual'][:, -1] * r['thick_annual'][:, -1]).sum().real / 1e3)
        b, G[i] = pm.mass_balance(x, grad=True)
        assert b == f[i, 0]
    return f, G


def fd_check(X, G, names):
    lo, hi = bounds(names)
    err = np.empty(len(X))
    for i, x in enumerate(X):
        gfd = np.empty(len(x))
        for j in range(len(x)):
            h = FD_REL_STEP * (hi[j] - lo[j])
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            gfd[j] = (pm.mass_balance(xp) - pm.mass_balance(xm)) / (2 * h)
        # compare in box-normalized coordinates, where the columns have comparable scale
        s = (hi - lo) / 2
        err[i] = np.linalg.norm((gfd - G[i]) * s) / np.linalg.norm(G[i] * s)
    return err


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, study, X, f, G):
    names = STUDIES[study]
    lo, hi = bounds(names)
    box = '; '.join(f'{n} in [{a:g}, {b:g}]' for n, a, b in zip(names, lo, hi))
    np.savetxt(os.path.join(out, f'inputs_{study}.csv'), X, fmt='%.17g', delimiter=',',
               header=(f'{len(X)} Monte Carlo draws, uniform on the box {box} (seed {SEEDS[study]}). {PROVENANCE}\n'
                       + ','.join(f'{n} ({UNIT_TXT[n]})' for n in names)))
    np.savetxt(os.path.join(out, f'outputs_{study}.csv'), f, fmt='%.17g', delimiter=',',
               header=(f'Glacier-wide specific mass balance 2000-2017 and glacier size at the end of 2017. {PROVENANCE}\n'
                       'B (m w.e. a-1),area_2017 (km2),volume_2017 (km3 ice)'))
    np.savetxt(os.path.join(out, f'gradients_{study}.csv'), G, fmt='%.17g', delimiter=',',
               header=('dB/dx with respect to the PHYSICAL inputs (complex step, the branch the model takes at x). '
                       + PROVENANCE + '\n'
                       + ','.join(f'dB/d{n} (m w.e. a-1 per {UNIT_TXT[n]})' for n in names)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--fd-rows', type=int, default=FD_ROWS, help=f'draws for the finite-difference check (default {FD_ROWS})')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here and (args.check or os.path.realpath(out) != here):
        ap.error('--out must be outside this data set\'s directory (not a folder in it; the directory itself only '
                 'without --check, to rewrite the shipped files)')
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()
    for study in ('m3', 'm5'):
        t0 = time.perf_counter()
        X = draw(study)
        f, G = run(X)
        t1 = time.perf_counter()
        write(out, study, X, f, G)
        print(f'{study}: M = {len(X)}, {t1 - t0:.0f} s ({(t1 - t0) / len(X) * 1e3:.0f} ms per draw with gradient); '
              f'B in [{f[:, 0].min():.3f}, {f[:, 0].max():.3f}] m w.e. a-1; area 2017 in [{f[:, 1].min():.3f}, '
              f'{f[:, 1].max():.3f}] km2 (initial {pm.AREA0.sum():.4f}); draws with no ice left in 2017: '
              f'{int((f[:, 1] == 0).sum())}')
        if args.fd_rows > 0:
            e = fd_check(X[:args.fd_rows], G[:args.fd_rows], STUDIES[study])
            print(f'{study}: complex-step gradients vs central differences (step {FD_REL_STEP:g} x box width), '
                  f'first {args.fd_rows} draws, box-normalized: median relative error {np.median(e):.1e}, '
                  f'max {e.max():.1e}')
        if args.check:
            for n in (f'inputs_{study}.csv', f'outputs_{study}.csv', f'gradients_{study}.csv'):
                a, b = read_csv(os.path.join(out, n)), read_csv(os.path.join(HERE, n))
                # difference relative to each column's largest magnitude
                rel = np.max(np.abs(a - b) / np.abs(b).max(axis=0))
                tol = 1e-15 if n.startswith('inputs') else 1e-9
                print(f'  {n}: max difference from the shipped file, relative to the column maximum, {rel:.1e} '
                      f'(tolerance {tol:g}): {"OK" if rel <= tol else "DIFFERENT"}')
    print(f'total {time.perf_counter() - t_all:.0f} s')


if __name__ == '__main__':
    main()
