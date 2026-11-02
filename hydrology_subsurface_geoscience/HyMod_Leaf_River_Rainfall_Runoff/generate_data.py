"""Write inputs.csv, outputs.csv and gradients.csv for the HyMod Leaf River data set.

    python generate_data.py                       # writes the three CSVs here (about 1 s)
    python generate_data.py --out /tmp/hymod --check

Draws M = 2000 independent uniform (plain Monte Carlo) samples of the five HyMod
parameters on the ranges of the paper's Fig. 4 with numpy's default_rng(SEED),
runs `hymod_model.qoi_and_grad` on the 365-day forcing in forcing.csv, and
writes the physical inputs, the three QoIs (NSE, mean flow, peak flow over days
31-365) and their exact (forward-mode, almost-everywhere) gradients with
respect to the physical parameters.

`--check` compares the new files with the ones next to this script and prints
the largest relative differences. `--forcing-from LeafCatch.txt` writes
forcing.csv (first 365 rows of SAFE's examples/data/LeafCatch.txt, values
unchanged) and forcing_other_years.csv (water years 1950, 1951 and 1959, used
only by the notebook's robustness section) if they are missing, and otherwise
checks the shipped ones against it.

Only numpy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hymod_model as hm  # noqa: E402

SEED = 2015
M = 2000

PROVENANCE = ('hymod_model.py (HyMod as run by SAFE; daily steps, zero initial storages), '
              f'forcing.csv days 1-365, QoIs over days {hm.WARMUP + 1}-365; written by generate_data.py')


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def sample(n=M, seed=SEED):
    rng = np.random.default_rng(seed)
    return hm.LOWER + (hm.UPPER - hm.LOWER) * rng.random((n, 5))


def write(out, X, f, G):
    cols = [f'{n} ({u})' for n, u in zip(hm.NAMES, hm.UNITS)]
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Monte Carlo samples of the 5 HyMod parameters, independent uniform '
                      f'on the ranges of Pianosi et al. (2015) Fig. 4 (numpy default_rng({SEED}))\n'
                      + ','.join(cols))
    qcols = [f'{n} ({u})' for n, u in zip(hm.QOI_NAMES, hm.QOI_UNITS)]
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.15e', delimiter=',',
               header='Nash-Sutcliffe efficiency vs observed flow, mean and peak simulated flow. '
                      + PROVENANCE + '\n' + ','.join(qcols))
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(G), -1), fmt='%.15e', delimiter=',',
               header='d(QoI)/d(parameter), physical units; column 5*k + j is QoI k, parameter j; '
                      'exact forward-mode derivatives (almost everywhere). ' + PROVENANCE + '\n'
                      + ','.join(f'd{q}/d{n}' for q in hm.QOI_NAMES for n in hm.NAMES))


#: Extra 365-day windows for the notebook's robustness section: (water year, first row of LeafCatch.txt, 1-based).
#: Row 1 is 1948-10-01 (14 610 rows = 1948-10-01 .. 1988-09-30), so these are 1 Oct .. 30 Sep exactly.
OTHER_YEARS = ((1950, 366), (1951, 731), (1959, 3653))

LICENCE_NOTE = ('the file states no licence; the source repository (SAFE-python / SAFE-matlab) is GPL-3.0; '
                'hosted with credit, see DATA.md')


def forcing_from(path):
    """Write forcing.csv and forcing_other_years.csv from SAFE's LeafCatch.txt if missing, else compare."""
    d = np.genfromtxt(path, comments='%')
    fmt = ['%d', '%.4f', '%.4f', '%.4f']
    out = os.path.join(HERE, 'forcing.csv')
    if not os.path.exists(out):
        np.savetxt(out, np.column_stack([np.arange(1, 366), d[:365]]), fmt=fmt, delimiter=',',
                   header='Leaf River catchment (Mississippi, USA), first 365 daily rows of LeafCatch.txt '
                          '(SAFE-python examples/data, commit 72c64e5; same values in SAFE-matlab example/hymod, '
                          'commit dc157e1); source data Sorooshian et al. (1983); values unchanged; '
                          + LICENCE_NOTE + '\nday,rain (mm/day),potential evapotranspiration (mm/day),'
                          'observed flow (mm/day)')
        print(f'wrote {out}')
    ship = read_csv(out)[:, 1:]
    print(f'LeafCatch.txt rows 1-365 equal forcing.csv: {np.array_equal(d[:365], ship)}')

    rows = np.concatenate([np.arange(r, r + 365) for _, r in OTHER_YEARS])
    wy = np.repeat([y for y, _ in OTHER_YEARS], 365)
    out = os.path.join(HERE, 'forcing_other_years.csv')
    if not os.path.exists(out):
        np.savetxt(out, np.column_stack([rows, wy, d[rows - 1]]), fmt=['%d'] + fmt, delimiter=',',
                   header='Leaf River catchment, water years (1 Oct - 30 Sep) 1950, 1951 and 1959: rows '
                          + ', '.join(f'{r}-{r + 364}' for _, r in OTHER_YEARS)
                          + ' of LeafCatch.txt (SAFE-python examples/data, commit 72c64e5); used only for the '
                          'notebook robustness check; source data Sorooshian et al. (1983); values unchanged; '
                          + LICENCE_NOTE + '\nLeafCatch.txt row,water year,rain (mm/day),'
                          'potential evapotranspiration (mm/day),observed flow (mm/day)')
        print(f'wrote {out}')
    ship = read_csv(out)
    print(f'LeafCatch.txt rows {", ".join(f"{r}-{r + 364}" for _, r in OTHER_YEARS)} equal forcing_other_years.csv: '
          f'{np.array_equal(ship[:, 0], rows) and np.array_equal(d[rows - 1], ship[:, 2:])}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--forcing-from', help="path to SAFE-python's LeafCatch.txt")
    args = ap.parse_args()
    if os.path.abspath(args.out) == HERE and args.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    if args.forcing_from:
        forcing_from(args.forcing_from)

    t0 = time.perf_counter()
    X = sample()
    f, G = hm.qoi_and_grad(X)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, f, G)
    print(f'wrote {args.out}: {len(X)} samples, {time.perf_counter() - t0:.2f} s')

    if args.check:
        X0 = read_csv(os.path.join(HERE, 'inputs.csv'))
        f0 = read_csv(os.path.join(HERE, 'outputs.csv'))
        G0 = read_csv(os.path.join(HERE, 'gradients.csv')).reshape(G.shape)
        rg = np.linalg.norm(G - G0, axis=2) / np.linalg.norm(G0, axis=2)
        print(f'inputs equal: {np.array_equal(X, X0)}; max relative difference: outputs '
              f'{np.max(np.abs(f - f0) / np.abs(f0)):.1e}, gradients {rg.max():.1e}')


if __name__ == '__main__':
    main()
