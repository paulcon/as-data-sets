"""Regenerate the data files of Point_Heat_Source_Consolidation.

    python generate_data.py                          writes the four CSVs here (about 1 s)
    python generate_data.py --out DIR --check        writes them to DIR and compares with the files here

Draws M = 1000 independent samples of the eight inputs by the inverse-CDF transform of
numpy.random.default_rng(1985).random((M, 8)) and evaluates heat_source_model.py:

    inputs.csv     (M, 8)    physical inputs K, C, rho, E, a_s, k_s, nu, phi
    outputs.csv    (M, 3)    dT (K), p (Pa), u_r (m) at r = 6 m, t = 365 d
    gradients.csv  (M, 24)   exact gradients of the three outputs w.r.t. the physical inputs
    fields.csv     (M, 300)  dT, p, u_r on the 10 x 10 (r, t) grid; column 100*field + 10*i_r + i_t
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heat_source_model as hm  # noqa: E402

SEED = 1985
M = 1000
SRC = ('heat_source_model.py (Booker & Savvidou 1985, Eq. 23; inputs and grid of UQTestFuns '
       'SaturatedPointHeatSource / Chaudhry et al. 2021); inverse-CDF draws from numpy '
       f'default_rng({SEED}).random((M, 8)); written by generate_data.py')


def compare(out, here, files):
    """Max relative difference (per element, NaNs must match) between the files in out and here."""
    for f in files:
        a = np.loadtxt(os.path.join(out, f), delimiter=',', comments='#', ndmin=2)
        b = np.loadtxt(os.path.join(here, f), delimiter=',', comments='#', ndmin=2)
        ok = np.isfinite(b)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        rel = np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300)) if ok.any() else 0.0
        print(f'{f}: max relative difference from the shipped file {rel:.1e}'
              + ('' if same_nan else ' (NaN pattern differs)'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    if args.check and os.path.abspath(args.out) == HERE:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    X = hm.sample(M, SEED)
    F, G = hm.headline(X)
    (dT, p, u), _ = hm.fields_and_grads(X)
    fields = np.stack([dT, p, u], axis=1).reshape(M, -1)
    units = [f'[{u_}]' for u_ in hm.UNITS]
    fmt = '%.17g'
    np.savetxt(os.path.join(args.out, 'inputs.csv'), X, delimiter=',', fmt=fmt,
               header=f'{M} samples of the 8 physical inputs, independent. {SRC}\n'
                      + ','.join(f'{n} {u_}' for n, u_ in zip(hm.NAMES, units)))
    np.savetxt(os.path.join(args.out, 'outputs.csv'), F, delimiter=',', fmt=fmt,
               header=f'temperature change dT, excess pore pressure p and radial displacement u_r at '
                      f'r = {hm.R_STAR:g} m, t = {hm.T_STAR / hm.DAY:g} d. {SRC}\n'
                      'dT [K],p [Pa],u_r [m]')
    qunits = {'dT': 'K', 'p': 'Pa', 'u_r': 'm'}
    gcols = [f'd{q}/d{n} [{qunits[q]} per {u_.strip("[]")}]' for q in qunits for n, u_ in zip(hm.NAMES, units)]
    np.savetxt(os.path.join(args.out, 'gradients.csv'), G.reshape(M, -1), delimiter=',', fmt=fmt,
               header=f'exact gradients with respect to the PHYSICAL inputs at r = {hm.R_STAR:g} m, '
                      f't = {hm.T_STAR / hm.DAY:g} d: columns 0-7 ddT/dx, 8-15 dp/dx, 16-23 du_r/dx. {SRC}\n'
                      + ','.join(gcols))
    fcols = [f'{q}(r={r:g} m;t={tl}) [{qunits[q]}]' for q in qunits for r in hm.R_GRID for tl in hm.T_LABELS]
    np.savetxt(os.path.join(args.out, 'fields.csv'), fields, delimiter=',', fmt=fmt,
               header=f'dT (K), p (Pa), u_r (m) on the 10 x 10 (r, t) grid; column = 100*field + 10*i_r + i_t '
                      f'(t fastest). {SRC}\n' + ','.join(fcols))
    print(f'wrote inputs.csv, outputs.csv, gradients.csv, fields.csv ({M} samples) to {args.out} in '
          f'{time.perf_counter() - t0:.1f} s')
    if args.check:
        compare(args.out, HERE, ('inputs.csv', 'outputs.csv', 'gradients.csv', 'fields.csv'))


if __name__ == '__main__':
    main()
