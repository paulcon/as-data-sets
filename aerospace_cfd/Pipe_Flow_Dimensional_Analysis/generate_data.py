"""Write laminar.csv, turbulent.csv and high_re.csv for the viscous pipe flow model.

    python generate_data.py --out DIR               # writes the three CSVs to DIR
    python generate_data.py --out DIR --check       # writes to DIR, compares with the files here
                                                    # (exit status 1 if any file differs beyond the tolerance)

For each flow case (Tables 2-4 of Constantine, del Rosario and Iaccarino, arXiv:1708.04303) it
draws M = 1000 independent Monte Carlo samples, uniform on the box of the physical variables
q = (rho, mu, D, eps, V), with numpy's default generator seeded with [SEED, k] (SEED = 1708,
k = 0, 1, 2 for laminar, turbulent, high_re), and computes the pressure loss per unit length
dp/dx and its exact gradient with respect to ln q (`pipe_model.pressure_gradient`).

It also checks the gradients against central finite differences in ln q (step 1e-6) at the
first FD_ROWS samples of each file and prints the largest relative error (row 2-norm).

The script never writes into its own directory: --out DIR (outside it) is required.
Only numpy and the standard library are imported. One process; about a second.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import pipe_model as pm  # noqa: E402

SEED = 1708
M = 1000
H_FD = 1e-6
FD_ROWS = 100
CHECK_RTOL = 1e-12      # tolerance of --check (relative, elementwise)
FILES = {r: f'{r}.csv' for r in pm.REGIMES}


def draw(k, regime):
    return pm.sample(regime, M, np.random.default_rng([SEED, k]))


def fd_check(Q, G):
    """Largest relative error (row 2-norm) of G = d(dp/dx)/d ln q against central differences."""
    X = np.log(Q)
    Gfd = np.empty_like(G)
    for j in range(5):
        e = np.zeros(5)
        e[j] = H_FD
        Gfd[:, j] = (pm.pressure_gradient(np.exp(X + e)) - pm.pressure_gradient(np.exp(X - e))) / (2 * H_FD)
    return np.max(np.linalg.norm(Gfd - G, axis=1) / np.linalg.norm(G, axis=1))


def write(path, regime, k, Q, dp, G):
    lo, up = pm.BOUNDS[regime]
    head1 = (f'Viscous pipe flow, {regime} case: {len(Q)} Monte Carlo samples uniform on the box of '
             f'{pm.SOURCES[regime]} of arXiv:1708.04303 (lower {lo.tolist()}, upper {up.tolist()}); '
             f'dp/dx = lam rho V^2/(2D), lam from the Colebrook equation (Newton); gradients exact, with '
             f'respect to ln q. pipe_model.py; numpy default_rng([{SEED}, {k}]); written by generate_data.py')
    cols = [f'{n} ({u})' for n, u in zip(pm.NAMES, pm.UNITS)] + ['dpdx (kg m^-2 s^-2)'] + \
           [f'd(dpdx)/d(ln {n}) (kg m^-2 s^-2)' for n in pm.NAMES]
    np.savetxt(path, np.column_stack([Q, dp, G]), fmt='%.17g', delimiter=',',
               header=head1 + '\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory, outside this directory')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    out = os.path.realpath(args.out)                      # symbolic links resolved
    if os.path.commonpath([out, HERE]) == HERE:
        ap.error('--out must be a directory outside this data set, so that the shipped files are not overwritten')
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    failed = 0
    for k, regime in enumerate(pm.REGIMES):
        Q = draw(k, regime)
        dp, G = pm.pressure_gradient(Q, grad=True)
        write(os.path.join(out, FILES[regime]), regime, k, Q, dp, G)
        Re = pm.reynolds(Q)
        msg = (f'{regime:9s}: M = {len(Q)}, Re in [{Re.min():.4g}, {Re.max():.4g}], '
               f'eps/D in [{pm.relative_roughness(Q).min():.3g}, {pm.relative_roughness(Q).max():.3g}], '
               f'dp/dx in [{dp.min():.4g}, {dp.max():.4g}] kg m^-2 s^-2; '
               f'gradient vs central differences (h = {H_FD:g}, first {FD_ROWS} rows): '
               f'max relative error {fd_check(Q[:FD_ROWS], G[:FD_ROWS]):.1e}')
        if args.check:
            new = np.loadtxt(os.path.join(out, FILES[regime]), delimiter=',')
            old = np.loadtxt(os.path.join(HERE, FILES[regime]), delimiter=',')
            rel = np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-300))
            failed += rel > CHECK_RTOL
            msg += f'; max relative difference from the shipped file {rel:.1e} (tolerance {CHECK_RTOL:g}: ' \
                   f'{"pass" if rel <= CHECK_RTOL else "FAIL"})'
        print(msg)
    print(f'wrote {len(pm.REGIMES)} files in {time.perf_counter() - t0:.2f} s')
    if failed:
        print(f'--check FAILED for {failed} file(s)')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
