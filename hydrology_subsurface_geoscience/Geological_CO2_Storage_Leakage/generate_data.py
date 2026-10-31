"""Write the input, output and gradient CSVs of the 1-D CO2 leakage model for the Ketzin and Quest sites.

    python generate_data.py                       # writes the six CSVs here
    python generate_data.py --out /tmp/c --check  # writes elsewhere, compares with the files here

For each site, draws M = 2000 independent samples of the inputs of `co2_leakage_model`
(Ketzin: m = 22, Table 3 and Sec. 3.1; Quest: m = 30, Table 4 and Sec. 3.2 of Amrhein, Most,
Walther 2026) by the inverse CDF of numpy.random.default_rng(SEEDS[site]).random((M, m)), and
evaluates the total leakage time t_total (years, Eq. 7), the minimum layer flow rate V_min
(m^3/year per m^2, Eq. 6), its smooth p-norm soft minimum V_soft (exponent 10), and the exact
gradients of log10 t_total and log10 V_soft with respect to the physical inputs.

It also checks the exact gradients against central finite differences (step 1e-6 times each
input's range) at every sample and prints the largest relative error (row 2-norm).

Only numpy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import co2_leakage_model as cm  # noqa: E402

SEEDS = {'ketzin': 2139, 'quest': 2140}
M = 2000
H_FD = 1e-6  # central-difference step in units of each input's range (check only)
FILES = [f'{s}_{k}.csv' for s in SEEDS for k in ('inputs', 'outputs', 'gradients')]


def provenance(site):
    return (f'co2_leakage_model.py (Amrhein, Most, Walther 2026, doi:10.20944/preprints202609.2139.v1, '
            f'Eqs. 3-8, 11-14); site {site}; inverse-CDF draws from numpy default_rng({SEEDS[site]})'
            f'.random((M, m)); written by generate_data.py')


def draw(site, M=M):
    return cm.sample(M, np.random.default_rng(SEEDS[site]), site)


def evaluate(X, site):
    out = np.column_stack([cm.leakage_time(X, site), cm.min_flow(X, site), cm.soft_min_flow(X, site)])
    G = np.column_stack([cm.grad_log10_leakage_time(X, site), cm.grad_log10_soft_min_flow(X, site)])
    return out, G


def fd_check(X, site):
    """Largest relative errors (row 2-norm) of the exact gradients vs central differences."""
    lo, hi = cm.bounds(site)
    errs = []
    for f, g in [(lambda Z: np.log10(cm.leakage_time(Z, site)), cm.grad_log10_leakage_time),
                 (lambda Z: np.log10(cm.soft_min_flow(Z, site)), cm.grad_log10_soft_min_flow)]:
        G = g(X, site)
        F = np.empty_like(G)
        for j in range(X.shape[1]):
            dx = np.zeros(X.shape[1])
            dx[j] = H_FD * (hi[j] - lo[j])
            F[:, j] = (f(X + dx) - f(X - dx)) / (2 * dx[j])
        errs.append(np.max(np.linalg.norm(F - G, axis=1) / np.linalg.norm(G, axis=1)))
    return errs


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, site, X, F, G):
    nm, un = cm.names(site), cm.units(site)
    m = len(nm)
    pv = provenance(site)
    np.savetxt(os.path.join(out, f'{site}_inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} samples of the {m} physical inputs ({site}), independent: logK_i symmetric '
                      f'triangular, h_i uniform +/- 5 m, p_pump symmetric triangular +/- 4.5 MPa, u_eta U[0,1]. '
                      f'{pv}\n' + ','.join(f'{n} [{u}]' for n, u in zip(nm, un)))
    np.savetxt(os.path.join(out, f'{site}_outputs.csv'), F, fmt='%.17g', delimiter=',',
               header=f'total leakage time (Eq. 7), minimum layer flow rate (Eq. 6) and its p-norm soft minimum '
                      f'(exponent {cm.Q_SOFT:g}), unit area A = 1 m^2. {pv}\n'
                      't_total [years],V_min [m^3/year],V_soft [m^3/year]')
    gT = [f'dlog10(t_total)/d{n} [1/({u})]' for n, u in zip(nm, un)]
    gV = [f'dlog10(V_soft)/d{n} [1/({u})]' for n, u in zip(nm, un)]
    np.savetxt(os.path.join(out, f'{site}_gradients.csv'), G, fmt='%.17g', delimiter=',',
               header=f'exact gradients with respect to the PHYSICAL inputs: columns 0-{m - 1} of log10 t_total, '
                      f'columns {m}-{2 * m - 1} of log10 V_soft. {pv}\n' + ','.join(gT + gV))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for site in SEEDS:
        t0 = time.perf_counter()
        X = draw(site)
        F, G = evaluate(X, site)
        write(args.out, site, X, F, G)
        t1 = time.perf_counter()
        print(f'{site}: wrote {len(X)} x {X.shape[1]} inputs in {t1 - t0:.2f} s; t_total in '
              f'[{F[:, 0].min():.1f}, {F[:, 0].max():.1f}] years, V_min in [{F[:, 1].min():.4g}, '
              f'{F[:, 1].max():.4g}] m^3/year, max log10(V_min/V_soft) = {np.log10(F[:, 1] / F[:, 2]).max():.4f}')
        eT, eV = fd_check(X, site)
        print(f'  exact gradients vs central differences (step 1e-6 range): max relative error '
              f'log10 t_total {eT:.1e}, log10 V_soft {eV:.1e}')
    if args.check:
        same = [np.array_equal(read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n)))
                for n in FILES]
        print('files identical to the shipped ones:', dict(zip(FILES, same)))


if __name__ == '__main__':
    main()
