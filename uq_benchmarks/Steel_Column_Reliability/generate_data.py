"""Write the six CSVs of the steel-column data set (two configurations x inputs, outputs, gradients).

    python generate_data.py --out DIR            # writes the six CSVs to DIR
    python generate_data.py --out DIR --check    # ... and compares them with the shipped files

DIR must be given. `--check` refuses DIR = this directory, so the shipped files are never
overwritten by a check; to rewrite the shipped files, pass `--out .` from this directory without
`--check`.

For each configuration of steel_column_model.CONFIGS ('benchmark', 'paper') it draws M = 2000
independent Monte Carlo samples u ~ N(0, I_9) with numpy's default generator (seed SEEDS[config]),
maps them to the physical inputs through the inverse CDFs, and evaluates the limit state G (MPa)
and its exact gradient with respect to the physical inputs. It then checks the gradients, mapped to
u, against central differences in u (step H_FD) at the first FD_ROWS samples and prints the
largest relative error (row 2-norm).

Only numpy, scipy and the standard library are imported. One process, about a second.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steel_column_model as sm  # noqa: E402

M = 2000
SEEDS = {'benchmark': 736, 'paper': 1997}
H_FD = 1e-6
FD_ROWS = 100
RTOL = 1e-12   # --check tolerance (relative, elementwise)


def build(config):
    mg = sm.Marginals(config)
    U, X = mg.sample(M, np.random.default_rng(SEEDS[config]))
    G, dG = sm.limit_state(X, grad=True)
    return mg, U, X, G, dG


def fd_check(config, U, dGu):
    E = np.eye(9) * H_FD
    fd = np.column_stack([(sm.limit_state_u(U + E[j], config) - sm.limit_state_u(U - E[j], config))
                          / (2 * H_FD) for j in range(9)])
    return np.max(np.linalg.norm(fd - dGu, axis=1) / np.linalg.norm(dGu, axis=1))


def write(out, config, X, G, dG):
    c = sm.CONFIGS[config]
    mean = sm.means(config)
    prov = (f'steel_column_model.py (Kuschel and Rackwitz 1997, pp. 326-327; s = 7500 mm); configuration '
            f"'{config}': design (b, d, h) = ({c['b']:g}, {c['d']:g}, {c['h']:g}) mm, mean of E = "
            f"{c['E_mean']:g} MPa; {M} Monte Carlo samples u ~ N(0, I_9), numpy default_rng({SEEDS[config]}), "
            'x = F^-1(Phi(u)); written by generate_data.py')
    cols = ','.join(f'{n} ({u})' for n, u in zip(sm.NAMES, sm.UNITS))
    np.savetxt(os.path.join(out, f'{config}_inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'Physical inputs; distributions {list(sm.DISTRIBUTIONS)}, means {mean.tolist()}, '
                      f'sd {sm.SD.tolist()}. {prov}\n{cols}')
    np.savetxt(os.path.join(out, f'{config}_outputs.csv'), G[:, None], fmt='%.17g', delimiter=',',
               header=f'Limit state G = Fs - P (1/As + F0/Ms * Eb/(Eb - P)); G <= 0 is failure. {prov}\n'
                      'G (MPa)')
    gcols = ','.join(f'dG/d{n} (MPa/{u})' for n, u in zip(sm.NAMES, sm.UNITS))
    np.savetxt(os.path.join(out, f'{config}_gradients.csv'), dG, fmt='%.17g', delimiter=',',
               header=f'Exact gradient of G with respect to the PHYSICAL inputs (closed form). {prov}\n'
                      + gcols.replace('MPa/MPa', '-'))


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory (required)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    if args.check and os.path.abspath(args.out) == HERE:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    for config in sm.CONFIGS:
        mg, U, X, G, dG = build(config)
        write(args.out, config, X, G, dG)
        Eb_P = sm.euler_load(X) / X[:, 1:4].sum(axis=1)
        print(f"{config}: M = {M}, G in [{G.min():.3f}, {G.max():.3f}] MPa, {int((G <= 0).sum())} samples with "
              f"G <= 0, Eb/P in [{Eb_P.min():.4f}, {Eb_P.max():.4f}], {int((X[:, 7] < 0).sum())} samples with F0 < 0")
        e = fd_check(config, U[:FD_ROWS], dG[:FD_ROWS] * mg.dx_du(U[:FD_ROWS]))
        print(f'  gradient check vs central differences in u (h = {H_FD:g}), first {FD_ROWS} samples: '
              f'max relative error {e:.1e}')
        if args.check:
            rel = [np.max(np.abs(read_csv(os.path.join(args.out, f'{config}_{n}.csv'))
                                 - read_csv(os.path.join(HERE, f'{config}_{n}.csv')))
                          / np.abs(read_csv(os.path.join(HERE, f'{config}_{n}.csv'))))
                   for n in ('inputs', 'outputs', 'gradients')]
            ok = all(r <= RTOL for r in rel)
            print(f'  max relative differences from the shipped files (inputs, outputs, gradients): '
                  f"{[f'{r:.1e}' for r in rel]} -> {'PASS' if ok else 'FAIL'} (tolerance {RTOL:g})")
    print(f'total {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
