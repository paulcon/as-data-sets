"""Write the six CSVs of the steel-column data set (two configurations x inputs, outputs, gradients).

    python generate_data.py --out DIR            # writes the six CSVs to DIR
    python generate_data.py --check              # writes them to a temporary directory and compares
                                                 # them with the shipped files
    python generate_data.py --out DIR --check    # the same, keeping the files in DIR

DIR must be given without `--check`. `--check` refuses this directory and any directory inside it,
so the shipped files are never overwritten by a check; it prints the largest differences from the
shipped files and fails only on structural mismatches (shapes, NaN pattern, inputs). To rewrite the
shipped files, pass `--out .` from this directory without `--check`.

For each configuration of steel_column_model.CONFIGS ('benchmark', 'paper') it draws M = 2000
independent Monte Carlo samples u ~ N(0, I_9) with numpy's default generator (seed SEEDS[config]),
maps them to the physical inputs through the inverse CDFs, and evaluates the limit state G (MPa)
and its exact gradient with respect to the physical inputs. It then checks the gradients, mapped to
u, against central differences in u (step H_FD) at the first FD_ROWS samples and prints the
largest relative error (row 2-norm).

Only numpy, scipy and the standard library are imported. One process, about a second.
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
import steel_column_model as sm  # noqa: E402

M = 2000
SEEDS = {'benchmark': 736, 'paper': 1997}
H_FD = 1e-6
FD_ROWS = 100
INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only


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
    ap.add_argument('--out', default=None,
                    help='output directory (required without --check; with --check, a temporary directory by default)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            ap.error('--out DIR is required without --check')
    here = os.path.realpath(HERE)
    out_real = os.path.realpath(args.out)
    if args.check and os.path.commonpath([out_real, here]) == here:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    bad, worst = [], (-1.0, '')
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
            for n in ('inputs', 'outputs', 'gradients'):
                f = f'{config}_{n}.csv'
                a, b = read_csv(os.path.join(args.out, f)), read_csv(os.path.join(HERE, f))
                if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                    bad.append(f'FAIL: {f}: shape {a.shape} or NaN pattern differs from the shipped {b.shape}')
                    continue
                r = np.nanmax(np.abs(a - b) / np.abs(b))
                print(f'  {f}: largest relative difference (elementwise) from the shipped file {r:.1e}')
                if n == 'inputs' and r > INPUT_RTOL:
                    bad.append(f'FAIL: {f} differs from the shipped inputs by {r:.1e} > {INPUT_RTOL:g}')
                worst = max(worst, (r, f))
    print(f'total {time.perf_counter() - t0:.1f} s')
    if args.check:
        print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
        for line in bad:
            print(line)
        if bad:
            sys.exit(1)

if __name__ == '__main__':
    main()
