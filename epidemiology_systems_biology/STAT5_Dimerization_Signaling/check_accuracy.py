"""Recompute the accuracy numbers of DATA.md for STAT5_Dimerization_Signaling.

    python check_accuracy.py --petab DIR      # all checks, 7-14 min; DIR is a clone of the PEtab benchmark collection
    PETAB_BENCHMARK_DIR=DIR python check_accuracy.py
    python check_accuracy.py --skip-petab     # all checks but 3
    python check_accuracy.py --petab DIR --skip-slow   # without check 2 (264 solves of 0.5-1.2 s), 3.5 min

Check 3 needs the PEtab benchmark collection, github.com/Benchmarking-Initiative/Benchmark-Models-PEtab,
at commit c44d5ede9bf9 (only problems/Boehm_JProteomeRes2014/v1/ is read). Without it, and without
--skip-petab, the script stops with an error saying what to fetch. Nothing else is needed: the
PEtab tables are read with the standard library.

1. The full PEtab box [-5, 5]^6 (log10): all 64 corners and 200 random points solved with
   `STAT5_model.qoi` (Radau, rtol = atol = 1e-10), with the median time per solve; then 300 further
   random points: the fraction with time-averaged pSTAT5A_rel below 1 %, the fraction between 5 and
   95 %, and the fraction of all observable values at the 15 times below 1 % or above 99 % (DATA.md:
   all solved, median 0.04-0.06 s; 79.0 %, 16 %, 54 %). Random points: numpy default_rng(1).
2. The study box (LOWER, UPPER): all 64 corners and 200 random points solved, with the median time
   per solve and the range of the time-averaged pSTAT5A_rel (DATA.md: all solved, median 0.5-1.2 s;
   0.93 to 99.06 %).
   Random points: numpy default_rng(2). `--skip-slow` leaves this check out.
3. Against the benchmark repository: at the PEtab best fit, the 3 observables at the 16 measurement
   times against the repository's simulations.tsv (largest absolute difference, per cent), and the
   Gaussian negative log-likelihood of the 48 measurements with the best-fit noise parameters
   (DATA.md: 2.0e-6; 138.222).
4. Outputs against a tighter solve: at rows 0, 200, 400, 600, 800, outputs.csv against `qoi` at
   rtol = atol = 1e-12 (largest relative difference; DATA.md: 4.6e-9).
5. Gradients against tighter sensitivities: same rows, gradients.csv against `qoi_and_grad` at
   rtol = atol = 1e-12 (relative 2-norm error of each output's 6-vector, worst output; DATA.md: 2.7e-9).
6. Gradients against central differences of `qoi` at rtol = atol = 1e-12, step h = 1e-3 in xi,
   same rows, against gradients.csv: relative 2-norm error for pSTAT5A_rel_mean, and the largest
   over the 48 outputs (DATA.md: 1.0e-7 to 5.6e-7; 9.5e-5). At the best fit, the error for
   pSTAT5A_rel_mean against `qoi_and_grad` at 1e-12 for h = 1e-2, 3e-3, 1e-3, 3e-4 (DATA.md:
   1.5e-4, 1.3e-5, 1.5e-6, 1.3e-7).

The re-solve of 3 samples with sensitivities is printed by the notebook, and `generate_data.py
--check` compares a rerun with the files. Imports numpy, scipy, the standard library and the
modules of this directory. Deterministic (fixed rows and seeds); it reads the files here and the
benchmark tables, and writes nothing.
"""
import argparse
import csv
import itertools
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import STAT5_model as sm  # noqa: E402
from generate_data import read_csv  # noqa: E402

PETAB_URL = 'https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab'
PETAB_COMMIT = 'c44d5ede9bf9'
PETAB_SUB = os.path.join('problems', 'Boehm_JProteomeRes2014', 'v1')
ROWS = [0, 200, 400, 600, 800]
MEAN = sm.QOI_NAMES.index('pSTAT5A_rel_mean')


def petab_dir(arg):
    d = arg or os.environ.get('PETAB_BENCHMARK_DIR')
    if not d or not os.path.isfile(os.path.join(d, PETAB_SUB, 'simulations.tsv')):
        sys.exit(f'error: check 3 needs the PEtab benchmark collection ({"not found at " + d if d else "no path given"}).\n'
                 f'Fetch {PETAB_URL} at commit {PETAB_COMMIT}, e.g.\n'
                 f'    git clone {PETAB_URL}.git DIR && git -C DIR checkout {PETAB_COMMIT}\n'
                 f'then rerun with --petab DIR (or PETAB_BENCHMARK_DIR=DIR), or with --skip-petab for the other checks.')
    return d


def read_tsv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def corners(lo, hi):
    return np.array([[(lo, hi)[b][j] for j, b in enumerate(bits)]
                     for bits in itertools.product((0, 1), repeat=len(lo))])


def solve_all(P):
    """qoi at each row of P; returns outputs, times and the number of failed solves."""
    F, T, fail = [], [], 0
    for p in P:
        t0 = time.perf_counter()
        try:
            F.append(sm.qoi(p))
        except (RuntimeError, ValueError):
            fail += 1
            F.append(np.full(len(sm.QOI_NAMES), np.nan))
        T.append(time.perf_counter() - t0)
    return np.array(F), np.array(T), fail


def rel_rows(G, Gref):
    return np.linalg.norm(G - Gref, axis=-1)/np.linalg.norm(Gref, axis=-1)


def cd(x, h):
    """Central differences of qoi (rtol = atol = 1e-12) with step h in xi, shape (48, 6)."""
    G = np.empty((len(sm.QOI_NAMES), len(x)))
    for j in range(len(x)):
        e = np.zeros(len(x)); e[j] = h
        G[:, j] = (sm.qoi(x + e, rtol=1e-12, atol=1e-12) - sm.qoi(x - e, rtol=1e-12, atol=1e-12))/(2*h)
    return G


def check_petab(d):
    base = os.path.join(d, PETAB_SUB)
    try:
        rev = subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = 'unknown (not a git checkout)'
    par = {r['parameterId']: float(r['nominalValue']) for r in read_tsv(os.path.join(base, 'parameters.tsv'))}
    same = (np.array_equal([par[n] for n in sm.NAMES], sm.NOMINAL_THETA)
            and par['ratio'] == sm.RATIO and par['specC17'] == sm.SPECC17)
    sims = read_tsv(os.path.join(base, 'simulations.tsv'))
    meas = read_tsv(os.path.join(base, 'measurements.tsv'))
    times = np.array(sorted({float(r['time']) for r in sims}))
    _, O = sm.trajectory(sm.NOMINAL, times)
    model = lambda r: O[sm.OBS.index(r['observableId']), np.searchsorted(times, float(r['time']))]  # noqa: E731
    dsim = max(abs(model(r) - float(r['simulation'])) for r in sims)
    nll = 0.
    for r in meas:
        s = par[r['noiseParameters']]
        nll += 0.5*math.log(2*math.pi*s*s) + 0.5*((float(r['measurement']) - model(r))/s)**2
    print(f'\n3. PEtab benchmark at {d} (commit {rev[:12]}); best-fit values equal STAT5_model.NOMINAL_THETA, ratio '
          f'and specC17: {same}')
    print(f'   observables at the {len(times)} measurement times against simulations.tsv ({len(sims)} values): '
          f'largest absolute difference {dsim:.1e} % (DATA.md: 2.0e-6)')
    print(f'   negative log-likelihood of the {len(meas)} measurements, best-fit noise parameters '
          f'({", ".join(f"{k} {par[k]:.4g}" for k in sorted({r["noiseParameters"] for r in meas}))}): '
          f'{nll:.3f} (DATA.md: 138.222)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--petab', help='clone of the PEtab benchmark collection (default: $PETAB_BENCHMARK_DIR)')
    ap.add_argument('--skip-petab', action='store_true', help='leave out check 3')
    ap.add_argument('--skip-slow', action='store_true', help='leave out check 2')
    a = ap.parse_args()
    d = None if a.skip_petab else petab_dir(a.petab)     # exits here if the benchmark is missing
    t_start = time.perf_counter()

    P = np.vstack([corners(sm.PETAB_LOWER, sm.PETAB_UPPER),
                   sm.PETAB_LOWER + (sm.PETAB_UPPER - sm.PETAB_LOWER)*np.random.default_rng(1).random((500, 6))])
    F, T, fail = solve_all(P)
    print(f'1. full PEtab box [-5, 5]^6: 64 corners and 200 random points, {fail} failed solves, median '
          f'{np.median(T[:264]):.2f} s per solve; 300 further random points, {np.isnan(F[264:, 0]).sum()} failed')
    F3 = F[264:]
    obs = F3[:, 3:]
    print(f'   of the 300: time-averaged pSTAT5A_rel below 1 % {np.mean(F3[:, MEAN] < 1):.1%}, between 5 and 95 % '
          f'{np.mean((F3[:, MEAN] > 5) & (F3[:, MEAN] < 95)):.1%}; observable values at the 15 times below 1 % or '
          f'above 99 % {np.mean((obs < 1) | (obs > 99)):.1%} (DATA.md: all solved, 0.04-0.06 s; 79.0 %, 16 %, 54 %)')

    P = np.vstack([corners(sm.LOWER, sm.UPPER),
                   sm.LOWER + (sm.UPPER - sm.LOWER)*np.random.default_rng(2).random((200, 6))])
    F, T, fail = solve_all(P) if not a.skip_slow else (None, None, None)
    if a.skip_slow:
        print('\n2. skipped (--skip-slow)')
    else:
        print(f'\n2. study box: 64 corners and 200 random points, {fail} failed solves, median {np.median(T):.2f} s per '
              f'solve; time-averaged pSTAT5A_rel {np.nanmin(F[:, MEAN]):.2f} to {np.nanmax(F[:, MEAN]):.2f} % '
              f'(corners {np.min(F[:64, MEAN]):.2f} to {np.max(F[:64, MEAN]):.2f} %) (DATA.md: all solved, 0.5-1.2 s; 0.93 to 99.06 %)')

    if d is None:
        print('\n3. skipped (--skip-petab)')
    else:
        check_petab(d)

    X = read_csv(os.path.join(HERE, 'inputs.csv'))
    F = read_csv(os.path.join(HERE, 'outputs.csv'))
    G = read_csv(os.path.join(HERE, 'gradients.csv')).reshape(len(X), len(sm.QOI_NAMES), 6)
    r4, r5, e_mean, e_all = [], [], [], []
    for r in ROWS:
        f12 = sm.qoi(X[r], rtol=1e-12, atol=1e-12)
        _, g12 = sm.qoi_and_grad(X[r], rtol=1e-12, atol=1e-12)
        e = rel_rows(G[r], cd(X[r], 1e-3))
        r4.append(np.max(np.abs(F[r] - f12)/np.abs(f12)))
        r5.append(np.max(rel_rows(G[r], g12)))
        e_mean.append(e[MEAN]); e_all.append(e.max())
    print(f'\n4. outputs.csv against qoi at rtol = atol = 1e-12, rows {ROWS}: largest relative difference '
          f'{max(r4):.1e} (per row {", ".join(f"{v:.1e}" for v in r4)}) (DATA.md: 4.6e-9)')
    print(f'5. gradients.csv against qoi_and_grad at 1e-12, same rows: worst output {max(r5):.1e} '
          f'(per row {", ".join(f"{v:.1e}" for v in r5)}) (DATA.md: 2.7e-9)')
    print(f'6. central differences, h = 1e-3, against gradients.csv, same rows: pSTAT5A_rel_mean {min(e_mean):.1e} to '
          f'{max(e_mean):.1e}; largest over the 48 outputs {max(e_all):.1e} (DATA.md: 1.0e-7 to 5.6e-7; 9.5e-5)')
    _, g_nom = sm.qoi_and_grad(sm.NOMINAL, rtol=1e-12, atol=1e-12)
    steps = (1e-2, 3e-3, 1e-3, 3e-4)
    errs = [rel_rows(cd(sm.NOMINAL, h)[MEAN], g_nom[MEAN]) for h in steps]
    print('   best fit, pSTAT5A_rel_mean against qoi_and_grad at 1e-12: '
          + ', '.join(f'h = {h:g}: {v:.1e}' for h, v in zip(steps, errs))
          + ' (DATA.md: 1.5e-4, 1.3e-5, 1.5e-6, 1.3e-7)')
    print(f'\nrun time {time.perf_counter() - t_start:.0f} s')


if __name__ == '__main__':
    main()
