"""Recompute the numbers of DATA.md (EU_DEMO_PROCESS_Major_Radius) that the notebook does not print.

    python check_accuracy.py --csv-only   checks 1-4, from demo_runs.csv (numpy and scipy; about a second)
    python check_accuracy.py              checks 1-7 (checks 5-7 need PROCESS v3.4.2; 7 PROCESS
                                          optimisations, about 10-20 CPU minutes)

Run it from any directory; it reads the shipped files next to this file and writes nothing there
(PROCESS's scratch files go to a temporary directory). Each line prints the recomputed value; DATA.md
quotes these values.

Checks:
 1. The MD5 checksum of demo_runs.csv.
 2. The design: the inputs against the first 64 points of scipy.stats.qmc.Sobol(d=6, scramble=True,
    seed=675).random_base2(7) mapped onto the Table I box (largest absolute difference), and the number
    of distinct rows.
 3. The two failed rows: ifail, VMCON iterations, and R0 against the warm start R0 of
    baseline_warmstart.json.
 4. Constraints at the converged runs: the largest sqsumsq (equality residuals), the ranges of P_net and
    t_burn, and the largest relative difference between the achieved P_sep B_T/(q95 A R0) and the
    input limit.
 5. Repeatability (PROCESS): rows 2 and 62 rerun with the warm start of the stored runs; R0 against the
    file.
 6. Start-point dependence (PROCESS): rows 1 and 5, and the failed rows 28 and 58, rerun from the deck's
    own starting values with R0 started at 9.0 m (cold start), with the stored runs' cap of 50 VMCON
    iterations per attempt; R0 and ifail against the file.
 7. The warm start at the baseline point itself (PROCESS): R0 against the cold-start baseline optimum
    held in baseline_warmstart.json.

Checks 5-7 need PROCESS v3.4.2 (see DATA.md, "Regenerating", for the install recipe), imported
through process_driver.py; without PROCESS the script stops with an install message after check 4.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_data as gd  # noqa: E402
from process_driver import run_one  # noqa: E402

COLD_ROWS = (1, 5, 28, 58)
WARM_ROWS = (2, 62)


def load():
    data = np.loadtxt(os.path.join(HERE, 'demo_runs.csv'), delimiter=',')
    with open(os.path.join(HERE, 'demo_runs.csv')) as fh:
        fh.readline()
        cols = [c.split(' [')[0] for c in fh.readline().lstrip('# ').strip().split(',')]
    return data, {c: data[:, i] for i, c in enumerate(cols)}


def check_csv(data, col, base):
    md5 = hashlib.md5(open(os.path.join(HERE, 'demo_runs.csv'), 'rb').read()).hexdigest()
    print(f'1. MD5 of demo_runs.csv: {md5}')
    X = data[:, 1:7]
    d = np.max(np.abs(X - gd.design(len(X))))
    print(f"2. inputs vs the first {len(X)} points of Sobol'(d = 6, seed {gd.SEED}) on the Table I box: "
          f'largest absolute difference {d:.1e}; {len(np.unique(X, axis=0))} distinct rows')
    conv = col['converged'] == 1
    for i in np.flatnonzero(~conv):
        print(f"3. failed row {int(col['index'][i])}: ifail {col['ifail'][i]:.0f}, VMCON iterations "
              f"{col['n_iter'][i]:.0f}, R0 {float(col['R0'][i])!r} m (warm start R0 {base['itv']['rmajor']!r} m)")
    ratio = col['PsepBt_qAR'][conv] / X[conv, 1]
    print(f"4. converged runs ({conv.sum()}): largest sqsumsq {col['sqsumsq'][conv].max():.2e}; "
          f"P_net {col['P_net'][conv].min():.6f} to {col['P_net'][conv].max():.6f} MW; "
          f"t_burn {col['t_burn'][conv].min():.6f} to {col['t_burn'][conv].max():.6f} s; "
          f'achieved P_sep B_T/(q95 A R0) / input limit: {ratio.min():.9f} to {ratio.max():.9f}')


def check_process(data, col, base):
    tpl = gd.template()
    warm = {k: v for k, v in base['itv'].items() if k != 'rmajor'}
    r0w = base['itv']['rmajor']
    with tempfile.TemporaryDirectory() as work, contextlib.chdir(work):   # PROCESS writes process.log here
        for i in WARM_ROWS:
            res, _ = run_one(tpl, data[i, 1:7], work, f'w{i}', rmajor0=r0w, maxcal=gd.MAXCAL, warm=warm)
            print(f"5. row {i}, warm start: R0 {res['rmajor']:.10f} m (file {col['R0'][i]:.10f}), "
                  f"ifail {res['ifail']:.0f}, relative difference {abs(res['rmajor'] - col['R0'][i]) / col['R0'][i]:.1e}; "
                  f"{res['cpu_s']:.0f} s CPU")
        for i in COLD_ROWS:
            res, _ = run_one(tpl, data[i, 1:7], work, f'c{i}', rmajor0=9.0, maxcal=gd.MAXCAL, warm=None)
            print(f"6. row {i}, cold start: R0 {res['rmajor']:.6f} m, ifail {res['ifail']:.0f} (file: R0 "
                  f"{col['R0'][i]:.6f} m, ifail {col['ifail'][i]:.0f}); difference {res['rmajor'] - col['R0'][i]:+.4f} m; "
                  f"{res['cpu_s']:.0f} s CPU")
        res, _ = run_one(tpl, gd.BASE, work, 'base', rmajor0=r0w, maxcal=gd.MAXCAL, warm=warm)
        print(f"7. baseline point, warm start: R0 {res['rmajor']:.6f} m, ifail {res['ifail']:.0f} (cold-start "
              f"optimum {r0w:.6f} m); difference {1e3 * (res['rmajor'] - r0w):+.1f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv-only', action='store_true', help='checks 1-4 only (no PROCESS needed)')
    a = ap.parse_args()
    t0 = time.perf_counter()
    data, col = load()
    base = json.load(open(gd.BASELINE_FILE))
    check_csv(data, col, base)
    if a.csv_only:
        return
    if importlib.util.find_spec('process') is None:
        sys.exit(gd.INSTALL.replace('rerun generate_data.py.', 'rerun check_accuracy.py (or use --csv-only).'))
    check_process(data, col, base)
    print(f'run time {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
