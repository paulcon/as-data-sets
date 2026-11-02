"""Recompute the simulator-noise and time-step numbers of the Accuracy section of DATA.md
(Pyloric_Circuit_Energy_Consumption).

    python check_accuracy.py --pyloric /path/to/pyloric

It needs the outside simulator pyloric (github.com/mackelab/pyloric, commit 89dd27d) with pandas
and Cython, set up as in DATA.md, "Regenerating"; without it the script exits with status 1 and
says what to install. Run it from any directory; it reads the shipped files next to this file
and runs pyloric through generate_data.run_one. Each line prints the recomputed value. About
40 pyloric runs (one to a few minutes).

Checks:
 1. Rows 0 and 1 rerun with the settings of the files (dt = 0.025 ms, noise seed 1): largest
    relative difference from outputs.csv.
 2. Simulator noise: rows 0-7 rerun with noise seeds 2, 3 and 4. Relative standard deviation of
    E_total over the four seeds (1 = the files): minimum, median, maximum and its row; the AB/PD
    spike counts of that row; the mean over the 8 rows of the noise variance of E_total as a share
    of the variance of E_total over the 1100 shipped samples; changes of the pyloric_like flag.
 3. Time step: rows 0-2 with the noise switched off (noise_std = 0) at dt = 0.025 ms and
    dt = 0.0125 ms: relative difference of E_total.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_data as gd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--pyloric', help='path to a clone of github.com/mackelab/pyloric at commit 89dd27d')
    a = ap.parse_args()
    _, simulate, summary_stats, prior_bounds, select_names = gd.import_pyloric(a.pyloric)  # exits 1 if missing
    t0 = time.perf_counter()
    lo, hi = prior_bounds(gd.pyloric_setups())
    tn, cn = select_names(gd.pyloric_setups())
    theta = gd.design(lo, hi)
    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',')
    Y = np.loadtxt(os.path.join(HERE, 'outputs.csv'), delimiter=',')
    dx = np.max(np.abs(gd.to_physical(theta) - X) / np.abs(X).clip(1e-300))
    print(f'0. design regenerated from seed {gd.SEED_X}: max relative difference from inputs.csv {dx:.1e}')

    def run(i, **kw):
        return gd.run_one(theta[i], simulate, summary_stats, tn, cn, **kw)

    # 1. reproduction of rows 0 and 1
    d = max(np.nanmax(np.abs(run(i) - Y[i]) / np.abs(Y[i]).clip(1e-300)) for i in (0, 1))
    print(f'1. rows 0 and 1, settings of the files: max relative difference from outputs.csv {d:.1e}')

    # 2. noise seeds
    rows = np.arange(8)
    runs = np.array([[Y[i]] + [run(i, seed=s) for s in (2, 3, 4)] for i in rows])     # (8, 4, 9)
    E = runs[:, :, 3]
    rsd = E.std(axis=1, ddof=1) / E.mean(axis=1)
    k = int(np.argmax(rsd))
    share = np.mean(E.var(axis=1, ddof=1)) / Y[:, 3].var(ddof=1)
    flips = int(np.sum(runs[:, 1:, 7] != runs[:, :1, 7]))
    print(f'2. rows 0-7, noise seeds 1-4: relative SD of E_total min {100 * rsd.min():.2f} %, median '
          f'{100 * np.median(rsd):.1f} %, max {100 * rsd.max():.1f} % (row {rows[k]}, AB/PD spikes '
          f'{int(runs[k, :, 4].min())}-{int(runs[k, :, 4].max())}); mean noise variance / variance over the '
          f'1100 samples {100 * share:.2f} %; pyloric_like changes {flips} of {runs[:, 1:].shape[0] * 3}')

    # 3. time step, noise off
    out = []
    for i in (0, 1, 2):
        e1 = run(i, noise_std=0.0)[3]
        e2 = run(i, noise_std=0.0, dt=gd.DT / 2)[3]
        out.append(f'row {i} {100 * abs(e2 - e1) / e1:.2f} %')
    print('3. noise off, dt = 0.025 vs 0.0125 ms, relative difference of E_total: ' + ', '.join(out))
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
