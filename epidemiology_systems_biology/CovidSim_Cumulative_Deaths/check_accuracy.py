"""Print the numbers of DATA.md's Accuracy section that neither the notebook nor convert_data.py prints.

    python check_accuracy.py                 # checks of the shipped files (numpy only)
    python check_accuracy.py --src DIR       # also the published DAS errors, from the authors' file
                                             # DIR/errors_n_neurons10.hdf5 (needs h5py, tested 3.16.0)

DIR is the COVID19/ folder of https://github.com/wedeling/deep_active_subspace_data (commit 880745c).
Nothing is written.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def shipped():
    X = np.loadtxt(os.path.join(HERE, 'covidsim_inputs.csv'), delimiter=',')
    f = np.loadtxt(os.path.join(HERE, 'covidsim_deaths_day801.csv'), delimiter=',')
    lo, hi = np.loadtxt(os.path.join(HERE, 'covidsim_input_ranges.csv'), delimiter=',')
    with np.load(os.path.join(HERE, 'covidsim_cumulative_deaths_daily.npz')) as a:
        D = a['deaths']
    with open(os.path.join(HERE, 'covidsim_inputs.csv')) as fh:
        fh.readline()
        names = [c.split(' [')[0] for c in fh.readline()[2:].strip().split(',')]
    print(f'death counts integer-valued: {bool(np.all(D == np.round(D)))} (all {D.size} daily values)')
    print(f'first daily column: max {D[:, 0].max():.0f}; runs with deaths at day 801 < 1000: {int(np.sum(f < 1000))}')
    inc = np.diff(D, axis=1)
    print(f'largest one-day increase: {inc.max():.0f}; median day of the largest increase: '
          f'{np.median(np.argmax(inc, axis=1) + 2):.0f}')
    Z = (X - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    print('range coverage of the inputs (share of [lower, upper] spanned by the samples): '
          f'{((X.max(0) - X.min(0)) / (hi - lo)).min():.4f} to {((X.max(0) - X.min(0)) / (hi - lo)).max():.4f}')
    # Kolmogorov-Smirnov statistic of each normalized input against U(-1, 1)
    ks = []
    for j in range(Z.shape[1]):
        z = np.sort(Z[:, j])
        F = (z + 1) / 2
        n = len(z)
        ks.append(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))
    print(f'KS statistic vs uniform: max {max(ks):.4f} (5% critical value for n = 3000: {1.358 / np.sqrt(3000):.4f})')
    C = np.corrcoef(Z, rowvar=False)
    off = np.abs(C[np.triu_indices_from(C, k=1)])
    print(f'largest |sample correlation| between two inputs: {off.max():.4f}')
    j = names.index('Mortality_factor')
    A = np.column_stack([np.ones(len(f)), Z])
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    print(f'Mortality_factor: corr with deaths {np.corrcoef(Z[:, j], f)[0, 1]:.4f}; linear coefficient '
          f'{c[1 + j]:.1f} deaths per half-range (largest |coefficient| {np.abs(c[1:]).max():.1f})')
    f_ratio = f / np.median(f)
    print(f'deaths at day 801 relative to the median: {f_ratio.min():.3f} to {f_ratio.max():.3f}')


def published(src):
    try:
        import h5py
    except ImportError:
        sys.exit('check_accuracy.py --src needs h5py (tested with 3.16.0): pip install h5py, then rerun.')
    path = os.path.join(src, 'errors_n_neurons10.hdf5')
    with open(path, 'rb') as fh:
        print(f'errors_n_neurons10.hdf5 SHA-256 {hashlib.sha256(fh.read()).hexdigest()}')
    fr = np.linspace(0.5, 0.1, 10)        # test fractions of the authors' notebook and recompute script
    size = (1 - fr) * 3000
    with h5py.File(path, 'r') as fh:
        for key in ['err_DAS', 'err_ANN', 'err_ANN_unconstrained']:
            e = fh[key][()] * 100
            print(f'{key}: replicas {e.shape[0]}; mean training / test error e [%] at '
                  f'{size[0]:.0f} training runs {e[:, 0, 0].mean():.2f} / {e[:, 0, 1].mean():.2f}, '
                  f'at {size[-1]:.0f}: {e[:, -1, 0].mean():.2f} / {e[:, -1, 1].mean():.2f}')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', help="the authors' COVID19/ folder (for errors_n_neurons10.hdf5)")
    a = ap.parse_args(argv)
    shipped()
    if a.src:
        published(a.src)
    else:
        print('published DAS errors: skipped (give --src)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
