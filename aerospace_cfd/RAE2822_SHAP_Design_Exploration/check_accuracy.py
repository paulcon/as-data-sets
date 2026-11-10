"""Recompute the Accuracy numbers of DATA.md that neither the notebook nor convert_data.py prints.

Stored precision of the outputs (decimal places written in the authors' file) against their spread.
Numpy only; reads the shipped CSV files of this directory (nothing is written).

    python check_accuracy.py
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    F = np.loadtxt(os.path.join(HERE, 'rae2822_hh_outputs.csv'), delimiter=',')
    with open(os.path.join(HERE, 'rae2822_hh_outputs.csv')) as fh:
        rows = [ln.strip().split(',') for ln in fh if not ln.startswith('#')]
    for j, q in enumerate(['Cd', '|Cm|']):
        dec = max(len(r[j].split('.')[1]) if '.' in r[j] else 0 for r in rows)
        step = 10.0**-dec
        sd = F[:, j].std(ddof=1)
        print(f'{q}: at most {dec} decimals written (resolution {step:.0e}); sample sd {sd:.3e}; '
              f'resolution / sd = {step / sd:.1e}')


if __name__ == '__main__':
    main()
