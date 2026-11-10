"""Recompute the numbers of DATA.md (BISON_Fuel_Performance_Surrogates) that neither the notebook nor
convert_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the two shipped CSVs next to this file. Deterministic; well under a
second; numpy and the standard library.

Checks:
 1. Stored precision: the largest number of significant digits written in each column.
 2. The R^2 ceiling set by the output rounding: 1 - (step^2 / 12) / variance, with step the stored
    rounding step (smallest difference between distinct values), for each output.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    with open(os.path.join(HERE, name)) as fh:
        fh.readline()
        cols = [c.split(' [')[0] for c in fh.readline()[2:].strip().split(',')]
        lines = [ln.strip() for ln in fh if ln.strip()]
    return np.loadtxt(os.path.join(HERE, name), delimiter=','), cols, lines


def sig_digits(text):
    mant = text.lower().split('e')[0].replace('-', '').replace('.', '').lstrip('0')
    return len(mant.rstrip('0')) if '.' not in text.lower().split('e')[0] else len(mant)


def main():
    for name in ('bison_fp_inputs.csv', 'bison_fp_outputs.csv'):
        V, cols, lines = read(name)
        print(f'1. {name}: largest number of significant digits written per column')
        for j, c in enumerate(cols):
            print(f'     {c:25s} {max(sig_digits(ln.split(",")[j]) for ln in lines)}')
    F, cols, _ = read('bison_fp_outputs.csv')
    print('2. R^2 ceiling from the stored rounding of each output')
    for j, c in enumerate(cols):
        step = np.min(np.diff(np.unique(F[:, j])))
        print(f'     {c:25s} step {step:.3g}; ceiling {1 - step**2 / 12 / F[:, j].var(ddof=1):.4f}')


if __name__ == '__main__':
    main()
