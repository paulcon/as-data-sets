"""Recompute the numbers of DATA.md (Four_Chamber_Heart_Electromechanics) that neither the notebook nor
convert_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the four shipped CSVs next to this file. Each line prints the
recomputed value; DATA.md quotes these values.

Checks:
 1. Stored precision of the four-chamber outputs: the largest number of decimals written in the
    volume and pressure columns, and whether every dp/dt value is a whole number; the standard
    deviations of the four dp/dt columns.
 2. Stored precision of the ToR-ORd outputs: whether TTP_Ca and RT90_Ca are whole numbers; their
    standard deviations.
 3. Range of the two conduction velocities CVf_v and CVf_a of the four-chamber inputs (the basis of
    the inferred unit m/s).
 4. ToR-ORd inputs: the largest number of the 2175 equal-width strata (between the sample minimum
    and maximum) that one input occupies, which shows that the rows are not one Latin hypercube
    of 2175 points.

Standard deviations use ddof = 1. Deterministic; takes well under a second. Imports numpy and the
standard library.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    """Values, column names and the data lines as text of one of this data set's CSVs."""
    with open(os.path.join(HERE, name)) as fh:
        fh.readline()
        cols = [c.split(' [')[0] for c in fh.readline()[2:].strip().split(',')]
        lines = [ln.strip() for ln in fh if ln.strip()]
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#'), cols, lines


def decimals(lines, j):
    """Largest number of digits after the decimal point in column j of the data lines."""
    return max(len(v.split('.')[1]) if '.' in v else 0 for v in (ln.split(',')[j] for ln in lines))


def main():
    X, xn, _ = read('fourchamber_inputs.csv')
    F, fn, flines = read('fourchamber_outputs.csv')
    XC, cn, _ = read('torord_ionic_inputs.csv')
    FC, qn, _ = read('torord_ionic_outputs.csv')

    vp = [j for j, q in enumerate(fn) if not q.startswith('dpdt')]
    dp = [j for j, q in enumerate(fn) if q.startswith('dpdt')]
    print(f'1. four-chamber volumes and pressures ({len(vp)} columns): at most '
          f'{max(decimals(flines, j) for j in vp)} decimals written; dp/dt ({len(dp)} columns) all whole '
          f'numbers: {bool(np.all(F[:, dp] == np.round(F[:, dp])))}; sd of dp/dt: '
          + ', '.join(f'{fn[j]} {F[:, j].std(ddof=1):.0f}' for j in dp) + ' mmHg/s')

    for q in ('TTP_Ca', 'RT90_Ca'):
        v = FC[:, qn.index(q)]
        print(f'2. ToR-ORd {q}: all whole numbers {bool(np.all(v == np.round(v)))}; sd {v.std(ddof=1):.1f} ms')

    for n in ('CVf_v', 'CVf_a'):
        v = X[:, xn.index(n)]
        print(f'3. {n}: {v.min():.3f} to {v.max():.3f}')

    N = len(XC)
    lo, hi = XC.min(axis=0), XC.max(axis=0)
    t = (XC - lo) / (hi - lo)
    occ = [np.unique(np.minimum(np.floor(t[:, j] * N), N - 1)).size for j in range(XC.shape[1])]
    print(f'4. ToR-ORd inputs: equal-width strata occupied per input (of {N}): {min(occ)} to {max(occ)}')


if __name__ == '__main__':
    main()
