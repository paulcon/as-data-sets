"""Recompute the numbers of DATA.md that neither the notebook nor generate_data.py prints (Zika_SEIR_SEI_Brazil).

    python check_accuracy.py          (20 to 70 s, depending on the load)

Prints, beside the values DATA.md quotes:

1. Outputs against a tighter solve: at seven rows (0-4 and the rows with the largest and
   smallest C52), the largest relative differences of C52 and J in outputs.csv (DOP853 at the
   shipped tolerances, with sensitivities) from Radau at rtol 1e-13, atol 1e-18 (scaled states).
2. The wider vector range of problem 3: the 1000 shipped input rows with E_vi and I_vi
   stretched from [0, 1e-3] to [0, 5e-3] (the seed-2018 draw on the wider box), solved with
   zika_model.qoi; the median C52, and its ratio to the total of data_cases.csv.

The gradient check against central differences is `generate_data.py --fd-check 20`, and the
notebook prints its re-solve of five rows.

Deterministic (fixed rows); imports numpy, scipy and zika_model; reads the CSVs next to it and
writes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zika_model as zm  # noqa: E402

WIDE = 5e-3     # upper bound of E_vi and I_vi in check 2


def read(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def radau_qoi(x, y):
    """(C52, J) from a Radau solve at rtol 1e-13, atol 1e-18."""
    C = zm.trajectory(x, rtol=1e-13, atol=1e-18, method='Radau')[:, 7]
    res = y - zm.new_cases(C)
    return np.array([C[-1], res @ res])


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    t0 = time.perf_counter()
    X, F = read('inputs.csv'), read('outputs.csv')
    y = zm.load_cases()

    rows = [0, 1, 2, 3, 4, int(np.argmax(F[:, 0])), int(np.argmin(F[:, 0]))]
    R = np.array([radau_qoi(X[r], y) for r in rows])
    d = np.max(np.abs(F[rows] - R) / np.abs(R), axis=0)
    print(f'1. rows {rows}: outputs.csv against Radau (rtol 1e-13, atol 1e-18): largest relative difference '
          f'C52 {d[0]:.1e}, J {d[1]:.1e}  (DATA.md: of order 1e-14, 6.0e-13)')

    P = X.copy()
    P[:, 8:10] *= WIDE / zm.UPPER[8:10]
    C52 = np.array([zm.qoi(p, y)[0] for p in P])
    med = np.median(C52)
    print(f'2. E_vi, I_vi on [0, {WIDE:g}], {len(P)} points: median C52 {med:.2e} people, '
          f'{med / y.sum():.1f} times the data total {y.sum():.0f}  (DATA.md: 2.2e6, eight times)')
    print(f'\n{time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
