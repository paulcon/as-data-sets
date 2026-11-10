"""Recompute the accuracy numbers of DATA.md that the notebook does not print (MHD_Generator).

    python check_accuracy.py          (about 1 s)

1. The u_avg row labelled 73 (run 72, the failed run). A quadratic in the five log-inputs
   (21 terms) is fitted by least squares to log u_avg on all 483 rows of
   MHD_Generator_Data_Uavg.txt. Printed: the absolute residual of label 73 divided by the median
   absolute residual, and the 95th percentile of that ratio over the 483 rows. DATA.md: 9 and 5.

The other numbers of DATA.md (the gradient check against central differences, the eigenvalues,
the label facts, the ratios (dp0/dx) df/d(dp0/dx) / f) are printed by the notebook.

Deterministic; imports numpy; reads the file next to it and writes nothing.
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def check_label_73():
    D = np.loadtxt(os.path.join(HERE, 'MHD_Generator_Data_Uavg.txt'), delimiter=',', skiprows=1)
    lab, L, y = D[:, 0].astype(int), np.log(D[:, 1:6]), np.log(D[:, 6])
    A = np.column_stack([np.ones(len(L))] + [L[:, i] for i in range(5)]
                        + [L[:, i]*L[:, j] for i in range(5) for j in range(i, 5)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = np.abs(y - A @ c)/np.median(np.abs(y - A @ c))
    i73 = np.flatnonzero(lab == 73)[0]
    print(f'1. quadratic in the log-inputs ({A.shape[1]} terms) fitted to log u_avg on all {len(y)} rows: '
          f'label 73 lies {r[i73]:.1f} times the median absolute residual off the fit; 95th percentile '
          f'of that ratio {np.percentile(r, 95):.1f} (DATA.md: 9 and 5)')


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    t0 = time.perf_counter()
    check_label_73()
    print(f'run time {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
