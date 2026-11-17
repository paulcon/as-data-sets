"""Check the post-processing of the OpenFAST runs of Wind_Turbine_Inflow_Loads (rainflow counting and per-set aggregation).

    python check_accuracy.py          (about a second)

1. openfast_driver.rainflow_dsum, which computed the damage sums sum(n S^m) in runs.csv, on two series
   whose rainflow counts follow by hand from three-point counting with the residue as half cycles:
   a. the turning points 0, 5, 1, 4, 2, 6, 0: one cycle each of range 2 (4-2), 4 (5-1) and 6 (0-6-0);
   b. A sin(2 pi t) at 20 points per period over N periods, starting and ending at 0: half cycles of
      range A at both ends and 2N - 1 half cycles of range 2A, so sum(n S^m) = (N - 1/2)(2A)^m + A^m.
2. sets.csv against openfast_driver.aggregate applied to runs.csv: the ultimate loads (mean over the ok
   seeds of max |x|, the paper's Eq. 5), the DELs and the seed means.

The time series of the runs were not kept, so the damage sums themselves cannot be recomputed here.
Imports numpy and the standard library.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import openfast_driver as drv  # noqa: E402


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    seq = np.array([0.0, 5.0, 1.0, 4.0, 2.0, 6.0, 0.0])
    for m in (0, 1, 3, 10):
        got, want = drv.rainflow_dsum(seq, m), 2.0 ** m + 4.0 ** m + 6.0 ** m
        print(f'check 1a, m = {m:2d}: rainflow_dsum {got:.17g}, by hand {want:.17g}, relative difference '
              f'{abs(got - want) / want:.1e}')
    A, N = 3.0, 50
    x = A * np.sin(2 * np.pi * np.arange(20 * N + 1) / 20)
    for m in (4, 10):
        got, want = drv.rainflow_dsum(x, m), (N - 0.5) * (2 * A) ** m + A ** m
        print(f'check 1b, m = {m:2d}: rainflow_dsum {got:.6e}, by hand {want:.6e}, relative difference '
              f'{abs(got - want) / want:.1e}')
    runs = list(csv.DictReader(open(os.path.join(HERE, 'runs.csv'), newline='')))
    rel = drv.compare_sets(drv.aggregate(runs, drv.read_design(HERE)), os.path.join(HERE, 'sets.csv'))
    print(f'check 2: sets.csv against the aggregation of runs.csv ({len(runs)} runs): largest relative difference '
          f'{rel:.1e}')
    if not np.isfinite(rel):
        sys.exit(1)


if __name__ == '__main__':
    main()
