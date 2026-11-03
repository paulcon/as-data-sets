"""Recompute the numbers of DATA.md (Kaesbach_Catchment_Subsurface_Flow) that neither the notebook
nor extract_data.py prints.

    python check_accuracy.py      (a few seconds)

Run it from any directory; it reads inputs_unit.csv and outputs.csv next to this file. Each line
prints the recomputed value; DATA.md quotes these values.

Checks:
 1. The stored coordinates of the five conductivities: runs in the lowest and highest 5 % of u
    (all 10,050 runs). Fig. 3 of the paper has its tallest bar at the highest ln K for the weathered
    Gipskeuper, which the stored u has at its low end: the map is reversed.
 2. "Sign checked" inputs: the Spearman correlation, over all 10,050 runs, of the stored u of the
    three recharge rates with the gauge-C discharge and the net boundary outflow, and of the km1-w
    anisotropy ratio with flooding and the stream-A flow.
 3. The ranges of the gauge-C discharge and the net boundary outflow over the behavioral runs (the
    ends of the colour bars of the paper's Fig. 6).

Deterministic. Imports numpy, scipy and the standard library.
"""
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaesbach_inputs as ki  # noqa: E402


def main():
    U = np.loadtxt(os.path.join(HERE, 'inputs_unit.csv'), delimiter=',')
    Y = np.loadtxt(os.path.join(HERE, 'outputs.csv'), delimiter=',')
    acc = ki.behavioral(Y)
    print(f'1. conductivities, runs with stored u < 0.05 / u > 0.95 (of {len(U)}): '
          + '; '.join(f'{ki.NAMES[j]} {int((U[:, j] < 0.05).sum())} / {int((U[:, j] > 0.95).sum())}'
                      for j in (4, 6, 8, 10, 11)))
    rho = lambda j, k: stats.spearmanr(U[:, j], Y[:, k])[0]
    print('2. Spearman correlation over all runs, stored u vs output:')
    for j in (29, 30, 31):
        print(f'   {ki.NAMES[j]}: gauge C {rho(j, 1):+.3f}, net boundary outflow {rho(j, 5):+.3f}')
    print(f'   aniso_km1w: flooding {rho(5, 0):+.3f}, stream A {rho(5, 2):+.3f}')
    for k in (1, 5):
        print(f'3. {ki.OUTPUT_NAMES[k]}, behavioral runs ({int(acc.sum())}): {Y[acc, k].min():.4f} to '
              f'{Y[acc, k].max():.4f} m^3/s')


if __name__ == '__main__':
    main()
