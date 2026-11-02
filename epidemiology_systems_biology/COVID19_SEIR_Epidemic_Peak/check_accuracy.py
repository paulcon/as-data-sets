"""Recompute the numbers of DATA.md (COVID19_SEIR_Epidemic_Peak) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py               all checks (about 45 s; see DATA.md)
    python check_accuracy.py --skip-slow   without checks 3 and 7

Run it from any directory; it reads the shipped files next to this file and evaluates
seir_model.py. Each line prints the recomputed value; DATA.md quotes these values.

Checks:
 1. Outputs: the shipped t_peak and I_peak of rows 0-19 against state-only LSODA solves at
    rtol = 1e-13, atol = 1e-11 (largest relative difference).
 2. Gradients: the shipped gradients of rows 0-19 against forward sensitivities solved at
    rtol = 1e-13, atol = 1e-11 (largest relative difference of the row 2-norm), and, at the input
    means, the tight sensitivities against central differences in z = ln x (step 1e-5, tight solves).
 3. Gradients against central differences in z = ln x, step 1e-4, solves at rtol = 1e-13, first
    100 samples: largest and median relative error (row 2-norm) (slow).
 4. Readings of the input table (p. 10): mean, sd and first-order Sobol indices of t_peak and
    I_peak from the paper's 27-run Gauss-Hermite design under five readings, and the ranges of
    t_peak and I_peak over the 27 nodes (compare with the axes of the paper's Fig. 3A).
 5. Initial state: t_peak at the input means and the 27-run means with (S, E, I)(0) =
    (N - 1, 0, 1) (the data), (N - 1, 1, 0) and (N, 0, 1).
 6. The peak near R0 = 1 (other inputs at their means): R0 = 1.05 and R0 = 1.02.
 7. Ranges of t_peak and I_peak over the paper's 1000-run design (10 nodes per input; compare
    with the axes of Fig. 3B) (slow).

Deterministic. Imports numpy, scipy (through seir_model) and the standard library.
"""
import argparse
import os
import sys
import time
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seir_model as sm  # noqa: E402

TIGHT = {'rtol': 1e-13, 'atol': 1e-11}


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def fd_z(x, h, **kw):
    """Central differences of (t_peak, I_peak) in z = ln x, shape (2, 3)."""
    z = np.log(x)
    cols = []
    for j in range(3):
        dz = np.zeros(3)
        dz[j] = h
        cols.append((np.array(sm.peak(np.exp(z + dz), **kw)) - np.array(sm.peak(np.exp(z - dz), **kw))) / (2 * h))
    return np.column_stack(cols)


def rel_rows(A, B):
    """Relative error of the rows of A against B (2-norm), shape (..., 2)."""
    return np.linalg.norm(A - B, axis=-1) / np.linalg.norm(B, axis=-1)


def gh_nodes(npts, mu, sigma):
    xi, _ = np.polynomial.hermite_e.hermegauss(npts)
    return np.array([np.exp(mu + sigma * xi[list(k)]) for k in product(range(npts), repeat=3)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-slow', action='store_true', help='skip checks 3 and 7')
    args = ap.parse_args()
    t0 = time.perf_counter()
    X, f = load('inputs.csv'), load('outputs.csv')
    G = load('gradients.csv').reshape(-1, 2, 3)

    # 1. outputs vs tight state-only solves
    rows = np.arange(20)
    ft = sm.peaks(X[rows], **TIGHT)
    d = np.abs(ft - f[rows]) / np.abs(f[rows])
    print(f'1. outputs, rows 0-19, vs LSODA rtol 1e-13 atol 1e-11: max relative difference '
          f't_peak {d[:, 0].max():.1e}, I_peak {d[:, 1].max():.1e}')

    # 2. gradients vs tight sensitivities; tight sensitivities vs differences at the means
    ft2, Gt = sm.peaks(X[rows], grad=True, **TIGHT)
    e = rel_rows(G[rows], Gt).max(axis=0)
    print(f'2. gradients, rows 0-19, vs sensitivities at rtol 1e-13: max relative difference (row 2-norm) '
          f't_peak {e[0]:.1e}, I_peak {e[1]:.1e}')
    _, _, Jm = sm.peak(sm.MEAN, grad=True, **TIGHT)
    em = rel_rows(fd_z(sm.MEAN, 1e-5, **TIGHT), Jm * sm.MEAN)
    print(f'   at the input means, tight sensitivities vs central differences in ln x (h = 1e-5, tight solves): '
          f't_peak {em[0]:.0e}, I_peak {em[1]:.0e}')

    # 3. gradients vs central differences h = 1e-4 at rtol 1e-13, first 100 samples
    if not args.skip_slow:
        err = np.array([rel_rows(fd_z(X[i], 1e-4, **TIGHT), G[i] * X[i]) for i in range(100)])
        print(f'3. gradients vs central differences in ln x (h = 1e-4, rtol 1e-13), first 100 samples: '
              f'max t_peak {err[:, 0].max():.1e}, I_peak {err[:, 1].max():.1e}; '
              f'median t_peak {np.median(err[:, 0]):.1e}, I_peak {np.median(err[:, 1]):.1e}')

    # 4. readings of the input table, paper's 27-run design
    lnm = np.log(sm.MEAN)
    readings = [('mean, variance of x (chosen)', sm.MU, sm.SIGMA),
                ('median = 1.4, 4.2, 3.3; s from the variance of x', lnm, sm.SIGMA),
                ('s = 0.025, 0.25, 0.25 for ln x; mean of x = 1.4, 4.2, 3.3', lnm - sm.SD**2 / 2, sm.SD),
                ('s = 0.025, 0.25, 0.25 for ln x; median = 1.4, 4.2, 3.3', lnm, sm.SD)]
    print('4. readings of the table on p. 10, 27-run Gauss-Hermite design '
          '(t_peak mean, sd in days; I_peak mean, sd x 10^3; first-order Sobol t_peak; I_peak):')
    for name, mu, s in readings:
        r = sm.gauss_hermite_moments(3, mu=mu, sigma=s)
        print(f'   {name}: {r["mean"][0]:.3f}, {r["std"][0]:.3f} | {r["mean"][1] / 1e3:.3f}, {r["std"][1] / 1e3:.3f} | '
              + ', '.join(f'{v:.3f}' for v in r['S1'][0]) + '; ' + ', '.join(f'{v:.3f}' for v in r['S1'][1]))
        if name.startswith('mean') or name.endswith('mean of x = 1.4, 4.2, 3.3'):
            F = sm.peaks(gh_nodes(3, mu, s))
            print(f'      over the 27 nodes: t_peak {F[:, 0].min():.1f}-{F[:, 0].max():.1f} days, '
                  f'I_peak {F[:, 1].min():.0f}-{F[:, 1].max():.0f}')

    # 5. initial state
    n = sm.N_POP
    for name, y0 in [('(N-1, 0, 1) (the data)', sm.Y0), ('(N-1, 1, 0)', np.array([n - 1, 1.0, 0.0])),
                     ('(N, 0, 1)', np.array([n, 0.0, 1.0]))]:
        tp, ip = sm.peak(sm.MEAN, y0=y0)
        r = sm.gauss_hermite_moments(3, y0=y0)
        print(f'5. (S, E, I)(0) = {name}: at the input means t_peak {tp:.3f} days, I_peak {ip:.1f}; '
              f'27-run means t_peak {r["mean"][0]:.3f} days, I_peak {r["mean"][1] / 1e3:.3f} x 10^3')

    # 6. near R0 = 1
    for r0 in (1.05, 1.02):
        x = np.array([r0, sm.MEAN[1], sm.MEAN[2]])
        try:
            tp, ip = sm.peak(x)
            print(f'6. R0 = {r0}: peak at day {tp:.0f} (I_peak {ip:.0f})')
        except RuntimeError as err:
            print(f'6. R0 = {r0}: {err}')

    # 7. ranges over the 1000-run design
    if not args.skip_slow:
        F = sm.peaks(gh_nodes(10, sm.MU, sm.SIGMA))
        print(f'7. over the 1000 nodes (10 per input): t_peak {F[:, 0].min():.1f}-{F[:, 0].max():.1f} days, '
              f'I_peak {F[:, 1].min():.0f}-{F[:, 1].max():.0f}')
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
