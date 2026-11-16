"""Accuracy checks for the Hodgkin-Huxley spike-peak data set (the numbers in DATA.md, "Accuracy").

    python check_accuracy.py

1. Gradients: the shipped gradients against central differences of the model (state-only solves
   at rtol 1e-12, atol 1e-14; step 1e-5 of each input's range) at the first 10 samples.
2. Solver tolerance: the shipped outputs against a solve at rtol 1e-12, atol 1e-14 at the
   first 20 samples.
3. Extrema: the event-located peak and trough against the largest and smallest v on a 0.001 ms
   grid within 0.1 ms of them, at the first 20 samples.
4. Parameter values: nominal outputs with Table 2 of Tennøe et al. (2018) against (a) m0 and n0
   exchanged (m0 = 0.0011, n0 = 0.0003, the values of Torres Valderrama et al. 2015, Table 2)
   and (b) ENa = 115 mV (Torres Valderrama et al. 2015, Table 2).

Exits with 0 when every check ran, 1 on any failure. Only numpy, scipy and the standard
library are imported.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hh_model as hm  # noqa: E402

TIGHT = dict(rtol=1e-12, atol=1e-14)
GROUPS = (('vpeak', [0]), ('tpeak', [1]), ('vmin', [2]), ('tmin', [3]), ('v(t)', list(range(4, hm.N_QOI))))


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def check_gradients(X, G, n=10):
    span = hm.UPPER - hm.LOWER
    h = 1e-5 * span
    err = np.zeros((n, hm.N_QOI))
    for k in range(n):
        Fd = np.empty((hm.N_QOI, hm.M_INPUTS))
        for j in range(hm.M_INPUTS):
            e = np.zeros(hm.M_INPUTS)
            e[j] = h[j]
            Fd[:, j] = (hm.qoi(X[k] + e, **TIGHT) - hm.qoi(X[k] - e, **TIGHT)) / (2 * h[j])
        Gk = G[k].reshape(hm.N_QOI, hm.M_INPUTS) * span / 2      # normalized coordinates
        Fk = Fd * span / 2
        err[k] = np.linalg.norm(Gk - Fk, axis=1) / np.linalg.norm(Gk, axis=1)
    print(f'1. gradients vs central differences (first {n} samples; relative 2-norm error per output, '
          'normalized coordinates):')
    for name, idx in GROUPS:
        print(f'   {name:6s} largest {err[:, idx].max():.1e}')
    print(f'   all outputs: largest {err.max():.1e}')


def check_tolerance(X, F, n=20):
    Ft = np.array([hm.qoi(x, **TIGHT) for x in X[:n]])
    print(f'2. shipped outputs vs a solve at rtol 1e-12, atol 1e-14 (first {n} samples), largest absolute difference:')
    for name, idx in GROUPS:
        print(f'   {name:6s} {np.max(np.abs(Ft[:, idx] - F[:n, idx])):.1e} {hm.QOI_UNITS[idx[0]]}')


def check_extrema(X, F, n=20):
    dv, dt = [], []
    for k in range(n):
        for col_v, col_t, sgn in ((0, 1, 1.0), (2, 3, -1.0)):
            t = np.arange(F[k, col_t] - 0.1, F[k, col_t] + 0.1 + 5e-4, 1e-3)
            v = hm.trajectory(X[k], t, **TIGHT)
            i = np.argmax(sgn * v)
            dv.append(abs(v[i] - F[k, col_v]))
            dt.append(abs(t[i] - F[k, col_t]))
    print(f'3. event-located extrema vs a 0.001 ms grid (first {n} samples): largest |dv| {max(dv):.1e} mV '
          f'(the grid value is at most the extremum), largest |dt| {max(dt):.1e} ms (grid spacing 1e-3 ms)')


def check_parameters():
    base = hm.qoi(hm.NOMINAL)
    swap = hm.NOMINAL.copy()
    swap[hm.NAMES.index('n0')], swap[hm.NAMES.index('m0')] = 0.0003, 0.0011
    ena = hm.NOMINAL.copy()
    ena[hm.NAMES.index('ENa')] = 115.0
    print('4. nominal outputs: Table 2 of Tennøe et al. (2018) vs alternatives')
    print(f'   Table 2:            vpeak {base[0]:.4f} mV at {base[1]:.4f} ms, vmin {base[2]:.4f} mV at {base[3]:.4f} ms')
    for label, p in (('m0, n0 exchanged', swap), ('ENa = 115 mV', ena)):
        q = hm.qoi(p)
        print(f'   {label + ":":19s} vpeak {q[0]:.4f} mV at {q[1]:.4f} ms, vmin {q[2]:.4f} mV at {q[3]:.4f} ms; '
              f'largest |change| in v(t) {np.max(np.abs(q[4:] - base[4:])):.3g} mV')


def main():
    t0 = time.perf_counter()
    try:
        X, F, G = load('inputs.csv'), load('outputs.csv'), load('gradients.csv')
        check_gradients(X, G)
        check_tolerance(X, F)
        check_extrema(X, F)
        check_parameters()
    except Exception as exc:  # any failure is exit code 1
        print(f'FAIL: {exc!r}', file=sys.stderr)
        sys.exit(1)
    print(f'run time {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
