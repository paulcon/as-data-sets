"""Recompute the numbers of DATA.md (HIV_Three_Stage_Reservoir) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py               all checks (a few minutes; see DATA.md)
    python check_accuracy.py --skip-slow   without checks 3, 4 and 7

Run it from any directory; it reads the shipped CSVs next to this file and evaluates hiv3_model.py.

Checks:
 1. Outputs: the shipped T5, T10, F of rows 0-19 against state-only solves at rtol = atol = 1e-12,
    with LSODA and with Radau (largest relative difference).
 2. Gradients: the shipped gradients of rows 0-4 against forward sensitivities solved at
    rtol = atol = 1e-12 (largest relative difference of the row 2-norm, per output).
 3. Finite-difference step: at rows 0-2, central differences in ln p with steps 1e-4 and 1e-6
    (tight solves) against the shipped gradients.
 4. Window: for rows 0-9 and the 10 rows with the largest F, T(t) on 0-100 years; the largest T after
    the shipped F must stay below 200 (so the 40-year window finds the last crossing).
 5. Initial virus: nominal trajectories with V(0) = 0.01 copies/mm^3 (used) and V(0) = 10: time of the
    acute T minimum, time to AIDS F, time at which V first exceeds 5000 after year 1, and the cosine
    between the two nominal normalized gradient directions of each output.
 6. Length of the year: T5, T10 and F at the nominal point with 365.25 days per year.
 7. Acute dip: the number of the 1000 samples whose T falls below 200 during the first 2 years (before
    recovering; F is the last crossing, so for them F is not the first time T falls below 200), and
    the lowest such value.
"""
import argparse
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hiv3_model as hm  # noqa: E402


def load():
    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',')
    f = np.loadtxt(os.path.join(HERE, 'outputs.csv'), delimiter=',')
    G = np.loadtxt(os.path.join(HERE, 'gradients.csv'), delimiter=',').reshape(-1, 3, 15)
    return X, f, G


def qoi_method(p, method, rtol, atol, y0=None, days_per_year=hm.DAYS_PER_YEAR):
    y0 = hm.Y0 if y0 is None else y0
    tend = hm.HORIZON_YEARS * days_per_year
    kw = dict(jac=hm.jac) if method in ('LSODA', 'Radau', 'BDF') else {}
    s = solve_ivp(hm.rhs, (0, tend), y0, args=(p,), method=method, rtol=rtol, atol=atol,
                  events=hm._event, dense_output=True, **kw)
    T = s.sol(np.array([5.0, 10.0]) * days_per_year)[0]
    return np.array([T[0], T[1], s.t_events[0][-1] / days_per_year])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--skip-slow', action='store_true')
    args = ap.parse_args()
    X, f, G = load()

    # 1
    for method in ('LSODA', 'Radau'):
        r = max(np.max(np.abs(qoi_method(X[i], method, 1e-12, 1e-12) - f[i]) / np.abs(f[i])) for i in range(20))
        print(f'check 1: outputs rows 0-19 vs {method} at rtol = atol = 1e-12: max relative difference {r:.1e}')

    # 2
    e = np.zeros(3)
    for i in range(5):
        _, Gt = hm.qoi(X[i], grad=True, rtol=1e-12, atol=1e-12)
        e = np.maximum(e, np.linalg.norm(G[i] - Gt, axis=1) / np.linalg.norm(Gt, axis=1))
    print(f'check 2: gradients rows 0-4 vs sensitivities at rtol = atol = 1e-12: max relative difference '
          f'T5 {e[0]:.1e}, T10 {e[1]:.1e}, F {e[2]:.1e}')

    if not args.skip_slow:
        # 3
        for h in (1e-4, 1e-6):
            e = np.zeros(3)
            for i in range(3):
                p = X[i]
                Gfd = np.empty((3, 15))
                for j in range(15):
                    d = np.zeros(15)
                    d[j] = h
                    Gfd[:, j] = (hm.qoi(p * np.exp(d), rtol=1e-12, atol=1e-12)
                                 - hm.qoi(p * np.exp(-d), rtol=1e-12, atol=1e-12)) / (2 * h)
                Gl = G[i] * p
                e = np.maximum(e, np.linalg.norm(Gfd - Gl, axis=1) / np.linalg.norm(Gl, axis=1))
            print(f'check 3: central differences in ln p, h = {h:g}, rows 0-2: max relative difference '
                  f'T5 {e[0]:.1e}, T10 {e[1]:.1e}, F {e[2]:.1e}')
        # 4
        rows = sorted(set(range(10)) | set(np.argsort(f[:, 2])[-10:].tolist()))
        worst = 0.0
        for i in rows:
            t = np.linspace(f[i, 2] * hm.DAYS_PER_YEAR + 1.0, 100 * hm.DAYS_PER_YEAR, 50001)
            T = hm.trajectory(X[i], np.concatenate([[0.0], t]))[0, 1:]
            worst = max(worst, T.max())
        print(f'check 4: {len(rows)} rows (F up to {f[rows, 2].max():.3f} years): largest T between F + 1 day and '
              f'100 years = {worst:.2f} cells/mm^3 (< 200: {worst < 200})')

    # 5
    ref = {}
    for V0 in (0.01, 10.0):
        y0 = hm.Y0.copy()
        y0[4] = V0
        t = np.linspace(0, 10 * 365, 365001)
        s = solve_ivp(hm.rhs, (0, t[-1]), y0, args=(hm.NOMINAL,), method='LSODA', jac=hm.jac,
                      rtol=1e-12, atol=1e-12, t_eval=t)
        T, V = s.y[0], s.y[4]
        j = np.argmin(T[:365 * 100])
        k = np.argmax((V > 5000) & (t > 365))
        F = qoi_method(hm.NOMINAL, 'LSODA', 1e-12, 1e-12, y0=y0)[2]
        print(f'check 5: V(0) = {V0:g} copies/mm^3: acute T minimum {T[j]:.1f} at {t[j] / 365:.3f} years; '
              f'F = {F:.3f} years; V > 5000 first after year 1 at {t[k] / 365:.3f} years')
        # nominal normalized gradient directions, by central differences in the normalized inputs
        h = 1e-4
        half = (hm.UPPER - hm.LOWER) / 2
        g = np.empty((3, 15))
        for i in range(15):
            d = np.zeros(15)
            d[i] = h * half[i]
            g[:, i] = (qoi_method(hm.NOMINAL + d, 'LSODA', 1e-12, 1e-12, y0=y0)
                       - qoi_method(hm.NOMINAL - d, 'LSODA', 1e-12, 1e-12, y0=y0)) / (2 * h)
        ref[V0] = g / np.linalg.norm(g, axis=1, keepdims=True)
    cos = np.sum(ref[0.01] * ref[10.0], axis=1)
    print(f'check 5: cosine between the nominal normalized gradients with V(0) = 0.01 and 10: '
          f'T5 {cos[0]:.4f}, T10 {cos[1]:.4f}, F {cos[2]:.4f}')

    # 6
    a = qoi_method(hm.NOMINAL, 'LSODA', 1e-12, 1e-12)
    b = qoi_method(hm.NOMINAL, 'LSODA', 1e-12, 1e-12, days_per_year=365.25)
    print(f'check 6: nominal (T5, T10, F) with 365 days/year {np.array2string(a, precision=4)}, '
          f'with 365.25 {np.array2string(b, precision=4)}')

    if not args.skip_slow:
        # 7
        t = np.linspace(0, 2 * 365, 2921)
        mins = np.array([hm.trajectory(p, t)[0].min() for p in X])
        dip = mins < hm.T_AIDS
        print(f'check 7: samples with T < 200 in the first 2 years: {dip.sum()} of {len(X)}'
              + (f' (lowest T {mins.min():.1f}; their F from {f[dip, 2].min():.3f} to {f[dip, 2].max():.3f} years)'
                 if dip.any() else ''))


if __name__ == '__main__':
    main()
