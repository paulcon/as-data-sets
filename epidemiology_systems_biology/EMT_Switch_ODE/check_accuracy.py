"""Recompute the numbers of DATA.md (EMT_Switch_ODE) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py               all checks (a few minutes; see DATA.md)
    python check_accuracy.py --skip-slow   without check 5

Run it from any directory; it reads the shipped files next to this file and evaluates
emt_model.py. Each line prints the recomputed value; DATA.md quotes these values.

Checks (both boxes, +/-10 % and +/-25 %):
 1. Outputs: E and S of rows 0-19 against integrations at rtol = 1e-12, atol = 1e-16 (with the
    same Newton polish), and the unpolished LSODA end state at t = 10,000 min against the files.
 2. Gradients: the shipped implicit-function gradients of rows 0-19 against the forward
    sensitivity equations integrated to t = 10,000 min (rtol 1e-11).
 3. Monostability: the number of equilibria at rows 0-499 on a 5x finer scan (20,000 points);
    and the end state of rows 0-99 started from the mesenchymal initial condition of Table S2
    (C = 0 row) instead of the C = 2 row.
 4. Vectorized bisection solver (emt_model.unique_equilibrium) against the files, all rows.
 5. The other seven treatment groups (500 uniform draws per box, seed 577): how many draws are
    bistable, on which branch the trajectory from the Table S2 initial condition ends, how
    many end with S < 0, and how many end with E below half its initial value (slow; it runs
    last, after check 6).
 6. The largest E of the +/-25 % file: its row, its Jacobian eigenvalues, and the equilibria along the
    segment from the Table 1 point to that draw (fractions 0.5 to 1.0 of the way, and 1.1 and 1.3,
    beyond the box).

Deterministic. Imports numpy, scipy (through emt_model) and the standard library.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emt_model as em  # noqa: E402

C, T = 2, em.T_ON


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-slow', action='store_true', help='skip check 5')
    args = ap.parse_args()
    t0 = time.perf_counter()
    for box in (10, 25):
        X, f = load(f'inputs_pm{box}.csv'), load(f'outputs_pm{box}.csv')
        G = load(f'gradients_pm{box}.csv').reshape(-1, 2, 11)
        rows = range(20)
        # 1
        d_tight, d_raw = 0.0, 0.0
        for i in rows:
            y = em.qoi(X[i], C, T, rtol=1e-12, atol=1e-16)
            d_tight = max(d_tight, np.max(np.abs(y - f[i, :2]) / np.abs(f[i, :2])))
            y_end = em.trajectory(X[i], C, T).y[:, -1]
            d_raw = max(d_raw, np.max(np.abs(y_end - f[i, :2]) / np.abs(f[i, :2])))
        print(f'+/-{box}%  1. outputs, rows 0-19: vs rtol 1e-12 max relative difference {d_tight:.1e}; '
              f'unpolished LSODA state at t = 10000 min vs files {d_raw:.1e}')
        # 2
        e = 0.0
        for i in rows:
            _, Gs = em.forward_sensitivities(X[i], C, T)
            e = max(e, np.max(np.linalg.norm(Gs - G[i], axis=1) / np.linalg.norm(G[i], axis=1)))
        print(f'        2. gradients, rows 0-19: implicit-function vs forward sensitivities, '
              f'max relative difference (row 2-norm) {e:.1e}')
        # 3
        n_eq = [len(em.equilibria(x, C, T, npts=20000)) for x in X[:500]]
        d_ic = 0.0
        for i in range(100):
            y = em.trajectory(X[i], C, T, y0=em.INITIAL[0]).y[:, -1]
            d_ic = max(d_ic, np.max(np.abs(y - f[i, :2]) / np.abs(f[i, :2])))
        print(f'        3. equilibria per draw, rows 0-499 (20000-point scan): {sorted(set(n_eq))}; '
              f'rows 0-99 from the mesenchymal initial condition: max relative difference {d_ic:.1e}')
        # 4
        u = em.unique_equilibrium(X, C, T)
        print(f'        4. unique_equilibrium vs files, all {len(X)} rows: max relative difference '
              f'{np.max(np.abs(u - f[:, :2]) / np.abs(f[:, :2])):.1e}')
    # 6
    X, f = load('inputs_pm25.csv'), load('outputs_pm25.csv')
    i = int(np.argmax(f[:, 0]))
    _, info = em.qoi(X[i], C, T, return_info=True)
    print(f'6. largest E in outputs_pm25.csv: row {i}, E = {f[i, 0]:.5f}, S = {f[i, 1]:.5f} ng/mL (next largest E '
          f'{np.sort(f[:, 0])[-2]:.5f}); Jacobian eigenvalues {np.array2string(info["eig"].real, precision=3)} per min; '
          'relative change from Table 1: ' + ', '.join(f'{n} {d:+.2f}' for n, d in zip(em.NAMES, X[i] / em.NOMINAL - 1)))
    for s in (0.5, 0.7, 0.8, 0.85, 0.9, 1.0, 1.1, 1.3):
        eq = em.equilibria(em.NOMINAL + s * (X[i] - em.NOMINAL), C, T, npts=20000)
        print(f'   fraction {s:.2f} of the way from Table 1: {len(eq)} equilibrium(a), E = '
              + ', '.join(f'{e:.5f}' for e in eq[:, 0]))
    # 5
    if not args.skip_slow:
        print('5. other treatment groups, 500 uniform draws per box (seed 577); counts of draws')
        for box in (10, 25):
            rng = np.random.default_rng(577)
            Xg = em.LOWER[box] + (em.UPPER[box] - em.LOWER[box]) * rng.random((500, 11))
            for TT in (0.0, em.T_ON):
                for CC in em.CONTACTS:
                    bi = epi = mes = neg = low = 0
                    for x in Xg:
                        y = em.trajectory(x, CC, TT).y[:, -1]
                        eq = em.equilibria(x, CC, TT)
                        neg += y[1] < 0
                        low += y[0] < 0.5 * em.INITIAL[CC][0]
                        if len(eq) >= 3:
                            bi += 1
                            k = np.argmin(np.abs(eq[:, 0] - y[0]))
                            epi += k == np.argmax(eq[:, 0])
                            mes += k == np.argmin(eq[:, 0])
                    print(f'   +/-{box}%, C = {CC}, T = {TT:.3g}: bistable {bi}; of these, end on the '
                          f'high-E (epithelial) branch {epi}, on the low-E (mesenchymal) branch {mes}; '
                          f'S < 0 at t = 10000 min {neg}; E(10^4) < E(0)/2 {low}')
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
