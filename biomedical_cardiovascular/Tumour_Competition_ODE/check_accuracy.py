"""Recompute the numbers of DATA.md (Tumour_Competition_ODE) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py               all checks (1-2 min; see DATA.md)
    python check_accuracy.py --skip-slow   without check 3

Run it from any directory; it reads lv_samples_*.npz next to this file and evaluates tumour_model.py.

Checks:
 1. Values: the stored q_pri, q_sec of 20 points of each design against (a) tumour_model at
    rtol = atol = 1e-12 and (b) an independent solve of Eqs. 3.1-3.2 in the original states S, R
    (Radau, rtol = 1e-12, atol = 1e-15, analytic Jacobian); largest relative difference of q_pri,
    largest absolute difference of q_sec.
    q_sec gradients are compared only where q_sec (1 - q_sec) > 1e-6: closer to 0 or 1 its
    gradient is below about 1e-6 and finite differences of q_sec are round-off.
 2. Gradients: the stored (float32) gradients of the same points against forward sensitivities
    solved at rtol = atol = 1e-12 (largest relative difference of the row 2-norm).
 3. Gradients against central differences (step 1e-6, solves at rtol = atol = 1e-12) at the first
    100 grid4 points: largest and median relative error (slow).
 4. The paper's gradient recipe (p. 2188 and its sample code): forward differences with step 1e-5
    on solves at rtol = 1e-6, atol = 1e-8 (scipy RK45 with max_step = 1 day, standing in for
    MATLAB's ode45 with the same options) against the exact gradients, at the first 100 grid4
    points: median and largest relative error of q_pri's gradient.
 5. The trapezoidal rule of Eq. (3.3) against the exact integral of S + R over [0, 56]
    (dense output of a tight solve, 5600 subintervals of Simpson's rule) at the first 200 grid4
    points: median and largest relative difference.
 6. The singular limit K -> 0: q_pri and q_sec as K_S -> 0 (other inputs 0.5) and the size of
    the gradient there.

Deterministic. Imports numpy, scipy (also through tumour_model) and the standard library.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp, simpson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tumour_model as tm  # noqa: E402

TIGHT = {'rtol': 1e-12, 'atol': 1e-12}


def rhs_sr(t, y, p):
    rS, rR, KS, KR, gS, gR = p
    S, R = y
    return [rS * S * (1 - (S + gR * R) / KS), rR * R * (1 - (R + gS * S) / KR)]


def jac_sr(t, y, p):
    rS, rR, KS, KR, gS, gR = p
    S, R = y
    return [[rS * (1 - (2 * S + gR * R) / KS), -rS * S * gR / KS],
            [-rR * R * gS / KR, rR * (1 - (2 * R + gS * S) / KR)]]


def q_direct(x):
    sol = solve_ivp(rhs_sr, (0, tm.T_END), [tm.S0, tm.R0], method='Radau', t_eval=tm.T_GRID, args=(x,),
                    jac=jac_sr, rtol=1e-12, atol=1e-15)
    S, R = sol.y
    return np.array([np.trapezoid(S + R, tm.T_GRID), S[-1] / (S[-1] + R[-1])])


def q_paper_fd(x, h=1e-5):
    def q(p):
        sol = solve_ivp(rhs_sr, (0, tm.T_END), [tm.S0, tm.R0], method='RK45', t_eval=tm.T_GRID, args=(p,),
                        rtol=1e-6, atol=1e-8, max_step=1.0)
        return np.trapezoid(sol.y[0] + sol.y[1], tm.T_GRID)
    q0 = q(x)
    g = np.empty(6)
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        g[j] = (q(x + e) - q0) / h
    return g


def fd_central(x, h=1e-6):
    g = np.empty((2, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        g[:, j] = (tm.qoi(x + e, **TIGHT) - tm.qoi(x - e, **TIGHT)) / (2 * h)
    return g


def exact_integral(x):
    z0 = [np.log(tm.S0), np.log(tm.R0)]
    sol = solve_ivp(tm._rhs, (0, tm.T_END), z0, method='LSODA', args=(x,), dense_output=True, **TIGHT)
    t = np.linspace(0, tm.T_END, 5601)
    return simpson(np.exp(sol.sol(t)).sum(axis=0), x=t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-slow', action='store_true', help='skip check 3')
    args = ap.parse_args()
    t0 = time.perf_counter()
    D = tm.load_samples()   # gradients stored as float32; promoted to float64 below
    rows = np.linspace(0, len(D['grid4_X']) - 1, 20).astype(int)

    # 1 and 2
    for key in ('grid4', 'sub8'):
        X, f, G = D[f'{key}_X'][rows], D[f'{key}_f'][rows], D[f'{key}_grad'][rows].astype(np.float64)
        qt, gt = tm.qois(X, grad=True, **TIGHT)
        qd = np.array([q_direct(x) for x in X])
        print(f'1. {key}, 20 points: vs tumour_model at rtol = atol = 1e-12: q_pri rel {np.max(np.abs(qt[:, 0] - f[:, 0]) / f[:, 0]):.1e}, '
              f'q_sec abs {np.max(np.abs(qt[:, 1] - f[:, 1])):.1e}; vs Radau in (S, R): q_pri rel '
              f'{np.max(np.abs(qd[:, 0] - f[:, 0]) / f[:, 0]):.1e}, q_sec abs {np.max(np.abs(qd[:, 1] - f[:, 1])):.1e}')
        e = np.linalg.norm(gt - G, axis=2) / np.maximum(np.linalg.norm(gt, axis=2), 1e-300)
        print(f'2. {key}, 20 points: gradients vs sensitivities at rtol = atol = 1e-12: max rel (row 2-norm) '
              f'q_pri {e[:, 0].max():.1e}, q_sec {e[(np.linalg.norm(gt[:, 1], axis=1) > 1e-8), 1].max():.1e} '
              f'(q_sec rows with |grad| > 1e-8)')

    X4, G4 = D['grid4_X'], D['grid4_grad'].astype(np.float64)
    if not args.skip_slow:
        with np.errstate(divide='ignore', invalid='ignore'):   # q_sec rows with a zero gradient are not compared
            e = np.array([np.linalg.norm(fd_central(X4[i]) - G4[i], axis=1) / np.linalg.norm(G4[i], axis=1)
                          for i in range(100)])
        q4 = D['grid4_f'][:100, 1]
        ok = q4 * (1 - q4) > 1e-6
        print(f'3. central differences (h = 1e-6, tight), first 100 grid4 points: q_pri max {e[:, 0].max():.1e}, '
              f'median {np.median(e[:, 0]):.1e}; q_sec ({ok.sum()} points with q_sec (1 - q_sec) > 1e-6) max {e[ok, 1].max():.1e}, '
              f'median {np.median(e[ok, 1]):.1e}')

    e = np.array([np.linalg.norm(q_paper_fd(X4[i]) - G4[i, 0]) / np.linalg.norm(G4[i, 0]) for i in range(100)])
    print(f'4. paper recipe (forward FD h = 1e-5, RK45 rtol 1e-6 atol 1e-8 max_step 1), first 100 grid4 points: '
          f'q_pri gradient median rel error {np.median(e):.1e}, max {e.max():.1e}')

    f4 = D['grid4_f'][:200, 0]
    ex = np.array([exact_integral(x) for x in X4[:200]])
    r = np.abs(f4 - ex) / ex
    print(f'5. trapezoidal rule (1-day bins) vs exact integral, first 200 grid4 points: median rel diff {np.median(r):.1e}, '
          f'max {r.max():.1e}')

    for KS in (1e-2, 1e-4, 1e-6, 1e-8):
        x = np.array([0.5, 0.5, KS, 0.5, 0.5, 0.5])
        q, g = tm.qoi(x, grad=True)
        print(f'6. K_S = {KS:.0e} (others 0.5): q_pri = {q[0]:.6f}, q_sec = {q[1]:.3e}, |grad q_pri| = {np.linalg.norm(g[0]):.4f}, '
              f'dq_pri/dK_S = {g[0, 2]:.4f}')
    try:
        tm.qoi(np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.5]))
    except ValueError as err:
        print(f'6. K_S = 0: ValueError: {err}')
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
