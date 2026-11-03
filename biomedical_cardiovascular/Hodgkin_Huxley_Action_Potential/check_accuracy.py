"""Recompute the numbers of DATA.md that neither the notebook nor generate_data.py prints (Hodgkin_Huxley_Action_Potential).

    python check_accuracy.py                 # all checks, 4 worker processes (10-12 minutes)
    python check_accuracy.py --workers 8
    python check_accuracy.py --skip-slow     # checks 1 and 4-6 only (about 1 minute)

Prints, beside the values DATA.md quotes:

1. Choice 3, the injected current: the nominal trajectory at I = 130, 140, ..., 170 uA/cm^2
   (DOP853, rtol 1e-10): v(5), the second maximum of v and its time, the trough after 9 ms and
   its time, and v(14). Then the lower currents of the sentence after the table: at I = 50, v(5)
   and the second maximum; at I = 0, 5, 10, 15, 20, the second maximum of v and its time, from a
   solve to 60 ms (the model's window ends at 14 ms).
2. Choice 5, the current term: on the first 300 shipped samples, the time-averaged normalized
   activity scores of v(t) (scaled to max 1), lambda1/lambda2 of vpeak, and the weight of C in
   w1 of vpeak (w1 signed so that vpeak increases along it), for the H-H form (the shipped
   gradients) and for the printed form (central differences of HH_model.qoi(form='printed'),
   step h = 1e-3 in normalized inputs).
3. The regime check: 2000 uniform random points (numpy default_rng(1)) and the 2048 corners of
   the box: the times and values of the first and second maxima of v, the number of maxima,
   how many points have a third one and when, and whether HH_model.qoi succeeds at every point.
4. Outputs against Radau (rtol 1e-13, atol 1e-14) at rows 0, 100, 200, 300, 400: the largest
   absolute differences from outputs.csv in vpeak (mV), tpeak (ms) and v(t) (mV).
5. Gradients against the same sensitivity solve at rtol 1e-13, atol 1e-14, rows 0, 100, 200, 300:
   the largest relative difference from gradients.csv (2-norm of each output's row of 11
   physical derivatives, worst output).
6. Gradients against central differences of HH_model.qoi, step h in normalized inputs:
   (a) at rtol 1e-13, atol 1e-14, h = 1e-3, rows 0, 100, 200, 300, the worst relative error over
   the 21 outputs; (b) at the shipped rtol 1e-10, h = 1e-3, rows 0, 100, ..., 900, the worst
   relative error for vpeak, tpeak and v(t) (errors of (a) and (b) in the row 2-norm of the
   physical derivatives, against gradients.csv); (c) at the nominal point, rtol 1e-10,
   h = 3e-3, 1e-3 and 3e-4, the worst relative error against HH_model.qoi_and_grad (row 2-norm of
   the derivatives in normalized inputs).

The notebook prints the other numbers of the Smoothness and Accuracy sections of DATA.md: the
ranges of vpeak and tpeak over the 1000 samples, the 81-point line along w1, and its re-solve of
3 samples. generate_data.py --check prints the comparison of a rerun with the shipped files.

Checks 2 and 3 take most of the run time (about 11 000 solves); --skip-slow leaves them out.
Deterministic (fixed rows and seed); imports numpy, scipy and HH_model; reads the CSVs next to
it and writes nothing.
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import HH_model as hm  # noqa: E402

HALF = (hm.UPPER - hm.LOWER) / 2
MID = (hm.UPPER + hm.LOWER) / 2
N_FORMS = 300                                   # samples for check 2
N_RANDOM, RANDOM_SEED = 2000, 1                 # check 3
RADAU_ROWS = [0, 100, 200, 300, 400]            # check 4
TIGHT_ROWS = [0, 100, 200, 300]                 # checks 5 and 6(a)
FD_ROWS = list(range(0, 1000, 100))             # check 6(b)
TIGHT = dict(rtol=1e-13, atol=1e-14)


def read(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def _vmin(t, y, p, I, form):
    """Event: dv/dt = 0 with dv/dt increasing (a local minimum of v)."""
    return hm._rhs(t, y, p, I, form)[0]


_vmin.direction = 1


def central_differences(p, h, **kw):
    """(21, 11) derivatives of HH_model.qoi w.r.t. the normalized inputs, step h."""
    g = np.empty((len(hm.QOI_NAMES), len(hm.NAMES)))
    for j in range(len(hm.NAMES)):
        e = np.zeros(len(hm.NAMES))
        e[j] = h * HALF[j]
        g[:, j] = (hm.qoi(p + e, **kw) - hm.qoi(p - e, **kw)) / (2 * h)
    return g


def row_error(g, ref):
    """Relative 2-norm error of each output's row of derivatives."""
    return np.linalg.norm(g - ref, axis=1) / np.linalg.norm(ref, axis=1)


def _printed(p):
    """vpeak and the (21, 11) normalized central-difference gradient for the printed form."""
    return hm.qoi(p, form='printed')[0], central_differences(p, 1e-3, form='printed')


def _maxima(p):
    """Times and values of the maxima of v, and whether HH_model.qoi succeeds."""
    sol = hm.trajectory(p, hm.TIMES)
    try:
        hm.qoi(p)
        ok = True
    except RuntimeError:
        ok = False
    return sol.t_events[0], sol.y_events[0][:, 0], ok


def radau_qoi(p, rtol, atol):
    """The 21 outputs of HH_model.qoi, solved with Radau and the analytic Jacobian."""
    p = tuple(float(x) for x in p)
    sol = solve_ivp(hm._rhs, (0., hm.T_END), hm._y0(p), method='Radau', t_eval=hm.TIMES,
                    args=(p, hm.I_INJ, hm.FORM), events=hm._vmax, rtol=rtol, atol=atol,
                    jac=lambda t, y, *a: hm._jacobians(y, p, hm.I_INJ, hm.FORM)[0])
    k = hm._peak(sol)
    return np.concatenate([[sol.y_events[0][k, 0], sol.t_events[0][k]], sol.y[0]])


def scores_and_vpeak(Gn, X, fpeak):
    """Activity scores of v(t) (scaled to max 1), lambda1/lambda2 and w1 of vpeak."""
    A = (Gn[:, 2:] ** 2).mean(axis=0)                         # (19, 11)
    S = (A / A.sum(axis=1, keepdims=True)).mean(axis=0)
    lam, W = np.linalg.eigh(Gn[:, 0].T @ Gn[:, 0] / len(Gn))
    lam, W = lam[::-1], W[:, ::-1]
    w1 = W[:, 0] * np.sign(np.cov((X - MID) / HALF @ W[:, 0], fpeak)[0, 1])
    return S / S.max(), lam[0] / lam[1], w1


def check1():
    print('1. Nominal trajectory against the injected current (DOP853, rtol 1e-10)')
    print('   I     v(5)   second peak      trough        v(14)')
    p = tuple(hm.NOMINAL)
    for I in [130., 140., 150., 160., 170.]:
        s = solve_ivp(hm._rhs, (0., hm.T_END), hm._y0(p), method='DOP853', t_eval=[5., 14.],
                      args=(p, I, hm.FORM), events=(hm._vmax, _vmin), rtol=hm.RTOL, atol=hm.ATOL)
        k = np.flatnonzero(s.t_events[1] > 9)[0]
        print(f'   {I:.0f}   {s.y[0][0]:4.1f}   {s.y_events[0][1, 0]:.1f} at {s.t_events[0][1]:.2f}   '
              f'{s.y_events[1][k, 0]:4.1f} at {s.t_events[1][k]:.2f}   {s.y[0][1]:.1f}')
    print('   (DATA.md: 130: 4.6, 50.5 at 8.79, 6.4 at 10.90, 24.0; 140: 5.6, 48.1 at 8.65, 7.9 at 10.78, 27.5;\n'
          '    150: 6.5, 46.0 at 8.52, 9.3 at 10.67, 30.2; 160: 7.4, 44.0 at 8.40, 10.8 at 10.58, 31.4;\n'
          '    170: 8.3, 42.3 at 8.29, 12.3 at 10.50, 31.3)')
    s = solve_ivp(hm._rhs, (0., hm.T_END), hm._y0(p), method='DOP853', t_eval=[5.], args=(p, 50., hm.FORM),
                  events=hm._vmax, rtol=hm.RTOL, atol=hm.ATOL)
    print(f'   I = 50: v(5) {s.y[0][0]:.1f}, second maximum {s.y_events[0][1, 0]:.1f} at {s.t_events[0][1]:.2f} '
          f'(DATA.md: -4.8; 77.7 at 11.02)')
    for I in [0., 5., 10., 15., 20.]:
        s = solve_ivp(hm._rhs, (0., 60.), hm._y0(p), method='DOP853', args=(p, I, hm.FORM),
                      events=hm._vmax, rtol=hm.RTOL, atol=hm.ATOL)
        print(f'   I = {I:2.0f}: second maximum {s.y_events[0][1, 0]:5.1f} at {s.t_events[0][1]:.2f}')
    print('   (DATA.md: after 14 ms for I <= 20, at 14.16 for I = 20; a spike of 92-96 mV for I = 10-20,\n'
          '    a few mV for I <= 5)')


def check2(pool, X, F, G):
    n = N_FORMS
    print(f'\n2. The two forms of the current term on the first {n} samples')
    S, ratio, w1 = scores_and_vpeak(G[:n] * HALF, X[:n], F[:n, 0])
    res = pool.map(_printed, X[:n], chunksize=5)
    fp = np.array([r[0] for r in res])
    Gp = np.array([r[1] for r in res])
    Sp, ratio_p, w1p = scores_and_vpeak(Gp, X[:n], fp)
    c = hm.NAMES.index('C')
    for label, s in [('H-H form', S), ('printed form', Sp)]:
        print(f'   {label}: activity scores of v(t), scaled to max 1: '
              + ', '.join(f'{hm.NAMES[j]} {s[j]:.3f}' for j in np.argsort(-s)[:6]))
    print('   (DATA.md: H-H form gK 1.000, ENa 0.971, gNa 0.315, h0 0.292, C 0.216, EK 0.200;\n'
          '    printed form C 1.000, ENa 0.704, gK 0.648, gNa 0.220, h0 0.187, EK 0.128)')
    print(f'   vpeak: lambda1/lambda2 = {ratio_p:.0f} (printed), {ratio:.0f} (H-H); weight of C in w1 '
          f'{w1p[c]:+.2f} (printed), {w1[c]:+.2f} (H-H)  (DATA.md: 64, 121; -0.31, -0.01)')


def check3(pool):
    print(f'\n3. Maxima of v at {N_RANDOM} random points (default_rng({RANDOM_SEED})) and the 2048 corners')
    rand = hm.LOWER + (hm.UPPER - hm.LOWER) * np.random.default_rng(RANDOM_SEED).random((N_RANDOM, 11))
    corners = np.array(list(itertools.product(*zip(hm.LOWER, hm.UPPER))))
    out = {}
    for label, P in [('random', rand), ('corners', corners)]:
        out[label] = pool.map(_maxima, P, chunksize=20)
    allres = out['random'] + out['corners']
    nmax = np.array([len(t) for t, v, ok in allres])
    t1 = np.array([t[0] for t, v, ok in allres])
    v1 = np.array([v[0] for t, v, ok in allres])
    t2 = np.array([t[1] for t, v, ok in allres])
    v2 = np.array([v[1] for t, v, ok in allres])
    t3 = np.array([t[2] for t, v, ok in allres if len(t) > 2])
    n3 = {k: sum(len(t) > 2 for t, v, ok in r) for k, r in out.items()}
    print(f'   first maximum at {t1.min():.2f}-{t1.max():.2f} ms ({v1.min():.0f}-{v1.max():.0f} mV), '
          f'second at {t2.min():.2f}-{t2.max():.2f} ms ({v2.min():.1f}-{v2.max():.1f} mV)')
    print(f'   number of maxima: {nmax.min()} to {nmax.max()}; a third maximum at {n3["corners"]} corners and '
          f'{n3["random"]} random points, at >= {t3.min():.2f} ms')
    print(f'   HH_model.qoi succeeds at {sum(ok for t, v, ok in allres)} of {len(allres)} points')
    print('   (DATA.md: first 0.46-0.77 ms (94-141 mV), second 7.76-9.32 ms (20.7-77.2 mV), 2 or 3 maxima,\n'
          '    third at 624 corners and 181 random points, at >= 13.29 ms; qoi succeeds at every point)')


def check4(X, F):
    d = np.array([np.abs(radau_qoi(X[r], **TIGHT) - F[r]) for r in RADAU_ROWS])
    print(f'\n4. Rows {RADAU_ROWS}, outputs.csv against Radau (rtol 1e-13, atol 1e-14): largest differences '
          f'{d[:, 0].max():.1e} mV (vpeak), {d[:, 1].max():.1e} ms (tpeak), {d[:, 2:].max():.1e} mV (v(t))'
          f'\n   (DATA.md: 2.7e-10, 2.7e-11, 5.2e-10)')


def check5(X, G):
    e = max(row_error(G[r], hm.qoi_and_grad(X[r], **TIGHT)[1]).max() for r in TIGHT_ROWS)
    print(f'\n5. Rows {TIGHT_ROWS}, gradients.csv against the sensitivity solve at rtol 1e-13: largest relative '
          f'difference {e:.1e}  (DATA.md: 5e-10)')


def check6(X, G):
    print('\n6. Gradients against central differences')
    e = max(row_error(central_differences(X[r], 1e-3, **TIGHT) / HALF, G[r]).max() for r in TIGHT_ROWS)
    print(f'   (a) rtol 1e-13, h = 1e-3, rows {TIGHT_ROWS}: worst relative error {e:.1e}  (DATA.md: 2.4e-7)')
    E = np.array([row_error(central_differences(X[r], 1e-3) / HALF, G[r]) for r in FD_ROWS])
    print(f'   (b) rtol 1e-10, h = 1e-3, rows 0, 100, ..., 900: vpeak {E[:, 0].max():.1e}, tpeak {E[:, 1].max():.1e}, '
          f'v(t) {E[:, 2:].max():.1e}  (DATA.md: 9.7e-7, 3.0e-6, 8.4e-5)')
    g = hm.qoi_and_grad(hm.NOMINAL)[1] * HALF
    errs = [row_error(central_differences(hm.NOMINAL, h), g).max() for h in [3e-3, 1e-3, 3e-4]]
    print(f'   (c) nominal point, rtol 1e-10: h = 3e-3 {errs[0]:.1e}, h = 1e-3 {errs[1]:.1e}, h = 3e-4 {errs[2]:.1e}'
          f'  (DATA.md: 2.9e-6, 5.5e-7, 8.0e-7)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--workers', type=int, default=4, help='worker processes for checks 2 and 3 (default 4)')
    ap.add_argument('--skip-slow', action='store_true', help='skip checks 2 and 3')
    a = ap.parse_args()
    t0 = time.perf_counter()
    X, F = read('inputs.csv'), read('outputs.csv')
    G = read('gradients.csv').reshape(len(X), len(hm.QOI_NAMES), len(hm.NAMES))
    check1()
    if not a.skip_slow:
        with Pool(a.workers) as pool:
            check2(pool, X, F, G)
            check3(pool)
    check4(X, F)
    check5(X, G)
    check6(X, G)
    print(f'\n{time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
