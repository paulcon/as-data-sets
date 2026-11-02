"""Fit the constants that the paper does not state: C_Q, A_PHI, B_PHI (DATA.md, "Calibration").

    python calibrate.py            # about 1 min on one core; prints the fitted values

The exact solution of co2_benchmark_model.py is fitted, by Nelder-Mead least squares, to two
of the authors' Monte Carlo reference curves in mc_reference_mean_std.csv (mean and standard
deviation of S over 0-250 m at t = 100 d):

  * the joint case (all three inputs uncertain; columns eal, sal), and
  * the injection-rate-only case (eir, sir), with porosity held at the nominal 0.15 of
    Table 1 and the held relative permeability degree n1 fitted.

The joint case alone fixes only C_Q / phi (the front depends on C_Q Q / phi); the
injection-rate-only case with phi = 0.15 separates them.  The fit uses every 4th of the
10,000 samples and, for speed, profiles tabulated for n on a grid of step 0.025 (8001
points on [0, 0.8] per profile).  The single-input cases ekd/skd and epo/spo are then fitted
with a held injection factor g = 1 + theta1 and held n, phi = 0.15 (these held values are not
in the paper either).

Consistency check (`cq_consistency`): C_Q alone is refitted, with every other value held at a
natural nominal (injection factor g = 1 + theta1 = 1, phi = 0.15, n = 2 or 3), to each reference
case separately, using all 10,000 samples.  The joint and injection-rate-only cases give
C_Q of about 0.64, the relative-permeability-only and porosity-only cases about 0.44-0.46:
one constant does not fit all four cases unless the held values absorb the difference
(the held g = 0.6806 fitted below, which is outside the benchmark's range of 1 + theta1).

numpy and scipy only.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import co2_benchmark_model as cm  # noqa: E402
from generate_data import read_csv  # noqa: E402

D = read_csv(os.path.join(HERE, 'mc_reference_mean_std.csv'))
C = read_csv(os.path.join(HERE, 'benchmark_input_samples.csv'))
x = D[:, 0]
NGRID = np.linspace(2.0, 4.0, 81)


def _hull_table(n, npts=8001):
    S = np.linspace(0.0, cm.S_LEFT, npts)
    f = cm.frac_flow(S, n)
    hull = []
    for p in zip(S, f):          # upper concave hull of (S, f) on [0, 0.8]
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(p)
    H = np.array(hull)
    sp = np.diff(H[:, 1]) / np.diff(H[:, 0])
    v = 0.5 * (sp[:-1] + sp[1:])
    return v[::-1].copy(), H[1:-1, 0][::-1].copy(), sp[0], sp[-1]


TABLES = [_hull_table(n) for n in NGRID]


def ensemble(g, kd, phi, cq):
    """Mean and std over samples of S(x) with Q = Q0 g, n = 2 + 2 kd."""
    P = np.empty((len(g), len(x)))
    for i in range(len(g)):
        xi = x ** 2 * phi[i] / (2 * cq * cm.Q0 * g[i] * cm.T_END)
        v, Sv, vmax, vmin = TABLES[int(round(kd[i] * 80))]
        s = np.interp(xi, v, Sv)
        s[xi > vmax] = 0.0
        s[xi <= vmin] = cm.S_LEFT
        P[i] = s
    return P.mean(0), P.std(0)


def err(ms, col):
    return np.mean((ms[0] - D[:, col]) ** 2) + np.mean((ms[1] - D[:, col + 1]) ** 2)


def fit_cq(g, kd, phi, col):
    """Fit C_Q alone to one reference case (columns col, col + 1); returns (C_Q, rms)."""
    r = minimize_scalar(lambda cq: err(ensemble(g, kd, phi, cq), col), bounds=(0.2, 1.2),
                        method='bounded', options={'xatol': 1e-5})
    return r.x, np.sqrt(r.fun / 2)


def cq_consistency():
    """C_Q fitted separately to each reference case with natural held values (all samples)."""
    k = len(C)
    one = np.ones(k)
    phi = cm.A_PHI + cm.B_PHI * C[:, 2]
    cases = [
        ('joint (porosity map fixed)', 1 + C[:, 0], C[:, 1], phi, 1),
        ('injection rate only, phi 0.15, n 3', 1 + C[:, 0], np.full(k, 0.5), 0.15 * one, 3),
        ('rel. perm. degree only, g 1, phi 0.15', one, C[:, 1], 0.15 * one, 5),
        ('porosity only, g 1, n 2', one, np.zeros(k), phi, 7),
        ('porosity only, g 1, n 3', one, np.full(k, 0.5), phi, 7),
    ]
    return [(name,) + fit_cq(g, kd, p, col) for name, g, kd, p, col in cases]


def main():
    sub = C[::4]
    k = len(sub)

    def loss(q):
        cq, a, b, kd1 = q
        phi = a + b * sub[:, 2]
        if cq <= 0 or np.any(phi <= 0) or not 0 <= kd1 <= 1:
            return 10.0
        return (err(ensemble(1 + sub[:, 0], sub[:, 1], phi, cq), 1)
                + err(ensemble(1 + sub[:, 0], np.full(k, kd1), np.full(k, 0.15), cq), 3))

    r = minimize(loss, [0.64, 0.09, 0.19, 0.27], method='Nelder-Mead',
                 options={'maxiter': 600, 'xatol': 1e-6, 'fatol': 1e-11})
    cq, a, b, kd1 = r.x
    print(f'C_Q = {cq:.5f}, A_PHI = {a:.5f}, B_PHI = {b:.5f}, held n in eir case = {2 + 2 * kd1:.3f}; '
          f'rms (4 curves) = {np.sqrt(r.fun / 4):.4f}')

    def loss23(q):
        g, kd3 = q
        if g <= 0 or not 0 <= kd3 <= 1:
            return 10.0
        return (err(ensemble(np.full(k, g), sub[:, 1], np.full(k, 0.15), cq), 5)
                + err(ensemble(np.full(k, g), np.full(k, kd3), a + b * sub[:, 2], cq), 7))

    r2 = minimize(loss23, [0.7, 0.3], method='Nelder-Mead', options={'maxiter': 300})
    print(f'held values in ekd/epo cases: g = 1 + theta1 = {r2.x[0]:.4f}, n (epo case) = '
          f'{2 + 2 * r2.x[1]:.3f}; rms (4 curves) = {np.sqrt(r2.fun / 4):.4f}')
    print('C_Q fitted alone to each case with natural held values (all samples):')
    for name, cq1, rms in cq_consistency():
        print(f'  {name:40s} C_Q = {cq1:.3f}  rms = {rms:.4f}')


if __name__ == '__main__':
    main()
