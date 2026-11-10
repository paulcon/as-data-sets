"""Recompute the numbers of DATA.md (Cavity_Shielding_Effectiveness) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the shipped files next to this file. Each line prints the
recomputed value; DATA.md quotes these values. A few seconds.

Checks:
 1. Special functions: the power series for Si and Cin in cavity_se_model against
    scipy.special.sici on [-12, 12] and on the arguments the shipped design uses (k l +- pi).
 2. Outputs: the shipped SE values (all four outputs, all 1000 rows) against an independent
    implementation of Eqs. (5)-(6) written here with scipy.special.sici (largest absolute
    difference, dB).
 3. Gradients: the shipped complex-step gradients against Richardson-extrapolated central
    differences (steps 1e-3 and 5e-4 of each range) of the independent scipy implementation, all
    rows, and of cavity_se_model itself at rows 0-19 (largest relative error, row 2-norm).
    (scipy's sici does not carry a 1e-30 imaginary part for |z| > 1, so it cannot be complex-stepped.)
 4. Smoothness across the slot resonance k l = pi: for the longest slot of the design space
    (l_p = 2.5 in, a = 3.6 in, other inputs at their midpoints), SE and dSE/dw on a 401-point
    frequency grid over +-0.1 GHz around the resonance; the second difference in frequency at the
    resonance divided by the median second difference (about 1: no kink).
 5. Reading of the slot length: SE at the nominal point and w1 of SE_2GHz when the model's
    slot length is the Table 1 value itself (l = l_p, i.e. the Table 1 values read as the arc
    length) instead of the arc length computed from them as projected lengths.

Deterministic. Imports numpy, scipy and the standard library.
"""
import os
import sys
import time

import numpy as np
from scipy.special import sici

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cavity_se_model as cm  # noqa: E402


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def se_scipy(X, f_hz, arc=True):
    """Independent implementation of Eqs. (5)-(6) with scipy's sici (complex-safe)."""
    X = np.atleast_2d(X)
    d, lp, w, h, a, sig = (X * np.array([cm.INCH, cm.INCH, cm.MIL, cm.INCH, cm.INCH, 1e7])).T
    ell = 2 * a * np.arcsin(lp / (2 * a)) if arc else lp
    om = 2 * np.pi * f_hz
    kl = om / cm.C0 * ell
    Rs = np.sqrt(om * cm.MU0 / (2 * sig))
    aeq = 2 * w / np.pi * np.exp(-(1 + np.pi * d / (2 * w)))
    L = cm.MU0 * np.pi / (2 * np.log(ell / aeq) + 2 * np.log(2) - 14 / 3)
    Rint = (L * d / (cm.MU0 * w)) ** 2 * 2 * Rs / d

    def Si(z):
        return sici(z)[0]

    def Cin(z):      # Cin is even
        zz = np.abs(z)
        return np.euler_gamma + np.log(zz) - sici(zz)[1]

    G = (-(1 + np.cos(kl)) + (kl / (2 * np.pi) + np.pi / (2 * kl)) * (Cin(kl + np.pi) - Cin(np.pi - kl))
         + kl / 2 * (1 - (np.pi / kl) ** 2) * (Si(kl + np.pi) - Si(np.pi - kl))) / (np.pi * cm.ETA0 * ell)
    Hj2 = ell * (8 / np.pi ** 2) / (Rint * np.pi ** 2 / (om * L * ell) ** 2 + G) / (2 * (2 * np.pi * a * (a + h)) * Rs)
    return 10 * np.log10(4 * Hj2)


def qois_scipy(X, arc=True):
    return np.column_stack([se_scipy(X, fg * 1e9, arc) for fg in cm.FREQS_GHZ]
                           + [np.mean([se_scipy(X, fg * 1e9, arc) for fg in cm.BAND_GHZ], axis=0)])


def fd_grad(fun, X, frac=1e-3):
    """Richardson-extrapolated central differences (steps frac and frac/2 of each range) of fun,
    a map from (M, 6) inputs to (M, q) outputs; shape (M, q, 6)."""
    rng_ = cm.UPPER - cm.LOWER
    cols = []
    for j in range(6):
        def cd(h):
            e = np.zeros(6)
            e[j] = h
            return (fun(X + e) - fun(X - e)) / (2 * h)
        h = frac * rng_[j]
        cols.append((4 * cd(h / 2) - cd(h)) / 3)
    return np.stack(cols, axis=-1)


def main():
    t0 = time.perf_counter()
    X, F = load('inputs.csv'), load('outputs.csv')
    G = load('gradients.csv').reshape(-1, 4, 6)

    # 1
    z = np.linspace(-12, 12, 4001)
    z = z[z != 0]
    si, ci = sici(np.abs(z))
    e_si = np.max(np.abs(cm._si(z) - np.sign(z) * si))
    e_cin = np.max(np.abs(cm._cin(z) - (np.euler_gamma + np.log(np.abs(z)) - ci)))
    kl = np.concatenate([cm.parts(X, fg * 1e9)['kl'] for fg in cm.BAND_GHZ])
    zz = np.concatenate([kl + np.pi, np.pi - kl])
    si2, ci2 = sici(np.abs(zz))
    e2 = max(np.max(np.abs(cm._si(zz) - np.sign(zz) * si2)),
             np.max(np.abs(cm._cin(zz) - (np.euler_gamma + np.log(np.abs(zz)) - ci2))))
    print(f'1. series vs scipy sici on [-12, 12]: max abs error Si {e_si:.1e}, Cin {e_cin:.1e}; '
          f'on the design arguments (|z| <= {np.abs(zz).max():.2f}): {e2:.1e}')

    # 2
    Fs = qois_scipy(X)
    print(f'2. shipped outputs vs independent scipy implementation, all rows: max abs difference '
          f'{np.max(np.abs(Fs - F)):.1e} dB')

    # 3
    Gs = fd_grad(qois_scipy, X)
    rel = np.linalg.norm(Gs - G, axis=2) / np.linalg.norm(G, axis=2)
    print(f'3a. shipped gradients vs Richardson central differences of the scipy implementation '
          f'(steps 1e-3, 5e-4 of each range), all rows: max relative error (row 2-norm) {rel.max():.1e}, '
          f'median {np.median(rel):.1e}')
    rows = slice(0, 20)
    Gr = fd_grad(cm.qois, X[rows])
    rel2 = np.linalg.norm(Gr - G[rows], axis=2) / np.linalg.norm(G[rows], axis=2)
    print(f'3b. shipped gradients vs Richardson central differences (1e-3, 5e-4 of range), rows 0-19: '
          f'max relative error {rel2.max():.1e} (per output: '
          + ', '.join(f'{q} {v:.1e}' for q, v in zip(cm.QOI_NAMES, rel2.max(axis=0))) + ')')

    # 4
    x = cm.NOMINAL.copy()
    x[1], x[4] = 2.5, 3.6
    ell = cm.parts(x, 1e9)['l'][0]
    f_res = cm.C0 / (2 * ell)
    fs = np.linspace(f_res - 0.1e9, f_res + 0.1e9, 401)
    se = np.array([cm.se_db(x, f)[0] for f in fs])
    gw = np.array([cm.se_db(x, f, grad=True)[1][0, 2] for f in fs])
    i0 = np.argmin(np.abs(fs - f_res))

    def jump(y):
        d2 = np.diff(y, 2)
        return np.abs(d2[i0 - 1]) / np.median(np.abs(d2))
    print(f'4. longest slot (l = {ell / cm.INCH:.4f} in): slot resonance k l = pi at {f_res / 1e9:.4f} GHz; '
          f'over +-0.1 GHz SE goes {se.min():.3f}-{se.max():.3f} dB and dSE/dw {gw.min():.4f}-{gw.max():.4f} dB/mil; '
          f'second difference of SE at the resonance / median: {jump(se):.2f}; of dSE/dw: {jump(gw):.2f} '
          '(about 1 means no kink)')

    # 5
    s0 = cm.se_db(cm.NOMINAL, 2e9)[0]
    s1 = se_scipy(cm.NOMINAL, 2e9, arc=False)[0]
    s0b = [cm.se_db(cm.NOMINAL, fg * 1e9)[0] for fg in (1, 3)]
    s1b = [se_scipy(cm.NOMINAL, fg * 1e9, arc=False)[0] for fg in (1, 3)]
    Gc = fd_grad(lambda Y: se_scipy(Y, 2e9, arc=False)[:, None], X)[:, 0, :]
    s = (cm.UPPER - cm.LOWER) / 2

    def w1(Gp):
        Gn = Gp * s
        lam, W = np.linalg.eigh(Gn.T @ Gn / len(Gn))
        v = W[:, -1]
        return v * np.sign(v[2]), lam[-1] / lam[-2]
    wa, ra = w1(G[:, 1])
    wb, rb = w1(Gc)
    print(f'5. nominal SE at 1, 2, 3 GHz: arc length {s0b[0]:.3f}, {s0:.3f}, {s0b[1]:.3f} dB; l = l_p '
          f'{s1b[0]:.3f}, {s1:.3f}, {s1b[1]:.3f} dB. SE_2GHz w1 (arc) {np.round(wa, 4)}, lambda1/lambda2 {ra:.1f}; '
          f'w1 (l = l_p) {np.round(wb, 4)}, lambda1/lambda2 {rb:.1f}')
    print(f'total {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
