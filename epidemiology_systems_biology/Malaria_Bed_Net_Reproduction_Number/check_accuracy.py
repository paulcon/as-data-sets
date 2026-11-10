"""Recompute the numbers of DATA.md that the notebook and generate_data.py do not print.

    python check_accuracy.py

Check 1: the gradients of gradients.csv against central differences at three relative steps,
         at all 1000 samples.
Check 2: alpha, alpha_s and alpha_r do not change the spectral radius of F V^{-1} (so they are
         not inputs of R_eff): the radius over their Table 3 ranges at 200 samples.
Check 3: the printed radicand of Eq. (18) against the corrected one, at all samples.
Check 4: the other reading of xi1 (xi1 = a b1 rho, without k_v/k_h; 15 inputs): its PRCC values
         and count of R_eff < 1 on the shipped design and on 50 further designs, against Fig. 5.

Only numpy and the standard library are imported. About 30 s.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import malaria_model as mm  # noqa: E402

SEED = 2019
# Fig. 5, p. 21, read from the bar ends on a 300 dpi rendering of the page (about +-0.005)
PAPER_FIG5 = {'beta_min': 0.863, 'gamma': -0.657, 'b2': 0.631, 'b1': 0.624, 'psi_h': -0.509, 'rho': 0.489,
              'b': -0.468, 'beta_max': 0.466, 'mu_h2': 0.291, 'mu_h': 0.255, 'e': -0.236, 'mu_v2': -0.234,
              'psi_v': 0.221, 'r_r': -0.216, 'mu_v': -0.128, 'T_s': -0.062, 'beta': -0.047, 'r_s': -0.021,
              'delta_h': -0.015, 'mu_v3': 0.0}


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',')


def kv_over_kh(X):
    c = {n: X[:, i] for i, n in enumerate(mm.NAMES)}
    kv = (c['psi_v'] - c['mu_v'] - c['mu_v3'] * c['b'] * c['gamma']) / c['mu_v2']
    kh = (c['psi_h'] - c['mu_h']) / c['mu_h2']
    return kv / kh


def main():
    t0 = time.perf_counter()
    X, F, G = load('inputs.csv'), load('outputs.csv'), load('gradients.csv')
    f = F[:, 0]

    print('Check 1: gradients vs central differences, all samples (row 2-norm relative error)')
    for h in (1e-5, 1e-6, 1e-7):
        err = []
        for x, g in zip(X, G):
            gfd = np.array([(mm.reff(x + h * x[j] * e) - mm.reff(x - h * x[j] * e)) / (2 * h * x[j])
                            for j, e in enumerate(np.eye(mm.M_IN))])
            err.append(np.linalg.norm(gfd - g) / np.linalg.norm(g))
        print(f'  relative step {h:g}: max {max(err):.1e}, median {np.median(err):.1e}')

    print('Check 2: spectral radius of F V^-1 with alpha, alpha_s, alpha_r over their Table 3 ranges')
    rng = np.random.default_rng([SEED, 2])
    worst = 0.0
    for x in X[:200]:
        r0 = np.max(np.abs(np.linalg.eigvals(mm.ngm(x))))
        for _ in range(5):
            a, s, r = rng.uniform([0.195, 0.001105, 0.001105], [0.39, 0.00221, 0.00221])
            r1 = np.max(np.abs(np.linalg.eigvals(mm.ngm(x, alpha=a, alpha_s=s, alpha_r=r))))
            worst = max(worst, abs(r1 - r0) / r0)
    print(f'  200 samples x 5 draws: max relative change of the radius {worst:.1e}')

    print('Check 3: radicand of Eq. (18)')
    neg, fixed_err = 0, 0.0
    for x, R in zip(X, f):
        q = mm._parts(x)
        beta, psi_v = q['p']['beta'], q['p']['psi_v']
        p1, p2, p3, c = q['p1'], q['p2'], q['p3'], q['xi1'] * q['k1']
        printed = -p2 * p1 * psi_v * c * (beta * p1 + beta - p3 - beta * p2 - p1)
        fixed = -p2 * p1 * psi_v * c * (beta * p1 - beta * p3 - beta * p2 - p1)
        neg += printed < 0
        fixed_err = max(fixed_err, abs(fixed / (p2 * p1 * psi_v) ** 2 - R) / R)
    print(f'  printed radicand negative at {neg} of {len(X)} samples; corrected radicand: '
          f'max |rho^2 - R_eff| / R_eff = {fixed_err:.1e}')

    print('Check 4: the reading xi1 = a b1 rho (no k_v/k_h)')
    pv = np.array([PAPER_FIG5[n] for n in mm.NAMES])
    rows = []
    designs = [X] + [mm.sample(1000, np.random.default_rng([SEED, r])) for r in range(1, 51)]
    for Xd in designs:
        Rd = mm.reffs(Xd)
        Ra = Rd / kv_over_kh(Xd)
        rows.append((mm.prcc(Xd, Rd), mm.prcc(Xd, Ra), int(np.sum(Rd < 1)), int(np.sum(Ra < 1))))
    for k, label in ((0, 'xi1 = a b1 rho kv/kh (this data set)'), (1, 'xi1 = a b1 rho')):
        P0 = rows[0][k]
        top = [mm.NAMES[j] for j in np.argsort(-np.abs(P0))[:5]]
        dev = np.array([np.max(np.abs(r[k] - pv)) for r in rows[1:]])
        cnt = np.array([r[2 + k] for r in rows[1:]])
        jp = mm.NAMES.index('psi_h')
        print(f'  {label}: shipped design: top five {top}, PRCC psi_h {P0[jp]:+.3f}, '
              f'max |PRCC - Fig. 5| {np.max(np.abs(P0 - pv)):.3f}, R_eff < 1 at {rows[0][2 + k]} of 1000')
        print(f'    50 further designs: max |PRCC - Fig. 5| median {np.median(dev):.3f} (range {dev.min():.3f}-{dev.max():.3f}); '
              f'PRCC psi_h {np.min([r[k][jp] for r in rows[1:]]):+.3f} to {np.max([r[k][jp] for r in rows[1:]]):+.3f}; '
              f'R_eff < 1 count {cnt.min()}-{cnt.max()} (median {int(np.median(cnt))}); paper: 210')
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
