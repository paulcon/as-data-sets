"""Recompute the numbers of DATA.md (Steel_Column_Reliability) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the shipped files next to this file and evaluates
steel_column_model.py. Deterministic; about 10 s. Imports numpy, scipy and the standard library.

Checks:
 1. Outputs and gradients: the shipped G and dG/dx of every row of both configurations against a
    fresh evaluation, and the shipped dG/dx against complex-step derivatives in x (step 1e-30;
    no subtractive cancellation), largest relative error (row 2-norm).
 2. Input maps: largest |u(x(u)) - u| on the shipped samples, and the sample mean and sd of each
    input of 10^6 draws (seed 0) against the table of p. 326 (largest relative difference).
 3. The paper's table of p. 327: FORM reliability index (SLSQP in u, exact gradients, starts
    u = 0.1, 0.2, 0.05 in turn) at each of the ten optimal designs with the mean of E as printed
    (21000 MPa) and with 210000 MPa (sd 4200 in both), against the printed beta*, and Eb/P at the
    input means of each design with E mean 21000 and 210000. Also the Weibull shape and scale for
    (mean, sd) = (21000, 4200) and (210000, 42000): same shape, scale ratio 10, so E/s^2 has the
    same distribution with s^2 ten times larger (only E/s^2 is identifiable from beta*).
 4. FORM along the cost line b d + 5 h = C_max for h = 100, 110, 130, 200 mm (C_max = 4000 with
    b = 200; C_max = 12000 with d = 30), E mean 21000 MPa, and Eb/P at the means.
"""
import os
import sys

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steel_column_model as sm  # noqa: E402

PAPER_TABLE = [  # (C_max, b*, d*, h*, beta*), p. 327
    (4000, 200.00, 17.50, 100.00, 3.132), (5000, 200.00, 22.50, 100.00, 4.961),
    (6000, 200.00, 27.50, 100.00, 6.369), (7000, 216.67, 30.00, 100.00, 7.427),
    (8000, 250.00, 30.00, 100.00, 8.249), (9000, 283.33, 30.00, 100.00, 8.967),
    (10000, 316.67, 30.00, 100.00, 9.605), (11000, 350.00, 30.00, 100.00, 10.180),
    (12000, 383.33, 30.00, 100.00, 10.709), (13000, 400.00, 30.00, 200.00, 11.065)]


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def form_beta(config, starts=(0.1, 0.2, 0.05)):
    """Signed FORM index; nan if SLSQP converges from none of the starts."""
    def g(u):
        return sm.limit_state_u(u, config)[0]

    def jac(u):
        return sm.limit_state_u(u, config, grad=True)[1][0]
    for t in starts:
        r = optimize.minimize(lambda u: u @ u, np.full(9, t), jac=lambda u: 2 * u, method='SLSQP',
                              constraints=[{'type': 'eq', 'fun': g, 'jac': jac}],
                              options={'ftol': 1e-14, 'maxiter': 1000})
        if r.success:
            return np.sign(g(np.zeros(9))) * np.sqrt(r.fun)
    return np.nan


def main():
    print('1. outputs and gradients')
    for config in sm.CONFIGS:
        X, G, dG = load(f'{config}_inputs.csv'), load(f'{config}_outputs.csv')[:, 0], load(f'{config}_gradients.csv')
        G2, dG2 = sm.limit_state(X, grad=True)
        h = 1e-30
        cs = np.column_stack([sm.limit_state(X + 1j * h * np.eye(9)[j]).imag / h for j in range(9)])
        print(f'   {config}: fresh evaluation, max rel. diff. G {np.max(np.abs(G2 - G) / np.abs(G)):.1e}, '
              f'dG/dx {np.max(np.linalg.norm(dG2 - dG, axis=1) / np.linalg.norm(dG, axis=1)):.1e}; '
              f'complex step, max rel. error of dG/dx {np.max(np.linalg.norm(cs - dG, axis=1) / np.linalg.norm(dG, axis=1)):.1e}')
    print('2. input maps')
    for config in sm.CONFIGS:
        mg = sm.Marginals(config)
        X = load(f'{config}_inputs.csv')
        U = mg.u_from_x(X)
        rt = np.max(np.abs(mg.x_from_u(U) - X) / np.abs(X))
        _, Xs = mg.sample(10 ** 6, np.random.default_rng(0))
        dm = np.abs(Xs.mean(0) - mg.mean) / mg.sd
        ds = np.abs(Xs.std(0) - mg.sd) / mg.sd
        print(f'   {config}: max rel. |x(u(x)) - x| {rt:.1e}; 10^6 draws: max |mean - table|/sd {dm.max():.1e}, '
              f'max |sd - table|/sd {ds.max():.1e}; Weibull shape k = {mg.wb_k:.4f}, scale = {mg.wb_lam:.2f} MPa')
    print('3. FORM at the designs of the table on p. 327 (mean of E = 21000 MPa as printed, and 210000 MPa)')
    print('   C_max      b*     d*     h*  beta* (paper)  beta (FORM, E 21000)  diff    beta (FORM, E 210000)  Eb/P at means (E 21000 | 210000)')
    for cmax, b, d, h, beta_p in PAPER_TABLE:
        rows = []
        for Em in (21000.0, 210000.0):
            cfg = {'b': b, 'd': d, 'h': h, 'E_mean': Em}
            x = sm.means(cfg)[None, :]
            rows.append(float(sm.euler_load(x)[0] / x[0, 1:4].sum()))
        beta = form_beta({'b': b, 'd': d, 'h': h, 'E_mean': 21000.0})
        b210 = form_beta({'b': b, 'd': d, 'h': h, 'E_mean': 210000.0})
        s210 = f'{b210:10.4f}' if np.isfinite(b210) else '  no conv.'
        print(f'   {cmax:5d} {b:7.2f} {d:6.2f} {h:6.2f}  {beta_p:8.3f}       {beta:10.4f}      {beta - beta_p:+.4f}'
              f'      {s210}          {rows[0]:.3f} | {rows[1]:.3f}')
    k1, l1 = sm._weibull_params(21000.0, 4200.0)
    k2, l2 = sm._weibull_params(210000.0, 42000.0)
    print(f'   Weibull (21000, 4200): k = {k1:.10f}, scale = {l1:.2f}; (210000, 42000): k = {k2:.10f}, '
          f'scale = {l2:.2f}; scale ratio {l2 / l1:.6f}')
    print('4. FORM along the cost line b d + 5 h = C_max (E mean 21000 MPa)')
    print('   C_max       b       d       h   beta     Eb/P at means')
    for cmax in (4000, 12000):
        for h in (100.0, 110.0, 130.0, 200.0):
            b, d = (200.0, (cmax - 5 * h) / 200.0) if cmax == 4000 else ((cmax - 5 * h) / 30.0, 30.0)
            cfg = {'b': b, 'd': d, 'h': h, 'E_mean': 21000.0}
            x = sm.means(cfg)[None, :]
            print(f'   {cmax:5d} {b:7.2f} {d:7.3f} {h:7.1f}  {form_beta(cfg):7.4f}   '
                  f'{sm.euler_load(x)[0] / x[0, 1:4].sum():.3f}')


if __name__ == '__main__':
    main()
