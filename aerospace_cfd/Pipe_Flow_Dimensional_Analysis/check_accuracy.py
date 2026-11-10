"""Recompute the numbers of DATA.md (Pipe_Flow_Dimensional_Analysis) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the shipped CSVs next to this file and evaluates
pipe_model.py. Deterministic; a few seconds. Imports numpy and the standard library.

Checks:
 1. Gradients: the shipped d(dp/dx)/d ln q of all 3000 rows against complex-step derivatives
    (step 1e-30 in ln q, Newton in complex arithmetic): largest relative error (row 2-norm).
 2. Outputs: the Colebrook residual of Eq. (60) at the friction factor recovered from the
    shipped dp/dx, lam = 2 D (dp/dx) / (rho V^2): largest absolute value over all rows; and the
    largest relative difference between the shipped dp/dx and a fresh model evaluation.
 3. Quadrature: the eigenvalues of C for pi = (dp/dx) D / (rho V^2) in ln q from the tensor
    Gauss-Legendre rule with 11 points per variable (the paper's rule) against 15 points:
    relative change of lambda_1 and lambda_2 and the largest change in z_1, z_2.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipe_model as pm  # noqa: E402


def load(regime):
    A = np.loadtxt(os.path.join(HERE, f'{regime}.csv'), delimiter=',')
    return A[:, :5], A[:, 5], A[:, 6:]


def quad_eig(regime, npts):
    Q, w = pm.gauss_legendre(regime, npts)
    _, G = pm.dimensionless_loss(Q, grad=True)
    C = (G * w[:, None]).T @ G
    lam, U = np.linalg.eigh(C)
    lam, U = lam[::-1], U[:, ::-1]
    U = U * np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(5)])   # sign rule: largest entry positive
    return lam, U


def main():
    t0 = time.perf_counter()
    print('1. complex-step check of the shipped gradients (step 1e-30 in ln q), all rows')
    for regime in pm.REGIMES:
        Q, dp, G = load(regime)
        X = np.log(Q)
        Gcs = np.empty_like(G)
        for j in range(5):
            Xc = X.astype(complex)
            Xc[:, j] += 1e-30j
            Gcs[:, j] = pm.pressure_gradient(np.exp(Xc)).imag / 1e-30
        err = np.linalg.norm(Gcs - G, axis=1) / np.linalg.norm(Gcs, axis=1)
        print(f'   {regime:9s}: max relative error {err.max():.1e} (median {np.median(err):.1e})')
    print('2. Colebrook residual at the shipped rows and model re-evaluation')
    for regime in pm.REGIMES:
        Q, dp, G = load(regime)
        lam = 2 * Q[:, 2] * dp / (Q[:, 0] * Q[:, 4] ** 2)
        res = pm.colebrook_residual(lam, pm.reynolds(Q), pm.relative_roughness(Q))
        rel = np.max(np.abs(pm.pressure_gradient(Q) - dp) / dp)
        print(f'   {regime:9s}: max |residual of Eq. (60)| {np.max(np.abs(res)):.1e}; '
              f'max relative difference model vs file {rel:.1e}')
    print('3. Gauss-Legendre 11 vs 15 points per variable, C of pi in ln q')
    for regime in pm.REGIMES:
        l11, U11 = quad_eig(regime, 11)
        l15, U15 = quad_eig(regime, 15)
        dl = np.abs(l15[:2] - l11[:2]) / l15[:2]
        dz = np.max(np.abs(U15[:, :2] - U11[:, :2]))
        print(f'   {regime:9s}: lambda_1, lambda_2 (11 points) {l11[0]:.6e}, {l11[1]:.6e}; relative change '
              f'to 15 points {dl[0]:.1e}, {dl[1]:.1e}; max change in z_1, z_2 {dz:.1e}')
    print(f'total {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
