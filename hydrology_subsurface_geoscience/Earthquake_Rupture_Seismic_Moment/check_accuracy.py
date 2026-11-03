"""Recompute the numbers of DATA.md (Earthquake_Rupture_Seismic_Moment) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py      (about a second)

Run it from any directory; it reads paper_tables.csv and design_inputs.csv next to this file. Each line
prints the recomputed value; DATA.md quotes these values.

Checks:
 1. The global quadratic (QPHD) estimate of the notebook, written out with numpy: a least-squares
    quadratic in the inputs normalized to [-1, 1]^3 and C = g g^T + H H / 3. Its eigenvalues and w1
    (these the notebook prints with active_subspaces).
 2. Rounding of the printed sizes: 200 perturbations of the 30 sizes by independent uniform errors in
    [-0.005, 0.005] m km (numpy default_rng(0)), the range of lambda_1/lambda_2 and the largest change
    of w1 (1 - |w1 . w1'|).

The fdfault reruns and the comparison of the input decks in DATA.md are an external run (fdfault is
not run here). Deterministic. Imports numpy and the standard library.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOWER = np.array([-120.0, 0.1, 0.9])
UPPER = np.array([-80.0, 0.4, 1.1])


def qphd(Z, f):
    m = Z.shape[1]
    iu = np.triu_indices(m)
    A = np.column_stack([np.ones(len(Z)), Z, Z[:, iu[0]] * Z[:, iu[1]]])
    c = np.linalg.lstsq(A, f, rcond=None)[0]
    g = c[1:m + 1]
    H = np.zeros((m, m))
    H[iu] = c[m + 1:]
    H = H + H.T                       # diagonal: 2 c_ii; off-diagonal: c_ij
    lam, W = np.linalg.eigh(np.outer(g, g) + H @ H / 3)
    return lam[::-1], W[:, ::-1]


def main():
    tab = np.loadtxt(os.path.join(HERE, 'paper_tables.csv'), delimiter=',')
    X = np.loadtxt(os.path.join(HERE, 'design_inputs.csv'), delimiter=',')[:, 3:]
    f = tab[:, 5]
    Z = 2 * (X - LOWER) / (UPPER - LOWER) - 1
    lam, W = qphd(Z, f)
    w1 = W[:, 0] * np.sign(W[1, 0])
    print(f'1. QPHD with numpy: eigenvalues {np.array2string(lam, precision=5)}; '
          f'lambda1/lambda2 = {lam[0] / lam[1]:.2f}; w1 = {np.array2string(w1, precision=4)}')
    rng = np.random.default_rng(0)
    ratios, dw = [], []
    for _ in range(200):
        lp, Wp = qphd(Z, f + rng.uniform(-0.005, 0.005, len(f)))
        ratios.append(lp[0] / lp[1])
        dw.append(1 - abs(Wp[:, 0] @ w1))
    print(f'2. sizes perturbed by U(-0.005, 0.005) m km, 200 times: lambda1/lambda2 from {min(ratios):.2f} '
          f'to {max(ratios):.2f}; largest 1 - |w1 . w1\'| = {max(dw):.1e}')


if __name__ == '__main__':
    main()
