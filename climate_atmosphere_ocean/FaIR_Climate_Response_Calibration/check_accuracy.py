"""Accuracy checks of the shipped gradients of FaIR_Climate_Response_Calibration (needs FaIR 2.1.3).

    python check_accuracy.py            # step study at three posterior members (about 1 min)

Three members are checked: those with the smallest, median and largest deterministic SSP2-4.5 warming
(2081-2100 relative to 1995-2014) in the shipped qoi.csv. For each, the central-difference gradients of
the four QoIs are recomputed with steps delta * sd(u_j) for delta = 1e-1, ..., 1e-5 (sd over the 841
members, as in generate_data.py, whose step is delta = 1e-3). Gradients are compared in the whitened
coordinates g_z = L^T g_u, with L the Cholesky factor of the posterior covariance of u, by the
relative 2-norm difference. Finally, the gradients in the notebook's coordinates v (aci_beta replaced by
ERFaci 2005-2014; fair_driver.to_erfaci_coordinates, chain rule from the shipped gradients) are compared,
whitened with the posterior covariance of v, with central differences taken directly in v (step
1e-3 sd(v_j); the points are mapped back with fair_driver.from_erfaci_coordinates). Writes nothing.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fair_driver as D  # noqa: E402

DELTAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
SHIPPED_DELTA = 1e-3


def main():
    if D.FAIR is None:
        sys.exit(f"FaIR is not installed. Install it (pip install fair=={D.FAIR_VERSION}; DATA.md gives the "
                 "full environment) and rerun check_accuracy.py.")
    ids, X, seeds = D.load_posterior()
    U = D.to_analysis(X)
    sd_u = U.std(axis=0, ddof=1)
    L = np.linalg.cholesky(np.cov(U.T))
    q = np.loadtxt(os.path.join(HERE, D.F_QOI), delimiter=",")
    dT = q[:, 1]
    order = np.argsort(dT)
    rows = [order[0], order[len(order) // 2], order[-1]]
    print("rows checked:", ", ".join(f"{r} (dT {dT[r]:.3f} K)" for r in rows))
    shipped = {k: np.loadtxt(os.path.join(HERE, D.GRAD_FILES[k]), delimiter=",")[rows] for k in D.QOI_NAMES}
    ens = D.FairEnsemble()
    # the base point is reproduced
    base = ens.run(X[rows])
    for k, qn in enumerate(D.QOI_NAMES):
        rel = np.max(np.abs(base[qn] - q[rows, k + 1]) / np.abs(q[rows, k + 1]))
        print(f"{qn}: rerun vs qoi.csv, max relative difference {rel:.1e}")
    G = {d: D.fd_gradients(ens, X[rows], d, sd_u) for d in DELTAS}
    rn = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)   # noqa: E731
    for qn in D.QOI_NAMES:
        print(f"\n{qn}: relative 2-norm difference of the whitened gradient")
        print("  row   " + "  ".join(f"d{a:.0e}/d{b:.0e}" for a, b in zip(DELTAS[:-1], DELTAS[1:]))
              + "   1e-3 vs shipped")
        for i, r in enumerate(rows):
            gz = {d: L.T @ G[d][qn][i] for d in DELTAS}
            if np.linalg.norm(gz[SHIPPED_DELTA]) == 0:
                print(f"  {r:4d}  gradient is zero")
                continue
            diffs = [rn(gz[a], gz[b]) for a, b in zip(DELTAS[:-1], DELTAS[1:])]
            s = rn(L.T @ shipped[qn][i], gz[SHIPPED_DELTA])
            print(f"  {r:4d}  " + "  ".join(f"{v:13.1e}" for v in diffs) + f"   {s:13.1e}")
    # gradients in v: chain rule (as in the notebook) against central differences taken in v
    V = D.to_erfaci_coordinates(X)
    sd_v = V.std(axis=0, ddof=1)
    Lv = np.linalg.cholesky(np.cov(V.T))
    h = SHIPPED_DELTA * sd_v
    blocks = []
    for j in range(D.M_PARAMS):
        for sgn in (1.0, -1.0):
            Vj = V[rows].copy()
            Vj[:, j] += sgn * h[j]
            blocks.append(Vj)
    res = ens.run(D.from_erfaci_coordinates(np.vstack(blocks)))
    print("\ngradients in v (chain rule from the shipped files) vs central differences in v, relative "
          "2-norm difference of the whitened gradient")
    for qn in D.QOI_NAMES:
        r3 = res[qn].reshape(D.M_PARAMS, 2, len(rows))
        g_fd = ((r3[:, 0] - r3[:, 1]) / (2 * h[:, None])).T              # (3, 45)
        _, g_cr = D.to_erfaci_coordinates(X[rows], shipped[qn])
        d = [rn(Lv.T @ g_cr[i], Lv.T @ g_fd[i]) for i in range(len(rows))]
        print(f"  {qn:20s} " + "  ".join(f"row {r}: {v:.1e}" for r, v in zip(rows, d)))


if __name__ == "__main__":
    main()
