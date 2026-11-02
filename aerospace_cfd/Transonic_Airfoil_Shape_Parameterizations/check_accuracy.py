"""Recompute the number of the Accuracy section of DATA.md that the notebook and convert_data.py
do not print (Transonic_Airfoil_Shape_Parameterizations).

    python check_accuracy.py          (about 3 s; numpy only)

Run it from any directory; it reads the shipped CSVs next to this file and imports
convert_data.py for the seeds, the regenerated designs and the dropped PARSEC rows.

The check: the effect of the 10 near-tie PARSEC drops on the directions. 28 PARSEC designs
have no SU2 run, and their rows (convert_data.DROPPED) were found by matching the outputs to
the regenerated designs. For 10 of them a row one or two places away fits almost as well
(the sum of squared standardized residuals of the set's own quadratic rises by less than 2,
the near-tie criterion of `convert_data.py --verify`). The script

 1. rebuilds the shipped PARSEC inputs from the seeds and DROPPED and checks that they equal
    parsec_inputs.csv bit for bit;
 2. finds the near ties from the shipped outputs (CL, CD) and the regenerated designs, and
    moves every one of them to its best alternative row (the 10 runs between a drop and its
    alternative then carry a neighbouring design);
 3. estimates the active subspaces of CL and CD (all three sample sets pooled, global
    quadratic, C = g g^T + H H / 3 in the normalized inputs, as in the notebook) with the
    shipped and with the moved rows, and prints |w1.w1'| and |w2.w2'|.

DATA.md quotes the change in |w.w| as "less than 5e-5". Deterministic (no randomness).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert_data as cd  # noqa: E402

QUOTED = 5e-5          # DATA.md: moving all 10 changes |w1.w1| and |w2.w2| by less than this
TIE = 2.0              # near-tie criterion of convert_data.verify_parsec_alignment


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def qphd_vectors(X, f):
    """Eigenvectors (descending) of C = g g^T + H H / 3 of the least-squares quadratic in X."""
    m = X.shape[1]
    i, j = np.triu_indices(m)
    c = np.linalg.lstsq(cd._quad_basis(X), f, rcond=None)[0]
    H = np.zeros((m, m))
    H[i, j] = c[m + 1:]
    H = H + H.T
    lam, W = np.linalg.eigh(np.outer(c[1:m + 1], c[1:m + 1]) + H @ H / 3)
    return W[:, ::-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.parse_args(argv)
    t0 = time.perf_counter()
    O = load('parsec_outputs.csv')
    Xship = load('parsec_inputs.csv')
    perm = np.array(cd.PARSEC_CODE_COLUMN) - 1

    rebuilt, moved, ties = [], [], []
    for N, seed in cd.SETS:
        X = cd.designs('parsec', N, seed)                       # code column order
        y = O[O[:, 0] == N][:, 3:5]
        dropped = list(cd.DROPPED[N])
        keep = np.setdiff1d(np.arange(N), dropped)
        rebuilt.append(X[keep][:, perm])
        c = np.linalg.lstsq(cd._quad_basis(X[keep]), y, rcond=None)[0]
        sd = y.std(axis=0)
        cost = lambda rows: (((y - cd._quad_basis(X[rows]) @ c) / sd) ** 2).sum()
        base = cost(keep)
        new = set(dropped)
        for d in dropped:
            best = None
            for t in range(-8, 9):
                dd = d + t
                if t == 0 or not 0 <= dd < N or dd in cd.DROPPED[N]:
                    continue
                dc = cost(np.setdiff1d(np.arange(N), sorted(set(dropped) - {d} | {dd}))) - base
                if dc < TIE and (best is None or dc < best[1]):
                    best = (dd, dc)
            if best is not None:
                ties.append((N, d, best[0], best[1]))
                new = (new - {d}) | {best[0]}
        moved.append(X[np.setdiff1d(np.arange(N), sorted(new))][:, perm])
    rebuilt, moved = np.vstack(rebuilt), np.vstack(moved)

    same = rebuilt.shape == Xship.shape and np.array_equal(rebuilt.view(np.uint64), Xship.view(np.uint64))
    print(f'1. PARSEC inputs rebuilt from the seeds and DROPPED equal parsec_inputs.csv bit for bit: {same}')
    print(f'2. near-tie drops (rise in the residual sum < {TIE:g}): {len(ties)} (DATA.md: 10)')
    for N, d, dd, dc in ties:
        print(f'     set {N:4d}: row {d} -> {dd} (+{dc:.2f})')
    print(f'   runs whose design row changes when all are moved: {int(np.sum(np.any(moved != rebuilt, axis=1)))}')
    worst = 0.0
    for k, q in ((3, 'CL'), (4, 'CD')):
        W0, W1 = qphd_vectors(rebuilt, O[:, k]), qphd_vectors(moved, O[:, k])
        dots = [abs(W0[:, j] @ W1[:, j]) for j in range(2)]
        worst = max(worst, *(1 - v for v in dots))
        print(f'3. {q}: |w1.w1\'| = {dots[0]:.7f}, |w2.w2\'| = {dots[1]:.7f}')
    print(f'   largest change 1 - |w.w\'| = {worst:.1e}  (DATA.md: less than {QUOTED:g})')
    print(f'run time {time.perf_counter() - t0:.1f} s')
    return 0 if same and worst < QUOTED else 1


if __name__ == '__main__':
    sys.exit(main())
