"""Recompute the numbers of the Accuracy section of DATA.md (Karst_Aquifer_LuKARS_Calibration).

    python check_accuracy.py               all checks (several minutes; see DATA.md for the time)
    python check_accuracy.py --skip-slow   only the checks that use the shipped files (about 10 s)

Run it from any directory; it reads the shipped files next to this file and evaluates
lukars_model.py, with the finite-difference helpers of generate_data.py. Each line prints the
recomputed value; DATA.md quotes these values.

Checks:
 1. The analytic Jacobian of the transform x -> p (Eqs. 28-33) against central differences
    (h = 1e-7), on the 1000 shipped samples.
 2. Exact gradients against untruncated central differences: h = 1e-4 (gradients_fd.csv),
    h = 1e-5 and h = 1e-6 (recomputed, all 1000 samples; slow), h = 1e-7 (first 20 samples).
 3. Row 338: the jump of f_d inside its h = 1e-4 stencil in x_12, located by bisection to an
    interval of 1e-10; its share of trace(C_FD); the FD eigenvalues without it.
 4. 300 fresh samples (numpy default_rng(7)): lambda_1 from exact gradients and from untruncated
    central differences at h = 1e-3, 1e-4, 1e-5, 1e-6, and from '%f'-rounded differences at h = 1e-4
    with the 75 mm and 70 mm bounds (slow).
 5. '%f'-rounded differences with the 70 mm bound on all 1000 samples at h = 3e-5 and h = 1e-6:
    lambda_1, and the leading components of w_1..w_4 at h = 1e-6 (slow). (h = 1e-5 is the shipped
    gradients_fd_rounded_lb70.csv, analysed in the notebook.)
 6. generate_data.round_like_percent_f against numpy.round(P, 6) on the shipped inputs.

Eigenvalues are those of C = G^T G / N with G the gradients in x; eigenvector signs follow the
notebook's rule (largest component positive). Deterministic. Imports numpy and the standard library.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lukars_model as lm  # noqa: E402
import generate_data as gd  # noqa: E402


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def eig(G):
    lam, W = np.linalg.eigh(G.T @ G / len(G))
    lam, W = lam[::-1], W[:, ::-1]
    return lam, W * np.sign(W[np.argmax(np.abs(W), axis=0), range(W.shape[1])])


def relerr(A, B):
    return np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)


def exact_grads(X):
    G = np.empty_like(X)
    for s in range(0, len(X), 250):
        G[s:s + 250] = lm.misfit_and_grad(X[s:s + 250])[1]
    return G


def check_jacobian(X):
    _, J = lm.to_physical(X, jac=True)                            # (N, 21, 21), dp/dx
    h = 1e-7
    Jfd = np.stack([(lm.to_physical(X + h * e) - lm.to_physical(X - h * e)) / (2 * h) for e in np.eye(21)], axis=2)
    err = np.linalg.norm(Jfd - J, axis=(1, 2)) / np.linalg.norm(J, axis=(1, 2))
    print(f'1. transform Jacobian vs central differences (h = 1e-7), 1000 samples: max relative error '
          f'(Frobenius) {err.max():.1e}')


def check_fd(X, G, Gfd, slow):
    e = relerr(Gfd, G)
    r = int(np.argmax(e))
    print(f'2. untruncated FD h = 1e-4 (gradients_fd.csv) vs exact: median {np.median(e):.1e}, '
          f'99th percentile {np.percentile(e, 99):.1e}, max {e.max():.2g} (row {r}); samples above 0.1: '
          f'{int(np.sum(e > 0.1))}')
    if slow:
        for h in (1e-5, 1e-6):
            e = relerr(gd.fd_gradients(X, h), G)
            print(f'   untruncated FD h = {h:.0e}, 1000 samples: median {np.median(e):.1e}, '
                  f'99th percentile {np.percentile(e, 99):.1e}, max {e.max():.2g}')
    e = relerr(gd.fd_gradients(X[:20], 1e-7), G[:20])
    print(f'   untruncated FD h = 1e-7, first 20 samples: max {e.max():.1e}')


def check_row338(X, G, Gfd):
    r, j, h = 338, 11, 1e-4
    x0 = X[r]
    e = np.eye(21)[j]

    def fg(s):
        f, g = lm.misfit_and_grad(x0 + s * e)
        return f[0], g[0, j]

    a, b = -h, h
    fa, ga = fg(a)
    fb, gb = fg(b)
    defect = lambda fa, ga, fb, gb, a, b: fb - fa - 0.5 * (ga + gb) * (b - a)  # noqa: E731
    while b - a > 1e-10:
        m = 0.5 * (a + b)
        fm, gm = fg(m)
        if abs(defect(fa, ga, fm, gm, a, m)) >= abs(defect(fm, gm, fb, gb, m, b)):
            b, fb, gb = m, fm, gm
        else:
            a, fa, ga = m, fm, gm
    J = defect(fa, ga, fb, gb, a, b)
    f0 = lm.misfit(x0)[0]
    print(f'3. row {r}: jump of f_d in x_12 within the h = 1e-4 stencil: {J:.4g} ({100 * abs(J) / f0:.2f} % of '
          f'f_d = {f0:.4g}), located at x_12 {0.5 * (a + b):+.6e} from the sample to within {b - a:.1e}; smooth change over '
          f'that interval {abs(0.5 * (ga + gb) * (b - a)):.0e}; (FD - exact) * 2h = {(Gfd[r, j] - G[r, j]) * 2 * h:.4g}')
    share = np.sum(Gfd[r] ** 2) / np.sum(Gfd ** 2)
    keep = np.arange(len(G)) != r
    lam = eig(G)[0]
    lam_k = eig(Gfd[keep])[0]
    print(f'   share of trace(C_FD) from row {r}: {share:.2f}; FD eigenvalues 1-3 without it vs exact (all 1000): '
          f'max relative difference {np.max(np.abs(lam_k[:3] / lam[:3] - 1)):.4f}')


def check_fresh():
    Xf = np.random.default_rng(7).uniform(-1, 1, size=(300, 21))
    lam = eig(exact_grads(Xf))[0][0]
    out = [f'exact {lam:.4g}']
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        out.append(f'h = {h:.0e} {eig(gd.fd_gradients(Xf, h))[0][0]:.4g}')
    print('4. 300 fresh samples (default_rng(7)), lambda_1: ' + '; '.join(out))
    for lb in (None, 70.0):
        l1 = eig(gd.fd_gradients(Xf, 1e-4, rounded=True, emin3_lb=lb))[0][0]
        print(f'   %f-rounded FD, h = 1e-4, e_min,3 bound {75 if lb is None else 70} mm: lambda_1 {l1:.4g}')


def check_rounded_steps(X):
    for h in (3e-5, 1e-6):
        lam, W = eig(gd.fd_gradients(X, h, rounded=True, emin3_lb=70.0))
        line = f'5. %f-rounded FD, 70 mm bound, 1000 samples, h = {h:.0e}: lambda_1 {lam[0]:.4g}'
        if h == 1e-6:
            comp = lambda w: ', '.join(f'No. {i + 1} {w[i]:+.2f}' for i in np.argsort(-np.abs(w))[:2])  # noqa: E731
            line += '; ' + ' | '.join(f'w{k + 1}: {comp(W[:, k])}' for k in range(4))
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-slow', action='store_true', help='skip the recomputed finite-difference sets')
    args = ap.parse_args()
    t0 = time.perf_counter()
    P = load('inputs.csv')
    X = lm.to_calibration(P)
    G = load('gradients.csv')
    Gfd = load('gradients_fd.csv')
    check_jacobian(X)
    check_fd(X, G, Gfd, not args.skip_slow)
    check_row338(X, G, Gfd)
    if not args.skip_slow:
        check_fresh()
        check_rounded_steps(X)
    same = np.array_equal(gd.round_like_percent_f(P), np.round(P, 6))
    print(f"6. round_like_percent_f equals np.round(P, 6) on all shipped inputs: {same}")
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
