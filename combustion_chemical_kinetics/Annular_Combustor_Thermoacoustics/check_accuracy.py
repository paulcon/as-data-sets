"""Recompute the numbers of DATA.md (Annular_Combustor_Thermoacoustics) that no other script prints.

    python check_accuracy.py            (about 45 s in the repo's .venv)

Run it from any directory inside a git checkout of this repository: checks 2-5 read the 2016
files Atacamac/atacamac-weak.txt and Atacamac/atacamac-strong.txt from the git history, at the
commit before 9c1b8a7 (the commit that removed them), with `git show 9c1b8a7^:<path>`. They are
read as numbers only. Without git or without that commit the script stops with status 3 after
check 1. Each line prints the recomputed value beside the value DATA.md quotes.

The 2016 files have 78 columns: Runs, n01..n19, tau01..tau19, maxFreq (the growth rate f_Im, 1/s),
dn01..dtau19 (its gradient with respect to the standardized inputs).

Checks:

 1. Choice 1 (L_i and alpha): the symmetric root of Magri et al. Table 2 case A (n = 0.5,
    tau = 0.6223 ms) with L_i = 0.04 m and alpha = 0.5, 0.862 and 1. DATA.md: 5136-5379 rad/s,
    growth rates -15 to -74 rad/s.
 2. Distribution of the 2016 inputs: mean column standard deviation as a fraction of the nominal
    n and tau (DATA.md 8.7 % and 4.3 %); the fraction of 2016 weak values above zero (10.37 %);
    and the model's risk factor P(f_Im > 0) for the weak case under the 2016 Gaussian
    distribution (nominal means, sd sqrt(3)/2 x 10 % of n and sqrt(3)/2 x 5 % of tau, the sd of
    the 2016 notebook), 1500 samples, numpy default_rng(2016) (DATA.md 10.1 %).
 3. Weak case, at the 492 inputs: correlation of the 2016 values with this model's f_Im (the
    larger root), largest and median difference, and the number of samples at which the larger
    root is the nearer one; the 2016 gradients against this model's gradient times the sample
    standard deviation of each 2016 input column (median cosine and median relative difference,
    2-norm, relative to the 2016 gradient).
 4. Strong case, at the 488 inputs: for each sample, the root whose growth rate is nearest the
    2016 value (largest and median difference); the 2016 gradient against that root's gradient
    (scaled as in check 3; median cosine); how often that root is the larger, the smaller, the
    higher-frequency one and the one nearer the symmetric root; the fraction of 2016 values above
    zero and of the larger root above zero.
 5. Strong-case w1 of the 2016 gradients: leading eigenvector of G^T G / 488 for the 38 gradient
    columns G, oriented so that its delay part is negative; mean of its gain part and of its delay
    part, and the gain share of |w1|^2 (DATA.md +0.168, -0.156, 0.54).

Printed elsewhere and left out here: the model's case-A and case-B roots (notebook and
generate_data.py), the Monte Carlo risk factors (15.33 %, 66.49 %, and 42.55 % for a random
component; notebook), the model's strong-case w1 (notebook), and the roots, gradients and file
comparison of generate_data.py (and its --check).

Deterministic (fixed seed). Imports numpy, the standard library and annular_combustor_model.py.
Writes nothing.
"""
import argparse
import io
import os
import subprocess
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import annular_combustor_model as acm  # noqa: E402

N = acm.N
COMMIT = '9c1b8a7^'                       # the parent of 9c1b8a7, which removed Atacamac/
CASE_A = (0.5, 6.223e-4)                  # Magri et al. Table 2, case A
GAUSS_M, GAUSS_SEED = 1500, 2016


def load_2016(case):
    """(X, f, G) of Atacamac/atacamac-<case>.txt at COMMIT: inputs (M, 38), maxFreq (M,), gradients (M, 38)."""
    path = f'Atacamac/atacamac-{case}.txt'
    try:
        res = subprocess.run(['git', '-C', HERE, 'show', f'{COMMIT}:{path}'], capture_output=True, text=True)
    except FileNotFoundError:
        print('git is not installed. Checks 2-5 read the 2016 files from the git history of this '
              'repository; install git and rerun this script inside a clone of it.', file=sys.stderr)
        sys.exit(3)
    if res.returncode != 0:
        print(f'Could not read {path} at commit {COMMIT}: {res.stderr.strip()}\n'
              'Run this script inside a full git clone of the repository (the history must contain '
              'commit 9c1b8a7 and its parent).', file=sys.stderr)
        sys.exit(3)
    A = np.loadtxt(io.StringIO(res.stdout), delimiter=',', skiprows=1)
    return A[:, 1:1 + 2 * N], A[:, 1 + 2 * N], A[:, 2 + 2 * N:]


def check_case_a():
    saved = acm.L_I, acm.ALPHA
    roots = []
    try:
        for alpha in (0.5, 0.862, 1.0):
            acm.L_I, acm.ALPHA = 0.04, alpha
            w = acm.symmetric_root(CASE_A)
            roots.append(w)
            print(f'1. case A, L_i = 0.04 m, alpha = {alpha}: omega = {w.real:.1f} {w.imag:+.2f}j rad/s')
    finally:
        acm.L_I, acm.ALPHA = saved
    re, im = [w.real for w in roots], [w.imag for w in roots]
    print(f'   range: {min(re):.0f}-{max(re):.0f} rad/s, growth rates {max(im):.0f} to {min(im):.0f} rad/s '
          f'(DATA.md 5136-5379 rad/s, -15 to -74 rad/s)')


def roots_and_gradients(X, case):
    n, tau = X[:, :N], X[:, N:]
    roots, _, _ = acm.split_roots(n, tau, case)
    G = np.stack([acm.root_gradient(roots[:, k], n, tau).imag / (2 * np.pi) for k in (0, 1)], axis=1)
    return roots, roots.imag / (2 * np.pi), G


def cosine_and_rel(g_model, g16):
    cos = np.sum(g_model * g16, axis=1) / (np.linalg.norm(g_model, axis=1) * np.linalg.norm(g16, axis=1))
    rel = np.linalg.norm(g_model - g16, axis=1) / np.linalg.norm(g16, axis=1)
    return cos, rel


def check_distribution(data):
    for case, (X, f16, _) in data.items():
        nb, tb = acm.CASES[case]
        print(f'2. {case}: 2016 inputs, mean column sd / nominal: n {100 * X[:, :N].std(0).mean() / nb:.2f} %, '
              f'tau {100 * X[:, N:].std(0).mean() / tb:.2f} % (DATA.md about 8.7 % and 4.3 %)')
    print(f'   2016 weak values above zero: {100 * (data["weak"][1] > 0).mean():.2f} % (DATA.md 10.37 %)')
    nb, tb = acm.CASES['weak']
    mu = np.r_[np.full(N, nb), np.full(N, tb)]
    sd = np.sqrt(3) / 2 * np.r_[np.full(N, 0.10 * nb), np.full(N, 0.05 * tb)]
    X = mu + sd * np.random.default_rng(GAUSS_SEED).standard_normal((GAUSS_M, 2 * N))
    f = np.concatenate([acm.growth_rate(X[s:s + 500], 'weak') for s in range(0, GAUSS_M, 500)])
    print(f'   model, weak case, 2016 Gaussian distribution, {GAUSS_M} samples (seed {GAUSS_SEED}): '
          f'P(f_Im > 0) = {100 * (f > 0).mean():.2f} % (DATA.md 10.1 %)')


def check_weak(X, f16, G16):
    roots, fr, G = roots_and_gradients(X, 'weak')
    d = np.abs(fr[:, 0] - f16)
    nearer = np.abs(fr - f16[:, None]).argmin(axis=1) == 0
    print(f'3. weak, {len(X)} samples: correlation with f_Im {np.corrcoef(fr[:, 0], f16)[0, 1]:.10f} '
          f'(DATA.md 0.9999999994); max difference {d.max():.2e}, median {np.median(d):.2e} 1/s '
          f'(DATA.md 1.4e-3, 9.3e-4); larger root nearer at {nearer.sum()} of {len(X)}')
    cos, rel = cosine_and_rel(G[:, 0] * X.std(0), G16)
    print(f'   gradients: median cosine {np.median(cos):.4f}, median relative difference {100 * np.median(rel):.1f} % '
          f'(DATA.md 0.9996, 2.9 %)')


def check_strong(X, f16, G16):
    roots, fr, G = roots_and_gradients(X, 'strong')
    rows = np.arange(len(X))
    k = np.abs(fr - f16[:, None]).argmin(axis=1)
    d = np.abs(fr[rows, k] - f16)
    print(f'4. strong, {len(X)} samples: nearest root within {d.max():.3f} 1/s (median {np.median(d):.3f}) '
          f'(DATA.md 0.025, 0.016)')
    cos, _ = cosine_and_rel(G[rows, k] * X.std(0), G16)
    print(f'   gradient of that root: median cosine {np.median(cos):.4f} (DATA.md 0.9996)')
    w, w_other = roots[rows, k], roots[rows, 1 - k]
    wc = acm.symmetric_root('strong')
    print(f'   that root is the larger at {(k == 0).sum()}, the smaller at {(k == 1).sum()}, the higher-frequency '
          f'one at {(w.real > w_other.real).sum()}, the one nearer the symmetric root at '
          f'{(np.abs(w - wc) < np.abs(w_other - wc)).sum()} (DATA.md 152, 336, 326, 229)')
    print(f'   above zero: 2016 values {100 * (f16 > 0).mean():.2f} %, larger root {100 * (fr[:, 0] > 0).mean():.2f} % '
          f'(DATA.md 37.91 %, 69.47 %)')


def check_w1(G16):
    _, V = np.linalg.eigh(G16.T @ G16 / len(G16))
    w1 = V[:, -1]
    if w1[N:].mean() > 0:
        w1 = -w1
    print(f'5. strong, 2016 gradients: w1 gain part {w1[:N].mean():+.3f}, delay part {w1[N:].mean():+.3f}, '
          f'gain share {np.sum(w1[:N] ** 2):.2f} (DATA.md +0.168, -0.156, 0.54)')


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    t0 = time.perf_counter()
    check_case_a()
    data = {case: load_2016(case) for case in acm.CASES}
    check_distribution(data)
    check_weak(*data['weak'])
    check_strong(*data['strong'])
    check_w1(data['strong'][2])
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
