"""Recompute the numbers of the Accuracy section of DATA.md (Point_Heat_Source_Consolidation).

    python check_accuracy.py                  all checks (about 8 s); the UQTestFuns comparison
                                              needs UQTestFuns at commit 82d5c8b (see below)
    python check_accuracy.py --skip-uqtestfuns  the numpy/scipy checks only

Run it from any directory; it reads the shipped CSVs next to this file and evaluates
heat_source_model.py. Each line prints the recomputed value beside the value quoted in DATA.md.

Checks (in the order of DATA.md, "Accuracy"):

 1. Against UQTestFuns: `evaluate` of the test function SaturatedPointHeatSource (an independent
    implementation with a Taylor branch near c = kappa) on the 1000 shipped inputs; largest relative
    difference over the nonzero field values, and whether the same values underflow to 0. The test
    function is not in the UQTestFuns 0.7.0 release; it needs the unreleased GitHub source,
        pip install "git+https://github.com/damar-wicaksono/uqtestfuns@82d5c8b1e2cc9e746ba7f22a53e3c7f1ad12d42f"
    If UQTestFuns is not importable, the other checks run and the script then exits with status 2.
 2. Against the 1985 Eq. (23) in its Y, Z form (Booker & Savvidou, p. 177 and pp. 178-179), on
    the samples with |1 - c/kappa| > 0.2.
 3. Field equations: u_r against a radial integration of M div(u) = p + b' dT, and the residuals
    of the heat and pressure equations by central differences (5 samples x 3 radii x 3 times).
 4. Gradients against central differences (step h * sd_j), at (6 m, 1 y) and over the grid.
 5. Least-squares linear trend against the mean gradient (normalized inputs).
 6. Divided differences of f and g against composite Gauss-Legendre quadrature (1500 random
    cases per function, both branches of heat_source_model.divided).

Deterministic (numpy default_rng with fixed seeds). Imports numpy, scipy and the standard library
(and UQTestFuns for check 1).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heat_source_model as hm  # noqa: E402

UQTF_PIP = 'pip install "git+https://github.com/damar-wicaksono/uqtestfuns@82d5c8b1e2cc9e746ba7f22a53e3c7f1ad12d42f"'
QOIS = ('dT', 'p', 'u_r')
DAY = hm.DAY


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def check_uqtestfuns(X, fields):
    try:
        from uqtestfuns.test_functions.saturated_point_heat_source import evaluate
    except ImportError:
        print('1. UQTestFuns: NOT RUN. UQTestFuns with the test function SaturatedPointHeatSource is not '
              'importable. Install the unreleased source with\n       ' + UQTF_PIP
              + '\n   and rerun this script.')
        return False
    U = evaluate(X, hm.Q, hm.A_W, hm.MU_W, hm.R_GRID, hm.T_GRID)        # (N, 3, nr, nt)
    quoted = ('2.8e-13', '1.5e-12', '1.2e-11')
    for k, q in enumerate(QOIS):
        a, b = U[:, k], fields[:, k]
        nz = b != 0
        rel = np.max(np.abs(a[nz] - b[nz]) / np.abs(b[nz]))
        same_zero = np.array_equal(a == 0, b == 0)
        print(f'1. UQTestFuns, {q}: max relative difference {rel:.1e} (DATA.md {quoted[k]}); '
              f'same values underflow to 0: {same_zero}')
    return True


def check_eq23(X, fields):
    """1985 Eq. (23): u_R = a_u Q/(4 pi K) [Y g(kappa) - Z g(c)], p = X/(1-c/kappa) Q/(4 pi K R)[f(kappa) - f(c)],
    with Y = [X/((1-c/kappa) a_u) + b'/a_u]/(lam + 2G) and Z = X/((1-c/kappa) a_u)/(lam + 2G) (p. 177)."""
    d = hm.derived(X)
    keep = np.abs(1 - d['c'] / d['kappa']) > 0.2
    Xs = X[keep]
    d = hm.derived(Xs)
    kap, c, q, M, bp, Xc, au = (d[k][:, None, None] for k in ('kappa', 'c', 'q', 'M', 'bp', 'Xc', 'a_u'))
    R = hm.R_GRID[None, :, None]
    T = hm.T_GRID[None, None, :]
    f = lambda A: hm._phi('f', A, R, T)  # noqa: E731
    g = lambda A: hm._phi('g', A, R, T)  # noqa: E731
    one = 1 - c / kap
    Y = (Xc / (one * au) + bp / au) / M
    Z = (Xc / (one * au)) / M
    paper = (q / R * f(kap), Xc / one * q / R * (f(kap) - f(c)), au * q * (Y * g(kap) - Z * g(c)))
    quoted = ('4.4e-16', '1.1e-14', '2.2e-13')
    for k, qn in enumerate(QOIS):
        a, b = paper[k], fields[keep][:, k]
        nz = b != 0
        rel = np.max(np.abs(a[nz] - b[nz]) / np.abs(b[nz]))
        print(f'2. Eq. (23) Y, Z form, {qn} ({keep.sum()} samples with |1 - c/kappa| > 0.2, DATA.md 842): '
              f'max relative difference {rel:.1e} (DATA.md {quoted[k]})')


def check_field_equations(X):
    rng = np.random.default_rng(31)
    rows = np.sort(rng.choice(len(X), 5, replace=False))
    radii = (2.0, 6.0, 25.0)
    times = (7 * DAY, 365 * DAY, 5 * 365 * DAY)
    err = []
    for i in rows:
        for r in radii:
            for t in times:
                u = hm.fields(X[i], [r], [t])[2][0, 0, 0]
                err.append(abs(hm.u_r_by_quadrature(X[i], r, t) / u - 1))
    print(f'3. u_r vs radial quadrature (400 nodes), 45 cases (samples {rows.tolist()}): '
          f'max relative difference {max(err):.1e} (DATA.md 2.7e-14)')
    cases = [(i, r, t) for i in rows for r in radii for t in times]
    for h in (1e-2, 1e-3, 1e-4):
        res = np.array([hm.pde_residuals(X[i], r, t, h=h) for i, r, t in cases])
        worst = [cases[j] for j in res.argmax(axis=0)]
        print(f'   PDE residuals, step h = {h:.0e} (relative): max heat {res[:, 0].max():.1e}, pressure '
              f'{res[:, 1].max():.1e} (both at r = {worst[0][1]:g} m, t = {worst[0][2] / DAY:g} d); '
              f'median heat {np.median(res[:, 0]):.1e}, pressure {np.median(res[:, 1]):.1e}')


def check_gradients(X, G, sd):
    quoted = {1e-4: ('6.4e-10', '3.0e-7', '5.0e-8'), 1e-5: ('1.0e-9', '3.1e-9', '6.9e-10'),
              1e-6: ('8.6e-9', '3.4e-9', '5.6e-9')}
    for h in (1e-4, 1e-5, 1e-6):
        step = h * sd
        FD = np.stack([(hm.headline(X + step[j] * e)[0] - hm.headline(X - step[j] * e)[0]) / (2 * step[j])
                       for j, e in enumerate(np.eye(8))], axis=2)
        err = np.linalg.norm((FD - G) * sd, axis=2) / np.linalg.norm(G * sd, axis=2)
        print(f'4. gradients vs central differences at (6 m, 1 y), step {h:.0e} sd, worst sample (dT, p, u_r): '
              + ', '.join(f'{e:.1e}' for e in err.max(axis=0)) + f' (DATA.md {", ".join(quoted[h])})')
    # every analysed grid cell, h = 1e-5
    step = 1e-5 * sd
    (dT, p, u), grads = hm.fields_and_grads(X)
    Fg = np.stack([dT, p, u], axis=1)                                     # (N, 3, nr, nt)
    Gg = np.stack(grads, axis=1)                                          # (N, 3, nr, nt, 8)
    FD = np.empty_like(Gg)
    for j, e in enumerate(np.eye(8)):
        fp = np.stack(hm.fields(X + step[j] * e), axis=1)
        fm = np.stack(hm.fields(X - step[j] * e), axis=1)
        FD[..., j] = (fp - fm) / (2 * step[j])
    med = np.median(np.abs(Fg), axis=0)                                   # (3, nr, nt)
    quoted = (('1.9e-9', '1.7e-10'), ('2.3e-8', '1.1e-10'), ('1.7e-8', '1.5e-10'))
    for k, q in enumerate(QOIS):
        cells = med[k] >= 1e-6 * med[k].max()
        num = np.linalg.norm((FD[:, k] - Gg[:, k]) * sd, axis=-1)[:, cells]
        den = np.linalg.norm(Gg[:, k] * sd, axis=-1)[:, cells]
        ok = den > 0
        e = num[ok] / den[ok]
        print(f'   {q}, all samples x {cells.sum()} analysed cells, step 1e-5 sd: worst {e.max():.1e}, '
              f'median {np.median(e):.1e} (DATA.md {quoted[k][0]}, {quoted[k][1]})')


def check_linear_trend(X, F, G, mean, sd):
    Z = (X - mean) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    quoted = (('0.99991', '1.6 %'), ('0.9953', '23 %'), ('0.9772', '28 %'))  # |b - g| / |g|
    for k, q in enumerate(QOIS):
        b = np.linalg.lstsq(A, F[:, k], rcond=None)[0][1:]
        g = (G[:, k] * sd).mean(axis=0)
        cos = b @ g / (np.linalg.norm(b) * np.linalg.norm(g))
        dn = np.linalg.norm(b - g) / np.linalg.norm(g)
        print(f'5. linear trend b vs mean gradient g, {q}: cosine {cos:.5f}, |b - g| / |g| {100 * dn:.1f} % '
              f'(DATA.md {quoted[k][0]}, {quoted[k][1]})')


def _composite_gl(n_panels=400, n_nodes=30):
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    edges = np.linspace(0.0, 1.0, n_panels + 1)
    s = (edges[:-1, None] + 0.5 * (x + 1) * np.diff(edges)[:, None]).ravel()
    ws = (0.5 * w * np.diff(edges)[:, None]).ravel()
    return s, ws


def check_divided(X):
    """Reference: D = a int_0^1 h'(b + s(a-b)) ds, dD/da = S + a int s h'' ds, dD/db = a int (1-s) h'' ds."""
    rng = np.random.default_rng(1500)
    d = hm.derived(X)
    n = 1500
    i = rng.integers(0, len(X), n)
    a = d['kappa'][i]
    b = d['c'][i].copy()
    near_half = np.arange(n) % 2 == 1                     # every other case: c close to kappa
    b[near_half] = a[near_half] * (1 + rng.choice([-1, 1], near_half.sum()) * 10 ** rng.uniform(-8, -0.5, near_half.sum()))
    r = hm.R_GRID[rng.integers(0, 10, n)]
    t = hm.T_GRID[rng.integers(0, 10, n)]
    s, ws = _composite_gl()
    A = b[:, None] + s * (a - b)[:, None]
    nearflag = hm._near(a, b, r, t)
    quoted = {'f': ('3.2e-13', '3.2e-13'), 'g': ('5.2e-14', '4.1e-13')}
    for kind in ('f', 'g'):
        h1 = hm._dphi(kind, A, r[:, None], t[:, None])
        h2 = hm._d2phi(kind, A, r[:, None], t[:, None])
        S = h1 @ ws
        Dref = a * S
        Daref = S + a * (h2 @ (ws * s))
        Dbref = a * (h2 @ (ws * (1 - s)))
        D, Da, Db = hm.divided(kind, a, b, r, t, grad=True)
        rel = lambda x, y: np.abs(x - y) / np.abs(y)  # noqa: E731
        with np.errstate(divide='ignore', invalid='ignore'):
            eD, ea, eb = rel(D, Dref), rel(Da, Daref), rel(Db, Dbref)
        print(f'6. divided difference {kind} ({nearflag.sum()} quadrature-branch and {(~nearflag).sum()} '
              f'direct-quotient cases): value {np.nanmax(eD):.1e}, d/dkappa {np.nanmax(ea):.1e}', end='')
        if kind == 'f':
            print(f', d/dc {np.nanmax(eb):.1e} (DATA.md all <= {quoted["f"][0]})')
        else:
            big = np.abs(b * Dbref) >= 1e-3 * np.abs(a * Daref)
            absb = np.abs(b * (Db - Dbref)) / np.abs(a * Daref)
            print(f' (DATA.md <= {quoted["g"][0]}); d/dc where |c dD/dc| >= 1e-3 |kappa dD/dkappa| '
                  f'({big.sum()} cases): {np.nanmax(eb[big]):.1e} (DATA.md <= {quoted["g"][1]}); elsewhere '
                  f'({(~big).sum()} cases): max relative {np.nanmax(eb[~big]) if (~big).any() else 0:.1e}, '
                  f'max |c error| / |kappa dD/dkappa| {np.nanmax(absb[~big]) if (~big).any() else 0:.1e} '
                  '(DATA.md below 2e-15)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-uqtestfuns', action='store_true',
                    help='skip check 1 (needs UQTestFuns at commit 82d5c8b) and exit 0 if the rest ran')
    args = ap.parse_args()
    t0 = time.perf_counter()
    X = load('inputs.csv')
    F = load('outputs.csv')
    G = load('gradients.csv').reshape(-1, 3, 8)
    fields = load('fields.csv').reshape(-1, 3, 10, 10)
    mean, sd = hm.mean_and_sd()
    have_uqtf = True
    if not args.skip_uqtestfuns:
        have_uqtf = check_uqtestfuns(X, fields)
    check_eq23(X, fields)
    check_field_equations(X)
    check_gradients(X, G, sd)
    check_linear_trend(X, F, G, mean, sd)
    check_divided(X)
    print(f'done in {time.perf_counter() - t0:.1f} s')
    if not have_uqtf:
        print('Check 1 was not run: install UQTestFuns with\n    ' + UQTF_PIP + '\nand rerun this script '
              '(or pass --skip-uqtestfuns to run only checks 2-6).')
        sys.exit(2)


if __name__ == '__main__':
    main()
