"""Recompute the numbers of DATA.md (Ice_Sheet_Basal_Melt_Ensemble) that neither the notebook nor
convert_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the two shipped CSVs next to this file. Each numbered line
prints the recomputed values that DATA.md quotes.

Checks:
 1. Sigmoid consistency of the stored design: M200 = Mmax * g(t0, tau) with
    g = K/(K-1) * (1/(1+exp(x)) - 1/K), x = -(200 - t0)/tau, K = 1 + exp(t0/tau) (Eqs. 1-5 of the
    paper evaluated at t = 200); largest relative error per basin.
 2. Scaled melt rates: the ratio of each scaled column (m/yr) to its unscaled fraction, per basin.
 3. The design: every input inside the Sect. 2.2 bounds; centred L2 discrepancy of each basin's
    3-D (M200, t0, tau) design on the unit cube against the mean of 20 seeded uniform random designs
    of the same size; correlations between the 15 inputs of the 500 runs (largest |r|, the pairs
    with |r| > 0.2); for comparison, the largest |r| of the first 500 points of an unscrambled
    15-D Sobol' sequence (scipy).
 4. The volumes: the initial value of all runs; the smallest nonzero year-to-year change and how
    close every change is to an integer multiple of it.
 5. The failed runs: their ensemble numbers, the first stored year written as 0, whether any
    succeeded run holds a 0; the rank of their year-100 sea-level contribution among all 500 runs
    and a two-sided Mann-Whitney test against the 491 succeeded runs; the smallest Mann-Whitney
    p-value over the 15 inputs (failed against succeeded runs, on the inputs).

Deterministic (fixed seeds); takes about a second. Imports numpy, scipy and the standard library.
"""
import os

import numpy as np
from scipy.stats import mannwhitneyu, qmc

HERE = os.path.dirname(os.path.abspath(__file__))
BASINS = ['amundsen', 'eais', 'peninsula', 'ross', 'weddell']
CAP = {'amundsen': 50.0, 'eais': 36.0, 'peninsula': 12.0, 'ross': 20.0, 'weddell': 16.0}  # m/yr, p. 2688


def read(name):
    path = os.path.join(HERE, name)
    with open(path) as fh:
        fh.readline()
        cols = [c.split(' [')[0] for c in fh.readline()[2:].strip().split(',')]
    return np.loadtxt(path, delimiter=','), cols


def main():
    D, dc = read('basal_melt_design.csv')
    V, _ = read('ice_volume_above_flotation.csv')
    col = {c: D[:, j] for j, c in enumerate(dc)}
    fail = col['flag'] == 1
    ok = ~fail

    errs = []
    for b in BASINS:
        t0, tau = col[f't0_{b}'], col[f'tau_{b}']
        K = 1 + np.exp(t0 / tau)
        x = -(200 - t0) / tau
        g = K / (K - 1) * (1 / (1 + np.exp(x)) - 1 / K)
        e1 = np.max(np.abs(col[f'Mmaxfrac_{b}'] * g - col[f'M200frac_{b}']) / col[f'M200frac_{b}'])
        e2 = np.max(np.abs(col[f'Mmax_{b}'] * g - col[f'M200_{b}']) / col[f'M200_{b}'])
        errs.append(f'{b} {max(e1, e2):.1e}')
    print('1. largest relative error of M200 = Mmax*g(t0,tau): ' + ', '.join(errs))

    rat = []
    for b in BASINS:
        r1 = col[f'M200_{b}'] / col[f'M200frac_{b}']
        r2 = col[f'Mmax_{b}'] / col[f'Mmaxfrac_{b}']
        dev = max(np.max(np.abs(r1 / CAP[b] - 1)), np.max(np.abs(r2 / CAP[b] - 1)))
        rat.append(f'{b} {np.median(r1):g} (largest relative deviation from the cap {dev:.0e})')
    print('2. scaled / unscaled melt rate: ' + '; '.join(rat))

    U = []
    inside = True
    for b in BASINS:
        m, t0, tau = col[f'M200_{b}'], col[f't0_{b}'], col[f'tau_{b}']
        inside &= bool(np.all((m >= 0) & (m <= CAP[b]) & (t0 >= 100) & (t0 <= 225) & (tau >= 10) & (tau <= 75)))
        U.append(np.c_[m / CAP[b], (t0 - 100) / 125, (tau - 10) / 65])
    rnd = np.mean([qmc.discrepancy(np.random.default_rng(s).random((500, 3))) for s in range(20)])
    cd = ', '.join(f'{b} {qmc.discrepancy(u):.1e}' for b, u in zip(BASINS, U))
    print(f'3a. all 500 runs inside the Sect. 2.2 bounds: {inside}; centred L2 discrepancy of each basin design '
          f'(500 points, unit cube): {cd}; mean of 20 uniform random designs {rnd:.1e}')
    names = [f'{p}_{b}' for b in BASINS for p in ('M200', 't0', 'tau')]
    X = np.hstack(U)
    C = np.corrcoef(X.T)
    iu = np.triu_indices(15, 1)
    big = [(abs(C[i, j]), names[i], names[j], C[i, j]) for i, j in zip(*iu) if abs(C[i, j]) > 0.2]
    big.sort(reverse=True)
    print(f'3b. input correlations (500 runs): largest |r| {np.max(np.abs(C[iu])):.2f}; '
          f'{len(big)} of {len(iu[0])} pairs with |r| > 0.2: '
          + ', '.join(f'{a}-{c} {r:+.2f}' for _, a, c, r in big)
          + f'; 1/sqrt(500) = {1 / np.sqrt(500):.3f}')
    S = qmc.Sobol(d=15, scramble=False).random(512)[:500]
    Cs = np.corrcoef(S.T)
    print(f'3c. first 500 points of an unscrambled 15-D Sobol\' sequence (scipy): largest |r| '
          f'{np.max(np.abs(Cs[iu])):.3f}')

    Vy = V[:, 1:]
    steps = np.diff(Vy[ok], axis=1).ravel()
    q = np.min(np.abs(steps[steps != 0]))
    k = steps / q
    print(f'4. initial volume of all 500 runs: {np.unique(Vy[:, 0])} mm SLE; smallest nonzero annual change '
          f'{q:.6f} mm; largest distance of a change/quantum from an integer {np.max(np.abs(k - np.round(k))):.1e} '
          f'({steps.size} changes of the 491 succeeded runs)')

    first0 = [int(np.where(Vy[i] == 0)[0][0]) + 1 for i in np.where(fail)[0]]
    slr100 = -(Vy[:, 99] - Vy[:, 0])
    pct = [(slr100 < v).mean() for v in slr100[fail]]
    p = mannwhitneyu(slr100[fail], slr100[ok]).pvalue
    pin = [mannwhitneyu(X[fail, j], X[ok, j]).pvalue for j in range(15)]
    jmin = int(np.argmin(pin))
    print(f'5. failed runs (flag = 1): ens {col["ens"][fail].astype(int).tolist()}; first stored year written as 0: '
          f'{first0}; zeros in succeeded runs: {int((Vy[ok] == 0).sum())}; year-100 contribution of the failed '
          f'runs {np.round(slr100[fail], 1).tolist()} mm, at fractions {np.round(pct, 2).tolist()} of the 500 runs '
          f'({int((slr100[fail] > np.median(slr100[ok])).sum())} of 9 above the median of the succeeded runs), '
          f'Mann-Whitney p = {p:.4f}; smallest input p-value {pin[jmin]:.3f} ({names[jmin]}), '
          f'Bonferroni-adjusted over 15 inputs {min(1.0, 15 * pin[jmin]):.2f}')


if __name__ == '__main__':
    main()
