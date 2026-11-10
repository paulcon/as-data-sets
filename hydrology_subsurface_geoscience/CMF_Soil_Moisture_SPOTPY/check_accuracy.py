"""Recompute the numbers of DATA.md's Accuracy section that neither the notebook nor convert_data.py prints.

    python check_accuracy.py                         # checks on the shipped CSV files only
    python check_accuracy.py <path of pone.0145180.s004.zip>   # also the objective check on the deposit

1. Objective: the deposited objective ``like`` equals minus the RMSE between each run's 239 stored
   soil-moisture values and one fixed (unknown, not deposited) observation vector. The observations
   are not in the deposit, so the check fits them: with d_i the run's simulations minus their mean over
   all valid MC and LHS runs, N RMSE_i^2 = |d_i|^2 - 2 d_i.u + c is linear in the unknown u (restricted
   to the span of the leading K = 40 right singular vectors of the d_i; the observation component
   outside that span only adds to c) and the constant c. A single (u, c) fitted to all 19,950 valid
   runs reproduces the stored RMSE to within its 2-decimal rounding if, and only if, the objective is
   such an RMSE. Needs the deposited archive (the simulations are not shipped); without it this check
   is skipped with a message naming the file.
2. Failed runs: every row with valid = 0 has a parameter outside the plausibility bounds of SPOTPY
   1.0.13's example setup ``spot_setup_cmf1d.py`` (alpha [0.0001, 0.6], ksat [0.01, 3], n [1.05, 1.4],
   porosity [0.4, 0.7]), and every row with valid = 1 is inside them, allowing for the 2-decimal
   rounding of the stored values.
3. Stored precision: decimals written and distinct values per column, and the rounding step relative
   to the sample standard deviation.

numpy only. Run time about 10 s with the archive.
"""

import io
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['alpha', 'n', 'ksat', 'porosity']
BOUNDS = {'alpha': (0.0001, 0.6), 'n': (1.05, 1.4), 'ksat': (0.01, 3.0), 'porosity': (0.4, 0.7)}
INNER = {'MC': ('S4_File_Simulation_results_of_the_CMF_model_with_MC.zip', 'CMF_MC.csv'),
         'LHS': ('S5_File_Simulation_results_of_the_CMF_model_with_LHS.zip', 'CMF_LHS.csv')}
FILES = {'MC': 'cmf_mc_samples.csv', 'LHS': 'cmf_lhs_samples.csv'}
K = 40


def load(label):
    return np.loadtxt(os.path.join(HERE, FILES[label]), delimiter=',')


def failed_rows():
    print('2. Failed runs against the plausibility bounds (half a rounding step, 0.005, allowed)')
    for label in FILES:
        A = load(label)
        P, valid = A[:, 2:6], A[:, 6] == 1
        inside = np.ones(len(A), bool)
        for j, name in enumerate(NAMES):
            lo, hi = BOUNDS[name]
            inside &= (P[:, j] >= lo - 0.005) & (P[:, j] <= hi + 0.005)
        strict = np.ones(len(A), bool)
        for j, name in enumerate(NAMES):
            lo, hi = BOUNDS[name]
            strict &= (P[:, j] > lo) & (P[:, j] < hi)
        print(f'   {label}: {np.sum(~valid)} rows valid = 0, of them {np.sum(~valid & ~strict)} with a stored '
              f'parameter outside or on a bound; {np.sum(valid)} rows valid = 1, of them '
              f'{np.sum(valid & ~inside)} outside the bounds by more than 0.005')


def precision():
    print('3. Stored precision')
    for label in FILES:
        A = load(label)
        A = A[A[:, 6] == 1]
        with open(os.path.join(HERE, FILES[label])) as fh:
            rows = [ln.split(',') for ln in fh.read().splitlines() if not ln.startswith('#')]
        dec = set()
        for r in rows:
            for v in r[1:6]:
                if v != 'nan':
                    dec.add(len(v.split('.')[1]) if '.' in v else 0)
        parts = []
        for j, name in enumerate(NAMES):
            x = A[:, 2 + j]
            parts.append(f'{name} {len(np.unique(x))} values, step/sd {0.01 / x.std(ddof=1):.3f}')
        print(f'   {label}: decimals written {sorted(dec)}; objective {len(np.unique(A[:, 1]))} values; '
              + '; '.join(parts))


def read_deposit(archive, label):
    inner, member = INNER[label]
    with zipfile.ZipFile(archive) as outer:
        data = outer.read(inner)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        text = zf.read(member).decode('ascii')
    A = np.genfromtxt(io.StringIO(text), delimiter=',', skip_header=1, comments=None,
                      missing_values='#NAME?', filling_values=np.nan)
    return A


def objective(archive):
    print('1. Objective = -RMSE of the stored simulations against one fixed observation vector')
    runs = {lab: read_deposit(archive, lab) for lab in INNER}
    sims, rmse, lab_of = [], [], []
    for lab, A in runs.items():
        ok = ~np.isnan(A[:, 0])
        print(f'   {lab}: {A.shape[0]} rows, {A.shape[1] - 6} simulated values per row, {np.sum(ok)} valid')
        sims.append(A[ok, 5:-1])
        rmse.append(-A[ok, 0])
        lab_of += [lab] * int(np.sum(ok))
    S, r, lab_of = np.vstack(sims), np.concatenate(rmse), np.array(lab_of)
    N = S.shape[1]
    D = S - S.mean(0)
    V = np.linalg.svd(D, full_matrices=False)[2][:K].T
    y = N * r ** 2 - np.sum(D ** 2, 1)
    G = np.column_stack([-2 * D @ V, np.ones(len(D))])
    sol = np.linalg.lstsq(G, y, rcond=None)[0]
    u, c = sol[:-1], sol[-1]
    r_fit = np.sqrt((np.sum(D ** 2, 1) - 2 * D @ V @ u + c) / N)
    print(f'   one fit to {len(r)} runs (K = {K}): c - |u|^2 = {c - u @ u:.1f} (must be >= 0)')
    for lab in INNER:
        e = r_fit[lab_of == lab] - r[lab_of == lab]
        print(f'   {lab}: |fitted - stored RMSE| rms {np.sqrt(np.mean(e ** 2)):.4f}, max {np.abs(e).max():.4f} '
              f'(objective and simulations stored to 0.01; rounding of the objective alone gives rms '
              f'{0.01 / np.sqrt(12):.4f})')


def main():
    if len(sys.argv) > 1:
        objective(sys.argv[1])
    else:
        print('1. skipped: give the path of pone.0145180.s004.zip (doi:10.1371/journal.pone.0145180.s004, '
              'the PLOS ONE supporting file S4; see DATA.md) as the argument to run this check')
    failed_rows()
    precision()


if __name__ == '__main__':
    main()
