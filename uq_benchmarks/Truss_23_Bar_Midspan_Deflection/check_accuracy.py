"""Recompute the numbers of DATA.md's Accuracy section that the notebook and generate_data.py do not print.

    python check_accuracy.py            # about 3.5-4 minutes, one process

1. Unit-load (virtual-work) form of the deflection. The truss is statically determinate, so the bar
   forces follow from equilibrium alone and V1 = sum_j P_j (a_j / (E1 A1) + b_j / (E2 A2)) with
   constants a_j, b_j (m^2 per unit load) computed here from `truss_model.bar_forces`. The shipped
   outputs and the gradients of this closed form are compared with the files at all 1000 rows.
2. Total Sobol' indices and moments by crude Monte Carlo with the stiffness solver, using the Jansen
   estimator with N = 500 000 base samples in standard-normal coordinates (N (m + 1) = 5 500 000 model
   runs, the number of runs of the paper's reference, Table 2), seed 737. The 95 % intervals are
   normal-theory intervals of the sample means (delta method for the ratio is not used: the
   variance's own error is printed separately).

3. Sampling spread of three total-index estimators at the paper's size N = 500 000, and where the
   paper's reference column (Table 2) falls against each. For each estimator, R = 100 independent
   replicates with N = 20 000 base samples (seeds 1000-1099) use the closed form of check 1 (the same
   function as the stiffness solver, much faster); the replicate sd is scaled by sqrt(20 000 / 500 000)
   to N = 500 000. With f_A = f(A), f_i = f(A_B^(i)) (A with column i from B), V the sample variance of
   f_A and f0 the sample mean of f_A:
     Jansen (1999)              T_i = mean((f_A - f_i)^2) / (2 V)
     Homma and Saltelli (1996)  T_i = 1 - (mean(f_A f_i) - f0^2) / V
     Sobol' (2007)              T_i = mean(f_A (f_A - f_i)) / V
   With f0 and V both taken from f_A, the Homma-Saltelli and Sobol' (2007) forms are algebraically
   the same up to the factor (N - 1)/N, so their spreads agree. The values of check 2 (Jansen,
   N = 500 000) serve as the true indices; each reference entry is given as (reference - true) / sd
   for each estimator.

Only numpy is imported (via truss_model).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import truss_model as tm  # noqa: E402

SEED_SOBOL = 737
N_SOBOL = 500_000
CHUNK = 50_000


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def closed_form_constants():
    """a_j, b_j with V1 = sum_j P_j (a_j / k1 + b_j / k2), k1 = E1 A1, k2 = E2 A2."""
    n, _ = tm.bar_forces('unit_mid')
    a = np.empty(6)
    b = np.empty(6)
    for j in range(6):
        P = np.zeros(6)
        P[j] = 1.0
        N, _ = tm.bar_forces(P)
        t = N * n * tm.BAR_LENGTHS
        a[j] = t[tm.BAR_GROUPS == 0].sum()
        b[j] = t[tm.BAR_GROUPS == 1].sum()
    return a, b


# Table 2, p. 2358: reference (crude Monte Carlo, 5 500 000 runs) and LAR sparse PC (70 runs), order of NAMES
TABLE2_REF = dict(E1=0.367, E2=0.010, A1=0.388, A2=0.014, P1=0.004, P2=0.031, P3=0.075, P4=0.079, P5=0.035, P6=0.005)
TABLE2_LAR = dict(E1=0.373, E2=0.012, A1=0.374, A2=0.013, P1=0.005, P2=0.037, P3=0.073, P4=0.079, P5=0.034, P6=0.004)
N_REP, R_REP, SEED_REP = 20_000, 100, 1000


def closed_form(XI, a, b):
    Z = tm.physical(XI)
    P = Z[:, 4:]
    return P @ a / (Z[:, 0] * Z[:, 2]) + P @ b / (Z[:, 1] * Z[:, 3])


def check1():
    Z, V, G = load('inputs.csv'), load('outputs.csv')[:, 0], load('gradients.csv')
    a, b = closed_form_constants()
    print('1. closed form V1 = sum_j P_j (a_j/(E1 A1) + b_j/(E2 A2))')
    print('   a_j (m^2):', np.array2string(a, precision=6))
    print('   b_j (m^2):', np.array2string(b, precision=6))
    k1, k2 = Z[:, 0] * Z[:, 2], Z[:, 1] * Z[:, 3]
    P = Z[:, 4:]
    Va, Vb = P @ a, P @ b
    Vc = Va / k1 + Vb / k2
    Gc = np.empty_like(G)
    Gc[:, 0] = -Va / k1**2 * Z[:, 2]
    Gc[:, 2] = -Va / k1**2 * Z[:, 0]
    Gc[:, 1] = -Vb / k2**2 * Z[:, 3]
    Gc[:, 3] = -Vb / k2**2 * Z[:, 1]
    Gc[:, 4:] = a / k1[:, None] + b / k2[:, None]
    print(f'   outputs.csv vs closed form, max relative difference: {np.max(np.abs(Vc / V - 1)):.1e}')
    # compare gradients scaled to standard-normal coordinates, so that all columns are comparable
    XI = load('inputs_standard.csv')
    J = tm.dphysical_dxi(XI)
    rel = np.linalg.norm((Gc - G) * J, axis=1) / np.linalg.norm(G * J, axis=1)
    print(f'   gradients.csv vs closed-form gradients (in xi, row 2-norm), max relative difference: {rel.max():.1e}')
    share = (Va / k1) / Vc
    print(f'   share of V1 from the horizontal bars (a-term): min {share.min():.4f}, max {share.max():.4f}')
    return a, b


def check2():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED_SOBOL)
    m = 10
    sum_f = sum_f2 = 0.0
    num = np.zeros(m)
    num2 = np.zeros(m)
    n_done = 0
    fA_all = []
    while n_done < N_SOBOL:
        n = min(CHUNK, N_SOBOL - n_done)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((n, m))
        fA = tm.deflection_xi(A)
        fA_all.append(fA)
        for i in range(m):
            ABi = A.copy()
            ABi[:, i] = B[:, i]
            d2 = (fA - tm.deflection_xi(ABi))**2 / 2
            num[i] += d2.sum()
            num2[i] += (d2**2).sum()
        n_done += n
        print(f'   ... {n_done} of {N_SOBOL} base samples, {time.perf_counter() - t0:.0f} s', flush=True)
    fA = np.concatenate(fA_all)
    mean, var = fA.mean(), fA.var(ddof=1)
    sd = np.sqrt(var)
    se_mean = sd / np.sqrt(N_SOBOL)
    se_var = np.sqrt((np.mean((fA - mean)**4) - var**2) / N_SOBOL)
    T = num / N_SOBOL / var
    se_T = np.sqrt(num2 / N_SOBOL - (num / N_SOBOL)**2) / np.sqrt(N_SOBOL) / var
    print(f'2. crude Monte Carlo, N = {N_SOBOL} base samples ({N_SOBOL * (m + 1)} runs), seed {SEED_SOBOL}, '
          f'{time.perf_counter() - t0:.0f} s')
    print(f'   mean V1 = {mean:.6f} +- {1.96 * se_mean:.6f} m; sd V1 = {sd:.6f} m '
          f'(variance relative 95 % half-width {1.96 * se_var / var:.4f}); cv {sd / mean:.4f}')
    print('   total Sobol indices (Jansen), 95 % half-width of the numerator only:')
    for i, nme in enumerate(tm.NAMES):
        print(f'     {nme:3s} {T[i]:.4f} +- {1.96 * se_T[i]:.1e}')
    print(f'   sum of total indices {T.sum():.4f}')
    return T


def check3(a, b, T_true):
    t0 = time.perf_counter()
    m = 10
    est = np.empty((R_REP, 3, m))
    for r in range(R_REP):
        rng = np.random.default_rng(SEED_REP + r)
        A = rng.standard_normal((N_REP, m))
        B = rng.standard_normal((N_REP, m))
        fA = closed_form(A, a, b)
        V, f0 = fA.var(ddof=1), fA.mean()
        for i in range(m):
            ABi = A.copy()
            ABi[:, i] = B[:, i]
            fi = closed_form(ABi, a, b)
            est[r, 0, i] = np.mean((fA - fi)**2) / (2 * V)
            est[r, 1, i] = 1 - (np.mean(fA * fi) - f0**2) / V
            est[r, 2, i] = np.mean(fA * (fA - fi)) / V
    sd = est.std(axis=0, ddof=1) * np.sqrt(N_REP / N_SOBOL)
    names = ('Jansen', 'Homma-Saltelli', "Sobol' 2007")
    f_big = closed_form(np.random.default_rng(1).standard_normal((10 ** 6, m)), a, b)
    print(f"3. sd of the total-index estimators at N = {N_SOBOL} ({R_REP} replicates at N = {N_REP}, scaled by "
          f"sqrt({N_REP}/{N_SOBOL}); closed form; {time.perf_counter() - t0:.0f} s); "
          f"E[V1]^2 / Var V1 = {f_big.mean() ** 2 / f_big.var():.1f} (10^6 draws, seed 1)")
    print('   input  true(check 2)  sd: ' + '  '.join(f'{n:>14s}' for n in names) +
          '   Table 2 ref  (ref - true)/sd: ' + '  '.join(f'{n:>14s}' for n in names) + '   LAR  (LAR - true)/sd (Homma-Saltelli)')
    for i, nme in enumerate(tm.NAMES):
        ref, lar = TABLE2_REF[nme], TABLE2_LAR[nme]
        z = (ref - T_true[i]) / sd[:, i]
        print(f'   {nme:3s}    {T_true[i]:.4f}         ' + '  '.join(f'{v:14.2g}' for v in sd[:, i]) +
              f'   {ref:.3f}                       ' + '  '.join(f'{v:+14.1f}' for v in z) +
              f'   {lar:.3f}  {(lar - T_true[i]) / sd[1, i]:+.1f}')
    zz = np.array([[(TABLE2_REF[n] - T_true[i]) / sd[k, i] for i, n in enumerate(tm.NAMES)] for k in range(3)])
    print('   largest |(ref - true)/sd| over the ten inputs: ' + ', '.join(f'{n} {np.abs(zz[k]).max():.1f}' for k, n in enumerate(names)))


if __name__ == '__main__':
    t = time.perf_counter()
    a, b = check1()
    T_true = check2()
    check3(a, b, T_true)
    print(f'total {time.perf_counter() - t:.0f} s')
