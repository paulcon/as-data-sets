"""Recompute the numbers of DATA.md (Speed_Reducer_Shaft_Limit_State) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py

Checks 1-2 need only numpy and scipy. Checks 3-5 compare with the uqtestfuns package
(pip install uqtestfuns==0.7.0); if it is not installed the script runs checks 1-2 and then exits
with an error that says what to install. Deterministic (check 5 uses seed 2026). About 3 s.

Checks:
 1. The map to standard normal coordinates: round trip to_normal(from_normal(xi)) on the shipped
    samples, and dx/dxi of shaft_model.dx_dxi against central differences of from_normal (step 1e-6).
 2. Gradients in standard normal coordinates, dg/dxi = dg/dx * dx/dxi, against central differences
    of g(from_normal(xi)) (step 1e-6), all shipped samples (largest relative error, row 2-norm).
 3. g of shaft_model against uqtestfuns RSSpeedReducerShaft at the shipped inputs.
 4. The marginals: Phi^{-1} of the uqtestfuns marginal CDFs at the shipped inputs against
    shaft_model.to_normal, and the uqtestfuns Gumbel parameters against MU_F, BETA_F.
 5. Failure probability P[g <= 0] from 10^7 samples drawn by uqtestfuns' own input sampler
    (seeded through numpy), with a 95% normal-approximation interval.
"""
import os
import sys
import time

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shaft_model as sm  # noqa: E402


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def main():
    t0 = time.perf_counter()
    X = load('inputs.csv')
    g = load('outputs.csv')[:, 0]
    G = load('gradients.csv')
    Z = sm.to_normal(X)

    # 1
    rt = np.max(np.abs(sm.to_normal(sm.from_normal(Z)) - Z))
    J = sm.dx_dxi(Z)
    Jfd = np.empty_like(J)
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-6
        Jfd[:, j] = (sm.from_normal(Z + e)[:, j] - sm.from_normal(Z - e)[:, j]) / 2e-6
    print(f'1. round trip xi -> x -> xi on the shipped samples: max |difference| {rt:.0e}; '
          f'dx/dxi vs central differences: max relative error {np.max(np.abs(Jfd - J) / J):.1e}')

    # 2
    Gz = G * J
    Gfd = np.empty_like(Gz)
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-6
        Gfd[:, j] = (sm.limit_state(sm.from_normal(Z + e)) - sm.limit_state(sm.from_normal(Z - e))) / 2e-6
    err = np.linalg.norm(Gfd - Gz, axis=1) / np.linalg.norm(Gz, axis=1)
    print(f'2. dg/dxi vs central differences in xi, all {len(X)} samples: max relative error {err.max():.1e}')

    try:
        import uqtestfuns as uq
    except ImportError:
        print('checks 3-5 need uqtestfuns 0.7.0: pip install uqtestfuns==0.7.0, then rerun '
              'python check_accuracy.py', file=sys.stderr)
        sys.exit(3)

    # 3
    fun = uq.RSSpeedReducerShaft()
    gu = fun(X)
    print(f'3. g vs uqtestfuns {uq.__version__} RSSpeedReducerShaft at the shipped inputs: max |difference| '
          f'/ max |g| {np.max(np.abs(gu - g)) / np.max(np.abs(g)):.0e}')

    # 4
    marg = fun.prob_input.marginals
    zu = np.column_stack([special.ndtri(marg[j].cdf(X[:, j])) for j in range(5)])
    pf = np.asarray(marg[2].parameters, float)
    print('4. Phi^-1(uqtestfuns marginal CDF) vs to_normal, max |difference| per input: '
          + ', '.join(f'{n} {np.max(np.abs(zu[:, j] - Z[:, j])):.0e}' for j, n in enumerate(sm.NAMES))
          + f'; Gumbel location, scale in uqtestfuns {pf[0]:.6f}, {pf[1]:.6f} '
          f'(shaft_model {sm.MU_F:.6f}, {sm.BETA_F:.6f})')

    # 5
    rng = np.random.default_rng(2026)
    n, k = 0, 0
    for _ in range(10):
        xs = fun.prob_input.get_sample(10**6, rng=rng)
        k += int(np.sum(fun(xs) <= 0))
        n += len(xs)
    p = k / n
    hw = 1.96 * np.sqrt(p * (1 - p) / n)
    print(f'5. P[g <= 0] from {n:.0e} uqtestfuns samples: {p:.4e} (95% interval {p - hw:.4e} to {p + hw:.4e})')
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
