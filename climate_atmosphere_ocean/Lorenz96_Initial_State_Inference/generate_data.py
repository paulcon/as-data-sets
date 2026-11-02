"""Write the synthetic truth, the observations, and the misfit samples with exact gradients.

    python generate_data.py                                  # everything, written here (about 4-5 min)
    python generate_data.py --rows 0:20 --out <scratch dir> --check   # a subset, compared with the files here

The setting is Sec. 5.2 of Schuster, Constantine and Sullivan (2017): the one-level Lorenz-96 system
with K = 36, inference variables x = (X_1(0), ..., X_36(0), F), prior N(0, 4 I_37), data at every
step t_i = 0.01 i of a forward solve on [0, 10] from a ground truth with F = 8, plus independent
N(0, 0.1) noise. The paper does not report its ground-truth initial state or its noise realization,
so both are drawn here from a saved seed:

    rng = numpy.random.default_rng(SEED_DATA)
    X_true(0) = 2 * rng.standard_normal(36)              # one draw from the prior N(0, 4 I)
    noise     = sqrt(0.1) * rng.standard_normal((1001, 36))

The M = 1000 input samples are independent draws from the prior, from a second seed
(SEED_SAMPLES), as plain Monte Carlo. For each sample this computes the misfit
Phi_T(x) = sum_{t_i <= T} |X(t_i; x) - d_i|^2 / (2 * 0.1) for the observation windows [0, T],
T in WINDOWS, and its exact gradient with respect to the physical inputs (the discrete
tangent-linear recursion of the Runge-Kutta scheme in lorenz96_model.py).

Files written: truth.csv, observations.csv, inputs.csv, outputs.csv, gradients.csv (see DATA.md).
`--check` compares what was computed with the files next to this script and prints the largest
relative differences. Only numpy and the standard library are imported (and lorenz96_model).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lorenz96_model as lm  # noqa: E402

SEED_DATA = 293          # truth initial state and observation noise
SEED_SAMPLES = 1712      # prior samples
M = 1000                 # number of prior samples
#: End times T of the observation windows [0, T]; 10 is the paper's window.
WINDOWS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
BATCH = 250              # samples per call (a few MB of tangent arrays per batch)
FMT = '%.17g'            # round-trip exact


def truth_and_data():
    """x_true (37,) and observations d (1001, 36), from SEED_DATA."""
    rng = np.random.default_rng(SEED_DATA)
    x_true = np.concatenate([np.sqrt(lm.PRIOR_VAR) * rng.standard_normal(lm.K), [lm.F_TRUE]])
    d = lm.synthetic_data(x_true, rng)
    return x_true, d


def prior_samples():
    """M x 37 prior draws, from SEED_SAMPLES."""
    rng = np.random.default_rng(SEED_SAMPLES)
    return np.sqrt(lm.PRIOR_VAR) * rng.standard_normal((M, lm.M_INPUTS))


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def compute(X, d):
    phi = np.empty((len(X), len(WINDOWS)))
    G = np.empty((len(X), len(WINDOWS), lm.M_INPUTS))
    for a in range(0, len(X), BATCH):
        phi[a:a + BATCH], G[a:a + BATCH] = lm.misfit_and_grad(X[a:a + BATCH], d, WINDOWS)
    return phi, G


def write(out, x_true, d, X, phi, G):
    names = list(lm.NAMES)
    t = np.arange(d.shape[0]) * lm.DT
    np.savetxt(os.path.join(out, 'truth.csv'), x_true[None], fmt=FMT, delimiter=',',
               header=f'synthetic ground truth x_true = (X_1(0), ..., X_36(0), F): initial state drawn '
                      f'from the prior N(0, 4) with numpy default_rng({SEED_DATA}), F = 8 (paper Sec. 5.2)\n'
                      + ','.join(names))
    np.savetxt(os.path.join(out, 'observations.csv'), np.column_stack([t, d]), fmt=FMT, delimiter=',',
               header=f'observations d_ik = X_k(t_i; x_true) + N(0, 0.1) noise, t_i = 0.01 i, i = 0..1000 '
                      f'(RK4, dt = 0.01; noise from default_rng({SEED_DATA}) after the truth draw)\n'
                      't,' + ','.join(f'd{k}' for k in range(1, lm.K + 1)))
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt=FMT, delimiter=',',
               header=f'{len(X)} Monte Carlo draws from the prior N(0, 4 I_37) (default_rng({SEED_SAMPLES})), '
                      'physical units (dimensionless state and forcing)\n' + ','.join(names))
    np.savetxt(os.path.join(out, 'outputs.csv'), phi, fmt=FMT, delimiter=',',
               header='data misfit Phi_T(x) = sum_{t_i <= T} |X(t_i; x) - d_i|^2 / (2 * 0.1), '
                      'dimensionless, for windows [0, T]\n' + ','.join(f'Phi_{T:g}' for T in WINDOWS))
    cols = [f'dPhi_{T:g}/d{n}' for T in WINDOWS for n in names]
    np.savetxt(os.path.join(out, 'gradients.csv'), G.reshape(len(X), -1), fmt=FMT, delimiter=',',
               header='exact gradients of Phi_T w.r.t. the physical inputs (discrete tangent-linear RK4); '
                      f'column {lm.M_INPUTS}*w + j is dPhi_(WINDOWS[w])/dx_j\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', default=None, help='a:b, a subset of the M samples')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if os.path.abspath(args.out) == HERE and (args.check or args.rows):
        ap.error('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')
    t0 = time.perf_counter()
    x_true, d = truth_and_data()
    X = prior_samples()
    rows = slice(None)
    if args.rows:
        a, b = (int(v) for v in args.rows.split(':'))
        rows = slice(a, b)
    X = X[rows]
    phi, G = compute(X, d)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, x_true, d, X, phi, G)
    print(f'{len(X)} samples, {len(WINDOWS)} windows: {time.perf_counter() - t0:.0f} s')
    if args.check:
        here = {n: read_csv(os.path.join(HERE, n)) for n in
                ('truth.csv', 'observations.csv', 'inputs.csv', 'outputs.csv', 'gradients.csv')}
        new = {n: read_csv(os.path.join(args.out, n)) for n in here}
        for n in here:
            ref = here[n] if n in ('truth.csv', 'observations.csv') else here[n][rows]
            rel = np.max(np.abs(new[n] - ref)) / np.max(np.abs(ref))
            print(f'{n:17s} largest difference / largest value: {rel:.1e}')


if __name__ == '__main__':
    main()
