"""Draw the samples and write inputs.csv, outputs.csv and gradients.csv.

    python generate_data.py                             # writes the three CSVs here
    python generate_data.py --out /tmp/osc --check      # writes elsewhere, compares with these

M = 1000 independent Monte Carlo draws of the eight lognormal inputs of `oscillator_model`
(x = exp(MU + SIGMA xi), xi ~ N(0, I_8) from numpy.random.default_rng(SEED)). For each draw it
writes the RMS deformation of the secondary spring from the paper's Eq. 37, the limit state
G = Fs - 3 ks rms, the exact RMS deformation from a Lyapunov solve of the 2-DOF system (no
perturbation approximation; for the accuracy check), and the exact gradients of rms and G with
respect to the log inputs z = ln x. It also prints a check of the gradients against central
finite differences in z (step 1e-6).

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oscillator_model as om  # noqa: E402

SEED = 1985
M = 1000
FMT = '%.17g'


def compute(seed=SEED, n=M):
    X = om.sample(n, seed)
    r, G = om.rms(X), om.limit_state(X)
    r_exact = np.sqrt(om.exact_mean_squares(X)[:, 0])
    F = np.column_stack([r, G, r_exact])
    D = np.column_stack([om.grad_log_rms(X), om.grad_log_limit_state(X)])
    return X, F, D


def fd_check(X, D, h=1e-6):
    Z = np.log(X)
    E = np.eye(8)
    fd_r = np.column_stack([(om.rms(np.exp(Z + h*e)) - om.rms(np.exp(Z - h*e)))/(2*h) for e in E[:7]])
    fd_G = np.column_stack([(om.limit_state(np.exp(Z + h*e)) - om.limit_state(np.exp(Z - h*e)))/(2*h)
                            for e in E])
    er = np.linalg.norm(fd_r - D[:, :7], axis=1)/np.linalg.norm(D[:, :7], axis=1)
    eG = np.linalg.norm(fd_G - D[:, 7:], axis=1)/np.linalg.norm(D[:, 7:], axis=1)
    return er.max(), eG.max()


def write(out, X, F, D):
    names = list(om.NAMES)
    desc = (f'{len(X)} Monte Carlo draws (numpy default_rng({SEED})) of the 8 independent lognormal '
            'inputs of the 2-DOF oscillator benchmark (Der Kiureghian and De Stefano 1991, via OpenTURNS); '
            'physical values, consistent units (not reported by the benchmark)')
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt=FMT, delimiter=',',
               header=desc + '\n' + ','.join(names))
    np.savetxt(os.path.join(out, 'outputs.csv'), F, fmt=FMT, delimiter=',',
               header='rms = sqrt(E[x_s^2]) from Igusa and Der Kiureghian (1985) Eq. 37 (RMS deformation '
                      'of the secondary spring); G = Fs - 3 ks rms (failure if G <= 0); rms_exact = exact RMS '
                      'deformation x_s - x_p of the 2-DOF system from a Lyapunov solve (accuracy check only). '
                      'Consistent units (not reported)\nrms,G,rms_exact')
    cols = [f'drms/dln_{n}' for n in names[:7]] + [f'dG/dln_{n}' for n in names]
    np.savetxt(os.path.join(out, 'gradients.csv'), D, fmt=FMT, delimiter=',',
               header='exact gradients with respect to the LOG inputs z = ln x: columns 0-6 d rms/d ln x_j '
                      '(j = mp..S0), columns 7-14 dG/d ln x_j (j = mp..Fs); divide by x_j for physical '
                      'gradients\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    t0 = time.perf_counter()
    X, F, D = compute()
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, F, D)
    er, eG = fd_check(X, D)
    print(f'wrote {args.out} in {time.perf_counter() - t0:.2f} s; gradient check vs central differences '
          f'(h = 1e-6 in ln x), max relative error (row 2-norm): rms {er:.1e}, G {eG:.1e}')
    if args.check:
        same = [np.array_equal(np.loadtxt(os.path.join(args.out, f), delimiter=','),
                               np.loadtxt(os.path.join(HERE, f), delimiter=','))
                for f in ('inputs.csv', 'outputs.csv', 'gradients.csv')]
        print('identical to the shipped files (inputs, outputs, gradients):', same)


if __name__ == '__main__':
    main()
