"""Draw the samples and write inputs.csv, outputs.csv and gradients.csv.

    python generate_data.py                             # writes the three CSVs here
    python generate_data.py --check                     # writes to a temporary directory, compares with these

M = 1000 independent Monte Carlo draws of the eight lognormal inputs of `oscillator_model`
(x = exp(MU + SIGMA xi), xi ~ N(0, I_8) from numpy.random.default_rng(SEED)). For each draw it
writes the RMS deformation of the secondary spring from the paper's Eq. 37, the limit state
G = Fs - 3 ks rms, the exact RMS deformation from a Lyapunov solve of the 2-DOF system (no
perturbation approximation; for the accuracy check), and the exact gradients of rms and G with
respect to the log inputs z = ln x. It also prints a check of the gradients against central
finite differences in z (step 1e-6). `--check` prints the largest differences
from the shipped files and fails only on structural mismatches (shapes, NaN pattern, inputs).

Only numpy, scipy and the standard library are imported.
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oscillator_model as om  # noqa: E402

SEED = 1985
M = 1000
FMT = '%.17g'
INPUT_RTOL = 1e-8  # --check: regenerated inputs (deterministic draws) may differ from the shipped ones by round-off only


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


def compare(out, here, files):
    """Print the largest relative difference (per element) between the files in out and here.

    Returns False on a structural mismatch: a different shape or NaN/inf pattern, or inputs.csv
    beyond INPUT_RTOL."""
    bad, worst = [], (-1.0, '')
    for f in files:
        a = np.loadtxt(os.path.join(out, f), delimiter=',', comments='#', ndmin=2)
        b = np.loadtxt(os.path.join(here, f), delimiter=',', comments='#', ndmin=2)
        if a.shape != b.shape or not (np.array_equal(np.isnan(a), np.isnan(b))
                                      and np.array_equal(np.isinf(a), np.isinf(b))):
            bad.append(f'FAIL: {f}: shape {a.shape} or NaN/inf pattern differs from the shipped {b.shape}')
            continue
        ok = np.isfinite(b)
        rel = np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300)) if ok.any() else 0.0
        print(f'{f}: largest relative difference (elementwise) from the shipped file {rel:.1e}')
        if f == 'inputs.csv' and rel > INPUT_RTOL:
            bad.append(f'FAIL: {f} differs from the shipped inputs by {rel:.1e} > {INPUT_RTOL:g}')
        worst = max(worst, (rel, f))
    print(f'largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})')
    for line in bad:
        print(line)
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None,
                    help='output directory (default: this directory; with --check, a temporary directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    if args.out is None:
        if args.check:
            args.out = tempfile.mkdtemp(prefix='check_')
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        else:
            args.out = HERE
    if args.check and os.path.abspath(args.out) == HERE:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    t0 = time.perf_counter()
    X, F, D = compute()
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, F, D)
    er, eG = fd_check(X, D)
    print(f'wrote {args.out} in {time.perf_counter() - t0:.2f} s; gradient check vs central differences '
          f'(h = 1e-6 in ln x), max relative error (row 2-norm): rms {er:.1e}, G {eG:.1e}')
    if args.check and not compare(args.out, HERE, ('inputs.csv', 'outputs.csv', 'gradients.csv')):
        sys.exit(1)


if __name__ == '__main__':
    main()
