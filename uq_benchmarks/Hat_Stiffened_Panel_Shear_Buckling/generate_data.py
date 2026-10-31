"""Write inputs.csv, outputs.csv and gradients.csv for the hat-stiffened panel shear buckling model.

    python generate_data.py                       # writes the three CSVs here
    python generate_data.py --out /tmp/p --check  # writes elsewhere, compares with the files here
    python generate_data.py --variant openturns --out /tmp/ot   # the OpenTURNS-as-coded thickness (not shipped)

Draws M = 5000 Monte Carlo samples of the ten inputs of `panel_model` (OpenTURNS
stiffened-panel use case: truncated normal E, nine uniforms, normal copula with Spearman
correlation -0.8, parameter -0.8135, between f_1 and f_2; the thickness t is the
memorandum's panel, U(8.02e-4, 8.181e-4) m, ten times the OpenTURNS range) by the inverse
CDF of numpy.random.default_rng(SEED).random((M, 10)), with SEED = 1994, and evaluates the
critical shear buckling load (N_xy)_cr (N/m) and its exact gradient with respect to the
physical inputs. `--variant openturns` uses the same draws with t in the OpenTURNS range
U(8.02e-5, 8.181e-5) m instead (the notebook recomputes this variant from the model).

It also checks the exact gradients at every sample against (1) central finite
differences with step 1e-6 times each input's standard deviation and (2) a complex step
(step 1e-30 times the input's value), and prints the largest relative errors (row
2-norm), and counts samples outside physical meaning.

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import panel_model as pm  # noqa: E402

SEED = 1994
M = 5000
H_FD = 1e-6  # central-difference step in units of each input's standard deviation (check only)

PROVENANCE = ('panel_model.py (Ko and Jackson, NASA TM-4644, 1994, formula and distributions from the OpenTURNS '
              'v1.25.1 stiffened-panel use case, thickness t = 10x the OpenTURNS range to match the memorandum); '
              f'Monte Carlo, inverse-CDF draws from numpy default_rng({SEED}).random((M, 10)); '
              'written by generate_data.py')


def draw(seed=SEED, M=M, variant='tm'):
    """M samples of the physical inputs, shape (M, 10)."""
    return pm.sample(M, np.random.default_rng(seed), variant=variant)


def gradient_check(X, variant='tm'):
    """Largest relative errors (row 2-norm) of the exact gradient vs central differences and a complex step."""
    _, cov = pm.mean_and_cov(variant=variant)
    sd = np.sqrt(np.diag(cov))
    G = pm.grad_buckling_load(X)
    F = np.empty_like(G)
    C = np.empty_like(G)
    for j in range(pm.M):
        e = np.zeros(pm.M)
        e[j] = H_FD * sd[j]
        F[:, j] = (pm.buckling_load(X + e) - pm.buckling_load(X - e)) / (2 * e[j])
        hc = 1e-30 * X[:, j]
        Xc = X.astype(complex)
        Xc[:, j] += 1j * hc
        C[:, j] = pm.buckling_load(Xc).imag / hc
    nG = np.linalg.norm(G, axis=1)
    return np.max(np.linalg.norm(F - G, axis=1) / nG), np.max(np.linalg.norm(C - G, axis=1) / nG)


def unphysical(X):
    """Number of samples outside physical meaning (any non-positive length or modulus, nu outside
    (0, 0.5), f_2 >= f_1, 2t >= h_c, h0 <= 0 or Q <= 0)."""
    q = pm.intermediates(X)
    bad = ((X <= 0).any(axis=1) | (X[:, 1] >= 0.5) | (X[:, 5] >= X[:, 4]) | (2 * X[:, 6] >= X[:, 2])
           | (q['h0'] <= 0) | (q['Q'] <= 0))
    return int(bad.sum())


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, X, f, G, variant='tm'):
    names = list(pm.NAMES)
    units = list(pm.UNITS)
    tdesc = ('t in [8.02e-4, 8.181e-4] m, the memorandum panel' if variant == 'tm'
             else 'VARIANT: t in [8.02e-5, 8.181e-5] m as coded in OpenTURNS')
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.17g', delimiter=',',
               header=f'{len(X)} Monte Carlo samples of the 10 physical inputs (truncated normal E, uniforms, '
                      f'normal copula with Spearman -0.8 (parameter -0.8135) between f_1 and f_2; {tdesc}). '
                      f'{PROVENANCE}\n'
                      + ','.join(f'{n} [{u}]' for n, u in zip(names, units)))
    np.savetxt(os.path.join(out, 'outputs.csv'), f[:, None], fmt='%.17g', delimiter=',',
               header=f'critical shear buckling load (N_xy)_cr (shear force per unit length). {PROVENANCE}\n'
                      'N_xy_cr [N/m]')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header='exact gradient of (N_xy)_cr with respect to the PHYSICAL inputs. '
                      f'{PROVENANCE}\n'
                      + ','.join(f'dN/d{n} [N/m per {u}]' if u != '-' else f'dN/d{n} [N/m]'
                                 for n, u in zip(names, units)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    ap.add_argument('--variant', choices=('tm', 'openturns'), default='tm',
                    help="thickness range: 'tm' (default, shipped) or 'openturns' (as coded there; not shipped)")
    args = ap.parse_args()
    if args.variant != 'tm' and os.path.abspath(args.out) == HERE:
        ap.error('--variant openturns must be written with --out to another directory (the shipped files are tm)')
    if args.variant != 'tm' and args.check:
        ap.error('--check compares with the shipped files, which are the tm variant')

    t0 = time.perf_counter()
    X = draw(variant=args.variant)
    f = pm.buckling_load(X)
    G = pm.grad_buckling_load(X)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, f, G, args.variant)
    t1 = time.perf_counter()
    print(f'wrote {args.out}: {len(X)} samples in {t1 - t0:.2f} s; (N_xy)_cr in [{f.min():.3f}, {f.max():.3f}] N/m')
    efd, ecs = gradient_check(X, args.variant)
    print(f'exact gradients: max relative error vs central differences (step 1e-6 sd) {efd:.1e}, '
          f'vs complex step {ecs:.1e}')
    print(f'samples outside physical meaning: {unphysical(X)}')
    if args.check:
        same = [np.array_equal(read_csv(os.path.join(args.out, n)), read_csv(os.path.join(HERE, n)))
                for n in ('inputs.csv', 'outputs.csv', 'gradients.csv')]
        print('files identical to the shipped ones (inputs, outputs, gradients):', same)


if __name__ == '__main__':
    main()
