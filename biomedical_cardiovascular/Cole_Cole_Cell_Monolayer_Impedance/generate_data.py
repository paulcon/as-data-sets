"""Write inputs.csv, outputs.csv and gradients.csv for the Cole-Cole cell-monolayer model.

    python generate_data.py                  # writes the three CSVs next to this script
    python generate_data.py --out /tmp/cole --check

Draws N = 1100 independent samples of (R_inf, R_0, alpha, tau) from lognormals with the
means and standard deviations of the paper's Table 1 (cole_model.LOG_MU, LOG_SIGMA), with
numpy's default_rng(SEED). Inputs are rounded to 12 significant digits before the model is
evaluated, so the files are consistent with the inputs as written. Every draw is kept; the
rows outside the physical domain (alpha >= 1, or R_0 <= R_inf) are left in the files and
removed in the notebook by `cole_model.is_physical` (DATA.md gives the counts).

For every row the script writes Re Z and Im Z (ohm) at the 31 angular frequencies OMEGA,
and their exact gradients with respect to the physical inputs. Run time: well under 1 s.

`--check` compares the new files with the ones next to this script.
Only numpy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cole_model as cm  # noqa: E402

SEED = 5861
N = 1100
#: angular frequencies in rad/s: 10^(k/4), k = 0..28 (1 to 1e7), plus 2e5 and 2e6,
#: the two values the paper's text singles out (p. 3 and p. 5).
OMEGA = np.sort(np.r_[10.0**(np.arange(29) / 4), 2e5, 2e6])
K = len(OMEGA)


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def draw():
    rng = np.random.default_rng(SEED)
    X = np.exp(cm.LOG_MU + cm.LOG_SIGMA * rng.standard_normal((N, 4)))
    return np.array([[float(f'{v:.12g}') for v in row] for row in X])


def compute(X):
    Z, G = cm.impedance_and_grad(X, OMEGA)
    f = np.concatenate([Z.real, Z.imag], axis=1)                      # (N, 2K)
    g = np.stack([G.real, G.imag], axis=1).reshape(len(X), -1)        # (N, 2*K*4)
    return f, g


def write(out, X, f, g):
    w = [f'{v:.6g}' for v in OMEGA]
    prov = (f'cole_model.py (Zhuang et al. 2022, Eq. 1, cell-monolayer term); lognormal inputs with '
            f'Table 1 means and sds, numpy default_rng({SEED}); written by generate_data.py')
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt='%.12g', delimiter=',',
               header=f'{N} lognormal samples of the 4 Cole parameters in physical units '
                      f'(all draws; rows with alpha >= 1 or R_0 <= R_inf are outside the physical '
                      f'domain and are kept). {prov}\n'
                      'R_inf (ohm),R_0 (ohm),alpha (-),tau (s)')
    np.savetxt(os.path.join(out, 'outputs.csv'), f, fmt='%.12e', delimiter=',',
               header=f'Re Z then Im Z (ohm) at omega = {", ".join(w)} rad/s. {prov}\n'
                      + ','.join([f'ReZ({s})' for s in w] + [f'ImZ({s})' for s in w]))
    names = [f'{p}Z({s})/d{n}' for p in ('dRe', 'dIm') for s in w for n in cm.NAMES]
    np.savetxt(os.path.join(out, 'gradients.csv'), g, fmt='%.12e', delimiter=',',
               header='exact gradients w.r.t. the physical inputs (ohm/ohm, ohm/ohm, ohm, ohm/s); '
                      'column 4*K*p + 4*k + j for part p (0 Re, 1 Im), frequency k, input j, '
                      f'so G.reshape(N, 2, {K}, 4). {prov}\n' + ','.join(names))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs next to this script')
    args = ap.parse_args()
    if os.path.abspath(args.out) == HERE and args.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    t0 = time.perf_counter()
    X = draw()
    f, g = compute(X)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, f, g)
    ok = cm.is_physical(X)
    print(f'wrote {args.out}: {N} rows ({ok.sum()} physical; alpha >= 1: {(X[:, 2] >= 1).sum()}, '
          f'R_0 <= R_inf: {(X[:, 1] <= X[:, 0]).sum()}), {K} frequencies, '
          f'{time.perf_counter() - t0:.2f} s')
    if args.check:
        for name, new in (('inputs.csv', X), ('outputs.csv', f), ('gradients.csv', g)):
            old = read_csv(os.path.join(HERE, name))
            back = read_csv(os.path.join(args.out, name))
            rel = np.max(np.abs(back - old) / (np.abs(old) + 1e-300))
            print(f'{name}: shape {old.shape}, max relative difference from the shipped file '
                  f'{rel:.1e}; identical text: '
                  f'{open(os.path.join(HERE, name)).read() == open(os.path.join(args.out, name)).read()}')


if __name__ == '__main__':
    main()
