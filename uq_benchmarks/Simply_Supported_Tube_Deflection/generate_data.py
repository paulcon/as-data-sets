"""Draw the samples and write inputs.csv, outputs.csv and gradients.csv.

    python generate_data.py                              # writes the three CSVs here
    python generate_data.py --out /tmp/tube --check      # writes elsewhere, compares with these

M = 1000 independent Monte Carlo draws (not Latin hypercube) of the six inputs of `tube_model`
from the distributions of the OpenTURNS use case "Vertical deflection of a tube"
(numpy.random.default_rng(SEED), columns drawn in the order F, L, a, De, di, E). For each draw it
writes the deflection at the load and the two end slopes (I = pi (De^4 - di^4)/64), and their exact
gradients with respect to the physical inputs. It prints two checks: the gradients against central
finite differences (relative step 1e-6 per input), and the closed forms against the unit-load
method (numerical quadrature) at the first 20 samples.

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tube_model as tm  # noqa: E402

SEED = 2011
M = 1000
FMT = '%.17g'
FILES = ('inputs.csv', 'outputs.csv', 'gradients.csv')


def compute(seed=SEED, n=M):
    X = tm.sample(n, seed)
    return X, tm.outputs(X), tm.gradients(X).reshape(n, 18)


def fd_check(X, D, rel=1e-6):
    """Max relative error (row 2-norm, per QoI) of the gradients against central differences."""
    G = D.reshape(len(X), 3, 6)
    fd = np.empty_like(G)
    for j in range(6):
        h = rel * np.abs(X[:, j])
        Xp, Xm = X.copy(), X.copy()
        Xp[:, j] += h
        Xm[:, j] -= h
        fd[:, :, j] = (tm.outputs(Xp) - tm.outputs(Xm)) / (2 * h[:, None])
    return np.max(np.linalg.norm(fd - G, axis=2) / np.linalg.norm(G, axis=2), axis=0)


def quad_check(X, F, k=20):
    """Max relative difference between the closed forms and the unit-load integrals, per QoI."""
    V = np.array([tm.virtual_work(x) for x in X[:k]])
    return np.max(np.abs(V / F[:k] - 1), axis=0)


def write(out, X, F, D):
    desc = (f'{len(X)} Monte Carlo draws (numpy default_rng({SEED})) of the 6 independent inputs of the OpenTURNS '
            'use case "Vertical deflection of a tube": F~N(1,0.1), L~N(1.5,0.01), a~U(0.7,1.2), '
            'De~Tri(0.75,0.8,0.85), di~Tri(0.09,0.1,0.11), E~N(200000,2000); physical values, consistent units '
            '(not stated by the benchmark)')
    np.savetxt(os.path.join(out, 'inputs.csv'), X, fmt=FMT, delimiter=',',
               header=desc + '\n' + ','.join(tm.NAMES))
    np.savetxt(os.path.join(out, 'outputs.csv'), F, fmt=FMT, delimiter=',',
               header='simply supported tube, downward point load F at x = a: deflection = v(a) = -F a^2 b^2/(3 E L I) '
                      '(length units), theta_left = v\'(0) = -F b (L^2-b^2)/(6 E L I), theta_right = v\'(L) = '
                      'F a (L^2-a^2)/(6 E L I) (rad); b = L - a, I = pi (De^4 - di^4)/64 (twice the OpenTURNS values, '
                      'which use /32)\n' + ','.join(tm.QOI_NAMES))
    cols = [f'd{q}/d{n}' for q in tm.QOI_NAMES for n in tm.NAMES]
    np.savetxt(os.path.join(out, 'gradients.csv'), D, fmt=FMT, delimiter=',',
               header='exact gradients with respect to the PHYSICAL inputs: columns 0-5 d deflection/d(F, L, a, De, di, E), '
                      '6-11 d theta_left/d(...), 12-17 d theta_right/d(...)\n' + ','.join(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory (default: this directory)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    t0 = time.perf_counter()
    X, F, D = compute()
    os.makedirs(args.out, exist_ok=True)
    write(args.out, X, F, D)
    t1 = time.perf_counter()
    for f, A in zip(FILES, (X, F, D)):
        if not np.array_equal(np.loadtxt(os.path.join(args.out, f), delimiter=','), A):
            raise RuntimeError(f'{f} did not round-trip exactly')
    print(f'wrote {args.out} in {t1 - t0:.3f} s')
    print('gradients vs central differences (relative step 1e-6), max relative error (row 2-norm), '
          'deflection / theta_left / theta_right:', np.array2string(fd_check(X, D), precision=2))
    print('closed forms vs unit-load quadrature, first 20 samples, max relative difference:',
          np.array2string(quad_check(X, F), precision=2))
    if args.check:
        same = [np.array_equal(np.loadtxt(os.path.join(args.out, f), delimiter=','),
                               np.loadtxt(os.path.join(HERE, f), delimiter=',')) for f in FILES]
        print('identical to the shipped files (inputs, outputs, gradients):', same)


if __name__ == '__main__':
    main()
