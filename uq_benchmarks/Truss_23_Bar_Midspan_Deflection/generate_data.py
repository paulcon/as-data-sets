"""Write the four CSVs of the 23-bar truss data set.

    python generate_data.py --out DIR --check       # writes to DIR (outside this directory) and compares
    python generate_data.py --out .                 # rewrites the shipped CSVs (explicit only)

Draws M = 1000 independent Monte Carlo samples xi ~ N(0, I_10) of the standard-normal coordinates
with numpy's default generator seeded with SEED = 2011, maps them to the physical inputs of
Table 1 of Blatman and Sudret (2011) through the inverse CDFs (`truss_model.physical`), and
computes the midspan deflection V1 (m) and its exact (adjoint) gradient with respect to the
physical inputs.

It also checks the gradients against central finite differences in xi (step 1e-5) at all M
samples and prints the largest relative error (row 2-norm).

Only numpy and the standard library are imported. One process.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import truss_model as tm  # noqa: E402

SEED = 2011
M = 1000
H_FD = 1e-5
FILES = ('inputs.csv', 'inputs_standard.csv', 'outputs.csv', 'gradients.csv')
TOL = {'inputs.csv': 1e-13, 'inputs_standard.csv': 1e-15, 'outputs.csv': 1e-12, 'gradients.csv': 1e-11}

PROVENANCE = ('truss_model.py (Blatman and Sudret 2011, Sec. 6.3, Fig. 8 and Table 1; linear bar FE, '
              '23 bars, 13 nodes); Monte Carlo, numpy default_rng(2011); written by generate_data.py')


def draw(seed=SEED, M=M):
    return tm.sample_xi(M, seed)


def fd_check(XI, G_xi):
    fd = np.empty_like(G_xi)
    for j in range(XI.shape[1]):
        e = np.zeros(XI.shape[1])
        e[j] = H_FD
        fd[:, j] = (tm.deflection_xi(XI + e) - tm.deflection_xi(XI - e)) / (2 * H_FD)
    return np.max(np.linalg.norm(fd - G_xi, axis=1) / np.linalg.norm(G_xi, axis=1))


def read_csv(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def write(out, XI, Z, V, G):
    names = ','.join(f'{n} ({u})' for n, u in zip(tm.NAMES, tm.UNITS))
    np.savetxt(os.path.join(out, 'inputs.csv'), Z, fmt='%.17g', delimiter=',',
               header=f'{len(Z)} samples of the physical inputs (lognormal E1, E2, A1, A2; Gumbel P1..P6; '
                      f'Table 1). {PROVENANCE}\n{names}')
    np.savetxt(os.path.join(out, 'inputs_standard.csv'), XI, fmt='%.17g', delimiter=',',
               header='The same samples in standard-normal coordinates xi, Z_i = F_i^-1(Phi(xi_i)) (Eq. 55 inverted). '
                      f'{PROVENANCE}\n' + ','.join(f'xi_{n} (-)' for n in tm.NAMES))
    np.savetxt(os.path.join(out, 'outputs.csv'), V[:, None], fmt='%.17g', delimiter=',',
               header=f'Midspan downward deflection V1. {PROVENANCE}\nV1 (m)')
    np.savetxt(os.path.join(out, 'gradients.csv'), G, fmt='%.17g', delimiter=',',
               header='Exact (adjoint) gradient of V1 with respect to the PHYSICAL inputs. '
                      f'{PROVENANCE}\n' + ','.join(f'dV1/d{n} (m/{u})' for n, u in zip(tm.NAMES, tm.UNITS)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory (required; pass . to rewrite the shipped files)')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory (needs --out)')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if args.check and out == HERE:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(out, exist_ok=True)

    t0 = time.perf_counter()
    XI = draw()
    Z = tm.physical(XI)
    V, G = tm.deflection(Z, grad=True)
    t1 = time.perf_counter()
    write(out, XI, Z, V, G)
    print(f'wrote {len(V)} samples ({t1 - t0:.2f} s for the model runs); '
          f'V1 range [{V.min():.5f}, {V.max():.5f}] m, mean {V.mean():.5f} m')
    print(f'smallest load {Z[:, 4:].min():.1f} N (all > 0: {bool((Z[:, 4:] > 0).all())})')
    err = fd_check(XI, G * tm.dphysical_dxi(XI))
    print(f'gradient check vs central differences in xi (h = {H_FD:g}), all {len(V)} samples: '
          f'max relative error {err:.1e}')
    failed = []
    if args.check:
        for n in FILES:
            a, b = read_csv(os.path.join(out, n)), read_csv(os.path.join(HERE, n))
            rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)) if a.shape == b.shape else np.inf
            print(f'{n}: max relative difference from the shipped file {rel:.1e} '
                  f'(tolerance {TOL[n]:g}) {"OK" if rel <= TOL[n] else "FAIL"}')
            if not rel <= TOL[n]:
                failed.append(n)
    print(f'total {time.perf_counter() - t0:.1f} s')
    if failed:
        sys.exit(f'--check FAILED for {", ".join(failed)}')


if __name__ == '__main__':
    main()
