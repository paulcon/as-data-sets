"""Recompute the accuracy numbers of DATA.md (Fire_Satellite_Coupled_System) that no other script prints.

    python check_accuracy.py                    # checks 1 and 2; check 2 needs OpenTURNS
    python check_accuracy.py --skip-openturns   # check 1 only

Run it from any directory; it reads the shipped CSVs next to this file and evaluates
fire_satellite_model.py. Each line prints the recomputed value beside the value DATA.md quotes.
In the repo's `.venv` check 1 takes 0.1 s (about 3 s with start-up); check 2 about 3 s.

Checks (DATA.md, "Accuracy"):

 1. Gradient step study: central differences of the real solve (fixed-point tolerance 1e-15) with
    steps 1e-4, 1e-5 and 1e-6 sd_j, at the first 50 shipped samples, against the stored
    complex-step gradients; the largest relative error (2-norm) of a gradient vector over the
    samples and the three outputs (DATA.md: about 1e-9, 1e-8 and 1e-7).
 2. Against OpenTURNS: `openturns.usecases.fire_satellite.FireSatelliteModel` evaluated at the mean
    of its input distribution, and the means of the three outputs over 5000 draws from that
    distribution (q sd 0.1) after `ot.RandomGenerator.SetSeed(1)` (DATA.md: 0.0116597, 1129.96,
    11.8626 at the mean; means 0.01159, 1129.2, 11.862). OpenTURNS is LGPL and is not
    shipped; it is only imported and run here. Install it in a separate environment with
        pip install openturns
    (DATA.md names the version used). Without it the script runs check 1, then stops with status 1
    and this message, unless --skip-openturns is given.

The notebook prints the other numbers of the section: the contraction factor, the outputs at the
OpenTURNS tolerance 1e-3, the 20-sample check at step 1e-4 sd, and the outputs of this model with
the OpenTURNS conventions (omega_max = 6000 rad/s, mu = 3.986004418e14).

Deterministic (fixed rows and seed). Imports numpy, scipy (through fire_satellite_model) and the
standard library, and openturns for check 2. Writes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fire_satellite_model as fm  # noqa: E402

N_ROWS = 50
STEPS = (1e-4, 1e-5, 1e-6)
OT_SEED = 1
OT_N = 5000


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def check_steps(X, G):
    Xc, Gc = X[:N_ROWS], G[:N_ROWS]
    quoted = ('of order 1e-9', 'of order 1e-8', 'of order 1e-7')
    for h, q in zip(STEPS, quoted):
        Gfd = np.empty_like(Gc)
        for j in range(9):
            dx = np.zeros(9)
            dx[j] = h * fm.SD[j]
            Gfd[:, :, j] = (fm.qoi(Xc + dx, tol=1e-15) - fm.qoi(Xc - dx, tol=1e-15)) / (2 * dx[j])
        rel = np.linalg.norm(Gfd - Gc, axis=2) / np.linalg.norm(Gc, axis=2)
        print(f'1. step {h:.0e} sd, rows 0-{N_ROWS - 1}: max relative error {rel.max():.1e} (DATA.md {q}); '
              f'per output (tau_tot, P_tot, A_sa) ' + ', '.join(f'{r:.1e}' for r in rel.max(0)))


def check_openturns():
    try:
        import openturns as ot
        from openturns.usecases.fire_satellite import FireSatelliteModel
    except ImportError:
        return False
    m = FireSatelliteModel()
    dist = m.distribution if hasattr(m, 'distribution') else m.inputDistribution   # renamed in 1.27
    print(f'2. OpenTURNS {ot.__version__}')
    y = np.array(m.model(dist.getMean()))
    print(f'   at the mean inputs: tau_tot = {y[0]:.6g}, P_tot = {y[1]:.6g}, A_sa = {y[2]:.6g} '
          f'(DATA.md 0.0116597, 1129.96, 11.8626)')
    ot.RandomGenerator.SetSeed(OT_SEED)
    Y = np.array(m.model(dist.getSample(OT_N)))
    mean = Y.mean(axis=0)
    print(f'   means over {OT_N} draws (seed {OT_SEED}): {mean[0]:.4g}, {mean[1]:.5g}, {mean[2]:.5g} '
          f'(DATA.md 0.01159, 1129.2, 11.862)')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('--skip-openturns', action='store_true', help='run check 1 only (no OpenTURNS needed)')
    args = ap.parse_args()
    t0 = time.perf_counter()
    X = load('inputs.csv')
    G = load('gradients.csv').reshape(-1, 3, 9)
    check_steps(X, G)
    if not args.skip_openturns and not check_openturns():
        print('2. OpenTURNS: NOT RUN. openturns is not importable. Install it in a separate environment '
              'with\n       pip install openturns\n   and rerun this script there (or pass --skip-openturns '
              'to run check 1 only).')
        sys.exit(1)
    print(f'done in {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
