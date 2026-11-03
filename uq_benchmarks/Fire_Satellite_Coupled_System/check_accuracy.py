"""Recompute the accuracy numbers of DATA.md (Fire_Satellite_Coupled_System) that no other script prints.

    python check_accuracy.py                    # checks 1-3; check 2 needs OpenTURNS 1.25
    python check_accuracy.py --skip-openturns   # checks 1 and 3 only

Run it from any directory; it reads the shipped CSVs next to this file and evaluates
fire_satellite_model.py. Each line prints the recomputed value beside the value DATA.md quotes.
In the repo's `.venv` checks 1 and 3 take 0.1 s (about 3 s with start-up); check 2 about 3 s.

Checks (DATA.md, "Accuracy"):

 1. Gradient step study: central differences of the real solve (fixed-point tolerance 1e-15) with
    steps 1e-4, 1e-5 and 1e-6 sd_j, at the first 50 shipped samples, against the stored
    complex-step gradients; the largest relative error (2-norm) of a gradient vector over the
    samples and the three outputs (DATA.md: about 1e-9, 1e-8 and 1e-7).
 2. Against OpenTURNS: `openturns.usecases.fire_satellite.FireSatelliteModel` evaluated at the mean
    of its input distribution, and the means of the three outputs over 5000 draws from that
    distribution (q sd 0.1) after `ot.RandomGenerator.SetSeed(1)` (DATA.md: 0.0116597, 1129.96,
    11.8626 at the mean; means 0.01159, 1129.2, 11.862). OpenTURNS is LGPL and is not
    shipped; it is only imported and run here. Install the version used for DATA.md in a separate
    environment (Python 3.11; 1.25 is the newest release with a pip wheel for macOS x86-64):
        python3.11 -m venv venv-ot
        venv-ot/bin/pip install openturns==1.25 numpy==2.4.6 scipy==1.17.1
    and rerun with venv-ot/bin/python check_accuracy.py. Without OpenTURNS the script runs checks 1
    and 3, then stops with status 1 and this message, unless --skip-openturns is given.
 3. Size of the gravity-gradient torque: tau_g / tau_dist from fire_satellite_model.solve at the
    mean inputs and its largest value over the 2000 shipped samples (DATA.md: 0.265 % and
    2.04 %). It bounds the effect of OpenTURNS's (2 R_E + H)^3 in tau_g, which enters tau_dist
    squared.

The notebook prints the other numbers of the section: the contraction factor, the outputs at the
OpenTURNS tolerance 1e-3, the 20-sample check at step 1e-4 sd, and the outputs of this model with
OpenTURNS's omega_max = 6000 rad/s (its mu, 3.986e14, is Table 1's).

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


def check_tau_g(X):
    """Check 3: tau_g as a share of tau_dist, at the mean inputs and over the shipped samples."""
    _, at_mean = fm.solve(fm.MEAN, full=True)
    _, at_all = fm.solve(X, full=True)
    r_mean = (at_mean['tau_g'] / at_mean['tau_dist'])[0]
    r_all = at_all['tau_g'] / at_all['tau_dist']
    print(f'3. tau_g / tau_dist: {100 * r_mean:.3f} % at the mean inputs, at most {100 * r_all.max():.3f} % '
          f'over the {len(X)} samples (DATA.md: 0.265 % and 2.04 %)')


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
    check_tau_g(X)
    if not args.skip_openturns and not check_openturns():
        print('2. OpenTURNS: NOT RUN. openturns is not importable. Install OpenTURNS 1.25 in a separate '
              'environment:\n       python3.11 -m venv venv-ot\n'
              '       venv-ot/bin/pip install openturns==1.25 numpy==2.4.6 scipy==1.17.1\n'
              '   and rerun with venv-ot/bin/python check_accuracy.py (or pass --skip-openturns to run '
              'checks 1 and 3 only).')
        sys.exit(1)
    print(f'done in {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
