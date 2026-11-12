"""Recompute the accuracy numbers quoted in DATA.md for H2_O2_Ignition_Delay_Kinetics (needs Cantera).

    python check_accuracy.py            # all checks, about 20 s on one core
    python check_accuracy.py --quick    # the nominal point and the ignition criterion only

Prints, beside each value DATA.md quotes:

1. tau at the nominal point (P0 = 1 atm, T0 = 900 K, phi = 2, Li et al. 2004 rate constants).
2. The ignition criterion: tau for temperature rises of 200, 800 and 1200 K against the 400 K
   used for the data, at the nominal point, at a slow sample (19-input row 440, tau = 0.48 s) and
   at a fast one (19-input row 737, tau = 0.150 ms).
3. tau re-solved at rtol 1e-12, atol 1e-22 at six shipped rows (19-input rows 780, 787, 870 and
   36-input rows 782, 414, 134; one fast-branch row in each case), against the stored tau.
4. Gradients at the same rows: central differences at rtol 1e-12 with the shipped step (1e-3 of
   each half-range) and with a step ten times smaller, against the stored gradients (relative
   2-norm error of the gradient in normalized coordinates).
5. Cantera's forward sensitivities (CVODES) at the three 19-input rows:
   dtau/dlnA_i = -T(tau) S_i(tau) / (dT/dt)(tau), with S_i = dlnT/dlnk_i, against the stored
   gradients.

The rows are fixed, and the computation is deterministic. Needs numpy, scipy and Cantera
(the shipped data were written with Cantera 3.2.0); without Cantera the script stops with an
error. Run it from any directory; it reads the shipped files next to it and writes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2o2_ignition_driver as hd  # noqa: E402

ROWS = {'19': [780, 787, 870], '36': [782, 414, 134]}
CRITERION_ROWS = {'slow': 440, 'fast': 737}          # 19-input rows


def load(case):
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=',', ndmin=2)  # noqa: E731
    return ld(f'inputs_{case}.csv'), ld(f'outputs_{case}.csv')[:, 0], ld(f'gradients_{case}.csv')


def normalized(G, case):
    lo, up = (hd.LOWER19, hd.UPPER19) if case == '19' else (hd.LOWER36, hd.UPPER36)
    return G * (up - lo) / 2


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def sensitivity_gradient(model, lnA, rtol=1e-9, atol=1e-18, srtol=1e-7, satol=1e-9):
    """dtau/dlnA_i from Cantera's forward sensitivities of T at the stored tau (19 inputs)."""
    ct = hd.ct
    tau = model.ignition_delay(lnA=lnA)
    model._set_lnA(lnA)
    model._set_Ea(None)
    gas = model.gas
    gas.TPX = hd.NOMINAL_IC['T0'], hd.NOMINAL_IC['P0'] * ct.one_atm, {'H2': 2.0 * hd.NOMINAL_IC['phi'], 'O2': 1.0}
    r = ct.IdealGasConstPressureReactor(gas, clone=False)
    net = ct.ReactorNet([r])
    for j in range(gas.n_reactions):
        r.add_sensitivity_reaction(j)
    net.rtol, net.atol = rtol, atol
    net.rtol_sensitivity, net.atol_sensitivity = srtol, satol
    net.max_steps = 200000
    net.advance(tau)
    g = r.phase
    tdot = -np.dot(g.partial_molar_enthalpies, g.net_production_rates) / (g.density * g.cp_mass)
    S = np.array([sum(net.sensitivity('temperature', j) for j in idx) for idx in hd.CANTERA_INDEX])
    return -r.T * S / tdot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='only checks 1 and 2')
    a = ap.parse_args()
    if hd.ct is None:
        print('Cantera is not installed. Install it (pip install cantera==3.2.0, the version that wrote '
              'the shipped files) and rerun check_accuracy.py.', file=sys.stderr)
        sys.exit(3)
    t0 = time.perf_counter()
    print(f'Cantera {hd.ct.__version__}, numpy {np.__version__}')
    model = hd.IgnitionModel()                        # rtol 1e-9, atol 1e-18: the settings of the data
    tight = hd.IgnitionModel(rtol=1e-12, atol=1e-22)
    X19, f19, G19 = load('19')

    tau0 = model.ignition_delay()
    print(f'\n1. tau at the nominal point: {tau0 * 1e3:.4f} ms (DATA.md: 86.13 ms)')

    print('\n2. ignition criterion: relative change of tau for a rise of 200, 800, 1200 K instead of 400 K')
    for label, lnA in [('nominal', None), ('slow, 19-input row %d' % CRITERION_ROWS['slow'], X19[CRITERION_ROWS['slow']]),
                       ('fast, 19-input row %d' % CRITERION_ROWS['fast'], X19[CRITERION_ROWS['fast']])]:
        base = model.ignition_delay(lnA=lnA)
        d = [model.ignition_delay(lnA=lnA, t_rise=dt) / base - 1 for dt in (200.0, 800.0, 1200.0)]
        print(f'   {label} (tau = {base:.4g} s): ' + ', '.join(f'{v:+.1e}' for v in d))
    if a.quick:
        print(f'\nrun time {time.perf_counter() - t0:.0f} s')
        return

    print('\n3. tau re-solved at rtol 1e-12, atol 1e-22 against the stored tau (relative difference)')
    data = {'19': (X19, f19, G19), '36': load('36')}
    for case, rows in ROWS.items():
        X, f, _ = data[case]
        fn = tight.tau19 if case == '19' else tight.tau36
        print(f'   {case} inputs: ' + ', '.join(f'row {i} (tau {f[i]:.3g} s) {abs(fn(X[i]) / f[i] - 1):.1e}' for i in rows))

    print('\n4. gradients at rtol 1e-12 against the stored ones (relative 2-norm error, normalized coordinates)')
    for case, rows in ROWS.items():
        X, f, G = data[case]
        fn = tight.tau19 if case == '19' else tight.tau36
        steps = hd.STEPS19 if case == '19' else hd.STEPS36
        for i in rows:
            g1 = tight.grad(fn, X[i], steps)
            g2 = tight.grad(fn, X[i], steps / 10)
            ref = normalized(G[i], case)
            print(f'   {case} inputs, row {i}: same step {rel(normalized(g1, case), ref):.1e}, '
                  f'step / 10 {rel(normalized(g2, case), ref):.1e}')

    print('\n5. CVODES forward sensitivities against the stored gradients (19 inputs; relative 2-norm error)')
    for i in ROWS['19']:
        gs = sensitivity_gradient(model, X19[i])
        print(f'   row {i} (tau {f19[i]:.3g} s): {rel(normalized(gs, "19"), normalized(G19[i], "19")):.1e}')
    print(f'\nrun time {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
