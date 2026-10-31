"""Regenerate the data files of Point_Heat_Source_Consolidation.

    python generate_data.py          (about 1 s)

Draws M = 1000 independent samples of the eight inputs by the inverse-CDF transform of
numpy.random.default_rng(1985).random((M, 8)) and evaluates heat_source_model.py:

    inputs.csv     (M, 8)    physical inputs K, C, rho, E, a_s, k_s, nu, phi
    outputs.csv    (M, 3)    dT (K), p (Pa), u_r (m) at r = 6 m, t = 365 d
    gradients.csv  (M, 24)   exact gradients of the three outputs w.r.t. the physical inputs
    fields.csv     (M, 300)  dT, p, u_r on the 10 x 10 (r, t) grid; column 100*field + 10*i_r + i_t
"""

import time

import numpy as np

import heat_source_model as hm

SEED = 1985
M = 1000
SRC = ('heat_source_model.py (Booker & Savvidou 1985, Eq. 23; inputs and grid of UQTestFuns '
       'SaturatedPointHeatSource / Chaudhry et al. 2021); inverse-CDF draws from numpy '
       f'default_rng({SEED}).random((M, 8)); written by generate_data.py')


def main():
    t0 = time.perf_counter()
    X = hm.sample(M, SEED)
    F, G = hm.headline(X)
    (dT, p, u), _ = hm.fields_and_grads(X)
    fields = np.stack([dT, p, u], axis=1).reshape(M, -1)
    units = [f'[{u_}]' for u_ in hm.UNITS]
    fmt = '%.17g'
    np.savetxt('inputs.csv', X, delimiter=',', fmt=fmt,
               header=f'{M} samples of the 8 physical inputs, independent. {SRC}\n'
                      + ','.join(f'{n} {u_}' for n, u_ in zip(hm.NAMES, units)))
    np.savetxt('outputs.csv', F, delimiter=',', fmt=fmt,
               header=f'temperature change dT, excess pore pressure p and radial displacement u_r at '
                      f'r = {hm.R_STAR:g} m, t = {hm.T_STAR / hm.DAY:g} d. {SRC}\n'
                      'dT [K],p [Pa],u_r [m]')
    qunits = {'dT': 'K', 'p': 'Pa', 'u_r': 'm'}
    gcols = [f'd{q}/d{n} [{qunits[q]} per {u_.strip("[]")}]' for q in qunits for n, u_ in zip(hm.NAMES, units)]
    np.savetxt('gradients.csv', G.reshape(M, -1), delimiter=',', fmt=fmt,
               header=f'exact gradients with respect to the PHYSICAL inputs at r = {hm.R_STAR:g} m, '
                      f't = {hm.T_STAR / hm.DAY:g} d: columns 0-7 ddT/dx, 8-15 dp/dx, 16-23 du_r/dx. {SRC}\n'
                      + ','.join(gcols))
    fcols = [f'{q}(r={r:g} m;t={tl}) [{qunits[q]}]' for q in qunits for r in hm.R_GRID for tl in hm.T_LABELS]
    np.savetxt('fields.csv', fields, delimiter=',', fmt=fmt,
               header=f'dT (K), p (Pa), u_r (m) on the 10 x 10 (r, t) grid; column = 100*field + 10*i_r + i_t '
                      f'(t fastest). {SRC}\n' + ','.join(fcols))
    print(f'wrote inputs.csv, outputs.csv, gradients.csv, fields.csv ({M} samples) in '
          f'{time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
