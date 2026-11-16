"""Accuracy checks for Irrigation_Water_Withdrawal: the model against a second transcription, and the Sobol' estimators.

    python check_accuracy.py          (about 1 s)

Check 1. An independent check of iww_model.py: the archived code's one-line expression for y (function
`full_model` in `code_ghm_auditing.R`, Zenodo doi:10.5281/zenodo.5528844 v1.0.4) transcribed into Python in
the order the code evaluates it,

    ET_0 = ((0.408 Delta A + gamma (900 / (T_a + 273)) w v) / Delta + gamma (1 + 0.34 w)) / 31
    y    = 10 (k_c ET_0 - P) I_a / (E_a E_c M_f),

sharing no code with iww_model.py, and compared with iww_model.iww and with outputs.csv on the shipped inputs
and at the midpoint of Table S4.

Check 2. sobol_tools.py (the Saltelli first-order and Jansen total-order estimators of the Supplementary
Materials, Eqs. 14-15, and the second-order estimator) applied to the Sobol' G function in 12
dimensions with N = 2^15 base rows, as in the notebook, against the exact indices. The coefficients a are
this script's choice, picked so that the indices span the sizes found for the irrigation model.

Check 3. The weather inputs of Table S4 against the supplement's own formulas: Eqs. 17-22 (pp. 18-19) evaluated
at the midpoint of Table S4's T_a (10 degC), with h_rel = 0.77 (p. 19) and z = 9 m (p. 19), compared with the
midpoints of Table S4's Delta, v and gamma.

Check 4. The input bounds as the archived code builds them (`code_ghm_auditing.R`, chunks `climate_uvalde`,
`kc_wheat` and `sample_matrix_full`), rebuilt from archived values typed in below (the archive's files are not
shipped), and y at their midpoint, against y at the Table S4 midpoint and the paper's 8600 m^3 (Comment p. 2;
Supplementary Materials p. 19). The constants:

- `uvalde_climate_data_january.csv`, days 6 and 7: Temp.Avg 52.1 and 51.5 degF, Hum.Avg 83.9 and 53.4 %. The code
  converts degF to degC, averages the two days, and applies Eqs. 17-22 with z = 9.1 m (the text says 9 m).
- `kc_wheat_new.csv`, the 8 points with 50 < x < 53 and y > 0: smallest 0.456431535, largest 1.144221956.
- `bos.dt.csv`, Type == "Sprinkler": E_a 0.49 to 0.88, E_c 0.64 to 0.96.
- typed in the code: A 350 +/- 15 %, w 2.81 +/- 5 %, M_f 0.5 to 0.97, P 0 to 0.1; Delta +/- 0.5 %, gamma +/- 0.1 %,
  T_a +/- 1 %, v +/- 4 % around the computed values.
- I_a: the code takes the smallest and largest area of map rasters that are not in the archive; the text's
  42.9 to 144.5 ha (p. 17) is used here.

The other accuracy numbers in DATA.md are printed by generate_data.py (gradients against central differences)
and by the notebook (model against outputs.csv). Deterministic; imports numpy, scipy and the standard library.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import iww_model as im  # noqa: E402
import sobol_tools as st  # noqa: E402


def archived_expression(X):
    D, g, A, Ta, w, v, kc, Ia, Ea, Ec, Mf, P = np.asarray(X, dtype=float).T
    ET0 = (((0.408 * D * A + g * (900 / (Ta + 273)) * w * v) / D + g * (1 + 0.34 * w))) / 31
    ETc = ET0 * kc
    Ep = Ea * Ec * Mf
    return ((ETc - P) * Ia) / Ep * 10


TEMP_AVG_F = (52.1, 51.5)        # uvalde_climate_data_january.csv, Temp.Avg on January 6 and 7 (degF)
HUM_AVG = (83.9, 53.4)           # the same file, Hum.Avg on January 6 and 7 (%)
REL_HUM = np.mean(HUM_AVG) / 100
KC_RANGE = (0.456431535, 1.144221956)   # kc_wheat_new.csv, 8 points with 50 < x < 53
EA_RANGE, EC_RANGE = (0.49, 0.88), (0.64, 0.96)   # bos.dt.csv, sprinkler rows
IA_RANGE = (42.9, 144.5)          # Supplementary Materials p. 17 (the code's rasters are not archived)


def code_bounds():
    """Lower and upper bounds of the 12 inputs (order of iww_model.NAMES) as the archived code builds them."""
    T = np.mean([(t - 32) / 1.8 for t in TEMP_AVG_F])
    e_sat = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
    Delta = (4098 * e_sat) / ((T + 237.3) ** 2)
    v = e_sat - e_sat * REL_HUM
    lam = 2.501 - 0.002361 * T
    p = 101.3 * ((293 - 0.0065 * 9.1) / 293) ** 5.256
    gamma = 0.0016286 * p / lam
    centre_rel = [(Delta, 0.005), (gamma, 0.001), (350.0, 0.15), (T, 0.01), (2.81, 0.05), (v, 0.04)]
    lo = [c * (1 - r) for c, r in centre_rel] + [KC_RANGE[0], IA_RANGE[0], EA_RANGE[0], EC_RANGE[0], 0.5, 0.0]
    hi = [c * (1 + r) for c, r in centre_rel] + [KC_RANGE[1], IA_RANGE[1], EA_RANGE[1], EC_RANGE[1], 0.97, 0.1]
    return np.array(lo), np.array(hi)


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',')
    f = np.loadtxt(os.path.join(HERE, 'outputs.csv'), delimiter=',')
    ref = archived_expression(X)
    rel = lambda a, b: np.max(np.abs(a - b) / np.abs(b))  # noqa: E731
    print(f'check 1: second transcription vs iww_model.iww, {len(X)} shipped inputs: max relative difference '
          f'{rel(im.iww(X), ref):.1e}; vs outputs.csv {rel(f, ref):.1e}')
    print(f'check 1: at the midpoint of Table S4, y = {archived_expression(im.NOMINAL[None])[0]:.4f} m^3 '
          f'(iww_model {im.iww(im.NOMINAL)[0]:.4f} m^3)')

    a = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 4.0, 9.0, 20.0, 50.0, 99.0, 99.0, 99.0])
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (2, 4), (3, 5), (0, 11)]
    A, B = st.base_matrices(2 ** 15, len(a), seed=2022)
    y = st.evaluate(lambda U: st.g_function(U, a), A, B, pairs)
    S, T, S2 = st.indices(y, pairs)
    S0, T0, S20 = st.g_function_indices(a, pairs)
    print(f'check 2: G function, k = 12, N = 2^15: exact S_1 = {S0[0]:.4f}, T_1 = {T0[0]:.4f}, '
          f'largest exact S_ij = {S20.max():.4f}')
    print(f'check 2: max |error|: first order {np.max(np.abs(S - S0)):.1e}, total order {np.max(np.abs(T - T0)):.1e}, '
          f'second order {np.max(np.abs(S2 - S20)):.1e}')

    Ta = im.NOMINAL[3]
    e_sat = 0.6108 * np.exp(17.27 * Ta / (Ta + 237.3))            # Eq. 18
    Delta = 4098 * e_sat / (Ta + 237.3) ** 2                      # Eq. 17
    v = e_sat - e_sat * 0.77                                      # Eq. 19, h_rel = 0.77
    lam = 2.501 - 0.002361 * Ta                                   # Eq. 21
    p = 101.3 * ((293 - 0.0065 * 9) / 293) ** 5.256              # Eq. 22, z = 9 m
    gamma = 0.0016286 * p / lam                                   # Eq. 20
    for n, val, j in [('Delta', Delta, 0), ('v', v, 5), ('gamma', gamma, 1)]:
        print(f'check 3: Eqs. 17-22 at T_a = {Ta:g} degC: {n} = {val:.4f}; Table S4 midpoint {im.NOMINAL[j]:.4f} '
              f'(range {im.LOWER[j]:g}-{im.UPPER[j]:g}); relative difference {(val - im.NOMINAL[j]) / im.NOMINAL[j]:+.3f}')

    lo, hi = code_bounds()
    mid = 0.5 * (lo + hi)
    y_code, y_s4 = im.iww(mid)[0], im.iww(im.NOMINAL)[0]
    print(f'check 4: code inputs: T_a = {mid[3]:.4f} degC, rel_hum = {REL_HUM:.4f} (p. 19: 0.77), Delta = {mid[0]:.5f}, '
          f'v = {mid[5]:.4f} kPa, gamma = {mid[1]:.5f} kPa/degC')
    for n, j in [('T_a', 3), ('v', 5), ('gamma', 1), ('Delta', 0)]:
        print(f'check 4: {n}: Table S4 midpoint {im.NOMINAL[j]:.5g} (range {im.LOWER[j]:g}-{im.UPPER[j]:g}, '
              f'half-range {100 * (im.UPPER[j] - im.LOWER[j]) / (im.UPPER[j] + im.LOWER[j]):.2f} %); code '
              f'{mid[j]:.5g} (range {lo[j]:.5g}-{hi[j]:.5g}, half-range {100 * (hi[j] - lo[j]) / (hi[j] + lo[j]):.2f} %); '
              f'Table S4 - code {(im.NOMINAL[j] - mid[j]) / mid[j]:+.3f} relative')
    print(f'check 4: k_c: code range {lo[6]:.5f}-{hi[6]:.5f}, Table S4 {im.LOWER[6]:g}-{im.UPPER[6]:g}')
    print(f'check 4: y at the code bounds midpoint {y_code:.1f} m^3 ({100 * (y_code - 8600) / 8600:+.1f} % from 8600); '
          f'at the Table S4 midpoint {y_s4:.1f} m^3 ({100 * (y_s4 - 8600) / 8600:+.1f} %)')
    for label, cols in [('the k_c bounds', [6]), ('the weather bounds (Delta, gamma, T_a, w, v)', [0, 1, 3, 4, 5]),
                        ('the A bounds', [2])]:
        x = im.NOMINAL.copy()
        x[cols] = mid[cols]
        print(f'check 4: Table S4 midpoint with only {label} from the code: y = {im.iww(x)[0]:.1f} m^3 '
              f'({100 * (im.iww(x)[0] - y_s4) / y_s4:+.2f} %)')


if __name__ == '__main__':
    main()
