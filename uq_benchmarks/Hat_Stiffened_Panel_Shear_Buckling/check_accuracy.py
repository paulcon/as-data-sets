"""Check panel_model.py against an independent transcription of the OpenTURNS formula (Hat_Stiffened_Panel_Shear_Buckling).

    python check_accuracy.py          (well under 1 s)

The OpenTURNS use case (python/src/usecases/stiffened_panel.py, tag v1.25.1) defines the model as
one symbolic formula,

    (5.35 + 4 (b_0/a)^2) (pi^2/b_0^2 E t^3 / (12 (1 - nu^2))) (1 + (2 p (h - 2 h_0) - h_c (f_1 - f_2)) / (4 h_0 ell))

with the substitutions h_0 = (A (h_c + 2 t) + t^2 (f_1 - f_2)) / (2 A_bar), A_bar = A + t (p + (f_1 - f_2)/2),
A = ell t and h = h_c + t. This script writes that formula out term by term, in a form deliberately
different from panel_model.py (no shared intermediates), evaluates it on the shipped inputs and on
the same draws with the OpenTURNS-as-coded thickness (t / 10), and prints the largest relative
differences from panel_model.buckling_load and from outputs.csv. Nothing from OpenTURNS is imported.

The other numbers of the Accuracy section of DATA.md are printed by the notebook (values against
outputs.csv, sample Spearman correlation) and by generate_data.py (gradients against central
differences and a complex step).

Deterministic; imports numpy and the standard library.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import panel_model as pm  # noqa: E402


def openturns_formula(X):
    """The OpenTURNS formula with every substitution written out (columns E, nu, h_c, ell, f_1, f_2, t, a, b_0, p)."""
    E, nu, h_c, ell, f_1, f_2, t, a, b_0, p = np.asarray(X, dtype=float).T
    A = ell * t
    A_bar = A + t * (p + (f_1 - f_2) / 2)
    h_0 = (A * (h_c + 2 * t) + t ** 2 * (f_1 - f_2)) / (2 * A_bar)
    h = h_c + t
    return ((5.35 + 4 * (b_0 / a) ** 2) * (np.pi ** 2 / b_0 ** 2 * E * t ** 3 / (12 * (1 - nu ** 2)))
            * (1 + (2 * p * (h - 2 * h_0) - h_c * (f_1 - f_2)) / (4 * h_0 * ell)))


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',')
    f = np.loadtxt(os.path.join(HERE, 'outputs.csv'), delimiter=',')
    rel = lambda a, b: np.max(np.abs(a - b) / np.abs(b))  # noqa: E731
    ot = openturns_formula(X)
    print(f'OpenTURNS formula vs panel_model.buckling_load, {len(X)} shipped inputs: max relative difference '
          f'{rel(ot, pm.buckling_load(X)):.1e}')
    print(f'OpenTURNS formula vs outputs.csv: max relative difference {rel(ot, f):.1e}')
    Xo = X.copy()
    Xo[:, 6] /= 10.0
    print(f'same draws with the OpenTURNS-as-coded thickness (t / 10): max relative difference '
          f'{rel(openturns_formula(Xo), pm.buckling_load(Xo)):.1e}')
    print(f'at the midpoint (NOMINAL): {openturns_formula(pm.NOMINAL[None])[0]:.6g} N/m '
          f'(panel_model {pm.buckling_load(pm.NOMINAL[None])[0]:.6g} N/m); with the OpenTURNS thickness '
          f'{openturns_formula(pm.NOMINAL_OPENTURNS[None])[0]:.6g} N/m')


if __name__ == '__main__':
    main()
