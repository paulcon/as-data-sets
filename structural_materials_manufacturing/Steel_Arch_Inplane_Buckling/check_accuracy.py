"""Check the shipped gradients against a complex step (Steel_Arch_Inplane_Buckling).

    python check_accuracy.py          (well under 1 s)

1. Gradients vs complex step. The formulas of Eqs. (1)-(3) and (6), with the I-section of
   DATA.md (choice 3), are written out again below in complex arithmetic, without the module's
   intermediates. Each input is perturbed by i h_j, with h_j = 1e-20 times its range, and
   Im N_acr(x + i h_j e_j) / h_j is the derivative. The branch of Eq. (2) or (3) is chosen by
   the real part of lambda. The script prints the largest relative error of the gradient vector
   (row 2-norm) over the 1000 shipped inputs against gradients_fixed.csv and
   gradients_pinned.csv. DATA.md: of order 1e-15 for both cases.

The other numbers of the Accuracy section of DATA.md are printed by the notebook (gradients
against central differences, model against outputs.csv, the branch values at lambda = 18.6
and 9.38) and by generate_data.py --check (rerun against the shipped files).

Deterministic; imports numpy and the standard library. Reads the files next to it and writes
nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arch_buckling_model as am  # noqa: E402

STEP = 1e-20   # complex step, as a fraction of each input's range


def load_complex(Z, case):
    """N_acr in kN for complex inputs Z with columns (S, Phi, E, D, B, tw, tf), E in MPa."""
    S, Phi, E, D, B, tw, tf = Z.T
    web = D - 2*tf
    area = 2*B*tf + web*tw
    inertia = (B*D**3 - (B - tw)*web**3)/12
    lam = S*Phi/4/np.sqrt(inertia/area)                      # Eq. (1)
    if case == 'fixed':                                      # Eq. (3), k = 0.35
        k, switch = 0.35, 18.6
        low = 0.36 + 0.0011*lam**2
        high = 0.6 + 0.4*np.sqrt(1 - 3.109*np.pi**4/lam**2)
    else:                                                    # Eq. (2), k = 0.5
        k, switch = 0.5, 9.38
        low = 0.15 + 0.006*lam**2
        high = 0.26 + 0.74*np.sqrt(1 - 0.63*np.pi**4/lam**2)
    n_cr = np.pi**2*(E*1e6)*inertia/(k*S)**2/1e3              # Eq. (6), N -> kN
    return np.where(lam.real < switch, low, high)*n_cr


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    t0 = time.perf_counter()
    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',').astype(complex)
    h = STEP*(am.UPPER - am.LOWER)
    print(f'1. gradients vs complex step (step {STEP:g} of each range), {len(X)} shipped inputs:')
    for case in ('fixed', 'pinned'):
        G = np.loadtxt(os.path.join(HERE, f'gradients_{case}.csv'), delimiter=',')
        G_cs = np.column_stack([load_complex(X + 1j*h[j]*e, case).imag/h[j]
                                for j, e in enumerate(np.eye(7))])
        err = np.linalg.norm(G_cs - G, axis=1)/np.linalg.norm(G, axis=1)
        print(f'   {case:6s}: max relative error of the gradient vector {err.max():.1e} '
              f'(DATA.md: of order 1e-15)')
    print(f'run time {time.perf_counter() - t0:.2f} s')


if __name__ == '__main__':
    main()
