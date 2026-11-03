"""Check flood_model.exact_moments against Monte Carlo (River_Flood_Dyke_Overflow).

    python check_accuracy.py          (about 1 s)

1. Quadrature vs Monte Carlo. 400 000 fresh samples of the eight inputs,
   flood_model.sample(400000, numpy.random.default_rng(0)), give the variance (np.var) of the
   overflow S and the sample DGSM nu_j = mean of (dS/dx_j)^2 (flood_model.grad_overflow). The
   script prints them beside exact_moments(): Var(S), and the largest relative difference of
   nu_j over the eight inputs. DATA.md: Var(S) 1.1745 (exact) vs 1.1704 (Monte Carlo); nu_j
   within 0.07 %.

The other numbers of the Accuracy section of DATA.md are printed by the notebook (model against
outputs.csv; mean and variance of S on the shipped 10 000 samples and their exact values) and by
generate_data.py (gradients against central differences).

Deterministic; imports numpy, scipy (through flood_model) and the standard library. Writes
nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flood_model as fm  # noqa: E402

M_MC = 400_000
SEED_MC = 0


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    t0 = time.perf_counter()
    ex = fm.exact_moments()
    X = fm.sample(M_MC, np.random.default_rng(SEED_MC))
    S = fm.overflow(X)
    nu = np.mean(fm.grad_overflow(X)**2, axis=0)
    rel = np.abs(nu - ex['nu'])/ex['nu']
    print(f'1. exact_moments vs {M_MC} Monte Carlo samples (seed {SEED_MC}):')
    print(f'   Var(S): exact {ex["var"]:.4f}, Monte Carlo {np.var(S):.4f} '
          f'(DATA.md: 1.1745 vs 1.1704)')
    print('   nu_j relative difference: ' + ', '.join(f'{n} {r:.1e}' for n, r in zip(fm.NAMES, rel)))
    print(f'   largest {100*rel.max():.2f} % (DATA.md: within 0.07 %)')
    print(f'run time {time.perf_counter() - t0:.2f} s')


if __name__ == '__main__':
    main()
