"""Recompute the accuracy numbers of DATA.md that the notebook does not print (Golinski_Speed_Reducer).

    python check_accuracy.py          (well under 1 s)

1. g3 and g5 against psdr. The two constraints as printed in psdr (J. M. Hokanson,
   psdr/demos/golinski.py, functions golinski_constraint3 and golinski_constraint5, commit
   c61374b of github.com/jeffrey-hokanson/PSDR, read on GitHub; psdr is not installed or
   imported) are transcribed below with psdr's variable names and order of operations:

       x3 = 17
       golinski_constraint3 = 1.93/(x2*x3*x6**4)*x4**3 - 1.
       golinski_constraint5 = np.sqrt((745*x4/x2/x3)**2 + 16.9e6)/(110.0*x6**3) - 1.

   with (x1, x2, x4, x5, x6, x7) = (b, m_t, l1, l2, d1, d2). The script prints the largest
   absolute difference from golinski_model.g3, g5 and from outputs.csv over the 1000 shipped
   inputs. DATA.md: 4.4e-16.
2. Other readings of the volume at Golinski's two Table 1 designs (crude Monte Carlo: b = 4.4,
   m_t = 0.6, l1 = 7.3, l2 = 8.1, d1 = 3.4, d2 = 5.0, z = 17; stray process: b = 3.6, m_t = 0.7,
   l1 = 6.6, l2 = 8.2, d1 = 2.8, d2 = 5.2, z = 18). The volume is summed term by term from the
   geometry of Fig. 11 (rim and web as in golinski_model) with the hub and shaft terms replaced:
   (a) hub (d_p^2 - d_k^2) 2b, as printed, with (pi/4) d_k^2 l_k shafts;
   (b) the same hub with (pi/4) d_k l_k shafts;
   (c) hub (pi/4)(d_p^2 - d_k^2) b, with (pi/4) d_k^2 l_k shafts;
   (d) hub (pi/4)(d_p^2 - d_k^2)(2 d_k - b/3), with (pi/4) d_k^2 l_k shafts.
   DATA.md: 2974.8 and 3029.8; 2800.8 and 2863.0; 2044.8 and 2303.8; 2471.5 and 2893.4 cm^3.
   The script first checks that the same sum with the reading used here, hub (pi/4)(d_p^2 - d_k^2)
   2 d_k, equals golinski_model.volume.

The other numbers of the Accuracy section are printed by the notebook (gradients against
central differences; expanded against geometric volume), and so is the volume of the reading
used here at the two designs (2671.92 and 3049.87 cm^3).

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
import golinski_model as gm  # noqa: E402

# Golinski (1970), Table 1, p. 300: inputs (b, m_t, l1, l2, d1, d2), z
DESIGNS = {'crude Monte Carlo': (np.array([4.4, 0.6, 7.3, 8.1, 3.4, 5.0]), 17),
           'stray process': (np.array([3.6, 0.7, 6.6, 8.2, 2.8, 5.2]), 18)}

HUBS = {'used':   lambda dp, d, b: np.pi/4*(dp**2 - d**2)*2*d,
        'as printed': lambda dp, d, b: (dp**2 - d**2)*2*b,
        'b':      lambda dp, d, b: np.pi/4*(dp**2 - d**2)*b,
        '2d-b/3': lambda dp, d, b: np.pi/4*(dp**2 - d**2)*(2*d - b/3)}
SHAFTS = {'d^2 l': lambda d, l: np.pi/4*d**2*l,
          'd l':   lambda d, l: np.pi/4*d*l}
READINGS = [('a', 'hub (d_p^2 - d_k^2) 2b, d_k^2 l_k shafts', 'as printed', 'd^2 l', (2974.8, 3029.8)),
            ('b', 'hub (d_p^2 - d_k^2) 2b, d_k l_k shafts', 'as printed', 'd l', (2800.8, 2863.0)),
            ('c', 'hub (pi/4)(d_p^2 - d_k^2) b', 'b', 'd^2 l', (2044.8, 2303.8)),
            ('d', 'hub (pi/4)(d_p^2 - d_k^2)(2 d_k - b/3)', '2d-b/3', 'd^2 l', (2471.5, 2893.4))]


def psdr_g3_g5(X):
    """golinski_constraint3 and golinski_constraint5 of psdr, transcribed (see the docstring)."""
    x1, x2, x4, x5, x6, x7 = np.asarray(X, dtype=float).T
    x3 = np.ones(x1.shape)*17
    g3 = 1.93/(x2*x3*x6**4)*x4**3 - 1.
    g5 = np.sqrt((745*x4/x2/x3)**2 + 16.9e6)/(110.0*x6**3) - 1.
    return g3, g5


def volume_reading(x, z, hub, shaft, ratio=3):
    """Volume of the two gear blanks and shafts of Fig. 11 with the given hub and shaft terms."""
    b, mt, l1, l2, d1, d2 = x
    V = 0.0
    for zk, d, l in ((z, d1, l1), (ratio*z, d2, l2)):
        d_st = mt*zk - 2.4*mt          # root diameter of the rim
        d_w = d_st - 4*mt              # inner diameter of the rim
        d_p = d + 2*0.7*d              # hub diameter
        V += np.pi*b/4*(d_st**2 - d_w**2) + np.pi*b/12*(d_w**2 - d_p**2) + hub(d_p, d, b) + shaft(d, l)
    return V


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    t0 = time.perf_counter()
    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',')
    F = np.loadtxt(os.path.join(HERE, 'outputs.csv'), delimiter=',')

    p3, p5 = psdr_g3_g5(X)
    d_model = max(np.max(np.abs(p3 - gm.g3(X))), np.max(np.abs(p5 - gm.g5(X))))
    d_file = max(np.max(np.abs(p3 - F[:, 1])), np.max(np.abs(p5 - F[:, 2])))
    print(f'1. g3, g5 vs the psdr expressions, {len(X)} shipped inputs: max absolute difference '
          f'{d_model:.1e} (golinski_model), {d_file:.1e} (outputs.csv) (DATA.md: 4.4e-16)')

    for name, (x, z) in DESIGNS.items():
        v_used = volume_reading(x, z, HUBS['used'], SHAFTS['d^2 l'])
        v_model = gm.volume(x, z=z)
        if abs(v_used - v_model) > 1e-12*v_model:
            sys.exit(f'term-by-term sum ({v_used}) differs from golinski_model.volume ({v_model}) at {name}')
    print('2. other readings of the volume at the Table 1 designs (crude Monte Carlo, stray process), cm^3:')
    for label, text, hub, shaft, quoted in READINGS:
        v = [volume_reading(x, z, HUBS[hub], SHAFTS[shaft]) for x, z in DESIGNS.values()]
        print(f'   ({label}) {text:42s} {v[0]:7.1f} {v[1]:7.1f}   (DATA.md: {quoted[0]:.1f} and {quoted[1]:.1f})')
    print(f'run time {time.perf_counter() - t0:.2f} s')


if __name__ == '__main__':
    main()
