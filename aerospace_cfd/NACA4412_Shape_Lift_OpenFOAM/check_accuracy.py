"""Recompute the numbers of DATA.md (NACA4412_Shape_Lift_OpenFOAM) that neither the notebook nor
convert_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the two shipped files next to this file. Checks:
 1. Stored precision of the lift histories: the number of significant digits (every value is an integer
    mantissa below 10^7 times a power of ten) and the range of the powers.
 2. Smallest |C_L| stored, and the number of values above 1 in magnitude (the start-up spikes) and the
    last time step at which one occurs.
 3. Precision of the shape weights: the smallest and largest weight, and the range of significant digits
    of their shortest round-trip form (full doubles from the authors' sampler, not rounded values).
 4. Size of the shipped files.

Deterministic; well under a second. Imports numpy and the standard library.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with np.load(os.path.join(HERE, 'lift_series.npz')) as z:
        diff, scale = z['mantissa_diff'], z['scale']
    mant = np.cumsum(diff, axis=1, dtype=np.int64)
    lift = mant / 10.0 ** scale
    print(f'1. lift: |mantissa| max {np.abs(mant).max()} (< 10^7, so at most 7 significant digits); '
          f'powers of ten 10^-{scale.max()} to 10^-{scale.min()}')
    big = np.abs(lift) > 1
    last = max(np.flatnonzero(b).max() + 1 for b in big)
    print(f'2. smallest |C_L| {np.abs(lift).min():.3g}; {int(big.sum())} values with |C_L| > 1, '
          f'the last at time step {last}')
    with open(os.path.join(HERE, 'shape_inputs.csv')) as fh:
        fields = [ln.strip().split(',')[:10] for ln in fh if not ln.startswith('#')]
    digits = [len(v.replace('.', '').replace('-', '').split('e')[0].lstrip('0')) for row in fields for v in row]
    X = np.array([[float(v) for v in row] for row in fields])
    print(f'3. weights {X.min():.3g} to {X.max():.5f}; significant digits of the shortest round-trip form {min(digits)}-{max(digits)}')
    for name in ('shape_inputs.csv', 'lift_series.npz'):
        print(f'4. {name}: {os.path.getsize(os.path.join(HERE, name)):,} bytes')


if __name__ == '__main__':
    main()
