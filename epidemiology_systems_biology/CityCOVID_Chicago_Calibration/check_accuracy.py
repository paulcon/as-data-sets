"""Recompute the numbers of DATA.md (CityCOVID_Chicago_Calibration) that neither the notebook nor
convert_data.py prints.

    python check_accuracy.py

Run it from any directory; it reads the shipped CSVs next to this file. Deterministic; well under a
second; numpy and the standard library only.

Checks:
 1. Seeding time, rows 501-700: each value equals 31 + 28 h(j), with h(j) the base-2 radical inverse
    (Van der Corput) of an integer j; prints the range of j, how many j are distinct, and the largest
    deviation. (Rows 1-500 and the other three inputs are reconstructed in the notebook.)
 2. Stored precision: the largest number of decimals written in the two output files, and the number
    of distinct seeding-time values.
 3. Observed data: the number of days with an observed death count and with an observed census, and
    the first and last day index of each.
 4. Rows 501-700: the largest absolute Pearson correlation between any two of the four calibrated inputs
    (their theta2 values are not the base-2 Halton points of the same indices as theta1, theta3, theta4).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def radical_inverse(i, base):
    r, f = 0.0, 1.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def lines(name):
    with open(os.path.join(HERE, name)) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]


def main():
    P = np.loadtxt(os.path.join(HERE, 'citycovid_inputs.csv'), delimiter=',')
    t2 = P[:, 6]
    table = {round(radical_inverse(j, 2), 12): j for j in range(1, 1 << 12)}
    u = (t2[500:] - 31.0) / 28.0
    js = np.array([table.get(round(v, 12), -1) for v in u])
    if np.any(js < 0):
        print('1. some rows 501-700 are not 31 + 28 h(j) for j < 4096')
    else:
        dev = np.abs(31.0 + 28.0 * np.array([radical_inverse(int(j), 2) for j in js]) - t2[500:]).max()
        print(f'1. rows 501-700: theta2 = 31 + 28 h(j), j from {js.min()} to {js.max()}, '
              f'{np.unique(js).size} distinct j, largest deviation {dev:.1e}; '
              f'{js.max() - js.min() + 1 - np.unique(js).size} integers of that range are not used')
    for name in ('citycovid_hospitalizations.csv', 'citycovid_deaths.csv'):
        dec = max(len(v.split('.')[1]) if '.' in v else 0 for ln in lines(name) for v in ln.split(','))
        print(f'2. {name}: at most {dec} decimals written')
    print(f'2. distinct seeding-time values: {np.unique(t2).size} of {t2.size}')
    O = np.loadtxt(os.path.join(HERE, 'observed_chicago.csv'), delimiter=',')
    for k, name in [(1, 'deaths'), (2, 'hospital census')]:
        ok = ~np.isnan(O[:, k])
        print(f'3. observed {name}: {ok.sum()} days, day index {O[ok, 0].min():.0f} to {O[ok, 0].max():.0f}')
    C = np.corrcoef(P[500:][:, [1, 6, 7, 8]], rowvar=False)
    r = np.abs(C[np.triu_indices(4, 1)]).max()
    print(f'4. rows 501-700: largest |correlation| between two calibrated inputs {r:.3f}')


if __name__ == '__main__':
    main()
