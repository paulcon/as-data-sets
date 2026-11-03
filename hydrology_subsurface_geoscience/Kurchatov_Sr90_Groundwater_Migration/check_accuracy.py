"""Recompute the numbers of DATA.md (Kurchatov_Sr90_Groundwater_Migration) that neither the notebook
nor convert_data.py prints.

    python check_accuracy.py      (under a second)

Run it from any directory; it reads marthedata.txt and inputs.csv next to this file. Each line prints
the recomputed value; DATA.md quotes these values.

Checks:
 1. Precision of the source file: the number of significant digits of each number string in
    marthedata.txt (trailing zeros of a decimal fraction not counted), as the median per column, for
    the unconditional inputs in rows 1-200 and rows 201-300, for the conditional inputs (dt1-dt3, i2,
    i3), for i1, and for the ten concentrations.
 2. The Weibull parameter order: the largest kd value of each layer, and the Kolmogorov-Smirnov
    p-value of each kd sample against the reverse reading of Table 1 (first number the scale, second
    the shape). (The notebook prints the p-values of the shape-then-scale reading.)

Deterministic. Imports numpy, scipy and the standard library.
"""
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
WEIBULL = {'kd1': (1.1597, 19.9875), 'kd2': (0.891597, 24.4455), 'kd3': (1.27363, 22.4986)}
CONDITIONAL = ('dt1', 'dt2', 'dt3', 'i2', 'i3')


def sig_digits(s):
    """Significant digits of a decimal number string (e.g. '0.0123' -> 3, '1.50' -> 2, '2.1e-05' -> 2)."""
    m = s.strip().lstrip('+-').lower().split('e')[0]
    digits = m.replace('.', '').lstrip('0')
    if '.' in m:
        digits = digits.rstrip('0') or digits
    else:
        digits = digits.rstrip('0') or '0'
    return max(len(digits), 1)


def main():
    lines = open(os.path.join(HERE, 'marthedata.txt'), encoding='ascii').read().splitlines()
    names = lines[0].split('\t')
    rows = [ln.split('\t') for ln in lines[1:] if ln.strip()]
    D = np.array([[sig_digits(s) for s in r] for r in rows])          # (300, 30)
    col = {n: j for j, n in enumerate(names)}
    uncond = [j for j, n in enumerate(names[:20]) if n not in CONDITIONAL and n != 'i1']
    med = lambda A: np.median(A, axis=0)
    print('1. significant digits of the number strings in marthedata.txt (median per column):')
    for lab, sl in (('rows 1-200', slice(0, 200)), ('rows 201-300', slice(200, 300))):
        m = med(D[sl][:, uncond])
        print(f'   unconditional inputs (14 columns), {lab}: median per column {m.min():g} to {m.max():g}')
    m = med(D[:, [col[n] for n in CONDITIONAL]])
    print(f'   conditional inputs dt1-dt3, i2, i3, all rows: median per column {m.min():g} to {m.max():g}')
    i1 = D[:, col['i1']]
    print(f'   i1, all rows: {i1.min()} to {i1.max()} significant digits; '
          f'{len(np.unique([r[col["i1"]] for r in rows]))} distinct values')
    m = med(D[:, 20:])
    print(f'   concentrations (10 columns): median per column {m.min():g} to {m.max():g}; '
          f'entries with fewer than 7 digits: {int(np.sum(D[:, 20:] < 7))} of {D[:, 20:].size}')

    X = np.loadtxt(os.path.join(HERE, 'inputs.csv'), delimiter=',')
    print('2. Weibull parameter order (Table 1 prints two numbers per kd):')
    for n, (a, b) in WEIBULL.items():
        x = X[:, col[n]]
        p_rev = stats.kstest(x, 'weibull_min', args=(b, 0, a)).pvalue
        print(f'   {n}: largest value {x.max():.1f}; KS p-value, shape {b:g} and scale {a:g} '
              f'(reverse reading): {p_rev:.1e}')


if __name__ == '__main__':
    main()
