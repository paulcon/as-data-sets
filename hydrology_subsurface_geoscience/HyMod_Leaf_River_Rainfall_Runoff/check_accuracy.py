"""Recompute the numbers of DATA.md (HyMod_Leaf_River_Rainfall_Runoff) that neither the notebook nor
generate_data.py prints.

    python check_accuracy.py      (a few seconds)

Run it from any directory; it evaluates hymod_model.py on the forcing next to this file. Each line
prints the recomputed value; DATA.md quotes these values.

Checks:
 1. The forward-mode gradients against central differences on 500 samples that are not in
    inputs.csv (numpy default_rng(1), uniform on the paper's ranges), with steps h = 1e-6 and 1e-3 as a
    fraction of each half-range: median and largest relative error of the gradient vector per QoI,
    and the number of samples above 1e-3. (The notebook does the same on rows 0-199 of inputs.csv.)
 2. The dates of LeafCatch.txt: the number of days from 1 October 1948 to 30 September 1988, and the
    first day of each window of forcing_other_years.csv, if row 1 is 1 October 1948.

The comparisons with SAFE itself in DATA.md (SAFEpython 0.2.0: daily flows, NSE, Qmax, run times, and
SAFE's own EET) are an external run and are not repeated here.

Deterministic. Imports numpy and the standard library.
"""
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hymod_model as hm  # noqa: E402


def check_fd():
    rng = np.random.default_rng(1)
    X = hm.LOWER + (hm.UPPER - hm.LOWER) * rng.random((500, 5))
    half = (hm.UPPER - hm.LOWER) / 2
    _, G = hm.qoi_and_grad(X)
    Gx = G * half
    for h in (1e-6, 1e-3):
        Gfd = np.empty_like(Gx)
        for j in range(5):
            dp = np.zeros(5)
            dp[j] = h * half[j]
            Gfd[:, :, j] = (hm.qoi(X + dp) - hm.qoi(X - dp)) / (2 * h)
        rel = np.linalg.norm(Gfd - Gx, axis=2) / np.linalg.norm(Gx, axis=2)
        print(f'1. 500 samples of default_rng(1), central differences h = {h:g}: relative error, median / max: '
              + '; '.join(f'{q} {np.median(rel[:, k]):.1e} / {rel[:, k].max():.1e} '
                          f'({int((rel[:, k] > 1e-3).sum())} > 1e-3)' for k, q in enumerate(hm.QOI_NAMES)))


def check_dates():
    d0, d1 = datetime.date(1948, 10, 1), datetime.date(1988, 9, 30)
    print(f'2. days from {d0} to {d1}, inclusive: {(d1 - d0).days + 1} (LeafCatch.txt has 14610 rows)')
    for row in (366, 731, 3653):
        print(f'   row {row} is {d0 + datetime.timedelta(days=row - 1)}')


if __name__ == '__main__':
    check_fd()
    check_dates()
