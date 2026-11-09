"""Recompute the numbers of DATA.md (Atmospheric_Selenium_Lifetime) that neither the notebook nor
convert_data.py prints.

    python check_accuracy.py                      # checks 1-2 (the shipped CSVs only)
    python check_accuracy.py <deposit folder>     # also check 3 (needs the deposited NetCDF files)

Checks:
 1. Stored precision of the physical inputs: the largest number of significant digits written in each group
    of columns of se_training_inputs.csv (the rate constants are written with 3).
 2. The rounding this causes: the largest relative difference between the rate constants as written and
    10**t of the authors' transformed file (full precision).
 3. Sensitivity of tau to the quadrature of the area integral: tau with cos(latitude) cell weights instead of
    the Gauss-Legendre weights, and with float32 instead of float64 accumulation (largest relative difference
    over the 400 training runs). Needs sensitivity_runs_data.nc of doi:10.3929/ethz-b-000357105; without the
    folder this check stops with a message saying so.

Deterministic; checks 1-2 take well under a second, check 3 about 5 s. numpy and scipy.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def lines(name):
    with open(os.path.join(HERE, name), encoding='utf-8') as fh:
        fh.readline()
        cols = [c.split(' [')[0] for c in fh.readline()[2:].strip().split(',')]
        return cols, [ln.strip().split(',') for ln in fh if ln.strip()]


def sig_digits(s):
    mant = s.lower().split('e')[0].replace('-', '').replace('.', '').lstrip('0')
    return len(mant.rstrip('0')) if '.' not in s.lower().split('e')[0] else len(mant)


cols, rows = lines('se_training_inputs.csv')
names = cols[2:]
groups = {'k1-k17': names[:17], 'fk18-fk21': names[17:21], 'emissions': names[21:25], 'speciation': names[25:30],
          'acc_coeff': ['acc_coeff'], 'r_aer': ['r_aer'], 'emiss_aer': ['emiss_aer']}
print('1. largest number of significant digits written, se_training_inputs.csv:')
for g, ns in groups.items():
    idx = [cols.index(n) for n in ns]
    print(f'   {g:11s} {max(sig_digits(r[i]) for r in rows for i in idx)}')
X = np.loadtxt(os.path.join(HERE, 'se_training_inputs.csv'), delimiter=',')[:, 2:]
T = np.loadtxt(os.path.join(HERE, 'se_training_inputs_transformed.csv'), delimiter=',')[:, 2:]
rel = np.abs(X[:, :17] / 10**T[:, :17] - 1)
print(f'2. rate constants as written vs 10**t (transformed file): largest relative difference {rel.max():.1e}, '
      f'median {np.median(rel):.1e}')

if len(sys.argv) < 2:
    print('3. skipped: give the folder with sensitivity_runs_data.nc (doi:10.3929/ethz-b-000357105) to run it')
    sys.exit(0)
from scipy.io import netcdf_file  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import convert_data  # noqa: E402

nc = netcdf_file(os.path.join(sys.argv[1], 'sensitivity_runs_data.nc'), 'r', mmap=False)
V = nc.variables
lat = np.asarray(V['lat'][:])
area_g, _ = convert_data.cell_areas(lat)
area_c = np.cos(np.radians(lat)) * area_g.sum() / np.cos(np.radians(lat)).sum()
tot = {}
for key, area, dt in (('gauss64', area_g, np.float64), ('cos64', area_c, np.float64),
                      ('gauss32', area_g.astype(np.float32), np.float32)):
    s = {}
    for k in convert_data.BURDENS + convert_data.WET + convert_data.DRY:
        a = np.asarray(V[k][:], dtype=dt)
        s[k] = (a * area[None, :, None].astype(dt)).sum(axis=(1, 2), dtype=dt).astype(np.float64)
    B = sum(s[k] for k in convert_data.BURDENS)
    F = sum(s[k] for k in convert_data.WET + convert_data.DRY)
    tot[key] = B / F / 86400.0
ref = np.loadtxt(os.path.join(HERE, 'se_training_outputs.csv'), delimiter=',')[:, -1]
print(f'3. tau, Gauss weights (float64) vs shipped: largest relative difference {np.abs(tot["gauss64"] / ref - 1).max():.1e}')
print(f'   tau with cos(latitude) weights: largest relative difference {np.abs(tot["cos64"] / ref - 1).max():.1e}')
print(f'   tau with float32 accumulation: largest relative difference {np.abs(tot["gauss32"] / ref - 1).max():.1e}')
