"""Recompute the numbers of DATA.md (Residential_Building_Stock_Energy_Use) that describe the full
deposited table, which neither the notebook nor convert_data.py prints.

    python check_accuracy.py "<folder>/Urban Building Energy Stock Datasets.zip"

The full table (1,030,022 rows) is not shipped here. Download the archive of version 1 of Mendeley
Data doi:10.17632/m6vv9k9gcd.1 (https://data.mendeley.com/datasets/m6vv9k9gcd/1, 52,851,275 bytes,
SHA-256 1f6bb3e1...c42c29) and give its path; without it the script stops and says so. The archive
is read in place (nothing is extracted). Takes about 30-40 s and about 1 GB of memory.

Checks (each line prints the recomputed value):
 1. Rows and columns of the deposited CSV; rows per building type x weather x renewables cell.
 2. Pearson correlation of each of the 15 numeric inputs with Energy_Use_Intensity over all rows
    (the quantity of the paper's Fig. 1), to three significant digits.
 3. Mean, sample standard deviation (ddof = 1), minimum and maximum of each input over all rows (the
    paper's Table 2), and the number of distinct levels.
 4. Mean, sample standard deviation (ddof = 1), minimum and maximum of each numeric output over
    all rows (the paper's Table 3); for Energy_Use_Intensity also the rows in [640, 650) and >= 650,
    and the maximum EUI in each building type x weather x renewables cell.
 5. The largest |correlation| between two inputs over all rows.
 6. Distinct HVAC_Efficiency levels in the rows with and without renewables; distinct
    Air_Change_Rate levels and the rows with Air_Change_Rate > 3 (by building type and renewables).
 7. The EUI identity over all rows: max |EUI - (HPCF*HCF*Heating_Usage + EPCF*Total_Electricity)/Area|
    and max |Total_Electricity - (Interior_Lighting + Interior_Equipment - 0.95*Photovoltaic_Power)|.

Imports numpy and the standard library.
"""
import collections
import csv
import hashlib
import io
import os
import sys
import zipfile

import numpy as np

SHA256 = '1f6bb3e1dfcb7ebefbd1a75f93d56dd12db2923aab58717f077ab4a643c42c29'
MEMBER = 'Urban Building Energy Stock Datasets/urban_building_stock_datasets.csv'
CATS = ['Simple_Building_Energy_Rating', 'Weather_File', 'Building_Type', 'Renewable_Energy_Usage',
        'Building_Energy_Rating']


def main(argv):
    if len(argv) < 2 or not os.path.isfile(argv[1]):
        print('check_accuracy.py needs the deposited archive "Urban Building Energy Stock Datasets.zip" '
              '(Mendeley Data doi:10.17632/m6vv9k9gcd.1, version 1, '
              'https://data.mendeley.com/datasets/m6vv9k9gcd/1). Download it and rerun: '
              'python check_accuracy.py <path of the zip>', file=sys.stderr)
        sys.exit(3)
    h = hashlib.sha256()
    with open(argv[1], 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    if h.hexdigest() != SHA256:
        sys.exit(f'archive SHA-256 {h.hexdigest()} differs from version 1 ({SHA256})')
    with zipfile.ZipFile(argv[1]) as zf, zf.open(MEMBER) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding='utf-8', newline=''))
        header = next(reader)
        ci = [header.index(c) for c in CATS]
        ni = [j for j in range(len(header)) if j not in ci]
        cats, vals = [], []
        for row in reader:
            cats.append(tuple(row[j] for j in ci[1:4]))
            vals.append([float(row[j]) for j in ni])
    A = np.array(vals)
    names = [header[j] for j in ni]
    col = {n: A[:, k] for k, n in enumerate(names)}
    print(f'1. deposited CSV: {A.shape[0]} rows, {len(header)} columns')
    for key, c in sorted(collections.Counter(cats).items()):
        print(f'   weather {key[0]:10s} type {key[1]:13s} renewables {key[2]:3s}: {c} rows')
    eui = col['Energy_Use_Intensity']
    inputs = names[:15]
    print('2. correlation with EUI over all rows: '
          + ', '.join(f'{n} {np.corrcoef(col[n], eui)[0, 1]:.3g}' for n in inputs))
    print('3. inputs over all rows (mean, sd, min, max, levels):')
    for n in inputs:
        v = col[n]
        print(f'   {n:30s} {v.mean():9.3f} {v.std(ddof=1):9.3f} {v.min():8.3f} {v.max():8.3f} {np.unique(v).size:4d}')
    print('4. outputs over all rows (mean, sd, min, max):')
    for n in names[15:]:
        v = col[n]
        print(f'   {n:38s} {v.mean():10.2f} {v.std(ddof=1):10.2f} {v.min():9.2f} {v.max():10.2f}')
    print(f'   EUI over all rows: mean {eui.mean():.2f}, min {eui.min():.2f}, max {eui.max():.4f}; '
          f'rows in [640, 650): {np.sum((eui >= 640) & (eui < 650))}; rows >= 650: {np.sum(eui >= 650)}')
    cell = np.array(['/'.join(c) for c in cats])
    cmax = [eui[cell == k].max() for k in sorted(set(cell))]
    print(f'   maximum EUI per building type x weather x renewables cell ({len(cmax)} cells): '
          f'{min(cmax):.4f} to {max(cmax):.4f}')
    R = np.corrcoef(np.array([col[n] for n in inputs]))
    np.fill_diagonal(R, 0.0)
    i, j = np.unravel_index(np.abs(R).argmax(), R.shape)
    print(f'5. largest |correlation| between two inputs over all rows: {R[i, j]:.3f} ({inputs[i]}, {inputs[j]})')
    ren = np.array([c[2] for c in cats])
    btype = np.array([c[1] for c in cats])
    hv = col['HVAC_Efficiency']
    ach = col['Air_Change_Rate']
    print(f'6. HVAC_Efficiency levels: renewables Yes {np.unique(hv[ren == "Yes"]).size}, '
          f'No {np.unique(hv[ren == "No"]).size}; Air_Change_Rate levels {np.unique(ach).size} '
          f'(max {ach.max():g}); rows with Air_Change_Rate > 3: {np.sum(ach > 3)}, of which '
          + ', '.join(f'{t}/{r} {np.sum((ach > 3) & (btype == t) & (ren == r))}'
                      for t in sorted(set(btype)) for r in ('No', 'Yes')))
    rebuilt = (col['Heating_Primary_Conversion_Factor'] * col['Heating_Conversion_Factor'] * col['Heating_Usage']
               + col['Electricity_Primary_Conversion_Factor'] * col['Total_Electricity_Energy']) \
        / col['Total_Building_Area']
    el = (col['Interior_Lighting_Energy'] + col['Interior_Equipment_Energy']
          - 0.95 * col['Photovoltaic_Power'] - col['Total_Electricity_Energy'])
    print(f'7. max |EUI - rebuilt EUI| over all rows: {np.abs(eui - rebuilt).max():.1e} kWh/(m2 yr); '
          f'max |electricity - (lighting + equipment - 0.95 PV)|: {np.abs(el).max():.4f} kWh/yr')


if __name__ == '__main__':
    main(sys.argv)
