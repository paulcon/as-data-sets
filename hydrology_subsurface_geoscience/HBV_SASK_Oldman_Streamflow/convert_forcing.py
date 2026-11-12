"""Convert the Oldman River HBV-SASK input files of VARS-TOOL into the CSV files used here.

    python convert_forcing.py --src "DIR/Oldman Basin" --out OUTDIR           # write the CSVs to OUTDIR
    python convert_forcing.py --src "DIR/Oldman Basin" --out OUTDIR --check   # also compare with the shipped CSVs

DIR is the `tutorials` folder of https://github.com/vars-tool/vars-tool (commit
df72ad2a3b773ca9f0fc285a0b7d1bc9e043a86d). The source folder holds four text files:
Precipitation_Temperature.inp (date, P mm/day, T degC), streamflow.inp (date, Q m3/s),
monthly_data.inp (12 rows: long-term mean T degC, mean PET mm/day) and initial_condition.inp
(watershed area and the initial SWE, SMS, S1, S2 in mm). The conversion only re-arranges the
numbers (dates become year, month, day columns). P and T have two decimals in the source and
are written with two; Q has at most three and is written with three, so the conversion is
lossless. --out must be outside this directory, neither this directory nor a subdirectory of it
(the shipped files are never overwritten); --check compares to a tolerance of 1e-12 (absolute).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = ('VARS-TOOL (Razavi et al. 2019, https://doi.org/10.1016/j.envsoft.2018.10.005), '
          'github.com/vars-tool/vars-tool commit df72ad2a, tutorials/Oldman Basin')
FILES = ('oldman_daily.csv', 'oldman_monthly.csv', 'oldman_basin.csv')


def read_dated(path, ncol):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            y, mo, d = (int(s) for s in p[0].split('-'))
            rows.append([y, mo, d] + [float(s) for s in p[1:1 + ncol]])
    return np.array(rows)


def convert(src):
    pt = read_dated(os.path.join(src, 'Precipitation_Temperature.inp'), 2)
    q = read_dated(os.path.join(src, 'streamflow.inp'), 1)
    if pt.shape[0] != q.shape[0] or np.any(pt[:, :3] != q[:, :3]):
        sys.exit('dates of Precipitation_Temperature.inp and streamflow.inp differ')
    daily = np.column_stack([pt, q[:, 3]])
    monthly = np.loadtxt(os.path.join(src, 'monthly_data.inp'))
    monthly = np.column_stack([np.arange(1, 13), monthly])
    ini = {}
    with open(os.path.join(src, 'initial_condition.inp')) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 2:
                ini[p[0]] = float(p[1])
    basin = np.array([[ini['WatershedArea_km2'], ini['initial_SWE'], ini['initial_SMS'], ini['S1'], ini['S2']]])
    return daily, monthly, basin


def write(out, daily, monthly, basin):
    np.savetxt(os.path.join(out, 'oldman_daily.csv'), daily, delimiter=',',
               fmt=['%d', '%d', '%d', '%.2f', '%.2f', '%.3f'],
               header=f'Oldman River at Waldron\'s Corner (WSC 05AA023), daily basin forcing and observed '
                      f'streamflow 1979-01-01 to 2008-12-31. Source: {SOURCE}\n'
                      'year,month,day,P (mm/day),T (degC),Qobs (m3/s)')
    np.savetxt(os.path.join(out, 'oldman_monthly.csv'), monthly, delimiter=',', fmt=['%d', '%.2f', '%.2f'],
               header=f'Long-term monthly mean temperature and potential evapotranspiration, Oldman basin. '
                      f'Source: {SOURCE}\nmonth,T_mean (degC),PET_mean (mm/day)')
    np.savetxt(os.path.join(out, 'oldman_basin.csv'), basin, delimiter=',', fmt='%.2f',
               header=f'Watershed area and initial states on 1979-01-01. Source: {SOURCE}\n'
                      'area (km2),SWE0 (mm),SMS0 (mm),S1_0 (mm),S2_0 (mm)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', required=True, help='the "Oldman Basin" folder of the VARS-TOOL tutorials')
    ap.add_argument('--out', required=True, help='output directory (not this directory)')
    ap.add_argument('--check', action='store_true', help='compare the converted files with the shipped ones')
    args = ap.parse_args()
    here, out = os.path.realpath(HERE), os.path.realpath(args.out)
    if os.path.commonpath([here, out]) == here:
        ap.error('--out must be outside this directory (not this directory or a subdirectory of it); '
                 'the shipped files are never overwritten')
    arrs = convert(args.src)
    os.makedirs(args.out, exist_ok=True)
    write(args.out, *arrs)
    print(f'wrote {", ".join(FILES)} to {args.out}: {arrs[0].shape[0]} days')
    if args.check:
        worst = 0.0
        for n in FILES:
            a = np.loadtxt(os.path.join(args.out, n), delimiter=',', ndmin=2)
            b = np.loadtxt(os.path.join(HERE, n), delimiter=',', ndmin=2)
            if a.shape != b.shape:
                sys.exit(f'{n}: shape {a.shape} differs from shipped {b.shape}')
            worst = max(worst, float(np.max(np.abs(a - b))))
        print(f'max absolute difference from the shipped files: {worst:.1e} (tolerance 1e-12): '
              f'{"OK" if worst <= 1e-12 else "FAIL"}')
        if worst > 1e-12:
            sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as err:
        print(f'file not found, or not of the expected kind: {err.filename}\n'
              'The source is the folder "tutorials/Oldman Basin" of a clone of github.com/vars-tool/vars-tool; see DATA.md, "Regenerating".', file=sys.stderr)
        sys.exit(3)
