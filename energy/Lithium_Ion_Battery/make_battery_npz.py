"""Convert the used columns of the three Hadigol et al. text files to battery_data.npz.

    python make_battery_npz.py --src DIR           # writes battery_data.npz here
    python make_battery_npz.py --src DIR --check   # compares with the shipped npz; writes nothing

DIR holds 0.25C.txt, 1C.txt and 4C.txt (22.4 MB each), archived on Zenodo with the voltage grids,
doi:10.5281/zenodo.24174045; the grids V_<rate>.txt are read from this directory. Without --src,
or without the files, the script exits with code 3 and says where to get them.

The conversion is lossless for the columns it keeps: every array in the npz is
checked to be bit-identical to the text values (each stored with 17 significant
digits, which float64 round-trips exactly). Nothing is rescaled or dropped:

- X (3600, 19): the normalized inputs, which are identical in all three files
  (checked) and so are stored once;
- voltage_<rate> (3600, 50): columns Vol01..Vol50, cell voltage [V] on the t* grid;
- time_<rate> (3600, 50): columns Cap01..Cap50, unchanged. These are inferred
  to be the elapsed discharge time [s] at each voltage of vgrid_<rate>; the
  notebook converts them to capacity [mAh cm^-2];
- vgrid_<rate> (50,): the voltage grid from V_<rate>.txt [V];
- run (3600,): the Run column, 1..3600;
- input_names (19,): the input column headers, verbatim;
- dropped_runs_<rate>: Run numbers whose final voltage (Vol50) is more than
  1 mV below the 2.8 V cutoff. This is a flag; the rows stay in the arrays and the
  notebook applies the same rule;
- source_md5 (6, 2): file name and md5 of each of the six source files, to check
  against the archived originals.

Not converted: the 150 concentration columns per file (ConA, ConS, ConC). They
are unused and are only in the Zenodo deposit of the original files.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOI = '10.5281/zenodo.24174045'

RATES = {'025C': ('0.25C.txt', 'V_025C.txt'), '1C': ('1C.txt', 'V_1C.txt'), '4C': ('4C.txt', 'V_4C.txt')}
CUTOFF = 2.8      # V, the discharge cutoff of every run
TOL = 1e-3        # V; a run whose last voltage is lower than CUTOFF - TOL overshot the cutoff


def md5(fname):
    with open(fname, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', help=f'directory that holds 0.25C.txt, 1C.txt and 4C.txt (doi:{DOI})')
    ap.add_argument('--check', action='store_true', help='compare with the shipped battery_data.npz; write nothing')
    args = ap.parse_args()
    missing = [f for f, _ in RATES.values() if not args.src or not os.path.isfile(os.path.join(args.src, f))]
    if missing:
        print(f'{", ".join(missing)} not found{" in " + args.src if args.src else ""}. Download them from '
              f'https://doi.org/{DOI} and rerun with --src DIR.', file=sys.stderr)
        sys.exit(3)
    out, md5s, X = {}, [], None
    for rate, (name, vname) in RATES.items():
        fname, vname = os.path.join(args.src, name), os.path.join(HERE, vname)
        with open(fname) as fh:
            header = fh.readline().strip().split(',')
        d = np.loadtxt(fname, delimiter=',', skiprows=1)
        assert d.shape == (3600, 270) and np.isfinite(d).all()
        assert header[20:70] == [f'Vol{k:02d}' for k in range(1, 51)]
        assert header[70:120] == [f'Cap{k:02d}' for k in range(1, 51)]
        assert np.array_equal(d[:, 0], np.arange(1, 3601))
        if X is None:
            X, names = d[:, 1:20].copy(), header[1:20]
        assert np.array_equal(d[:, 1:20], X) and header[1:20] == names
        out[f'voltage_{rate}'] = d[:, 20:70].copy()
        out[f'time_{rate}'] = d[:, 70:120].copy()
        out[f'vgrid_{rate}'] = np.loadtxt(vname)
        out[f'dropped_runs_{rate}'] = d[d[:, 69] < CUTOFF - TOL, 0].astype(np.int64)
        md5s.append([name, md5(fname)])
        md5s.append([os.path.basename(vname), md5(vname)])
    out.update(X=X, run=np.arange(1, 3601), input_names=np.array(names), source_md5=np.array(md5s))
    path = os.path.join(HERE, 'battery_data.npz')
    if args.check:
        with np.load(path) as z:
            ok = sorted(z.files) == sorted(out) and all(
                z[k].dtype == out[k].dtype and np.array_equal(z[k], out[k]) for k in out)
        print(f'battery_data.npz: every array identical to the conversion: {ok}')
        sys.exit(0 if ok else 1)
    np.savez_compressed(path, **out)

    # check: reload and compare with the text files, bit for bit
    z = np.load(path)
    for rate, (name, vname) in RATES.items():
        fname, vname = os.path.join(args.src, name), os.path.join(HERE, vname)
        d = np.loadtxt(fname, delimiter=',', skiprows=1)
        assert np.array_equal(z['X'], d[:, 1:20])
        assert np.array_equal(z[f'voltage_{rate}'], d[:, 20:70])
        assert np.array_equal(z[f'time_{rate}'], d[:, 70:120])
        assert np.array_equal(z[f'vgrid_{rate}'], np.loadtxt(vname))
        print(f'{name}: identical; dropped runs {z[f"dropped_runs_{rate}"].tolist()}')
    print(f'wrote battery_data.npz: {len(z.files)} arrays')


if __name__ == '__main__':
    main()
