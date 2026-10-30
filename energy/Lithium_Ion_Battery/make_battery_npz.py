"""Convert the used columns of the three Hadigol et al. text files to battery_data.npz.

Run from this directory: python make_battery_npz.py

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
- source_md5 (3, 2): md5 of each source file, to check against the archived originals.

Not converted: the 150 concentration columns per file (ConA, ConS, ConC). They
are unused and go to the archival deposit of the original files only.
"""
import hashlib

import numpy as np

RATES = {'025C': ('0.25C.txt', 'V_025C.txt'), '1C': ('1C.txt', 'V_1C.txt'), '4C': ('4C.txt', 'V_4C.txt')}
CUTOFF = 2.8      # V, the discharge cutoff of every run
TOL = 1e-3        # V; a run whose last voltage is lower than CUTOFF - TOL overshot the cutoff


def md5(fname):
    with open(fname, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    out, md5s, X = {}, [], None
    for rate, (fname, vname) in RATES.items():
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
        md5s.append([fname, md5(fname)])
        md5s.append([vname, md5(vname)])
    out.update(X=X, run=np.arange(1, 3601), input_names=np.array(names), source_md5=np.array(md5s))
    np.savez_compressed('battery_data.npz', **out)

    # check: reload and compare with the text files, bit for bit
    z = np.load('battery_data.npz')
    for rate, (fname, vname) in RATES.items():
        d = np.loadtxt(fname, delimiter=',', skiprows=1)
        assert np.array_equal(z['X'], d[:, 1:20])
        assert np.array_equal(z[f'voltage_{rate}'], d[:, 20:70])
        assert np.array_equal(z[f'time_{rate}'], d[:, 70:120])
        assert np.array_equal(z[f'vgrid_{rate}'], np.loadtxt(vname))
        print(f'{fname}: identical; dropped runs {z[f"dropped_runs_{rate}"].tolist()}')
    print(f'wrote battery_data.npz: {len(z.files)} arrays')


if __name__ == '__main__':
    main()
