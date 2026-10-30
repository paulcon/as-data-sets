"""Rewrite the 2016 HyShot II data files in physical units, and check that nothing is lost.

The 2016 files, as on the tag ``legacy-master`` (directory ``HyShotII/``):

* ``HyShotII30.txt``: 52 rows, no header; 7 inputs normalized to [-1, 1] and the
  exit pressure at fuel plenum pressure 4.8 bar (JCP 2015). Rows 51-52 are the
  validation runs at the corners of the input box.
* ``HyShotII35.txt``: a header and 16 rows: run, 7 inputs, exit pressure at
  phi = 0.30 and at phi = 0.35 (the book's section 5.1). Rows 1-14 have physical
  inputs; rows 15-16, the validation runs, have normalized inputs and a ``0``
  placeholder in the phi = 0.30 column (those two runs are unavailable).

This script writes ``HyShotII_phi030.csv`` and ``HyShotII_phi035.csv``: physical
inputs, a ``#`` header with names and units, a ``validation`` flag column, and
``nan`` for the two missing phi = 0.30 values. Numbers are written with Python's
shortest round-trip ``repr``, so every value that was physical in the old files
(the 14 design rows of HyShotII35.txt and every exit pressure) is bit-identical.

Normalized inputs z become physical inputs x = mid + half * z, with
mid = (upper + lower) / 2 and half = (upper - lower) / 2 (the map of
``active_subspaces.Uniform``). A float64 x cannot carry every bit of z (near the
top of the stagnation-pressure range one ulp of x is 1.5e-15 in z), so the check
is: z recovered as (x - mid) / half from the written text equals the old z to
within 2e-15, which is the float64 resolution of the physical values.

Usage, from this directory:

    python convert_legacy_data.py           # write the two csv files
    python convert_legacy_data.py --check   # verify the committed files; write nothing
    python convert_legacy_data.py --legacy-dir DIR   # read the old files from DIR instead of git

By default the old files are read with ``git show legacy-master:HyShotII/<name>``
and checked against their SHA-256 hashes.
"""
import argparse
import hashlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Normalization bounds (2016 notebook, cell 5; JCP 2015 Table 2 in the units of the data).
LOWER = np.array([2.6, 0.1, 122.5, 1.6448e7, 3.0551e6, 0.03, 0.087])
UPPER = np.array([4.6, 1.9, 367.5, 1.9012e7, 3.4280e6, 0.07, 0.203])
MID, HALF = (UPPER + LOWER) / 2, (UPPER - LOWER) / 2
INPUTS = ['angle_of_attack [deg]', 'turbulence_intensity [%]', 'turbulence_length_scale [mm]',
          'stagnation_pressure [Pa]', 'stagnation_enthalpy [J/kg]',
          'cowl_transition_location [m]', 'ramp_transition_location [m]']

LEGACY = {  # file name on legacy-master: SHA-256
    'HyShotII30.txt': '0fee8bbb9f86e407dc0a9438eb0782e1c4f3b518dc8060950f59c4a06b56b4b0',
    'HyShotII35.txt': 'a6c57c8c0777cf0faae151a6a1e669e0d7db6ac9e200f1aa3b71d880f862e545',
}
TOL = 2e-15  # bound on |z_recovered - z_old| for converted inputs

HEADER30 = """\
# HyShot II scramjet: exit pressure at fuel plenum pressure 4.8 bar (fuel/air equivalence ratio
# phi = 0.30 at nominal inflow), the 52 runs of Constantine, Emory, Larsson and Iaccarino,
# J. Comput. Phys. 302 (2015) 1-20, doi:10.1016/j.jcp.2015.09.001.
# Stored data (Joe RANS flamelet simulations, Stanford CTR); see DATA.md. Inputs in physical units;
# normalize with z = (x - (upper + lower)/2) / ((upper - lower)/2), bounds in DATA.md.
# exit_pressure: pressure averaged over the last 10 mm of the combustor (0.64 <= x <= 0.65 m).
# validation = 1: the two runs made after w was computed, near opposite corners of the input box.
# Written by convert_legacy_data.py from HyShotII30.txt (tag legacy-master), which stored the
# inputs normalized; converting back reproduces them to 2e-15.
"""
HEADER35 = """\
# HyShot II scramjet: exit pressure at fuel/air equivalence ratios phi = 0.30 and phi = 0.35,
# the 16 runs of P. G. Constantine, Active Subspaces, SIAM 2015, section 5.1 (Table 5.2),
# doi:10.1137/1.9781611973860. The same 16 input points for both phi.
# Stored data (Joe RANS flamelet simulations, Stanford CTR); see DATA.md. Inputs in physical units;
# normalize with z = (x - (upper + lower)/2) / ((upper - lower)/2), bounds in DATA.md.
# exit_pressure: pressure averaged over the last 10 mm of the combustor (0.64 <= x <= 0.65 m).
# validation = 1: the two runs made after w was computed, near opposite corners of the input box.
# exit_pressure_phi030 is nan for runs 15-16: those two runs are unavailable (author, 2026-10-30).
# Written by convert_legacy_data.py from HyShotII35.txt (tag legacy-master), whose runs 15-16 had
# normalized inputs and 0 placeholders; converting back reproduces the inputs to 2e-15.
"""


def legacy_text(name, legacy_dir):
    if legacy_dir is None:
        raw = subprocess.run(['git', 'show', f'legacy-master:HyShotII/{name}'], cwd=HERE,
                             check=True, capture_output=True).stdout
    else:
        raw = (Path(legacy_dir) / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != LEGACY[name]:
        sys.exit(f'{name}: SHA-256 {digest} is not the legacy file')
    return raw.decode()


def fmt(v):
    v = float(v)
    return 'nan' if np.isnan(v) else repr(v)


def write_csv(header, columns, rows):
    out = io.StringIO()
    out.write(header)
    out.write('# columns: ' + ', '.join(columns) + '\n')
    for row in rows:
        out.write(','.join(str(int(v)) if k in (0, len(row) - 1) else fmt(v)
                           for k, v in enumerate(row)) + '\n')
    return out.getvalue()


def convert(legacy_dir=None):
    d30 = np.loadtxt(io.StringIO(legacy_text('HyShotII30.txt', legacy_dir)), delimiter=',')
    d35 = np.loadtxt(io.StringIO(legacy_text('HyShotII35.txt', legacy_dir)), delimiter=',', skiprows=1)
    assert d30.shape == (52, 8) and d35.shape == (16, 10)

    # phi = 0.30, JCP 2015: all inputs normalized in the old file.
    z30 = d30[:, :7]
    x30 = MID + HALF * z30
    val30 = np.r_[np.zeros(50), np.ones(2)]
    rows30 = np.column_stack([np.arange(1, 53), x30, d30[:, 7], val30])
    text30 = write_csv(HEADER30, ['run'] + INPUTS + ['exit_pressure [bar]', 'validation'], rows30)

    # book section 5.1: rows 1-14 physical, rows 15-16 normalized with 0 in the phi = 0.30 column.
    assert np.array_equal(d35[:, 0], np.arange(1, 17))
    assert np.all(d35[-2:, 8] == 0) and np.all(d35[:-2, 8] > 0)
    x35 = d35[:, 1:8].copy()
    x35[-2:] = MID + HALF * d35[-2:, 1:8]
    f35_030 = d35[:, 8].copy()
    f35_030[-2:] = np.nan
    val35 = np.r_[np.zeros(14), np.ones(2)]
    rows35 = np.column_stack([d35[:, 0], x35, f35_030, d35[:, 9], val35])
    text35 = write_csv(HEADER35, ['run'] + INPUTS + ['exit_pressure_phi030 [bar]',
                                                     'exit_pressure_phi035 [bar]', 'validation'], rows35)

    # Round trip: parse what will be written and recover the old values.
    n30 = np.loadtxt(io.StringIO(text30), delimiter=',')
    n35 = np.loadtxt(io.StringIO(text35), delimiter=',')
    err30 = np.abs((n30[:, 1:8] - MID) / HALF - z30).max()
    err35 = np.abs((n35[-2:, 1:8] - MID) / HALF - d35[-2:, 1:8]).max()
    checks = {
        'phi030 inputs: max |z_recovered - z_old|': (err30, err30 <= TOL),
        'phi035 corner inputs: max |z_recovered - z_old|': (err35, err35 <= TOL),
        'phi035 physical inputs, runs 1-14, bit-identical': (None, np.array_equal(n35[:-2, 1:8], d35[:-2, 1:8])),
        'exit pressures bit-identical (phi030 file)': (None, np.array_equal(n30[:, 8], d30[:, 7])),
        'exit pressures bit-identical (phi035 file)': (None, np.array_equal(n35[:, 9], d35[:, 9])
                                                       and np.array_equal(n35[:-2, 8], d35[:-2, 8])),
        'missing phi = 0.30 values are exactly runs 15-16': (None, np.array_equal(np.isnan(n35[:, 8]), val35 == 1)),
        'converted inputs inside the bounds': (None, bool(np.all((n30[:, 1:8] >= LOWER) & (n30[:, 1:8] <= UPPER))
                                                          and np.all((n35[:, 1:8] >= LOWER) & (n35[:, 1:8] <= UPPER)))),
    }
    for name, (value, ok) in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ('' if value is None else f': {value:.1e}'))
    if not all(ok for _, ok in checks.values()):
        sys.exit('round trip failed')
    return {'HyShotII_phi030.csv': text30, 'HyShotII_phi035.csv': text35}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='compare with the committed files, write nothing')
    parser.add_argument('--legacy-dir', help='directory holding HyShotII30.txt and HyShotII35.txt')
    args = parser.parse_args()
    files = convert(args.legacy_dir)
    for name, text in files.items():
        path = HERE / name
        if args.check:
            same = path.exists() and path.read_text() == text
            print(f"{'ok  ' if same else 'FAIL'} {name} matches the conversion")
            if not same:
                sys.exit(1)
        else:
            path.write_text(text)
            print(f'wrote {name}')


if __name__ == '__main__':
    main()
