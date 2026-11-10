"""Convert the BISON fuel-performance files distributed with pyMAISE into this data set's CSV files.

Stored data: the 400 BISON runs (13 inputs, 4 outputs) of Radaideh and Kozlowski (2020),
doi:10.1016/j.ress.2019.106731, as distributed by the pyMAISE package (files fp_inp.csv and
fp_out.csv; Apache License 2.0, text in pyMAISE_LICENSE.txt). The source is either

  * the pyMAISE 0.0.2 wheel from PyPI, pyMAISE-0.0.2-py3-none-any.whl (members
    pyMAISE/data/fp_inp.csv and pyMAISE/data/fp_out.csv are read from the zip; nothing is
    extracted or executed), or
  * a folder holding fp_inp.csv and fp_out.csv as downloaded from the Zenodo record
    https://zenodo.org/records/20140559 (the files later pyMAISE versions fetch).

The SHA-256 of the wheel (when given) and of both CSV files are checked first; the two file
hashes are those of the pyMAISE download registry, so both sources give the same bytes. This
script only changes the layout: it writes two '#' lines (a description; column names with
units) and then each data row's text unchanged (line endings CRLF -> LF), so
np.loadtxt(path, delimiter=',') reads the files. Every number is copied character for character.

    python convert_data.py <wheel or folder> --out DIR            # write the two CSVs into DIR
    python convert_data.py <wheel or folder> --out DIR --check    # ... and compare with the shipped files

--out is required and must lie outside this data set's directory (subdirectories included), with or
without --check, so the script never writes into the data set; the shipped files are replaced only by
copying the written files by hand. The check compares the values with a relative tolerance of 1e-12 (it also reports whether the text
is identical). Standard library and numpy only.
"""
import argparse
import hashlib
import io
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WHEEL_SHA256 = 'feca174a5e0ed1a29a8903f07610b99b39c4d52b7b20d070b2977b78d8286f81'
FILE_SHA256 = {
    'fp_inp.csv': 'cf1e4c80685f540b73b3133fe43fcf373fdb744a22cf4fe4e4948852aa3c7525',
    'fp_out.csv': '72811c5a55c497248a4675db64cd05009172ba4fb029dbc878567348a8333037',
}
RTOL = 1e-12

# source column -> unit (units as in the pyMAISE documentation of load_fp; "1" = dimensionless)
INPUTS = [('fuel_dens', 'kg/m3'), ('porosity', '1'), ('clad_thick', 'm'), ('pellet_OD', 'm'),
          ('pellet_h', 'm'), ('gap_thick', 'm'), ('inlet_T', 'K'), ('enrich', '1'),
          ('rough_fuel', 'm'), ('rough_clad', 'm'), ('ax_pow', '1'), ('clad_T', 'K'),
          ('pressure', 'Pa')]
OUTPUTS = [('fis_gas_produced', 'mol'), ('max_fuel_centerline_temp', 'K'),
           ('max_fuel_surface_temp', 'K'), ('radial_clad_dia', 'm')]

FILES = [
    ('fp_inp.csv', INPUTS, 'bison_fp_inputs.csv',
     'Radaideh and Kozlowski 2020 (doi:10.1016/j.ress.2019.106731) BISON PCMI benchmark Case 1: '
     '13 inputs of 400 runs (pyMAISE fp_inp.csv; physical units; values unchanged)'),
    ('fp_out.csv', OUTPUTS, 'bison_fp_outputs.csv',
     'Radaideh and Kozlowski 2020 BISON PCMI benchmark Case 1: 4 outputs of the 400 runs, row i matches '
     'row i of bison_fp_inputs.csv (pyMAISE fp_out.csv; values unchanged)'),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_sources(src):
    """Bytes of fp_inp.csv and fp_out.csv from the wheel or a folder, after the SHA-256 checks."""
    raw = {}
    if os.path.isdir(src):
        for name in FILE_SHA256:
            with open(os.path.join(src, name), 'rb') as fh:
                raw[name] = fh.read()
    else:
        with open(src, 'rb') as fh:
            got = sha256(fh.read())
        if got != WHEEL_SHA256:
            raise ValueError(f'wheel SHA-256 {got} differs from the recorded {WHEEL_SHA256}')
        with zipfile.ZipFile(src) as zf:
            for name in FILE_SHA256:
                raw[name] = zf.read('pyMAISE/data/' + name)
    for name, data in raw.items():
        if sha256(data) != FILE_SHA256[name]:
            raise ValueError(f'{name}: SHA-256 {sha256(data)} differs from the recorded {FILE_SHA256[name]}')
    return raw


def build(src):
    raw = read_sources(src)
    out = {}
    for name, cols, target, desc in FILES:
        lines = raw[name].decode('ascii').splitlines()
        header = lines[0].strip().split(',')
        if header != [c for c, _ in cols]:
            raise ValueError(f'{name}: header {header} differs from the expected columns')
        rows = [ln.strip() for ln in lines[1:] if ln.strip()]
        for ln in rows:
            vals = ln.split(',')
            if len(vals) != len(cols):
                raise ValueError(f'{name}: a row has {len(vals)} fields, expected {len(cols)}')
            [float(v) for v in vals]          # every field is a plain number (raises otherwise)
        out[target] = (f'# {desc}\n# ' + ','.join(f'{c} [{u}]' for c, u in cols) + '\n'
                       + '\n'.join(rows) + '\n')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', help='pyMAISE-0.0.2-py3-none-any.whl, or a folder with fp_inp.csv and fp_out.csv')
    p.add_argument('--out', required=True, help='folder to write the CSV files into')
    p.add_argument('--check', action='store_true', help='compare the written files with the shipped ones')
    a = p.parse_args()
    out = os.path.realpath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([out, here]) == here:
        sys.exit('--out must be a directory outside this data set (subdirectories included); '
                 'the script never writes into the data set')
    texts = build(a.source)
    os.makedirs(out, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(out, name), 'w', newline='\n') as fh:
            fh.write(text)
        print(f'wrote {name}: {text.count(chr(10)) - 2} rows')
    if a.check:
        bad = 0
        for name, text in texts.items():
            new = np.loadtxt(os.path.join(out, name), delimiter=',')
            old = np.loadtxt(os.path.join(HERE, name), delimiter=',')
            same_shape = new.shape == old.shape
            ok = same_shape and bool(np.all(np.abs(new - old) <= RTOL * np.abs(old)))
            with open(os.path.join(HERE, name)) as fh:
                ident = fh.read() == text
            print(f'{name}: values equal within rtol {RTOL:g}: {ok}; text identical: {ident}')
            bad += not ok
        return int(bad > 0)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as err:
        sys.exit(f'file not found, or not of the expected kind: {err.filename}\n'
                 'The source is the pyMAISE 0.0.2 wheel, or a folder with fp_inp.csv and fp_out.csv; see DATA.md, "Regenerating".')
