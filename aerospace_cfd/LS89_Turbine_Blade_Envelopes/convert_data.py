"""Convert blade_envelopes.npz (equadratures/data-sets) to this repository's CSV format.

Wong, Seshadri, Scillitoe, Duncan and Parks, "Blade Envelopes Part I: Concept and
Methodology", J. Turbomach. 144(6), 2022, doi:10.1115/1.4053239, ran SU2 (2-D RANS,
Spalart-Allmaras) on 1000 FFD-deformed VKI LS89 turbine blades. The runs are published
as one NumPy archive in https://github.com/equadratures/data-sets
(blade_envelopes/blade_envelopes.npz). Its arrays are

    X                 (1000, 20)  FFD design variables, scaled to [-1, 1]
    loss              (1000,)     stagnation pressure loss coefficient Yp (3 entries are inf)
    x_coords          (240,)      blade-surface x coordinates, m (the same for every design)
    y_coords_random   (1000, 240) blade-surface y coordinates of the 1000 designs, m
    train_y_coords    (5000, 240) y coordinates of the H = 5000 inactive-subspace samples
    y_coords_loss     (500, 240)  y coordinates of a further 500 designs (no loss values)

This script copies the first four, which the notebook uses, into four CSV files:

    inputs.csv    1000 x 20    X
    outputs.csv   1000 x 1     loss (inf kept)
    blade_x.csv   240 x 1      x_coords
    blade_y.csv   1000 x 240   y_coords_random

Each file has two '#' lines (a description; the column names with units), then one row
per line, every number written with Python's repr (the shortest decimal that reads back
as the same double; 'inf' for infinity). The conversion is lossless: the script reads
each file back with np.loadtxt and checks it equals the source array bit for bit.
The last two arrays (train_y_coords, y_coords_loss) are not used and not copied.

Usage:
    python convert_data.py --npz /path/to/blade_envelopes.npz           # write the CSVs
    python convert_data.py --npz /path/to/blade_envelopes.npz --check   # compare with the shipped CSVs
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SOURCE_URL = ('https://raw.githubusercontent.com/equadratures/data-sets/main/'
              'blade_envelopes/blade_envelopes.npz')
SOURCE_SHA256 = '519dc43c8125a98ee5a2fadf6329eeef2edc4d68382aad2b8994929bae870395'

M, m, N = 1000, 20, 240
NAMES = tuple(f'x{j}' for j in range(1, m + 1))      # FFD design variables, scaled
LOWER = -np.ones(m)
UPPER = np.ones(m)

CREDIT = ('SU2 RANS runs of Wong et al., J. Turbomach. 144(6) 2022, doi:10.1115/1.4053239, from '
          'equadratures/data-sets blade_envelopes.npz (sha256 ' + SOURCE_SHA256[:12] + '...)')

FILES = {
    'inputs.csv': ('X', (M, m),
                   f'FFD design variables of the 1000 LS89 designs, scaled to [-1,1]; {CREDIT}',
                   [f'{n} (-)' for n in NAMES]),
    'outputs.csv': ('loss', (M,),
                    f'stagnation pressure loss coefficient Yp of the 1000 designs (inf = no value); {CREDIT}',
                    ['Yp (-)']),
    'blade_x.csv': ('x_coords', (N,),
                    f'x coordinates of the 240 blade-surface points, the same for every design; {CREDIT}',
                    ['x (m)']),
    'blade_y.csv': ('y_coords_random', (M, N),
                    f'y coordinates of the 240 blade-surface points of the 1000 designs; {CREDIT}',
                    [f'y{k:03d} (m)' for k in range(1, N + 1)]),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def to_text(a, description, columns):
    """The CSV text of a 1-D or 2-D float array, every value written with repr."""
    a2 = a.reshape(len(a), -1)
    out = io.StringIO()
    out.write(f'# {description}\n# {",".join(columns)}\n')
    for row in a2:
        out.write(','.join(repr(float(v)) for v in row) + '\n')
    return out.getvalue()


def same_bits(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def build(npz_path):
    """Return {file name: text} built from the npz, after checking its hash and arrays."""
    digest = sha256(npz_path)
    if digest != SOURCE_SHA256:
        sys.exit(f'{npz_path}: sha256 {digest} is not the expected {SOURCE_SHA256}')
    texts = {}
    with np.load(npz_path, allow_pickle=False) as d:
        for name, (key, shape, description, columns) in FILES.items():
            a = np.ascontiguousarray(d[key], dtype=np.float64)
            if a.shape != shape or d[key].dtype != np.float64:
                sys.exit(f'{key}: expected float64 {shape}, got {d[key].dtype} {a.shape}')
            text = to_text(a, description, columns)
            back = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2).reshape(shape)
            if not same_bits(a, back):
                sys.exit(f'{name}: the text does not read back bit for bit')
            texts[name] = text
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--npz', required=True, help='path to blade_envelopes.npz (' + SOURCE_URL + ')')
    ap.add_argument('--check', action='store_true', help='compare with the shipped CSVs, write nothing')
    args = ap.parse_args()
    texts = build(args.npz)
    for name, text in texts.items():
        path = HERE / name
        if args.check:
            ok = path.exists() and path.read_text(encoding='ascii') == text
            print(f'{name}: {"identical" if ok else "DIFFERS"}')
            if not ok:
                sys.exit(1)
        else:
            path.write_text(text, encoding='ascii')
            print(f'wrote {name}: {len(text.splitlines()) - 2} rows, {path.stat().st_size / 1e6:.2f} MB')
    print('every array reads back bit for bit')


if __name__ == '__main__':
    main()
