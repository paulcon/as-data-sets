"""Convert 3Dfan_blades.npz (equadratures/data-sets) to this repository's CSV format.

Seshadri, Shahpar, Constantine, Parks and Adams, "Turbomachinery Active Subspace
Performance Maps", J. Turbomach. 140(4):041003, 2018, doi:10.1115/1.4038839, ran the
Rolls-Royce RANS solver Hydra on PADRAM-perturbed variants of a transonic research fan
blade with 25 design variables. The equadratures/data-sets repository publishes CFD
results for three Rolls-Royce research fan blades with this parameterization as one NumPy
archive, 3Dfan_blades/3Dfan_blades.npz, whose arrays are (all float64)

    X_a  (548, 25)   design variables of blade A, normalized to [-1, 1]
    y1_a (548,)      non-dimensional efficiency of blade A
    y2_a (548,)      non-dimensional pressure ratio of blade A
    X_b  (311, 25), y1_b (311,), y2_b (311,)   the same for blade B
    X_c  (547, 25), y1_c (547,), y2_c (547,)   the same for blade C

This script writes one CSV per blade, with the 25 inputs followed by the two outputs:

    blade_A.csv   548 x 27
    blade_B.csv   311 x 27
    blade_C.csv   547 x 27

Each file has two '#' lines (a description; the column names with units), then one row
per line, every number written with Python's repr (the shortest decimal that reads back
as the same double). The conversion is lossless: the script reads each file back with
np.loadtxt and checks it equals the source arrays bit for bit. Row order is kept.

Column order. The archive does not name its columns. We use the order of the paper's
eigenvector bar charts (Figs. 8(b), 13(b), 22(b)): dihedral, leading-edge recamber,
trailing-edge recamber, sweep, skew, each at 0, 25, 50, 75 and 100 % span. The active
subspaces computed from blade A in this order agree with those bar charts (see DATA.md).

Usage:
    python convert_data.py --npz /path/to/3Dfan_blades.npz           # write the CSVs
    python convert_data.py --npz /path/to/3Dfan_blades.npz --check   # compare with the shipped CSVs
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SOURCE_URL = ('https://raw.githubusercontent.com/equadratures/data-sets/main/'
              '3Dfan_blades/3Dfan_blades.npz')
SOURCE_COMMIT = '62e0a429adf03df417811a0252ea4127178a5a1c'
SOURCE_SHA256 = '9fe1e837be536c6777877b6d7bc4416d4a06b04ffd0bd9a868bff325ac60fe29'

m = 25
DOFS = ('dihedral', 'le_recamber', 'te_recamber', 'sweep', 'skew')
SPANS = (0, 25, 50, 75, 100)
NAMES = tuple(f'{dof}_{s:03d}' for dof in DOFS for s in SPANS)   # e.g. 'le_recamber_050'
LOWER = -np.ones(m)          # the normalized bounds; the physical bounds are not published
UPPER = np.ones(m)
OUTPUTS = ('eta_nd', 'pr_nd')
BLADES = {'A': 548, 'B': 311, 'C': 547}

CREDIT = ('CFD of three Rolls-Royce research fan blades used in Seshadri et al., J. Turbomach. '
          '140(4) 2018, doi:10.1115/1.4038839 and Aeronaut. J. 124(1279) 2020, doi:10.1017/aer.2020.50; '
          'shared with permission from Rolls-Royce plc in equadratures/data-sets 3Dfan_blades.npz '
          '(sha256 ' + SOURCE_SHA256[:12] + '...)')


def columns():
    return [f'{n} (-)' for n in NAMES] + [f'{o} (-)' for o in OUTPUTS]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def to_text(a, description, cols):
    out = io.StringIO()
    out.write(f'# {description}\n# {",".join(cols)}\n')
    for row in a:
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
        expected = {f'{p}_{b.lower()}' for b in BLADES for p in ('X', 'y1', 'y2')}
        if set(d.files) != expected:
            sys.exit(f'unexpected keys {sorted(d.files)}')
        for b, M in BLADES.items():
            s = b.lower()
            X, y1, y2 = d[f'X_{s}'], d[f'y1_{s}'], d[f'y2_{s}']
            for name, a, shape in [('X', X, (M, m)), ('y1', y1, (M,)), ('y2', y2, (M,))]:
                if a.dtype != np.float64 or a.shape != shape:
                    sys.exit(f'{name}_{s}: expected float64 {shape}, got {a.dtype} {a.shape}')
            a = np.column_stack([X, y1, y2])
            desc = (f'blade {b}: {M} designs; columns 1-25 the PADRAM design variables normalized to '
                    f'[-1,1], 26 non-dimensional efficiency (npz y1_{s}), 27 non-dimensional pressure '
                    f'ratio (npz y2_{s}); {CREDIT}')
            text = to_text(a, desc, columns())
            back = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2)
            if not same_bits(np.ascontiguousarray(a), back):
                sys.exit(f'blade {b}: the text does not read back bit for bit')
            texts[f'blade_{b}.csv'] = text
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--npz', required=True, help='path to 3Dfan_blades.npz (' + SOURCE_URL + ')')
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
