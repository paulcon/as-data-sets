"""Convert probes.npz (equadratures/data-sets) to this repository's CSV format.

Scillitoe, Ubald, Seshadri and Shahpar, "Design Space Exploration of Stagnation Temperature
Probes via Dimension Reduction", Proc. ASME Turbo Expo 2020, GT2020-16277,
doi:10.1115/GT2020-16277, ran 3-D conjugate RANS simulations (ANSYS Fluent) of 128 variants
of a single-shrouded Kiel stagnation-temperature probe, each at six Mach numbers. The
equadratures/data-sets repository publishes the results as one NumPy archive,
probes/probes.npz, whose arrays are (all float64)

    X  (128, 7)   the 7 design parameters, normalized to [-1, 1]
    y1 (128, 1)   loss objective O_Yp (mean stagnation pressure loss), standardized to [-1, 1]
    y2 (128, 1)   recovery objective O_Rr (|dRr/dM|), standardized to [-1, 1]
    y3 (128, 1)   recovery ratio Rr at Mach 0.8, standardized to [-1, 1]

This script writes three CSV files:

    inputs_normalized.csv  128 x 7  the archive's X, unchanged, in the archive's column order
    outputs.csv            128 x 3  y1, y2, y3, unchanged
    inputs.csv             128 x 7  the design parameters in physical units, in the order of
                                    the paper's Table 1, computed from X (see below)

Each file has two '#' lines (a description; the column names with units), then one row per
line, every number written with Python's repr (the shortest decimal that reads back as the
same double). inputs_normalized.csv and outputs.csv read back bit for bit equal to the
archive. Row order is kept.

Physical inputs. The archive does not name its columns. Its column j is Table 1's parameter
COLUMN_MAP[j], and its 5th column (the bleed-hole angle) has the opposite sign to the
paper's normalized coordinate. Both are confirmed by the authors' own fitted subspaces in their
companion repository (github.com/ascillitoe/probe-subspaces, commit 072a0b0,
Example_Notebooks/Yp_subspace.joblib and Rr_subspace.joblib, whose 7 x 7 subspace matrices were
read from the raw bytes, never unpickled): under this map their loss direction equals Eq. 12
(cosine 0.99997) and their recovery subspace gives the vectors of Fig. 11. The code's baseline
vector (dim_red.py, line 28) is Table 1's x0 under the same map. Details in DATA.md;
check_accuracy.py recomputes the subspace numbers. Eq. 4 as intended is
x = (2 x_phys - x_U - x_L) / (x_U - x_L); the paper prints the numerator as 2 x_phys - (x_U - x_L),
a misprint that does not map Table 1 to [-1, 1]. With x the normalized value in the paper's
convention, Eq. 4 inverted gives the physical value
    x_phys = x_L + (x + 1) (x_U - x_L) / 2
with x_L, x_U from Table 1 (LOWER, UPPER). The physical values are deltas or scale factors
relative to the proprietary baseline design x0 (NOMINAL).

Usage:
    python convert_data.py --npz /path/to/probes.npz           # write the CSVs
    python convert_data.py --npz /path/to/probes.npz --check   # compare with the shipped CSVs
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SOURCE_URL = 'https://raw.githubusercontent.com/equadratures/data-sets/main/probes/probes.npz'
SOURCE_COMMIT = '62e0a429adf03df417811a0252ea4127178a5a1c'
SOURCE_SHA256 = '3b9766c5956d9b29a7f7865c1dfe9ae0d5ea42ed08b103c1bb45855abc6a1fd4'

M, m = 128, 7
# Table 1 (p. 3), in its order: name, description, units, x_L, x_0, x_U
TABLE1 = (
    ('hole_ellipse', 'ellipsoidal bleed hole (scale)', '-', 0.50, 1.00, 1.25),
    ('hole_move', 'move bleed hole (translate)', 'mm', -0.3, 0.0, 0.0),
    ('hole_angle', 'angle bleed hole (translate inner opening)', 'mm', 0.0, 0.0, 1.0),
    ('kiel_le', 'Kiel leading edge (translate)', 'mm', -2.0, 0.0, 2.0),
    ('kiel_outer', 'Kiel outer diameter (scale)', '-', 0.90, 1.00, 1.25),
    ('kiel_inner', 'Kiel inner diameter (scale)', '-', 0.75, 1.00, 1.10),
    ('hole_diameter', 'bleed hole diameter (scale)', '-', 0.50, 1.00, 1.50),
)
NAMES = tuple(t[0] for t in TABLE1)
DESCRIPTIONS = tuple(t[1] for t in TABLE1)
UNITS = tuple(t[2] for t in TABLE1)
LOWER = np.array([t[3] for t in TABLE1])
NOMINAL = np.array([t[4] for t in TABLE1])
UPPER = np.array([t[5] for t in TABLE1])

# archive column j (0-based) holds Table 1 parameter COLUMN_MAP[j] times SIGN[j]
COLUMN_MAP = (6, 5, 4, 3, 2, 1, 0)       # the archive's columns are Table 1 in reverse order
SIGN = (1, 1, 1, 1, -1, 1, 1)            # the archive's hole-angle column is negated

OUTPUTS = (('Yp_std', 'loss objective O_Yp, Eq. 2, standardized', 'y1'),
           ('ORr_std', 'recovery objective O_Rr = |dRr/dM|, Eq. 3, standardized', 'y2'),
           ('Rr08_std', 'recovery ratio Rr at Mach 0.8, standardized', 'y3'))

CREDIT = ('CFD (ANSYS Fluent RANS) of 128 Kiel stagnation temperature probe designs from Scillitoe, Ubald, '
          'Seshadri and Shahpar, ASME GT2020-16277, doi:10.1115/GT2020-16277; published in '
          'equadratures/data-sets probes/probes.npz (sha256 ' + SOURCE_SHA256[:12] + '...), no licence stated')


def to_paper_normalized(X_archive):
    """Archive X -> normalized inputs in Table 1 order and the paper's sign convention."""
    Z = np.empty_like(X_archive)
    for j, (k, s) in enumerate(zip(COLUMN_MAP, SIGN)):
        Z[:, k] = s * X_archive[:, j]
    return Z


def to_physical(Z):
    """Eq. 4 (as intended) inverted: normalized (Table 1 order) -> physical."""
    return LOWER + (Z + 1.0) * (UPPER - LOWER) / 2.0


def to_normalized(Xphys):
    """Eq. 4 as intended, (2 x - x_U - x_L) / (x_U - x_L): physical -> normalized (Table 1 order)."""
    return (2.0 * Xphys - UPPER - LOWER) / (UPPER - LOWER)


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
    """Return ({file name: text}, max |Eq. 4 round trip error|) after checking the archive."""
    digest = sha256(npz_path)
    if digest != SOURCE_SHA256:
        sys.exit(f'{npz_path}: sha256 {digest} is not the expected {SOURCE_SHA256}')
    with np.load(npz_path, allow_pickle=False) as d:
        if sorted(d.files) != ['X', 'y1', 'y2', 'y3']:
            sys.exit(f'unexpected keys {sorted(d.files)}')
        X = d['X']
        Y = np.column_stack([d[k][:, 0] for k in ('y1', 'y2', 'y3')])
        for k in d.files:
            want = (M, m) if k == 'X' else (M, 1)
            if d[k].dtype != np.float64 or d[k].shape != want:
                sys.exit(f'{k}: expected float64 {want}, got {d[k].dtype} {d[k].shape}')
    Xphys = to_physical(to_paper_normalized(X))
    texts, arrays = {}, {}
    arrays['inputs_normalized.csv'] = (
        X, 'the archive X verbatim (design parameters normalized to [-1,1]); archive column order = Table 1 '
           'parameters 7,6,5,4,3,2,1 with column 5 (hole angle) of opposite sign to the paper '
           "(confirmed by the authors' fitted subspaces in ascillitoe/probe-subspaces@072a0b0, see DATA.md); "
           + CREDIT,
        [f'archive_x{j + 1} (-)' for j in range(m)])
    arrays['outputs.csv'] = (
        Y, 'the archive y1, y2, y3 verbatim, each standardized by the authors to [-1,1] (physical values not '
           'published); ' + CREDIT,
        [f'{o[0]} (-)' for o in OUTPUTS])
    arrays['inputs.csv'] = (
        Xphys, 'design parameters in physical units (deltas or scale factors relative to the proprietary baseline, '
               'Table 1 order), computed from the archive X by Eq. 4 inverted with Table 1 bounds; ' + CREDIT,
        [f'{n} ({u})' for n, u in zip(NAMES, UNITS)])
    for name, (a, desc, cols) in arrays.items():
        text = to_text(a, desc, cols)
        back = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2)
        if not same_bits(np.ascontiguousarray(a), back):
            sys.exit(f'{name}: the text does not read back bit for bit')
        texts[name] = text
    # the physical inputs map back to the archive X (Eq. 4, then the column map) to rounding
    Zback = to_normalized(np.loadtxt(io.StringIO(texts['inputs.csv']), delimiter=','))
    Xback = np.column_stack([s * Zback[:, k] for k, s in zip(COLUMN_MAP, SIGN)])
    err = float(np.abs(Xback - X).max())
    if err > 1e-14:
        sys.exit(f'Eq. 4 round trip error {err:.2e} is larger than 1e-14')
    if np.any(Xphys < LOWER - 1e-12) or np.any(Xphys > UPPER + 1e-12):
        sys.exit('a physical input lies outside the Table 1 bounds')
    return texts, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--npz', required=True, help='path to probes.npz (' + SOURCE_URL + ')')
    ap.add_argument('--check', action='store_true', help='compare with the shipped CSVs, write nothing')
    args = ap.parse_args()
    texts, err = build(args.npz)
    for name, text in texts.items():
        path = HERE / name
        if args.check:
            ok = path.exists() and path.read_text(encoding='ascii') == text
            print(f'{name}: {"identical" if ok else "DIFFERS"}')
            if not ok:
                sys.exit(1)
        else:
            path.write_text(text, encoding='ascii')
            print(f'wrote {name}: {len(text.splitlines()) - 2} rows, {path.stat().st_size} bytes')
    print('inputs_normalized.csv and outputs.csv read back bit for bit equal to the archive')
    print(f'inputs.csv maps back to the archive X with max |error| {err:.1e}')


if __name__ == '__main__':
    main()
