"""Copy the outlet cell row of the deposited CFD fields of Piscopo et al. (2026) and reduce it to outlet emissions.

Stored data: the 175 RANS solutions (Ansys Fluent 22.1, C-type staged combustor) were computed by the
authors and deposited at Zenodo, doi:10.5281/zenodo.19348021 (v2.0, CC BY 4.0), as three NumPy files:
``P.npy`` (175 x 3 operating conditions), ``xy.npy`` (60231 x 2 cell-centre coordinates, axial z and
radius r in m) and ``X.npy`` (903465 x 175: 15 fields stacked by feature, one column per case). This
script checks their SHA-256, memory-maps ``X.npy`` and writes two files:

* ``h2nh3_outlet_cells.npz``: a lossless copy of the deposited values the reduction uses. ``P`` (175 x 3,
  from P.npy), ``cell`` (37 row indices into xy.npy), ``z`` and ``r`` (37, from xy.npy), ``features`` (the
  15 field names) and ``fields`` (15 x 37 x 175, the X.npy values of those cells for every case), all
  float64 as deposited.
* ``h2nh3_outlet.csv``: one row per case, computed from the cell values above by the formula below. The
  operating conditions are copied unchanged (written as the shortest decimal string that reads back as
  the same float64, Python ``repr``). Nothing else is fitted or changed.

Outlet formula (this data set's reconstruction; the paper says only that the outlet quantities "can be
computed from a surface integral at the outlet from the field files", p. 6):

1. Outlet cells: the last row of cells below the outlet plane, i.e. every cell whose centre lies within
   0.5 mm of the highest cell centre (z_c = 0.64541-0.64542 m, 37 cells from the axis to the outlet
   wall; the row below is 1 mm lower). Their cell-centre values stand in for the outlet face values.
2. Face areas: the radial face edges are the midpoints between neighbouring centres, the inner edge is
   the axis (r = 0), and the outer edge is the wall, placed as far beyond the last centre as the last
   centre lies beyond its inner edge (r_wall = 2 r_37 - e_37 = 32.58 mm). Annulus k has area
   A_k = pi (e_{k+1}^2 - e_k^2) (axisymmetric, full 2 pi).
3. Weights: mass flux through each annulus, w_k = rho_k v_z,k A_k (v_z > 0 in every outlet cell of every
   case); the outlet mass flow is sum_k w_k. Mass-flux-weighted averages
   Xbar = sum_k w_k X_k / sum_k w_k of the mole fractions of NO, NO2, NH3, O2, H2O and of T.
4. Dry basis and 16 % O2: with Xbar_dry = Xbar / (1 - Xbar_H2O),
   C_16 = 1e6 Xbar_dry (21 - 16) / (21 - 100 Xbar_O2,dry)   [ppmv, dry, at 16 % O2].
   The reference 21 % is the O2 mole fraction of the simulated inlet air (0.21/0.79 in the fields; see
   check_accuracy.py). NOx = NO + NO2 (N2O is not among the deposited fields).
5. For comparison, the area-weighted NOx and T (w_k = A_k) are written too.

    python extract_data.py <folder with P.npy, xy.npy, X.npy> --out DIR          # write both files to DIR
    python extract_data.py <folder> --out DIR --check   # write to DIR, compare both with the shipped files
    python extract_data.py --cells h2nh3_outlet_cells.npz --out DIR --check
                        # rebuild the CSV from the shipped cell values alone (no download), compare

``--out`` is required (nothing is written by default) and must be outside this directory (not the
directory or a folder in it). ``--check`` compares the cell values at a relative tolerance of 1e-15, the CSV's input columns as
text and its outlet columns at a relative tolerance of 1e-9. Needs numpy only; reads about 4 MB of the
1.26 GB ``X.npy`` and hashes all three deposited files (a few seconds).
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = 'h2nh3_outlet.csv'
CELLFILE = 'h2nh3_outlet_cells.npz'
RTOL = 1e-9          # CSV outlet columns (written with 12 significant digits)
RTOL_CELLS = 1e-15   # copied cell values

SHA256 = {
    'P.npy': 'e7ba58b83cb6937fed3604ebbfc11e3133581684803cfdaa7ae0975731861c9b',
    'xy.npy': 'b5857e4bb34abbc8cadadec669733d39cd8ad330998a5095ba80a8bd37780f77',
    'X.npy': '1ec080ded6c9fe879b1cef334f2d438679c046dbf51b9b1f2ec4baec3e9934ef',
}

# Field order of the rows of X.npy (README.ipynb of the deposit). Species: mole fractions;
# rho kg/m3; p gauge pressure Pa; velocities m/s (v_z axial, v_r radial); T K.
FEATURES = ['N2', 'NH2', 'NO2', 'H2O', 'OH', 'O2', 'H2', 'NH3', 'NO', 'rho', 'p', 'vmag', 'vr', 'vz', 'T']
O2_AIR = 21.0      # % O2 of the simulated (dry) inlet air
O2_REF = 16.0      # % O2 reference of the paper's Table 2

COLUMNS = [
    ('case', '1', 'case number (column of X.npy plus one)'),
    ('phi1', '1', 'rich-stage equivalence ratio (P.npy column 1)'),
    ('xH2', 'mol/mol', 'H2 mole fraction in the fuel (P.npy column 2)'),
    ('p0', 'Pa', 'operating pressure (P.npy column 3)'),
    ('dup_of', '1', 'case number of an earlier case with identical inputs, else 0'),
    ('NOx', 'ppmvd@16%O2', 'outlet NO + NO2, mass-flux weighted'),
    ('NO', 'ppmvd@16%O2', 'outlet NO, mass-flux weighted'),
    ('NO2', 'ppmvd@16%O2', 'outlet NO2, mass-flux weighted'),
    ('NH3', 'ppmvd@16%O2', 'outlet NH3 slip, mass-flux weighted'),
    ('O2_dry', '%', 'outlet O2, dry, mass-flux weighted'),
    ('H2O', 'mol/mol', 'outlet H2O (wet), mass-flux weighted'),
    ('T_out', 'K', 'outlet temperature, mass-flux weighted'),
    ('NOx_area', 'ppmvd@16%O2', 'outlet NO + NO2, area weighted'),
    ('T_out_area', 'K', 'outlet temperature, area weighted'),
    ('mdot', 'kg/s', 'outlet mass flow'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def load(folder, verify=True):
    """Return P, xy and the memory-mapped X of the deposit, after the SHA-256 check."""
    for name, digest in SHA256.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            print(f'missing {name} in the given folder: download Zenodo record 19348021 '
                  '(doi:10.5281/zenodo.19348021) and rerun', file=sys.stderr)
            sys.exit(3)
        if verify and sha256(path) != digest:
            sys.exit(f'{name}: SHA-256 differs from the deposited file')
    P = np.load(os.path.join(folder, 'P.npy'))
    xy = np.load(os.path.join(folder, 'xy.npy'))
    X = np.load(os.path.join(folder, 'X.npy'), mmap_mode='r')
    if X.shape != (len(FEATURES) * xy.shape[0], P.shape[0]):
        sys.exit(f'unexpected shapes P {P.shape}, xy {xy.shape}, X {X.shape}')
    return P, xy, X


def outlet_cells(xy, offset=0.0):
    """Indices (sorted by radius) of the cell row whose centres lie within 0.5 mm of z_max - offset."""
    z, r = xy[:, 0], xy[:, 1]
    idx = np.where(np.abs(z - (z.max() - offset)) < 0.5e-3)[0]
    return idx[np.argsort(r[idx])]


def face_edges(rc, rule='midpoint'):
    """Radial face edges of a row of cells with centres rc (axis at 0)."""
    if rule == 'midpoint':
        e = np.r_[0.0, 0.5 * (rc[1:] + rc[:-1])]
        return np.r_[e, 2 * rc[-1] - e[-1]]
    e = [0.0]                      # 'centred': every centre midway between its two edges
    for c in rc:
        e.append(2 * c - e[-1])
    return np.array(e)


def outlet_fields(X, idx, n_cells):
    """Dict feature -> (len(idx), n_cases) array of the outlet cell values."""
    return {f: np.asarray(X[[j * n_cells + c for c in idx], :]) for j, f in enumerate(FEATURES)}


def averages(G, w):
    """Weighted averages over the outlet cells (weights w, shape (cells, cases))."""
    return {f: (G[f] * w).sum(0) / w.sum(0) for f in G}


def corrected(av, species):
    """ppmv, dry basis, at 16 % O2, of the summed mole fractions of `species`."""
    dry = 1.0 - av['H2O']
    o2d = 100.0 * av['O2'] / dry
    return 1e6 * sum(av[s] for s in species) / dry * (O2_AIR - O2_REF) / (O2_AIR - o2d)


def reduce_cells(r, G, edges='midpoint', weight='mass'):
    """The outlet quantities of every case from the outlet cell row (centres r, fields G)."""
    e = face_edges(r, edges)
    A = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    n_cases = G['T'].shape[1]
    if np.any(G['vz'] <= 0):
        raise ValueError('reverse flow at the outlet row')
    mflux = G['rho'] * G['vz'] * A[:, None]
    w = {'mass': mflux, 'area': np.repeat(A[:, None], n_cases, axis=1)}[weight]
    av = averages(G, w)
    area = averages(G, np.repeat(A[:, None], n_cases, axis=1))
    out = {
        'NOx': corrected(av, ['NO', 'NO2']), 'NO': corrected(av, ['NO']), 'NO2': corrected(av, ['NO2']),
        'NH3': corrected(av, ['NH3']), 'O2_dry': 100.0 * av['O2'] / (1.0 - av['H2O']),
        'H2O': av['H2O'], 'T_out': av['T'], 'NOx_area': corrected(area, ['NO', 'NO2']),
        'T_out_area': area['T'], 'mdot': mflux.sum(0),
    }
    out['_n_cells'] = len(r)
    out['_r_wall'] = e[-1]
    return out


def reduce_cases(P, xy, X, offset=0.0, edges='midpoint', weight='mass'):
    """The outlet quantities of every case, read from the deposit (dict name -> (n_cases,) array)."""
    idx = outlet_cells(xy, offset)
    return reduce_cells(xy[idx, 1], outlet_fields(X, idx, xy.shape[0]), edges, weight)


def cells_from_deposit(P, xy, X):
    """The arrays of h2nh3_outlet_cells.npz: deposited values, copied without change."""
    idx = outlet_cells(xy)
    if len(idx) != 37:
        raise ValueError(f'expected 37 outlet cells, found {len(idx)}')
    G = outlet_fields(X, idx, xy.shape[0])
    return {'P': np.array(P, dtype=np.float64), 'cell': idx.astype(np.int64),
            'z': xy[idx, 0].copy(), 'r': xy[idx, 1].copy(), 'features': np.array(FEATURES),
            'fields': np.stack([G[f] for f in FEATURES])}


def load_cells(path):
    """Read h2nh3_outlet_cells.npz; return its arrays and the fields as a dict feature -> (37, 175)."""
    with np.load(path, allow_pickle=False) as d:
        cells = {k: d[k] for k in d.files}
    if list(cells['features']) != FEATURES:
        sys.exit(f'{path}: unexpected field names {list(cells["features"])}')
    G = {f: cells['fields'][j] for j, f in enumerate(FEATURES)}
    return cells, G


def duplicates(P):
    dup = np.zeros(P.shape[0], dtype=int)
    for i in range(P.shape[0]):
        same = np.where(np.all(P[:i] == P[i], axis=1))[0]
        if len(same):
            dup[i] = same[0] + 1
    return dup


def build_text(cells):
    P = cells['P']
    q = reduce_cells(cells['r'], {f: cells['fields'][j] for j, f in enumerate(FEATURES)})
    dup = duplicates(P)
    head = ('# Piscopo et al. 2026 H2/NH3 staged combustor (C-type), 175 RANS cases (Zenodo '
            '10.5281/zenodo.19348021): operating conditions copied from P.npy; outlet quantities '
            'reduced by extract_data.py from the outlet cell row of X.npy (h2nh3_outlet_cells.npz; '
            'mass-flux weighted; ppmv dry at 16% O2)')
    names = ','.join(f'{c} [{u}]' for c, u, _ in COLUMNS)
    lines = [head, '# ' + names]
    for i in range(P.shape[0]):
        vals = [str(i + 1), repr(float(P[i, 0])), repr(float(P[i, 1])), repr(float(P[i, 2])), str(dup[i])]
        vals += ['%.12g' % q[c][i] for c, _, _ in COLUMNS[5:]]
        lines.append(','.join(vals))
    return '\n'.join(lines) + '\n', q


def check_csv(path):
    new = np.loadtxt(path, delimiter=',')
    old = np.loadtxt(os.path.join(HERE, OUTFILE), delimiter=',')
    with open(path) as f1, open(os.path.join(HERE, OUTFILE)) as f2:
        t1 = [ln.split(',')[:5] for ln in f1.read().splitlines()[2:]]
        t2 = [ln.split(',')[:5] for ln in f2.read().splitlines()[2:]]
    same_inputs = t1 == t2
    rel = np.max(np.abs(new[:, 5:] - old[:, 5:]) / np.maximum(np.abs(old[:, 5:]), 1e-300))
    ok = same_inputs and new.shape == old.shape and rel <= RTOL
    print(f'{OUTFILE}: inputs identical as text: {same_inputs}; largest relative difference of the outlet '
          f'columns {rel:.1e} (tolerance {RTOL:.0e}): {"agree" if ok else "DIFFER"}')
    return ok


def check_cells(path):
    new, _ = load_cells(path)
    old, _ = load_cells(os.path.join(HERE, CELLFILE))
    ok = sorted(new) == sorted(old)
    worst = 0.0
    for k in new:
        if k == 'features' or not ok:
            continue
        a, b = np.asarray(new[k], dtype=np.float64), np.asarray(old[k], dtype=np.float64)
        if a.shape != b.shape:
            ok = False
            continue
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))))
    ok = ok and worst <= RTOL_CELLS
    print(f'{CELLFILE}: largest relative difference of the copied values {worst:.1e} '
          f'(tolerance {RTOL_CELLS:.0e}): {"agree" if ok else "DIFFER"}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('folder', nargs='?', help='folder holding the deposited P.npy, xy.npy and X.npy')
    ap.add_argument('--cells', help=f'rebuild the CSV from this {CELLFILE} instead of the deposit')
    ap.add_argument('--out', required=True, help='directory to write the files into')
    ap.add_argument('--check', action='store_true', help='compare the written files with the shipped ones')
    a = ap.parse_args()
    if (a.folder is None) == (a.cells is None):
        sys.exit('give either the deposit folder or --cells FILE')
    out = os.path.realpath(a.out)
    inside = os.path.commonpath([out, os.path.realpath(HERE)]) == os.path.realpath(HERE)
    if inside:
        sys.exit("--out must be outside this data set's directory (not the directory or a folder in it)")
    os.makedirs(out, exist_ok=True)
    if a.cells is not None:
        if not os.path.isfile(a.cells):
            sys.exit(f'missing {a.cells}')
        cells, _ = load_cells(a.cells)
    else:
        P, xy, X = load(a.folder)
        cells = cells_from_deposit(P, xy, X)
        np.savez(os.path.join(out, CELLFILE), **cells)
        print(f'wrote {CELLFILE}: {cells["fields"].shape[0]} fields x {cells["fields"].shape[1]} cells x '
              f'{cells["fields"].shape[2]} cases')
    text, q = build_text(cells)
    path = os.path.join(out, OUTFILE)
    with open(path, 'w') as fh:
        fh.write(text)
    print(f'wrote {OUTFILE}: {cells["P"].shape[0]} cases, {q["_n_cells"]} outlet cells, '
          f'outlet wall radius {1e3 * q["_r_wall"]:.2f} mm')
    if a.check:
        ok = check_csv(path)
        if a.cells is None:
            ok = check_cells(os.path.join(out, CELLFILE)) and ok
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
