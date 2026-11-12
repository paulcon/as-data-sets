"""Recompute the accuracy numbers of DATA.md that neither the notebook nor extract_data.py prints.

    python check_accuracy.py <folder with the deposited P.npy, xy.npy, X.npy>

Needs the Zenodo deposit (doi:10.5281/zenodo.19348021; X.npy is 1.26 GB) and numpy; without the
files the script stops with an error that says what to download. It writes nothing. Sections 1 and 2
read the outlet rows and three whole fields of X.npy (about 250 MB) through a memory map; section 3
needs two full columns of X.npy, and because the file is stored row by row it reads the whole 1.26 GB
file once, in blocks of about 28 MB. Peak memory is about 320 MB; run time about 3 s with the
file in the page cache (longer from a cold disk). It prints:

1. How much the outlet NOx and temperature change with the choices of the outlet reduction:
   area or molar-flux weighting instead of mass-flux weighting; the cell row 1 mm upstream of the
   outlet row; face edges with every centre midway between its edges instead of midpoint edges;
   a 20.95 % instead of a 21 % O2 reference.
2. Checks of the fields: the sum of the nine stored mole fractions in the outlet cells; the O2 and
   N2 mole fractions of the inlet air (cells below z = 0 at 900 +- 1 K); the outlet mass flow per
   atmosphere of operating pressure (eq. 1 of the paper scales the mass flows with p0).
3. The two pairs of cases with identical inputs: whether their fields are identical, and if not,
   which fields differ and by how much.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract_data as ed  # noqa: E402

MOLAR_MASS = {'N2': 28.0134, 'NH2': 16.0226, 'NO2': 46.0055, 'H2O': 18.0153, 'OH': 17.0073,
              'O2': 31.9988, 'H2': 2.01588, 'NH3': 17.0305, 'NO': 30.0061}  # g/mol


def read_columns(path, cols, block=20000):
    """Columns `cols` of the C-ordered X.npy, read in blocks of rows (bounded memory)."""
    with open(path, 'rb') as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        if fortran or len(shape) != 2:
            sys.exit('X.npy: expected a C-ordered 2-D array')
        out = np.empty((shape[0], len(cols)), dtype=dtype)
        for start in range(0, shape[0], block):
            rows = min(block, shape[0] - start)
            buf = np.fromfile(fh, dtype=dtype, count=rows * shape[1]).reshape(rows, shape[1])
            out[start:start + rows] = buf[:, cols]
    return out


def rel(a, b):
    return np.max(np.abs(a / b - 1))


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)   # the downloaded files are needed
        sys.exit(3)
    P, xy, X = ed.load(sys.argv[1], verify=False)
    n = xy.shape[0]
    base = ed.reduce_cases(P, xy, X)
    print('1. Sensitivity of the outlet reduction (largest relative change over the 175 cases)')
    area = ed.reduce_cases(P, xy, X, weight='area')
    print(f'   area instead of mass-flux weighting: NOx {rel(area["NOx"], base["NOx"]):.2e}, '
          f'T_out {rel(area["T_out"], base["T_out"]):.2e}, O2_dry {rel(area["O2_dry"], base["O2_dry"]):.2e}')
    idx = ed.outlet_cells(xy)
    e = ed.face_edges(xy[idx, 1])
    A = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    G = ed.outlet_fields(X, idx, n)
    wmix = sum(G[s] * MOLAR_MASS[s] for s in MOLAR_MASS)
    w = G['rho'] * G['vz'] * A[:, None] / wmix
    av = ed.averages(G, w)
    nox_molar = ed.corrected(av, ['NO', 'NO2'])
    print(f'   molar-flux instead of mass-flux weighting: NOx {rel(nox_molar, base["NOx"]):.2e}, '
          f'T_out {rel(av["T"], base["T_out"]):.2e}')
    up = ed.reduce_cases(P, xy, X, offset=1.0e-3)
    print(f'   cell row 1 mm upstream ({up["_n_cells"]} cells): NOx {rel(up["NOx"], base["NOx"]):.2e}, '
          f'T_out {rel(up["T_out"], base["T_out"]):.2e}')
    cen = ed.reduce_cases(P, xy, X, edges='centred')
    print(f'   centred face edges (wall at {1e3 * cen["_r_wall"]:.3f} mm against '
          f'{1e3 * base["_r_wall"]:.3f} mm): NOx {rel(cen["NOx"], base["NOx"]):.2e}, '
          f'mdot {rel(cen["mdot"], base["mdot"]):.2e}')
    o2d = base['O2_dry']
    f2095 = (20.95 - 16) / (20.95 - o2d) / ((21 - 16) / (21 - o2d))
    print(f'   20.95 % instead of 21 % O2 reference: NOx x {f2095.min():.4f} to {f2095.max():.4f}')

    print('2. Checks of the fields')
    s = sum(G[k] for k in MOLAR_MASS)
    print(f'   sum of the nine mole fractions in the outlet cells: {s.min():.5f} to {s.max():.5f}')
    T = np.asarray(X[14 * n:15 * n, :])
    cold = (np.abs(T - 900.0) < 1.0) & (xy[:, 0] < 0)[:, None]
    for k, name in [(5, 'O2'), (0, 'N2')]:
        F = np.asarray(X[k * n:(k + 1) * n, :])
        v = F[cold]
        print(f'   inlet air {name} (cells below z = 0 at 900 +- 1 K, {cold.sum()} cell values): '
              f'{v.min():.5f} to {v.max():.5f}')
    del T
    r = base['mdot'] / (P[:, 2] / 101325.0)
    print(f'   outlet mass flow per atm: {r.min():.5f} to {r.max():.5f} kg/s (median {np.median(r):.5f})')

    print('3. Cases with identical inputs')
    dup = ed.duplicates(P)
    pairs = [(i, dup[i] - 1) for i in np.where(dup > 0)[0]]
    cols = read_columns(os.path.join(sys.argv[1], 'X.npy'), [c for pr in pairs for c in pr])
    for k_pair, (i, j) in enumerate(pairs):
        a = cols[:, 2 * k_pair]
        b = cols[:, 2 * k_pair + 1]
        if np.array_equal(a, b):
            print(f'   case {i + 1} = case {j + 1}: all {a.size} field values identical')
            continue
        diffs = []
        for k, f in enumerate(ed.FEATURES):
            d = np.abs(a[k * n:(k + 1) * n] - b[k * n:(k + 1) * n])
            if d.max() > 0:
                scale = np.abs(b[k * n:(k + 1) * n]).max()
                diffs.append(f'{f} (max |diff| {d.max():.3g}, {d.max() / scale:.1e} of max |{f}|, '
                             f'{int(np.sum(d > 0))} cells)')
        print(f'   case {i + 1} vs case {j + 1}: fields differ in ' + '; '.join(diffs))
        print(f'      outlet NOx {base["NOx"][i]:.4f} vs {base["NOx"][j]:.4f} ppmvd, '
              f'T_out {base["T_out"][i]:.4f} vs {base["T_out"][j]:.4f} K')


if __name__ == '__main__':
    main()
