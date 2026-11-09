"""Independent check of the AirfRANS force coefficients in airfrans_forces.csv (DATA.md, "Accuracy").

    python check_accuracy.py --zip Dataset.zip --out DIR [--rows N] [--max-seconds S]
    (the archive path may also be given by the environment variable AIRFRANS_ZIP)

The shipped coefficients were computed by the airfrans library (pressure of the internal field at the
wall nodes; wall shear stress from the vtk point gradient of the velocity; normals and lengths of the
aerofoil patch). This script recomputes them a second way, with numpy and the standard library only
(its own reader of the VTK XML files, read from the archive in memory, nothing extracted; it reads ascii,
inline binary and appended raw or base64 data, zlib-compressed or not, with 32- or 64-bit headers):

  * pressure force  F_p = - sum_f p_f n_f L_f  over the faces f of the aerofoil patch, with p_f the
    face pressure stored as cell data of ``*_aerofoil.vtp`` (OpenFOAM's face values; if the file has
    only point values, the mean of the face's two end points), and the outward
    unit normal n_f and length L_f computed here from the patch's point coordinates;
  * viscous force   F_v = sum_f nu (u_c . t_f) / d_f t_f L_f  over the same wall faces, where u_c is the
    velocity stored at the centre of the wall cell next to face f (cell data of ``*_internal.vtu``; if the
    file has only point values, the mean of the cell's four point values),
    t_f the unit tangent of the face, and d_f the distance from the cell centre (mean of its four
    vertices) to the face line: a one-sided wall gradient, as in OpenFOAM's wall shear stress
    (no-slip wall, nu_t = 0 at the wall);
  * C_D and C_L are the components of (F_p + F_v) / (U_inf^2 / 2) (kinematic pressure, chord 1 m) along
    and normal to the free stream.

The viscosity is nu = 1.5498148e-5 m^2/s, the value airfrans uses and the current case generator
(Extrality/NACA_simulation) writes into OpenFOAM's transportProperties, so that the comparison isolates the
integration. The inlet velocities of the archive were set with the paper's Table 6 value, 1.56e-5 (DATA.md);
which value the solver used is not recorded in the files. The viscous parts scale with nu.

The script also checks the inputs: the wall nodes of the internal field lie on the aerofoil patch, and it
prints the range of U_inf against Re in [2e6, 6e6] for both viscosities.

Results are appended to DIR/check_rows.csv (resumable; ``--max-seconds`` stops early, rerun to
continue); when all requested cases are done the script prints the largest differences from the shipped
table, which arrays it read (OpenFOAM's cell values or the point-value fallback above) and the SHA-256 of
this script, so that a log identifies the version and the branch that produced it. ``--rows N`` checks N
evenly spaced cases (default: all 1000). DIR must lie outside this data set's directory. The script reads
airfrans_forces.csv and airfrans_case_names.txt from this directory (DATA.md says how to make them).
"""

import argparse
import base64
import hashlib
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NU = -3.400747e-6 + 3.452139e-8 * 298.15 + 1.00881778e-10 * 298.15 ** 2 - 1.363528e-14 * 298.15 ** 3
DTYPES = {'Float32': '<f4', 'Float64': '<f8', 'Int64': '<i8', 'Int32': '<i4', 'Int8': '<i1', 'UInt8': '<u1',
          'UInt32': '<u4', 'UInt64': '<u8'}
ROWFILE = 'check_rows.csv'


def _blocks(read, hdt, compressed):
    """Data bytes of one array: ``read(nbytes, header)`` returns the next header or data bytes."""
    hsz = hdt.itemsize
    if not compressed:
        n = int(np.frombuffer(read(hsz, True), hdt)[0])
        return read(n, False)
    nb = int(np.frombuffer(read(hsz, True), hdt)[0])
    rest = np.frombuffer(read(hsz * (2 + nb), True), hdt)
    data = read(int(rest[2:].sum()), False)
    out, pos = [], 0
    for size in rest[2:]:
        out.append(zlib.decompress(data[pos:pos + int(size)]))
        pos += int(size)
    return b''.join(out)


def _b64len(n):
    return 4 * -(-n // 3)


def _inline(text, hdt, compressed):
    """Bytes of one base64 array (inline, or at its offset in appended base64 data). VTK encodes the
    header and the data separately; an uncompressed array encoded in one piece is also accepted."""
    hsz = hdt.itemsize
    if compressed:
        nb = int(np.frombuffer(base64.b64decode(text[:_b64len(hsz)])[:hsz], hdt)[0])
        hlen = _b64len(hsz * (3 + nb))
        head = np.frombuffer(base64.b64decode(text[:hlen])[:hsz * (3 + nb)], hdt)
        data = base64.b64decode(text[hlen:hlen + _b64len(int(head[3:].sum()))])
        out, pos = [], 0
        for size in head[3:]:
            out.append(zlib.decompress(data[pos:pos + int(size)]))
            pos += int(size)
        return b''.join(out)
    hlen = _b64len(hsz)
    n = int(np.frombuffer(base64.b64decode(text[:hlen])[:hsz], hdt)[0])
    try:
        data = base64.b64decode(text[hlen:hlen + _b64len(n)])
    except ValueError:
        data = b''
    if len(data) != n:                      # header and data encoded together
        data = base64.b64decode(text[:_b64len(hsz + n)])[hsz:hsz + n]
    return data


def read_vtk(raw):
    """Points, connectivity, offsets, point data and cell data of a VTK XML PolyData/UnstructuredGrid
    (ascii, inline binary or appended raw/base64 data; zlib or no compression; UInt32 or UInt64 headers)."""
    appended, appended_enc = b'', True
    k = raw.find(b'<AppendedData')
    if k >= 0:
        j = raw.index(b'_', raw.index(b'>', k)) + 1
        stop = raw.rfind(b'</AppendedData>')
        appended_enc = b'encoding="raw"' not in raw[k:j]
        appended = raw[j:stop]
        if appended_enc:
            appended = b''.join(appended.split())
        raw = raw[:k] + b'</VTKFile>'
    root = ET.fromstring(raw)
    comp = root.get('compressor')
    if comp is not None and 'ZLib' not in comp:
        raise ValueError(f'unsupported compressor {comp}')
    compressed = comp is not None
    hdt = np.dtype({'UInt32': '<u4', 'UInt64': '<u8'}[root.get('header_type', 'UInt32')])
    if root.get('byte_order', 'LittleEndian') != 'LittleEndian':
        raise ValueError('big-endian file')
    piece = root.find('./*/Piece')

    def values(a):
        dt = np.dtype(DTYPES[a.get('type')])
        fmt = a.get('format')
        if fmt == 'ascii':
            return np.array(a.text.split(), dtype=dt)
        if fmt == 'binary':
            return np.frombuffer(_inline(''.join(a.text.split()), hdt, compressed), dt)
        off = int(a.get('offset'))
        if not appended_enc:
            pos = [off]

            def read(n, header):
                b = appended[pos[0]:pos[0] + n]
                pos[0] += n
                return b
            return np.frombuffer(_blocks(read, hdt, compressed), dt)
        return np.frombuffer(_inline(appended[off:].decode('ascii'), hdt, compressed), dt)

    def arrays(tag):
        node = piece.find(tag)
        res = {}
        for a in (node if node is not None else []):
            v = values(a)
            nc = int(a.get('NumberOfComponents', '1'))
            res[a.get('Name')] = v.reshape(-1, nc) if nc > 1 else v
        return res

    pts = values(piece.find('Points/DataArray')).reshape(-1, 3).astype(float)
    topo = piece.find('Lines') if piece.find('Lines') is not None else piece.find('Cells')
    conn = {a.get('Name'): values(a) for a in topo}
    return pts, conn['connectivity'], conn['offsets'], arrays('PointData'), arrays('CellData')


def case_coefficients(zf, name):
    """CD, CD_p, CD_v, CL, CL_p, CL_v of one case, two input checks, and which arrays were read
    (1 = cell values written by OpenFOAM, 0 = point-value fallback) for the pressure and the velocity."""
    parts = name.split('_')
    U, alpha = float(parts[2]), np.deg2rad(float(parts[3]))
    # aerofoil patch: line segments with face pressure
    pa, ca, oa, pda, cda = read_vtk(zf.read(f'Dataset/{name}/{name}_aerofoil.vtp'))
    if not np.all(np.diff(oa) == 2):
        raise ValueError('aerofoil patch is not made of 2-point lines')
    seg = ca.reshape(-1, 2)
    a, b = pa[seg[:, 0], :2], pa[seg[:, 1], :2]
    e = b - a
    L = np.hypot(e[:, 0], e[:, 1])
    t = e / L[:, None]
    n = np.column_stack([t[:, 1], -t[:, 0]])          # one of the two unit normals of each face
    mid = 0.5 * (a + b)
    centroid = (mid * L[:, None]).sum(0) / L.sum()
    if np.sum(((mid - centroid) * n).sum(1) * L) < 0:  # make n point out of the closed contour
        n = -n
    p_cells = 'p' in cda
    if p_cells:                                        # face values written by OpenFOAM
        p = cda['p'].astype(float)
    else:                                              # else the mean of the two end-point values
        p = pda['p'].astype(float)[seg].mean(axis=1)
    Fp = -(p[:, None] * n * L[:, None]).sum(0)
    # internal field: wall cells and their centre velocity
    pi, ci, oi, pdi, cdi = read_vtk(zf.read(f'Dataset/{name}/{name}_internal.vtu'))
    if not np.all(np.diff(oi) == 4) or oi[0] != 4:
        raise ValueError('internal field is not made of quadrilaterals')
    quads = ci.reshape(-1, 4)
    wall_node = np.all(pdi['U'][:, :2] == 0, axis=1)
    # wall nodes of the internal mesh coincide with the patch points (to rounding)
    key = lambda xy: [tuple(r) for r in np.round(xy, 9)]
    patch_pts = set(key(pa[:, :2]))
    wall_ok = len(set(key(pi[wall_node, :2])) - patch_pts) == 0
    u_cells = 'U' in cdi
    if u_cells:                                        # cell-centre velocity written by OpenFOAM
        cell_u = cdi['U'][:, :2].astype(float)
    else:                                              # else the mean of the cell's four point values
        cell_u = pdi['U'][:, :2].astype(float)[quads].mean(axis=1)
    nwall = wall_node[quads].sum(1)
    cells = np.nonzero(nwall == 2)[0]
    Fv = np.zeros(2)
    used = 0
    for c in cells:
        q = quads[c]
        w = q[wall_node[q]]
        x0, x1 = pi[w[0], :2], pi[w[1], :2]
        ev = x1 - x0
        Lf = np.hypot(*ev)
        tf = ev / Lf
        xc = pi[q, :2].mean(0)
        d = abs((xc - x0)[0] * tf[1] - (xc - x0)[1] * tf[0])
        uc = cell_u[c]
        Fv += NU * (uc @ tf) / d * tf * Lf
        used += 1
    q = 0.5 * U ** 2
    basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    cp, cv = basis @ Fp / q, basis @ Fv / q
    wall_len_ratio = 0.0
    if used:
        wall_len_ratio = used / len(L)
    return ([cp[0] + cv[0], cp[0], cv[0], cp[1] + cv[1], cp[1], cv[1]], wall_ok, wall_len_ratio,
            float(p_cells), float(u_cells))


def inside(path, folder):
    """True if path is folder or lies inside it (after resolving links)."""
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return os.path.commonpath([path, folder]) == folder


def read_shipped():
    if not os.path.exists(os.path.join(HERE, 'airfrans_forces.csv')):
        raise SystemExit('check_accuracy.py compares with airfrans_forces.csv in this directory, which is not '
                         'there: make it with convert_data.py and copy it here (DATA.md, "Running the notebook '
                         'from the archive"), then rerun.')
    with open(os.path.join(HERE, 'airfrans_case_names.txt')) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    data = np.loadtxt(os.path.join(HERE, 'airfrans_forces.csv'), delimiter=',')
    return names, data


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--zip', default=os.environ.get('AIRFRANS_ZIP'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', type=int, default=None)
    ap.add_argument('--max-seconds', type=float, default=None)
    args = ap.parse_args(argv)
    if not args.zip or not os.path.exists(args.zip):
        raise SystemExit('check_accuracy.py needs Dataset.zip of AirfRANS (download and size in DATA.md): '
                         'give --zip PATH or set AIRFRANS_ZIP, then rerun.')
    out = os.path.abspath(args.out)
    if inside(out, HERE):
        ap.error('--out must be a folder outside this data set\'s directory (not the directory or a folder in it)')
    os.makedirs(out, exist_ok=True)
    names, data = read_shipped()
    idx = list(range(len(names)))
    if args.rows:
        step = len(names) / args.rows
        idx = [int(i * step) for i in range(args.rows)]
    rowfile = os.path.join(out, ROWFILE)
    done = {}
    if os.path.exists(rowfile):
        for ln in open(rowfile):
            s = ln.strip().split(',')
            if len(s) == 11:
                done[s[0]] = [float(v) for v in s[1:]]
    t0 = time.time()
    with zipfile.ZipFile(args.zip) as zf, open(rowfile, 'a') as fh:
        for i in idx:
            if names[i] in done:
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print(f'stopped after {time.time() - t0:.0f} s; {len(done)} of {len(idx)} cases done; rerun to continue')
                return 3
            coef, wall_ok, ratio, p_cells, u_cells = case_coefficients(zf, names[i])
            vals = coef + [float(wall_ok), ratio, p_cells, u_cells]
            fh.write(','.join([names[i]] + [repr(float(v)) for v in vals]) + '\n')
            fh.flush()
            done[names[i]] = vals
    mine = np.array([done[names[i]] for i in idx])
    ref = data[idx][:, 7:13]
    U = data[idx, 1]
    print(f'independent integration of {len(idx)} cases (numpy VTK reader; face pressure; one-sided wall gradient '
          f'at wall-cell centres) against airfrans in airfrans_forces.csv; '
          f'{int(np.sum((data[idx, 0] == 4) & (data[idx, 4] > 0)))} of them cambered NACA 4-digit (P > 0):')
    labels = ['CD', 'CD_p', 'CD_v', 'CL', 'CL_p', 'CL_v']
    for j, lab in enumerate(labels):
        diff = mine[:, j] - ref[:, j]
        rel = np.abs(diff) / np.abs(ref[:, j])
        print(f'  {lab:5s}: median |rel. diff| {np.median(rel):.2e}, 95th pct {np.percentile(rel, 95):.2e}, '
              f'max {rel.max():.2e}; median diff {np.median(diff):+.2e}, max |diff| {np.abs(diff).max():.2e}')
    print(f'  wall nodes of the internal field lie on the aerofoil patch in all cases: {bool(mine[:, 6].all())}')
    print(f'  wall cells found per patch face: {mine[:, 7].min():.3f} to {mine[:, 7].max():.3f}')
    print(f'  arrays used: face pressure from the cell values of aerofoil.vtp in {int(mine[:, 8].sum())} of '
          f'{len(idx)} cases (else the point-value mean); wall-cell velocity from the cell values of internal.vtu '
          f'in {int(mine[:, 9].sum())} of {len(idx)} cases (else the point-value mean)')
    lo, hi = 2e6 * NU, 6e6 * NU
    print(f'  nu = {NU:.7e} m^2/s: U_inf in [{lo:.3f}, {hi:.3f}] m/s for Re in [2e6, 6e6]; '
          f'shipped U_inf {data[:, 1].min():.3f} to {data[:, 1].max():.3f} m/s '
          f'(with nu = 1.56e-5: Re {data[:, 1].min() / 1.56e-5:.4g} to {data[:, 1].max() / 1.56e-5:.4g})')
    with open(os.path.abspath(__file__), 'rb') as fh:
        print(f'  check_accuracy.py SHA-256 {hashlib.sha256(fh.read()).hexdigest()}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
