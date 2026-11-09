"""Geometry and triangular mesh of the wrench of Lam, Zahm, Marzouk and Willcox (2020), Sec. 5.2.

The geometry is the one of the authors' MATLAB code (wrenchmark, O. Zahm, Inria GitLab), rebuilt
here from its shape parameters: a handle (rectangle [-2.8, 2.8] x [-0.4, 0.4]), a ring on the left
(circle of radius 1 at (-3.5, 0) with a hexagonal hole of circumradius 0.6, joined to the handle
by straight tapers), and an open jaw on the right (circle of radius 1.4 at (3.5, 0) minus a
hexagon of circumradius 1.1 centred at (3.95, -0.26), joined to the handle by two circular
fillets of radius 1.305 tangent to the handle and the head). Length unit: nondimensional.

Boundary conditions (paper Fig. 5a, p. 13): u = 0 on the two inner jaw faces that the hexagon
cuts, a unit downward traction on the straight top edge of the handle, and the quantity of
interest is the vertical displacement at the left vertex (-4.1, 0) of the ring's hexagonal hole.

The mesh generator is written for this data set (numpy and scipy only; no outside mesher): the
boundary is resampled at spacing ~h with its corners kept, interior nodes start on an
equilateral lattice and are relaxed by a truss-spring iteration with fixed boundary nodes (the
idea of Persson and Strang, SIAM Rev. 46 (2004) 329-345, implemented anew), and the final mesh is
the Delaunay triangulation restricted to the domain. H_FINE gives 2197 triangles, the number in
the paper (p. 13).

    python wrench_mesh.py --out DIR      # writes the three mesh CSVs to DIR
"""
import argparse
import os

import numpy as np
from scipy.spatial import Delaunay

# Shape parameters (wrenchmark Wrench.initGeo)
L_HALF = 3.5        # ring and head centres at x = -/+ L_HALF
E_HALF = 0.4        # half thickness of the handle
R_HEAD = 1.4        # head (jaw) radius
R_RING = 1.0        # ring radius
C_TAPER = 0.85      # taper start height on the ring side
R_JAW_HEX = 1.1     # circumradius of the jaw hexagon
R_RING_HEX = 0.6    # circumradius of the ring's hexagonal hole
R_FILLET = (E_HALF**2 + (L_HALF - 1.4)**2 - R_HEAD**2) / (2 * R_HEAD - 2 * E_HALF)   # 1.305

H_FINE = 0.097      # boundary spacing
N_ELEMENTS = 2197   # number of triangles in the paper (p. 13)
N_ITER = 300        # spring-relaxation iterations
POI = np.array([-L_HALF - R_RING_HEX, 0.0])   # point of interest (-4.1, 0)

PIECE_KIND = {'neumann': 1, 'dirichlet': 2, 'free': 0}


def _seg_circle(p0, p1, c, r):
    """Parameters t in [0, 1] where the segment p0 + t (p1 - p0) meets the circle |x - c| = r."""
    d = p1 - p0
    f = p0 - c
    a, b, cc = d @ d, 2 * f @ d, f @ f - r * r
    disc = np.sqrt(b * b - 4 * a * cc)
    ts = [(-b - disc) / (2 * a), (-b + disc) / (2 * a)]
    return [t for t in ts if -1e-12 <= t <= 1 + 1e-12]


def _ang(p, c):
    return np.arctan2(p[1] - c[1], p[0] - c[0])


def boundary_pieces():
    """The boundary as a list of (kind, sampler, length) pieces.

    The outer loop runs clockwise starting at the left end of the handle's top edge; the hole is a
    separate loop. A sampler maps s in [0, 1] (arc-length fraction) to points.
    """
    l, e = L_HALF, E_HALF
    c_head, c_ring = np.array([l, 0.0]), np.array([-l, 0.0])
    c_f1, c_f2 = np.array([1.4, e + R_FILLET]), np.array([1.4, -e - R_FILLET])
    # taper lines (from the polygon (-l/2, 0), (-l + 0.5, +-c2))
    q0 = np.array([-l / 2, 0.0])
    q1u, q1d = np.array([-l + 0.5, C_TAPER]), np.array([-l + 0.5, -C_TAPER])
    tu = (C_TAPER - e) / C_TAPER        # where the taper meets y = +-e (from q1 toward q0)
    TLu, TLd = q1u + tu * (q0 - q1u), q1d + tu * (q0 - q1d)
    t_ring_u = max(_seg_circle(q0, q1u, c_ring, R_RING))
    t_ring_d = max(_seg_circle(q0, q1d, c_ring, R_RING))
    Ru, Rd = q0 + t_ring_u * (q1u - q0), q0 + t_ring_d * (q1d - q0)
    # fillet / head tangent points
    T1 = c_f1 + R_FILLET * (c_head - c_f1) / np.linalg.norm(c_head - c_f1)
    T2 = c_f2 + R_FILLET * (c_head - c_f2) / np.linalg.norm(c_head - c_f2)
    # jaw hexagon
    hc = np.array([0.45 + l, -0.26])
    ang = np.arange(6) / 6 * 2 * np.pi + np.pi / 6
    HV = hc + R_JAW_HEX * np.c_[np.cos(ang), np.sin(ang)]   # vertices at 30, 90, ..., 330 deg
    tA = min(_seg_circle(HV[1], HV[0], c_head, R_HEAD))      # on edge 90 -> 30 deg
    A = HV[1] + tA * (HV[0] - HV[1])
    tB = min(_seg_circle(HV[3], HV[4], c_head, R_HEAD))      # on edge 210 -> 270 deg
    B = HV[3] + tB * (HV[4] - HV[3])

    def seg(p, q):
        return (lambda s: p[None, :] + np.asarray(s)[:, None] * (q - p)[None, :], np.linalg.norm(q - p))

    def arc(c, r, a0, a1):
        def f(s):
            a = a0 + np.asarray(s) * (a1 - a0)
            return c[None, :] + r * np.c_[np.cos(a), np.sin(a)]
        return f, abs(a1 - a0) * r

    def unwrap(a0, a1, direction):
        """a1 shifted by 2 pi so that a0 -> a1 runs in the given direction (+1 ccw, -1 cw)."""
        if direction < 0:
            while a1 > a0:
                a1 -= 2 * np.pi
        else:
            while a1 < a0:
                a1 += 2 * np.pi
        return a1

    a_T1, a_A = _ang(T1, c_head), _ang(A, c_head)
    a_B, a_T2 = _ang(B, c_head), _ang(T2, c_head)
    a_Rd, a_Ru = _ang(Rd, c_ring), _ang(Ru, c_ring)
    outer = [
        ('neumann',) + seg(TLu, np.array([1.4, e])),
        ('free',) + arc(c_f1, R_FILLET, -np.pi / 2, _ang(T1, c_f1)),
        ('free',) + arc(c_head, R_HEAD, a_T1, unwrap(a_T1, a_A, -1)),
        ('dirichlet',) + seg(A, HV[1]),
        ('free',) + seg(HV[1], HV[2]),
        ('free',) + seg(HV[2], HV[3]),
        ('dirichlet',) + seg(HV[3], B),
        ('free',) + arc(c_head, R_HEAD, a_B, unwrap(a_B, a_T2, -1)),
        ('free',) + arc(c_f2, R_FILLET, _ang(T2, c_f2), np.pi / 2),
        ('free',) + seg(np.array([1.4, -e]), TLd),
        ('free',) + seg(TLd, Rd),
        ('free',) + arc(c_ring, R_RING, a_Rd, unwrap(a_Rd, a_Ru, -1)),
        ('free',) + seg(Ru, TLu),
    ]
    hc2 = np.array([-l, 0.0])
    ang2 = np.arange(6) / 6 * 2 * np.pi
    HV2 = hc2 + R_RING_HEX * np.c_[np.cos(ang2), np.sin(ang2)]
    hole = [('free',) + seg(HV2[k], HV2[(k + 1) % 6]) for k in range(6)]
    return outer, hole


def _sample_loop(pieces, h):
    """Nodes of a closed loop (each piece split into round(length/h) equal parts, >= 1) and, per
    segment, its kind code."""
    pts, kinds = [], []
    for kind, f, length in pieces:
        n = max(1, int(round(length / h)))
        s = np.arange(n) / n
        pts.append(f(s))
        kinds += [PIECE_KIND[kind]] * n
    return np.vstack(pts), np.array(kinds)


def _inside(P, loops):
    """Even-odd point-in-polygon test of points P (N, 2) against closed polylines."""
    inside = np.zeros(len(P), bool)
    for V in loops:
        a, b = V, np.roll(V, -1, axis=0)
        x, y = P[:, :1], P[:, 1:]
        cond = (a[None, :, 1] > y) != (b[None, :, 1] > y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = a[None, :, 0] + (y - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / (b[None, :, 1] - a[None, :, 1])
        inside ^= (np.sum(cond & (x < xint), axis=1) % 2).astype(bool)
    return inside


def _dist_to_segments(P, A, B):
    d = B - A
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(-1) / (d * d).sum(-1)[None], 0, 1)
    proj = A[None] + t[..., None] * d[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(axis=1)


def initial_nodes(h, n_elements=None):
    """Boundary nodes, their kinds, and the interior nodes for boundary spacing h.

    Interior nodes start on an equilateral lattice of spacing h, at distance > 0.55 h from the
    boundary. With n_elements, the lattice spacing is reduced in steps of 0.1 % until there are at
    least (n_elements - V_bnd) / 2 lattice nodes, and the surplus nodes closest to the boundary
    are dropped, so that a conforming triangulation has exactly n_elements triangles
    (T = 2 V_int + V_bnd for a domain with one hole).
    """
    outer, hole = boundary_pieces()
    Vo, ko = _sample_loop(outer, h)
    Vh, kh = _sample_loop(hole, h)
    loops = [Vo, Vh]
    nb = len(Vo) + len(Vh)
    Ab = np.vstack([Vo, Vh])
    Bb = np.vstack([np.roll(Vo, -1, 0), np.roll(Vh, -1, 0)])
    if n_elements is not None and (n_elements - nb) % 2:
        raise ValueError(f'{n_elements} triangles need an odd number of boundary nodes; h = {h} gives {nb}')
    target = None if n_elements is None else (n_elements - nb) // 2
    hl = h
    while True:
        dy = hl * np.sqrt(3) / 2
        ys = np.arange(-1.5, 1.5 + dy, dy)
        xs = np.arange(-4.6, 5.0 + hl, hl)
        X, Y = np.meshgrid(xs, ys)
        X = X + (np.arange(len(ys)) % 2)[:, None] * hl / 2
        P = np.c_[X.ravel(), Y.ravel()]
        P = P[_inside(P, loops)]
        dist = _dist_to_segments(P, Ab, Bb)
        P, dist = P[dist > 0.55 * h], dist[dist > 0.55 * h]
        if target is None or len(P) >= target:
            break
        hl *= 0.999
    if target is not None:
        keep = np.sort(np.argsort(-dist, kind='stable')[:target])
        P = P[keep]
    return Vo, ko, Vh, kh, P


def element_count(h):
    """Triangles a valid triangulation of the domain has with these nodes: 2 V_int + V_bnd (one hole)."""
    Vo, _, Vh, _, P = initial_nodes(h)
    return 2 * len(P) + len(Vo) + len(Vh)


def _triangulate(p, loops):
    t = Delaunay(p).simplices
    cen = p[t].mean(axis=1)
    return t[_inside(cen, loops)]


def make_mesh(h=H_FINE, n_elements=N_ELEMENTS, n_iter=N_ITER):
    """Return nodes (N, 2), triangles (T, 3) counter-clockwise, boundary segments (S, 3) as
    (node_a, node_b, kind code) and the index of the point-of-interest node."""
    Vo, ko, Vh, kh, P = initial_nodes(h, n_elements)
    loops = [Vo, Vh]
    nb = len(Vo) + len(Vh)
    p = np.vstack([Vo, Vh, P])
    Ab = np.vstack([Vo, Vh])
    Bb = np.vstack([np.roll(Vo, -1, 0), np.roll(Vh, -1, 0)])
    p_tri = p.copy()
    t = _triangulate(p, loops)
    for it in range(n_iter):
        if np.max(np.sqrt(((p - p_tri) ** 2).sum(1))) > 0.1 * h:
            p_tri = p.copy()
            t = _triangulate(p, loops)
        bars = np.unique(np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1), axis=0)
        bv = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.sqrt((bv ** 2).sum(1))
        L0 = 1.2 * np.sqrt((L ** 2).mean())
        F = np.maximum(L0 - L, 0.0)
        Fv = (F / L)[:, None] * bv
        Ft = np.zeros_like(p)
        np.add.at(Ft, bars[:, 0], Fv)
        np.add.at(Ft, bars[:, 1], -Fv)
        Ft[:nb] = 0.0
        p_new = p + 0.2 * Ft
        moved = np.arange(nb, len(p))
        ok = _inside(p_new[moved], loops) & (_dist_to_segments(p_new[moved], Ab, Bb) > 0.25 * h)
        p[moved[ok]] = p_new[moved[ok]]
    t = _triangulate(p, loops)
    # counter-clockwise
    a = p[t]
    det = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])
    t[det < 0] = t[det < 0][:, [0, 2, 1]]
    # boundary segments
    io = np.arange(len(Vo))
    ih = len(Vo) + np.arange(len(Vh))
    segs = np.vstack([np.c_[io, np.roll(io, -1), ko], np.c_[ih, np.roll(ih, -1), kh]])
    poi = int(np.argmin(((p - POI) ** 2).sum(1)))
    _validate(p, t, segs, nb, poi)
    return p, t, segs, poi


def _validate(p, t, segs, nb, poi):
    T = len(t)
    if T != 2 * (len(p) - nb) + nb:
        raise RuntimeError(f'triangulation is not conforming: {T} triangles, expected {2 * (len(p) - nb) + nb}')
    if len(np.unique(t)) != len(p):
        raise RuntimeError('some nodes are not used by any triangle')
    edges = set(map(tuple, np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1).tolist()))
    missing = [s for s in np.sort(segs[:, :2], axis=1).tolist() if tuple(s) not in edges]
    if missing:
        raise RuntimeError(f'{len(missing)} boundary segments are not mesh edges')
    if np.linalg.norm(p[poi] - POI) > 1e-12:
        raise RuntimeError('the point of interest is not a mesh node')


def quality(p, t):
    """Smallest and largest interior angle (degrees) and the area range."""
    a = p[t]
    ang = []
    for i in range(3):
        u = a[:, (i + 1) % 3] - a[:, i]
        v = a[:, (i + 2) % 3] - a[:, i]
        ang.append(np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1))))
    ang = np.array(ang)
    area = 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1]))
    return ang.min(), ang.max(), area.min(), area.max(), area.sum()


NODES_FILE, ELEMENTS_FILE, BOUNDARY_FILE = 'mesh_nodes.csv', 'mesh_elements.csv', 'mesh_boundary.csv'


def write_mesh(out, p, t, segs, poi, h=H_FINE):
    tag = (f'wrench_mesh.py, h = {h}, {N_ITER} relaxation iterations, scipy.spatial.Delaunay; '
           f'{len(p)} nodes, {len(t)} triangles; point of interest = node {poi} at (-4.1, 0)')
    np.savetxt(os.path.join(out, NODES_FILE), p, fmt='%.17g', delimiter=',',
               header=f'Mesh node coordinates (nondimensional length). {tag}\nx,y')
    np.savetxt(os.path.join(out, ELEMENTS_FILE), t, fmt='%d', delimiter=',',
               header=f'Triangles: 0-based node indices, counter-clockwise; row k is input k (log Young modulus of element k). {tag}\n'
                      'node_a,node_b,node_c')
    np.savetxt(os.path.join(out, BOUNDARY_FILE), segs, fmt='%d', delimiter=',',
               header=f'Boundary segments: 0-based node indices and kind (0 traction-free, 1 unit downward traction, '
                      f'2 clamped u = 0). {tag}\nnode_a,node_b,kind')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--h', type=float, default=H_FINE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p, t, segs, poi = make_mesh(args.h)
    write_mesh(args.out, p, t, segs, poi, args.h)
    q = quality(p, t)
    print(f'{len(p)} nodes, {len(t)} triangles; angles {q[0]:.1f}-{q[1]:.1f} deg; area {q[4]:.6f}')


if __name__ == '__main__':
    main()
