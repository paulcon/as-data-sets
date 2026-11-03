"""Recompute the numbers of DATA.md that neither the notebook nor generate_data.py prints.

    python check_accuracy.py            # all checks, a few seconds; numpy and scipy only

Prints, beside the values DATA.md quotes:

1. Sampling design (thermal_compliance.npz): dtype and range of theta, duplicate rows, values
   at exactly 0 or 1, per-column kurtosis (Pearson, biased; 1.8 for a uniform), the range of E,
   and the sign of the gradients.
2. Heat source: the integral of f = 56 (1 - |x| - |y|)^6 over Omega, inside the diamond
   |x| + |y| <= 1 and in the four corner triangles outside it (exact Gauss rules), and the energy
   at theta = 0.5 for the two boundary lists (boundary='paper', all 128 boundary nodes, and
   boundary='casm', the CASM notebook's 124 nodes) with the source as written and with it set to
   zero outside the diamond.
3. Discretization: E at theta = 0.5 on 16 x 16, 32 x 32 and 64 x 64 meshes.
4. Load quadrature: the change of E when the load vector is integrated exactly (collapsed 6 x 6
   Gauss rule per element) instead of with the 4-point Hammer rule, at theta = 0.5 and at sample 0.

Deterministic. Reads thermal_compliance.npz next to this file and writes nothing; runs from any
directory. The checks against the author's stored files (dfData.npy, fData.npy, Xtest.npy) are
in `generate_data.py --casm DIR`.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import thermal_model as tm  # noqa: E402


def gauss_triangle(n):
    """Collapsed (Duffy) n x n Gauss-Legendre rule on the reference triangle (0,0), (1,0), (0,1).

    Exact for polynomials of total degree up to 2n - 2."""
    g, w = np.polynomial.legendre.leggauss(n)
    u, wu = (g + 1)/2, w/2
    U, V = np.meshgrid(u, u, indexing='ij')
    W = np.outer(wu, wu)
    xi = U
    eta = V*(1 - U)
    return xi.ravel(), eta.ravel(), (W*(1 - U)).ravel()


def integrate_on_triangle(fun, a, b, c, n=8):
    xi, eta, w = gauss_triangle(n)
    a, b, c = map(np.asarray, (a, b, c))
    p = a[None] + xi[:, None]*(b - a)[None] + eta[:, None]*(c - a)[None]
    det = abs((b - a)[0]*(c - a)[1] - (b - a)[1]*(c - a)[0])
    return det*np.sum(w*fun(p[:, 0], p[:, 1]))


def source_clipped(x, y):
    return 56.0*np.clip(1.0 - np.abs(x) - np.abs(y), 0.0, None)**6


def model_with_source(src, boundary='paper', nx=tm.NX):
    """Build the model with another heat source (the module's `source` is swapped temporarily)."""
    keep = tm.source
    tm.source = src
    try:
        return tm.ThermalCompliance(nx=nx, boundary=boundary)
    finally:
        tm.source = keep


def check_design():
    print('1. Sampling design (thermal_compliance.npz)')
    d = np.load(os.path.join(HERE, 'thermal_compliance.npz'))
    th, E, G = d['theta'], d['E'], d['dE_dtheta']
    print(f'   theta {th.shape} {th.dtype}, range [{th.min():.3g}, {th.max():.7f}]; '
          f'entries at exactly 0 or 1: {int(np.sum((th == 0) | (th == 1)))}; '
          f'duplicate rows: {len(th) - len(np.unique(th, axis=0))}')
    z = th.astype(np.float64)
    z = z - z.mean(axis=0)
    kurt = (z**4).mean(axis=0)/(z**2).mean(axis=0)**2
    print(f'   per-column kurtosis {kurt.min():.2f} to {kurt.max():.2f} (mean {kurt.mean():.3f}; 1.8 for a uniform)')
    print(f'   E in [{E.min():.4f}, {E.max():.4f}]; samples with E > E_max = {tm.E_MAX}: {np.sum(E > tm.E_MAX)} of {len(E)}; '
          f'all gradient entries < 0: {bool(np.all(G < 0))}')


def check_source():
    print('2. Heat source and boundary list')
    f = tm.source
    tot = dia = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            o, px, py, cc = (0, 0), (sx, 0), (0, sy), (sx, sy)
            dia += integrate_on_triangle(f, o, px, py)
            tot += integrate_on_triangle(f, o, px, py) + integrate_on_triangle(f, cc, px, py)
    print(f'   integral of f over Omega {tot:.6f}; inside |x| + |y| <= 1 {dia:.6f}; '
          f'in the four corner triangles {tot - dia:.6f} ({(tot - dia)/tot:.1%} of the total)')
    half = np.full(tm.M_INPUTS, 0.5)
    E = {}
    for src, sname in ((tm.source, 'as written'), (source_clipped, 'clipped')):
        for bnd in ('casm', 'paper'):
            E[sname, bnd] = model_with_source(src, boundary=bnd).energy(half)
            print(f'   theta = 0.5, source {sname:10s}, boundary={bnd!r:8s}: E = {E[sname, bnd]:.6f}')
    for sname in ('as written', 'clipped'):
        print(f'   source {sname}: the full boundary changes E by '
              f'{100*(E[sname, "paper"]/E[sname, "casm"] - 1):+.4f} % from the CASM list')
    print(f'   clipping the source changes E (full boundary) by '
          f'{100*(E["clipped", "paper"]/E["as written", "paper"] - 1):+.2f} %')


def check_mesh():
    print('3. Discretization: E at theta = 0.5 (boundary="paper")')
    Es = {}
    for nx in (16, 32, 64):
        model = tm.ThermalCompliance(nx=nx)
        Es[nx] = model.energy(np.full(model.n_elem, 0.5))
        print(f'   {nx} x {nx} mesh ({model.n_elem} elements): E = {Es[nx]:.6f}')
    print(f'   16 x 16 relative to 64 x 64: {100*(Es[16]/Es[64] - 1):+.2f} %')


def check_load():
    print('4. Load quadrature: exact load integration against the 4-point rule')
    model = tm.ThermalCompliance()
    xi, eta, w = gauss_triangle(6)
    phi, _ = tm._p2_basis(xi, eta)                                  # (6, q)
    v = model.coord[model.conn[:, :3]]
    T = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=2)
    detT = np.abs(np.linalg.det(T))
    xq = v[:, 0][:, None, :] + np.einsum('eij,qj->eqi', T, np.column_stack([xi, eta]))
    Fe = np.einsum('q,eq,iq->ei', w, tm.source(xq[..., 0], xq[..., 1]), phi)*detT[:, None]
    F = np.zeros(model.n_nodes)
    np.add.at(F, model.conn, Fe)
    data = np.load(os.path.join(HERE, 'thermal_compliance.npz'))
    for name, th in (('theta = 0.5', np.full(model.n_elem, 0.5)),
                     ('sample 0', data['theta'][0].astype(np.float64))):
        E4 = model.energy(th)
        keep = model.F
        model.F = F
        U = model.solve(th)
        model.F = keep
        Ex = 0.5*float(F @ U)
        print(f'   {name}: E = {E4:.6f} (4-point rule), {Ex:.6f} (exact load); relative change {abs(Ex - E4)/E4:.1e}')


def main():
    t0 = time.time()
    check_design()
    check_source()
    check_mesh()
    check_load()
    print(f'run time {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
