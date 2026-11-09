"""Recompute the accuracy numbers of DATA.md that the notebook and generate_data.py do not print.

    python check_accuracy.py

1. Adjoint gradients against complex-step derivatives (step 1e-30, exact to rounding) and
   central differences (step 1e-3) along random directions, at rows 0-9 of inputs.csv.
2. The scaling identity sum_k df/dx_k = -f (f is proportional to 1/E for a uniform change of
   E), at rows 0-9.
3. Linear solves at rows 0-9: normwise backward error ||K u - b||_inf / (||K||_inf ||u||_inf), and
   the 1-norm condition number of K at row 0, whose product with the unit round-off bounds the
   relative accuracy of the computed displacements and gradients.
4. Discretization: f at the nominal field (x = 0) on the shipped mesh and on one and two uniform
   refinements (each triangle split into four, children keep the parent's modulus), and the
   eigenvalues 1-5 of the active subspace matrix from the first 200 samples on the shipped mesh
   and on one refinement (gradient with respect to a parent triangle = sum over its children).
   The refinement keeps the polygonal boundary, so it measures the finite element error on the
   meshed geometry, not the error of approximating the curved boundary.

Only numpy, scipy and the standard library are imported. Runs in about a minute.
"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wrench_mesh  # noqa: E402
import wrench_model as wm  # noqa: E402

ROWS = 10
N_EIG = 200


def read_csv(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def refine(p, t, segs):
    """Uniform (red) refinement; returns nodes, triangles, boundary segments and the parent of
    each child triangle."""
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.ravel()
    mid = len(p) + inv.reshape(3, -1).T                 # midpoint node of edges (01, 12, 20)
    p2 = np.vstack([p, 0.5 * (p[ue[:, 0]] + p[ue[:, 1]])])
    a, b, c = t.T
    mab, mbc, mca = mid.T
    t2 = np.vstack([np.c_[a, mab, mca], np.c_[mab, b, mbc], np.c_[mca, mbc, c], np.c_[mab, mbc, mca]])
    parent = np.tile(np.arange(len(t)), 4)
    key = {tuple(k): len(p) + i for i, k in enumerate(ue.tolist())}
    s2 = []
    for na, nb, kind in segs.tolist():
        m = key[tuple(sorted((na, nb)))]
        s2 += [(na, m, kind), (m, nb, kind)]
    return p2, t2, np.array(s2), parent


def model_on(p, t, segs, tmp):
    poi = int(np.argmin(((p - wm.POI) ** 2).sum(1)))
    wrench_mesh.write_mesh(tmp, p, t, segs, poi)
    return wm.Wrench(tmp, load_kl=False)


def main():
    t0 = time.perf_counter()
    model = wm.Wrench(HERE)
    xi = read_csv('inputs.csv')
    f_ship = read_csv('outputs.csv')[:, 0]
    rng = np.random.default_rng(0)

    print(f'1-3. gradients and solves at rows 0-{ROWS - 1}')
    e_cs, e_fd, e_sc, res = [], [], [], []
    for i in range(ROWS):
        x = model.field(xi[i])
        f, g = model.f_and_grad(x)
        for _ in range(3):
            v = rng.standard_normal(model.n_el)
            cs = model.f(x + 1e-30j * v).imag / 1e-30
            fd = (model.f(x + 1e-3 * v) - model.f(x - 1e-3 * v)) / 2e-3
            e_cs.append(abs(g @ v - cs) / abs(cs))
            e_fd.append(abs(g @ v - fd) / abs(fd))
        e_sc.append(abs(g.sum() + f) / abs(f))
        K = model.stiffness(np.exp(x))
        u = model.displacement(x).ravel()[model.free]
        res.append(np.abs(K @ u - model.b).max() / (abs(K).sum(axis=1).max() * np.abs(u).max()))
        if i == 0:
            cond = abs(K).sum(axis=0).max() * np.abs(np.linalg.inv(K.toarray())).sum(axis=0).max()
        assert abs(f - f_ship[i]) <= 1e-9 * abs(f_ship[i])
    print(f'  adjoint vs complex step, 30 directions: max relative error {max(e_cs):.1e}, median {np.median(e_cs):.1e}')
    print(f'  adjoint vs central differences (h = 1e-3): max relative error {max(e_fd):.1e}')
    print(f'  scaling identity |sum_k df/dx_k + f| / |f|: max {max(e_sc):.1e}')
    print(f'  solve backward error ||K u - b|| / (||K|| ||u||) (inf-norms): max {max(res):.1e}; '
          f'cond_1(K) at row 0: {cond:.1e}')

    print('4. uniform refinement of the shipped mesh')
    p, t, s = wm.read_mesh(HERE)
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        p1, t1, s1, par1 = refine(p, t, s)
        p2, t2, s2, par2 = refine(p1, t1, s1)
        m1 = model_on(p1, t1, s1, tmp1)
        m2 = model_on(p2, t2, s2, tmp2)
        f0 = [model.f(np.zeros(len(t))), m1.f(np.zeros(len(t1))), m2.f(np.zeros(len(t2)))]
        print(f'  f(x = 0) on {len(t)}, {len(t1)}, {len(t2)} triangles: ' + ', '.join(f'{v:.6f}' for v in f0))
        rich = f0[2] + (f0[2] - f0[1]) ** 2 / ((f0[1] - f0[0]) - (f0[2] - f0[1]))
        print(f'  relative change {len(t)} -> {len(t1)}: {abs(f0[1] - f0[0]) / abs(f0[1]):.2e}; '
              f'{len(t1)} -> {len(t2)}: {abs(f0[2] - f0[1]) / abs(f0[2]):.2e}; '
              f'shipped mesh vs Aitken extrapolation: {abs(f0[0] - rich) / abs(rich):.2e}')
        G0 = read_csv('gradients.csv')[:N_EIG]
        G1 = np.empty_like(G0)
        for i in range(N_EIG):
            x = model.field(xi[i])
            _, g1 = m1.f_and_grad(x[par1])
            G1[i] = model.Phi.T @ np.bincount(par1, weights=g1, minlength=len(t))
        l0 = np.linalg.svd(G0, compute_uv=False)[:5] ** 2 / N_EIG
        l1 = np.linalg.svd(G1, compute_uv=False)[:5] ** 2 / N_EIG
        print(f'  eigenvalues 1-5 from the first {N_EIG} samples, shipped mesh: ' + ', '.join(f'{v:.5g}' for v in l0))
        print(f'  same samples, {len(t1)} triangles:                 ' + ', '.join(f'{v:.5g}' for v in l1))
        print('  relative change: ' + ', '.join(f'{abs(a - b) / b:.2e}' for a, b in zip(l0, l1)))
    print(f'total {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
