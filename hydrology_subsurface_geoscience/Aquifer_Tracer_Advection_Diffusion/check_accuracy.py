"""Recompute the numbers of DATA.md (Aquifer_Tracer_Advection_Diffusion) that neither the notebook
nor generate_data.py prints.

    python check_accuracy.py               all checks (several minutes; see DATA.md for the time)
    python check_accuracy.py --skip-slow   checks 1-5 only (about a minute)

Run it from any directory; it reads the shipped files next to this file and evaluates
tracer_model.py. Each line prints the recomputed value; DATA.md quotes these values.

Checks:
 1. kle_basis.csv against the squared-exponential kernel matrix (length scale 0.5, sigma 1, no
    quadrature weights) on its 12 x 12 nodes: orthonormality of v_d = phi_d / sqrt(lambda_d), the
    eigen-residual, the match of lambda_d with the 10 largest eigenvalues, lambda_11, and the share
    of the field variance carried by the 10 modes.
 2. Time step: q at rows 0-3 of inputs.csv with 16, 64 and 1024 Crank-Nicolson steps, on f0
    (order 11) and on order 21 (16 steps is the coarsest time step of PyApprox's fidelities).
 3. Mesh: q at rows 0-3 with 64 steps on orders 7, 11, 15, 21, 27 and 35 (order 7 is PyApprox's
    coarsest mesh, order 11 its f0).
 4. q(0) on f0 (compared with PyApprox 1.0.3 by pyapprox_reference.py --compare, which needs
    PyApprox; the PyApprox value is quoted, not recomputed) and the time of one f0 value.
 5. Gradients against central differences with h = 1e-4 and 1e-5 at rows 0-4 on order 21 and f0
    (generate_data.py --fd prints h = 1e-3).
 6. (slow) The paper's printed versions of two settings, on order 21, rows 0-299, exact gradients:
    (a) the source s/(2 pi h^2) exp(-|x - x_src|^2 / (2 h^2)) instead of the code's
    s exp(-|x - x_src|^2 / h^2); (b) the space-time QoI, the integral of q(t) over [0, T] by the
    trapezoid rule over the Crank-Nicolson levels, with tangent-linear gradients (checked against
    central differences at row 0). For each: lambda_k / lambda_k+1, k = 1, 2, 3, and the distance
    of its k-dimensional subspace (k = 1, 2, 3) from the code's (gradients.csv, same rows).

Eigenvalues are those of C = G^T G / N. Deterministic. Imports numpy, scipy and the standard library.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tracer_model as tm  # noqa: E402


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=",", ndmin=2)


def eig(G):
    lam, W = np.linalg.eigh(G.T @ G / len(G))
    return lam[::-1], W[:, ::-1]


def sdist(A, B, k):
    return np.linalg.norm(A[:, :k] @ A[:, :k].T - B[:, :k] @ B[:, :k].T, 2)


def check_kle():
    pts, phi = tm.load_kle_basis()
    lam = (phi ** 2).sum(axis=0)
    V = phi / np.sqrt(lam)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * 0.5 ** 2))
    ev = np.linalg.eigvalsh(K)[::-1]
    orth = np.abs(V.T @ V - np.eye(V.shape[1])).max()
    res = np.abs(K @ V - V * lam).max()
    match = np.abs(lam - ev[:10]).max()
    print(f"1. KLE basis: v_d orthonormal to {orth:.0e}; eigen-residual max |K v - lambda v| {res:.0e}; "
          f"lambda_d vs the 10 largest kernel eigenvalues, max |difference| {match:.0e}; "
          f"lambda = {', '.join(f'{v:.3f}' for v in lam)}; lambda_11 = {ev[10]:.4f}; "
          f"10 modes carry {lam.sum() / len(pts):.2%} of the field variance on the {len(pts)} nodes")


def check_time_step(X):
    rows = range(4)
    for order in (tm.ORDER_F0, tm.ORDER_MAIN):
        q = {n: np.array([tm.TracerModel(order=order, nsteps=n).qoi(X[r]) for r in rows])
             for n in (16, 64, 1024)}
        d64 = 100 * np.abs(q[64] - q[1024]) / q[1024]
        d16 = 100 * np.abs(q[16] - q[64]) / q[64]
        print(f"2. order {order}, rows 0-3: |q(64 steps) - q(1024)| / q(1024) = "
              f"{', '.join(f'{v:.3f}' for v in d64)} %; |q(16) - q(64)| / q(64) = "
              f"{', '.join(f'{v:.3f}' for v in d16)} %")


def check_mesh(X):
    rows = range(4)
    orders = (7, 11, 15, 21, 27, 35)
    q = {p: np.array([tm.TracerModel(order=p).qoi(X[r]) for r in rows]) for p in orders}
    for p in orders[:-1]:
        d = 100 * np.abs(q[p] - q[35]) / q[35]
        print(f"3. order {p} vs order 35, rows 0-3, 64 steps: |dq|/q = {', '.join(f'{v:.2f}' for v in d)} %")
    d = 100 * np.abs(q[7] - q[11]) / q[11]
    print(f"   order 7 vs order 11 (f0): |dq|/q = {', '.join(f'{v:.1f}' for v in d)} %; "
          f"q at row 0: order 7 {q[7][0]:.3f}, order 11 {q[11][0]:.3f}")


def check_q0():
    m = tm.TracerModel(order=tm.ORDER_F0)
    t0 = time.perf_counter()
    for _ in range(20):
        q0 = m.qoi(np.zeros(10))
    dt = (time.perf_counter() - t0) / 20
    print(f"4. q(0) on f0 = {q0:.12f} (PyApprox 1.0.3: 0.537336364600886, external run, see DATA.md); "
          f"one f0 value takes {dt:.3f} s here")


def check_fd(X):
    for order in (tm.ORDER_MAIN, tm.ORDER_F0):
        m = tm.TracerModel(order=order)
        for h in (1e-4, 1e-5):
            errs = []
            for r in range(5):
                g = m.qoi_and_grad(X[r])[1]
                fd = np.array([(m.qoi(X[r] + h * e) - m.qoi(X[r] - h * e)) / (2 * h) for e in np.eye(10)])
                errs.append(np.linalg.norm(fd - g) / np.linalg.norm(g))
            print(f"5. order {order}, h = {h:.0e}, rows 0-4: relative 2-norm error "
                  f"{', '.join(f'{v:.1e}' for v in errs)}")


class PaperSource(tm.TracerModel):
    """The source as the paper prints it (p. 7): s / (2 pi h^2) exp(-|x - x_src|^2 / (2 h^2))."""

    def __init__(self, **kw):
        super().__init__(**kw)
        r2 = (self.pts[0] - tm.SOURCE_LOC[0]) ** 2 + (self.pts[1] - tm.SOURCE_LOC[1]) ** 2
        h = tm.SOURCE_SCALE
        self.source = tm.SOURCE_AMP / (2 * np.pi * h ** 2) * np.exp(-r2 / (2 * h ** 2))


def space_time(model, z, grad=True):
    """Integral over [0, T] of q(t) (trapezoid rule over the Crank-Nicolson levels) and its gradient.
    The time loop is that of TracerModel.solve; the QoI is accumulated at every level."""
    k = model.diffusivity(z)
    kphi = k[:, None] * model.phi
    L, R = model._operators(k)
    I, B, n = model.iidx, model.bidx, model.n
    dt2 = 0.5 * model.dt
    Id = np.eye(n)
    Es = np.empty((n, n)); Es[I] = L[I]; Es[B] = R[B]
    rs = np.zeros(n); rs[I] = -model.source[I]
    lus = lu_factor(Es)
    u = lu_solve(lus, rs)
    gs = np.zeros((n, model.m)); gs[I] = -model._dL(kphi, u)[I]; gs -= model._dR(kphi, u)
    du = lu_solve(lus, gs)
    E0 = np.empty((n, n)); E0[I] = Id[I]; E0[B] = R[B]
    E1 = np.empty((n, n)); E1[I] = Id[I] - dt2 * L[I]; E1[B] = R[B]
    lu0, lu1 = lu_factor(E0), lu_factor(E1)
    w = np.full(model.nsteps + 1, model.dt); w[0] = w[-1] = dt2
    Q = w[0] * (model.qvec @ u)
    dQ = w[0] * (model.qvec @ du)
    for step in range(model.nsteps):
        f_n = model.source if step == 0 else 0.0
        r0 = np.zeros(n); r0[I] = (u + dt2 * (L @ u + f_n))[I]
        Y0 = lu_solve(lu0, r0)
        r1 = np.zeros(n); r1[I] = (u + dt2 * (L @ (Y0 - u)))[I]
        Y1 = lu_solve(lu1, r1)
        if grad:
            g0 = np.zeros((n, model.m)); g0[I] = (du + dt2 * (model._dL(kphi, u) + L @ du))[I]
            g0 -= model._dR(kphi, Y0)
            dY0 = lu_solve(lu0, g0)
            g1 = np.zeros((n, model.m))
            g1[I] = (du + dt2 * (model._dL(kphi, Y0 - u) + L @ (dY0 - du)) + dt2 * model._dL(kphi, Y1))[I]
            g1 -= model._dR(kphi, Y1)
            dY1 = lu_solve(lu1, g1)
            du = dY0 + dY1 - du
        u = Y0 + Y1 - u
        Q += w[step + 1] * (model.qvec @ u)
        if grad:
            dQ += w[step + 1] * (model.qvec @ du)
    return Q, dQ


def check_paper_variants(X, G):
    rows = 300
    base = tm.TracerModel()
    h = 1e-3
    q0, g0 = space_time(base, X[0])
    fd = np.array([(space_time(base, X[0] + h * e, False)[0] - space_time(base, X[0] - h * e, False)[0]) / (2 * h)
                   for e in np.eye(10)])
    print(f"6. space-time QoI, row 0: tangent-linear gradient vs central differences (h = 1e-3): "
          f"{np.linalg.norm(fd - g0) / np.linalg.norm(g0):.1e}")
    lam_c, W_c = eig(G[:rows])
    print(f"   code (gradients.csv, rows 0-{rows - 1}): ratios "
          + ", ".join(f"{lam_c[k] / lam_c[k + 1]:.2f}" for k in range(3)))
    paper = PaperSource()
    variants = {
        "paper's printed source": lambda z: paper.qoi_and_grad(z)[1],
        "space-time QoI": lambda z: space_time(base, z)[1],
    }
    for name, fn in variants.items():
        t0 = time.perf_counter()
        Gv = np.array([fn(X[r]) for r in range(rows)])
        lam, W = eig(Gv)
        ds = [sdist(W, W_c, k) for k in (1, 2, 3)]
        print(f"   {name}: ratios " + ", ".join(f"{lam[k] / lam[k + 1]:.2f}" for k in range(3))
              + "; distance to the code's subspace, k = 1, 2, 3: "
              + ", ".join(f"{d:.3f} ({np.degrees(np.arcsin(min(d, 1))):.1f} deg)" for d in ds)
              + f"  [{time.perf_counter() - t0:.0f} s]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-slow", action="store_true", help="skip check 6")
    args = ap.parse_args()
    t0 = time.perf_counter()
    X = load("inputs.csv")
    G = load("gradients.csv")
    check_kle()
    check_time_step(X)
    check_mesh(X)
    check_q0()
    check_fd(X)
    if not args.skip_slow:
        check_paper_variants(X, G)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
