"""Transient advection-diffusion of a tracer with a log-normal KLE diffusivity.

A numpy/scipy re-implementation of the forward model behind PyApprox's
``setup_multi_index_advection_diffusion_benchmark(..., time_scenario=True)``
(PyApprox 1.0.3, MIT licence; J. D. Jakeman, Environ. Model. Softw. 2023,
doi:10.1016/j.envsoft.2023.105825). No PyApprox code is copied; the
discretisation was rebuilt from the PyApprox 1.0.3 source and checked
against it value by value (see DATA.md, "Accuracy").

Model (unit square, x = (x1, x2) in [0, 1]^2, t in (0, T]):

    dc/dt = div(K grad c) - v . grad c + G(x, t)
    K grad c . n + alpha c = 0 on the boundary          (Robin, alpha = 0.1)
    K(x, z) = exp( sum_d phi_d(x) z_d ),  z_d ~ N(0, 1) i.i.d.
    G(x, 0) = s exp(-|x - x_src|^2 / h^2),  G(x, t > 0) = 0

with s = 100, h = 0.1, x_src = (0.25, 0.75), v = (1, 0), T = 0.2.
The initial condition c0 is the steady state with the source on.

Discretisation: Chebyshev-Gauss-Lobatto collocation of polynomial order
`order` in each direction ((order+1)^2 nodes), boundary rows replaced by the
Robin condition, and PyApprox's two-stage Crank-Nicolson ("im_crank2") with
64 steps of dt = T/64. The default order is ORDER_MAIN = 21, the converged
mesh of the data set's main files; order=ORDER_F0=11 is PyApprox's
highest-fidelity member f0 (12 x 12 nodes), which this code matches to
rounding. The source is evaluated at t = 0 in the first step, as in
PyApprox. The KLE basis phi_d is read from kle_basis.csv (computed by
PyApprox 1.0.3 on the 12 x 12 mesh); finer meshes interpolate the log
field from that mesh, as PyApprox's InterpolatedMeshKLE does.

QoI:  q(z) = integral over S = [0.75, 1] x [0, 0.25] of c(x, T, z) dx,
computed with an (order+2)-point Gauss-Legendre rule per direction on S
applied to the collocation interpolant.

Gradients dq/dz are exact derivatives of the discrete model, computed in
tangent-linear (forward sensitivity) mode alongside the forward solve.
"""
import os

import numpy as np
from scipy.linalg import lu_factor, lu_solve

_HERE = os.path.dirname(os.path.abspath(__file__))
KLE_FILE = os.path.join(_HERE, "kle_basis.csv")

# Model constants (PyApprox 1.0.3 defaults; paper Sec. 3, p. 7-8)
SOURCE_AMP = 100.0
SOURCE_SCALE = 0.1
SOURCE_LOC = (0.25, 0.75)
VELOCITY = (1.0, 0.0)
ROBIN_ALPHA = 0.1
FINAL_TIME = 0.2
SUBDOMAIN = (0.75, 1.0, 0.0, 0.25)
KLE_ORDER = 11          # the KLE basis lives on the 12 x 12 mesh
ORDER_F0 = 11           # PyApprox's highest-fidelity mesh f0
ORDER_MAIN = 21         # converged mesh used for the main data files (default)
M_KLE = 10              # number of KLE terms shipped in kle_basis.csv


def cheb_points(order):
    """Chebyshev-Gauss-Lobatto points on [-1, 1], ascending."""
    return -np.cos(np.linspace(0.0, np.pi, order + 1))


def cheb_derivative_matrix(order):
    """Collocation derivative matrix on the ascending points of cheb_points."""
    x = cheb_points(order)
    c = np.ones(order + 1)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(order + 1)
    X = np.tile(x, (order + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(order + 1))
    D -= np.diag(D.sum(axis=1))
    return D


def lagrange_matrix(order, xx):
    """Rows: values at xx (in [-1, 1]) of the Lagrange basis on cheb_points."""
    x = cheb_points(order)
    w = (-1.0) ** np.arange(order + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    xx = np.atleast_1d(np.asarray(xx, dtype=float))
    diff = xx[:, None] - x[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-15, rtol=0.0)
    diff[exact] = 1.0
    tmp = w / diff
    B = tmp / tmp.sum(axis=1, keepdims=True)
    rows = np.where(exact.any(axis=1))[0]
    for r in rows:
        B[r] = exact[r].astype(float)
    return B


def load_kle_basis(path=KLE_FILE):
    data = np.loadtxt(path, delimiter=",")
    return data[:, :2], data[:, 2:]


class TracerModel:
    """Discrete transient advection-diffusion model with exact gradients.

    Parameters
    ----------
    order : int
        Polynomial order per direction (mesh is (order+1)^2 nodes). Default
        21 (ORDER_MAIN, converged; the data set's main files); 11 (ORDER_F0)
        is PyApprox's highest-fidelity mesh f0.
    nsteps : int
        Number of Crank-Nicolson steps over [0, T]. 64 (dt = 0.003125) is
        PyApprox's highest-fidelity time step.
    m : int
        Number of KLE terms used (<= 10).
    """

    def __init__(self, order=ORDER_MAIN, nsteps=64, m=M_KLE, kle_path=KLE_FILE):
        if not (1 <= m <= M_KLE):
            raise ValueError(f"m must be in 1..{M_KLE}")
        if order < 2 or nsteps < 1:
            raise ValueError("order >= 2 and nsteps >= 1 required")
        self.order, self.nsteps, self.m = int(order), int(nsteps), int(m)
        self.dt = FINAL_TIME / self.nsteps
        n1 = self.order + 1
        self.n = n1 * n1
        # 1D operators on [0, 1] (derivative scaled by 2 from [-1, 1])
        D1 = 2.0 * cheb_derivative_matrix(self.order)
        self.D1 = D1
        I1 = np.eye(n1)
        # node index = i1 + n1 * i2 (x1 varies fastest)
        self.Dx = np.kron(I1, D1)
        self.Dy = np.kron(D1, I1)
        s = 0.5 * (cheb_points(self.order) + 1.0)
        self.pts = np.vstack([np.tile(s, n1), np.repeat(s, n1)])
        # boundaries: left, right (with corners), bottom, top (no corners)
        ii = np.arange(n1)
        self.bndry = [ii * n1, ii * n1 + self.order,
                      np.arange(1, self.order), self.n - n1 + np.arange(1, self.order)]
        self.normals = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
        self.bidx = np.concatenate(self.bndry)
        mask = np.ones(self.n, bool)
        mask[self.bidx] = False
        self.iidx = np.where(mask)[0]
        # KLE basis on this mesh: interpolate from the 12x12 KLE mesh
        kle_pts, phi = load_kle_basis(kle_path)
        phi = phi[:, :self.m]
        if self.order == KLE_ORDER:
            if not np.allclose(kle_pts, self.pts.T, atol=1e-14, rtol=0):
                raise RuntimeError("KLE mesh does not match the model mesh")
            self.phi = phi.copy()
        else:
            can = 2.0 * self.pts - 1.0
            Bx = lagrange_matrix(KLE_ORDER, can[0])
            By = lagrange_matrix(KLE_ORDER, can[1])
            nk = KLE_ORDER + 1
            # value at node p = sum_{a,b} Bx[p,a] By[p,b] phi[a + nk*b]
            phi3 = phi.reshape(nk, nk, self.m)        # [b, a, d]
            self.phi = np.einsum("pa,pb,bad->pd", Bx, By, phi3)
        # source term at t = 0
        r2 = (self.pts[0] - SOURCE_LOC[0]) ** 2 + (self.pts[1] - SOURCE_LOC[1]) ** 2
        self.source = SOURCE_AMP * np.exp(-r2 / SOURCE_SCALE ** 2)
        # QoI: Gauss-Legendre (order+2 points per direction) on the subdomain
        gx, gw = np.polynomial.legendre.leggauss(self.order + 2)
        a1, b1, a2, b2 = SUBDOMAIN
        q1 = a1 + (gx + 1.0) * 0.5 * (b1 - a1)
        q2 = a2 + (gx + 1.0) * 0.5 * (b2 - a2)
        w1 = gw * 0.5 * (b1 - a1)
        w2 = gw * 0.5 * (b2 - a2)
        Q1 = lagrange_matrix(self.order, 2.0 * q1 - 1.0)
        Q2 = lagrange_matrix(self.order, 2.0 * q2 - 1.0)
        # qoi = sum_{i,j} w1_i w2_j u(q1_i, q2_j) = (w2 Q2) x (w1 Q1) . u
        self.qvec = np.kron(w2 @ Q2, w1 @ Q1)

    # ------------------------------------------------------------------
    def diffusivity(self, z):
        z = np.asarray(z, dtype=float).ravel()
        if z.shape[0] != self.m:
            raise ValueError(f"expected {self.m} KLE coefficients")
        if not np.all(np.isfinite(z)):
            raise ValueError("non-finite input")
        return np.exp(self.phi @ z)

    def _operators(self, k):
        Dx, Dy = self.Dx, self.Dy
        L = Dx @ (k[:, None] * Dx) + Dy @ (k[:, None] * Dy) \
            - VELOCITY[0] * Dx - VELOCITY[1] * Dy
        R = np.zeros((self.n, self.n))
        for idx, (nx, ny) in zip(self.bndry, self.normals):
            R[idx] = k[idx, None] * (nx * Dx[idx] + ny * Dy[idx])
            R[idx, idx] += ROBIN_ALPHA
        return L, R

    def _apply_x(self, X):
        """Dx @ X for X of shape (n, m), using the Kronecker structure."""
        n1 = self.order + 1
        X3 = X.reshape(n1, n1, -1)                 # [i2, i1, d]
        return np.einsum("ab,jbd->jad", self.D1, X3).reshape(self.n, -1)

    def _apply_y(self, X):
        n1 = self.order + 1
        X3 = X.reshape(n1, n1, -1)
        return np.einsum("ab,bid->aid", self.D1, X3).reshape(self.n, -1)

    def _dL(self, kphi, u):
        """Columns d: (dL/dz_d) u."""
        return (self._apply_x(kphi * (self.Dx @ u)[:, None])
                + self._apply_y(kphi * (self.Dy @ u)[:, None]))

    def _dR(self, kphi, u):
        """Columns d: (dR/dz_d) u (nonzero on boundary rows only)."""
        out = np.zeros((self.n, self.m))
        ux, uy = self.Dx @ u, self.Dy @ u
        for idx, (nx, ny) in zip(self.bndry, self.normals):
            out[idx] = kphi[idx] * (nx * ux[idx] + ny * uy[idx])[:, None]
        return out

    def solve(self, z, grad=False, return_states=False):
        """Return q(z), and dq/dz if grad=True."""
        k = self.diffusivity(z)
        kphi = k[:, None] * self.phi
        L, R = self._operators(k)
        I, B, n = self.iidx, self.bidx, self.n
        dt2 = 0.5 * self.dt
        Id = np.eye(n)
        # steady state with the source on
        Es = np.empty((n, n)); Es[I] = L[I]; Es[B] = R[B]
        rs = np.zeros(n); rs[I] = -self.source[I]
        lus = lu_factor(Es)
        u = lu_solve(lus, rs)
        u0 = u.copy()
        if grad:
            gs = np.zeros((n, self.m))
            gs[I] = -self._dL(kphi, u)[I]
            gs -= self._dR(kphi, u)
            du = lu_solve(lus, gs)
        E0 = np.empty((n, n)); E0[I] = Id[I]; E0[B] = R[B]
        E1 = np.empty((n, n)); E1[I] = Id[I] - dt2 * L[I]; E1[B] = R[B]
        lu0, lu1 = lu_factor(E0), lu_factor(E1)
        for step in range(self.nsteps):
            f_n = self.source if step == 0 else 0.0   # source on only at t = 0
            # stage 0
            r0 = np.zeros(n)
            r0[I] = (u + dt2 * (L @ u + f_n))[I]
            Y0 = lu_solve(lu0, r0)
            # stage 1 (forcing at t_{n+1} > 0 is zero)
            r1 = np.zeros(n)
            r1[I] = (u + dt2 * (L @ (Y0 - u)))[I]
            Y1 = lu_solve(lu1, r1)
            if grad:
                dLu = self._dL(kphi, u)
                g0 = np.zeros((n, self.m))
                g0[I] = (du + dt2 * (dLu + L @ du))[I]
                g0 -= self._dR(kphi, Y0)
                dY0 = lu_solve(lu0, g0)
                g1 = np.zeros((n, self.m))
                g1[I] = (du + dt2 * (self._dL(kphi, Y0 - u) + L @ (dY0 - du))
                         + dt2 * self._dL(kphi, Y1))[I]
                g1 -= self._dR(kphi, Y1)
                dY1 = lu_solve(lu1, g1)
                du = dY0 + dY1 - du
            u = Y0 + Y1 - u
        q = float(self.qvec @ u)
        if not np.isfinite(q):
            raise FloatingPointError("non-finite QoI")
        out = [q]
        if grad:
            out.append(self.qvec @ du)
        if return_states:
            out.append((u0, u))
        return out[0] if len(out) == 1 else tuple(out)

    def qoi(self, z):
        return self.solve(z)

    def qoi_and_grad(self, z):
        return self.solve(z, grad=True)


_DEFAULT = None


def tracer_qoi(z):
    """q(z) on the default (order-21, 64-step) discretisation."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TracerModel()
    return _DEFAULT.qoi(z)


def tracer_qoi_and_grad(z):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TracerModel()
    return _DEFAULT.qoi_and_grad(z)
