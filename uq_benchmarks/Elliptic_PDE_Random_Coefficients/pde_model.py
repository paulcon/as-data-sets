"""Elliptic PDE model -div(a grad u) = 1 on [0,1]^2, numpy/scipy only.

Replaces the numpad-based pde_model.py files in
uq_benchmarks/Elliptic_PDE_Random_Coefficients/ and
uq_benchmarks/PDE_Inverse_Problem_MCMC/. The two copies must stay
byte-identical (uq_benchmarks/check_pde_model.py checks this). Reproduces
the legacy finite-volume residual exactly (same grid, same face averaging,
same boundary padding) and computes gradients with a hand-coded discrete
adjoint. See specs/numpad_replacement.md.

Grid: nx x nx cells (nx = 100), coefficient a is cell-centred, u lives on
the (nx+1) x (nx+1) vertices; unknowns are the (nx-1) x (nx-1) interior
vertices. Array axis 0 = "row" i (s2, bottom..top), axis 1 = column j
(s1, left..right).
  u[0, :] = u[nx, :] = 0   (Dirichlet, bottom and top)
  u[:, 0] = 0              (Dirichlet, left)
  u[:, nx] = u[:, nx-1]    (ghost copy -> zero flux on the right faces)
"""
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _EllipticPDE:
    """Shared discretization, solve, and adjoint machinery."""

    def __init__(self, corrcase='short', datadir=None):
        if datadir is None:
            datadir = os.path.dirname(os.path.abspath(__file__))
        kl = np.load(os.path.join(datadir, 'kl-{}.npz'.format(corrcase)))
        K, lam = kl['K'], kl['lam']
        self.B = K * np.sqrt(lam.reshape((1, K.shape[1])))
        self.f = 1.0
        self.nx = int(np.sqrt(self.B.shape[0]))
        self.dx = 1.0 / self.nx
        self.m = K.shape[1]
        self._build_index()

    # ------------------------------------------------------------------
    # static index structure of the stiffness matrix
    # ------------------------------------------------------------------
    def _build_index(self):
        nx, n = self.nx, self.nx - 1
        # node id on the padded (nx+1)x(nx+1) vertex grid; -1 = not an unknown
        nid = -np.ones((nx + 1, nx + 1), dtype=np.int64)
        nid[1:nx, 1:nx] = np.arange(n * n).reshape(n, n)
        # horizontal faces (flux along axis 1) between vertices (i,j),(i,j+1),
        # i = 1..nx-1, j = 0..nx-2 (the j = nx-1 face is the zero-flux one)
        hp, hq = nid[1:nx, 0:nx - 1], nid[1:nx, 1:nx]
        # vertical faces (flux along axis 0) between vertices (i,j),(i+1,j),
        # i = 0..nx-1, j = 1..nx-1
        vp, vq = nid[0:nx, 1:nx], nid[1:nx + 1, 1:nx]
        p = np.concatenate([hp.ravel(), vp.ravel()])
        q = np.concatenate([hq.ravel(), vq.ravel()])
        self._nh = hp.size
        self._p, self._q = p, q
        # COO pattern: +k at (p,p),(q,q); -k at (p,q),(q,p), Dirichlet nodes dropped
        rows, cols, sgn, face = [], [], [], []
        fidx = np.arange(p.size)
        for r, c, s in ((p, p, 1.0), (q, q, 1.0), (p, q, -1.0), (q, p, -1.0)):
            keep = (r >= 0) & (c >= 0)
            rows.append(r[keep]); cols.append(c[keep])
            sgn.append(np.full(keep.sum(), s)); face.append(fidx[keep])
        self._rows = np.concatenate(rows)
        self._cols = np.concatenate(cols)
        self._sgn = np.concatenate(sgn)
        self._face = np.concatenate(face)
        self.N = n * n
        # fixed CSC pattern: map every COO entry to its slot in the CSC data array
        key = self._cols * self.N + self._rows
        ukey, self._slot = np.unique(key, return_inverse=True)
        self._csc_indices = (ukey % self.N).astype(np.int32)
        self._csc_indptr = np.searchsorted(ukey // self.N, np.arange(self.N + 1)).astype(np.int32)

    # ------------------------------------------------------------------
    def coeff(self, x):
        """Cell coefficient a = exp(B x), shape (nx, nx)."""
        x = np.asarray(x, dtype=float).reshape(self.m)
        return np.exp(self.B @ x).reshape(self.nx, self.nx)

    def _face_coeff(self, a):
        nx = self.nx
        a_hor = 0.5 * (a[1:, :] + a[:-1, :])        # (nx-1, nx)
        a_ver = 0.5 * (a[:, 1:] + a[:, :-1])        # (nx, nx-1)
        return np.concatenate([a_hor[:, :nx - 1].ravel(), a_ver.ravel()])

    def stiffness(self, a):
        """SPD matrix A with A u = 1 equivalent to legacy residual(u) = 0."""
        k = self._face_coeff(a) / self.dx**2
        data = np.bincount(self._slot, weights=self._sgn * k[self._face],
                           minlength=self._csc_indices.size)
        return sp.csc_matrix((data, self._csc_indices, self._csc_indptr),
                             shape=(self.N, self.N))

    def solve(self, x):
        """Return (u as (nx-1,nx-1) array, a, splu factorization)."""
        a = self.coeff(x)
        lu = spla.splu(self.stiffness(a), permc_spec='MMD_AT_PLUS_A')
        u = lu.solve(np.full(self.N, self.f))
        return u.reshape(self.nx - 1, self.nx - 1), a, lu

    def _adjoint_grad(self, u, a, lu, c):
        """d(c^T u)/dx for c given on the unknowns, via one adjoint solve.

        A(a) u = f, q = c^T u  =>  dq/da_face = -lam^T (dA/da_face) u
                                             = -(du_face)(dlam_face)/dx^2,
        then chain rule through face averaging, a = exp(Bx): dq/dx = B^T (a * dq/da).
        """
        lam = lu.solve(c.ravel())       # A is symmetric: A^T = A
        uu = np.concatenate([u.ravel(), [0.0]])   # index -1 -> Dirichlet zero
        ll = np.concatenate([lam, [0.0]])
        p, q = self._p, self._q
        g_face = -(uu[q] - uu[p]) * (ll[q] - ll[p]) / self.dx**2
        nx = self.nx
        g_hor = np.zeros((nx - 1, nx))
        g_hor[:, :nx - 1] = g_face[:self._nh].reshape(nx - 1, nx - 1)
        g_ver = g_face[self._nh:].reshape(nx, nx - 1)
        g_a = np.zeros((nx, nx))
        g_a[1:, :] += 0.5 * g_hor
        g_a[:-1, :] += 0.5 * g_hor
        g_a[:, 1:] += 0.5 * g_ver
        g_a[:, :-1] += 0.5 * g_ver
        return self.B.T @ (a * g_a).ravel()

    def residual(self, u, a):
        """Legacy numpad residual, plain numpy (for testing)."""
        nx, dx = self.nx, self.dx
        u = np.hstack([np.zeros([nx - 1, 1]), u, u[:, -1:]])
        u = np.vstack([np.zeros([1, nx + 1]), u, np.zeros([1, nx + 1])])
        a_hor = 0.5 * (a[1:, :] + a[:-1, :])
        a_ver = 0.5 * (a[:, 1:] + a[:, :-1])
        a_dudy = a_hor * (u[1:-1, 1:] - u[1:-1, :-1]) / dx
        a_dudx = a_ver * (u[1:, 1:-1] - u[:-1, 1:-1]) / dx
        return (a_dudy[:, 1:] - a_dudy[:, :-1]) / dx \
            + (a_dudx[1:, :] - a_dudx[:-1, :]) / dx + self.f


class PDESolver(_EllipticPDE):
    """Elliptic_PDE QoI: trapezoid average of u on the last unknown column."""

    def _qoi_weights(self):
        n = self.nx - 1
        w = np.zeros((n, n))
        w[:, -1] = self.dx
        w[0, -1] = w[-1, -1] = 0.5 * self.dx
        return w

    def qoi(self, x):
        u, _, _ = self.solve(x)
        u_bnd = u[:, -1]
        return 0.5 * self.dx * np.sum(u_bnd[1:] + u_bnd[:-1])

    def grad_qoi(self, x):
        u, a, lu = self.solve(x)
        return self._adjoint_grad(u, a, lu, self._qoi_weights())

    def qoi_and_grad(self, x):
        """(qoi, grad_qoi) from one factorization: one solve plus one adjoint."""
        u, a, lu = self.solve(x)
        w = self._qoi_weights()
        return np.sum(w * u), self._adjoint_grad(u, a, lu, w)


class PDEMisfit(_EllipticPDE):
    """MCMC model: sensor observations u[indz, -1] and Gaussian misfit."""

    def __init__(self, corrcase='short', datadir=None):
        super().__init__(corrcase, datadir)
        # unknown row r is vertex row r+1, so rows 19..79 are the sensors at
        # s2 = 0.2, ..., 0.8 (s1 = 0.99), as in the paper. Legacy: 20..80.
        self.indz = [19, 29, 39, 49, 59, 69, 79]

    def observations(self, x):
        u, _, _ = self.solve(x)
        return u[self.indz, -1]

    def set_data(self, d, sig2):
        self.d = np.asarray(d, dtype=float).reshape(len(self.indz))
        self.sig2 = sig2

    def misfit(self, x):
        obs = self.observations(x)
        r = obs - self.d
        return 0.5 * np.dot(r, r) / self.sig2

    def misfit_grad(self, x):
        u, a, lu = self.solve(x)
        c = np.zeros_like(u)
        c[self.indz, -1] = (u[self.indz, -1] - self.d) / self.sig2
        return self._adjoint_grad(u, a, lu, c)
