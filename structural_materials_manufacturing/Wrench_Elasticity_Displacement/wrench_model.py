"""Plane-stress linear elasticity of a wrench with a random Young's modulus (numpy and scipy only).

A new implementation of the high-fidelity model of R. R. Lam, O. Zahm, Y. M. Marzouk and
K. E. Willcox, "Multifidelity dimension reduction via active subspaces", SIAM J. Sci. Comput.
42(2), A929-A956 (2020), doi:10.1137/18M1214123, Sec. 5.2, pp. 12-14, written from the paper and
from the problem set-up of the authors' MATLAB code (wrenchmark, O. Zahm, Inria GitLab; not
copied).

Model. Find u: Omega -> R^2 with div(K : eps(u)) = 0, eps(u) = (grad u + grad u^T)/2 and the
plane-stress Hooke tensor K : eps = E/(1+nu) eps + nu E/(1-nu^2) tr(eps) I, nu = 0.3 (p. 12);
u = 0 on the two inner jaw faces, a unit downward traction on the top edge of the handle, traction
free elsewhere (Fig. 5a). Galerkin P1 finite elements on the 2197-triangle mesh in the mesh CSVs,
with E piecewise constant: E = exp(x_k) on triangle k.

Inputs. x = log E^h in R^2197 (one value per triangle), x ~ N(0, Sigma),
Sigma_ij = exp(-|c_i - c_j|^2 / l0), l0 = 1 (pp. 12-13), with c_i the incentre of triangle i (as in
wrenchmark). Sigma is numerically singular, so the field is written x = Phi xi, xi ~ N(0, I_r),
Phi = U_r D_r^(1/2) from the eigenpairs of Sigma with eigenvalue >= 10 eps lambda_max (the
truncation of wrenchmark); the r columns of Phi are stored in kl_basis.npz.

Output. f(x) = u_2^h(PoI), the vertical displacement at the node (-4.1, 0) (p. 13); nondimensional.

Gradient. Adjoint: K(E) u = b, K(E) q = l (K symmetric), df/dx_k = -E_k q_k^T K_k u_k, where K_k
is the stiffness of triangle k at E = 1; df/dxi = Phi^T df/dx.
"""
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

HERE = os.path.dirname(os.path.abspath(__file__))
NU = 0.3            # Poisson ratio (p. 12)
L0 = 1.0            # covariance length parameter (p. 13)
TRUNC = 10 * np.finfo(float).eps   # relative eigenvalue cut-off of Sigma (wrenchmark)
POI = np.array([-4.1, 0.0])
KL_FILE = 'kl_basis.npz'


def read_mesh(directory=HERE):
    """Nodes (N, 2), triangles (T, 3) and boundary segments (S, 3) from the mesh CSVs."""
    p = np.loadtxt(os.path.join(directory, 'mesh_nodes.csv'), delimiter=',')
    t = np.loadtxt(os.path.join(directory, 'mesh_elements.csv'), delimiter=',').astype(np.intp)
    s = np.loadtxt(os.path.join(directory, 'mesh_boundary.csv'), delimiter=',').astype(np.intp)
    return p, t, s


def incentres(p, t):
    a = p[t]
    la = np.linalg.norm(a[:, 1] - a[:, 2], axis=1)
    lb = np.linalg.norm(a[:, 2] - a[:, 0], axis=1)
    lc = np.linalg.norm(a[:, 0] - a[:, 1], axis=1)
    return (la[:, None] * a[:, 0] + lb[:, None] * a[:, 1] + lc[:, None] * a[:, 2]) / (la + lb + lc)[:, None]


def covariance(p, t, l0=L0):
    """Sigma_ij = exp(-|c_i - c_j|^2 / l0) at the incentres c_i."""
    c = incentres(p, t)
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / l0)


def kl_basis(Sigma, trunc=TRUNC):
    """Phi = U_r D_r^(1/2) (columns by decreasing eigenvalue, each column's largest-magnitude
    entry positive) and the kept eigenvalues D_r."""
    lam, U = np.linalg.eigh(Sigma)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    keep = lam >= trunc * lam[0]
    lam, U = lam[keep], U[:, keep]
    idx = np.argmax(np.abs(U), axis=0)
    U = U * np.sign(U[idx, np.arange(U.shape[1])])
    return U * np.sqrt(lam), lam


class Wrench:
    """The wrench model on a mesh directory (default: this directory).

    f(x), grad f(x) for x = log E per triangle (physical coordinates), and f, grad in the
    standardized coordinates xi of x = Phi xi.
    """

    def __init__(self, directory=HERE, load_kl=True):
        self.p, self.t, self.segs = read_mesh(directory)
        self.n_el = len(self.t)
        self.n_nodes = len(self.p)
        self._element_matrices()
        self._assembly_maps()
        self._load_and_observation()
        self.Phi = self.kl_eigenvalues = None
        if load_kl:
            with np.load(os.path.join(directory, KL_FILE)) as z:
                self.Phi = z['Phi']
                self.kl_eigenvalues = z['eigenvalues']

    # --- finite elements -------------------------------------------------------------------
    def _element_matrices(self):
        a = self.p[self.t]                      # (T, 3, 2)
        x, y = a[..., 0], a[..., 1]
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
        area = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
        if np.any(area <= 0):
            raise ValueError('triangles must be counter-clockwise with positive area')
        B = np.zeros((self.n_el, 3, 6))
        B[:, 0, 0::2] = b
        B[:, 1, 1::2] = c
        B[:, 2, 0::2] = c
        B[:, 2, 1::2] = b
        B /= (2 * area)[:, None, None]
        D = np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]]) / (1 - NU ** 2)
        self.area = area
        self.Ke = area[:, None, None] * np.einsum('kij,il,klm->kjm', B, D, B)   # (T, 6, 6), E = 1
        dofs = np.empty((self.n_el, 6), np.intp)
        dofs[:, 0::2] = 2 * self.t
        dofs[:, 1::2] = 2 * self.t + 1
        self.dofs = dofs

    def _assembly_maps(self):
        clamped = np.unique(self.segs[self.segs[:, 2] == 2][:, :2])
        fixed = np.zeros(2 * self.n_nodes, bool)
        fixed[2 * clamped] = fixed[2 * clamped + 1] = True
        self.clamped_nodes = clamped
        self.free = np.flatnonzero(~fixed)
        g2f = -np.ones(2 * self.n_nodes, np.intp)
        g2f[self.free] = np.arange(len(self.free))
        rows = np.repeat(g2f[self.dofs], 6, axis=1).ravel()
        cols = np.tile(g2f[self.dofs], (1, 6)).ravel()
        ok = (rows >= 0) & (cols >= 0)
        n = len(self.free)
        key = rows[ok] * n + cols[ok]
        ukey, inv = np.unique(key, return_inverse=True)
        self._ok, self._inv = ok, inv
        self._rows, self._cols = ukey // n, ukey % n
        self._nfree = n
        self._g2f = g2f

    def _load_and_observation(self):
        b = np.zeros(2 * self.n_nodes)
        loaded = self.segs[self.segs[:, 2] == 1]
        length = np.linalg.norm(self.p[loaded[:, 0]] - self.p[loaded[:, 1]], axis=1)
        np.add.at(b, 2 * loaded[:, 0] + 1, -0.5 * length)       # unit downward traction
        np.add.at(b, 2 * loaded[:, 1] + 1, -0.5 * length)
        self.load_length = length.sum()
        self.b = b[self.free]
        self.poi_node = int(np.argmin(((self.p - POI) ** 2).sum(1)))
        ell = np.zeros(2 * self.n_nodes)
        ell[2 * self.poi_node + 1] = 1.0
        self.ell = ell[self.free]

    def stiffness(self, E):
        """Sparse stiffness matrix on the free degrees of freedom for element moduli E (T,)."""
        vals = (np.asarray(E)[:, None, None] * self.Ke).reshape(-1)[self._ok]
        data = np.bincount(self._inv, weights=vals.real, minlength=len(self._rows))
        if np.iscomplexobj(vals):     # complex-step checks
            data = data + 1j * np.bincount(self._inv, weights=vals.imag, minlength=len(self._rows))
        return sp.csc_matrix((data, (self._rows, self._cols)), shape=(self._nfree, self._nfree))

    def displacement(self, x):
        """Nodal displacement field (N, 2) for x = log E."""
        K = self.stiffness(np.exp(x))
        u = np.zeros(2 * self.n_nodes, dtype=np.result_type(x, float))
        u[self.free] = splu(K).solve(self.b.astype(u.dtype))
        return u.reshape(-1, 2)

    # --- physical coordinates x = log E --------------------------------------------------
    def f(self, x):
        """Vertical displacement at the point of interest for x = log E (T,)."""
        x = np.asarray(x)
        lu = splu(self.stiffness(np.exp(x)))
        u = lu.solve(self.b.astype(np.result_type(x, float)))
        return u @ self.ell

    def f_and_grad(self, x):
        """f(x) and the adjoint gradient df/dx (T,)."""
        x = np.asarray(x, float)
        E = np.exp(x)
        lu = splu(self.stiffness(E))
        u = lu.solve(self.b)
        q = lu.solve(self.ell)                       # K symmetric: adjoint = same factor
        uf = np.zeros(2 * self.n_nodes)
        qf = np.zeros(2 * self.n_nodes)
        uf[self.free], qf[self.free] = u, q
        ue, qe = uf[self.dofs], qf[self.dofs]        # (T, 6)
        g = -E * np.einsum('ki,kij,kj->k', qe, self.Ke, ue)
        return u @ self.ell, g

    # --- standardized coordinates xi --------------------------------------------------------
    def field(self, xi):
        """x = Phi xi (log E per triangle) for xi of shape (r,) or (M, r)."""
        return np.asarray(xi) @ self.Phi.T

    def f_xi(self, xi):
        return self.f(self.field(xi))

    def f_and_grad_xi(self, xi):
        """f and df/dxi = Phi^T df/dx at xi (r,)."""
        fx, g = self.f_and_grad(self.field(xi))
        return fx, self.Phi.T @ g


def sample(model, xi):
    """f, df/dxi and df/dx for the rows of xi (M, r)."""
    M = len(xi)
    f = np.empty(M)
    gx = np.empty((M, model.n_el))
    for i in range(M):
        f[i], gx[i] = model.f_and_grad(model.field(xi[i]))
    return f, gx @ model.Phi, gx
