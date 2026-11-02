"""Two-material steady heat conduction on [-1, 1]^2: thermal energy (heat compliance)
of a material distribution, and its exact gradient.

The model of Luneau (2025), Sec. 4.3, Eqs. (8)-(9): find u in H^1_0(Omega) with

    int_Omega (k1 theta + k2 (1 - theta)) grad u . grad v dx = int_Omega f v dx   for all v,

Omega = [-1, 1]^2, k1 = 2, k2 = 1, f(x, y) = 56 (1 - |x| - |y|)^6, and the
quantity of interest is the thermal energy

    E(theta) = 1/2 int_Omega (k1 theta + k2 (1 - theta)) |grad u|^2 dx = 1/2 int f u dx.

The discretization is the one of the author's code (pyTOP `fem2d.py`, MIT licence,
P.-A. Luneau, used through github.com/paluneau/CASM): a structured mesh of
16 x 16 squares, each cut into two triangles along the diagonal from its
lower-right to its upper-left corner (D = 512 elements), second-order (P2)
Lagrange elements (33 x 33 = 1089 nodes), theta piecewise constant per element,
and the 4-point degree-3 Hammer rule for the element integrals (exact for the
P2 stiffness, approximate for the load f v). The element and node numbering
are the code's, so theta[e] is the code's design variable e (mapped from the
code's x in [-1, 1] by theta = (x + 1)/2). This module is an independent
vectorized numpy/scipy implementation, not a copy of that code.

Gradient. E = 1/2 F^T U with K(theta) U = F, so E is self-adjoint and

    dE/dtheta_e = -1/2 (k1 - k2) u_e^T K_e u_e,

with K_e the unit-conductivity element stiffness and u_e the element's nodal
values: the exact derivative of the discrete E (one solve per sample).

Boundary. `boundary="paper"` (the default) puts u = 0 at all 128 boundary nodes,
as the paper states. `boundary="casm"` uses the node list hard-coded in the
CASM notebook `thermal_design_casm.ipynb`, which omits four boundary nodes
(32, 1023, 1055 and 1088: the lower-right and upper-right corners and the
nodes next to the top on the left and right edges); it is kept only to check
this implementation against the files that notebook wrote.

Only numpy and scipy are imported.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

K1 = 2.0        #: conductivity of material 1 (theta = 1), Sec. 4.3
K2 = 1.0        #: conductivity of material 2 (theta = 0), Sec. 4.3
E_MAX = 1.35    #: energy threshold of the design constraint E <= E_max, Sec. 4.3
NX = 16         #: squares per side (2 * NX**2 = 512 triangles)
M_INPUTS = 2*NX*NX

# Hammer 4-point rule on the reference triangle (area 1/2), as in pyTOP fem2d.RefElement
_QX = np.array([1/3, 1/5, 1/5, 3/5])
_QY = np.array([1/3, 1/5, 3/5, 1/5])
_QW = np.array([-0.28125, 0.2604166666666667, 0.2604166666666667, 0.2604166666666667])


def source(x, y):
    """Heat source f(x, y) = 56 (1 - |x| - |y|)^6 (Sec. 4.3, as written; no clipping)."""
    return 56.0*(1.0 - np.abs(x) - np.abs(y))**6


def _p2_basis(xi, eta):
    """P2 basis on the reference triangle, node order (0,0), (1,0), (0,1), mid 01, mid 02, mid 12."""
    phi = np.array([2*xi**2 + 4*xi*eta - 3*xi + 2*eta**2 - 3*eta + 1,
                    xi*(2*xi - 1),
                    eta*(2*eta - 1),
                    4*xi*(1 - xi - eta),
                    4*eta*(1 - xi - eta),
                    4*xi*eta])
    dphi = np.array([[4*xi + 4*eta - 3, 4*xi - 1, 0*xi, 4 - 8*xi - 4*eta, -4*eta, 4*eta],
                     [4*xi + 4*eta - 3, 0*xi, 4*eta - 1, -4*xi, 4 - 8*eta - 4*xi, 4*xi]])
    return phi, dphi


def mesh(nx=NX):
    """Node coordinates (N, 2) and element connectivity (2 nx^2, 6), pyTOP numbering.

    Node r*(2nx+1) + c is at (-1 + c h/2, -1 + r h/2), h = 2/nx. Square (i, j)
    (row i, column j) holds elements 2(nx i + j) (lower-left triangle) and
    2(nx i + j) + 1 (upper-right triangle).
    """
    n1 = 2*nx + 1
    s = np.linspace(-1.0, 1.0, n1)
    X, Y = np.meshgrid(s, s)
    coord = np.column_stack([X.ravel(), Y.ravel()])
    i, j = np.meshgrid(np.arange(nx), np.arange(nx), indexing='ij')
    b = (2*i*n1 + 2*j).ravel()
    lower = np.column_stack([b, b + 2, b + 2*n1, b + 1, b + n1, b + n1 + 1])
    upper = np.column_stack([b + 2*n1 + 2, b + 2*n1, b + 2, b + 2*n1 + 1, b + n1 + 2, b + n1 + 1])
    conn = np.empty((2*nx*nx, 6), dtype=int)
    conn[0::2] = lower
    conn[1::2] = upper
    return coord, conn


def boundary_nodes(nx=NX, boundary='paper'):
    """Dirichlet nodes: all boundary nodes ('paper') or the CASM notebook's list ('casm')."""
    n1 = 2*nx + 1
    if boundary == 'paper':
        r, c = np.divmod(np.arange(n1*n1), n1)
        return np.flatnonzero((r == 0) | (r == n1 - 1) | (c == 0) | (c == n1 - 1))
    if boundary == 'casm':
        if nx != 16:
            raise ValueError("boundary='casm' is defined only for nx = 16")
        return np.sort(np.r_[np.arange(0, 32), np.arange(1056, 1088),
                             np.arange(33, 1023, 33), np.arange(65, 1055, 33)])
    raise ValueError(f"unknown boundary {boundary!r}")


class ThermalCompliance:
    """Assembled FE operators for one mesh; `energy`, `grad`, `energy_and_grad`.

    theta : array of shape (2 nx^2,), material-1 fraction per element, each in [0, 1].
    """

    def __init__(self, nx=NX, k1=K1, k2=K2, boundary='paper'):
        self.nx, self.k1, self.k2, self.boundary = nx, float(k1), float(k2), boundary
        self.coord, self.conn = mesh(nx)
        self.n_elem = self.conn.shape[0]
        self.n_nodes = self.coord.shape[0]
        v = self.coord[self.conn[:, :3]]                       # (D, 3, 2) vertices
        T = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=2)   # (D, 2, 2) affine map
        detT = np.linalg.det(T)
        invTt = np.linalg.inv(T).transpose(0, 2, 1)
        Ke = np.zeros((self.n_elem, 6, 6))
        Fe = np.zeros((self.n_elem, 6))
        for q in range(4):
            phi, dphi = _p2_basis(_QX[q], _QY[q])
            g = invTt @ dphi                                    # (D, 2, 6) physical gradients
            Ke += _QW[q]*detT[:, None, None]*np.einsum('eki,ekj->eij', g, g)
            xq = v[:, 0] + np.einsum('eij,j->ei', T, [_QX[q], _QY[q]])
            Fe += _QW[q]*detT[:, None]*source(xq[:, 0], xq[:, 1])[:, None]*phi[None, :]
        self.Ke = Ke                        #: unit-conductivity element stiffness (D, 6, 6)
        self.area = 0.5*detT                #: element areas
        F = np.zeros(self.n_nodes)
        np.add.at(F, self.conn, Fe)
        fixed = boundary_nodes(nx, boundary)
        self.free = np.setdiff1d(np.arange(self.n_nodes), fixed)
        self.F = F
        self._rows = np.repeat(self.conn, 6, axis=1).ravel()
        self._cols = np.tile(self.conn, (1, 6)).ravel()

    def _check(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_elem,):
            raise ValueError(f"theta must have shape ({self.n_elem},), got {theta.shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta has non-finite entries")
        if theta.min() < 0.0 or theta.max() > 1.0:
            raise ValueError("theta must lie in [0, 1]")
        return theta

    def conductivity(self, theta):
        """Element conductivities k1 theta + k2 (1 - theta)."""
        return self.k2 + (self.k1 - self.k2)*np.asarray(theta)

    def solve(self, theta):
        """Nodal temperature U (zero at the Dirichlet nodes)."""
        theta = self._check(theta)
        kap = self.conductivity(theta)
        data = (kap[:, None, None]*self.Ke).ravel()
        K = sp.csr_matrix((data, (self._rows, self._cols)), shape=(self.n_nodes,)*2)
        Kf = K[self.free][:, self.free].tocsc()
        U = np.zeros(self.n_nodes)
        U[self.free] = spla.spsolve(Kf, self.F[self.free])
        if not np.all(np.isfinite(U)):
            raise FloatingPointError("FE solve returned non-finite values")
        return U

    def energy_and_grad(self, theta):
        """E(theta) = 1/2 F.U and dE/dtheta (shape (D,)), from one solve."""
        U = self.solve(theta)
        E = 0.5*float(self.F @ U)
        ue = U[self.conn]
        dE = -0.5*(self.k1 - self.k2)*np.einsum('ei,eij,ej->e', ue, self.Ke, ue)
        return E, dE

    def energy(self, theta):
        """Thermal energy E(theta)."""
        return 0.5*float(self.F @ self.solve(theta))

    def grad(self, theta):
        """Exact gradient dE/dtheta."""
        return self.energy_and_grad(theta)[1]

    def volume(self, theta):
        """Volume of material 1, V(theta) = sum of area_e theta_e (the paper's objective)."""
        return float(self.area @ np.asarray(theta, dtype=float))

    def element_centroids(self):
        """Element centroids (D, 2), for plotting a field over the elements."""
        return self.coord[self.conn[:, :3]].mean(axis=1)

    def element_triangles(self):
        """Vertex coordinates (D, 3, 2), for plotting a field over the elements."""
        return self.coord[self.conn[:, :3]]


_DEFAULT = None


def energy(theta):
    """E(theta) on the paper's model (16 x 16 mesh, u = 0 on the whole boundary)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ThermalCompliance()
    return _DEFAULT.energy(theta)


def energy_and_grad(theta):
    """E(theta) and dE/dtheta on the paper's model."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ThermalCompliance()
    return _DEFAULT.energy_and_grad(theta)
