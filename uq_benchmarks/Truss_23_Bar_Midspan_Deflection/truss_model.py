"""Midspan deflection of the 23-bar plane truss of Blatman and Sudret (J. Comput. Phys. 230, 2011), Sec. 6.3.

Geometry (Fig. 8, p. 2357): a simply supported plane truss of span 24 m and height 2 m. The bottom
chord has 7 nodes at x = 0, 4, ..., 24 m (y = 0); the top chord has 6 nodes at x = 2, 6, ..., 22 m
(y = 2 m). The 23 bars are the 6 bottom-chord and 5 top-chord bars (11 horizontal bars, Young's
modulus E1 and area A1) and the 12 diagonals (E2, A2) joining each top node to the two bottom nodes
below it. The left end (node 0) is pinned and the right end (node 6) is on a roller (Fig. 8). The
six vertical loads P1..P6 act downward at the six top nodes, left to right. The output V1 is the
downward deflection of the midspan bottom node (x = 12 m), in metres, positive downward
(the arrow V1 in Fig. 8).

Node numbering, bar connectivity and the support conditions are read from Fig. 8 (the paper gives
no table of them); they are this module's reading of the figure.

Inputs (Table 1, p. 2358), independent:

    E1, E2 (Pa)  lognormal, mean 2.10e11, sd 2.10e10
    A1 (m^2)     lognormal, mean 2.0e-3,  sd 2.0e-4
    A2 (m^2)     lognormal, mean 1.0e-3,  sd 1.0e-4
    P1..P6 (N)   Gumbel (largest value), mean 5.0e4, sd 7.5e3

Column order here is that of the paper's Eq. (54): E1, E2, A1, A2, P1, ..., P6.

Standard-normal coordinates. As in the paper's Eq. (55), each input Z_i is written as
Z_i = F_i^{-1}(Phi(xi_i)) with xi_i standard normal (an isoprobabilistic transform through the
CDF): lognormal Z = exp(mu + s xi); Gumbel Z = u - beta ln(-ln Phi(xi)), with
beta = sd sqrt(6)/pi and u = mean - gamma_E beta. `physical(XI)` gives Z, and
`dphysical_dxi(XI)` the diagonal Jacobian dZ_i/dxi_i used to map gradients.

Finite elements: linear-elastic two-node bar elements, small displacements; the global stiffness
is K = E1 A1 K_h + E2 A2 K_d, where K_h and K_d assemble (1/L_e) c c^T blocks of the horizontal
and the diagonal bars. Gradients are exact (adjoint): with K u = f, V1 = -q^T u (q picks the vertical
displacement of the midspan node) and K lambda = q,

    dV1/dE1 = A1 lambda^T K_h u,  dV1/dA1 = E1 lambda^T K_h u  (likewise E2, A2 with K_d),
    dV1/dP_j = lambda at the vertical dof of top node j.

Only numpy and the standard library (math.erfc) are imported.
"""
import math

import numpy as np

#: Input names in column order (Eq. (54) of the paper).
NAMES = ('E1', 'E2', 'A1', 'A2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6')
UNITS = ('Pa', 'Pa', 'm^2', 'm^2', 'N', 'N', 'N', 'N', 'N', 'N')
DESCRIPTIONS = ("Young's modulus of the 11 horizontal bars",
                "Young's modulus of the 12 diagonal bars",
                'cross-section area of the 11 horizontal bars',
                'cross-section area of the 12 diagonal bars') + tuple(
    f'vertical (downward) load at top node {j} (x = {2 + 4 * (j - 1)} m)' for j in range(1, 7))
DISTRIBUTIONS = ('lognormal',) * 4 + ('Gumbel (max)',) * 6

#: Means and standard deviations (Table 1, p. 2358).
MEAN = np.array([2.10e11, 2.10e11, 2.0e-3, 1.0e-3] + [5.0e4] * 6)
SD = np.array([2.10e10, 2.10e10, 2.0e-4, 1.0e-4] + [7.5e3] * 6)

IS_LOGNORMAL = np.array([True] * 4 + [False] * 6)
#: Lognormal parameters: ln Z ~ N(LN_MU, LN_S^2) (first four columns; NaN elsewhere).
LN_S = np.where(IS_LOGNORMAL, np.sqrt(np.log1p((SD / MEAN)**2)), np.nan)
LN_MU = np.where(IS_LOGNORMAL, np.log(MEAN) - LN_S**2 / 2, np.nan)
#: Gumbel (largest value) parameters: F(z) = exp(-exp(-(z - GUM_U)/GUM_BETA)) (last six columns).
EULER_GAMMA = 0.57721566490153286061
GUM_BETA = np.where(IS_LOGNORMAL, np.nan, SD * math.sqrt(6.0) / math.pi)
GUM_U = np.where(IS_LOGNORMAL, np.nan, MEAN - EULER_GAMMA * GUM_BETA)

QOI_NAME = 'V1'
QOI_UNITS = 'm'

# ---------------------------------------------------------------- geometry (Fig. 8)
SPAN = 24.0
HEIGHT = 2.0
NODES = np.array([[4.0 * i, 0.0] for i in range(7)] + [[2.0 + 4.0 * j, HEIGHT] for j in range(6)])
#: Bars as (node a, node b, group) with group 0 = horizontal (E1, A1), 1 = diagonal (E2, A2).
BARS = ([(i, i + 1, 0) for i in range(6)]                 # bottom chord, 6 bars
        + [(7 + j, 8 + j, 0) for j in range(5)]           # top chord, 5 bars
        + [b for j in range(6) for b in ((j, 7 + j, 1), (7 + j, j + 1, 1))])   # 12 diagonals
N_DOF = 2 * len(NODES)
FIXED = (0, 1, 2 * 6 + 1)            # node 0: ux, uy (pin); node 6: uy (roller)
FREE = np.array([d for d in range(N_DOF) if d not in FIXED])
MID_NODE = 3                         # bottom node at x = 12 m
LOAD_NODES = tuple(range(7, 13))     # top nodes, P1..P6 from left to right


def _group_stiffness():
    """Unit-property stiffness matrices K_h, K_d (EA = 1) on the free dofs."""
    K = np.zeros((2, N_DOF, N_DOF))
    for a, b, g in BARS:
        d = NODES[b] - NODES[a]
        L = np.hypot(*d)
        c = np.concatenate([-d, d]) / L
        idx = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        K[g][np.ix_(idx, idx)] += np.outer(c, c) / L
    return K[:, FREE][:, :, FREE]


K_GROUPS = _group_stiffness()
_free_index = {d: k for k, d in enumerate(FREE)}
Q_MID = _free_index[2 * MID_NODE + 1]                       # vertical dof of the midspan node
LOAD_DOFS = np.array([_free_index[2 * n + 1] for n in LOAD_NODES])


def _check_inputs(Z):
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[1] != 10:
        raise ValueError(f'expected 10 inputs per row (E1, E2, A1, A2, P1..P6), got {Z.shape[1]}')
    if not np.all(np.isfinite(Z)):
        raise ValueError('inputs must be finite')
    if np.any(Z[:, :4] <= 0):
        raise ValueError('Young\'s moduli and areas must be positive')
    return Z


def deflection(Z, grad=False, chunk=20000):
    """Midspan downward deflection V1 (m) for physical inputs Z, shape (M, 10) or (10,).

    Returns V of shape (M,), and with grad=True also dV/dZ of shape (M, 10) (physical units).
    """
    Z = _check_inputs(Z)
    M = len(Z)
    V = np.empty(M)
    G = np.empty((M, 10)) if grad else None
    for s in range(0, M, chunk):
        z = Z[s:s + chunk]
        k1 = z[:, 0] * z[:, 2]
        k2 = z[:, 1] * z[:, 3]
        K = k1[:, None, None] * K_GROUPS[0] + k2[:, None, None] * K_GROUPS[1]
        f = np.zeros((len(z), len(FREE)))
        f[:, LOAD_DOFS] = -z[:, 4:]                       # downward loads
        if grad:
            rhs = np.stack([f, np.broadcast_to(np.eye(len(FREE))[Q_MID], f.shape)], axis=2)
            sol = np.linalg.solve(K, rhs)
            u, lam = sol[:, :, 0], sol[:, :, 1]
        else:
            u = np.linalg.solve(K, f[:, :, None])[:, :, 0]
        V[s:s + chunk] = -u[:, Q_MID]
        if grad:
            t1 = np.einsum('ni,ij,nj->n', lam, K_GROUPS[0], u)
            t2 = np.einsum('ni,ij,nj->n', lam, K_GROUPS[1], u)
            g = G[s:s + chunk]
            g[:, 0] = z[:, 2] * t1
            g[:, 2] = z[:, 0] * t1
            g[:, 1] = z[:, 3] * t2
            g[:, 3] = z[:, 1] * t2
            g[:, 4:] = lam[:, LOAD_DOFS]
    return (V, G) if grad else V


def displacements(z):
    """All nodal displacements (13, 2) in metres for one input vector z (10,)."""
    z = _check_inputs(z)[0]
    K = z[0] * z[2] * K_GROUPS[0] + z[1] * z[3] * K_GROUPS[1]
    f = np.zeros(len(FREE))
    f[LOAD_DOFS] = -z[4:]
    u = np.zeros(N_DOF)
    u[FREE] = np.linalg.solve(K, f)
    return u.reshape(-1, 2)


def bar_forces(loads):
    """Bar axial forces (23,) in N (tension positive) and reactions (3,) from nodal equilibrium alone.

    The truss is statically determinate (23 bars + 3 reactions = 26 equilibrium equations), so
    the forces do not depend on E or A. `loads` is the (6,) vector of downward top-node loads, or
    the string 'unit_mid' for a unit downward load at the midspan bottom node.
    """
    B = np.zeros((N_DOF, len(BARS) + len(FIXED)))
    for e, (a, b, _) in enumerate(BARS):
        d = NODES[b] - NODES[a]
        c = d / np.hypot(*d)
        B[2 * a:2 * a + 2, e] += c          # a tensile bar pulls node a toward b
        B[2 * b:2 * b + 2, e] -= c
    for k, dof in enumerate(FIXED):
        B[dof, len(BARS) + k] = 1.0
    p = np.zeros(N_DOF)
    if isinstance(loads, str):
        p[2 * MID_NODE + 1] = -1.0
    else:
        p[[2 * n + 1 for n in LOAD_NODES]] = -np.asarray(loads, dtype=float)
    sol = np.linalg.solve(B, -p)
    return sol[:len(BARS)], sol[len(BARS):]


BAR_LENGTHS = np.array([np.hypot(*(NODES[b] - NODES[a])) for a, b, _ in BARS])
BAR_GROUPS = np.array([g for _, _, g in BARS])


def virtual_work(z):
    """V1 by the unit-load method, sum_e N_e n_e L_e / (E A)_e, for one input vector z (10,).

    An independent check of `deflection` that does not assemble or solve the stiffness matrix.
    """
    z = _check_inputs(z)[0]
    N, _ = bar_forces(z[4:])
    n, _ = bar_forces('unit_mid')
    EA = np.where(BAR_GROUPS == 0, z[0] * z[2], z[1] * z[3])
    return float(np.sum(N * n * BAR_LENGTHS / EA))


# ---------------------------------------------------------------- standard-normal coordinates
_erfc = np.frompyfunc(math.erfc, 1, 1)


def _upper_tail(x):
    """Q(x) = 1 - Phi(x), computed without cancellation."""
    return 0.5 * _erfc(np.asarray(x, dtype=float) / math.sqrt(2.0)).astype(float)


def _gumbel_parts(xi):
    """Phi(xi), w = -ln Phi(xi) and phi(xi), accurate in both tails."""
    Phi = _upper_tail(-np.asarray(xi, dtype=float))         # Phi(x) = Q(-x)
    w = -np.log1p(-_upper_tail(xi))           # -ln Phi, accurate when Phi is near 1
    small = Phi < 0.5
    w = np.where(small, -np.log(np.where(small, Phi, 1.0)), w)
    phi = np.exp(-0.5 * np.asarray(xi, dtype=float)**2) / math.sqrt(2.0 * math.pi)
    return Phi, w, phi


def physical(XI):
    """Physical inputs Z (M, 10) from standard-normal coordinates XI (M, 10), Eq. (55) inverted."""
    XI = np.atleast_2d(np.asarray(XI, dtype=float))
    Z = np.empty_like(XI)
    Z[:, :4] = np.exp(LN_MU[:4] + LN_S[:4] * XI[:, :4])
    _, w, _ = _gumbel_parts(XI[:, 4:])
    Z[:, 4:] = GUM_U[4:] - GUM_BETA[4:] * np.log(w)
    return Z


def dphysical_dxi(XI):
    """Diagonal Jacobian dZ_i/dxi_i (M, 10) of `physical`."""
    XI = np.atleast_2d(np.asarray(XI, dtype=float))
    J = np.empty_like(XI)
    J[:, :4] = LN_S[:4] * np.exp(LN_MU[:4] + LN_S[:4] * XI[:, :4])
    Phi, w, phi = _gumbel_parts(XI[:, 4:])
    J[:, 4:] = GUM_BETA[4:] * phi / (Phi * w)
    return J


def deflection_xi(XI, grad=False):
    """V1 and (optionally) its gradient with respect to the standard-normal coordinates xi."""
    XI = np.atleast_2d(np.asarray(XI, dtype=float))
    Z = physical(XI)
    if not grad:
        return deflection(Z)
    V, G = deflection(Z, grad=True)
    return V, G * dphysical_dxi(XI)


def sample_xi(M, rng):
    """M independent standard-normal draws (M, 10) from a numpy Generator or seed."""
    return np.random.default_rng(rng).standard_normal((M, 10))


def gumbel_cdf(z, j):
    """CDF of input column j (4..9) at z (physical units); used for checks."""
    return np.exp(-np.exp(-(np.asarray(z, dtype=float) - GUM_U[j]) / GUM_BETA[j]))
