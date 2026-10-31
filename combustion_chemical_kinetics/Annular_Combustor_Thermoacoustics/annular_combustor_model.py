"""ATACAMAC network model of an annular combustor: plenum + chamber + N = 19 burners.

Re-implementation (numpy only) of the dispersion relation of Bauerheim et al.,
J. Fluid Mech. 789 (2016) 534-566, eqs. (2.2)-(2.11), written as a nonlinear
eigenproblem as in Magri, Bauerheim & Juniper, J. Comput. Phys. (2016), eq. (30):

    D(omega; n, tau) = det( P_N ... P_2 P_1 - I ) = 0,   P_i = R(omega) T_i(omega, n_i, tau_i),

with time dependence exp(-j omega t), so Im(omega) > 0 means a growing (UNSTABLE)
mode. R is the 4x4 propagation (rotation) matrix over one sector, T_i the
interaction matrix of burner i with coupling factors Gamma_{i,1..4} (JFM eqs.
2.4-2.7) and flame parameter F_i = (rho0 c0)/(rho_u c_u) (1 + n_i exp(j omega tau_i))
(JFM eq. 2.8).

The quantity of interest is the growth rate of the first azimuthal mode,
f_Im = Im(f) = Im(omega)/(2 pi) [s^-1], taken as the larger of the two growth
rates of the split pair (JFM Sec. 4.2 and App. A.1).

Roots are located with the argument principle on a circle around the nominal
(symmetric) root, which also checks that exactly two roots (the split pair) are
inside it, and polished with deflated Newton steps. Gradients are exact: by the
implicit function theorem, d omega/dp = -(y^H dA/dp x)/(y^H dA/domega x), with
A = M - I and x, y its right and left null vectors (the adjoint formula of
Magri et al.); all derivatives of A are analytic.

Constants: JFM Table 1 and Magri et al. Table 1 (burner length L_i = 0.0551 m and
flame position alpha = 0.862 from Magri et al.; see DATA.md).
Only numpy is imported.
"""
import numpy as np

# ---- constants -------------------------------------------------------------
N = 19                 #: number of burners
L_C = 0.54             #: chamber half-perimeter L_c [m]      (JFM Table 1)
L_P = 0.54             #: plenum half-perimeter L_p [m]       (JFM Table 1)
S_C = 0.00785          #: chamber cross-section [m^2]         (JFM Table 1)
S_P = 0.00785          #: plenum cross-section [m^2]          (JFM Table 1)
S_I = 0.00028          #: burner cross-section [m^2]          (JFM Table 1)
L_I = 0.0551           #: burner length [m]                   (Magri et al. Table 1; JFM Table 1 says 0.04)
ALPHA = 0.862          #: flame position z_f = alpha L_i      (Magri et al. Table 1; not in JFM)
RHO_U, C_U = 10.93, 519.0   #: fresh gas (plenum, upstream of flame) [kg/m^3], [m/s]
RHO_B, C_B = 4.70, 832.0    #: burnt gas (chamber, downstream of flame) [kg/m^3], [m/s]
KAPPA = RHO_B * C_B / (RHO_U * C_U)
OMEGA0 = np.pi * C_B / L_C  #: first chamber azimuthal pulsation, pi c0 / L_c [rad/s] (JFM p. 19)

#: Nominal FTF parameters (JFM Table 2): (n_bar, tau_bar [s])
CASES = {'weak': (0.5, 0.635e-3), 'strong': (1.75, 0.735e-3)}
SD_FRAC_N, SD_FRAC_TAU = 0.10, 0.05      #: sigma_n = 10 % n_bar, sigma_tau = 5 % tau_bar (JFM p. 13)
HALF_FRAC_N = np.sqrt(3) * SD_FRAC_N     #: uniform half-width = sqrt(3) sigma
HALF_FRAC_TAU = np.sqrt(3) * SD_FRAC_TAU
NAMES = [f'n{i:02d}' for i in range(1, N + 1)] + [f'tau{i:02d}' for i in range(1, N + 1)]

#: radius [rad/s] of the circle around the nominal root that must contain the split pair
RADIUS = {'weak': 300.0, 'strong': 450.0}
K_CONTOUR = 64

_EYE4 = np.eye(4, dtype=complex)


def _case(case):
    """(n_bar, tau_bar, radius) for a case name in CASES or a tuple (n_bar, tau_bar[, radius])."""
    if isinstance(case, str):
        nb, tb = CASES[case]
        return nb, tb, RADIUS[case]
    nb, tb, *r = case
    return float(nb), float(tb), float(r[0]) if r else 450.0


def bounds(case):
    """Lower and upper bounds (38,) of the uniform inputs [n_1..n_19, tau_1..tau_19]."""
    nb, tb, _ = _case(case)
    lo = np.r_[np.full(N, nb * (1 - HALF_FRAC_N)), np.full(N, tb * (1 - HALF_FRAC_TAU))]
    up = np.r_[np.full(N, nb * (1 + HALF_FRAC_N)), np.full(N, tb * (1 + HALF_FRAC_TAU))]
    return lo, up


def _rot(x):
    c, s = np.cos(x), np.sin(x)
    R = np.zeros(np.shape(x) + (2, 2), complex)
    R[..., 0, 0] = R[..., 1, 1] = c
    R[..., 0, 1], R[..., 1, 0] = -s, s
    dR = np.zeros_like(R)
    dR[..., 0, 0] = dR[..., 1, 1] = -s
    dR[..., 0, 1], dR[..., 1, 0] = -c, c
    return R, dR


def sectors(w, n, tau, deriv=False):
    """Sector matrices P_i = R T_i for omega w (B,) and n, tau (B, N).

    Returns P (B, N, 4, 4); with deriv=True also dP/domega (B, N, 4, 4) and the
    pieces for parameter derivatives: R (B, 4, 4), dT/dF (B, N, 4, 4) and
    e = exp(j omega tau) (B, N), since dF/dn = KAPPA e and dF/dtau = KAPPA n j omega e.
    """
    w = np.asarray(w, complex)
    wb = w[:, None]
    e = np.exp(1j * wb * tau)
    F = KAPPA * (1 + n * e)
    a_w, b_w = (1 - ALPHA) * L_I / C_B, ALPHA * L_I / C_U
    a, b = a_w * wb, b_w * wb
    C, S, c, s = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    den = C * s + F * S * c
    hp, hc = 0.5 * S_I / S_P, 0.5 * S_I / S_C
    num1 = C * c - F * S * s
    num4 = F * C * c - S * s
    G1, G2, G3, G4 = -hp * num1 / den, hp / den, hc * F / den, -hc * num4 / den
    T = np.broadcast_to(_EYE4, G1.shape + (4, 4)).copy()
    T[..., 1, 0], T[..., 1, 2], T[..., 3, 0], T[..., 3, 2] = 2 * G1, 2 * G2, 2 * G3, 2 * G4

    xp, xc = w * 2 * L_P / (N * C_U), w * 2 * L_C / (N * C_B)
    Rp, dRp = _rot(xp)
    Rc, dRc = _rot(xc)
    R = np.zeros(w.shape + (4, 4), complex)
    R[:, :2, :2], R[:, 2:, 2:] = Rp, Rc
    P = R[:, None] @ T
    if not deriv:
        return P

    # partial derivatives of the coupling factors with respect to a, b, F
    dden = {'a': -S * s + F * C * c, 'b': C * c - F * S * s, 'F': S * c}
    dn1 = {'a': -S * c - F * C * s, 'b': -C * s - F * S * c, 'F': -S * s}
    dn4 = {'a': -F * S * c - C * s, 'b': -F * C * s - S * c, 'F': C * c}

    def dG(v):
        g1 = -hp * (dn1[v] - num1 / den * dden[v]) / den
        g2 = -hp * dden[v] / den ** 2
        g3 = hc * ((1.0 if v == 'F' else 0.0) - F / den * dden[v]) / den
        g4 = -hc * (dn4[v] - num4 / den * dden[v]) / den
        D = np.zeros(G1.shape + (4, 4), complex)
        D[..., 1, 0], D[..., 1, 2], D[..., 3, 0], D[..., 3, 2] = 2 * g1, 2 * g2, 2 * g3, 2 * g4
        return D

    TF = dG('F')
    F_w = KAPPA * n * 1j * tau * e
    T_w = dG('a') * a_w + dG('b') * b_w + TF * F_w[..., None, None]
    dR = np.zeros_like(R)
    dR[:, :2, :2] = dRp * (2 * L_P / (N * C_U))
    dR[:, 2:, 2:] = dRc * (2 * L_C / (N * C_B))
    P_w = dR[:, None] @ T + R[:, None] @ T_w
    return P, P_w, R, TF, e


def _chain(P):
    """Prefix Q_i = P_{i-1}..P_1 and suffix S_i = P_N..P_{i+1}; returns Q, S, M."""
    B = P.shape[0]
    Q = np.empty_like(P)
    S = np.empty_like(P)
    acc = np.broadcast_to(_EYE4, (B, 4, 4)).copy()
    for i in range(N):
        Q[:, i] = acc
        acc = P[:, i] @ acc
    M = acc
    acc = np.broadcast_to(_EYE4, (B, 4, 4)).copy()
    for i in range(N - 1, -1, -1):
        S[:, i] = acc
        acc = acc @ P[:, i]
    return Q, S, M


def dispersion(w, n, tau):
    """D(omega) = det(M - I) for w (B,), n, tau (B, N)."""
    P = sectors(w, n, tau)
    return np.linalg.det(_chain(P)[2] - _EYE4)


def _A_and_Aw(w, n, tau):
    """A = M - I and dA/domega = sum_i S_i P_i' Q_i, shapes (B, 4, 4)."""
    P, P_w, *_ = sectors(w, n, tau, deriv=True)
    Q, S, M = _chain(P)
    A_w = np.einsum('bnij,bnjk,bnkl->bil', S, P_w, Q)
    return M - _EYE4, A_w


def log_derivative(w, n, tau):
    """D'(omega)/D(omega) = tr(A^{-1} dA/domega), A = M - I; shape (B,). Not at a root."""
    A, A_w = _A_and_Aw(w, n, tau)
    return np.trace(np.linalg.solve(A, A_w), axis1=1, axis2=2)


def newton_step(w, n, tau):
    """D(omega)/D'(omega), computed stably near a root from the SVD A = U diag(s) V^H:
    D/D' = prod(s) / sum_i prod_{j != i}(s_j) (U^H A' V)_ii (zero at an exact root)."""
    A, A_w = _A_and_Aw(w, n, tau)
    U, s, Vh = np.linalg.svd(A)
    d = np.einsum('bji,bjk,bik->bi', U.conj(), A_w, Vh.conj())
    others = np.stack([np.prod(np.delete(s, i, axis=1), axis=1) for i in range(4)], 1)
    return np.prod(s, axis=1) / np.sum(others * d, axis=1)


def symmetric_root(case, w_start=None, tol=1e-12, maxit=50):
    """First azimuthal (p = 1) chamber root of the symmetric configuration.

    Uses the Bloch condition det(P - exp(2 pi j/N) I) = 0 (the root is a double
    root of D), solved by the secant method from OMEGA0. `case` is a name in CASES
    or a tuple (n_bar, tau_bar[, radius]).
    """
    nb, tb, _ = _case(case)
    lam = np.exp(2j * np.pi / N)

    def g(w):
        P = sectors(np.array([w]), np.full((1, N), nb), np.full((1, N), tb))[0, 0]
        return np.linalg.det(P - lam * _EYE4)

    w0 = OMEGA0 + 0j if w_start is None else complex(w_start)
    w1 = w0 * (1 + 1e-4)
    g0, g1 = g(w0), g(w1)
    for _ in range(maxit):
        w2 = w1 - g1 * (w1 - w0) / (g1 - g0)
        w0, g0, w1, g1 = w1, g1, w2, g(w2)
        if abs(w1 - w0) < tol * abs(w1):
            return w1
    raise RuntimeError('secant did not converge')


def split_roots(n, tau, case, chunk=40, newton_iter=6):
    """The two roots of D near the nominal symmetric root, for each sample.

    n, tau: (B, N). Returns roots (B, 2) sorted by decreasing Im, the
    argument-principle count (B,) inside the circle (2 on the right branch), and
    the final Newton correction size (B,).
    """
    n, tau = np.atleast_2d(n), np.atleast_2d(tau)
    B = n.shape[0]
    wc = symmetric_root(case)
    r = _case(case)[2]
    t = np.exp(2j * np.pi * np.arange(K_CONTOUR) / K_CONTOUR)
    roots = np.empty((B, 2), complex)
    count = np.empty(B)
    step = np.empty(B)
    for s0 in range(0, B, chunk):
        sl = slice(s0, min(B, s0 + chunk))
        b = sl.stop - sl.start
        w = (wc + r * t)[None, :].repeat(b, 0).ravel()
        nn = np.repeat(n[sl], K_CONTOUR, 0)
        tt = np.repeat(tau[sl], K_CONTOUR, 0)
        ld = log_derivative(w, nn, tt).reshape(b, K_CONTOUR)
        # (1/2 pi j) \oint z^k D'/D domega, z = (omega - wc)/r, domega = j r t dphi
        mom = [np.mean(ld * r * t * t ** k, axis=1) for k in range(3)]
        count[sl] = mom[0].real
        s1, s2 = mom[1], mom[2]
        disc = np.sqrt(2 * s2 - s1 ** 2 + 0j)
        z = np.stack([(s1 + disc) / 2, (s1 - disc) / 2], 1)
        roots[sl] = wc + r * z
    # polish: Newton deflated by the partner root, g = D/(omega - omega_other)
    for it in range(newton_iter):
        step = np.zeros(B)
        for k in (0, 1):
            sk = newton_step(roots[:, k], n, tau)
            gap = roots[:, k] - roots[:, 1 - k]
            dw = sk * gap / (gap - sk)
            roots[:, k] -= dw
            step = np.maximum(step, np.abs(dw))
    order = np.argsort(-roots.imag, axis=1)
    roots = np.take_along_axis(roots, order, 1)
    return roots, count, step


def root_gradient(w, n, tau):
    """d omega/d(n_1..n_N, tau_1..tau_N) at simple roots w (B,); shape (B, 2N), complex."""
    P, P_w, R, TF, e = sectors(w, n, tau, deriv=True)
    Q, S, M = _chain(P)
    U, sv, Vh = np.linalg.svd(M - _EYE4)
    x = Vh[:, -1, :].conj()             # right null vector
    y = U[:, :, -1]                      # left null vector: y^H A = 0
    yh = y.conj()
    left = np.einsum('bi,bnij->bnj', yh, S)              # y^H S_i
    right = np.einsum('bnij,bj->bni', Q, x)              # Q_i x
    den = np.einsum('bni,bnij,bnj->b', left, P_w, right)
    lr = np.einsum('bni,bij->bnj', left, R)              # y^H S_i R
    core = np.einsum('bni,bnij,bnj->bn', lr, TF, right)  # y^H S_i R dT/dF Q_i x
    wb = np.asarray(w)[:, None]
    dn = KAPPA * e * core
    dtau = KAPPA * n * 1j * wb * e * core
    return -np.concatenate([dn, dtau], 1) / den[:, None]


def growth_rate(X, case, grad=False, return_info=False):
    """QoI f_Im = max over the split pair of Im(omega)/(2 pi) [s^-1].

    X: (B, 38) physical inputs [n_1..n_19, tau_1..tau_19 (s)]. With grad=True also
    returns the exact gradient (B, 38) with respect to the physical inputs (of the
    component that attains the max). With return_info=True also returns a dict
    with both roots, the argument-principle count and the last Newton step.
    """
    X = np.atleast_2d(np.asarray(X, float))
    n, tau = X[:, :N], X[:, N:]
    roots, count, step = split_roots(n, tau, case)
    bad = np.abs(count - 2) > 1e-3
    if np.any(bad):
        raise RuntimeError(f'{bad.sum()} samples: argument-principle count != 2 '
                           f'(roots left the circle or a third root entered)')
    f = roots[:, 0].imag / (2 * np.pi)
    out = [f]
    if grad:
        G = root_gradient(roots[:, 0], n, tau).imag / (2 * np.pi)
        out.append(G)
    if return_info:
        out.append({'roots': roots, 'count': count, 'newton_step': step,
                    'omega_nominal': symmetric_root(case)})
    return out[0] if len(out) == 1 else tuple(out)
