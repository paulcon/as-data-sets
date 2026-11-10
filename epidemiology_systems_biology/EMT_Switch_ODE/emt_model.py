"""E-cadherin / Slug switch model of epithelial-mesenchymal transition (EMT), as used by
K. I. Gasior, npj Syst. Biol. Appl. 12:143 (2026), doi:10.1038/s41540-026-00757-5
(model of Gasior et al. 2022, their ref. 15).

Equations (1)-(2) of the paper (p. 10), with membrane-bound E-cadherin E (ng/mL) and the
transcription factor Slug S (ng/mL), cell-cell contact C (cells) and exogenous TGF-beta T (ng/mL):

    dE/dt = alpha1 / (1 + (S/ICS)^n1) + k0 (C/ICC)^n2 / (1 + (C/ICC)^n2) - beta1 E
    dS/dt = alpha2 - k1 (E/ICE)^n3 / (1 + (E/ICE)^n3) + k2 (T/ICT)^n4 / (1 + (T/ICT)^n4) - beta2 S

Hill coefficients n1, n2, n3, n4 = 3, 4, 2, 3 are held fixed (Table 1, p. 9). The 11 inputs, in
the paper's order, are alpha1, alpha2, beta1, beta2, k0, k1, k2, ICS, ICE, ICT, ICC. Nominal
values: Table 1 (p. 9). Ranges: Supplementary Table S1 (+/-10 % and +/-25 %; the alpha2 and k1
ranges are shifted there, with their widths kept). Initial conditions: Supplementary Table S2
(the same for both TGF-beta levels; epithelial for C >= 1, mesenchymal for C = 0).

The quantities of interest are E and S at t = 10,000 min (p. 10). Here they are computed by
integrating from the initial condition to t = T_END with LSODA and then polishing the end state
to the equilibrium it has reached with Newton's method (the slowest decay rate at the stored draws of
this data set is 0.36 per minute, so the state at t = 10,000 min equals the equilibrium to
rounding). Gradients with respect to the 11 physical inputs follow from the implicit function
theorem at that equilibrium, dy/dp = -J_y^{-1} J_p; `forward_sensitivities` integrates the
sensitivity equations instead, for checking.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

NAMES = ('alpha1', 'alpha2', 'beta1', 'beta2', 'k0', 'k1', 'k2', 'ICS', 'ICE', 'ICT', 'ICC')
UNITS = ('ng/(mL min)', 'ng/(mL min)', '1/min', '1/min', 'ng/(mL min)', 'ng/(mL min)',
         'ng/(mL min)', 'ng/mL', 'ng/mL', 'ng/mL', 'cells')
DESCRIPTIONS = ('E-cadherin production', 'Slug production', 'E-cadherin degradation',
                'Slug degradation', 'rate at which E-cadherin moves to the membrane due to contact',
                'rate of Slug suppression by membrane-bound E-cadherin',
                'rate of Slug upregulation by TGF-beta',
                'half-maximal Slug for inhibiting E-cadherin production',
                'half-maximal E-cadherin for inhibiting Slug upregulation',
                'half-maximal TGF-beta for activating Slug',
                'half-maximal number of cells drawing E-cadherin to the membrane')
NOMINAL = np.array([0.19355, 0.080, 6.035, 0.830, 0.1245, 0.080, 0.040,
                    0.01920, 0.010, 3.64e-6, 2.17])            # Table 1, p. 9
HILL = (3, 4, 2, 3)                                            # n1..n4, Table 1, p. 9
# Supplementary Table S1 (p. 1 of the supplement), as printed
LOWER = {10: np.array([0.1742, 0.0772, 5.4315, 0.7470, 0.1120, 0.0669, 0.0360,
                       0.0173, 0.0090, 3.276e-6, 1.9530]),
         25: np.array([0.1452, 0.0772, 4.5263, 0.6225, 0.0938, 0.0429, 0.0300,
                       0.0144, 0.0075, 2.73e-6, 1.6275])}
UPPER = {10: np.array([0.2129, 0.0932, 6.6385, 0.9130, 0.1370, 0.0829, 0.0440,
                       0.0211, 0.0110, 4.004e-6, 2.3870]),
         25: np.array([0.2419, 0.1172, 7.5438, 1.0375, 0.1556, 0.0829, 0.0500,
                       0.024, 0.0125, 4.55e-6, 2.7125])}
# Supplementary Table S2: (E(0), S(0)) in ng/mL for each contact level
INITIAL = {0: (0.0003, 0.0963), 1: (0.0280, 0.0109), 2: (0.0399, 0.0057), 6: (0.0522, 0.0034)}
T_ON = 3.64e-6       # exogenous TGF-beta, ng/mL (pp. 10-11)
CONTACTS = (0, 1, 2, 6)
T_END = 1.0e4        # minutes (p. 10)
RTOL, ATOL = 1e-10, 1e-14
QOI_NAMES = ('E', 'S')


def _terms(y, x, C, T):
    E, S = y
    a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC = x
    n1, n2, n3, n4 = HILL
    s, q, r, u = S / ICS, C / ICC, E / ICE, T / ICT
    A = 1.0 + s**n1
    hc = q**n2 / (1.0 + q**n2)
    he = r**n3 / (1.0 + r**n3)
    ht = u**n4 / (1.0 + u**n4)
    return E, S, a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC, s, q, r, u, A, hc, he, ht


def rhs(t, y, x, C, T):
    """Right-hand side of Eqs. (1)-(2)."""
    E, S, a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC, s, q, r, u, A, hc, he, ht = _terms(y, x, C, T)
    return np.array([a1 / A + k0 * hc - b1 * E, a2 - k1 * he + k2 * ht - b2 * S])


def jac_y(y, x, C, T):
    """Jacobian of the right-hand side with respect to (E, S), shape (2, 2)."""
    E, S, a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC, s, q, r, u, A, hc, he, ht = _terms(y, x, C, T)
    n1, n2, n3, n4 = HILL
    dhe_dE = n3 * r**(n3 - 1) / (ICE * (1.0 + r**n3)**2)
    return np.array([[-b1, -a1 * n1 * s**(n1 - 1) / (ICS * A**2)],
                     [-k1 * dhe_dE, -b2]])


def jac_p(y, x, C, T):
    """Jacobian of the right-hand side with respect to the 11 physical inputs, shape (2, 11)."""
    E, S, a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC, s, q, r, u, A, hc, he, ht = _terms(y, x, C, T)
    n1, n2, n3, n4 = HILL
    J = np.zeros((2, 11))
    J[0, 0] = 1.0 / A
    J[0, 2] = -E
    J[0, 4] = hc
    J[0, 7] = a1 * n1 * s**n1 / (ICS * A**2)
    J[0, 10] = -k0 * n2 * q**n2 / (ICC * (1.0 + q**n2)**2)
    J[1, 1] = 1.0
    J[1, 3] = -S
    J[1, 5] = -he
    J[1, 6] = ht
    J[1, 8] = k1 * n3 * r**n3 / (ICE * (1.0 + r**n3)**2)
    J[1, 9] = -k2 * n4 * u**n4 / (ICT * (1.0 + u**n4)**2)
    return J


def equilibria(x, C=2, T=T_ON, npts=4000):
    """All equilibria (E, S) of Eqs. (1)-(2), shape (k, 2), sorted by S.

    dE/dt = 0 gives E as a function of S; the remaining scalar equation in S is scanned on a
    grid over every S an equilibrium can have, (alpha2 - k1)/beta2 <= S <= (alpha2 + k2)/beta2,
    and each sign change is refined with Brent's method. One equilibrium means the treatment group
    is monostable at x; three mean bistable (two stable, one saddle)."""
    x = np.asarray(x, float)
    a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC = x
    n2 = HILL[1]
    hc = (C / ICC)**n2 / (1.0 + (C / ICC)**n2)

    def Ebar(S):
        return (a1 / (1.0 + (S / ICS)**HILL[0]) + k0 * hc) / b1

    def g(S):
        return rhs(0.0, (Ebar(S), S), x, C, T)[1]

    lo = min(0.0, (a2 - k1) / b2) - 1e-9
    hi = (a2 + k2) / b2 + 1e-9
    grid = np.linspace(lo, hi, npts)
    vals = np.array([g(S) for S in grid])
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) <= 0)[0]:
        if vals[i] == 0.0:
            roots.append(grid[i])
        elif vals[i + 1] != 0.0:
            roots.append(brentq(g, grid[i], grid[i + 1], xtol=1e-17, rtol=1e-15))
    roots = np.unique(np.array(roots))
    return np.array([[Ebar(S), S] for S in roots]).reshape(-1, 2)


def trajectory(x, C=2, T=T_ON, t_eval=None, y0=None, rtol=RTOL, atol=ATOL, t_end=T_END):
    """Integrate Eqs. (1)-(2) from the initial condition of Table S2 (or y0); returns the
    solve_ivp result."""
    x = np.asarray(x, float)
    y0 = INITIAL[C] if y0 is None else y0
    sol = solve_ivp(rhs, (0.0, t_end), np.array(y0, float), args=(x, C, T), method='LSODA',
                    jac=lambda t, y, *a: jac_y(y, *a), rtol=rtol, atol=atol, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f'integration failed: {sol.message}')
    return sol


def qoi(x, C=2, T=T_ON, grad=False, rtol=RTOL, atol=ATOL, return_info=False):
    """E and S (ng/mL) at t = 10,000 min, and optionally their gradients (2, 11) with respect to
    the physical inputs.

    The end state of the LSODA trajectory is polished to the equilibrium it has reached by
    Newton's method; RuntimeError is raised if the end state is not within 1e-6 (relative) of an
    equilibrium, or if Newton's method does not converge. The iteration stops when the correction
    is below 1e-15 (relative). At some inputs the corrections cycle at the rounding level, a few
    1e-15, without meeting that; the state after 20 iterations is then accepted if its last
    correction is below 1e-12. With return_info, a dict is also returned with the integrated end
    state, the polish correction, the eigenvalues of the Jacobian and the number of equilibria."""
    x = np.asarray(x, float)
    sol = trajectory(x, C, T, rtol=rtol, atol=atol)
    y_end = sol.y[:, -1].copy()
    y = y_end.copy()
    for _ in range(20):
        F = rhs(0.0, y, x, C, T)
        dy = np.linalg.solve(jac_y(y, x, C, T), -F)
        y = y + dy
        if np.all(np.abs(dy) <= 1e-15 * np.maximum(np.abs(y), 1e-12)):
            break
    else:
        if not np.all(np.abs(dy) <= 1e-12 * np.maximum(np.abs(y), 1e-12)):
            raise RuntimeError('Newton polish did not converge')
    corr = np.abs(y - y_end) / np.abs(y)
    if np.max(corr) > 1e-6:
        raise RuntimeError(f'state at t = {T_END:g} min is not at an equilibrium (relative gap {np.max(corr):.1e})')
    out = [y]
    Jy = jac_y(y, x, C, T)
    if grad:
        out.append(-np.linalg.solve(Jy, jac_p(y, x, C, T)))
    if return_info:
        out.append({'y_end': y_end, 'polish': corr, 'eig': np.linalg.eigvals(Jy),
                    'n_equilibria': len(equilibria(x, C, T))})
    return tuple(out) if len(out) > 1 else y


def qois(X, C=2, T=T_ON, grad=False, **kw):
    """qoi for each row of X (M, 11): returns f (M, 2) and, with grad, G (M, 2, 11)."""
    X = np.atleast_2d(np.asarray(X, float))
    res = [qoi(x, C, T, grad=grad, **kw) for x in X]
    if grad:
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])
    return np.array(res)


def forward_sensitivities(x, C=2, T=T_ON, rtol=1e-11, atol=1e-16, t_end=T_END):
    """(E, S) at t_end and their gradients (2, 11) from the forward sensitivity equations
    dZ/dt = J_y Z + J_p, Z(0) = 0 (the initial condition does not depend on the inputs)."""
    x = np.asarray(x, float)

    def aug(t, z):
        y = z[:2]
        Z = z[2:].reshape(2, 11)
        return np.concatenate([rhs(t, y, x, C, T), (jac_y(y, x, C, T) @ Z + jac_p(y, x, C, T)).ravel()])

    z0 = np.concatenate([np.array(INITIAL[C], float), np.zeros(22)])
    sol = solve_ivp(aug, (0.0, t_end), z0, method='LSODA', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'integration failed: {sol.message}')
    return sol.y[:2, -1], sol.y[2:, -1].reshape(2, 11)


def count_equilibria(X, C=2, T=T_ON, npts=4000, chunk=512):
    """Vectorized number of equilibria for many input rows X (M, 11), shape (M,) int.

    The same scalar equation in S as `equilibria` (E from dE/dt = 0), evaluated on `npts`
    equally spaced points of its bracket for every row; the count is the number of sign changes
    (plus exact zeros). 1 means monostable, 3 bistable. Two roots closer together than the grid
    spacing can be missed, so this is a scan, not a proof; `equilibria` refines single rows."""
    X = np.atleast_2d(np.asarray(X, float))
    n1, n2, n3, n4 = HILL
    out = np.empty(len(X), int)
    grid = np.linspace(0.0, 1.0, npts)[None, :]
    for s0 in range(0, len(X), chunk):
        a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC = (col[:, None] for col in X[s0:s0 + chunk].T)
        hc = (C / ICC)**n2 / (1.0 + (C / ICC)**n2)
        ht = (T / ICT)**n4 / (1.0 + (T / ICT)**n4)
        lo = np.minimum(0.0, (a2 - k1) / b2) - 1e-9
        hi = (a2 + k2 * ht) / b2 + 1e-9
        S = lo + (hi - lo) * grid
        r = ((a1 / (1.0 + (S / ICS)**n1) + k0 * hc) / b1 / ICE)**n3
        sg = np.sign(a2 - k1 * r / (1.0 + r) + k2 * ht - b2 * S)
        out[s0:s0 + chunk] = np.sum(sg[:, :-1] * sg[:, 1:] < 0, axis=1) + np.sum(sg == 0, axis=1)
    return out


def unique_equilibrium(X, C=2, T=T_ON, iters=64):
    """Vectorized equilibrium (E, S) for many input rows X (M, 11), shape (M, 2).

    Valid only where the treatment group is monostable (one equilibrium, as `equilibria` or
    `count_equilibria` reports), because then the trajectory from any initial condition ends
    there; at a bistable row the bisection returns one of the three equilibria, not necessarily
    the one the trajectory from Table S2 reaches. The scalar
    equation for S (see `equilibria`) is solved by `iters` bisection steps on its bracket (64 steps
    shrink it below double-precision resolution) and E follows from dE/dt = 0. Used for the
    variance-based (Sobol') indices, which need many runs."""
    X = np.atleast_2d(np.asarray(X, float))
    a1, a2, b1, b2, k0, k1, k2, ICS, ICE, ICT, ICC = X.T
    n1, n2, n3, n4 = HILL
    hc = (C / ICC)**n2 / (1.0 + (C / ICC)**n2)
    ht = (T / ICT)**n4 / (1.0 + (T / ICT)**n4)

    def g(S):
        E = (a1 / (1.0 + (S / ICS)**n1) + k0 * hc) / b1
        return a2 - k1 * (E / ICE)**n3 / (1.0 + (E / ICE)**n3) + k2 * ht - b2 * S, E

    lo = np.minimum(0.0, (a2 - k1) / b2) - 1e-9
    hi = (a2 + k2 * ht) / b2 + 1e-9
    glo = g(lo)[0]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        gm = g(mid)[0]
        same = np.sign(gm) == np.sign(glo)
        lo = np.where(same, mid, lo)
        glo = np.where(same, gm, glo)
        hi = np.where(same, hi, mid)
    S = 0.5 * (lo + hi)
    E = g(S)[1]
    return np.column_stack([E, S])
