"""Viscous pipe flow model of Constantine, del Rosario and Iaccarino (arXiv:1708.04303, 2017), Sec. 6.

Five independent variables q = (rho, mu, D, eps, V) in the order of the paper's dimension
matrix (Eq. (62), p. 32): fluid density rho (kg/m^3), fluid viscosity mu (kg/(m s)), pipe
diameter D (m), pipe roughness eps (m) and bulk velocity V (m/s). The dependent variable is the
pressure loss per unit length dp/dx (kg m^-2 s^-2, i.e. Pa/m).

    Re = rho V D / mu,                                     Reynolds number
    1/sqrt(lam) = -2.0 log10(eps/(3.7 D) + 2.51/(Re sqrt(lam))),   Colebrook, Eq. (60), p. 30
    dp/dx = lam rho V^2 / (2 D),                           Eq. (61), p. 30

The paper solves the Colebrook equation with a Newton method for every flow case, laminar ones
included (p. 30); `law="poiseuille"` gives lam = 64/Re (Eq. (59)) instead, for comparison only.

The dimensionless pressure loss is pi = (dp/dx) exp(-w^T ln q) with w = (1, 0, -1, 0, 2)
(Sec. 6.1, p. 32), i.e. pi = (dp/dx) D / (rho V^2) = lam / 2.

Gradients are exact (implicit differentiation of the Colebrook equation) and are taken with
respect to x = ln q, the coordinates of the paper's analysis (Sec. 2.3 and 3.2).

Only numpy is imported.
"""
import numpy as np

NAMES = ('rho', 'mu', 'D', 'eps', 'V')
SYMBOLS = (r'\rho', r'\mu', 'D', r'\varepsilon', 'V')
UNITS = ('kg/m^3', 'kg/(m s)', 'm', 'm', 'm/s')
DESCRIPTIONS = ('fluid density', 'fluid viscosity', 'pipe diameter', 'pipe roughness', 'fluid velocity')

REGIMES = ('laminar', 'turbulent', 'high_re')
# Bounds of the independent variables (Tables 2-4, pp. 31-32), in the order of NAMES.
BOUNDS = {
    'laminar': (np.array([1.0e-1, 1.0e-6, 5.0e-1, 3.0e-5, 2.5e-2]),
                np.array([1.4e-1, 1.0e-5, 8.0e-1, 8.0e-5, 3.0e-2])),
    'turbulent': (np.array([1.0e-1, 1.0e-6, 5.0e-1, 5.0e-4, 2.0e0]),
                  np.array([1.4e-1, 1.0e-5, 1.0e0, 2.0e-3, 4.0e0])),
    'high_re': (np.array([1.0e-1, 1.0e-6, 5.0e-1, 1.0e-2, 5.0e2]),
                np.array([1.4e-1, 1.0e-5, 1.0e0, 4.0e-2, 7.0e2])),
}
SOURCES = {'laminar': 'Table 2, p. 31', 'turbulent': 'Table 3, p. 31', 'high_re': 'Table 4, p. 32'}

# Dimension matrix, rows kg, m, s; columns rho, mu, D, eps, V (Eq. (62), p. 32; Table 1, p. 30).
DIM_MATRIX = np.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                       [-3.0, -1.0, 1.0, 1.0, 1.0],
                       [0.0, -1.0, 0.0, 0.0, -1.0]])
DIM_DPDX = np.array([1.0, -2.0, -2.0])                 # dimension vector of dp/dx (Sec. 6.1)
W_PI = np.array([1.0, 0.0, -1.0, 0.0, 2.0])            # D W_PI = DIM_DPDX (Sec. 6.1, p. 32)
W_RE = np.array([1.0, -1.0, 1.0, 0.0, 1.0])            # exponents of Re = rho V D / mu
W_RR = np.array([0.0, 0.0, -1.0, 1.0, 0.0])            # exponents of eps / D
RE_CRIT = 3.0e3                                        # critical Reynolds number quoted on p. 29

NEWTON_RTOL = 1e-15
NEWTON_MAXIT = 50
_LN10 = np.log(10.0)


def reynolds(Q):
    """Reynolds number rho V D / mu of the rows of Q, shape (M,)."""
    Q = np.atleast_2d(Q)
    return Q[:, 0] * Q[:, 4] * Q[:, 2] / Q[:, 1]


def relative_roughness(Q):
    """Relative roughness eps / D of the rows of Q, shape (M,)."""
    Q = np.atleast_2d(Q)
    return Q[:, 3] / Q[:, 2]


def colebrook(Re, rr):
    """Solve the Colebrook equation (60) for s = 1/sqrt(lam) by Newton's method.

    Returns lam and the derivatives of ln lam with respect to ln a and ln b, where
    a = rr / 3.7 and b = 2.51 / Re. Works for complex arguments (complex-step checks).
    Raises RuntimeError if Newton's method does not converge.
    """
    a = rr / 3.7
    b = 2.51 / Re
    s = np.full(np.shape(Re), 8.0, dtype=np.result_type(Re, rr, float))
    for _ in range(NEWTON_MAXIT):
        t = a + b * s
        F = s + 2.0 * np.log10(t)
        Fs = 1.0 + 2.0 / _LN10 * b / t
        ds = F / Fs
        s = s - ds
        if np.all(np.abs(ds) <= NEWTON_RTOL * np.abs(s)):
            break
    else:
        raise RuntimeError('Colebrook Newton iteration did not converge')
    t = a + b * s
    Fs = 1.0 + 2.0 / _LN10 * b / t
    ds_dlna = -(2.0 / _LN10 * a / t) / Fs
    ds_dlnb = -(2.0 / _LN10 * b * s / t) / Fs
    return s ** -2, -2.0 * ds_dlna / s, -2.0 * ds_dlnb / s


def colebrook_residual(lam, Re, rr):
    """Residual 1/sqrt(lam) + 2 log10(rr/3.7 + 2.51/(Re sqrt(lam))) of Eq. (60)."""
    s = 1.0 / np.sqrt(lam)
    return s + 2.0 * np.log10(rr / 3.7 + 2.51 / Re * s)


def _check(Q):
    Q = np.atleast_2d(np.asarray(Q))
    if Q.ndim != 2 or Q.shape[1] != 5:
        raise ValueError('Q must have shape (M, 5): rho, mu, D, eps, V')
    if np.isrealobj(Q) and not np.all(Q > 0):
        raise ValueError('all independent variables must be positive')
    return Q


def friction_factor(Q, law='colebrook', grad=False):
    """Darcy friction factor lam (dimensionless), shape (M,); with grad=True also
    d ln lam / d ln q, shape (M, 5)."""
    Q = _check(Q)
    Re, rr = reynolds(Q), relative_roughness(Q)
    if law == 'colebrook':
        lam, dla, dlb = colebrook(Re, rr)
        # ln a = ln eps - ln D - ln 3.7 and ln b = ln 2.51 - ln Re
        dln = np.outer(dla, W_RR) - np.outer(dlb, W_RE)
    elif law == 'poiseuille':
        lam = 64.0 / Re
        dln = -np.outer(np.ones_like(Re), W_RE)
    else:
        raise ValueError('law must be "colebrook" or "poiseuille"')
    return (lam, dln) if grad else lam


def pressure_gradient(Q, law='colebrook', grad=False):
    """Pressure loss per unit length dp/dx = lam rho V^2 / (2 D) (kg m^-2 s^-2), shape (M,);
    with grad=True also d(dp/dx)/d ln q, shape (M, 5), in the same units."""
    Q = _check(Q)
    lam, dln = friction_factor(Q, law=law, grad=True)
    dp = lam * Q[:, 0] * Q[:, 4] ** 2 / (2.0 * Q[:, 2])
    return (dp, dp[:, None] * (dln + W_PI)) if grad else dp


def dimensionless_loss(Q, law='colebrook', grad=False):
    """Dimensionless pressure loss pi = (dp/dx) D / (rho V^2) = lam / 2, shape (M,);
    with grad=True also d pi / d ln q, shape (M, 5)."""
    lam, dln = friction_factor(Q, law=law, grad=True)
    pi = lam / 2.0
    return (pi, pi[:, None] * dln) if grad else pi


def null_basis():
    """Orthonormal basis W (5, 2) of the null space of DIM_MATRIX: the last two right singular
    vectors (Sec. 6.1, p. 32). Column signs are numpy's SVD convention."""
    return np.linalg.svd(DIM_MATRIX)[2][3:].T


def sample(regime, M, rng):
    """M independent draws, uniform on the box of `regime` in the physical variables q."""
    lo, up = BOUNDS[regime]
    return lo + rng.random((M, 5)) * (up - lo)


def gauss_legendre(regime, npts=11):
    """Tensor-product Gauss-Legendre rule for the uniform density on the box of `regime`
    (npts points per variable): points (npts^5, 5) in q and weights (npts^5,) summing to 1."""
    lo, up = BOUNDS[regime]
    t, w = np.polynomial.legendre.leggauss(npts)
    grids = np.meshgrid(*([t] * 5), indexing='ij')
    wg = np.meshgrid(*([w / 2.0] * 5), indexing='ij')
    T = np.column_stack([g.ravel() for g in grids])
    weights = np.prod(np.column_stack([g.ravel() for g in wg]), axis=1)
    return lo + (T + 1.0) / 2.0 * (up - lo), weights
