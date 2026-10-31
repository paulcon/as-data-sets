"""Growth rate of the most unstable mode of the linearized 1-D Vlasov-Poisson system.

Reimplementation, from the paper only, of the dispersion-relation model of

    S. Terrab and S. Pankavich, "Global sensitivity analysis of plasma instabilities via
    active subspaces", Commun. Nonlinear Sci. Numer. Simul. (2024), article 107994,
    doi:10.1016/j.cnsns.2024.107994 (preprint arXiv:2106.08552v2, Eqs. (3)-(5)).

Two equilibria, all quantities dimensionless (s = sigma^2 is the "thermal velocity" variance):

* Bi-Maxwellian (``kind='bm'``), inputs x = (k, mu1, mu2, s1, s2, beta), paper Eq. (5):

      k^2 eps(k, w) = k^2 + beta/s1 [1 + A1 Z(A1)] + (1 - beta)/s2 [1 + A2 Z(A2)],
      A_i = (u - mu_i)/sqrt(2 s_i),  u = w/k.

* Two-Stream (``kind='ts'``), inputs x = (k, mu, s), paper Eq. (3):

      k^2 eps(k, w) = k^2 - [1 - 2A^2 + 2(A - A^3) Z(A)],  A = (u - mu)/sqrt(2 s).

Z is the plasma dispersion function, Z(z) = i sqrt(pi) w(z) with w the Faddeeva function
(scipy.special.wofz); this is the Landau-continued Z, analytic in the whole plane, so the
dispersion function D(u) = k^2 eps is entire in the phase velocity u.

Quantity of interest: gamma = Im(w) = k Im(u*) for the root u* of D with the LARGEST imaginary
part (the most unstable mode; when every root is damped, the least-damped one).

Root finding (`most_unstable_root`):

1. scan |D| on a grid of the complex u-plane that covers both beams and Im u in [-2.5, 3];
   every interior local minimum of |D| is a Newton start; Newton (analytic dD/du) is run on
   all starts at once; converged roots are deduplicated.
2. certify: count the zeros of D in the rectangle  Re u in [re_lo, re_hi],
   Im u in [Im u* - DELTA, im_top]  with the argument principle (adaptive sampling of the
   contour, phase steps < 0.5 rad) and compare with the number of roots found there. The top
   and side edges are placed where |D - k^2| < k^2/2 (D tends to k^2 for large |u| away from
   the Landau-damped sector), so no root lies outside. On a mismatch the scan is repeated on
   a 3x finer grid; a second mismatch raises `RootError`.

Every failure (Newton without convergence, a certification mismatch, an input outside its
limits, a non-finite value) raises `RootError` (a RuntimeError); a number is returned only for
a certified root.

Gradients: implicit function theorem, du/dx_j = -(dD/dx_j)/(dD/du) at the root, and
dgamma/dx_j = Im(u) [x_j is k] + k Im(du/dx_j).  Derivatives are with respect to the PHYSICAL
inputs.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy.special import wofz

SQPI = np.sqrt(np.pi)

# --- inputs ----------------------------------------------------------------------------------
NAMES = {'bm': ('k', 'mu1', 'mu2', 's1', 's2', 'beta'), 'ts': ('k', 'mu', 's')}
SYMBOLS = {'bm': ('k', 'mu_1', 'mu_2', 'sigma_1^2', 'sigma_2^2', 'beta'),
           'ts': ('k', 'mu', 'sigma^2')}
DESCRIPTIONS = {
    'bm': ('perturbation wavenumber', 'mean velocity of beam 1', 'mean velocity of beam 2',
           'thermal velocity (variance) of beam 1', 'thermal velocity (variance) of beam 2',
           'density fraction of beam 1 (scaling parameter)'),
    'ts': ('perturbation wavenumber', 'mean velocity', 'thermal velocity (variance)'),
}
#: Baselines, paper Table I (Two-Stream) and Table II (Double Beam, Bump-on-Tail).
NOMINAL_TS = np.array([0.5, 0.0, 1.0])
NOMINAL_DB = np.array([0.5, 0.0, 4.0, 0.5, 0.5, 0.5])
NOMINAL_BOT = np.array([0.5, 0.0, 4.0, 0.5, 0.5, 0.8])
#: Global Bi-Maxwellian box, paper Sec. V.B.3 (p. 29).
GLOBAL_LOWER = np.array([0.4, -0.1, 3.5, 0.25, 0.25, 0.5])
GLOBAL_UPPER = np.array([0.6, 0.1, 4.5, 0.75, 0.75, 0.99])


def percent_box(nominal, pct):
    """Bounds of the paper's percentage studies (p. 11): nominal*(1 -+ pct/100), and
    -+ pct/100 (absolute) for a zero nominal."""
    nominal = np.asarray(nominal, float)
    half = np.where(nominal == 0, pct/100, np.abs(nominal)*pct/100)
    return nominal - half, nominal + half


class RootError(RuntimeError):
    """The most unstable root could not be found and certified."""


# --- dispersion function ------------------------------------------------------------------------
def Z(z):
    """Plasma dispersion function (Landau-continued), Z(z) = i sqrt(pi) w(z)."""
    return 1j*SQPI*wofz(z)


def _check(x, kind):
    x = np.asarray(x, float)
    if kind not in NAMES:
        raise ValueError(f"kind must be 'bm' or 'ts', not {kind!r}")
    if x.shape != (len(NAMES[kind]),):
        raise ValueError(f'x must have {len(NAMES[kind])} entries {NAMES[kind]}')
    if not np.all(np.isfinite(x)):
        raise RootError(f'non-finite input {x}')
    if x[0] <= 0:
        raise RootError(f'k must be > 0 (got {x[0]})')
    if kind == 'bm':
        if x[3] <= 0 or x[4] <= 0:
            raise RootError('s1, s2 must be > 0')
        if not 0 < x[5] < 1:
            raise RootError(f'beta must be in (0, 1) (got {x[5]})')
    elif x[2] <= 0:
        raise RootError('s must be > 0')
    return x


def dispersion(u, x, kind='bm'):
    """D(u) = k^2 eps(k, k u) at complex phase velocities u (any shape)."""
    u = np.asarray(u, complex)
    if kind == 'bm':
        k, mu1, mu2, s1, s2, beta = x
        a1 = (u - mu1)/np.sqrt(2*s1)
        a2 = (u - mu2)/np.sqrt(2*s2)
        return k**2 + beta/s1*(1 + a1*Z(a1)) + (1 - beta)/s2*(1 + a2*Z(a2))
    k, mu, s = x
    a = (u - mu)/np.sqrt(2*s)
    return k**2 - (1 - 2*a**2 + 2*(a - a**3)*Z(a))


def dispersion_du(u, x, kind='bm'):
    """dD/du (analytic; Z'(z) = -2(1 + z Z(z)))."""
    u = np.asarray(u, complex)
    if kind == 'bm':
        k, mu1, mu2, s1, s2, beta = x
        out = 0
        for mu, s, c in ((mu1, s1, beta/s1), (mu2, s2, (1 - beta)/s2)):
            r = np.sqrt(2*s)
            a = (u - mu)/r
            z = Z(a)
            out = out + c*(z - 2*a*(1 + a*z))/r          # c R'(a) da/du, R = 1 + a Z
        return out
    k, mu, s = x
    r = np.sqrt(2*s)
    a = (u - mu)/r
    return -_ts_T_prime(a)/r


def _ts_T_prime(a):
    z = Z(a)
    zp = -2*(1 + a*z)
    return -4*a + 2*(1 - 3*a**2)*z + 2*(a - a**3)*zp


def dispersion_dx(u, x, kind='bm'):
    """Partial derivatives of D with respect to the physical inputs at fixed u (complex vector)."""
    u = complex(u)
    if kind == 'bm':
        k, mu1, mu2, s1, s2, beta = x
        g = np.zeros(6, complex)
        g[0] = 2*k
        R = []
        for j, (mu, s, c) in enumerate(((mu1, s1, beta/s1), (mu2, s2, (1 - beta)/s2))):
            r = np.sqrt(2*s)
            a = (u - mu)/r
            z = Z(a)
            Ra = 1 + a*z
            Rp = z - 2*a*Ra
            R.append(Ra)
            g[1 + j] = -c*Rp/r                        # d/dmu
            g[3 + j] = -c*Ra/s - c*Rp*a/(2*s)         # d/ds  (c = w/s, a = (u - mu)/sqrt(2 s))
        g[5] = R[0]/s1 - R[1]/s2
        return g
    k, mu, s = x
    r = np.sqrt(2*s)
    a = (u - mu)/r
    Tp = _ts_T_prime(a)
    return np.array([2*k, Tp/r, Tp*a/(2*s)], complex)


# --- root finding -------------------------------------------------------------------------------
IM_LO, IM_TOP = -2.5, 3.0     # initial scan of Im u
GRID_H = 0.05                 # scan spacing in u
DELTA = 0.05                  # certification rectangle starts DELTA below Im u*
NEWTON_TOL = 1e-13
NEWTON_MAXIT = 60


def _newton(u, x, kind):
    u = np.array(u, complex)
    ok = np.zeros(u.shape, bool)
    for _ in range(NEWTON_MAXIT):
        with np.errstate(all='ignore'):
            step = dispersion(u, x, kind)/dispersion_du(u, x, kind)
        step = np.where(ok | ~np.isfinite(step), 0, step)
        # damp very long steps
        big = np.abs(step) > 1.0
        with np.errstate(all='ignore'):
            step = np.where(big, step/np.abs(step), step)
        u = u - step
        ok |= np.abs(step) < NEWTON_TOL*(1 + np.abs(u))
        if ok.all():
            break
    with np.errstate(all='ignore'):
        resid = np.abs(dispersion(u, x, kind))
    good = ok & np.isfinite(u) & (resid < 1e-9)
    return u[good]


def _dedupe(roots, tol=1e-7):
    out = []
    for r in roots[np.argsort(-roots.imag)]:
        if all(abs(r - q) > tol*(1 + abs(r)) for q in out):
            out.append(r)
    return np.array(out, complex)


def _re_box(x, kind, im_lo):
    if kind == 'bm':
        mus, smax = (x[1], x[2]), max(x[3], x[4])
    else:
        mus, smax = (x[1],), x[2]
    L = 4*np.sqrt(2*smax) + 1.5 + 1.2*abs(min(im_lo, 0))
    return min(mus) - L, max(mus) + L


def _scan(x, kind, im_lo, im_hi, h):
    re_lo, re_hi = _re_box(x, kind, im_lo)
    ur = np.arange(re_lo, re_hi + h/2, h)
    ui = np.arange(im_lo, im_hi + h/2, h)
    U = ur[None, :] + 1j*ui[:, None]
    with np.errstate(all='ignore'):
        A = np.abs(dispersion(U, x, kind))
    A[~np.isfinite(A)] = np.inf
    P = np.pad(A, 1, constant_values=np.inf)
    nb = np.stack([P[1 + di:P.shape[0] - 1 + di, 1 + dj:P.shape[1] - 1 + dj]
                   for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj])
    mins = A <= nb.min(axis=0)
    return _dedupe(_newton(U[mins], x, kind))


def count_zeros(x, kind, re_lo, re_hi, im_lo, im_hi, max_pts=2**18):
    """Number of zeros of D in the rectangle (argument principle, adaptive sampling).

    Returns (count, min |D| / k^2 on the contour). Raises RootError when the winding number
    is not resolved."""
    corners = [re_lo + 1j*im_lo, re_hi + 1j*im_lo, re_hi + 1j*im_hi, re_lo + 1j*im_hi]
    per_unit = 64
    while True:
        pts = []
        for a, b in zip(corners, corners[1:] + corners[:1]):
            n = max(16, int(np.ceil(abs(b - a)*per_unit)))
            pts.append(a + (b - a)*np.arange(n)/n)
        pts = np.concatenate(pts + [corners[:1]])
        with np.errstate(all='ignore'):
            d = dispersion(pts, x, kind)
        if not np.all(np.isfinite(d)):
            raise RootError('non-finite D on the certification contour')
        dphi = np.angle(d[1:]/d[:-1])
        if np.max(np.abs(dphi)) < 0.5:
            w = dphi.sum()/(2*np.pi)
            if abs(w - round(w)) > 1e-3:
                raise RootError(f'winding number {w} is not an integer')
            return int(round(w)), float(np.min(np.abs(d))/x[0]**2)
        per_unit *= 2
        if len(pts) > max_pts:
            raise RootError('argument principle not resolved (a root on the contour?)')


def _far_field_ok(x, kind, re_lo, re_hi, im_lo, im_hi):
    """|D - k^2| < k^2/2 on the top edge and on the side edges: no roots beyond them."""
    k2 = x[0]**2
    t = np.linspace(0, 1, 400)
    edges = np.concatenate([re_lo + (re_hi - re_lo)*t + 1j*im_hi,
                            re_lo + 1j*(im_lo + (im_hi - im_lo)*t),
                            re_hi + 1j*(im_lo + (im_hi - im_lo)*t)])
    with np.errstate(all='ignore'):
        dev = np.abs(dispersion(edges, x, kind) - k2)
    return bool(np.all(dev < 0.5*k2))


def most_unstable_root(x, kind='bm', return_info=False, grid_h=GRID_H, delta=DELTA):
    """The root u* (phase velocity w/k) of D with the largest imaginary part, certified.

    info: dict with 'roots' (all roots found above the certification line), 'all_roots' (every
    root the scan found, by decreasing imaginary part; below the line not certified), 'count'
    (argument-principle count), 'refined' (True when the fine rescan was needed),
    'gap' (Im u* - Im of the next root found, in u units; nan if none).
    grid_h (scan spacing) and delta (depth of the certification rectangle below Im u*) are
    solver settings; the defaults were used for the shipped data."""
    x = _check(x, kind)
    if not (0 < grid_h <= 0.5 and 0 < delta <= 1):
        raise ValueError('need 0 < grid_h <= 0.5 and 0 < delta <= 1')
    refined = False
    im_lo = IM_LO
    for attempt in range(4):
        h = grid_h if not refined else grid_h/3
        roots = _scan(x, kind, im_lo, IM_TOP, h)
        if roots.size == 0:
            im_lo -= 2.0
            continue
        ustar = roots[0]
        bottom = ustar.imag - delta
        # keep the bottom edge away from any root found
        while np.any(np.abs(roots.imag - bottom) < 0.005):
            bottom -= 0.01
        re_lo, re_hi = _re_box(x, kind, min(bottom, im_lo))
        im_top = IM_TOP
        while not _far_field_ok(x, kind, re_lo, re_hi, bottom, im_top):
            im_top += 2.0
            re_lo, re_hi = re_lo - 2.0, re_hi + 2.0
            if im_top > 40:
                raise RootError('far field never reached |D - k^2| < k^2/2')
        inside = roots[(roots.imag > bottom) & (roots.real > re_lo) & (roots.real < re_hi)]
        count, _ = count_zeros(x, kind, re_lo, re_hi, bottom, im_top)
        if count == inside.size:
            if return_info:
                nxt = roots[1].imag if roots.size > 1 else np.nan
                return ustar, dict(roots=inside, count=count, refined=refined,
                                   gap=ustar.imag - nxt, all_roots=roots)
            return ustar
        if count > inside.size and ustar.imag > IM_TOP - 0.5:
            raise RootError('root above the scan window')
        if refined:
            break
        refined = True
    raise RootError(f'certification failed at x = {x.tolist()} ({kind}): the argument principle '
                    f'and the roots found disagree')


def growth_rate(x, kind='bm', **solver):
    """gamma = Im(omega) of the most unstable mode."""
    x = _check(x, kind)
    return float(x[0]*most_unstable_root(x, kind, **solver).imag)


def growth_rate_and_grad(x, kind='bm', return_info=False, **solver):
    """gamma, omega_r = Re(omega), and dgamma/dx (physical inputs) by the implicit function
    theorem."""
    x = _check(x, kind)
    u, info = most_unstable_root(x, kind, return_info=True, **solver)
    k = x[0]
    Du = complex(dispersion_du(u, x, kind))
    if abs(Du) < 1e-10:
        raise RootError('dD/du = 0 at the root (double root): gamma is not differentiable')
    du = -dispersion_dx(u, x, kind)/Du
    g = k*du.imag
    g[0] += u.imag
    out = (float(k*u.imag), float(k*u.real), g)
    return out + (info,) if return_info else out
