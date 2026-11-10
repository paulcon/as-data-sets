"""Limit state of the undamped yielding oscillator of Rajashekhar and Ellingwood (1993), Example 2.

Paper: Sec. 4.2, Fig. 10 and Eq. (12), p. 217; Table 7, p. 218:

    g(x) = 3 r - |z_max|,    failure when g <= 0,

with z_max the largest displacement of the single-degree-of-freedom oscillator of Fig. 10: a
mass m held by a linear spring c1 and, in parallel, a spring c2 that yields at displacement r
(yield force c2 r), loaded from rest by a rectangular pulse of height F1 and duration t1, with
no damping. Fig. 10 draws r beside spring c2 and a bilinear restoring force (slope c1 + c2 up to
z = r, then a smaller slope, unloading parallel to the first slope). This module reads that as
c1 linear in parallel with an elastic-perfectly-plastic c2 with elastic unloading and the same
yield displacement in compression (kinematic, not isotropic: the elastic range of c2 is always
[-r, r] in its own elastic deformation s = z - p, p the plastic offset). Those last two points
(perfectly plastic c2; symmetric compressive yield) are this module's reading of the sketch.

Between switching events the equation of motion is linear with constant coefficients,

    elastic (|s| < r):      m z'' + c1 z + c2 (z - p) = F(t),
    yielding (s = +-r):     m z'' + c1 z +- c2 r       = F(t),    while the velocity keeps its sign,

so each phase is a harmonic motion about its own equilibrium, with frequency
w0 = sqrt((c1 + c2)/m) (elastic) or w1 = sqrt(c1/m) (yielding). `response` solves the motion
exactly, event by event (pulse end t = t1; s reaching +-r; velocity zero), in closed form, until
the pulse has ended and the motion is elastic for good, after which it is periodic. z_max is the
largest |z(t)| over all t >= 0, so no time window or integrator is involved.

Gradients: forward-mode differentiation (dual numbers) of the same event-by-event closed form,
including the dependence of every event time on the inputs. This is the exact derivative of z_max
wherever z_max is differentiable; z_max is only piecewise smooth (it has kinks where the pulse end
coincides with a yield event or with the end of yielding, and where the largest |z| switches
from one peak to another; `response` returns the margins to these events).

The closed-form peak of the *linear* oscillator of stiffness c1 + c2 (the form of the later
benchmark literature, e.g. UQTestFuns 0.7.0 `UndampedOscillator`, which credits Gayton et al.
2003) is kept as `limit_state_linear` for the secondary study:

    z_lin = 2 F1 / (m w0^2) sin(w0 t1 / 2),    valid for w0 t1 / 2 <= pi/2.

Inputs x = (m, c1, c2, r, F1, t1), independent, with the means and standard deviations of
Table 7 (p. 218), lognormal in the paper (p. 217). Units are not stated (consistent units).

Only numpy is imported.
"""
import numpy as np

NAMES = ('m', 'c1', 'c2', 'r', 'F1', 't1')
DESCRIPTIONS = ('mass', 'stiffness of spring 1 (linear)', 'stiffness of spring 2 (yields at r)',
                'yield displacement of spring 2', 'pulse load', 'pulse duration')
UNITS = ('-',) * 6   # not stated in the sources; consistent units
MEAN = np.array([1.00, 1.00, 0.10, 0.50, 1.00, 1.00])   # Table 7, p. 218
SD = np.array([0.05, 0.10, 0.01, 0.05, 0.20, 0.20])     # Table 7, p. 218
# lognormal parameters with the mean and sd of x itself: ln x ~ N(MU, SIGMA^2)
SIGMA = np.sqrt(np.log1p((SD / MEAN) ** 2))
MU = np.log(MEAN) - 0.5 * SIGMA ** 2

TWO_PI = 2.0 * np.pi
MAX_EVENTS = 5000         # safety cap on the number of events per sample (raises if reached)


# ----------------------------------------------------------------------------------------------
# minimal forward-mode dual numbers: value v (N,) and derivative d (N, 6) or None
# ----------------------------------------------------------------------------------------------
class _D:
    __slots__ = ('v', 'd')

    def __init__(self, v, d=None):
        self.v = v
        self.d = d

    @staticmethod
    def lift(a, like):
        if isinstance(a, _D):
            return a
        a = np.broadcast_to(np.asarray(a, dtype=float), like.v.shape).copy()
        return _D(a, None if like.d is None else np.zeros_like(like.d))

    def __add__(self, o):
        o = _D.lift(o, self)
        return _D(self.v + o.v, None if self.d is None else self.d + o.d)
    __radd__ = __add__

    def __neg__(self):
        return _D(-self.v, None if self.d is None else -self.d)

    def __sub__(self, o):
        return self + (-_D.lift(o, self))

    def __rsub__(self, o):
        return _D.lift(o, self) - self

    def __mul__(self, o):
        o = _D.lift(o, self)
        d = None if self.d is None else self.d * o.v[:, None] + o.d * self.v[:, None]
        return _D(self.v * o.v, d)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _D.lift(o, self)
        v = self.v / o.v
        d = None if self.d is None else (self.d - o.d * v[:, None]) / o.v[:, None]
        return _D(v, d)

    def __rtruediv__(self, o):
        return _D.lift(o, self) / self

    def chain(self, f, df):
        return _D(f, None if self.d is None else self.d * df[:, None])


def _sqrt(a):
    s = np.sqrt(a.v)
    with np.errstate(divide='ignore', invalid='ignore'):
        return a.chain(s, np.where(s > 0, 0.5 / s, 0.0))


def _sin(a):
    return a.chain(np.sin(a.v), np.cos(a.v))


def _cos(a):
    return a.chain(np.cos(a.v), -np.sin(a.v))


def _arccos(a):
    c = np.clip(a.v, -1.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = np.where(np.abs(c) < 1.0, -1.0 / np.sqrt(1.0 - c * c), 0.0)
    return a.chain(np.arccos(c), dd)


def _atan2(y, x):
    den = x.v * x.v + y.v * y.v
    v = np.arctan2(y.v, x.v)
    if y.d is None:
        return _D(v)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(den[:, None] > 0, (x.v[:, None] * y.d - y.v[:, None] * x.d) / den[:, None], 0.0)
    return _D(v, d)


def _abs(a):
    return a.chain(np.abs(a.v), np.sign(a.v))


def _where(cond, a, b):
    d = None
    if a.d is not None:
        d = np.where(cond[:, None], a.d, b.d)
    return _D(np.where(cond, a.v, b.v), d)


def _shift(a, c):
    """a + c with c a plain array (an integer number of periods): derivative unchanged."""
    return _D(a.v + c, a.d)


# ----------------------------------------------------------------------------------------------
# the yielding oscillator, solved event by event
# ----------------------------------------------------------------------------------------------
_E, _PP, _PM = 0, 1, -1            # modes: elastic, yielding in tension (s = +r), in compression
_EV_T1, _EV_UP, _EV_DN, _EV_V0, _EV_PEND = 0, 1, 2, 3, 4


def response(X, grad=False, full=None):
    """Exact response of the Fig. 10 oscillator for inputs X of shape (M, 6) or (6,).

    Returns a dict with
      z_max     largest |z(t)| over t >= 0;
      g         3 r - z_max;
      grad      (if grad) exact gradient of g with respect to the physical inputs, (M, 6);
      grad_z    (if grad) the same for z_max;
      n_yield   number of yielding phases;
      yielded   whether spring 2 yields at all;
      p_final   final plastic offset of spring 2;
      n_events  number of events solved;
      peak_index which velocity-zero peak has the largest |z| (1 = the first);
      peak_sign  sign of z there;
      peak_gap   (zmax - second largest peak |z|) / zmax, the relative margin to a switch of the
                 peak that attains z_max (where z_max has a kink); with full=False only over the
                 peaks solved before the stop;
      margins   (if grad) dict of first-order distances, in the standardized log inputs
                y = (ln x - MU)/SIGMA, to two switching sets:
                'pulse_end'   the pulse ends at the same time as a yield event or the end of yielding,
                'yield_onset' a peak just touches the yield level (yielding starts or stops there).
                The restoring force is continuous at both, so z_max stays differentiable there
                (its curvature can jump); they are reported so that finite differences across
                them can be checked.
    Each margin is the minimum over the trajectory of |gap| / ||grad_y gap||.

    After the pulse no external work is done and yielding dissipates energy, so the energy
    E = m v^2/2 + c1 z^2/2 + c2 s^2/2 never increases and every later |z| is at most
    sqrt(2 E / c1). With full=False (the default without grad) a sample stops as soon as that
    bound is at most the largest |z| found so far; z_max is unchanged by the stop. With full=True
    (the default with grad, needed for the peak-switch margin) every event is solved until the
    motion is elastic for good.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return _response(X, grad, full)


def _take(a, idx):
    return _D(a.v[idx], None if a.d is None else a.d[idx])


def _put(a, idx, b):
    a.v[idx] = b.v
    if a.d is not None:
        a.d[idx] = b.d


def _response(X, grad, full):
    """Event-by-event solution; see response."""
    if full is None:
        full = grad
    X = np.asarray(X, dtype=float)
    single = X.ndim == 1
    X = np.atleast_2d(X)
    if not np.all(np.isfinite(X)) or np.any(X <= 0):
        raise ValueError('oscillator inputs must be finite and positive')
    N = len(X)

    def dual(v, seed=None):
        d = None
        if grad:
            d = np.zeros((len(v), 6))
            if seed is not None:
                d[:, seed] = 1.0
        return _D(np.array(v, dtype=float), d)

    P = [dual(X[:, j], j) for j in range(6)]                 # parameters (full size)
    S = {n: dual(np.zeros(N)) for n in ('t', 'z', 'v', 'p', 'zmax')}
    S['second'] = dual(np.full(N, -np.inf))
    mode = np.full(N, _E)
    on = np.ones(N, bool)
    active = np.ones(N, bool)
    n_yield = np.zeros(N, int)
    n_ev = np.zeros(N, int)
    npk = np.zeros(N, int)
    ipk = np.zeros(N, int)
    zsign = np.zeros(N)
    marg = {'pulse_end': np.full(N, np.inf), 'yield_onset': np.full(N, np.inf)}
    s_y = SIGMA * X                       # d/dy_i = s_i x_i d/dx_i
    big = np.inf

    def dist(gap, sy):
        """first-order standardized distance |gap| / ||grad_y gap||."""
        nrm = np.linalg.norm(gap.d * sy, axis=1)
        return np.where(nrm > 0, np.abs(gap.v) / nrm, np.inf)

    it = 0
    while active.any():
        it += 1
        if it > MAX_EVENTS:
            raise RuntimeError(f'oscillator response: more than {MAX_EVENTS} events for {int(active.sum())} samples')
        idx = np.flatnonzero(active)
        n = len(idx)
        m, c1, c2, r, F1, t1 = (_take(q, idx) for q in P)
        t, z, v, p, zmax, second = (_take(S[q], idx) for q in ('t', 'z', 'v', 'p', 'zmax', 'second'))
        md, pon, sy = mode[idx], on[idx], s_y[idx]
        zero = _D.lift(0.0, m)
        BIG = _D.lift(big, m)
        k = c1 + c2
        w0 = _sqrt(k / m)
        w1 = _sqrt(c1 / m)
        F = _where(pon, F1, zero)
        el = md == _E
        om_ = _where(el, w0, w1)
        ze = _where(el, (F + c2 * p) / k, _where(md == _PP, (F - c2 * r) / c1, (F + c2 * r) / c1))
        a = z - ze
        b = v / om_
        R = _sqrt(a * a + b * b)
        psi0 = _atan2(-b, a)              # z - ze = R cos(psi), v = -R w sin(psi), psi' = w
        cu = (p + r - ze) / R
        cd = (p - r - ze) / R
        tol = 1e-12
        reach_u = el & (np.abs(cu.v) < 1 - tol)
        reach_d = el & (np.abs(cd.v) < 1 - tol)

        def phase_time(target, rows, period):
            dpsi = target - psi0
            kk = np.floor(dpsi.v / period)
            dpsi = _shift(dpsi, -kk * period)                        # in [0, period)
            dpsi = _shift(dpsi, np.where(dpsi.v < 1e-9, period, 0.0))
            return _where(rows, dpsi / om_, BIG)

        cand = [None] * 5
        cand[_EV_T1] = _where(pon, t1 - t, BIG)
        cand[_EV_UP] = phase_time(-_arccos(cu), reach_u, TWO_PI)
        cand[_EV_DN] = phase_time(_arccos(cd), reach_d, TWO_PI)
        cand[_EV_V0] = phase_time(zero, el, np.pi)
        cand[_EV_PEND] = phase_time(_where(md == _PP, zero, _D.lift(np.pi, m)), ~el, TWO_PI)
        imm = (~el) & (np.abs(b.v) <= 1e-14 * np.maximum(R.v, 1e-300)) & \
              np.where(md == _PP, a.v >= 0, a.v <= 0)                # yield reached with zero velocity
        cand[_EV_PEND] = _where(imm, zero, cand[_EV_PEND])

        T = np.stack([c.v for c in cand], axis=1)
        ev = np.argmin(T, axis=1)

        def record_peak(zz, rows):
            nonlocal zmax, second
            npk[idx[rows]] += 1
            aa = _abs(zz)
            newmax = rows & (aa.v > zmax.v)
            newsec = rows & ~newmax & (aa.v > second.v)
            second = _where(newmax, zmax, _where(newsec, aa, second))
            zmax = _where(newmax, aa, zmax)
            ipk[idx[newmax]] = npk[idx[newmax]]
            zsign[idx[newmax]] = np.sign(zz.v[newmax])

        # pulse over and elastic for good: periodic motion with peaks ze +- R, done
        free = el & ~pon & ~reach_u & ~reach_d
        go = ~free
        if free.any():
            record_peak(_where(ze.v >= 0, ze + R, ze - R), free)
            if grad:
                gap = _where((ze.v + R.v) - (p.v + r.v) > (p.v - r.v) - (ze.v - R.v),
                             (ze + R) - (p + r), (p - r) - (ze - R))
                d_ = dist(gap, sy)
                rows = idx[free]
                marg['yield_onset'][rows] = np.minimum(marg['yield_onset'][rows], d_[free])
        n_ev[idx[go]] += 1

        if grad:
            # pulse end against the nearest yield event or end of yielding
            others = np.stack([cand[_EV_UP].v, cand[_EV_DN].v, cand[_EV_PEND].v], axis=1)
            jo = np.argmin(others, axis=1)
            rows = go & pon & np.isfinite(others[np.arange(n), jo])
            if rows.any():
                other = _where(jo == 0, cand[_EV_UP], _where(jo == 1, cand[_EV_DN], cand[_EV_PEND]))
                d_ = dist(cand[_EV_T1] - other, sy)
                marg['pulse_end'][idx[rows]] = np.minimum(marg['pulse_end'][idx[rows]], d_[rows])

        tau = cand[0]
        for e in range(1, 5):
            tau = _where(ev == e, cand[e], tau)
        tau = _where(go, tau, zero)
        ph = psi0 + om_ * tau
        z_new = ze + R * _cos(ph)
        v_new = -(R * om_) * _sin(ph)
        t_new = t + tau
        up, dn = go & (ev == _EV_UP), go & (ev == _EV_DN)
        v0, pe, te = go & (ev == _EV_V0), go & (ev == _EV_PEND), go & (ev == _EV_T1)
        z_new = _where(up, p + r, _where(dn, p - r, z_new))          # exact values at the events
        v_new = _where(v0 | pe, zero, v_new)
        t_new = _where(te, t1, t_new)
        if grad and (up | dn).any():
            # yielding starts: distance to a touch (the elastic peak exactly at the yield level)
            rows = up | dn
            gap = _where(up, (ze + R) - (p + r), (p - r) - (ze - R))
            d_ = dist(gap, sy)
            marg['yield_onset'][idx[rows]] = np.minimum(marg['yield_onset'][idx[rows]], d_[rows])
        z = _where(go, z_new, z)
        v = _where(go, v_new, v)
        t = _where(go, t_new, t)
        md = np.where(up, _PP, np.where(dn, _PM, md))
        n_yield[idx] += up | dn
        p = _where(pe & (md == _PP), z - r, _where(pe & (md == _PM), z + r, p))
        md = np.where(pe, _E, md)
        pon = pon & ~te
        record_peak(z, v0 | pe)
        done = free.copy()
        if not full:
            xs = X[idx]
            sv = np.where(md == _PP, xs[:, 3], np.where(md == _PM, -xs[:, 3], z.v - p.v))
            E = 0.5 * (xs[:, 0] * v.v ** 2 + xs[:, 1] * z.v ** 2 + xs[:, 2] * sv ** 2)
            done |= ~pon & (2.0 * E <= xs[:, 1] * zmax.v ** 2)
        for q, val in (('t', t), ('z', z), ('v', v), ('p', p), ('zmax', zmax), ('second', second)):
            _put(S[q], idx, val)
        mode[idx], on[idx] = md, pon
        active[idx[done]] = False

    zmax, second = S['zmax'], S['second']
    with np.errstate(invalid='ignore'):
        peak_gap = np.where(np.isfinite(second.v), (zmax.v - second.v) / zmax.v, np.inf)
    out = {'z_max': zmax.v, 'g': 3.0 * X[:, 3] - zmax.v, 'n_yield': n_yield,
           'yielded': n_yield > 0, 'p_final': S['p'].v, 'n_events': n_ev,
           'peak_index': ipk, 'peak_sign': zsign, 'peak_gap': peak_gap}
    if grad:
        gz = zmax.d
        gg = -gz.copy()
        gg[:, 3] += 3.0
        out['grad'] = gg
        out['grad_z'] = gz
        out['margins'] = marg
    if single:
        out = {k_: (v_[0] if isinstance(v_, np.ndarray) else {kk: vv[0] for kk, vv in v_.items()})
               for k_, v_ in out.items()}
    return out


def limit_state(X, grad=False):
    """g = 3 r - z_max of the yielding oscillator (Fig. 10). With grad=True also the exact
    gradient with respect to the physical inputs (where z_max is differentiable)."""
    res = response(X, grad=grad)
    return (res['g'], res['grad']) if grad else res['g']


# ----------------------------------------------------------------------------------------------
# the linear closed form (secondary study)
# ----------------------------------------------------------------------------------------------
def parts_linear(X):
    """Return (z_lin, B) for inputs X of shape (M, 6) or (6,), with B = w0 t1 / 2."""
    X = np.asarray(X, dtype=float)
    m, c1, c2, r, F1, t1 = np.moveaxis(X, -1, 0)
    k = c1 + c2
    w0 = np.sqrt(k / m)
    B = 0.5 * w0 * t1
    return 2.0 * F1 / k * np.sin(B), B


def limit_state_linear(X, grad=False):
    """g_lin = 3 r - |z_lin| with the closed-form peak of the linear oscillator of stiffness c1 + c2.

    With grad=True also the exact gradient with respect to the physical inputs. Where z_lin = 0
    (the kink of |.|) the gradient uses sign(0) = 0; no sample of the designs here is near it.
    """
    X = np.asarray(X, dtype=float)
    m, c1, c2, r, F1, t1 = np.moveaxis(X, -1, 0)
    k = c1 + c2
    w0 = np.sqrt(k / m)
    B = 0.5 * w0 * t1
    A = 2.0 * F1 / k
    sB, cB = np.sin(B), np.cos(B)
    z = A * sB
    g = 3.0 * r - np.abs(z)
    if not grad:
        return g
    dz_dm = A * cB * (-B / (2.0 * m))
    dz_dk = -A / k * sB + A * cB * B / (2.0 * k)
    dz_dF1 = 2.0 / k * sB
    dz_dt1 = A * cB * 0.5 * w0
    sgn = np.sign(z)
    G = np.stack([-sgn * dz_dm, -sgn * dz_dk, -sgn * dz_dk, 3.0 * np.ones_like(g),
                  -sgn * dz_dF1, -sgn * dz_dt1], axis=-1)
    return g, G


# ----------------------------------------------------------------------------------------------
# sampling and failure probability
# ----------------------------------------------------------------------------------------------
def sample_lognormal(M, rng):
    """M draws of the paper's inputs (lognormal, Table 7 mean and sd of x), shape (M, 6)."""
    return np.exp(MU + SIGMA * rng.standard_normal((M, 6)))


def sample_normal(M, rng):
    """M draws of the UQTestFuns 'Gayton2003' inputs (normal, same mean and sd), shape (M, 6)."""
    return MEAN + SD * rng.standard_normal((M, 6))


def wilson(nf, N, z=1.959963984540054):
    """95 % Wilson interval for a binomial proportion nf / N."""
    p = nf / N
    den = 1 + z ** 2 / N
    c = (p + z ** 2 / (2 * N)) / den
    h = z * np.sqrt(p * (1 - p) / N + z ** 2 / (4 * N ** 2)) / den
    return c - h, c + h


def failure_probability(sampler, N, rng, chunk=10 ** 5, mean=None, sd=None, model=limit_state,
                        positive_only=False):
    """Plain Monte Carlo estimate of P[g <= 0] with N draws in chunks.

    sampler is sample_lognormal or sample_normal; for the normal sampler, mean and sd may
    replace MEAN and SD (e.g. the reduced pulse loads of Echard et al. 2013). model is
    limit_state (yielding oscillator) or limit_state_linear. With positive_only, draws with a
    non-positive input are counted and dropped (the yielding model needs positive inputs).
    Returns (pf, n_fail, n_used, 95 % Wilson interval (lo, hi), n_dropped).
    """
    nf = 0
    done = 0
    used = 0
    dropped = 0
    while done < N:
        n = min(chunk, N - done)
        if mean is None:
            X = sampler(n, rng)
        else:
            X = mean + sd * rng.standard_normal((n, 6))
        if positive_only:
            keep = (X > 0).all(axis=1)
            dropped += int((~keep).sum())
            X = X[keep]
        nf += int(np.count_nonzero(model(X) <= 0))
        used += len(X)
        done += n
    pf = nf / used
    return pf, nf, used, wilson(nf, used), dropped
