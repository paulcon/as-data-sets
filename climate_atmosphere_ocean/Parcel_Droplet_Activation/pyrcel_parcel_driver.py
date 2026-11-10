"""Drive the pyrcel adiabatic cloud-parcel model for Parcel_Droplet_Activation.

One run: a single lognormal aerosol mode in 200 log-spaced dry-radius bins, equilibrated at
100 % relative humidity (initial supersaturation S0 = 0, as in Rothenberg and Wang 2016,
p. 1260), lifted adiabatically at a constant updraft speed V. The response is

    S_max = max of the parcel supersaturation S over the first Z_CAP = 400 m of ascent,

S decimal (S = RH - 1). It is located on a saved grid of 2001 heights (0 and 2000 geometric
steps from 1 cm to 400 m) and refined by a cubic Hermite interpolant built from the saved
states and the exact dS/dt (the method of pyrcel's integrator.max_supersaturation). When S
is still rising at 400 m the maximum is the value at 400 m, and `peaked` is 0.

Gradients of log10(S_max) are computed by JAX forward-mode automatic differentiation through
the bin construction, the equilibration root-find (implicit differentiation) and the diffrax
Kvaerno5 solve. The derivative of the peak location drops out (envelope theorem), so the
Hermite peak parameter is held fixed in the derivative, as in pyrcel.

The activated fraction uses Eq. (10) of the paper,
    N_act / N = (1/2) erfc( 2 ln(S_m / S_max) / (3 sqrt(2) ln sigma_g) ),
with S_m the approximate kappa-Koehler critical supersaturation of a dry particle of radius
mu_g at the initial temperature T; its gradient follows by the chain rule.

Inputs, in the coordinates of Table 1 (p. 1259) of the paper:
    log10 N [log10 cm-3], log10 mu_g [log10 um], sigma_g [-], kappa [-],
    log10 V [log10 m s-1], T [K], P [Pa], alpha_c [-]

Needs pyrcel 2.0.0 (pip install pyrcel==2.0.0; it brings JAX, diffrax, equinox and
optimistix). Without it, `pm` is None and every call raises ParcelError naming what to install.
pyrcel is (c) Daniel Rothenberg, BSD 3-clause (LICENSE-pyrcel.txt); this module imports it and
copies none of its code.
"""
import os
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import diffrax as dfx
    import pyrcel as pm
    from jax.scipy.special import erfc
    from pyrcel import constants as pc
    from pyrcel.equilibrate import equilibrate_initial_state
    from pyrcel.integrator import _TERM, atol_vector
    from pyrcel.parcel_aux import parcel_ode_sys
    from pyrcel.thermo import sigma_w
    from pyrcel.updraft import ConstantV
except ImportError:  # pragma: no cover
    pm = None

PYRCEL_VERSION = "2.0.0"
NAMES = ["log10_N", "log10_mu_g", "sigma_g", "kappa", "log10_V", "T", "P", "alpha_c"]
UNITS = ["log10 cm-3", "log10 um", "-", "-", "log10 m s-1", "K", "Pa", "-"]
LOWER = np.array([1.0, -3.0, 1.2, 0.0, -2.0, 240.0, 50000.0, 0.1])
UPPER = np.array([4.0, 1.0, 3.0, 1.2, 1.0, 310.0, 105000.0, 1.0])
NOMINAL = np.array([3.0, np.log10(0.05), 2.0, 0.54, np.log10(0.5), 283.0, 85000.0, 1.0])  # Fig. 3, p. 1265
NBINS = 200
Z_CAP = 400.0          # m of ascent searched for the supersaturation maximum
NZ = 2000              # geometric save heights from 1 cm to Z_CAP (plus z = 0)
S0 = 0.0               # initial supersaturation (100 % RH), paper p. 1260
RTOL = 1e-7            # pyrcel's default state rtol (integrator.STATE_RTOL)
MAX_STEPS = 200_000
S_IDX = 6              # index of S in pyrcel's state vector


class ParcelError(RuntimeError):
    pass


def _solve(y0, args, ts, rtol):
    """pyrcel's parcel ODE (Kvaerno5, PID step control, pyrcel's per-component atol) saved at
    ts; the same call as pyrcel.integrator._solve, with diffrax's forward-mode adjoint so that
    jax.jacfwd can differentiate it."""
    return dfx.diffeqsolve(
        _TERM, dfx.Kvaerno5(), t0=ts[0], t1=ts[-1], dt0=None, y0=y0, args=args,
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol_vector(NBINS)),
        saveat=dfx.SaveAt(ts=ts), max_steps=MAX_STEPS, throw=False, adjoint=dfx.ForwardMode())


def _need():
    if pm is None:
        raise ParcelError("pyrcel is not installed. Install it (pip install pyrcel==2.0.0, the "
                          "version that wrote the shipped files) and rerun.")


def _bins(logN, logmu, sg):
    """Dry radii (m) and number concentrations (m-3) of the 200 bins: log-spaced edges on
    [mu_g / (10 sigma_g), 10 sigma_g mu_g] and trapezoidal integration of dN/dr, the same
    construction as pyrcel.AerosolSpecies(..., bins=200)."""
    mu = 10.0 ** logmu
    N = 10.0 ** logN
    rs = jnp.logspace(jnp.log10(mu / (10.0 * sg)), jnp.log10(mu * 10.0 * sg), NBINS + 1)
    lns = jnp.log(sg)

    def pdf(r):
        return N / (jnp.sqrt(2.0 * jnp.pi) * lns * r) * jnp.exp(-jnp.log(r / mu) ** 2 / (2.0 * lns ** 2))

    Nis = 0.5 * (rs[1:] - rs[:-1]) * (pdf(rs[:-1]) + pdf(rs[1:]))
    return jnp.sqrt(rs[:-1] * rs[1:]) * 1e-6, Nis * 1e6


def _hermite_peak(t0, t1, S0_, S1, d0, d1):
    """Maximum of the cubic Hermite interpolant on [t0, t1]; the location is held fixed."""
    h = t1 - t0

    def p(u):
        return ((2 * u ** 3 - 3 * u ** 2 + 1) * S0_ + (-2 * u ** 3 + 3 * u ** 2) * S1
                + h * (u ** 3 - 2 * u ** 2 + u) * d0 + h * (u ** 3 - u ** 2) * d1)

    sg = jax.lax.stop_gradient
    a0, a1, b0, b1, hs = sg(S0_), sg(S1), sg(d0), sg(d1), sg(h)
    # dp/du = A u^2 + B u + C
    A = 6.0 * (a0 - a1) + 3.0 * hs * (b0 + b1)
    B = 6.0 * (a1 - a0) - 2.0 * hs * (2.0 * b0 + b1)
    C = hs * b0
    disc = B * B - 4.0 * A * C
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    As = jnp.where(jnp.abs(A) > 1e-300, A, 1.0)
    Bs = jnp.where(jnp.abs(B) > 1e-300, B, 1.0)
    cands = jnp.array([(-B + sq) / (2 * As), (-B - sq) / (2 * As), -C / Bs, 0.0, 1.0])
    ok = jnp.array([(disc >= 0) & (jnp.abs(A) > 1e-300), (disc >= 0) & (jnp.abs(A) > 1e-300),
                    (jnp.abs(A) <= 1e-300) & (jnp.abs(B) > 1e-300), True, True])
    ok = ok & (cands >= 0.0) & (cands <= 1.0)
    vals = jnp.where(ok, jax.vmap(p)(cands), -jnp.inf)
    u = sg(cands[jnp.argmax(vals)])
    return p(u)


def _core(theta, rtol):
    """log10 S_max and diagnostics for one input vector (JAX-traceable)."""
    logN, logmu, sg, kappa, logV, T, P, ac = (theta[i] for i in range(8))
    r_d, Ni = _bins(logN, logmu, sg)
    kap = jnp.full(NBINS, kappa)
    y0 = equilibrate_initial_state(T, S0, P, r_d, kap, Ni)
    V = 10.0 ** logV
    args = (r_d, Ni, kap, ac, ConstantV(V))
    zs = jnp.concatenate([jnp.zeros(1), jnp.geomspace(0.01, Z_CAP, NZ)])
    ts = zs / V
    sol = _solve(y0, args, ts, rtol)
    S = sol.ys[:, S_IDX]
    i = jax.lax.stop_gradient(jnp.argmax(S))
    n = ts.shape[0]
    ic = jnp.clip(i, 1, n - 2)
    ym, yc, yp = sol.ys[ic - 1], sol.ys[ic], sol.ys[ic + 1]
    dm, dc, dp = (parcel_ode_sys(ts[j], y, args)[S_IDX] for j, y in ((ic - 1, ym), (ic, yc), (ic + 1, yp)))
    lo = _hermite_peak(ts[ic - 1], ts[ic], ym[S_IDX], yc[S_IDX], dm, dc)
    hi = _hermite_peak(ts[ic], ts[ic + 1], yc[S_IDX], yp[S_IDX], dc, dp)
    smax = jnp.where(jax.lax.stop_gradient(lo >= hi), lo, hi)
    aux = {"ok": (sol.result == dfx.RESULTS.successful) & jnp.all(jnp.isfinite(y0))
           & jnp.all(y0[7:] > r_d) & (smax > 0),
           "steps": sol.stats["num_steps"], "peaked": i < n - 1, "z_smax": zs[i],
           "S_end": S[-1], "log10_smax": jnp.log10(smax)}
    return jnp.log10(smax), aux


def _act_frac(log10_smax, theta):
    """Eq. (10) activated fraction, S_m from approximate kappa-Koehler theory at T."""
    logmu, sg, kappa, T = theta[1], theta[2], theta[3], theta[5]
    rd = 10.0 ** logmu * 1e-6
    A = 2.0 * pc.Mw * sigma_w(T) / (pc.R * T * pc.rho_w)
    s_m = jnp.sqrt(4.0 * A ** 3 / (27.0 * kappa * rd ** 3))
    smax = 10.0 ** log10_smax
    u = 2.0 * jnp.log(s_m / smax) / (3.0 * jnp.sqrt(2.0) * jnp.log(sg))
    return 0.5 * erfc(u)


_jit = {}


def _fns(rtol):
    _need()
    if rtol not in _jit:
        f = jax.jit(lambda th: _core(th, rtol))
        vg = jax.jit(jax.jacfwd(lambda th: _core(th, rtol), has_aux=True))
        af = jax.jit(jax.value_and_grad(_act_frac, argnums=(0, 1)))
        _jit[rtol] = (f, vg, af)
    return _jit[rtol]


def _pack(x, val, aux, t0):
    out = {"log10_smax": float(val), "ok": bool(aux["ok"]), "steps": int(aux["steps"]),
           "peaked": int(bool(aux["peaked"])), "z_smax": float(aux["z_smax"]),
           "S_end": float(aux["S_end"])}
    out["wall_s"] = time.perf_counter() - t0
    return out


def run(x, rtol=RTOL):
    """One parcel run without gradients. Returns a dict with log10_smax, act_frac, ok,
    peaked, z_smax (m), S_end (S at 400 m), steps, wall_s."""
    x = np.asarray(x, float)
    if x[3] <= 0:
        raise ParcelError("kappa must be > 0: an insoluble mode has no equilibrium at 100 % RH")
    f, _, af = _fns(rtol)
    t0 = time.perf_counter()
    val, aux = f(jnp.asarray(x))
    out = _pack(x, val, aux, t0)
    a, _ = af(val, jnp.asarray(x))
    out["act_frac"] = float(a)
    return out


def run_with_grad(x, rtol=RTOL):
    """As run(), plus grad_log10_smax and grad_act_frac (arrays of 8, paper coordinates)."""
    x = np.asarray(x, float)
    if x[3] <= 0:
        raise ParcelError("kappa must be > 0: an insoluble mode has no equilibrium at 100 % RH")
    _, vg, af = _fns(rtol)
    t0 = time.perf_counter()
    g, aux = vg(jnp.asarray(x))
    val = aux["log10_smax"]
    out = _pack(x, val, aux, t0)
    a, (da_ds, da_dx) = af(val, jnp.asarray(x))
    g = np.asarray(g)
    out["act_frac"] = float(a)
    out["grad_log10_smax"] = g
    out["grad_act_frac"] = float(da_ds) * g + np.asarray(da_dx)
    return out


def fd_gradient(x, step=1e-3, rtol=RTOL):
    """Central differences of log10_smax and act_frac in the paper coordinates, step
    h_i = step * (UPPER_i - LOWER_i) / 2 (one-sided, second order, at the box faces)."""
    x = np.asarray(x, float)
    h = step * (UPPER - LOWER) / 2.0
    base = run(x, rtol)
    G = {k: np.zeros(8) for k in ("log10_smax", "act_frac")}
    for i in range(8):
        e = np.zeros(8)
        e[i] = h[i]
        if x[i] - h[i] > max(LOWER[i], 0.0 if i == 3 else -np.inf) and x[i] + h[i] <= UPPER[i]:
            fp, fm = run(x + e, rtol), run(x - e, rtol)
            for k in G:
                G[k][i] = (fp[k] - fm[k]) / (2 * h[i])
        elif x[i] + 2 * h[i] <= UPPER[i]:
            f1, f2 = run(x + e, rtol), run(x + 2 * e, rtol)
            for k in G:
                G[k][i] = (-3 * base[k] + 4 * f1[k] - f2[k]) / (2 * h[i])
        else:
            f1, f2 = run(x - e, rtol), run(x - 2 * e, rtol)
            for k in G:
                G[k][i] = (3 * base[k] - 4 * f1[k] + f2[k]) / (2 * h[i])
    return G, base
