"""Real-data acceptance checks for V1-01 (RBF search interval) and V1-03 (polynomial intervals).

Run with the branch under test first on the path (see the task). One line per check:
PASS|FAIL|SKIP name: measured vs target.
"""
import os, sys, time, warnings
import numpy as np
import scipy.stats as st
import scipy.optimize as op

T0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import active_subspaces as ac  # noqa: E402  (branch src is first on PYTHONPATH)
from common import (R, hydrology_grass, hyshot30, HYSHOT_XL, HYSHOT_XU, ORDER_G, YL, YU,  # noqa: E402
                    _rd)

print("active_subspaces file:", ac.__file__)
RES = []


def report(status, name, msg):
    RES.append(status)
    print(f"{status} {name}: {msg}", flush=True)


def check(name, ok, measured, target):
    report("PASS" if ok else "FAIL", name, f"{measured} vs {target}")


def wcodes(w):
    return sorted({getattr(x.message, "code", None) or type(x.message).__name__ for x in w})


# ---------------------------------------------------------------- V1-01
def elliptic(case):
    sys.path.append(f"{R}/specs/prototypes/g2")
    from fvsolver import PDESolver
    fn = f"{R}/uq_benchmarks/Elliptic_PDE_Random_Coefficients/kl-{case}.npz"
    if not os.path.exists(fn):
        return None
    s = PDESolver(fn)
    rng = np.random.default_rng(2014)
    X = rng.standard_normal((300, 100))
    fg = [s.qoi_and_grad(x) for x in X]
    f0 = np.array([a for a, _ in fg]); G = np.array([b for _, b in fg])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = ac.active_subspace(space=ac.Gaussian.reference(100), grads=G)
    Xt = rng.standard_normal((500, 100)); ft = np.array([s.qoi_and_grad(x)[0] for x in Xt])
    return s, f0, sub, Xt, ft


def fit_case(s, f0, sub, Xt, ft, n, **kw):
    y1 = np.array([-3, -1.5, 0, 1.5, 3.])
    sn = ac.with_dimension(sub, n)
    Yd = y1[:, None] if n == 1 else np.array(np.meshgrid(y1, y1, indexing="ij")).reshape(2, -1).T
    Xd = Yd @ sn.W1.T
    fd = np.array([s.qoi_and_grad(x)[0] for x in Xd])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rs = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind="rbf", degree=2, **kw)
    err = np.mean(np.abs(ac.predict(rs, Xt) - ft) / np.abs(ft))
    return rs, err, w


try:
    for case, thr in (("long", {1: (0.12, 0.095, 0.178), 2: (0.33, 0.300, None)}),
                      ("short", {1: (0.01, 0.008, None), 2: (0.01, 0.008, None)})):
        d = elliptic(case)
        if d is None:
            report("SKIP", f"V1-01 elliptic {case}", f"missing {R}/uq_benchmarks/Elliptic_PDE_Random_Coefficients/kl-{case}.npz")
            continue
        s, f0, sub, Xt, ft = d
        fv = f0.var()
        for n, (tol, rep, paper) in thr.items():
            rs, err, w = fit_case(s, f0, sub, Xt, ft, n, f_variance=fv)
            pr = rs.provenance.parameters
            print(f"   [{case} n={n}] log10_eta={rs.log10_eta:.4f} bounds={np.round(pr['log10_eta_bounds'], 4).tolist()} "
                  f"source={pr['log10_eta_bounds_source']} f_variance={pr['f_variance']:.4e} warnings={wcodes(w)}")
            check(f"V1-01 beta={'1' if case == 'long' else '0.01'} n={n} mean rel err", err <= tol,
                  f"{err:.4f} (report {rep}" + (f", paper {paper}" if paper else "") + ")", f"<= {tol}")
            if case == "short" and n == 1:
                check("V1-01 beta=0.01 empty_bounds issued", "accuracy.empty_bounds" in wcodes(w),
                      wcodes(w), "contains accuracy.empty_bounds")
            if case == "long" and n == 1:
                check("V1-01 provenance records f_variance and source",
                      abs(pr["f_variance"] - fv) <= 1e-12 * fv and pr["log10_eta_bounds_source"] is not None,
                      f"{pr['f_variance']:.6e}, {pr['log10_eta_bounds_source']}", f"{fv:.6e}")
        if case == "long":
            rs, err, w = fit_case(s, f0, sub, Xt, ft, 1)
            fw = [x for x in w if getattr(x.message, "code", None) == "accuracy.flat_objective"]
            # D-212: the default plateau reaches the lower end of the grid (the polynomial-tail
            # limit), so it no longer warns.
            check("V1-01 n=1 default: lower-edge plateau, no accuracy.flat_objective (D-212)",
                  len(fw) == 0, wcodes(w), "no accuracy.flat_objective")
            for x in fw:
                print(f"   flat_objective: {x.message}; details={getattr(x.message, 'details', None)}")
except Exception as e:  # keep going to V1-03
    import traceback; traceback.print_exc()
    report("FAIL", "V1-01 elliptic (exception)", repr(e))

# ---------------------------------------------------------------- V1-03
def hand(s, Yq, level=0.95, kind="prediction", side="two"):
    """Hand formula of r07b for a 1-D active variable, quadratic."""
    B = np.vander(s.Y[:, 0], 3, increasing=True); M, p = B.shape
    r = s.residuals; s2 = r @ r / (M - p)
    Bq = np.vander(Yq, 3, increasing=True); h = np.einsum("ij,jk,ik->i", Bq, np.linalg.inv(B.T @ B), Bq)
    q = 0.5 + level / 2 if side == "two" else level
    t = st.t.ppf(q, M - p); g = s.evaluate(Yq[:, None])
    half = t * np.sqrt(s2 * ((1 + h) if kind == "prediction" else h))
    if side == "two":
        return g - half, g + half
    return (g - half, np.full_like(g, np.inf)) if side == "lower" else (np.full_like(g, -np.inf), g + half)


try:
    if not os.path.exists(f"{R}/aerospace_cfd/HyShotII_Scramjet/HyShotII_phi030.csv"):
        report("SKIP", "V1-03 HyShotII 0.30", f"missing {R}/aerospace_cfd/HyShotII_Scramjet/HyShotII_phi030.csv")
    else:
        X, f = hyshot30(); space = ac.Uniform(HYSHOT_XL, HYSHOT_XU)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sub = ac.active_subspace(X=X, f=f, space=space, estimator="ols", n=1)
            s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
        yb = np.abs(sub.W1[:, 0]).sum()
        Yq = np.array([[-yb], [0.0], [yb]])
        lo, hi = s.interval(Yq)
        print(f"   yb={yb:.4f} lower={np.round(lo, 4)} upper={np.round(hi, 4)}")
        check("V1-03 HyShotII 0.30 95% PI lower", np.allclose(lo, [2.0348, 2.3672, 2.7402], atol=1e-4), np.round(lo, 4), "[2.0348 2.3672 2.7402]")
        check("V1-03 HyShotII 0.30 95% PI upper", np.allclose(hi, [2.0961, 2.4065, 2.7982], atol=1e-4), np.round(hi, 4), "[2.0961 2.4065 2.7982]")
        cl, ch = s.interval([[0.0]], kind="confidence")
        check("V1-03 HyShotII 0.30 95% CI at y=0", np.allclose([cl[0], ch[0]], [2.3836, 2.3901], atol=1e-4),
              np.round([cl[0], ch[0]], 4), "[2.3836, 2.3901]")
        # predict_interval at physical points mapped to the same y
        ys = np.array([-1.0, 0.0, 0.5, 1.5])
        Zq = ys[:, None] * sub.W1[:, 0][None, :]
        Xq = space.from_normalized(Zq)
        pi = ac.predict_interval(s, Xq)
        l2, u2 = s.interval((Zq @ sub.W1[:, 0])[:, None])
        okp = np.allclose(pi.lower, l2, rtol=1e-9) and np.allclose(pi.upper, u2, rtol=1e-9)
        check("V1-03 predict_interval (physical) == interval (active)", okp,
              f"max|diff|={max(np.abs(pi.lower - l2).max(), np.abs(pi.upper - u2).max()):.2e}", "<= 1e-9 rel")
        print(f"   dof={pi.dof} residual_scale={pi.residual_scale:.6f}")
        # one-sided 99% lower vs hand
        l1, u1 = s.interval(Yq, level=0.99, side="lower")
        hl, _ = hand(s, Yq[:, 0], level=0.99, side="lower")
        check("V1-03 side=lower level=0.99 == one-sided t", np.allclose(l1, hl, rtol=1e-10) and np.all(np.isposinf(u1)),
              f"max rel diff {np.max(np.abs(l1 - hl) / np.abs(hl)):.2e}", "<= 1e-10, upper=+inf")
        lu, uu = s.interval(Yq, level=0.99, side="upper")
        _, hu = hand(s, Yq[:, 0], level=0.99, side="upper")
        check("V1-03 side=upper level=0.99 == one-sided t", np.allclose(uu, hu, rtol=1e-10) and np.all(np.isneginf(lu)),
              f"max rel diff {np.max(np.abs(uu - hu) / np.abs(hu)):.2e}", "<= 1e-10, lower=-inf")
except Exception as e:
    import traceback; traceback.print_exc()
    report("FAIL", "V1-03 HyShotII 0.30 (exception)", repr(e))

# Hydrology, six cases
try:
    hy = [f"{R}/hydrology_subsurface_geoscience/Land_Surface_Energy_Fluxes/{n}" for n in ("SC_ScatterParameters_grass.txt", "SC_ScatterParameters_bare.txt",
                                          "ScatterSummary_grass.txt", "ScatterSummary_bare.txt")]
    miss = [p for p in hy if not os.path.exists(p)]
    if miss:
        report("SKIP", "V1-03 Hydrology", f"missing {miss[0]}")
    else:
        Xg, Fg = hydrology_grass()
        Zg = Xg.copy(); uni = [j for j in range(19) if j != 16]
        Zg[:, uni] = 2 * (Xg[:, uni] - YL) / (YU - YL) - 1; Zg[:, 16] = (np.log(Xg[:, 16]) + 5.11) / .52
        hb, Xb = _rd(hy[1]); Xb = Xb[:, [hb.index(n) for n in ("z0m", "soilcol", "watertable")]]
        Fb = np.loadtxt(hy[3])
        Zb = Xb.copy(); Zb[:, 0] = (np.log(Xb[:, 0]) + 8.83) / 1.47
        Zb[:, 1] = 2 * (Xb[:, 1] - 1) / 7 - 1; Zb[:, 2] = 2 * (Xb[:, 2] + 10) / 9 - 1
        worst = 0.0
        for tag, Z, F in (("grass", Zg, Fg), ("bare", Zb, Fb)):
            m = Z.shape[1]; sp = ac.Gaussian(np.zeros(m), np.ones(m))
            for j, qoi in enumerate(("LH", "SH", "GH")):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sub = ac.active_subspace(X=Z, f=F[:, j], space=sp, estimator="ols", n=1)
                    s = ac.fit_surrogate(X=Z, f=F[:, j], subspace=sub, kind="poly", degree=2)
                Yq = np.linspace(-3, 3, 13)
                dmax = 0.0
                for kind in ("prediction", "confidence"):
                    lo, hi = s.interval(Yq[:, None], kind=kind)
                    hl, hh = hand(s, Yq, kind=kind)
                    dmax = max(dmax, np.max(np.abs(lo - hl) / np.abs(hl)), np.max(np.abs(hi - hh) / np.abs(hh)))
                worst = max(worst, dmax)
                if tag == "grass" and qoi == "LH":
                    l0, h0 = s.interval([[0.0]]); c0l, c0h = s.interval([[0.0]], kind="confidence")
                    pih, cih = (h0[0] - l0[0]) / 2, (c0h[0] - c0l[0]) / 2
                    check("V1-03 Hydrology grass LH PI half-width y=0", abs(pih - 1.3919e5) <= 1e-4 * 1.3919e5 + 5,
                          f"{pih:.4e}", "1.3919e5 (rtol 1e-4)")
                    check("V1-03 Hydrology grass LH CI half-width y=0", abs(cih - 1.0119e4) <= 1e-4 * 1.0119e4 + 0.5,
                          f"{cih:.4e}", "1.0119e4 (rtol 1e-4)")
                print(f"   hydrology {tag} {qoi}: max rel diff vs hand = {dmax:.2e}")
        check("V1-03 Hydrology six cases == hand formula", worst <= 1e-10, f"{worst:.2e}", "<= 1e-10 rel")
except Exception as e:
    import traceback; traceback.print_exc()
    report("FAIL", "V1-03 Hydrology (exception)", repr(e))

# HyShotII phi = 0.35 safe-box limit
try:
    fn35 = f"{R}/aerospace_cfd/HyShotII_Scramjet/HyShotII_phi035.csv"
    if not os.path.exists(fn35):
        report("SKIP", "V1-03 HyShotII 0.35", f"missing {fn35}")
    else:
        d35 = np.loadtxt(fn35, delimiter=",")  # run, 7 physical inputs, f(0.30), f(0.35), validation
        X35 = d35[:, 1:8]
        f35 = d35[:, 9]
        sp = ac.Uniform(HYSHOT_XL, HYSHOT_XU)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sub = ac.active_subspace(X=X35[:-2], f=f35[:-2], space=sp, estimator="ols", n=1)
            s = ac.fit_surrogate(X=X35, f=f35, subspace=sub, kind="poly", degree=2)
        w = sub.W1[:, 0]
        # Legacy limit: g(y) + 2.33 s sqrt(1 + h(y)) with the exact quadratic leverage h; the library's
        # prediction half-width is t_nu s sqrt(1 + h), so rescale it by 2.33 / t_nu.
        res = ac.predict_interval(s, sp.from_normalized(np.zeros((1, 7))))
        tq = st.t.ppf(0.975, res.dof)

        def lim(y):
            lo, hi = s.interval(np.array([[y]]))
            g = s.evaluate(np.array([[y]]))[0]
            return g + (hi[0] - g) / tq * 2.33
        y0 = op.brentq(lambda y: lim(y) - 2.8, -3, 3)
        xmin = -np.sign(w)
        obj = lambda x: -np.sum(np.log(np.abs(x - xmin) + 1e-300))
        r = op.minimize(obj, np.zeros(7), bounds=[(-1, 1)] * 7, method="SLSQP",
                        constraints={"type": "ineq", "fun": lambda x: y0 - x @ w})
        xop = sp.from_normalized(r.x[None, :])[0]; xm = sp.from_normalized(xmin[None, :])[0]
        aoa = max(xop[0], xm[0])
        print(f"   phi=0.35 ymax={y0:.4f}, AoA range [{min(xop[0], xm[0]):.4f}, {aoa:.4f}], "
              f"TI {np.sort([xop[1], xm[1]])[1]:.4f}, H0 {np.sort([xop[4], xm[4]])[0]:.4e}, slsqp ok={r.success}")
        check("V1-03 HyShotII 0.35 AoA limit (xmin=-sign(w))", r.success and abs(aoa - 3.193) < 5e-4,
              f"{aoa:.4f}", "3.193")
except Exception as e:
    import traceback; traceback.print_exc()
    report("FAIL", "V1-03 HyShotII 0.35 (exception)", repr(e))

print(f"{RES.count('PASS')} passed, {RES.count('FAIL')} failed, {RES.count('SKIP')} skipped")
print(f"runtime {time.time() - T0:.1f} s")
