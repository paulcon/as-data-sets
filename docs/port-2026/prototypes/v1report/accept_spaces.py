"""Acceptance checks, branch as-port-spaces: V1-06, V1-10, V1-12 (real data)."""
import io, time, warnings
import numpy as np
import active_subspaces as ac
from common import *

print("active_subspaces from:", ac.__file__)
warnings.simplefilter("ignore")
T0 = time.time()
res = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def rep(ok, name, text):
    tag = "PASS" if ok else "FAIL"
    res[tag] += 1
    print(f"{tag} {name}: {text}", flush=True)


def skip(name, why):
    res["SKIP"] += 1
    print(f"SKIP {name}: {why}", flush=True)


def guard(name, fn):
    try:
        fn()
    except FileNotFoundError as e:
        skip(name, f"missing data file {e.filename}")
    except Exception as e:
        rep(False, name, f"{type(e).__name__}: {str(e)[:200]}")


# ---------------- V1-06 Hydrology
def v106_hydrology():
    X, F = hydrology_grass()
    Xt = X.copy(); Xt[:, 16] = np.log(X[:, 16])
    uni = [j for j in range(19) if j != 16]
    Z = Xt.copy()
    Z[:, uni] = 2 * (X[:, uni] - YL) / (YU - YL) - 1
    Z[:, 16] = (Xt[:, 16] + 5.11) / .52
    sp = ac.Product([ac.Uniform(YL[:16], YU[:16]), ac.Gaussian([-5.11], [0.52 ** 2]),
                     ac.Uniform(YL[16:], YU[16:])])
    rep(sp.kind == "product" and sp.dim == 19, "V1-06 hydrology kind", f"kind={sp.kind} dim={sp.dim} vs product/19")
    wk = ac.Gaussian(np.zeros(19), np.ones(19), names=ORDER_G)
    subs = {}
    for k, (lab, tgt) in enumerate((("LH", 6.68147e10), ("SH", 7.68786e10), ("GH", 6.77473e9))):
        s = ac.active_subspace(X=Xt, f=F[:, k], space=sp, estimator="ols", n=1)
        sw = ac.active_subspace(X=Z, f=F[:, k], space=wk, estimator="ols", n=1)
        subs[lab] = s
        l, lw = s.eigenvalues[0], sw.eigenvalues[0]
        rel = abs(l - lw) / abs(lw)
        rep(rel <= 1e-10, f"V1-06 hydrology OLS lambda1 {lab} vs workaround",
            f"{l:.15g} vs {lw:.15g} (rel {rel:.2e}, target 1e-10); 6-digit {l:.6g} vs {tgt:.6g}")
        dot = abs(float(s.W[:, 0] @ sw.W[:, 0]))
        rep(abs(dot - 1) <= 1e-10, f"V1-06 hydrology |w.w_legacy| {lab}", f"{dot:.15f} vs 1 (tol 1e-10)")
    s = subs["LH"]
    d = np.diag(s.space.reference_covariance)
    exp = np.full(19, 1 / 3); exp[16] = 1.0
    rep(np.allclose(d, exp, rtol=1e-14, atol=0) and np.allclose(d[uni], 1 / 3) and d[16] == 1.0,
        "V1-06 reference_covariance", f"18 cols {d[uni].min():.6f}..{d[uni].max():.6f} vs 1/3; col16 {d[16]} vs 1")
    try:
        ac.active_domain(s)
        rep(False, "V1-06 active_domain unsupported", "no exception raised")
    except ac.UnsupportedError as e:
        rep(e.code == "unsupported.space", "V1-06 active_domain unsupported", f"{type(e).__name__} code={e.code} vs unsupported.space")
    rep(ac.Product.from_dict(sp.to_dict()) == sp, "V1-06 space to_dict/from_dict", "equal")
    rep(ac.from_dict(s.to_dict()) == s, "V1-06 result to_dict/from_dict", "equal")
    buf = io.BytesIO(); s.save(buf); buf.seek(0)
    s2 = ac.load(buf)
    rep(s2 == s and s2.space == sp, "V1-06 result save/load (BytesIO)", "equal")


# ---------------- V1-06 Reentry
def v106_reentry():
    d = np.loadtxt(f"{R}/Reentry_freestream_conditions/SA_lhs_all.txt", delimiter=",", skiprows=1)
    Xu0, Xn0, pst = d[:, 1:4], d[:, 4:8], d[:, 8]
    lb, ub = np.array([16.3, 13.7, 0.001]), np.array([24.3, 17.3, 0.002])
    sd = np.array([.12, .12, .10, .10])
    print(f"  reentry M={len(d)} normal-input means {Xn0.mean(0).round(4)} sd {Xn0.std(0).round(4)}")
    X = np.hstack([Xu0, Xn0])
    sp = ac.Product([ac.Uniform(lb, ub), ac.Gaussian(np.zeros(4), sd ** 2)])
    s = ac.active_subspace(X=X, f=pst, space=sp, estimator="qphd", n=2)
    Z = np.hstack([2 * (Xu0 - lb) / (ub - lb) - 1, Xn0 / sd])
    sw = ac.active_subspace(X=Z, f=pst, space=ac.Gaussian(np.zeros(7), np.ones(7)), estimator="qphd", n=2)
    l2, l2w = s.eigenvalues[1], sw.eigenvalues[1]
    print(f"  p_st QPHD lambda2: Product {l2:.4g}, hand-normalized Gaussian workaround {l2w:.4g} (report: 4.97e4)")
    rep(abs(l2 / 1.66e4 - 1) <= 0.01, "V1-06 reentry QPHD p_st lambda2", f"{l2:.4g} vs 1.66e4 +-1%")
    rep(abs(l2w / 4.97e4 - 1) <= 0.01, "V1-06 reentry workaround lambda2", f"{l2w:.4g} vs 4.97e4 +-1% (informational)")


# ---------------- V1-10 NREL
def v110_nrel():
    d = wind(); Xw = d[:, 1:6]; Fw = d[:, 6:]
    names = ["Vhub", "Hs", "WaveDir", "Tp", "PlatformDir"]
    try:
        ac.local_linear_gradients(X=Xw, f=Fw[:, 0], space=ac.Uniform.reference(5, names=names), centers=np.arange(5))
        rep(False, "V1-10 NREL out_of_space", "no exception raised")
        return
    except ac.InputError as e:
        print("  message:", str(e))
        print("  details:", e.details)
        rep(e.code == "input.out_of_space", "V1-10 code", f"{e.code} vs input.out_of_space")
        rep(e.details.get("n_outside") == 724, "V1-10 n_outside", f"{e.details.get('n_outside')} vs 724")
        rep("np.float64" not in str(e), "V1-10 no np.float64 in message", f"{'np.float64' not in str(e)}")
        rep(e.details.get("name") == "PlatformDir", "V1-10 name", f"{e.details.get('name')!r} vs 'PlatformDir'")
        keys = {"n_outside", "row", "column", "name", "value", "bound", "side"}
        rep(keys <= set(e.details), "V1-10 details keys", f"missing {sorted(keys - set(e.details))}")


# ---------------- V1-12
def v112():
    rep(ac.Uniform.reference(2, names=["a", "b"]) == ac.Uniform([-1, -1], [1, 1], names=("a", "b")),
        "V1-12 Uniform.reference", "equal")
    rep(ac.Gaussian.reference(2, names=["a", "b"]) == ac.Gaussian([0, 0], [1, 1], names=("a", "b")),
        "V1-12 Gaussian.reference", "equal")


guard("V1-06 hydrology", v106_hydrology)
guard("V1-06 reentry", v106_reentry)
guard("V1-10", v110_nrel)
guard("V1-12", v112)
print(f"{res['PASS']} passed, {res['FAIL']} failed, {res['SKIP']} skipped  ({time.time()-T0:.1f}s)")
