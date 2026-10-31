"""Acceptance checks, branch as-port-subspaces: V1-02, V1-04, V1-05, V1-07, V1-08, V1-11 (real data)."""
import os, time, tracemalloc, warnings
import numpy as np
from scipy.special import roots_legendre
import active_subspaces as ac
from common import *

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
        rep(False, name, f"{type(e).__name__}: {str(e)[:150]}")


def ols_hat(space, X, f):
    A = np.hstack([np.ones((len(X), 1)), space.to_normalized(X)])
    return np.linalg.lstsq(A, f, rcond=None)[0][1:]


def rev(W):
    return int((np.sum(W[1:] * W[:-1], 1) < 0).sum())


# ---------------- V1-04
def v104_lithium():
    X, V, C = battery("0.25C")
    sp = ac.Uniform(-np.ones(19), np.ones(19), names=BATTERY_NAMES)
    for lab, F, cols in (("capacity", C, range(50)), ("voltage", V, range(1, 49))):
        W = []; bad = 0
        for i in cols:
            s = ac.active_subspace(X=X, f=F[:, i], space=sp, estimator="ols", n=1)
            w = s.W[:, 0]; W.append(w)
            bad += (w @ ols_hat(sp, X, F[:, i])) <= 0
        r = rev(np.array(W))
        rep(bad == 0 and r == 0, f"V1-04 Lithium 0.25C {lab}", f"w.uhat<=0 in {bad}/{len(W)}, reversals {r}/{len(W)-1} vs 0, 0")
guard("V1-04 Lithium", v104_lithium)


def v104_stomatal():
    h, Xs, B, out = stomatal()
    sp = ac.Uniform(B[:, 0], B[:, 1], names=h)
    tot = 0; nser = 0
    for k, (hh, F) in out.items():
        W = np.array([ac.active_subspace(X=Xs[1:], f=F[1:, i], space=sp, estimator="ols", n=1).W[:, 0] for i in range(F.shape[1])])
        r = rev(W); tot += r; nser += 1
        if r: print(f"   reversals in {k}: {r}")
    rep(tot == 0 and nser == 6, "V1-04 Stomatal", f"reversals {tot} over {nser} series vs 0 in all six")
guard("V1-04 Stomatal", v104_stomatal)

W_HY = np.array([0.6506, 0.5565, -0.0002, 0.3685, -0.3566, -0.0196, 0.0607])


def v104_hyshot():
    X, f = hyshot30(); sp = ac.Uniform(HYSHOT_XL, HYSHOT_XU)
    s = ac.active_subspace(X=X, f=f, space=sp, estimator="ols", n=1)
    d = np.abs(s.W[:, 0] - W_HY).max()
    rep(d <= 1e-4 + 1e-12, "V1-04 HyShotII w(0.30)", f"max|diff|={d:.2e} vs 1e-4; w={np.round(s.W[:,0],4)}")
guard("V1-04 HyShotII", v104_hyshot)


def v104_hort():
    X, fs, fu = subsurface()
    s = ac.active_subspace(X=X, f=fu, space=ac.Gaussian.reference(100), estimator="ols", n=1)
    w1 = s.W[0, 0]
    rep(abs(w1 + 0.627) <= 1e-3, "V1-04 Subsurface Hortonian w1", f"{w1:.4f} vs -0.627 +-1e-3")
guard("V1-04 Hortonian", v104_hort)

# ---------------- V1-02
def v102_align(label, X, f, sp):
    kw = dict(X=X, f=f, space=sp, estimator="ols", n=1)
    s = ac.active_subspace(**kw, nboot=100, rng=0, bootstrap_vectors=1)
    Wb = s.bootstrap_eigenvectors
    w = s.W[:, 0]; ip = Wb[:, :, 0] @ w
    d = np.sqrt(np.maximum(1 - (w @ Wb[:, :, 0].T) ** 2, 0))
    err = np.abs(s.bootstrap_distances[:, 0] - d).max()
    rep(Wb.shape == (100, len(w), 1) and (ip > 0).all() and err <= 1e-12, f"V1-02 {label}",
        f"shape {Wb.shape}, min inner {ip.min():.4f} > 0, max dist err {err:.1e} vs 1e-12")
    return s


def v102():
    X, V, C = battery("0.25C")
    sp = ac.Uniform(-np.ones(19), np.ones(19), names=BATTERY_NAMES)
    s = v102_align("Lithium 0.25C idx 2", X, C[:, 2], sp)
    se = s.bootstrap_eigenvectors[:, :, 0].std(0, ddof=1).max()
    rep(0.0012 <= se <= 0.0028, "V1-02 Lithium max component SE (B=100)", f"{se:.4f} vs [0.0012, 0.0028] (hand 0.0019)")
    kw = dict(X=X, f=C[:, 2], space=sp, estimator="ols", n=1, nboot=20, rng=0)
    a = ac.active_subspace(**kw); b = ac.active_subspace(**kw, bootstrap_vectors=0)
    same = all(np.array_equal(getattr(a, k), getattr(b, k)) for k in
               ("eigenvalues", "eigenvectors", "bootstrap_eigenvalues", "bootstrap_distances", "bootstrap_dets")) \
        and a.bootstrap_eigenvectors is None and b.bootstrap_eigenvectors is None
    try:
        same = same and (a == b)
    except Exception:
        pass
    rep(same, "V1-02 bootstrap_vectors=0 equals default", f"arrays equal and bootstrap_eigenvectors None: {same}")
guard("V1-02 Lithium", v102)


def v102_hy():
    X, f = hyshot30()
    v102_align("HyShotII phi=0.30", X, f, ac.Uniform(HYSHOT_XL, HYSHOT_XU))
guard("V1-02 HyShotII", v102_hy)

# ---------------- V1-05
def v105_hiv():
    Xh, Fh, Gh = hiv(); sp = ac.Uniform.reference(27)
    nt = Gh.shape[1] // 27
    subs = [ac.active_subspace(grads=Gh[:, 27 * i:27 * (i + 1)], space=sp) for i in range(nt)]
    raw = rev(np.array([s.W[:, 0] for s in subs]))
    out = [subs[0]]
    for s in subs[1:]:
        out.append(ac.orient(s, by="reference", reference=out[-1]))
    r = rev(np.array([s.W[:, 0] for s in out]))
    rep(r == 0, "V1-05 HIV chained orient", f"reversals {r}/{nt - 1} vs 0 (unoriented {raw}/{nt - 1})")
guard("V1-05 HIV", v105_hiv)


def v105_phys():
    X, f = hyshot30(); sp = ac.Uniform(HYSHOT_XL, HYSHOT_XU)
    s = ac.active_subspace(X=X, f=f, space=sp, estimator="ols", n=1)
    ok = True; msg = []
    for sgn in (1, -1):
        o = ac.orient(s, by="reference", reference=sgn * s.W[:, :1])
        ph = o.physical_directions()[:, 0]
        v = o.W[:, 0] / ((HYSHOT_XU - HYSHOT_XL) / 2)
        ip = ph @ v
        ok &= ip > 0 and np.sign(o.W[:, 0] @ s.W[:, 0]) == sgn
        msg.append(f"{ip:.3g}")
    rep(bool(ok), "V1-05 physical_directions follows orient", f"inner products (+ref, -ref) = {msg} > 0")
    for by, kw in (("covariance", dict(X=X, f=f)),):
        o = ac.orient(s, by=by, **kw)
        rep(o.W[:, 0] @ ols_hat(sp, X, f) > 0, "V1-05 HyShotII orient covariance", "w.uhat > 0")
guard("V1-05 physical", v105_phys)
skip("V1-05 MCMC y2 mean", "no MCMC data path in r02/r03/common")

# ---------------- V1-07 / V1-08
names8 = ['b1', 'b2', 'b3', 'rho1', 'g1', 'g2', 'om', 'psi']
LB = {"Liberia": np.array([.1, .1, .05, .41, .0276, .081, .25, .0833]), "Sierra Leone": np.array([.1, .1, .05, .41, .0275, .1236, .25, .0833])}
UB = {"Liberia": np.array([.4, .4, .2, 1, .1702, .21, .5, .7]), "Sierra Leone": np.array([.4, .4, .2, 1, .1569, .384, .5, .7])}
R04 = {"Liberia": [0.228, 0.006, 0.291, 0.003, 0.099, 0.29, 0.002, 1.0],
       "Sierra Leone": [0.184, 0.005, 0.082, 0.002, 0.046, 0.133, 0.002, 1.0]}


def R0(x):
    b1, b2, b3, r1, g1, g2, om, p = x.T
    return (b1 + b2 * r1 * g1 / om + b3 * p / g2) / (g1 + p)


def R0_grad(x):
    b1, b2, b3, r1, g1, g2, om, p = x.T; D = g1 + p; R = R0(x)
    return np.column_stack([1 / D, r1 * g1 / om / D, p / g2 / D, b2 * g1 / om / D, (b2 * r1 / om - R) / D,
                            -b3 * p / g2 ** 2 / D, -b2 * r1 * g1 / om ** 2 / D, (b3 / g2 - R) / D])


def rule(k):
    t, w = roots_legendre(k); w = w / 2
    T = np.stack(np.meshgrid(*([t] * 8), indexing='ij'), -1).reshape(-1, 8)
    Wq = np.prod(np.stack(np.meshgrid(*([w] * 8), indexing='ij'), -1).reshape(-1, 8), 1)
    return T, Wq


def v107_108():
    T, Wq = rule(6)
    for nm in ("Liberia", "Sierra Leone"):
        lb, ub = LB[nm], UB[nm]
        sp = ac.Uniform(lb, ub, names=names8)
        G = R0_grad(sp.from_normalized(T))
        sub = ac.active_subspace(space=sp, grads=G, quad_weights=Wq)
        n = 2
        a = ac.activity_scores(sub, n)
        ref = (sub.eigenvalues[:n] * sub.W[:, :n] ** 2).sum(1)
        e1 = np.abs(a - ref).max()
        ratio = a / a.max()
        e2 = np.abs(ratio - np.array(R04[nm])).max()
        rep(e1 <= 1e-14 * max(1, ref.max()) and int(np.argmax(a)) == 7 and e2 <= 1e-3 + 1e-12,
            f"V1-07 {nm} activity scores",
            f"max|diff formula|={e1:.1e} vs 1e-14, argmax={names8[int(np.argmax(a))]}, max|ratio - r04.out|={e2:.1e} vs 1e-3")
        Gn = G * (ub - lb) / 2
        C = (Gn * Wq[:, None]).T @ Gn
        h = ac.subspace_from_matrix(C, space=sp)
        dl = abs(h.eigenvalues[0] - sub.eigenvalues[0]); dw = np.abs(np.abs(h.W[:, 0] @ sub.W[:, 0]) - 1)
        wd = np.abs(h.W[:, 0] * np.sign(h.W[:, 0] @ sub.W[:, 0]) - sub.W[:, 0]).max()
        rep(dl <= 1e-10 and wd <= 1e-10, f"V1-08 {nm} 6^8 subspace_from_matrix",
            f"lam1 {h.eigenvalues[0]:.10f} vs {sub.eigenvalues[0]:.10f} diff {dl:.1e}, w1 max diff {wd:.1e} (both vs 1e-10)")
guard("V1-07/08", v107_108)


def v108_8():
    mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    if mem < 16:
        return skip("V1-08 8^8 rule", f"only {mem:.0f} GB RAM")
    t0 = time.time()
    T, Wq = rule(8); lb, ub = LB["Liberia"], UB["Liberia"]
    sp = ac.Uniform(lb, ub, names=names8)
    G = R0_grad(sp.from_normalized(T)); del T
    Gn = G * (ub - lb) / 2
    C = np.zeros((8, 8))
    for i in range(0, len(Wq), 2_000_000):
        s = slice(i, i + 2_000_000); C += (Gn[s] * Wq[s, None]).T @ Gn[s]
    del Gn
    t1 = time.time()
    h = ac.subspace_from_matrix(C, space=sp)
    s2 = ac.active_subspace(space=sp, grads=G, quad_weights=Wq)
    el = time.time() - t0
    dl = abs(h.eigenvalues[0] - s2.eigenvalues[0])
    wd = np.abs(h.W[:, 0] * np.sign(h.W[:, 0] @ s2.W[:, 0]) - s2.W[:, 0]).max()
    rep(el < 180 and dl <= 1e-10 and wd <= 1e-10 and abs(h.eigenvalues[0] - 1.0217) < 5e-5, "V1-08 8^8 rule",
        f"lam1 {h.eigenvalues[0]:.4f} (r05_8: 1.0217), diff to quad_weights {dl:.1e}, w1 diff {wd:.1e}, total {el:.0f}s vs <180s")
guard("V1-08 8^8", v108_8)


def v108_mem():
    t0 = time.time()
    T, Wq = rule(7); lb, ub = LB["Liberia"], UB["Liberia"]
    sp = ac.Uniform(lb, ub, names=names8)
    G = R0_grad(sp.from_normalized(T)); del T
    tracemalloc.start()
    sub = ac.active_subspace(space=sp, grads=G, quad_weights=Wq)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    r = peak / G.nbytes
    rep(r <= 2.1, "V1-08 peak NumPy memory 7^8", f"{peak/1e9:.2f} GB = {r:.2f} x G ({G.nbytes/1e9:.2f} GB) vs <=2.1 (was 3.1); lam1={sub.eigenvalues[0]:.4f}; {time.time()-t0:.0f}s")
guard("V1-08 memory", v108_mem)

# ---------------- V1-11
def v111():
    X, fs, fu = subsurface()
    t = time.time()
    q = ac.active_subspace(X=X, f=fs, space=ac.Gaussian.reference(100), estimator="qphd")
    el = time.time() - t
    lam = np.abs(q.eigenvalues[:4])
    tgt = np.array([12252.5, 630.6, 424.3, 363.6])
    err = np.abs(lam.round(1) - tgt).max()
    rep(err < 1e-9, "V1-11 QPHD eigenvalues", f"{np.round(lam, 4)} vs {tgt} (r15.out has 1 decimal only: compared at 0.1; 1e-8 relative not testable)")
    rep(el <= 0.5 * 40.8, "V1-11 QPHD time", f"{el:.1f}s vs <= 20.4s (0.5 x gelsd 40.8s)")
guard("V1-11", v111)

print(f"{res['PASS']} passed, {res['FAIL']} failed, {res['SKIP']} skipped  (total {time.time()-T0:.0f}s)")
