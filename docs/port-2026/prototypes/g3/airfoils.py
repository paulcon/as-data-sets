import sys, time, warnings, numpy as np, active_subspaces as ac
np.set_printoptions(precision=4, suppress=False, linewidth=160)
fn, m, b = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
d = np.genfromtxt(fn, delimiter=",", names=True)
names = d.dtype.names
d = np.loadtxt(fn, delimiter=",", skiprows=1)
X, fl, fd = d[:, 1:m+1], d[:, m+1], d[:, m+2]
Gl, Gd = d[:, m+3:2*m+3], d[:, 2*m+3:]
space = ac.Uniform(-b*np.ones(m), b*np.ones(m), names=list(names[1:m+1]))
for nm, f, G in [("lift", fl, Gl), ("drag", fd, Gd)]:
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = ac.active_subspace(grads=G, space=space, nboot=200, rng=0, qoi=nm)
    el = time.time() - t
    # legacy: eig of G^T G / M with UNSCALED grads, sign: first component positive
    C = G.T @ G / len(G); e, W = np.linalg.eigh(C); e, W = e[::-1], W[:, ::-1]
    W = W * np.sign(W[0]); 
    print(nm, f"v1 time {el:.2f}s, n chosen {s.n}, warnings {[str(x.message)[:80] for x in w]}")
    print("  v1 eig[:6]      ", s.eigenvalues[:6])
    print("  legacy eig[:6]  ", e[:6], " ratio v1/legacy", s.eigenvalues[0]/e[0])
    print("  gap ratios lambda_k/lambda_k+1", (s.eigenvalues[:5]/s.eigenvalues[1:6]))
    for k in range(2):
        v1 = s.W[:, k]; lg = W[:, k]
        print(f"  w{k+1} v1 vs legacy(first>0): sign {np.sign(v1@lg):+.0f}  max|diff| after sign {np.abs(v1-np.sign(v1@lg)*lg).max():.2e}")
    print("  w1 v1:", s.W[:, 0])
    print("  |w1| first half vs second half:", np.abs(s.W[:m//2, 0]).sum(), np.abs(s.W[m//2:, 0]).sum())
    ss = ac.with_dimension(s, 2)
    for n in (1, 2):
        sk = ac.with_dimension(s, n)
        sur = ac.fit_surrogate(X=X, f=f, subspace=sk, kind="poly", degree=2)
        print(f"  poly2 n={n} loo R2 {sur.loo_r_squared:.4f}")
    if m == 50:
        t=time.time()
        for n in range(1, 4):
            sk = ac.with_dimension(s, n)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sur = ac.fit_surrogate(X=X, f=f, subspace=sk, kind="rbf", degree=2)
            print(f"  rbf2 n={n} loo R2 {sur.loo_r_squared:.4f}  ({time.time()-t:.1f}s)")
        dom = ac.active_domain(ss); print("  domain", dom.kind, dom.vertices.shape)
