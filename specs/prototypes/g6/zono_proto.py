import time, numpy as np, active_subspaces as ac
rng = np.random.default_rng(0)
def randomized(W, nmax, rng):
    m, n = W.shape
    target = ac.zonotope_vertex_count(m, n)
    S = set(); count = 0
    batch = 1000
    while len(S) < target and count < nmax:
        Z = rng.standard_normal((batch, n))
        sg = np.sign(Z @ W.T).astype(np.int8)
        for s in sg:
            S.add(s.tobytes()); S.add((-s).tobytes())
        count += batch
    return np.array([np.frombuffer(b, dtype=np.int8) for b in S]) @ W, count
for m, n in [(5,2),(10,2),(10,3),(10,4),(20,3)]:
    A = rng.standard_normal((m, m)); A = A @ A.T
    space = ac.Uniform(lower=-np.ones(m), upper=np.ones(m))
    X = rng.uniform(-1, 1, (200, m)); G = X @ A  # grads of x^T A x /2
    sub = ac.active_subspace(space=space, X=X, grads=G, n=n)
    t=time.time(); dom = ac.active_domain(sub); te=time.time()-t
    Vex = np.asarray(dom.vertices); W1 = np.asarray(sub.W1)
    t=time.time(); Vr, cnt = randomized(W1, 10**6, rng); tr=time.time()-t
    key = lambda V: set(map(tuple, np.round(V, 8)))
    ex, rd = key(Vex), key(Vr)
    print(f"m={m} n={n} count={ac.zonotope_vertex_count(m,n)} exact={len(ex)} ({te:.3f}s) rand={len(rd)} after {cnt} samples ({tr:.2f}s) rand_subset_of_exact={rd<=ex}")
