# Item 5: tensor quadrature C (Ebola), memory and subspace_from_basis
import numpy as np, time, sys, resource, tracemalloc, warnings
from scipy.special import roots_legendre
import active_subspaces as ac
lb = np.array([.1, .1, .05, .41, .0276, .081, .25, .0833]); ub = np.array([.4, .4, .2, 1, .1702, .21, .5, .7])
def R0(x):
    b1,b2,b3,r1,g1,g2,om,p = x.T
    return (b1 + b2*r1*g1/om + b3*p/g2)/(g1 + p)
def R0_grad(x):
    b1,b2,b3,r1,g1,g2,om,p = x.T; D=g1+p; R=R0(x)
    return np.column_stack([1/D, r1*g1/om/D, p/g2/D, b2*g1/om/D, (b2*r1/om - R)/D, -b3*p/g2**2/D, -b2*r1*g1/om**2/D, (b3/g2 - R)/D])
k = int(sys.argv[1])
t, w = roots_legendre(k); w = w / 2
T = np.stack(np.meshgrid(*([t] * 8), indexing='ij'), -1).reshape(-1, 8)
W = np.prod(np.stack(np.meshgrid(*([w] * 8), indexing='ij'), -1).reshape(-1, 8), 1)
space = ac.Uniform(lower=lb, upper=ub)
X = space.from_normalized(T); del T
G = R0_grad(X)
print(f"k={k}: N={len(W)}, sum(w)-1={W.sum()-1:.1e}, G bytes {G.nbytes/1e9:.2f} GB")
tracemalloc.start()
t0 = time.time()
try:
    sub = ac.active_subspace(space=space, grads=G, quad_weights=W)
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  active_subspace: {time.time()-t0:.1f}s, numpy peak inside call {peak/1e9:.2f} GB ({peak/G.nbytes:.1f} x G), lambda1={sub.eigenvalues[0]:.4f}")
except Exception as e:
    print("  ->", type(e).__name__, getattr(e, "code", None), str(e)[:200])
tracemalloc.stop()
# hand chunked C in normalized coordinates (what the notebook writes for 8^8)
Gn = G * (ub - lb) / 2
C = (Gn * W[:, None]).T @ Gn
lam, V = np.linalg.eigh(C)
print(f"  hand C: lambda1={lam[-1]:.4f}")
b = ac.subspace_from_basis(space.directions_to_physical(V[:, ::-1][:, :2]) if False else (V[:, ::-1][:, :2] / ((ub-lb)/2)[:, None]), space=space)
print("  subspace_from_basis eigenvalues:", b.eigenvalues[:3], "W1 vs hand |cos|:", np.abs(np.sum(b.W1 * V[:, ::-1][:, :2], 0)).round(12))
import matplotlib; matplotlib.use("Agg")
try:
    ac.plotting.plot_eigenvalues(b)
except Exception as e:
    print("  plot_eigenvalues(basis) ->", type(e).__name__, e.code, "|", str(e)[:120])
print(f"  maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
