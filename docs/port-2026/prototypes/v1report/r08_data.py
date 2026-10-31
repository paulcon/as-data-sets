# Item 8: non-finite f in plots (Subsurface), data outside the declared space (NREL_Wind)
import numpy as np, warnings, time
import matplotlib; matplotlib.use("Agg")
import active_subspaces as ac
from common import *
X, fs, fu = subsurface()
sp = ac.Gaussian.reference(100)
sub = ac.active_subspace(X=X, f=fu, space=sp, estimator="ols", n=1)
print("Subsurface Hortonian zeros:", int((fu == 0).sum()), "of", len(fu))
with np.errstate(divide="ignore"):
    lf = np.log10(fu)
try:
    ac.plotting.plot_sufficient_summary(sub, X=X, f=lf)
except Exception as e:
    print("plot_sufficient_summary(f=log10 f) ->", type(e).__name__, e.code, "|", str(e)[:140])
m = fu > 0
fig, ax = ac.plotting.plot_sufficient_summary(sub, X=X[m], f=np.log10(fu[m]))
print("masked idiom X[f>0], log10(f[f>0]) works:", len(ax.collections[0].get_offsets()))
d = wind(); Xw = d[:, 1:6]; Fw = d[:, 6:]
out = (np.abs(Xw) > 1).any(1)
print("NREL rows outside [-1,1]^5:", int(out.sum()), "per column", (np.abs(Xw) > 1).sum(0))
try:
    ac.local_linear_gradients(X=Xw, f=Fw[:, 0], space=ac.Uniform.reference(5), centers=np.arange(5))
except Exception as e:
    print("local_linear_gradients(space=Uniform.reference(5)) ->", type(e).__name__, e.code, "|", str(e)[:140])
lo = np.minimum(Xw.min(0), -1); hi = np.maximum(Xw.max(0), 1)
sp2 = ac.Uniform(lo, hi, names=["Vhub", "Hs", "WaveDir", "Tp", "PlatformDir"])
rng = np.random.default_rng(0)
t = time.time()
g = ac.local_linear_gradients(X=Xw, f=Fw[:, 0], space=sp2, p=60, centers=rng.choice(len(Xw), 2000, replace=False))
print(f"local_linear_gradients 2000 centers, M=50000: {time.time()-t:.1f}s")
t = time.time()
g = ac.local_linear_gradients(X=Xw[:5000], f=Fw[:5000, 0], space=sp2, p=60)
print(f"local_linear_gradients all 5000 centers, M=5000: {time.time()-t:.1f}s")
s = ac.active_subspace(grads=g, nboot=100, rng=1)
print("wind EVG n:", s.n)
# Gaussian identity space accepts anything (the other stopgap)
s3 = ac.active_subspace(X=Xw, f=Fw[:, 0], space=ac.Gaussian(np.zeros(5), np.ones(5)), estimator="ols", n=1)
print("Gaussian identity space accepts out-of-box rows: True")
