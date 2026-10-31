# Item 1: mixed Uniform x Gaussian space (Hydrology grass: 18 uniform + log z0m Gaussian)
import numpy as np, warnings, inspect
import active_subspaces as ac
from common import *
X, F = hydrology_grass()
Xt = X.copy(); Xt[:, 16] = np.log(X[:, 16])            # log z0m ~ N(-5.11, 0.52^2)
lo = np.insert(YL, 16, -np.inf); hi = np.insert(YU, 16, np.inf)
print("public space classes:", [n for n in dir(ac.spaces) if n[0].isupper()])
# (a) a Uniform with infinite bounds for the Gaussian column
try:
    ac.Uniform(lower=lo, upper=hi)
except Exception as e:
    print("(a) Uniform(+-inf) ->", type(e).__name__, e.code, "|", str(e)[:110])
# (b) Uniform.reference on hand-normalized data
Z = Xt.copy(); uni = [j for j in range(19) if j != 16]
Z[:, uni] = 2 * (X[:, uni] - YL) / (YU - YL) - 1; Z[:, 16] = (Xt[:, 16] + 5.11) / .52
try:
    ac.active_subspace(X=Z, f=F[:, 0], space=ac.Uniform.reference(19), estimator="ols", n=1)
except Exception as e:
    print("(b) Uniform.reference(19), normalized data ->", type(e).__name__, e.code, "|", str(e)[:110])
# (c) workaround: Gaussian identity space on hand-normalized data
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    s = ac.active_subspace(X=Z, f=F[:, 0], space=ac.Gaussian(np.zeros(19), np.ones(19), names=ORDER_G), estimator="ols", n=1)
print("(c) Gaussian identity workaround: lambda1 = %.6g, warnings %s" % (s.eigenvalues[0], codes(w)))
print("    space.kind =", s.space.kind, "-> sample() would draw N(0,1) for the 18 uniform inputs")
# (d) same with the 18 uniform columns only, physical Uniform: correct space but drops z0m
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    s2 = ac.active_subspace(X=X[:, uni], f=F[:, 0], space=ac.Uniform(YL, YU), estimator="ols", n=1)
print("(d) Uniform on 18 columns: warnings", codes(w))
# (e) QPHD needs Sigma: wrong Sigma under the Gaussian workaround
print("reference_covariance Gaussian identity diag[:3] =", s.space.reference_covariance.diagonal()[:3], "(true 1/3 for uniform cols)")
# Stomatal: discrete inputs
h, Xs, B, out = stomatal()
for nm in ("wj1", "wj2"):
    if nm in h:
        j = h.index(nm); print(f"Stomatal {nm}: values {np.unique(Xs[1:, j])}, bounds {B[j]}")
sp = ac.Uniform(B[:, 0], B[:, 1], names=h)
print("Stomatal Uniform(bounds).contains(all rows):", bool(sp.contains(Xs).all()))
