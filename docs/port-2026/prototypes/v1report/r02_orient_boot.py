# Items 2 and 3: eigenvector bootstrap replicates; data-defined orientation
import numpy as np, warnings, dataclasses, time
import active_subspaces as ac
from common import *
X, V, C = battery("0.25C")
space = ac.Uniform(-np.ones(19), np.ones(19), names=BATTERY_NAMES)
flips = 0; prev = None; jumps_v1 = 0; jumps_cov = 0
t = time.time()
with warnings.catch_warnings(record=True) as wr:
    warnings.simplefilter("always")
    for i in range(50):
        s = ac.active_subspace(X=X, f=C[:, i], space=space, estimator="ols", n=1)
        w = s.W1[:, 0]
        sc = np.sign(np.cov(X @ w, C[:, i])[0, 1])
        flips += sc < 0
        if prev is not None:
            jumps_v1 += (w @ prev[0]) < 0
            jumps_cov += ((sc * w) @ prev[1]) < 0
        prev = (w, sc * w)
el = time.time() - t
print(f"0.25C capacity, 50 OLS fits: v1 sign disagrees with sign(cov(Xw,f)) at {flips}/50; "
      f"consecutive-voltage sign reversals: v1 {jumps_v1}/49, cov-oriented {jumps_cov}/49; "
      f"{len(wr)} warnings {set(codes(wr))}; {el:.2f}s")
s = ac.active_subspace(X=X, f=C[:, 2], space=space, estimator="ols", n=1, nboot=100, rng=0)
print("SubspaceResult fields:", [f.name for f in dataclasses.fields(s)])
print("bootstrap arrays:", s.bootstrap_eigenvalues.shape, s.bootstrap_distances.shape, s.bootstrap_dets.shape)
print("rank-one OLS, bootstrap_distances[:,0] (sin angle w, w^(b)): median %.4f max %.4f" %
      (np.median(s.bootstrap_distances[:, 0]), s.bootstrap_distances[:, 0].max()))
# Undocumented workaround: dataclasses.replace with a flipped column
W = s.eigenvectors.copy(); W[:, 0] *= -1
try:
    s2 = dataclasses.replace(s, eigenvectors=W)
    print("dataclasses.replace(sub, eigenvectors=-W) accepted: W1[0] %.3f -> %.3f; provenance kept: %s" %
          (s.W1[0, 0], s2.W1[0, 0], s2.provenance == s.provenance))
except Exception as e:
    print("dataclasses.replace failed:", type(e).__name__, getattr(e, "code", None), e)
# plot uses sub.eigenvectors: sufficient summary is mirrored when the hand-oriented w differs
import matplotlib; matplotlib.use("Agg")
fig, ax = ac.plotting.plot_sufficient_summary(s, X=X, f=C[:, 2])
xs = ax.collections[0].get_offsets()[:, 0]
print("plot_sufficient_summary x == X @ v1 w:", np.allclose(xs, X @ s.W1[:, 0]))
# Hand bootstrap of the sign-aligned weights (what the notebook must write)
rng = np.random.default_rng(0); B = 100; Wb = np.empty((B, 19))
w0 = s.W1[:, 0] * np.sign(np.cov(X @ s.W1[:, 0], C[:, 2])[0, 1])
for b in range(B):
    i = rng.integers(0, len(X), len(X))
    sb = ac.active_subspace(X=X[i], f=C[i, 2], space=space, estimator="ols", n=1) if False else None
    A = np.hstack([np.ones((len(X), 1)), X[i]]); u = np.linalg.lstsq(A, C[i, 2], rcond=None)[0][1:]
    Wb[b] = u / np.linalg.norm(u)
print("hand bootstrap SE of sign-aligned w (idx 2): max %.4f" % Wb.std(0, ddof=1).max())
