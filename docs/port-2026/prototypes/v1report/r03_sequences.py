# Item 3/10: orientation across sequences of subspaces (one QoI per call)
import numpy as np, warnings, time
import active_subspaces as ac
from common import *
warnings.simplefilter("ignore", ac.ValidityWarning)
def seq(Xp, F, space, cols):
    v1 = []; cv = []
    for i in cols:
        s = ac.active_subspace(X=Xp, f=F[:, i], space=space, estimator="ols", n=1)
        w = s.W1[:, 0]; v1.append(w)
        A = np.hstack([np.ones((len(Xp), 1)), space.to_normalized(Xp)])
        a = np.linalg.lstsq(A, F[:, i], rcond=None)[0][1:]
        cv.append(w * np.sign(w @ a))
    v1 = np.array(v1); cv = np.array(cv)
    disagree = int((np.sum(v1 * cv, 1) < 0).sum())
    jv = int((np.sum(v1[1:] * v1[:-1], 1) < 0).sum()); jc = int((np.sum(cv[1:] * cv[:-1], 1) < 0).sum())
    return disagree, jv, jc, len(cols)
for rate in ("0.25C", "1C", "4C"):
    X, V, C = battery(rate)
    sp = ac.Uniform(-np.ones(19), np.ones(19))
    print(rate, "capacity: (v1 != a_hat sign, v1 reversals between neighbours, a_hat reversals, N) =", seq(X, C, sp, range(50)))
    print(rate, "voltage  :", seq(X, V, sp, range(1, 49)))
h, Xs, B, out = stomatal()
sp = ac.Uniform(B[:, 0], B[:, 1], names=h)
for k, (hh, F) in out.items():
    print("Stomatal", k, F.shape, seq(Xs[1:], F[1:], sp, range(F.shape[1])))
# vector f
X, V, C = battery("0.25C")
try:
    ac.active_subspace(X=X, f=C, space=ac.Uniform.reference(19), estimator="ols", n=1)
except Exception as e:
    print("f (3600, 50) ->", type(e).__name__, e.code, "|", str(e)[:120])
t = time.time()
for i in range(50): ac.active_subspace(X=X, f=C[:, i], space=ac.Uniform.reference(19), estimator="ols", n=1)
print("50 OLS calls (M=3600, m=19): %.2fs" % (time.time() - t))
Xh, Fh, Gh = hiv()
print("HIV shapes", Xh.shape, Fh.shape, Gh.shape)
t = time.time()
nt = Gh.shape[1] // 27
subs = [ac.active_subspace(grads=Gh[:, 27 * i:27 * (i + 1)], space=ac.Uniform.reference(27)) for i in range(nt)]
print("HIV %d AS calls: %.2fs; n chosen:" % (nt, time.time() - t), [s.n for s in subs])
W = np.array([s.W1[:, 0] for s in subs])
print("HIV v1 w1 reversals between consecutive times:", int((np.sum(W[1:] * W[:-1], 1) < 0).sum()), "/%d" % (nt - 1))
