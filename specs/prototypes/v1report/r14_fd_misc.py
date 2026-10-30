# Item 14 (FD for many outputs) and misc API checks (names, reference spaces)
import numpy as np, inspect, warnings
import active_subspaces as ac
from common import *
sp = ac.Uniform(np.full(3, 0.5), np.full(3, 1.5), names=["a", "b", "c"])
calls = {"n": 0, "rows": 0}
def traj(X):                       # stand-in for one ODE solve per row returning 21 output times
    calls["n"] += 1; calls["rows"] += len(X)
    t = np.linspace(0, 1, 21)
    return np.exp(-np.outer(X @ np.array([1.0, 0.5, 0.1]), t))     # (k, 21)
X = sp.sample(4, rng=0)
try:
    ac.finite_difference_gradients(traj, X=X, space=sp)
except Exception as e:
    print("vector-valued model (k,21) ->", type(e).__name__, e.code, "|", str(e)[:120])
calls.update(n=0, rows=0)
G = [ac.finite_difference_gradients(lambda Z, j=j: traj(Z)[:, j], X=X, space=sp) for j in range(21)]
print(f"21 scalar FD calls: {calls['n']} model calls, {calls['rows']} trajectory rows (= 21 x M(m+1) = {21*4*4})")
print("finite_difference_gradients signature:", inspect.signature(ac.finite_difference_gradients))
print("GradientResult has steps/central_mask/values:", hasattr(G[0], "steps"), G[0].steps.shape)
# names
for cls in (ac.Uniform, ac.Gaussian):
    print(cls.__name__ + ".reference signature:", inspect.signature(cls.reference))
try:
    ac.Uniform.reference(3, names=["a", "b", "c"])
except TypeError as e:
    print("Uniform.reference(3, names=...) -> TypeError", str(e)[:80])
hdr = open(f"{R}/Lithium_Battery/0.25C.txt").readline().strip().split(",")[1:20]
s = ac.Uniform(-np.ones(19), np.ones(19), names=hdr)
print("LaTeX header names accepted:", s.names[:3])
print("Uniform signature:", inspect.signature(ac.Uniform))
print("Gaussian signature:", inspect.signature(ac.Gaussian))
