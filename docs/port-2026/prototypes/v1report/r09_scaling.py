# Item 9: eigenvalue conventions (scaled outputs, physical inputs) and item 11 (dimension rules, R2)
import numpy as np, warnings
import active_subspaces as ac
from common import *
D = np.loadtxt("dev/LM104X.txt", delimiter=",", skiprows=1)
X, F = D[:, 1:106], D[:, 106:109]; G = {'lift': D[:, 109:214], 'drag': D[:, 214:319], 'ea': D[:, 319:424]}
sp = ac.Uniform(np.full(105, -.3), np.full(105, .3))
for j, k in enumerate(G):
    f = F[:, j]; c = 2.0 / np.ptp(f)                      # min-max scaling of f to [-1, 1]
    s = ac.active_subspace(space=sp, grads=c * G[k])
    print(f"LM104X {k}: min-max scaled outputs via grads=c*G: lambda1={s.eigenvalues[0]:.3g} lambda2={s.eigenvalues[1]:.2g}")
Xn, fl, fd, Gl, Gd = airfoil("NACA0012", 18)
M = 122
s_norm = ac.active_subspace(grads=Gl[:M], space=ac.Uniform(np.full(18, -.01), np.full(18, .01)))
s_phys = ac.active_subspace(grads=Gl[:M], space=ac.Uniform.reference(18))
e = np.linalg.eigvalsh(Gl[:M].T @ Gl[:M] / M)[::-1]
print("NACA0012 lift rows 1-122: normalized lambda1 %.4g; physical-gradient lambda1 via Uniform.reference %.4g; eigh(G'G/M) %.4g (book Fig 5.11 ~8e2)" % (s_norm.eigenvalues[0], s_phys.eigenvalues[0], e[0]))
# Item 11: ONERA-M6 drag dimension rules and R2 conventions
Xo, flo, fdo, Glo, Gdo = airfoil("ONERA-M6", 50)
spo = ac.Uniform(np.full(50, -.05), np.full(50, .05))
for gap in ("linear", "log"):
    s = ac.active_subspace(grads=Gdo, space=spo, dimension_rule=ac.subspaces.EVG(gap=gap))
    print(f"ONERA-M6 drag EVG(gap={gap!r}): n={s.n}")
s = ac.active_subspace(grads=Gdo, space=spo, nboot=200, rng=0)
for rule in ("rs", "li"):
    try:
        print(f"ONERA-M6 drag {rule}: n={ac.with_dimension(s, rule).n if False else ac.active_subspace(grads=Gdo, space=spo, nboot=200, rng=0, dimension_rule=rule).n}")
    except Exception as ex:
        print(rule, type(ex).__name__, getattr(ex, 'code', None), str(ex)[:100])
print("ONERA-M6 drag |lambda| tail:", np.array2string(np.abs(s.eigenvalues[[0,1,2,10,30,47,48,49]]), precision=2))
for n in (1, 2, 3):
    sur = ac.fit_surrogate(X=Xo, f=fdo, subspace=ac.with_dimension(s, n), kind="rbf", degree=2)
    rin = 1 - np.sum(sur.residuals ** 2) / np.sum((fdo - fdo.mean()) ** 2)
    print(f"ONERA-M6 drag RBF n={n}: loo_r_squared={sur.loo_r_squared:.3f}, 1-SSres/SStot from .residuals={rin:.4f}")
