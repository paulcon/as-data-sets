# Item 6 follow-up: lower bound from the variance of the training f instead of the gradient sample
import numpy as np, sys, warnings
sys.path.insert(0, '/Users/paulconstantine/PycharmProjects/as-data-sets/specs/prototypes/g2')
from fvsolver import PDESolver
import active_subspaces as ac
warnings.simplefilter('ignore')
for case in ("short",):
    s = PDESolver(f'/Users/paulconstantine/PycharmProjects/as-data-sets/Elliptic_PDE/kl-{case}.npz')
    rng = np.random.default_rng(2014)
    X = rng.standard_normal((300, 100)); fg = [s.qoi_and_grad(x) for x in X]
    f0 = np.array([a for a, _ in fg]); G = np.array([b for _, b in fg])
    sub = ac.active_subspace(space=ac.Gaussian.reference(100), grads=G)
    Xt = rng.standard_normal((500, 100)); ft = np.array([s.qoi_and_grad(x)[0] for x in Xt])
    lam = np.abs(sub.eigenvalues); S = lam.sum()
    y1 = np.array([-3, -1.5, 0, 1.5, 3.])
    for n in (1, 2):
        sn = ac.with_dimension(sub, n)
        Yd = y1[:, None] if n == 1 else np.array(np.meshgrid(y1, y1, indexing='ij')).reshape(2, -1).T
        Xd = Yd @ sn.W1.T; fd = np.array([s.qoi_and_grad(x)[0] for x in Xd])
        err = lambda r: np.mean(np.abs(ac.predict(r, Xt) - ft) / np.abs(ft))
        r0 = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2)
        lo_g = np.log10(2 * f0.var() / S); print("  short: 2*sig2/S =", 10**lo_g, "> 2*C1 = 2 -> (4.13) interval empty"); rg = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind="rbf", degree=2, log10_eta=min(lo_g, np.log10(2.0)))
        lo_d = min(np.log10(2 * fd.var() / S), np.log10(2.0) - 1e-3)
        rd = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2, log10_eta_bounds=(lo_d, np.log10(2.0)))
        print(f"{case} n={n}: default {err(r0):.3f} | bounds from gradient-sample var {err(rg):.3f} | bounds from design var {err(rd):.3f} (lo={lo_d:.2f})")
