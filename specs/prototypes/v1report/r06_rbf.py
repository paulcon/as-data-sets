# Item 6: RBF default on Elliptic_PDE beta = 1 (kl-long), vs CDW 2014 Sec. 4.2-4.3 (alpha bounded by (4.13))
import numpy as np, sys, warnings, scipy.linalg, scipy.optimize
from scipy.spatial.distance import cdist
sys.path.insert(0, '/Users/paulconstantine/PycharmProjects/as-data-sets/specs/prototypes/g2')
from fvsolver import PDESolver
import active_subspaces as ac
s = PDESolver('/Users/paulconstantine/PycharmProjects/as-data-sets/Elliptic_PDE/kl-long.npz')
rng = np.random.default_rng(2014); space = ac.Gaussian.reference(100)
X = rng.standard_normal((300, 100)); fg = [s.qoi_and_grad(x) for x in X]
f0 = np.array([a for a, _ in fg]); G = np.array([b for _, b in fg])
sub = ac.active_subspace(space=space, grads=G)
Xt = rng.standard_normal((500, 100)); ft = np.array([s.qoi_and_grad(x)[0] for x in Xt])
lam = np.abs(sub.eigenvalues); S = lam.sum(); sig2 = f0.var()
print(f"sum|lam|={S:.4e} lam1={lam[0]:.4e} sigma_hat^2 (300 samples)={sig2:.4e}; CDW (4.13): {sig2/S:.3f} <= alpha <= C1 = 1")
y1 = np.array([-3, -1.5, 0, 1.5, 3.])
def cdw(Yd, fd, n, lo, hi):
    """CDW 2014 Sec. 4.3 step 5: product SE kernel, l_i^2 = alpha*S/lam_i, sigma^2 = alpha*S, eta^2 = alpha*sum_{q>n} lam, quadratic mean, ML in alpha on [lo, hi]."""
    from itertools import combinations_with_replacement as cwr
    k = Yd.shape[1]
    def basis(Y):
        cols = [np.ones(len(Y))] + [Y[:, i] for i in range(k)] + [Y[:, i] * Y[:, j] for i, j in cwr(range(k), 2)]
        return np.column_stack(cols)
    B = basis(Yd); tail = lam[n:].sum()
    def parts(a):
        l2 = a * S / lam[:k]
        Kc = np.exp(-0.5 * cdist(Yd / np.sqrt(l2), Yd / np.sqrt(l2), 'sqeuclidean'))
        Sig = a * S * Kc + a * tail * np.eye(len(fd))
        L = np.linalg.cholesky(Sig)
        Bt = scipy.linalg.solve_triangular(L, B, lower=True); ftt = scipy.linalg.solve_triangular(L, fd, lower=True)
        beta = np.linalg.lstsq(Bt, ftt, rcond=None)[0]; r = ftt - Bt @ beta
        return 0.5 * r @ r + np.log(np.diag(L)).sum(), l2, Sig, beta
    res = scipy.optimize.minimize_scalar(lambda la: parts(10 ** la)[0], bounds=(np.log10(lo), np.log10(hi)), method='bounded')
    a = 10 ** res.x; _, l2, Sig, beta = parts(a)
    alpha_w = np.linalg.solve(Sig, fd - B @ beta)
    def pred(Y):
        Kx = a * S * np.exp(-0.5 * cdist(Y / np.sqrt(l2), Yd / np.sqrt(l2), 'sqeuclidean'))
        return basis(Y) @ beta + Kx @ alpha_w
    return a, pred
warnings.simplefilter('ignore')
for n in (1, 2):
    sn = ac.with_dimension(sub, n)
    Yd = y1[:, None] if n == 1 else np.array(np.meshgrid(y1, y1, indexing='ij')).reshape(2, -1).T
    Xd = Yd @ sn.W1.T; fd = np.array([s.qoi_and_grad(x)[0] for x in Xd])
    err = lambda p: np.mean(np.abs(p - ft) / np.abs(ft))
    rs = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2)
    print(f"n={n} v1 default: log10_eta={rs.log10_eta:.2f} ell={np.round(rs.length_scales,4)} nugget={rs.nugget[0]:.1e} mean rel err {err(ac.predict(rs, Xt)):.3f}")
    rs1 = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2, log10_eta=-np.log10(S / lam[0]))
    print(f"n={n} v1 log10_eta fixed (ell1=1): mean rel err {err(ac.predict(rs1, Xt)):.3f}")
    b = (np.log10(2 * sig2 / S), np.log10(2.0))
    rsb = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2, log10_eta_bounds=b)
    print(f"n={n} v1 with log10_eta_bounds from CDW (4.13) (eta=2 alpha) {np.round(b,3)}: log10_eta={rsb.log10_eta:.3f} mean rel err {err(ac.predict(rsb, Xt)):.3f}")
    a, pred = cdw(Yd, fd, n, sig2 / S, 1.0)
    print(f"n={n} CDW Sec. 4.3 by hand (alpha in [{sig2/S:.3f}, 1]): alpha={a:.3f} mean rel err {err(pred(Yd @ np.eye(n) if False else (Xt @ sn.W1))):.3f}")
    a2, pred2 = cdw(Yd, fd, n, 1e-10, 10.0)
    print(f"n={n} CDW kernel, alpha unbounded [1e-10, 10]: alpha={a2:.2e} mean rel err {err(pred2(Xt @ sn.W1)):.3f}")
    # v1 objective landscape
    grid = np.arange(-10, 1.01, 1.0)
    Js = []
    for u in grid:
        try:
            Js.append(ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2, log10_eta=u).objective)
        except Exception as e:
            Js.append(np.nan)
    print(f"   v1 J(log10_eta) on {grid[0]}..{grid[-1]}: {np.round(Js, 2)}")
