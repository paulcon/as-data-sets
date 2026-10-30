import numpy as np, sys, warnings
sys.path.insert(0, '.')
from fvsolver import PDESolver
import active_subspaces as ac
warnings.simplefilter('ignore')
s = PDESolver('/Users/paulconstantine/PycharmProjects/as-data-sets/uq_benchmarks/Elliptic_PDE_Random_Coefficients/kl-long.npz')
rng = np.random.default_rng(2014); space = ac.Gaussian.reference(100)
X = rng.standard_normal((300, 100)); fg=[s.qoi_and_grad(x) for x in X]; f=np.array([a for a,_ in fg]); G=np.array([b for _,b in fg])
sub = ac.active_subspace(space=space, grads=G)
Xt = rng.standard_normal((500, 100)); ft = np.array([s.qoi_and_grad(x)[0] for x in Xt])
lam = sub.eigenvalues; r = np.sum(np.abs(lam))/lam[0]; print('sum|lam|/lam1 =', r)
y1 = np.array([-3, -1.5, 0, 1.5, 3.])
for n in (1, 2):
    sn = ac.with_dimension(sub, n)
    Yd = y1[:, None] if n == 1 else np.array(np.meshgrid(y1, y1, indexing='ij')).reshape(2, -1).T
    Xd = Yd @ sn.W1.T; fd = np.array([s.qoi_and_grad(x)[0] for x in Xd])
    for le in [None, -np.log10(r)]:
        rs = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2, log10_eta=le)
        e = np.abs(ac.predict(rs, Xt)-ft)/np.abs(ft)
        print(f'n={n} log10_eta={le} ls={rs.length_scales} mean rel err {e.mean():.3e}')
# full-space RBF, equal cost 3M+P
for P in (5, 25):
    Xf = rng.standard_normal((900+P, 100)); ff = np.array([s.qoi_and_grad(x)[0] for x in Xf])
    rs = ac.fit_surrogate(X=Xf, f=ff, space=space, subspace=None, kind='rbf', degree=1)
    e = np.abs(ac.predict(rs, Xt)-ft)/np.abs(ft); print(f'full RBF ({900+P} pts) mean rel err {e.mean():.3e}')
