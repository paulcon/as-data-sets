import numpy as np, sys, warnings
sys.path.insert(0, '.')
from fvsolver import PDESolver
import active_subspaces as ac
warnings.simplefilter('ignore')
R='/Users/paulconstantine/PycharmProjects/as-data-sets/uq_benchmarks/Elliptic_PDE_Random_Coefficients/'
d = np.load('elliptic_proto.npz'); s = PDESolver(R+'kl-long.npz')
space = ac.Gaussian.reference(100)
w1 = d['long_W'][:, 0]
sub = ac.subspace_from_basis(w1[:, None], space=space)
rng = np.random.default_rng(5); Xt = rng.standard_normal((300, 100)); ft = np.array([s.qoi_and_grad(x)[0] for x in Xt]); yt = Xt@w1
y1 = np.array([-3, -1.5, 0, 1.5, 3.]); fd = np.array([s.qoi_and_grad(v*w1)[0] for v in y1])
print('train f', fd)
print('test y range', yt.min(), yt.max())
for kw in [dict(kind='rbf', degree=2), dict(kind='rbf', degree=2, log10_eta=0.0), dict(kind='rbf', degree=1), dict(kind='poly', degree=2), dict(kind='poly', degree=4)]:
    rs = ac.fit_surrogate(X=y1[:, None]*w1[None, :], f=fd, subspace=sub, **kw)
    p = ac.predict(rs, Xt); e = np.abs(p-ft)/np.abs(ft)
    extra = f" ls={getattr(rs,'length_scales',None)}" if kw['kind']=='rbf' else ''
    print(kw, f'mean rel {e.mean():.3e} median {np.median(e):.3e} max {e.max():.2e}', extra)
# full eigen-subspace RBF (true v1 path with eigenvalues)
sub2 = ac.with_dimension(ac.active_subspace(space=space, grads=np.array([s.qoi_and_grad(x)[1] for x in Xt[:100]])), 1)
