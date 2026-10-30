import numpy as np, time, sys, warnings
sys.path.insert(0, '.')
from fvsolver import PDESolver
import active_subspaces as ac
R='/Users/paulconstantine/PycharmProjects/as-data-sets/Elliptic_PDE/'
rng = np.random.default_rng(2014)
space = ac.Gaussian.reference(100)
X = rng.standard_normal((300, 100)); Xtest = rng.standard_normal((500, 100))
out = {}
for case in ['long', 'short']:
    s = PDESolver(R+f'kl-{case}.npz'); t = time.time()
    fg = [s.qoi_and_grad(x) for x in X]; f = np.array([v for v, _ in fg]); G = np.array([g for _, g in fg])
    print(case, '300 value+grad in', round(time.time()-t, 1), 's; f range', f.min(), f.max())
    sub = ac.active_subspace(space=space, grads=G, nboot=100, rng=0)
    lam = sub.eigenvalues
    print('  n chosen (EVG):', sub.n, ' eigenvalues[:5]', lam[:5])
    print('  normalized singular values sqrt(lam/lam1):', np.round(np.sqrt(lam[:5]/lam[0]), 4))
    w1 = sub.W[:, 0]; print('  w1 top-5 |components| idx', np.argsort(-np.abs(w1))[:5], ' w1[:5]', np.round(w1[:5], 3))
    print('  sign rule: max |w1| comp positive?', w1[np.argmax(np.abs(w1))] > 0)
    g0 = s.qoi_and_grad(np.zeros(100))[1]; print('  local SENS at origin: top-2 coords (1-based)', np.argsort(-np.abs(g0))[:2]+1, '; argmax|w1| (1-based)', np.argmax(np.abs(w1))+1)
    ftest = np.array([s.qoi_and_grad(x)[0] for x in Xtest])
    for n in (1, 2):
        sn = ac.with_dimension(sub, n)
        y1 = np.array([-3, -1.5, 0, 1.5, 3.])
        Yd = y1[:, None] if n == 1 else np.array(np.meshgrid(y1, y1, indexing='ij')).reshape(2, -1).T
        Xd = Yd @ sn.W1.T
        fd = np.array([s.qoi_and_grad(x)[0] for x in Xd])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rs = ac.fit_surrogate(X=Xd, f=fd, subspace=sn, kind='rbf', degree=2)
            pred = ac.predict(rs, Xtest)
        err = np.abs(ftest - pred)/np.abs(ftest)
        print(f'  n={n}: AS-RBF mean rel err on 500 test pts {err.mean():.2e}; median log10 {np.median(np.log10(err)):.2f}')
    out[case] = dict(lam=lam, W=sub.W[:, :2])
np.savez('elliptic_proto.npz', **{f'{k}_{kk}': v for k, d in out.items() for kk, v in d.items()})
