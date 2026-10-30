import sys, time
import numpy as np
sys.path.insert(0, '.')
from pde_model import PDESolver, PDEMisfit
REPO = '/Users/paulconstantine/PycharmProjects/as-data-sets'
rng = np.random.default_rng(20261029)

for case in ['short', 'long']:
    S = PDESolver(case, datadir=REPO + '/Elliptic_PDE')
    print(f'==== corrcase={case}  nx={S.nx} N={S.N} m={S.m}  nnz(A)={S.stiffness(S.coeff(np.zeros(S.m))).nnz}')
    loga = S.B @ rng.standard_normal((S.m, 2000))
    print(f'  pointwise std of log a over prior: min {loga.std(1).min():.3f} max {loga.std(1).max():.3f}')
    # (a) residual
    res = []
    for k in range(5):
        x = rng.standard_normal(S.m)
        u, a, _ = S.solve(x)
        r = S.residual(u, a)
        res.append((np.abs(r).max(), np.abs(u).max(), a.min(), a.max()))
    print('  (a) max|residual| per sample:', ' '.join(f'{r[0]:.1e}' for r in res),
          ' | max u', f'{max(r[1] for r in res):.3e}', ' a range', f'{min(r[2] for r in res):.2e}..{max(r[3] for r in res):.2e}')
    # (b) FD in random directions, qoi
    errs = []
    for k in range(5):
        x = rng.standard_normal(S.m); v = rng.standard_normal(S.m); v /= np.linalg.norm(v)
        g = S.grad_qoi(x); h = 1e-4
        fd = (S.qoi(x + h*v) - S.qoi(x - h*v)) / (2*h)
        errs.append(abs(fd - g @ v) / abs(g @ v))
    print('  (b) qoi grad vs central FD (h=1e-4), rel err:', ' '.join(f'{e:.1e}' for e in errs))
    # coordinate check, full gradient at one point
    x = rng.standard_normal(S.m); g = S.grad_qoi(x); h = 1e-5
    fdg = np.array([(S.qoi(x + h*e) - S.qoi(x - h*e)) / (2*h) for e in np.eye(S.m)])
    print(f'  (b) full FD gradient (100 coords, h=1e-5): ||g-fd||/||g|| = {np.linalg.norm(g-fdg)/np.linalg.norm(g):.1e}; q={S.qoi(x):.5e} ||g||={np.linalg.norm(g):.3e}')
    # misfit
    M = PDEMisfit(case, datadir=REPO + '/Elliptic_PDE')
    xt = rng.standard_normal(M.m); obs = M.observations(xt)
    sig2 = 1e-4*np.linalg.norm(obs)**2
    M.set_data(obs + np.sqrt(sig2)*rng.standard_normal(obs.shape), sig2)
    errs = []
    for k in range(5):
        x = rng.standard_normal(M.m); v = rng.standard_normal(M.m); v /= np.linalg.norm(v)
        g = M.misfit_grad(x); h = 1e-5
        fd = (M.misfit(x + h*v) - M.misfit(x - h*v)) / (2*h)
        errs.append(abs(fd - g @ v) / abs(g @ v))
    print('  (b) misfit grad vs central FD (h=1e-5), rel err:', ' '.join(f'{e:.1e}' for e in errs))
    # (d) timings
    X = rng.standard_normal((50, S.m))
    for name, fn in [('qoi', S.qoi), ('grad_qoi', S.grad_qoi), ('misfit', M.misfit), ('misfit_grad', M.misfit_grad)]:
        fn(X[0]); t = time.perf_counter()
        for xx in X: fn(xx)
        print(f'  (d) {name:12s} {1e3*(time.perf_counter()-t)/len(X):6.1f} ms/call')
    t = time.perf_counter(); 
    for xx in X: S.stiffness(S.coeff(xx))
    print(f'  (d) assembly only  {1e3*(time.perf_counter()-t)/len(X):6.1f} ms/call')
