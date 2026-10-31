import sys, time, numpy as np, active_subspaces as ac
np.set_printoptions(precision=4, linewidth=160)
fn = sys.argv[1]; t0 = time.time()
names = open(fn).readline().strip().split(',')
D = np.loadtxt(fn, delimiter=',', skiprows=1)
X, F = D[:,1:106], D[:,106:109]; G = {'lift': D[:,109:214], 'drag': D[:,214:319], 'ea': D[:,319:424]}
sp = ac.Uniform(lower=np.full(105,-.3), upper=np.full(105,.3), names=tuple(names[1:106]))
for j, k in enumerate(['lift','drag','ea']):
    f = F[:,j]
    s = ac.active_subspace(space=sp, X=X, grads=G[k], nboot=100, rng=0)
    out = []
    for n in (1, 2):
        s2 = ac.active_subspace(space=sp, X=X, grads=G[k], n=n)
        for deg in (1, 2):
            sur = ac.fit_surrogate(X=X, f=f, space=sp, subspace=s2, kind='poly', degree=deg)
            out.append(f'n{n}d{deg} R2={1-np.sum(np.asarray(sur.residuals)**2)/np.sum((f-f.mean())**2):.3f}')
    w1 = np.asarray(s.W1)[:,0]
    print(k, 'eig[:4]', np.asarray(s.eigenvalues)[:4], 'auto n', s.n, *out, '| top-3 |w1| idx', np.argsort(-abs(w1))[:3]+1)
print(f'{time.time()-t0:.1f}s')
