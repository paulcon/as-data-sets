import numpy as np, scipy.io as sio, os, io, time, warnings
import active_subspaces as ac
D='/Users/paulconstantine/PycharmProjects/as-data-sets/MCMC/'
for fn in ['Vanilla_Chain.mat','Vanilla_Acceptance.mat','AV_Chain.mat','AV_Acceptance.mat','Gradient_Samples.mat']:
    m = sio.loadmat(D+fn); k = [k for k in m if not k.startswith('__')]
    print(fn, m['__header__'][:60], {kk: (m[kk].shape, m[kk].dtype) for kk in k})
Xv = sio.loadmat(D+'Vanilla_Chain.mat')['Chain']; av = sio.loadmat(D+'Vanilla_Acceptance.mat')['Accept'].ravel()
Y = sio.loadmat(D+'AV_Chain.mat')['Chain']; aa = sio.loadmat(D+'AV_Acceptance.mat')['Accept'].ravel()
G = sio.loadmat(D+'Gradient_Samples.mat')['Samples']
print('acceptance vanilla %.4f AV %.4f' % (av.mean(), aa.mean()))
# consistency: accepted flag at i means row i+1 differs from row i
diff = np.any(Xv[1:] != Xv[:-1], axis=1); print('vanilla: rows changed == accept[:-1]?', np.array_equal(diff, av[:-1].astype(bool)), 'unique rows', 1+diff.sum())
print('first rows norms', np.linalg.norm(Xv[:3], axis=1), 'last', np.linalg.norm(Xv[-1]))
sub = ac.active_subspace(space=ac.Gaussian.reference(100), grads=G, nboot=100, rng=0)
lam = sub.eigenvalues; print('v1 eigenvalues[:7]', lam[:7]); print(' ratios', lam[:6]/lam[1:7]); print(' EVG n =', sub.n)
# legacy eigenvectors (np.linalg.eigh of G^T G / M, legacy sign: first component? check)
C = G.T@G/G.shape[0]; e, W = np.linalg.eigh(C); e, W = e[::-1], W[:, ::-1]
print('max rel diff eigenvalues legacy-eigh vs v1', np.max(np.abs(e[:10]-lam[:10])/lam[:10]))
for j in range(3):
    print(f' w{j+1}: |<v1,eigh>|={abs(sub.W[:,j]@W[:,j]):.12f}  v1 sign of first comp {np.sign(sub.W[0,j])}, argmax comp idx {np.argmax(np.abs(sub.W[:,j]))}')
# Y chain mapping check: Y supposedly = W1^T x. stats
print('AV chain mean', Y.mean(0), 'std', Y.std(0))
yv = Xv @ sub.W[:, :4]; print('vanilla projected (v1 W) mean last half', yv[50000:].mean(0), 'std', yv[50000:].std(0))
# storage options
t=time.time(); b=io.BytesIO(); np.savez_compressed(b, chain=Xv); print('npz_compressed float64 MB', len(b.getvalue())/1e6, round(time.time()-t,1),'s')
b=io.BytesIO(); np.savez_compressed(b, chain=Xv[av.astype(bool) | np.r_[True, np.zeros(len(av)-1, bool)]]); 
idx = np.r_[0, 1+np.flatnonzero(diff)]; b=io.BytesIO(); np.savez_compressed(b, states=Xv[idx], accept=av.astype(np.uint8)); print('unique states+accept MB', len(b.getvalue())/1e6)
b=io.BytesIO(); np.savez_compressed(b, chain=Xv[::10]); print('thin x10 float64 MB', len(b.getvalue())/1e6)
b=io.BytesIO(); np.savez_compressed(b, chain=Xv[::10].astype(np.float32)); print('thin x10 float32 MB', len(b.getvalue())/1e6)
