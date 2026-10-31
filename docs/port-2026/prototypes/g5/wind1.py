import numpy as np, time, warnings
import active_subspaces as ac
d=np.load('wind.npy'); X=d[:,1:6]; F=d[:,6:]
names=['Vhub','Hs','WaveDir','Tp','PlatformDir']
out=(np.abs(X)>1).any(1); print('rows outside [-1,1]^5:',out.sum(), 'per col',(np.abs(X)>1).sum(0))
space=ac.Uniform.reference(5)
try:
    ac.local_linear_gradients(X=X[:200],f=F[:200,0],space=space)
    ac.local_linear_gradients(X=X,f=F[:,0],space=space,centers=np.arange(5))
    print('v1 accepted X outside support')
except Exception as e: print('v1 error:',type(e).__name__, getattr(e,'code',None), str(e)[:200])
# option: box from data bounds
lo=np.minimum(X.min(0),-1); hi=np.maximum(X.max(0),1)
sp2=ac.Uniform(lower=lo,upper=hi,names=names)
rng=np.random.default_rng(0)
for i in range(7):
    t=time.time()
    cen=rng.choice(len(X),2000,replace=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        g=ac.local_linear_gradients(X=X,f=F[:,i],space=space if out.sum()==0 else sp2,p=60,centers=cen)
        s=ac.active_subspace(grads=g,nboot=100,rng=1)
    # legacy-like: 81 centers, normalized coordinates directly (identity)
    G=np.empty((81,5)); idx=rng.integers(len(X),size=81)
    for k,ii in enumerate(idx):
        D2=((X-X[ii])**2).sum(1); ind=np.argsort(D2)[:60]
        A=np.hstack([np.ones((60,1)),X[ind]]); G[k]=np.linalg.lstsq(A,F[ind,i],rcond=None)[0][1:]
    lamL,WL=np.linalg.eigh(G.T@G/81); lamL=lamL[::-1]; WL=WL[:,::-1]
    print(f'out {i}: v1(2000 ctrs, box space) eig',np.array2string(s.eigenvalues,precision=2), 'w1',s.W1[:,0].round(2),'n',s.n,'%.1fs'%(time.time()-t),'warn',len(w))
    print(f'       legacy-like(81 ctrs, identity) eig',np.array2string(lamL,precision=2),'w1',(WL[:,0]*np.sign(WL[np.argmax(abs(WL[:,0])),0])).round(2))
