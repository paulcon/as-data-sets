import numpy as np, time, warnings
import active_subspaces as ac
d=np.load('pv.npy'); X=d[:,1:6]; f=d[:,6]; G=d[:,7:]
xl=np.array([0.05989,np.log(2.2e-11),1.0,0.16625,93.75]); xu=np.array([0.23958,np.log(2.2e-7),2.0,0.665,375.0])
Xt=X.copy(); Xt[:,1]=np.log(X[:,1])
names=['I_SC','log I_S','n','R_S','R_P']
space=ac.Uniform(lower=xl,upper=xu,names=names)
Gphys=G/((xu-xl)/2)           # undo normalization so v1 re-applies it
t=time.time()
s=ac.active_subspace(grads=Gphys,space=space,nboot=500,rng=2016)
print('v1 time %.1fs'%(time.time()-t))
np.set_printoptions(precision=4,linewidth=150)
print('eig',s.eigenvalues, 'ratios',s.eigenvalues[:-1]/s.eigenvalues[1:], 'n(EVG)',s.n)
print('W1',s.W[:,0]); print('W2',s.W[:,1])
# reference route
s2=ac.active_subspace(grads=G,space=ac.Uniform.reference(5),nboot=0)
print('reference-space eig equal',np.allclose(s2.eigenvalues,s.eigenvalues), np.allclose(abs(s2.W),abs(s.W)))
# wrong route (double scaling) for illustration
s3=ac.active_subspace(grads=G,space=space)
print('double-scaled W1 (WRONG)',s3.W[:,0], 'eig',s3.eigenvalues[:2])
# legacy-style direct
C=G.T@G/len(G); lam,W=np.linalg.eigh(C); print('direct eigh',lam[::-1])
lo,hi=ac.bootstrap_interval(abs(s.bootstrap_eigenvalues),kind='minmax'); print('boot minmax lo',lo,'hi',hi)
print('mean subspace dist',ac.mean_distance(s) if hasattr(ac,'mean_distance') else None)
XXn=space.to_normalized(Xt); print('normalized X range',XXn.min(0),XXn.max(0))
# legacy normalizer w/ typo
xu_t=xu.copy(); xu_t[0]=0.23598; XXl=2*(Xt-xl)/(xu_t-xl)-1; print('legacy XX ISC max',XXl[:,0].max())
# eigenvector bootstrap replicate (hand coded, as legacy cell 9) - timing with 1e4
rng=np.random.default_rng(0); M=len(G); t=time.time(); B=1000
w1=np.empty((B,5))
for b in range(B):
    i=rng.integers(M,size=M); _,_,Vt=np.linalg.svd(G[i],full_matrices=False); v=Vt[0]; w1[b]=v*np.sign(v[np.argmax(abs(v))])
print('1000 eigvec replicates %.2fs; w1 range'%(time.time()-t), w1.min(0), w1.max(0))
y=Xt; 
for deg in (1,2,3):
    sur=ac.fit_surrogate(X=Xt,f=f,subspace=ac.with_dimension(s,1),kind='poly',degree=deg); print('poly deg',deg,'LOO R2 %.5f'%sur.loo_r_squared)
