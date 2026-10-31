import numpy as np, warnings, time
import active_subspaces as ac
d=np.load('0.25C.npy'); X=d[:,1:20]; C=7.5/36000*d[:,70:120]
sp=ac.Uniform.reference(19)
for idx in [2,17,38]:
    t=time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s=ac.active_subspace(X=X,f=C[:,idx],space=sp,estimator='qphd')
    e=np.abs(s.eigenvalues[:5]); print(idx,'qphd |eig|[:5]',np.array2string(e,precision=3),'log10 gap 1-2 %.2f'%np.log10(e[0]/e[1]),'%.1fs'%(time.time()-t))
