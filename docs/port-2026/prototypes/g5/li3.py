import numpy as np, warnings, time
import active_subspaces as ac
CAPFAC={'0.25C':7.5/36000.,'1C':30./36000.,'4C':120./36000.}
def coef(X,f):
    A=np.hstack((np.ones((X.shape[0],1)),X)); u=np.linalg.lstsq(A,f,rcond=None)[0][1:]; return u/np.linalg.norm(u)
def boot(X,f,rng,nb=100):
    M=X.shape[0]; a=np.empty((nb,X.shape[1]))
    for b in range(nb):
        i=rng.integers(M,size=M); a[b]=coef(X[i],f[i])
    return a.std(0)
names=["eps_a","eps_s","eps_c","brugg_a","brugg_s","brugg_c","t_plus","D","D_a","D_c","sig_a","sig_c","k_a","k_c","r_a","r_c","L_a","L_s","L_c"]
space=ac.Uniform.reference(19)
rng=np.random.default_rng(0)
for f in ['0.25C','1C','4C']:
    d=np.load(f+'.npy')
    if f=='4C': d=np.delete(d,1537,0)
    X=d[:,1:20]; V=d[:,20:70]; C=CAPFAC[f]*d[:,70:120]
    t=time.time()
    aC=np.array([coef(X,C[:,i]) for i in range(50)])
    aV=np.array([coef(X,V[:,i]) for i in range(1,49)])
    # v1
    maxdiff=0; flips=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(50):
            s=ac.active_subspace(X=X,f=C[:,i],space=space,estimator='ols',n=1)
            w=s.W1[:,0]
            if np.dot(w,aC[i])<0: flips+=1
            maxdiff=max(maxdiff,np.abs(np.abs(w)-np.abs(aC[i])).max())
    print(f,'cap: v1 vs legacy max|diff| (abs)',maxdiff,' v1 sign flips vs regression sign:',flips,'/50')
    seC=np.array([boot(X,C[:,i],rng) for i in range(50)]); seV=np.array([boot(X,V[:,i],rng) for i in range(1,49)])
    print('  boot SE cap max %.4f at idx %d ; volt max %.4f ; time %.1fs'%(seC.max(),seC.max(1).argmax(),seV.max(),time.time()-t))
    print('  frac cap SE>0.005: %.3f, >0.01: %d entries'%((seC>0.005).mean(),(seC>0.01).sum()))
    for idx in ([2,17,38] if f=='0.25C' else [5,34,48]):
        w=aC[idx]; top=np.argsort(-np.abs(w))[:4]
        r=np.corrcoef(X@w,C[:,idx])[0,1]
        print('  cap idx',idx,'top',[(names[j],round(w[j],2)) for j in top],'R2 lin %.3f'%r**2)
    for idx in [2,38,47]:
        w=aV[idx-1]; top=np.argsort(-np.abs(w))[:4]
        r=np.corrcoef(X@w,V[:,idx])[0,1]
        print('  vol idx',idx,'top',[(names[j],round(w[j],2)) for j in top],'R2 lin %.3f'%r**2)
    np.savez(f'li_res_{f}.npz',aC=aC,aV=aV,seC=seC,seV=seV)
