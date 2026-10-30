import numpy as np, sys, time, warnings
import active_subspaces as ac
d=sys.argv[1]
np.set_printoptions(linewidth=200,precision=4,suppress=True)
X=np.loadtxt(f'{d}/inputs-ShortCorr.GW02.txt')
fs=np.loadtxt(f'{d}/RunOff02_Saturated_10k.txt'); fu=np.loadtxt(f'{d}/RunOff02_Unsaturated_10k.txt')
sp=ac.Gaussian.reference(100)
def leg_sign(W): s=np.sign(W[0]); s[s==0]=1; return W*s
def r2_in(Y,f,deg):
    s=ac.fit_surrogate(X=None,f=f,subspace=None) if False else None
def poly_r2(Y,f,deg):
    # legacy in-sample R2: total-degree polynomial LS
    from itertools import combinations_with_replacement
    M,n=Y.shape; cols=[np.ones(M)]
    for k in range(1,deg+1):
        for c in combinations_with_replacement(range(n),k): cols.append(np.prod(Y[:,c],axis=1))
    B=np.column_stack(cols); c=np.linalg.lstsq(B,f,rcond=None)[0]
    return 1-np.sum((B@c-f)**2)/(M*np.var(f))
res={}
for name,f in [('sat',fs),('unsat',fu)]:
    t=time.time()
    with warnings.catch_warnings(record=True) as Wn:
        warnings.simplefilter('always')
        ols=ac.active_subspace(X=X,f=f,space=sp,estimator='ols',n=1)
        t1=time.time()
        q=ac.active_subspace(X=X,f=f,space=sp,estimator='qphd')
        t2=time.time()
    A=np.hstack([np.ones((len(f),1)),X]); u=np.linalg.lstsq(A,f,rcond=None)[0][1:]
    wl=u/np.linalg.norm(u); wl=wl*np.sign(wl[0])
    print(f'== {name}: ols {t1-t:.2f}s qphd {t2-t1:.1f}s warnings={[str(w.message)[:80] for w in Wn]}')
    print(' OLS lam v1', ols.eigenvalues[0], 'legacy', u@u, ' |cos| ', abs(wl@ols.W1[:,0]), 'sign v1 vs legacy', np.sign(wl@ols.W1[:,0]), 'sign v1 vs u', np.sign(u@ols.W1[:,0]))
    print(' OLS w[:10]', wl[:10], ' sum|w[30:]|^2', np.sum(wl[30:]**2))
    print(' QPHD eig[:8]', q.eigenvalues[:8], ' default n', q.n)
    print(' QPHD/OLS first-vector |cos|', abs(q.W1[:,0]@wl), ' max|diff| after align', np.abs(q.W1[:,0]*np.sign(q.W1[:,0]@wl)-wl).max())
    # legacy QPHD with gamma=1: C = g g^T + H H^T, should equal v1 with Sigma=I
    Wq=leg_sign(q.eigenvectors.copy())
    for nm,W,k in [('linSS',None,None)]: pass
    k=1 if name=='sat' else 3
    # legacy linear subspace partition(k): eigh of rank-1 C
    e,V=np.linalg.eigh(np.outer(u,u)); idx=np.argsort(abs(e))[::-1]; V=leg_sign(V[:,idx])
    r_lin=[poly_r2(X@V[:,:k],f,dg) for dg in (1,2,3)]
    r_q=[poly_r2(X@Wq[:,:k],f,dg) for dg in (1,2,3)]
    r_lin1=[poly_r2(X@V[:,:1],f,dg) for dg in (1,2,3)]
    print(f' in-sample R2 (k={k}) linSS', np.round(r_lin,4), ' quadSS', np.round(r_q,4), ' linSS 1D', np.round(r_lin1,4))
    sub_k=ac.with_dimension(q,k)
    for dg in (1,2,3):
        s=ac.fit_surrogate(X=X,f=f,subspace=sub_k,kind='poly',degree=dg)
        rin=1-np.sum(s.residuals**2)/np.sum((f-f.mean())**2)
        print(f'  v1 poly deg{dg} on qphd k={k}: in-sample {rin:.4f} loo {s.loo_r_squared:.4f}')
    res[name]=(ols,q)
print('zeros in fu', (fu==0).sum())
