import numpy as np, time
from scipy.special import roots_legendre
exec(open('ebola_proto.py').read().split('# nominal')[0])
def C_tensor(lb,ub,k):
    t,w=roots_legendre(k); w=w/2
    # chunk over first two dims
    C=np.zeros((8,8)); grids=np.stack(np.meshgrid(*([t]*6),indexing='ij'),-1).reshape(-1,6); wg=np.prod(np.stack(np.meshgrid(*([w]*6),indexing='ij'),-1).reshape(-1,6),1)
    for a in range(k):
        for b in range(k):
            T=np.column_stack([np.full(len(grids),t[a]),np.full(len(grids),t[b]),grids]); W=w[a]*w[b]*wg
            X=lb+(ub-lb)/2*(T+1); G=R0_grad(X)*(ub-lb)/2; C+=(G*W[:,None]).T@G
    return C
for lb,ub,n in [(lb_L,ub_L,'L'),(lb_S,ub_S,'S')]:
    for k in [7,8]:
        t0=time.time(); e,V=np.linalg.eigh(C_tensor(lb,ub,k)); e=e[::-1]; print(n,k,np.array2string(e,precision=4), np.round(V[:,-1]*np.sign(V[7,-1]),4), '%.1fs'%(time.time()-t0))
