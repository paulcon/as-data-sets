exec(open('ebola_proto.py').read().split('# nominal')[0])
from scipy.special import roots_legendre
import itertools
def tensor(k,m=8):
    t,w=roots_legendre(k); w=w/2
    return np.array(list(itertools.product(t,repeat=m))), np.prod(np.array(list(itertools.product(w,repeat=m))),axis=1)
T,W=tensor(6)
for lb,ub,n in [(lb_L,ub_L,'L'),(lb_S,ub_S,'S')]:
    X=lb+(ub-lb)/2*(T+1); G=R0_grad(X)*(ub-lb)/2; C=(G*W[:,None]).T@G; e,V=np.linalg.eigh(C); e=e[::-1]; V=V[:,::-1]
    a=(e[:2]*V[:,:2]**2).sum(1); print(n, np.round(a/a.sum(),3), 'mean R0', (R0(X)*W).sum())
