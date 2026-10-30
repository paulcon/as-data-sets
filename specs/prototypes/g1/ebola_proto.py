import numpy as np, time, itertools
from scipy.special import roots_legendre
import active_subspaces as ac
lb_L = np.array([.1, .1, .05, .41, .0276, .081, .25, .0833]); ub_L = np.array([.4, .4, .2, 1, .1702, .21, .5, .7])
lb_S = np.array([.1, .1, .05, .41, .0275, .1236, .25, .0833]); ub_S = np.array([.4, .4, .2, 1, .1569, .384, .5, .7])
def R0(x):
    b1,b2,b3,r1,g1,g2,om,p = x.T
    return (b1 + b2*r1*g1/om + b3*p/g2)/(g1 + p)
def R0_grad(x):
    b1,b2,b3,r1,g1,g2,om,p = x.T; D=g1+p; R=R0(x)
    return np.column_stack([1/D, r1*g1/om/D, p/g2/D, b2*g1/om/D, (b2*r1/om - R)/D, -b3*p/g2**2/D, -b2*r1*g1/om**2/D, (b3/g2 - R)/D])
# nominal R0
nomL=np.array([[.376,.135,.163,.98,.0542,.174,.325,.5]]); nomS=np.array([[.251,.395,.079,.76,.051,.0833,.370,.442]])
print('R0 nominal L,S', R0(nomL), R0(nomS))
# FD check grad
x=lb_L+(ub_L-lb_L)*np.random.default_rng(0).random((3,8)); g=R0_grad(x)
for j in range(8):
    e=np.zeros(8); e[j]=1e-7; print(j, np.max(np.abs((R0(x+e)-R0(x))/1e-7 - g[:,j])), end='; ')
print()
def tensor(k,m=8):
    t,w=roots_legendre(k); w=w/2
    T=np.array(list(itertools.product(t,repeat=m))); W=np.prod(np.array(list(itertools.product(w,repeat=m))),axis=1)
    return T,W
for lb,ub,name in [(lb_L,ub_L,'L'),(lb_S,ub_S,'S')]:
  for k in [3,4,5,6]:
    t0=time.time(); T,W=tensor(k); X=lb+(ub-lb)/2*(T+1); G=R0_grad(X)*(ub-lb)/2
    C=(G*W[:,None]).T@G; e,V=np.linalg.eigh(C); e=e[::-1]; V=V[:,::-1]
    print(name,k,'sumw',W.sum(),'eig',np.array2string(e,precision=4),'w1',np.round(V[:,0]*np.sign(V[np.argmax(abs(V[:,0])),0]),4),f'{time.time()-t0:.1f}s')
  # v1
  space=ac.Uniform(lower=lb,upper=ub,names=['b1','b2','b3','rho1','g1','g2','om','psi'])
  T,W=tensor(5); X=lb+(ub-lb)/2*(T+1)
  t0=time.time(); sub=ac.active_subspace(space=space,X=X,grads=R0_grad(X),quad_weights=W); print('v1 quad k=5',sub.eigenvalues, sub.n, sub.W[:,0], f'{time.time()-t0:.1f}s')
  rng=np.random.default_rng(42); Xmc=space.sample(10000,rng=rng)
  sub2=ac.active_subspace(space=space,grads=R0_grad(Xmc),nboot=100,rng=rng); print('v1 MC 1e4',sub2.eigenvalues, sub2.n, sub2.W[:,0])
  print('legacy-scaled (x 2^-8) k=6 eig would be', e/256)
