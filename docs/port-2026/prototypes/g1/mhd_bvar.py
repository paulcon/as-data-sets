import numpy as np, itertools
from scipy.special import roots_legendre
import mhd_gradients_copy as g
lb,ub=g.lb,g.ub
t,w=roots_legendre(11); w=w/2
T=np.array(list(itertools.product(t,repeat=5))); W=np.prod(np.array(list(itertools.product(w,repeat=5))),1)
X=lb+(ub-lb)/2*(T+1)
print('u range', g.uavg(X).min(), g.uavg(X).max(), ' B range', g.Bind(X).min(), g.Bind(X).max())
def B_var(lx, l=1., mu0=1., p=1):
    mu,rho,dp,eta,B0=np.exp(lx).T; Ha=B0*l/np.sqrt(eta*mu)
    return dp*l*mu0/(2*B0**p)*(1-2/Ha*np.tanh(Ha/2))
def fdgrad(f,X,h=1e-6):
    return np.column_stack([(f(X+h*np.eye(5)[j])-f(X-h*np.eye(5)[j]))/(2*h)*(ub[j]-lb[j])/2 for j in range(5)])
for kw in [dict(),dict(p=2),dict(l=2.),dict(l=0.5),dict(mu0=4*np.pi*1e-7)]:
    f=lambda x: B_var(x,**kw); G=fdgrad(f,X); C=(G*W[:,None]).T@G; e,V=np.linalg.eigh(C); v=V[:,-1]*np.sign(V[0,-1])
    print(kw, 'range %.3g-%.3g'%(f(X).min(),f(X).max()), 'eig', np.round(e[::-1][:2],4), 'ratio %.1f'%(e[-1]/e[-2]), 'w1', np.round(v,3))
print('--- grid')
target=np.array([.38,0,-.56,.49,-.56])
res=[]
for l in [0.25,0.4,0.5,0.6,0.75,1,1.5,2]:
  for p in [0,1,2]:
    f=lambda x: B_var(x,l=l,p=p); G=fdgrad(f,X); C=(G*W[:,None]).T@G; e,V=np.linalg.eigh(C); v=V[:,-1]*np.sign(V[0,-1])
    res.append((np.linalg.norm(v-target),l,p,e[-1],e[-1]/e[-2],np.round(v,3)))
for r in sorted(res,key=lambda r:r[0])[:5]: print('err %.3f l=%g p=%d lam1=%.3g ratio=%.1f'%r[:5], r[5])
