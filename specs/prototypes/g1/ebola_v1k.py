import numpy as np, time, sys, resource
from scipy.special import roots_legendre
exec(open('ebola_proto.py').read().split('# nominal')[0])
k=int(sys.argv[1]); t,w=roots_legendre(k); w=w/2
T=np.stack(np.meshgrid(*([t]*8),indexing='ij'),-1).reshape(-1,8); W=np.prod(np.stack(np.meshgrid(*([w]*8),indexing='ij'),-1).reshape(-1,8),1)
space=ac.Uniform(lower=lb_S,upper=ub_S)
t0=time.time(); X=space.from_normalized(T); G=R0_grad(X); del T
sub=ac.active_subspace(space=space,grads=G,quad_weights=W/W.sum())
print(k, len(W), sub.eigenvalues[:3], '%.1fs'%(time.time()-t0), 'maxrss GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9)
