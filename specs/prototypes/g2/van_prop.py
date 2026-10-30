import numpy as np, sys
sys.path.insert(0,'.')
from fvsolver import PDESolver
s=PDESolver('/Users/paulconstantine/PycharmProjects/as-data-sets/MCMC/kl-short.npz')
rng=np.random.default_rng(1510); xt=rng.standard_normal(100); obs=s.observations(xt); sig2=1e-4*obs@obs; d=obs+np.sqrt(sig2)*rng.standard_normal(7)
mis=lambda x:.5*np.sum((s.observations(x)-d)**2)/sig2
for std in [0.1, np.sqrt(0.1)]:
    r=np.random.default_rng(7); x=r.standard_normal(100); f=mis(x); acc=[]
    for i in range(400):
        xp=x+std*r.standard_normal(100); fp=mis(xp)
        a=np.log(r.uniform())<(f-fp)+.5*(x@x-xp@xp); acc.append(a)
        if a: x,f=xp,fp
    print(f'std={std:.3f}: acceptance all {np.mean(acc):.2f}, last 200 {np.mean(acc[200:]):.2f}')
