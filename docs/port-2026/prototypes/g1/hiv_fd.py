import numpy as np, time
from hiv_model_copy import Tcells
exec(open('hiv_load.py').read().split('labels=')[0].split("print('load")[0])
nominal = np.array([10, .15, 5, .2, 55.6, 3.87e-3, 1e-6, 4.5e-4, 7.45e-4, 5.22e-4, 3e-6, 3.3e-4, 6e-9, .537, .285, 7.79e-6, 1e-6, 4e-5, .01, .28, .05, .005, .005, .015, 2.39, 3e-4, .97])
xl=.975*nominal; xu=1.025*nominal
times = np.array([5, 15, 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 750, 1000, 1600, 1800, 2000, 2200, 2400, 2800, 3400])
t0=time.time(); f=Tcells(X[:2], np.arange(1,3401)); print('2 solves to 3400 days: %.2fs'%(time.time()-t0))
print('recomputed vs shipped outputs, max rel diff', np.max(np.abs(f[:,times-1]-F[:2])/F[:2]))
# one trajectory FD for all times, sample k
k=int(np.argmax(F[:,-1]<0.1001)); print('frozen sample', k)
for s in [0, k]:
  x0=X[s:s+1]; base=Tcells(x0, times)[0]
  for hn in [1e-7, 1e-4, 1e-2]:
    G=np.empty((21,27))
    for j in range(27):
        xp=x0.copy(); xp[0,j]+=hn*(xu[j]-xl[j])/2
        G[:,j]=(Tcells(xp,times)[0]-base)/hn   # derivative wrt normalized coord
    for i in [2,20]:
        ship=DF[s,27*i:27*(i+1)]
        print(f'sample {s} h={hn:g} t={times[i]}: rel err vs shipped {np.linalg.norm(G[i]-ship)/np.linalg.norm(ship):.2e}  |g|={np.linalg.norm(G[i]):.3g} |ship|={np.linalg.norm(ship):.3g}')
