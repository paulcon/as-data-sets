import numpy as np, sys, time
d=sys.argv[1]
t=time.time(); X=np.loadtxt(f'{d}/inputs-ShortCorr.GW02.txt'); print('X',X.shape, 'load s',round(time.time()-t,2))
fs=np.loadtxt(f'{d}/RunOff02_Saturated_10k.txt'); fu=np.loadtxt(f'{d}/RunOff02_Unsaturated_10k.txt')
print('fs',fs.shape,fs.min(),fs.max(),fs.mean()); print('fu',fu.shape,fu.min(),fu.max(),fu.mean(),'zeros',(fu==0).sum(),'neg',(fu<0).sum())
print('X mean range',X.mean(0).min(),X.mean(0).max(),'std range',X.std(0,ddof=1).min(),X.std(0,ddof=1).max())
print('X min/max',X.min(),X.max())
C=np.corrcoef(X.T); print('max offdiag |corr|',np.abs(C-np.eye(100)).max())
from scipy import stats
print('KS vs N(0,1) pooled', stats.kstest(X.ravel(),'norm'))
print('KS vs U(-1,1)?', (np.abs(X)<=1).mean())
print('dup rows',100*0+len(np.unique(X,axis=0)))
np.save(f'{sys.argv[2]}/X.npy',X)
