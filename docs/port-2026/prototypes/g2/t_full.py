import numpy as np, time, warnings, active_subspaces as ac
warnings.simplefilter('ignore')
rng=np.random.default_rng(0); X=rng.standard_normal((1000,100)); f=np.exp(0.3*X[:,0])+0.01*X[:,1:].sum(1)
t=time.time(); s=ac.fit_surrogate(X=X,f=f,space=ac.Gaussian.reference(100),subspace=None,kind='rbf',degree=1); print('full rbf fit', round(time.time()-t,1),'s', 'loo R2', s.loo_r_squared)
