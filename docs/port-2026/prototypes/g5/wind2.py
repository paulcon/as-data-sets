import numpy as np, time, warnings
import active_subspaces as ac
d=np.load('wind.npy'); X=d[:,1:6]; F=d[:,6:]
names=['Vhub','Hs','WaveDir','Tp','PlatformDir']
lo=np.minimum(X.min(0),-1); hi=np.maximum(X.max(0),1)
space=ac.Uniform(lower=lo,upper=hi,names=names)
legacy=[[0.9745,0.9751,0.9825],[0.9693,0.972,0.975],[0.732,0.868,0.9403],[0.7958,0.903,0.92],[0.209,0.7271,0.7456],[0.761,0.8628,0.8719],[0.7712,0.8483,0.8586]]
parts=[1,1,3,3,2,2,2]
rng=np.random.default_rng(2016)
T=time.time()
for i in range(7):
    cen=rng.choice(len(X),2000,replace=False)
    g=ac.local_linear_gradients(X=X,f=F[:,i],space=space,p=60,centers=cen)
    s=ac.active_subspace(grads=g,n=parts[i])
    r=[]
    for deg in (3,4,5):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            sur=ac.fit_surrogate(X=X,f=F[:,i],subspace=s,kind='poly',degree=deg)
        pred=ac.predict(sur,X); ins=1-np.sum((F[:,i]-pred)**2)/np.sum((F[:,i]-F[:,i].mean())**2)
        r.append((round(ins,4),round(sur.loo_r_squared,4)))
    print(i,'legacy',legacy[i],' v1 (in-sample, LOO)',r, 'warns',len(w))
print('total %.1fs'%(time.time()-T))
