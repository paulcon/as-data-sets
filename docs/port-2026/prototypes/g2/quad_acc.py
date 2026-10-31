import numpy as np
Q=.5*np.array([[np.sqrt(2),np.sqrt(2)],[-np.sqrt(2),np.sqrt(2)]]); A=Q@np.diag([1,.01])@Q.T; d=.9
for sig2 in [0.01,0.1]:
    rng=np.random.default_rng(0)
    f=lambda x:(d-.5*x@A@x)**2/(2*sig2)
    lp=lambda x:-f(x)-.5*x@x
    x=np.array([1.,1.]); l=lp(x); acc=0; N=200000
    for i in range(N):
        xp=x+np.sqrt(.5)*rng.standard_normal(2); lpp=lp(xp)
        if np.log(rng.uniform())<=lpp-l: x,l=xp,lpp; acc+=1
    print('sigma^2=',sig2,'vanilla acceptance',acc/N)
