import numpy as np
from scipy.special import lambertw
from scipy.optimize import minimize_scalar, brentq
def pmax(p, T=25.0, Vth=None):
    Isc,Is,n,Rs,Rp=p
    if Vth is None: Vth=1.380649e-23*(273.15+T)/1.602176634e-19
    a=n*Vth
    IL=Isc+Is*np.expm1(Isc*Rs/a)+Isc*Rs/Rp
    def I(V):
        arg=np.log(Rs*Rp*Is/(a*(Rs+Rp)))+Rp*(Rs*(IL+Is)+V)/(a*(Rs+Rp))
        W=lambertw(np.exp(arg)).real if arg<600 else arg-np.log(arg)
        return (Rp*(IL+Is)-V)/(Rs+Rp)-(a/Rs)*W
    Voc=brentq(I,0,5)
    r=minimize_scalar(lambda V:-V*I(V),bounds=(0,Voc),method='bounded',options={'xatol':1e-12})
    return -r.fun
xl=np.array([0.05989,np.log(2.2e-11),1.0,0.16625,93.75]); xu=np.array([0.23958,np.log(2.2e-7),2.0,0.665,375.0])
def phys(x):
    q=xl+(x+1)*(xu-xl)/2; q=q.copy(); q[1]=np.exp(q[1]); return q
def norm(p):
    q=np.array(p,float); q[1]=np.log(q[1]); return 2*(q-xl)/(xu-xl)-1
if __name__=='__main__':
    d=np.load('pv.npy'); X=d[:,1:6]; f=d[:,6]; G=d[:,7:]
    np.set_printoptions(linewidth=160,precision=5)
    for Vth in [None, 0.025852, 0.02569]:
        errs=[]; gerr=[]
        for i in range(8):
            fi=pmax(X[i],Vth=Vth); errs.append(fi/f[i]-1)
            x=norm(X[i]); h=1e-6; g=np.zeros(5)
            for j in range(5):
                e=np.zeros(5); e[j]=h; g[j]=(pmax(phys(x+e),Vth=Vth)-pmax(phys(x-e),Vth=Vth))/(2*h)
            gerr.append(g/G[i]-1)
        print('Vth',Vth,'rel err f',np.array(errs)); print(' rel err grad (rows 0..3)\n',np.array(gerr)[:4])
