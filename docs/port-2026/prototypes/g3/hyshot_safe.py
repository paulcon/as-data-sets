import sys, numpy as np, scipy.optimize as op
np.set_printoptions(precision=4, suppress=True, linewidth=150)
D=sys.argv[1]
xl = np.array([2.6, 0.1, 122.5, 1.6448e7, 3.0551e6, 0.03, 0.087]); xu = np.array([4.6, 1.9, 367.5, 1.9012e7, 3.4280e6, 0.07, 0.203])
d35 = np.loadtxt(f"{D}/HyShotII35.txt", delimiter=",", skiprows=1); d30 = np.loadtxt(f"{D}/HyShotII30.txt", delimiter=",")
Z35 = d35[:,1:8].copy(); Z35[:-2] = 2*(Z35[:-2]-xl)/(xu-xl)-1; f35=d35[:,9]
Z30 = d30[:,:7]; f30 = d30[:,7]
def ols(Z,f):
    u=np.linalg.lstsq(np.hstack([np.ones((len(f),1)),Z]),f,rcond=None)[0][1:]; return u/np.linalg.norm(u)
def analyse(Z,f,w,intdiv, label, exact=False):
    y=Z@w; M=len(y)
    V=np.vander(y,3,increasing=True); c=np.linalg.lstsq(V,f,rcond=None)[0]
    pred=lambda t: np.vander(t,3,increasing=True)@c
    sf=np.sqrt(np.sum((f-pred(y))**2)/(M-3)); sy=np.sum((y-y.mean())**2)
    R2=1-np.sum((f-pred(y))**2)/np.sum((f-f.mean())**2)
    oneM = 0 if intdiv else 1/M
    if exact:
        AtAi=np.linalg.inv(V.T@V)
        pl=lambda t: pred(t)+2.33*sf*np.sqrt(1+np.einsum('ij,jk,ik->i',np.vander(t,3,increasing=True),AtAi,np.vander(t,3,increasing=True)))
    else:
        pl=lambda t: pred(t)+2.33*sf*np.sqrt(1+oneM+(t-y.mean())**2/sy)
    t=np.linspace(-2,3,800); ymax=t[np.argmin(abs(pl(t)-2.8))]
    ybox=np.sum(np.abs(w))
    xmin=Z[-2]
    res=op.minimize(lambda x:-np.sum(np.log(np.abs(x-xmin))), Z[-1], bounds=[(-1,1)]*7, constraints={'type':'ineq','fun':lambda x: ymax-x@w})
    A=np.sort(np.vstack([xl+(xu-xl)*(xmin+1)/2, xl+(xu-xl)*(res.x+1)/2]).T,axis=1)
    print(f"{label}: R2={R2:.4f} ymax={ymax:.4f} (box max w^Tx={ybox:.4f}) pl at y-range ends {pl(np.array([-ybox,ybox]))}")
    print("   AoA", A[0], " TI", A[1], " H0", A[4])
for nm,Z,f in [("0.30",Z30,f30),("0.35",Z35,f35)]:
    w=ols(Z[:-2],f[:-2])
    analyse(Z,f,w,True,nm+" py2 1/M=0")
    analyse(Z,f,w,False,nm+" py3 1/M")
    analyse(Z,f,w,False,nm+" exact quad PI",exact=True)
print("--- paper Table 3 w for 0.35")
wp=np.array([0.7066,0.5008,0.0289,0.2051,-0.4490,-0.0591,0.0432]); wp/=np.linalg.norm(wp)
analyse(Z35,f35,wp,False,"0.35 paper w")
# which variants reproduce paper w35?
for nm,Zv in [("with extremes",Z35),("no extremes",Z35[:-2])]:
    print(nm, ols(Zv,f35[:len(Zv)]))
# 0.30 outputs in 35 file
print("0.30 col in 35 file", d35[:,8])
