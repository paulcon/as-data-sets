import numpy as np, sys
d=sys.argv[1]
def rd(fn):
    with open(fn) as f: hdr=[h.strip('"') for h in f.readline().split()]
    return hdr, np.loadtxt(fn, skiprows=1, ndmin=2)
order_g=['dewmx','lai','lai0','sai','xl','dleaf','rhol_vis','rhol_nir','rhos_vis','rhos_nir','taul_vis','taul_nir','taus_vis','taus_nir','fieldcap','wiltpt','z0m','soilcol','watertable']
hg,Xg=rd(f'{d}/SC_ScatterParameters_grass.txt'); Xg=Xg[:,[hg.index(n) for n in order_g]]
F=np.loadtxt(f'{d}/ScatterSummary_grass.txt')
yl=np.array([.05,1,.05,.5,-.3,0,.06,.35,.3,.5,.01,.1,.15,.3,.3,.1,1,-10]); yu=np.array([.15,4,3,4,.4,.16,.13,.55,.4,.6,.09,.45,.25,.4,1,.3,8,-1])
def norm(X):
    Z=X.astype(float).copy(); u=[j for j in range(19) if j!=16]
    Z[:,u]=2*(X[:,u]-yl)/(yu-yl)-1; Z[:,16]=(np.log(X[:,16])+5.11)/.52; return Z
XX=norm(Xg)
for i,lab in enumerate(['LH','SH','GH']):
    f=F[:,i]; A=np.hstack([np.ones((300,1)),XX]); a=np.linalg.lstsq(A,f,rcond=None)[0][1:]; w=a/np.linalg.norm(a)
    print(lab,'lai weight (a-orient)',round(w[1],3),' top', [order_g[j]+f'{w[j]:+.2f}' for j in np.argsort(-abs(w))[:6]])
    if i==0:
        y=XX@w; c=np.polyfit(y,f,2)
        for wt in (-1,):
            nom=np.array([[.1,2,.5,4,-.3,.04,.11,.58,.36,.58,.07,.25,.22,.38,1,.1,.03,2,wt]])
            yn=(norm(nom)@w)[0]; fn=np.polyval(c,yn)
            res=[]
            for lai in (1,3):
                n2=nom.copy(); n2[0,1]=lai; res.append(np.polyval(c,(norm(n2)@w)[0])/fn-1)
            print(f'  wt={wt}: y_nom={yn:.3f}  lai-1: {100*res[0]:+.1f}%  lai+1: {100*res[1]:+.1f}%')
        # with clipping rhol_nir to .55
