import numpy as np, sys, warnings
import active_subspaces as ac
d=sys.argv[1]
np.set_printoptions(linewidth=200,precision=4,suppress=True)
def rd(fn):
    with open(fn) as f: hdr=[h.strip('"') for h in f.readline().split()]
    return hdr, np.loadtxt(fn, skiprows=1, ndmin=2)
order_g=['dewmx','lai','lai0','sai','xl','dleaf','rhol_vis','rhol_nir','rhos_vis','rhos_nir','taul_vis','taul_nir','taus_vis','taus_nir','fieldcap','wiltpt','z0m','soilcol','watertable']
order_b=['z0m','soilcol','watertable']
hg,Xg=rd(f'{d}/SC_ScatterParameters_grass.txt'); Xg=Xg[:,[hg.index(n) for n in order_g]]
hb,Xb=rd(f'{d}/SC_ScatterParameters_bare.txt'); Xb=Xb[:,[hb.index(n) for n in order_b]]
Fg=np.loadtxt(f'{d}/ScatterSummary_grass.txt'); Fb=np.loadtxt(f'{d}/ScatterSummary_bare.txt')
yl=np.array([.05,1,.05,.5,-.3,0,.06,.35,.3,.5,.01,.1,.15,.3,.3,.1,1,-10]); yu=np.array([.15,4,3,4,.4,.16,.13,.55,.4,.6,.09,.45,.25,.4,1,.3,8,-1])
def norm(X, lo, hi, iz, mu, sd):
    Z=X.copy().astype(float)
    uni=[j for j in range(X.shape[1]) if j!=iz]
    Z[:,uni]=2*(X[:,uni]-lo)/(hi-lo)-1
    Z[:,iz]=(np.log(X[:,iz])-mu)/sd
    return Z
XXg=norm(Xg,yl,yu,16,-5.11,.52); XXb=norm(Xb,yl[-2:],yu[-2:],0,-8.83,1.47)
print('grass normalized min/max uniform cols', XXg[:,np.r_[0:16,17:19]].min(), XXg[:,np.r_[0:16,17:19]].max())
def legacy_ols(X,f):
    A=np.hstack([np.ones((len(f),1)),X]); u=np.linalg.lstsq(A,f,rcond=None)[0][1:]
    w=u/np.linalg.norm(u); w=w*np.sign(w[0]) if w[0]!=0 else w
    return w, u@u, u
def r2_quad(y,f):
    B=np.vander(y,3); c=np.linalg.lstsq(B,f,rcond=None)[0]; r=B@c-f
    return 1-np.sum(r**2)/(len(f)*np.var(f))
labels=['LH','SH','GH']
for name,XX,F,names in [('bare',XXb,Fb,order_b),('grass',XXg,Fg,order_g)]:
    m=XX.shape[1]
    sp=ac.Gaussian.reference(m)
    for i in range(3):
        f=F[:,i]
        wl,lam,u=legacy_ols(XX,f)
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter('always')
            sub=ac.active_subspace(X=XX,f=f,space=sp,estimator='ols',n=1)
        w1=sub.W1[:,0]
        print(f'{name} {labels[i]}: lam legacy {lam:.6g} v1 {sub.eigenvalues[0]:.6g} | |w_leg.w_v1|={abs(wl@w1):.12f} sign(w_leg.w_v1)={np.sign(wl@w1):+.0f} sign(u.w_v1)={np.sign(u@w1):+.0f} | quadR2(in-sample)={r2_quad(XX@wl,f):.2f} warns={[str(x.message)[:60] for x in W]}')
        if name=='grass' or True: print('   w_legacy', wl)
# v1 with Uniform physical space on grass uniform columns only to show the ValidityWarning
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter('always')
    sp=ac.Uniform(lower=yl, upper=yu)
    Xu=np.hstack([Xg[:,:16],Xg[:,17:]])
    sub=ac.active_subspace(X=Xu,f=Fg[:,0],space=sp,estimator='ols',n=1)
    print('uniform-only warnings:', [type(x.message).__name__+': '+str(x.message)[:150] for x in W])
# nominal
nom=np.array([.1,2,.5,4,-.3,.04,.11,.58,.36,.58,.07,.25,.22,.38,1,.1,.03,2,-1])
print('nominal outside bounds:', [(order_g[j], nom[j]) for j in range(19) if j!=16 and not (yl[j if j<16 else j-1] <= nom[j] <= yu[j if j<16 else j-1])])
print('nominal log z0m z-score', (np.log(.03)+5.11)/.52)
print('--- v1 surrogate check')
for name,XX,F in [('bare',XXb,Fb),('grass',XXg,Fg)]:
    sp=ac.Gaussian.reference(XX.shape[1])
    for i in range(3):
        f=F[:,i]; sub=ac.active_subspace(X=XX,f=f,space=sp,estimator='ols',n=1)
        s=ac.fit_surrogate(X=XX,f=f,subspace=sub,kind='poly',degree=2)
        rin=1-np.sum(s.residuals**2)/np.sum((f-f.mean())**2)
        y=XX@sub.W1[:,0]
        print(f'{name} {labels[i]} in-sample {rin:.3f} loo {s.loo_r_squared:.3f} y-range [{y.min():.2f},{y.max():.2f}]')
