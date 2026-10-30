import numpy as np, time, warnings
warnings.simplefilter('ignore')
exec(open('hiv_fd2.py').read().split('s=0; q=X[s]')[0])
def rhs_f(t,x,q):
    return [0.0]*7 if x[0] < .1 else rhs(t,x,q)
def T_ivp_f(q):
    s=solve_ivp(rhs_f,(0,times[-1]),[1000,0,0,30,0,5/.015,1],method='LSODA',t_eval=times,args=(q,),rtol=1e-10,atol=1e-10); return s.y[0]
rng=np.random.default_rng(1); S=list(rng.choice(1000,12,replace=False))+[3]
for s in S:
    q=X[s]; h=1e-4; G=np.empty((21,27))
    for j in range(27):
        e=np.zeros(27); e[j]=h*half[j]; G[:,j]=(T_ivp_f(q+e)-T_ivp_f(q-e))/(2*h)
    i=20; ship=DF[s,27*i:27*(i+1)]
    c=abs(G[i]@ship)/(np.linalg.norm(G[i])*np.linalg.norm(ship)+1e-300)
    print(f's={s} F3400={F[s,20]:.4g} |g|={np.linalg.norm(G[i]):.3g} |ship|={np.linalg.norm(ship):.3g} cos={c:.3f}; t=2800 rel={np.linalg.norm(G[19]-DF[s,27*19:27*20])/np.linalg.norm(G[19]):.1e}')
# legacy path reproduction for sample 0, t=3400, h=1e-7 (as in notebook cell 5)
t0=time.time(); p=X[:1]; hh=1e-7*half
base=Tcells(p,np.linspace(1,3400,3400))[0,-1]; g=[]
for j in range(27):
    pp=p.copy(); pp[0,j]+=hh[j]; g.append((Tcells(pp,np.linspace(1,3400,3400))[0,-1]-base)/hh[j]*half[j])
g=np.array(g); ship=DF[0,-27:]; print('legacy-path py3 repro t=3400 s0: relerr', np.linalg.norm(g-ship)/np.linalg.norm(ship), '%.1fs'%(time.time()-t0))
