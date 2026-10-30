import numpy as np, time, warnings
warnings.simplefilter('ignore')
exec(open('hiv_fd2.py').read().split('s=0; q=X[s]')[0])
def ev(t,x,q): return x[0]-0.1
ev.terminal=True; ev.direction=-1
def T_ev(q):
    s=solve_ivp(rhs,(0,times[-1]),[1000,0,0,30,0,5/.015,1],method='LSODA',t_eval=times,args=(q,),rtol=1e-10,atol=1e-10,events=ev)
    out=np.full(len(times),0.1); out[:len(s.t)]=s.y[0]; return out, (s.t_events[0][0] if len(s.t_events[0]) else np.nan)
fro=np.where(F[:,-1]<0.1001)[0]
for s in fro[:4]:
    t0=time.time(); q=X[s]; h=1e-4; G=np.empty((21,27)); base,tc=T_ev(q)
    for j in range(27):
        e=np.zeros(27); e[j]=h*half[j]; G[:,j]=(T_ev(q+e)[0]-T_ev(q-e)[0])/(2*h)
    i=20; ship=DF[s,27*i:27*(i+1)]
    print(f's={s} crossing t={tc:.1f} F3400 shipped={F[s,20]:.6g} |g_event|={np.linalg.norm(G[i]):.3g} |ship|={np.linalg.norm(ship):.3g} ship={np.round(ship[[5,8,13,21,24]],2)} ({time.time()-t0:.0f}s)', flush=True)
