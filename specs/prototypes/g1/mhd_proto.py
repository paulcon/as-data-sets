import numpy as np, itertools, time, warnings
from scipy.special import roots_legendre
import active_subspaces as ac
import mhd_gradients_copy as g
np.set_printoptions(precision=4, linewidth=150)
lb,ub=g.lb,g.ub
rng=np.random.default_rng(0); x=lb+(ub-lb)*rng.random((5,5))
for fn,gfn in [(g.uavg,g.uavg_grad),(g.Bind,g.Bind_grad)]:
    G=gfn(x); err=[]
    for j in range(5):
        e=np.zeros(5); e[j]=1e-6; fd=(fn(x+e)-fn(x-e))/2e-6*(ub[j]-lb[j])/2
        err.append(np.max(abs(fd-G[:,j])/np.max(abs(G))))
    print(fn.__name__,'grad FD max rel err per input', np.array(err))
t,w=roots_legendre(11); w=w/2
T=np.array(list(itertools.product(t,repeat=5))); W=np.prod(np.array(list(itertools.product(w,repeat=5))),1)
X=lb+(ub-lb)/2*(T+1)
names=['log_mu','log_rho','log_dpdx','log_eta','log_B0']
space=ac.Uniform(lower=lb,upper=ub,names=names)
for fn,gfn in [(g.uavg_grad,'u'),(g.Bind_grad,'B')]:
    Gn=fn(X); Gp=Gn/((ub-lb)/2)  # grad wrt log-inputs
    t0=time.time(); sub=ac.active_subspace(space=space,grads=Gp,quad_weights=W); 
    print(gfn,'eig',sub.eigenvalues,'n',sub.n,'legacy(x2^-5)',sub.eigenvalues[:2]/32, '%.2fs'%(time.time()-t0)); print('  W1,W2', sub.W[:,0], sub.W[:,1])
# generator
d='/Users/paulconstantine/PycharmProjects/as-data-sets/MHD/'
lbg=np.log([.001,.1,.1,.1,.1]); ubg=np.log([.01,10,.5,10,1])
spg=ac.Uniform(lower=lbg,upper=ubg,names=names)
for fn in ['Uavg','Bind']:
    D=np.loadtxt(d+f'MHD_Generator_Data_{fn}.txt',delimiter=',',skiprows=1)[:,1:]
    Xp,f,df=D[:,:5],D[:,5],D[:,6:]
    print(fn, D.shape, 'X in range?', np.all(np.log(Xp)>=lbg-1e-12), np.all(np.log(Xp)<=ubg+1e-12), 'f range',f.min(),f.max())
    sub=ac.active_subspace(space=spg,X=np.log(Xp),grads=Xp*df,nboot=500,rng=0)
    print('  eig',sub.eigenvalues,'n',sub.n,' rel',sub.eigenvalues/sub.eigenvalues.sum()); print('  w1',sub.W[:,0],' w2',sub.W[:,1])
    print('  minmax band λ1', ac.bootstrap_interval(abs(sub.bootstrap_eigenvalues),kind='minmax')[:,0])
Du=np.loadtxt(d+'MHD_Generator_Data_Uavg.txt',delimiter=',',skiprows=1); Db=np.loadtxt(d+'MHD_Generator_Data_Bind.txt',delimiter=',',skiprows=1)
print('same X in both files?', np.array_equal(Du[:,1:6],Db[:,1:6]))
