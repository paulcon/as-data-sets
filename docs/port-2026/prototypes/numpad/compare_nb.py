import sys; sys.path.insert(0,'.')
import numpy as np, scipy.io as IO
from pde_model import PDESolver, PDEMisfit
R='/Users/paulconstantine/PycharmProjects/as-data-sets'
rng=np.random.default_rng(7)
t=np.array([-3,-1.5,0,1.5,3.])
for case in ['long','short']:
    S=PDESolver(case, datadir=R+'/Elliptic_PDE')
    print(f'== {case}: q(0) = {S.qoi(np.zeros(100)):.5f}')
    X=rng.standard_normal((300,100)); G=np.array([S.grad_qoi(x) for x in X]); Q=np.array([S.qoi(x) for x in X])
    C=G.T@G/300; e,W=np.linalg.eigh(C); e,W=e[::-1],W[:,::-1]
    W=W*np.sign(W[0]) 
    print('  eigs[:5]', np.array2string(e[:5],precision=3), ' Q range', f'{Q.min():.4f}..{Q.max():.4f}')
    print('  w1[:8]', np.array2string(W[:8,0],precision=3)); print('  w2[:8]', np.array2string(W[:8,1],precision=3))
    k=np.argmax(np.abs(W[:,0])); print('  argmax|w1| =',k)
    for kk in sorted({k,0}):
        print(f'  q(t e_{kk}) t=-3..3:', np.array2string(np.array([S.qoi(tt*np.eye(100)[kk]) for tt in t]),precision=4))
    print('  q(t*w1) t=-3..3:', np.array2string(np.array([S.qoi(tt*W[:,0]) for tt in t]),precision=4), '(sign of w1 arbitrary)')
# MCMC Gradient_Samples
df=IO.loadmat(R+'/uq_benchmarks/PDE_Inverse_Problem_MCMC/Gradient_Samples.mat')['Samples']
e0,W0=np.linalg.eigh(df.T@df/df.shape[0]); e0,W0=e0[::-1],W0[:,::-1]
print('== stored Gradient_Samples eigs[:6]', np.array2string(e0[:6],precision=3), ' mean|g|', f'{np.linalg.norm(df,axis=1).mean():.2f}')
M=PDEMisfit('short', datadir=R+'/MCMC')
for trial in range(2):
    xt=rng.standard_normal(100); obs=M.observations(xt); sig2=1e-4*np.linalg.norm(obs)**2
    M.set_data(obs+np.sqrt(sig2)*rng.standard_normal(7), sig2)
    G=np.array([M.misfit_grad(x) for x in rng.standard_normal((500,100))])
    e,W=np.linalg.eigh(G.T@G/500); e,W=e[::-1],W[:,::-1]
    print(f'  synthetic data #{trial}: eigs[:6]', np.array2string(e[:6],precision=3), ' mean|g|', f'{np.linalg.norm(G,axis=1).mean():.2f}',
          ' |cos(w1,w1_stored)|', f'{abs(W[:,0]@W0[:,0]):.3f}', ' subspace(2) dist', f'{np.linalg.norm(W[:,:2]@W[:,:2].T-W0[:,:2]@W0[:,:2].T,2):.3f}')
