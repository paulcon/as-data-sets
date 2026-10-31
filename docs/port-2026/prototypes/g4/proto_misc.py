import numpy as np, warnings, matplotlib; matplotlib.use('Agg')
import active_subspaces as ac, active_subspaces.plotting as acp
rng=np.random.default_rng(0)
X=rng.standard_normal((200,5)); f=np.maximum(X[:,0],0)
sp=ac.Gaussian.reference(5)
sub=ac.active_subspace(X=X,f=f,space=sp,estimator='ols',n=1)
with np.errstate(divide='ignore'):
    lf=np.log10(f)
try:
    acp.plot_sufficient_summary(sub,X=X,f=lf); print('log10 with -inf: OK')
except Exception as e: print('log10 with -inf ->', type(e).__name__, getattr(e,'code',None), str(e)[:150])
s=ac.fit_surrogate(X=X,f=f,subspace=sub,kind='poly',degree=2)
print('poly attrs', [a for a in dir(s) if not a.startswith('_')][:40])
print('W1 shape',sub.W1.shape, 'eigs', sub.eigenvalues[:3])
# Uniform.reference with gaussian column
try:
    ac.active_subspace(X=X,f=f,space=ac.Uniform.reference(5),estimator='ols',n=1)
except Exception as e: print('Uniform.reference with |x|>1 ->', type(e).__name__, getattr(e,'code',None))
print(ac.__version__)
