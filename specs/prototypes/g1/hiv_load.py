import numpy as np, time
import active_subspaces as ac
d='/Users/paulconstantine/PycharmProjects/as-data-sets/HIV/'
t0=time.time()
X=np.loadtxt(d+'inputs.csv',delimiter=',',skiprows=1)[:,1:]
F=np.loadtxt(d+'outputs.csv',delimiter=',',skiprows=1)[:,1:]
DF=np.loadtxt(d+'gradients.csv',delimiter=',',skiprows=1)[:,1:]
print('load s',time.time()-t0, X.shape,F.shape,DF.shape)
idx=np.loadtxt(d+'inputs.csv',delimiter=',',skiprows=1)[:,0]; print('index 0..999?', np.all(idx==np.arange(1000)))
nominal = np.array([10, .15, 5, .2, 55.6, 3.87e-3, 1e-6, 4.5e-4, 7.45e-4, 5.22e-4, 3e-6, 3.3e-4, 6e-9, .537, .285, 7.79e-6, 1e-6, 4e-5, .01, .28, .05, .005, .005, .015, 2.39, 3e-4, .97])
xl=.975*nominal; xu=1.025*nominal
XX=2*(X-xl)/(xu-xl)-1; print('XX range', XX.min(), XX.max())
print('nan?', np.isnan(DF).sum(), np.isnan(F).sum(), 'F range', F.min(), F.max())
print('digits in inputs.csv line:', open(d+'inputs.csv').readlines()[1][:80])
times = np.array([5, 15, 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 750, 1000, 1600, 1800, 2000, 2200, 2400, 2800, 3400])
labels=['s1','s2','s3','p1','C1','K1','K2','K3','K4','K5','K6','K7','K8','K9','K10','K11','K12','K13','d1','d2','d3','d4','d5','d6','d7','a1','psi']
space=ac.Uniform(lower=xl,upper=xu,names=labels)
ref=ac.Uniform.reference(27)
np.set_printoptions(precision=3, linewidth=150)
for i in [2, 20]:
    G=DF[:,27*i:27*(i+1)]
    sub=ac.active_subspace(space=ref, grads=G, nboot=100, rng=0)
    # equivalently physical grads
    sub_p=ac.active_subspace(space=space, grads=G/((xu-xl)/2))
    print('t=',times[i],'eig[:5]',sub.eigenvalues[:5],'n=',sub.n, 'same as physical?', np.allclose(sub.eigenvalues, sub_p.eigenvalues))
    w=sub.W[:,0]; top=np.argsort(-abs(w))[:6]; print('  top w1', [(labels[j], round(w[j],3)) for j in top])
    y=XX@w; print('  corr(f, y1)', np.corrcoef(F[:,i], y)[0,1])
# small-sample check: first 20 rows
for i in [2]:
    sub=ac.active_subspace(space=ref, grads=DF[:20,27*i:27*(i+1)]); w=sub.W[:,0]; top=np.argsort(-abs(w))[:5]; print('N=20 t=24 top', [(labels[j], round(w[j],3)) for j in top], sub.eigenvalues[:3])
print('samples with T<=0.1001 at each time:', [(t,int((F[:,i]<0.1001).sum())) for i,t in enumerate(times)])
print('zero-gradient rows per time:', [int((np.abs(DF[:,27*i:27*(i+1)]).max(1)==0).sum()) for i in range(21)])
