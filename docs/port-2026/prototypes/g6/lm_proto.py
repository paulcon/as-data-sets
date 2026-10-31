import sys, numpy as np, active_subspaces as ac
np.set_printoptions(precision=3, linewidth=160)
fn = sys.argv[1]
hdr = open(fn).readline().strip().split(',')
D = np.loadtxt(fn, delimiter=',', skiprows=1)
print('shape', D.shape, 'hdr', len(hdr), 'nan', np.isnan(D).sum(), 'run col first/last', D[0,0], D[-1,0], 'unique runs', len(np.unique(D[:,0])))
print('outputs hdr', hdr[106:109], 'grad hdr samples', hdr[109], hdr[213], hdr[214], hdr[318], hdr[319], hdr[-1])
X = D[:,1:106]; F = D[:,106:109]; GL, GD, GA = D[:,109:214], D[:,214:319], D[:,319:]
print('X min/max', X.min(), X.max(), 'cols with any nonzero', np.sum(np.any(X!=0,0)), 'row0 all zero', np.all(X[0]==0))
print('X per-col max |x| quantiles', np.quantile(np.abs(X).max(0), [0, .5, 1]))
print('F min/max', F.min(0), F.max(0))
print('grad cols all-zero: lift', np.sum(np.all(GL==0,0)), 'drag', np.sum(np.all(GD==0,0)), 'ea', np.sum(np.all(GA==0,0)))
dup = len(np.unique(X.round(12), axis=0)); print('unique X rows', dup)
for lab, f, G in [('lift', F[:,0], GL), ('drag', F[:,1], GD), ('ea', F[:,2], GA)]:
    A = np.hstack([np.ones((len(f),1)), X]); coef = np.linalg.lstsq(A, f, rcond=None)[0][1:]
    print(f'{lab}: corr(OLS coef, meanG) {np.corrcoef(coef, G.mean(0))[0,1]:.3f}, median ratio {np.median(G.mean(0)/coef):.3f}')
    sp = ac.Uniform(lower=-0.3*np.ones(105), upper=0.3*np.ones(105))
    try:
        s = ac.active_subspace(space=sp, X=X, grads=G, nboot=0)
        e = np.asarray(s.eigenvalues); print('   eig[:8]/eig0', e[:8]/e[0], 'auto n', s.n)
    except Exception as ex: print('   v1 error:', type(ex).__name__, str(ex)[:300])
