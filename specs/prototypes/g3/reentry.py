import sys, warnings, numpy as np, active_subspaces as ac
np.set_printoptions(precision=4, suppress=False, linewidth=160)
d = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
lb, ub = np.array([16.3, 13.7, 0.001]), np.array([24.3, 17.3, 0.002]); sd = np.array([.12, .12, .10, .10])
X = np.hstack([2*(d[:, 1:4]-lb)/(ub-lb)-1, d[:, 4:8]/sd])
pst, qst = d[:, 8], d[:, 9]
names = ["p_inf","M_inf","gamma","r1","r2","r3","r4"]
def quad_C(X, f, Sigma):
    M, m = X.shape; ju, ku = np.triu_indices(m, 1)
    A = np.hstack([np.ones((M,1)), X, X*X, X[:,ju]*X[:,ku]])
    c = np.linalg.lstsq(A, f, rcond=None)[0]
    g = c[1:m+1]; H = np.diag(2*c[m+1:2*m+1]); H[ju,ku] = c[2*m+1:]; H[ku,ju] = c[2*m+1:]
    resid = f - A@c; R2 = 1 - resid@resid/np.sum((f-f.mean())**2)
    C = np.outer(g,g) + H@Sigma@H
    e, W = np.linalg.eigh(C); e, W = e[::-1], W[:, ::-1]
    W = W*np.sign(W[np.argmax(abs(W),0), range(m)]); return e, W, R2
ref = ac.Gaussian(np.zeros(7), np.ones(7), names=names)
for nm, f in [("p_st", pst), ("q_st", qst)]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = ac.active_subspace(X=X, f=f, space=ref, estimator="qphd", n=2, qoi=nm)
    eL, WL, R2 = quad_C(X, f, np.eye(7))                         # = legacy (gamma=1 everywhere)
    eC, WC, _ = quad_C(X, f, np.diag([1/3]*3+[1]*4))              # correct mixed second moment
    print(nm, "quadratic fit R2", round(R2,6), "warnings", [str(x.message)[:60] for x in w])
    print("  v1 Gaussian.reference eig", s.eigenvalues)
    print("  numpy Sigma=I        eig", eL, " max|dW1:2|", np.abs(abs(s.W[:,:2])-abs(WL[:,:2])).max())
    print("  numpy mixed Sigma    eig", eC)
    print("  W1 (Sigma=I)  ", WL[:,0], "\n  W1 (mixed)    ", WC[:,0])
    print("  W2 (Sigma=I)  ", WL[:,1], "\n  W2 (mixed)    ", WC[:,1])
    # OLS check
    u = np.linalg.lstsq(np.hstack([np.ones((len(f),1)),X]), f, rcond=None)[0][1:]; print("  OLS dir", u/np.linalg.norm(u))
