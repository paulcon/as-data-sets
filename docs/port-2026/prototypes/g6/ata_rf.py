import sys, numpy as np, active_subspaces as ac
d = sys.argv[1]; w0 = np.pi*832/.54; rng = np.random.default_rng(42)
for case, nbar, tbar in [('weak', .5, .000635), ('strong', 1.75, .000735)]:
    D = np.loadtxt(f'{d}/atacamac-{case}.txt', delimiter=',', skiprows=1)
    X, f, G = D[:,1:39], D[:,39], D[:,40:]
    sd = np.r_[np.full(19, nbar*.05*np.sqrt(3)), np.full(19, tbar*.025*np.sqrt(3))]; mu = np.r_[np.full(19, nbar), np.full(19, tbar)]
    def cov(X):
        n, t = X[:,:19], X[:,19:]; return np.hstack([n*np.cos(w0*t), n*np.sin(w0*t)])
    Gp_ = G/sd; n, t = X[:,:19], X[:,19:]; a, b = n*np.cos(w0*t), n*np.sin(w0*t); r = np.hypot(a,b)
    fn, ft = Gp_[:,:19], Gp_[:,19:]
    Xp = cov(X); m0, s0 = Xp.mean(0), Xp.std(0, ddof=1); XXp = (Xp-m0)/s0
    res = {}
    for lab, Ga in [('ok', fn*a/r - ft*b/(w0*r**2)), ('legacy', fn*np.abs(a)/r - ft*b/(w0*r**2))]:
        Gp = np.hstack([Ga, fn*b/r + ft*a/(w0*r**2)])*s0
        s = ac.active_subspace(space=ac.Gaussian.reference(38), X=XXp, grads=Gp, n=3)
        sur = ac.fit_surrogate(X=XXp, f=f, space=ac.Gaussian.reference(38), subspace=s, kind='poly', degree=3)
        Xmc = mu + sd*rng.standard_normal((50000, 38))
        Y = ((cov(Xmc)-m0)/s0) @ np.asarray(s.W1)
        ft_ = np.asarray(sur.evaluate(Y)).ravel()
        res[lab] = 100*np.mean(ft_ > 0)
    print(f'{case}: RF cubic-surrogate MC  correct={res["ok"]:.2f}%  legacy-chain-rule={res["legacy"]:.2f}%  (legacy notebook printed weak 9.392 / strong 27.16; paper MC reference 14.77 / 39.55; data frac {100*np.mean(f>0):.2f}%)')
