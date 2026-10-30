import sys, numpy as np, active_subspaces as ac
np.set_printoptions(precision=3, linewidth=150)
d = sys.argv[1]; w0 = np.pi*832/.54
def r2(f, fh): return 1 - np.sum((f-fh)**2)/np.sum((f-f.mean())**2)
for case, nbar, tbar in [('weak', .5, .000635), ('strong', 1.75, .000735)]:
    D = np.loadtxt(f'{d}/atacamac-{case}.txt', delimiter=',', skiprows=1)
    X, f, G = D[:,1:39], D[:,39], D[:,40:]
    sd = np.r_[np.full(19, nbar*.05*np.sqrt(3)), np.full(19, tbar*.025*np.sqrt(3))]
    mu = np.r_[np.full(19, nbar), np.full(19, tbar)]
    sp = ac.Gaussian(mean=mu, cov=sd**2)
    Gphys = G/sd   # stored G is w.r.t. standardized inputs (likely)
    sub = ac.active_subspace(space=sp, X=X, grads=Gphys, nboot=100, rng=0)
    sub_ref = ac.active_subspace(space=ac.Gaussian.reference(38), X=(X-mu)/sd, grads=G, nboot=0)
    ev = np.asarray(sub.eigenvalues)
    print(f'== {case}: orig coords eig[:7]/eig0 {ev[:7]/ev[0]}  (ref-space path identical: {np.allclose(ev, sub_ref.eigenvalues)})  auto n={sub.n}')
    print('   w1 n-part |.| mean', np.abs(np.asarray(sub.W1)[:19,0]).mean().round(3), 'tau-part', np.abs(np.asarray(sub.W1)[19:,0]).mean().round(3))
    n, t = X[:,:19], X[:,19:]
    a, b = n*np.cos(w0*t), n*np.sin(w0*t); r = np.hypot(a, b)
    fn, ft = Gphys[:,:19], Gphys[:,19:]
    Ga_ok = fn*a/r - ft*b/(w0*r**2);   Gb = fn*b/r + ft*a/(w0*r**2)
    Ga_nb = fn*np.abs(a)/r - ft*b/(w0*r**2)   # legacy notebook: sqrt(1+(b/a)^2)^-1 = |a|/r
    Xp = np.hstack([a, b]); sdp = Xp.std(0, ddof=1); XXp = (Xp-Xp.mean(0))/sdp
    for lab, Ga in [('correct chain rule', Ga_ok), ('legacy chain rule', Ga_nb)]:
        Gp = np.hstack([Ga, Gb])*sdp
        s = ac.active_subspace(space=ac.Gaussian.reference(38), X=XXp, grads=Gp, n=3)
        e = np.asarray(s.eigenvalues)
        out = [lab, 'eig[:6]/eig0', e[:6]/e[0]]
        for deg in (1, 2, 3):
            sur = ac.fit_surrogate(X=XXp, f=f, space=ac.Gaussian.reference(38), subspace=s, kind='poly', degree=deg)
            out.append(f'R2 deg{deg}={1-np.sum(np.asarray(sur.residuals)**2)/np.sum((f-f.mean())**2):.3f}')
        print('  ', *out)
    full = ac.fit_surrogate(X=XXp, f=f, space=ac.Gaussian.reference(38), subspace=None, kind='poly', degree=1)
    print(f'   full linear R2 {1-np.sum(np.asarray(full.residuals)**2)/np.sum((f-f.mean())**2):.3f}; data frac f>0 {np.mean(f>0):.4f}')
