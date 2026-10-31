import sys, numpy as np, active_subspaces as ac
d = sys.argv[1]
for case, nbar, tbar in [('weak', .5, .000635), ('strong', 1.75, .000735)]:
    fn = f'{d}/atacamac-{case}.txt'
    hdr = open(fn).readline().strip().split(',')
    D = np.loadtxt(fn, delimiter=',', skiprows=1)
    print(f'== {case}: shape {D.shape}, ncols hdr {len(hdr)}, nan {np.isnan(D).sum()}, header tail {hdr[38:41]} ... {hdr[-1]}')
    X, f, G = D[:,1:39], D[:,39], D[:,40:]
    n, t = X[:,:19], X[:,19:]
    print(f'  n: mean {n.mean():.4f} std {n.std(ddof=1):.4f} min {n.min():.4f} max {n.max():.4f}  (paper sd {0.1*nbar:.4f}; nb sd {nbar*.05*np.sqrt(3):.4f})')
    print(f'  tau: mean {t.mean():.3e} std {t.std(ddof=1):.3e} min {t.min():.3e} max {t.max():.3e} (paper sd {0.05*tbar:.3e}; nb sd {tbar*.025*np.sqrt(3):.3e})')
    # uniform? ratio (max-min)/sd: uniform -> 2*sqrt3=3.46 ; gaussian n=480*19 -> ~7-8
    print(f'  range/sd n {(n.max()-n.min())/n.std():.2f}, tau {(t.max()-t.min())/t.std():.2f}; kurtosis n {((n-n.mean())**4).mean()/n.var()**2:.2f} (unif 1.8, gauss 3)')
    print(f'  f: min {f.min():.3f} max {f.max():.3f} frac>0 {np.mean(f>0):.4f}')
    # gradient scaling test: OLS slopes vs mean gradients
    A = np.hstack([np.ones((len(f),1)), X]); coef = np.linalg.lstsq(A, f, rcond=None)[0][1:]
    gm = G.mean(0); sd = X.std(0, ddof=1)
    r_phys = np.corrcoef(coef, gm)[0,1]; 
    print(f'  median ratio meanG/OLS: n {np.median(gm[:19]/coef[:19]):.3e} tau {np.median(gm[19:]/coef[19:]):.3e}; sd n {sd[:19].mean():.3e} sd tau {sd[19:].mean():.3e}')
    a = n*np.cos(np.pi*832/.54*t)
    print(f'  a=Re(F) sign: frac a<0 {np.mean(a<0):.3f}')
