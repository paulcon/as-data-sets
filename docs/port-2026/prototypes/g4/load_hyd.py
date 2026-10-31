import numpy as np, sys
d=sys.argv[1]
def rd(fn):
    with open(fn) as f: hdr=[h.strip('"') for h in f.readline().split()]
    return hdr, np.loadtxt(fn, skiprows=1)
for c in ['bare','grass']:
    h,X=rd(f'{d}/SC_ScatterParameters_{c}.txt'); F=np.loadtxt(f'{d}/ScatterSummary_{c}.txt')
    print(c,X.shape,F.shape,h)
    np.set_printoptions(linewidth=200,precision=4,suppress=True)
    print(' min',X.min(0)); print(' max',X.max(0))
    if 'z0m' in h:
        z=np.log(X[:,h.index('z0m')]); print(' log z0m mean/std',z.mean(),z.std(ddof=1), 'min/max', z.min(), z.max())
    print(' F min',F.min(0)); print(' F max',F.max(0)); print(' F mean', F.mean(0))
    print(' unique per col', [len(np.unique(X[:,j])) for j in range(X.shape[1])])
