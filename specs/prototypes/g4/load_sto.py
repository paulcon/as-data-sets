import numpy as np, sys
d=sys.argv[1]
np.set_printoptions(linewidth=220,precision=5,suppress=False)
def rd(fn):
    with open(fn) as f: hdr=[h.strip('"') for h in f.readline().split()]
    return hdr, np.loadtxt(fn, skiprows=1, ndmin=2)
h,X=rd(f'{d}/stomata_realization_inputs.txt'); print('X',X.shape,h)
hb,B=rd(f'{d}/param_ranges.txt'); print('bounds',B.shape,hb)
for j,n in enumerate(h): print(f'{n:8s} lb={B[j,0]:<10g} ub={B[j,1]:<10g} min={X[:,j].min():<10g} max={X[:,j].max():<10g} uniq={len(np.unique(X[:,j]))}')
print('rows all-equal dup?', len(np.unique(X,axis=0)))
for fn in ['mo_avg_rssha','mo_avg_rssun','mo_avg_tran_veg','JJA_hr_avg_rssha','JJA_hr_avg_rssun','JJA_hr_avg_tran_veg']:
    hh,F=rd(f'{d}/Tonzi_Stomata_SC.{fn}.txt')
    print(fn,F.shape,hh[:3],'... min',F.min(),'max',F.max(),'nan',np.isnan(F).sum(), 'nonpos',(F<=0).sum())
