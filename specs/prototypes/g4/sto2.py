import numpy as np, sys
d=sys.argv[1]
def rd(fn):
    with open(fn) as f: hdr=[h.strip('"') for h in f.readline().split()]
    return hdr, np.loadtxt(fn, skiprows=1, ndmin=2)
names,X=rd(f'{d}/stomata_realization_inputs.txt'); _,B=rd(f'{d}/param_ranges.txt')
XX=2*(X-B[:,0])/(B[:,1]-B[:,0])-1
A=np.hstack([np.ones((301,1)),XX])
mx=0
for q in ['mo_avg_tran_veg','mo_avg_rssha','mo_avg_rssun','JJA_hr_avg_tran_veg','JJA_hr_avg_rssha','JJA_hr_avg_rssun']:
    h,F=rd(f'{d}/Tonzi_Stomata_SC.{q}.txt')
    W=[]
    for i in range(F.shape[1]):
        a=np.linalg.lstsq(A,F[:,i],rcond=None)[0][1:]; W.append(a/np.linalg.norm(a))
    W=np.array(W); j=names.index('bp'); k=names.index('mp'); p=names.index('ppcd')
    print(f'{q:22s} bp range [{W[:,j].min():+.2f},{W[:,j].max():+.2f}] mp [{W[:,k].min():+.2f},{W[:,k].max():+.2f}] max|ppcd| {abs(W[:,p]).max():.3f}')
print('tran mm/d range (JJA hourly):', end=' ')
h,F=rd(f'{d}/Tonzi_Stomata_SC.JJA_hr_avg_tran_veg.txt'); print(F.min()*86400, F.max()*86400)
