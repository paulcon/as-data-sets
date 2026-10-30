import numpy as np, time, sys
for f in ['0.25C.txt','1C.txt','4C.txt']:
    t=time.time()
    d=np.loadtxt(f,delimiter=',',skiprows=1)
    print(f,d.shape,'%.1fs'%(time.time()-t))
    run=d[:,0]; X=d[:,1:20]
    print(' run col', run[:3], run[-3:], 'unique',len(np.unique(run)))
    print(' X min',X.min(0).round(3)); print(' X max',X.max(0).round(3))
    print(' X mean',X.mean(0).round(3)); print(' X std',X.std(0).round(3))
    V=d[:,20:70]; C=d[:,70:120]
    print(' V range',V.min(),V.max(),' C range',C.min(),C.max())
    print(' nan',np.isnan(d).sum())
    # outlier search
    for name,B in [('V',V),('C',C),('ConA',d[:,120:170]),('ConS',d[:,170:220]),('ConC',d[:,220:270])]:
        z=np.abs((B-np.median(B,0))/(B.std(0)+1e-300)).max(1)
        i=np.argsort(z)[-3:]
        print('  ',name,'max |z| rows',i,z[i].round(1))
    if f=='4C.txt':
        print(' row1537',d[1537,:3],d[1537,20:30],d[1537,70:80])
    np.save(sys.argv[1]+'/'+f.replace('.txt','')+'.npy',d)
