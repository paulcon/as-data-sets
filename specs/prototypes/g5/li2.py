import numpy as np
for f,v in [('0.25C','V_025C.txt'),('1C','V_1C.txt'),('4C','V_4C.txt')]:
    d=np.load(f+'.npy'); vt=np.loadtxt('/Users/paulconstantine/PycharmProjects/as-data-sets/Lithium_Battery/'+v)
    V=d[:,20:70]; C=d[:,70:120]
    print(f,'vtime',vt[[0,1,2,17,34,38,48,49]].round(3), 'monotone dec',np.all(np.diff(vt)<0))
    print('  V col std first/last',V[:,0].std(),V[:,-1].std(), 'V last col range',V[:,-1].min(),V[:,-1].max())
    print('  C col0 std',C[:,0].std(),'C col0 range',C[:,0].min(),C[:,0].max(),' C col49 std',C[:,-1].std())
    print('  C monotone in col', np.all(np.diff(C,axis=1)>=0))
    print('  V monotone dec', np.mean(np.all(np.diff(V,axis=1)<=0,axis=1)))
d0=np.load('0.25C.npy');d1=np.load('1C.npy');d4=np.load('4C.npy')
print('X identical across files', np.array_equal(d0[:,:20],d1[:,:20]), np.array_equal(d0[:,:20],d4[:,:20]))
print('row 2968 4C V', d4[2968,20:70].round(3))
print('row 1537 4C V', d4[1537,20:70].round(3))
print('median 4C V', np.median(d4[:,20:70],0).round(3))
