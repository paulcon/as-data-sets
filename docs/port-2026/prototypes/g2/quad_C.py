import numpy as np
Q=.5*np.array([[np.sqrt(2),np.sqrt(2)],[-np.sqrt(2),np.sqrt(2)]])
pts,w=np.polynomial.hermite_e.hermegauss(50); w=w/np.sqrt(2*np.pi)
X=np.array(np.meshgrid(pts,pts,indexing='ij')).reshape(2,-1).T; W=np.outer(w,w).ravel()
for eps in [0.01,0.95]:
    A=Q@np.diag([1,eps])@Q.T
    for sig2 in [0.01,0.1]:
        m=.5*np.einsum('ij,jk,ik->i',X,A,X); G=-(1/sig2)*(0.9-m)[:,None]*(X@A)
        C=(G*W[:,None]).T@G; print(f'eps={eps} sigma^2={sig2}: eig(C)=',np.sort(np.linalg.eigvalsh(C))[::-1])
