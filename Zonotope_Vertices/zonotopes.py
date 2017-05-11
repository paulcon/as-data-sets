import numpy as np
import qp_solver as qp
from scipy.spatial import ConvexHull, Delaunay
from scipy.misc import comb

def nzv(m, n):
    """
    Compute the number of zonotope vertices for a linear map from R^m to R^n.

    :param int m: The dimension of the hypercube.
    :param int n: The dimension of the low-dimesional subspace.

    :return: N, The number of vertices defining the zonotope.
    :rtype: int
    """
    if not isinstance(m, int):
        raise TypeError('m should be an integer.')

    if not isinstance(n, int):
        raise TypeError('n should be an integer.')

    N = 0
    for i in range(n):
        N = N + comb(m-1,i)
    N = 2*N
    return int(N)
    
def unique_rows(S):
    """
    Return the unique rows from S.
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """
    #T = np.ascontiguousarray(S).view(np.dtype((np.void, S.dtype.itemsize * S.shape[1])))
    #_, idx = np.unique(T, return_index=True)
    #return S[idx]
    T = S.view(np.dtype((np.void, S.dtype.itemsize * S.shape[1])))
    return np.unique(T).view(S.dtype).reshape(-1, S.shape[1])

def polyarea(Y):
    """
    Compute the area of the polygon defined by columns of Y. 
    http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    """
    ch = ConvexHull(Y)
    x, y = Y[ch.vertices,0], Y[ch.vertices,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def tetrahedron_volume(a, b, c, d):
    """
    Volume of tetrahedron.
    http://stackoverflow.com/questions/24733185/volume-of-convex-hull-with-qhull-from-scipy
    """
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def convex_hull_volume(Y):
    """
    Compute the volume of the polygon defined by columns of Y. 
    http://stackoverflow.com/questions/24733185/volume-of-convex-hull-with-qhull-from-scipy
    """
    ch = ConvexHull(Y)
    dt = Delaunay(Y[ch.vertices])
    tets = dt.points[dt.simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))

def zonotope_vertices(W, Nsamples=1e6, maxcount=1e4):
    """
    Compute the vertices of the zonotope.

    :param ndarray W1: m-by-n matrix that contains the eigenvector bases of the
        n-dimensional active subspace.
    :param int Nsamples: number of samples

    :return: Y, nzv-by-n matrix that contains the zonotope vertices.
    :rtype: ndarray

    """

    m, n = W.shape
    num_verts = nzv(m,n)
    
    # initialize
    Z = np.random.normal(size=(Nsamples, n))
    S = unique_rows(np.sign(np.dot(Z, W.transpose())))
    S = unique_rows(np.vstack((S, -S)))
    N = S.shape[0]
    
    count = 0
    while N < num_verts:
        Z = np.random.normal(size=(Nsamples, n))
        S0 = unique_rows(np.sign(np.dot(Z, W.transpose())))
        S0 = unique_rows(np.vstack((S0, -S0)))
        S = unique_rows(np.vstack((S, S0)))
        N = S.shape[0]
        count += 1
        if count > maxcount:
            break
        
    Y = np.dot(S, W)
    
    if num_verts > Y.shape[0]:
        print 'Warning: {} of {} vertices found.'.format(Y.shape[0], num_verts)
    
    return Y
    
def zonotope_hausdist(Z, V):
    qps = qp.QPSolver()
    n = Z.shape[1]
    Q = np.eye(n)
    
    # get distances from Z vertices
    distZvert = np.zeros(( Z.shape[0], 1 ))
    ch = ConvexHull(V)
    A = ch.equations[:,:n]
    b = ch.equations[:,n].reshape(( A.shape[0], 1 ))
    for i in range(Z.shape[0]):
        z = Z[i,:].reshape(( n, 1 ))
        c = -2*z
        x = qps.quadratic_program_ineq(c, Q, A, b)
        distZvert[i,0] = np.linalg.norm(x - z)
        
    # get distances from V vertices
    distVvert = np.zeros(( V.shape[0], 1 ))
    ch = ConvexHull(Z)
    A = ch.equations[:,:n]
    b = ch.equations[:,n].reshape(( A.shape[0], 1 ))
    for i in range(V.shape[0]):
        v = V[i,:].reshape(( n, 1 ))
        c = -2*v
        x = qps.quadratic_program_ineq(c, Q, A, b)
        distVvert[i,0] = np.linalg.norm(x - v)
    
    return np.max( np.vstack(( distZvert, distVvert )) )
    
    
def zonotope_haus_errors(W, samples, truth):
    n = W.shape[1]
    N = samples.shape[0]
    Nstop = N
    errorz = np.zeros((N,))
    
    S = np.vstack((np.sign(np.dot(samples[0,:], W.transpose())), -np.sign(np.dot(samples[0,:], W.transpose()))))
    errorz[0] = -1.0
    if n==2:

        for i in np.arange(1,N):
            S0 = np.vstack((np.sign(np.dot(samples[i,:], W.transpose())), -np.sign(np.dot(samples[i,:], W.transpose()))))
            S = unique_rows(np.vstack((S, S0)))
            Y = np.dot(S, W);
            if Y.shape[0] < 3:
                E = -1.0
            else:
                E = zonotope_hausdist(truth, Y)
            
            if np.fabs(E) < np.sqrt(np.finfo(float).eps):
                Nstop = i
                break
            else:
                errorz[i] = E
                
    elif n==3:
        for i in np.arange(1,N):
            S0 = np.vstack((np.sign(np.dot(samples[i,:], W.transpose())), -np.sign(np.dot(samples[i,:], W.transpose()))))
            S = unique_rows(np.vstack((S, S0)))
            Y = np.dot(S, W);
            if Y.shape[0] < 5:
                E = -1.0
            else:
                E = zonotope_hausdist(truth, Y)
                
            if np.fabs(E) < np.sqrt(np.finfo(float).eps):
                Nstop = i
                break
            else:
                errorz[i] = E
        
    else:
        raise Exception('Wrong dimensions.')
        
    return errorz, Nstop

def zonotope_haus_errors2(W, samples, truth):
    n = W.shape[1]
    N = np.round(np.logspace(1, np.log10(samples.shape[0]), 9)).astype(int)
    errorz = np.zeros((len(N),))

    for i in np.arange(len(N)):
        S0 = unique_rows( np.sign( np.dot( samples[:N[i],:], W.transpose() ) ) )
        S = unique_rows(np.vstack((S0, -S0)))
        Y = np.dot(S, W);
        if Y.shape[0] < n+1:
            E = -1.0
        else:
            E = zonotope_hausdist(truth, Y)
        
        if np.fabs(E) < np.sqrt(np.finfo(float).eps):
            break
        else:
            errorz[i] = E

    return errorz, N

def zonotope_errors(W, Z, truth):
    n = W.shape[1]
    N = Z.shape[0]
    
    Nstop = N
    errorz = np.zeros((N,))
    
    S = np.vstack((np.sign(np.dot(Z[0,:], W.transpose())), -np.sign(np.dot(Z[0,:], W.transpose()))))
    errorz[0] = 1.0
    if n==2:

        for i in np.arange(1,N):
            S0 = np.vstack((np.sign(np.dot(Z[i,:], W.transpose())), -np.sign(np.dot(Z[i,:], W.transpose()))))
            S = unique_rows(np.vstack((S, S0)))
            Y = np.dot(S, W);
            if Y.shape[0] < 3:
                A = 0.0
            else:
                A = polyarea(Y)
            E = np.fabs(truth - A)/truth
            if np.fabs(E) < np.sqrt(np.finfo(float).eps):
                Nstop = i
                break
            else:
                errorz[i] = E
                
    elif n==3:
        for i in np.arange(1,N):
            S0 = np.vstack((np.sign(np.dot(Z[i,:], W.transpose())), -np.sign(np.dot(Z[i,:], W.transpose()))))
            S = unique_rows(np.vstack((S, S0)))
            Y = np.dot(S, W);
            if Y.shape[0] < 5:
                A = 0.0
            else:
                A = convex_hull_volume(Y)
            E = np.fabs(truth - A)/truth
            if np.fabs(E) < np.sqrt(np.finfo(float).eps):
                Nstop = i
                break
            else:
                errorz[i] = E
        
    else:
        raise Exception('Wrong dimensions.')
        
    return errorz, Nstop
    

def zonotope_nstop(W, Z):
    m, n = W.shape
    Nsamples = Z.shape[0]
    num_verts = nzv(m, n)
    
    sample_step = 1000
    lb, ub = 0, sample_step
    count = 1    

    S = unique_rows(np.vstack((np.sign(np.dot(Z[lb:ub,:], W.transpose())), -np.sign(np.dot(Z[lb:ub,:], W.transpose())))))
    N = S.shape[0]
    
    while N < num_verts:
        lb, ub = count*sample_step, (count+1)*sample_step
        count += 1
        
        S0 = unique_rows(np.vstack((np.sign(np.dot(Z[lb:ub,:], W.transpose())), -np.sign(np.dot(Z[lb:ub,:], W.transpose())))))
        S = unique_rows(np.vstack((S, S0)))
        N = S.shape[0]

        if ub > Nsamples-1:
            print 'Warning: count is Nsamples'
            break
        
    return 0.5*(lb+ub)
    
    
    
    
