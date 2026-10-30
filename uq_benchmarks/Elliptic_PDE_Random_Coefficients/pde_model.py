import numpad as npd
import numpy as np

class PDESolver(object):
    
    def __init__(self, corrcase='short'):
        
        # load and construct Karhunen-Loeve bases
        kl = np.load('kl-{}.npz'.format(corrcase))
        K, lam = kl['K'], kl['lam']
        self.B = K*np.sqrt(lam.reshape((1, K.shape[1])))
        
        # right hand side
        self.f = 1.0

        # number of cells in spatial discretization and grid spacing
        self.nx = int(np.sqrt(self.B.shape[0]))
        self.dx = 1./self.nx

        # number of parameters
        self.m = K.shape[1]
                
    def set_coeff(self, x):
        # given a value for the parameters (KL coefficients),
        # construct the coefficients of the differential operator
        _e = 2.7182818284590451
        self.x = npd.array(x.reshape((self.m, 1)))
        a = npd.dot(self.B, self.x)
        self.a = _e**a.reshape((self.nx, self.nx))
        
    def residual(self, u):
        # compute the PDE residual for use in "solve"
        u = np.hstack([np.zeros([self.nx-1,1]), u, u[:,-1:]])
        u = np.vstack([np.zeros([1,self.nx+1]), u, np.zeros([1,self.nx+1])])
        a_hor = 0.5 * (self.a[1:,:] + self.a[:-1,:])
        a_ver = 0.5 * (self.a[:,1:] + self.a[:,:-1])
        a_dudy = a_hor * (u[1:-1,1:] - u[1:-1,:-1]) / self.dx
        a_dudx = a_ver * (u[1:,1:-1] - u[:-1,1:-1]) / self.dx
        res = (a_dudy[:,1:] - a_dudy[:,:-1]) / self.dx \
              + (a_dudx[1:,:] - a_dudx[:-1,:]) / self.dx + self.f
        return res
        
    def qoi(self, x):
        
        # set the operator coefficients with the given x
        self.set_coeff(x)
        
        # solve the PDE
        u = npd.solve(self.residual, np.zeros([self.nx-1, self.nx-1]), verbose=False)
        
        # get the PDE solution on the boundary
        u_bnd = u[:, -1]
        
        # compute the average on the boundary
        q = 0.5*self.dx*np.sum(u_bnd[1:] + u_bnd[:-1])
        
        # return a number from numpad
        return npd.value(q)
        
    def grad_qoi(self, x):
        
        # set the operator coefficients with the given x 
        self.set_coeff(x)
        
        # solve the PDE
        u = npd.solve(self.residual, np.zeros([self.nx-1, self.nx-1]), verbose=False)
        
        # get the PDE solution on the boundary
        u_bnd = u[:, -1]
        
        # compute the average on the boundary
        q = 0.5*self.dx*np.sum(u_bnd[1:] + u_bnd[:-1])
        
        # return the gradient of q as a numpy array
        return q.diff(self.x).todense()
    
