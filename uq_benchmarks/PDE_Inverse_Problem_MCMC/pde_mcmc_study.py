import numpy as np
import active_subspaces as asub
import pde_model as pdem
import pdb
        
class ActiveSubspaceAcceleratedMCMC():
    def __init__(self, model, ss):
        
        # set PDE model
        self.model = model
        
        # set active subspace and inactive subspace bases
        self.W1 = ss.W1
        self.W2 = ss.W2
        
    def misfit(self, y, NN):
        
        # monte carlo of conditional expectation given y
        gz = np.zeros((NN,1))
        m = self.W1.shape[0]
        
        for i in range(NN):
            z = np.random.normal(size=(self.W2.shape[1], 1))
            x = np.dot(self.W1, y).reshape((m,1)) + np.dot(self.W2, z).reshape((m,1))
            gz[i,0] += self.model.misfit(x)
            
        return np.mean(gz), np.var(gz)
        
    def sample_active_variables(self, y0, N, M, prop_sig=0.1):
        """ 
        Algorithm 1 from Constantine, Kent, Bui-Thanh, "Accelerating MCMC with
        active subspaces" SISC (2016)
        
        Runs random walk MCMC (Metropolis-Hastings) on the coordinates of the 
        active subspace. Samples independently according to prior on 
        coordinates of the inactive subspace.
        
        Inputs:
            y0, initial point 
            N, number of samples
            M, number of samples for MC evaluation of misfit
            prop_sig, symmetric proposal variance
        
        Outputs:
            Y, samples of active variables 
            MF, misfit values
            accept, array indicating whether step was accepted (1) or rejected (0)
        """
        
        # dimension of active subspace
        n = self.W1.shape[1]
        
        # initialize output arrays
        Y = np.zeros((N, n))
        MF = np.zeros((N, 1))
        accept = np.zeros((N, 1))
        
        for i in range(N):
            
            # compute the approximate misfit
            mf = self.misfit(y0, M)[0]
            
            # store misfits and chains
            MF[i] = mf
            Y[i,:] = y0.reshape((n,))
            
            # proposal step in the active variables
            yc = y0 + np.random.normal(scale=prop_sig, size=y0.shape)
            
            # compute log of the acceptance ratio
            gamma = (mf-self.misfit(yc, M))[0] + 0.5*(np.dot(y0.T, y0)-np.dot(yc.T, yc))
            
            # compute the acceptance ratio
            alpha = np.minimum(1.0,np.exp(gamma))
            
            # accept / reject
            t = np.random.uniform(0.0,1.0)
            if alpha>t:
                y0 = yc
                accept[i] = 1.0
            
        return Y, MF, accept
        
    def sample_original_variables(self, Y, P):
        """
        For each sample of the active variables from MCMC, sample P inactive 
        variables according to the prior.
        
        Inputs:
            Y, array of active variables samples from MCMC
            P, number of inactive variable samples from prior
            
        Outputs:
            X, array of samples from original variables
        
        """
        
        # number of active variable samples and dimension of active subspace
        N, n = Y.shape
        
        # dimension of original parameter space
        m = self.model.m
        
        # initialize array for samples
        X = np.zeros((N*P, m))
        
        for i in range(N):
            
            # sample from prior on inactive variables
            Z = np.random.normal(size=(P, m-n))
            
            # compute P samples of original variables
            XX = np.dot(self.W1, Y[i,:].reshape((n, ))).reshape((m, 1)) + \
                np.dot(self.W2, Z.transpose()).reshape((m, P))
                
            # store
            X[i*P:(i+1)*P,:] = XX.transpose()
            
        return X
        
def compute_misfit_active_subspace(N, pde_model, return_ss=True):
    
    # sample from standard normal prior
    m = pde_model.m
    X = np.random.normal(size=(N, m))
    
    # compute the gradient of the misfit for each sample
    sr = asub.utils.simrunners.SimulationGradientRunner(pde_model.misfit_grad)
    df = sr.run(X)
    
    # compute the active subspace
    ss = asub.Subspaces()
    ss.compute(df=df, nboot=100)
    
    if return_ss: return ss
    else: return df
        
if __name__ == "__main__":
    
    # instantiate the PDE model
    model = pdem.PDESolver()
    m = model.m
    
    # generate synthetic observations for the inverse problem
    x_true = np.random.normal(size=(m, 1))
    obs = model.observations(x_true)
    
    # perturb observations by noise for synthetic data
    sig2 = 0.0001
    data = obs + np.sqrt(sig2)*np.random.normal(size=obs.shape)
    model.set_data(data, sig2)
    
    # compute the active subspace for the model's misfit given data
    N = 100
    ss = compute_misfit_active_subspace(N, model)
    print 'Active subspace estimated!'
    
    # plot active subspace metrics
    asub.utils.plotters.eigenvalues(ss.eigenvals[:10, 0], e_br=ss.e_br[:10,:])
    asub.utils.plotters.subspace_errors(ss.sub_br[:10,:])
    asub.utils.plotters.eigenvectors(ss.eigenvecs[:,:2])
    
    # set active subspace dimension to 2
    n = 2
    ss.partition(n)
    
    # instantiate an active subspace accelerated MCMC sampler
    asamcmc = ActiveSubspaceAcceleratedMCMC(model, ss)
    
    # sample the active variables with MCMC
    y0 = np.random.normal(size=(n, 1))
    N = 100
    M = 10
    Y, MF, accept = asamcmc.sample_active_variables(y0, N, M)
    print 'Active subspace sampled, acceptance ratio: {:04.2f}'.format(np.mean(accept))
    
    # sample inactive variables from prior and transform to original variables
    P = 10
    X = asamcmc.sample_original_variables(Y, P)
    print 'Parameter space sampled'
    
    # compute posterior mean and variance
    post_mean = np.mean(X, axis=0)
    post_var = np.var(X, axis=0)
    

