#This script generates data for PDE MCMC

import numpy as np
import active_subspaces as asub
import pde_model as pdem
from pde_mcmc_study import *
import time
import scipy.io as IO

begin = time.time()


# instantiate the PDE model
model = pdem.PDESolver()
m = model.m

# generate synthetic observations for the inverse problem
x_true = np.random.normal(size=(m, 1))
obs = model.observations(x_true)

# perturb observations by noise for synthetic data
sig2 = 0.0001*np.linalg.norm(obs)**2
data = obs + np.sqrt(sig2)*np.random.normal(size=obs.shape)
model.set_data(data, sig2)

#######################################################################
#Vanilla MCMC

N = 100000 #Number of MCMC replicates on full space
prop_sig = .1 #Proposal standard deviation
y0 = np.random.normal(size=(m, 1)) #MCMC starting point

# initialize output arrays
YV = np.zeros((N, m))
MF = np.zeros((N, 1))
acceptV = np.zeros((N, 1))

for i in range(N):

    # compute the approximate misfit
    mf = model.misfit(y0)
    
    # store misfits and chains
    MF[i] = mf
    YV[i,:] = y0.reshape((m,))
    
    # proposal step in the active variables
    yc = y0 + np.random.normal(scale=prop_sig, size=y0.shape)
    
    # compute log of the acceptance ratio
    gamma = (mf - model.misfit(yc)) + 0.5*(np.dot(y0.T, y0) - np.dot(yc.T, yc))
    
    # compute the acceptance ratio
    alpha = np.minimum(1.0, np.exp(gamma))
    
    # accept / reject
    t = np.random.uniform(0.0, 1.0)
    if alpha > t:
        y0 = yc
        acceptV[i] = 1.0

#Store Data from Vanilla MCMC
IO.savemat('Vanilla_Chain', {'Chain' : YV})
IO.savemat('Vanilla_Acceptance', {'Accept' : acceptV})

#End vanilla MCMC
######################################################################



#Active Variable MCMC:

# compute the active subspace for the model's misfit given data
N = 1000 #Number of replicates to estimate the active subspace
df_explore = compute_misfit_active_subspace(N, model, False)
IO.savemat('Gradient_Samples', {'Samples' : df_explore})

ss = asub.subspaces.Subspaces()
ss.compute(df=df_explore)
ss.partition(2)
n = ss.W1.shape[1]

# instantiate an active subspace accelerated MCMC sampler
asamcmc = ActiveSubspaceAcceleratedMCMC(model, ss)

# sample the active variables with MCMC
y0 = np.random.normal(size=(n, 1))
N = 10000 #Number of MCMC samples on the active subspace
M = 10 #Number of points to estimate E(f|y)
Y, MF, accept = asamcmc.sample_active_variables(y0, N, M)

#Store data from active variable MCMC
IO.savemat('AV_Chain', {'Chain' : Y})
IO.savemat('AV_Acceptance', {'Accept' : accept})

end = time.time()
span = end - begin
print 'Took {} seconds or {} minutes or {} hours'.format(span, span/60., span/(60.**2))