import numpy as np

coth = lambda x: 1./np.tanh(x)
sech = lambda x: 1./np.cosh(x)
csch = lambda x: 1./np.sinh(x)

l = 1.0
mu0 = 1.0

lb = np.log(np.array([.05, 1, .5, .5, .1])); ub = np.log(np.array([.2, 5, 3, 3, 1]))

#average velocity
def uavg(x):
    x = np.exp(x)
    return -x[:,2]*x[:,3]/x[:,4]**2*(1 - x[:,4]*l/np.sqrt(x[:,3]*x[:,0])*coth(x[:,4]*l/\
                                    np.sqrt(x[:,3]*x[:,0])))

#induced magnetic field
def Bind(x):
    x = np.exp(x)
    return x[:,2]*l*mu0/(2*x[:,4])*(1 - 2*np.sqrt(x[:,3]*x[:,0])/(x[:,4]*l)*np.tanh(x[:,4]*\
                l/(2*np.sqrt(x[:,3]*x[:,0]))))

#gradient of average velocity
def uavg_grad(x):
    x = np.exp(x)
    mu = x[:,0]; rho = x[:,1]; dp0 = x[:,2]; eta = x[:,3]; B0 = x[:,4]
    
    dudmu = -dp0*eta/B0**2*(B0*l/(2*np.sqrt(eta))*mu**(-3./2)*coth(B0*l/np.sqrt(eta*mu)) - \
            (B0*l)**2*np.sqrt(eta/mu)*csch(B0*l/np.sqrt(eta*mu))**2/(2*(eta*mu)**(3./2)))
    dudrho = 0.0*np.empty(x.shape[0])
    duddp0 = -eta/B0**2*(1 - B0*l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu)))
    dudeta = -dp0/B0**2*(1 - B0*l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu))) - \
        dp0*eta/B0**2*(B0*l/(2*np.sqrt(mu))*eta**(-3./2)*coth(B0*l/np.sqrt(eta*mu)) - \
            (B0*l)**2*np.sqrt(mu/eta)*csch(B0*l/np.sqrt(eta*mu))**2/(2*(eta*mu)**(3./2)))
    dudB0 = 2*dp0*eta/B0**3*(1 - B0*l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu))) - \
        dp0*eta/B0**2*(-l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu)) + B0*l**2/(eta*mu)*\
                                                csch(B0*l/np.sqrt(eta*mu))**2)
        
    dudmu = dudmu[:,None]; dudrho = dudrho[:,None]; duddp0 = duddp0[:,None]
    dudeta = dudeta[:,None]; dudB0 = dudB0[:,None]
    return np.hstack((dudmu, dudrho, duddp0, dudeta, dudB0))*x*(ub - lb).reshape((1, 5))/2.

#gradient of induced magnetic field
def Bind_grad(x):
    x = np.exp(x)
    mu = x[:,0]; rho = x[:,1]; dp0 = x[:,2]; eta = x[:,3]; B0 = x[:,4]
    
    dBdmu = dp0*l*mu0/(2*B0)*(-(B0*l)**-1*np.sqrt(eta/mu)*np.tanh(B0*l/(2*np.sqrt(eta*mu))) +\
                eta*np.sqrt(eta*mu)*sech(B0*l/(2*np.sqrt(eta*mu)))**2/(2*(eta*mu)**(3./2)))
    dBdrho = 0.0*np.empty(x.shape[0])
    dBddp0 = l*mu0/(2*B0)*(1 - 2*np.sqrt(eta*mu)/(B0*l)*np.tanh(B0*l/(2*np.sqrt(eta*mu))))
    dBdeta = dp0*l*mu0/(2*B0)*(-(B0*l)**-1*np.sqrt(mu/eta)*np.tanh(B0*l/(2*np.sqrt(eta*mu)))+\
                mu*np.sqrt(eta*mu)*sech(B0*l/(2*np.sqrt(eta*mu)))**2/(2*(eta*mu)**(3./2)))
    dBdB0 = -dp0*l*mu0/(2*B0**2)*(1 - 2*np.sqrt(eta*mu)/(B0*l)*np.tanh(B0*l/\
        (2*np.sqrt(eta*mu)))) + dp0*l*mu0/(2*B0)*(2*np.sqrt(eta*mu)/(B0**2*l)*np.tanh(\
        B0*l/(2*np.sqrt(eta*mu))) - B0**-1*sech(B0*l/(2*np.sqrt(eta*mu)))**2)
        
    dBdmu = dBdmu[:,None]; dBdrho = dBdrho[:,None]; dBddp0 = dBddp0[:,None]
    dBdeta = dBdeta[:,None]; dBdB0 = dBdB0[:,None]
    return np.hstack((dBdmu, dBdrho, dBddp0, dBdeta, dBdB0))*x*(ub - lb).reshape((1, 5))/2.

