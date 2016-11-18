import numpy as np
import pde_model as pdem

if __name__ == "__main__":
    
    # instantiate the PDE model
    model = pdem.PDESolver()
    m = model.m
    
    # sample random input point
    x = np.random.normal(size=(m, 1))
    
    # evaluate the quantity of interest
    q = model.qoi(x)
    print q
    
    # evaluate the gradient of the quantity of interest
    dq = model.grad_qoi(x)
    print dq
    
    

