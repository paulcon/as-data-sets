from HIV_model import *
import numpy as np
import pandas as pn
import active_subspaces as ac
import time

bt = time.time()

nominal = np.array([10, .15, 5, .2, 55.6, 3.87e-3, 1e-6, 4.5e-4, 7.45e-4, 5.22e-4, 3e-6,\
    3.3e-4, 6e-9, .537, .285, 7.79e-6, 1e-6, 4e-5, .01, .28, .05, .005, .005, .015, 2.39, 3e-4, .97])
    
xl = .975*nominal; xu = 1.025*nominal

N = 1000

p = np.random.uniform(-1, 1, (N, len(xl)))
p = .5*((np.diag(xu - xl)).dot(p.T) + xu[:,None] + xl[:,None]).T

pn.DataFrame(p).to_csv('inputs.csv')

times = np.linspace(1, 3400, 3400)

f = Tcells(p, times)

times = np.array([5, 15, 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 750, 1000, 1600, 1800, 2000, 2200, 2400, 2800, 3400])
pn.DataFrame(f[:,times-1]).to_csv('outputs.csv')

df = np.empty((N, 27*len(times)))
h = 1e-7*(xu - xl).reshape((1, len(xl)))/2.

for i in range(len(times)):
    df[:,27*i:27*(i+1)] = ac.gradients.finite_difference_gradients(p, lambda x: Tcells(x, np.linspace(1, times[i], times[i]))[:,-1], h)
    df[:,27*i:27*(i+1)] *= .5*(xu - xl).reshape((1, len(xl)))

pn.DataFrame(df).to_csv('gradients.csv')

et = time.time()

print 'Took {:.4} seconds or {:.4} hours'.format(et - bt, (et - bt)/60**2)

