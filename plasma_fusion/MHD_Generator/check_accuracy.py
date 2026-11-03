"""Recompute the two accuracy numbers of DATA.md that the notebook does not print (MHD_Generator).

    python check_accuracy.py          (about 1 s)

1. The ported Hartmann functions against the 2016 ones. `uavg`, `Bind`, `uavg_grad` and
   `Bind_grad` of hartmann_model.py are evaluated on the 11^5 Gauss-Legendre nodes of the
   notebook and compared with the 2016 formulas, after removing the 2016 scaling of the
   gradients to the normalized inputs (a factor (ub - lb)/2 per input). Printed: the largest
   difference relative to the largest absolute value of the 2016 function (each of the 4
   functions; for a gradient, over all nodes and components). DATA.md: about 1e-14 (rounding level).

   The 2016 formulas below are a transcription of MHD/gradients.py at the tag `legacy-master`
   (`git show legacy-master:MHD/gradients.py`): each expression is copied as written there, with
   the module constants l = mu0 = 1 and the Hartmann bounds lb, ub. One change: the
   rho-derivatives are np.zeros, where the 2016 file has 0.0*np.empty(...), which can give NaN.
   Nothing is read from git or imported from hartmann_model.py for the 2016 side, so the check
   does not depend on the module it checks.

2. The u_avg row labelled 73 (run 72, the failed run). A quadratic in the five log-inputs
   (21 terms) is fitted by least squares to log u_avg on all 483 rows of
   MHD_Generator_Data_Uavg.txt. Printed: the absolute residual of label 73 divided by the median
   absolute residual, and the 95th percentile of that ratio over the 483 rows. DATA.md: 9 and 5.

The other numbers of DATA.md (the gradient check against central differences, the eigenvalues,
the label facts, the ratios (dp0/dx) df/d(dp0/dx) / f) are printed by the notebook.

Deterministic; imports numpy, scipy and hartmann_model.py; reads the files next to it and writes
nothing.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.special import roots_legendre

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True                  # write nothing into the directory
sys.path.insert(0, HERE)
import hartmann_model as hm  # noqa: E402


# --- transcription of MHD/gradients.py at legacy-master ----------------------------------------
coth = lambda x: 1./np.tanh(x)  # noqa: E731
sech = lambda x: 1./np.cosh(x)  # noqa: E731
csch = lambda x: 1./np.sinh(x)  # noqa: E731

l = 1.0
mu0 = 1.0

lb = np.log(np.array([.05, 1, .5, .5, .1])); ub = np.log(np.array([.2, 5, 3, 3, 1]))


def uavg_2016(x):
    x = np.exp(x)
    return -x[:,2]*x[:,3]/x[:,4]**2*(1 - x[:,4]*l/np.sqrt(x[:,3]*x[:,0])*coth(x[:,4]*l/\
                                    np.sqrt(x[:,3]*x[:,0])))


def Bind_2016(x):
    x = np.exp(x)
    return x[:,2]*l*mu0/(2*x[:,4])*(1 - 2*np.sqrt(x[:,3]*x[:,0])/(x[:,4]*l)*np.tanh(x[:,4]*\
                l/(2*np.sqrt(x[:,3]*x[:,0]))))


def uavg_grad_2016(x):
    x = np.exp(x)
    mu = x[:,0]; rho = x[:,1]; dp0 = x[:,2]; eta = x[:,3]; B0 = x[:,4]

    dudmu = -dp0*eta/B0**2*(B0*l/(2*np.sqrt(eta))*mu**(-3./2)*coth(B0*l/np.sqrt(eta*mu)) - \
            (B0*l)**2*np.sqrt(eta/mu)*csch(B0*l/np.sqrt(eta*mu))**2/(2*(eta*mu)**(3./2)))
    dudrho = np.zeros(x.shape[0])
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


def Bind_grad_2016(x):
    x = np.exp(x)
    mu = x[:,0]; rho = x[:,1]; dp0 = x[:,2]; eta = x[:,3]; B0 = x[:,4]

    dBdmu = dp0*l*mu0/(2*B0)*(-(B0*l)**-1*np.sqrt(eta/mu)*np.tanh(B0*l/(2*np.sqrt(eta*mu))) +\
                eta*np.sqrt(eta*mu)*sech(B0*l/(2*np.sqrt(eta*mu)))**2/(2*(eta*mu)**(3./2)))
    dBdrho = np.zeros(x.shape[0])
    dBddp0 = l*mu0/(2*B0)*(1 - 2*np.sqrt(eta*mu)/(B0*l)*np.tanh(B0*l/(2*np.sqrt(eta*mu))))
    dBdeta = dp0*l*mu0/(2*B0)*(-(B0*l)**-1*np.sqrt(mu/eta)*np.tanh(B0*l/(2*np.sqrt(eta*mu)))+\
                mu*np.sqrt(eta*mu)*sech(B0*l/(2*np.sqrt(eta*mu)))**2/(2*(eta*mu)**(3./2)))
    dBdB0 = -dp0*l*mu0/(2*B0**2)*(1 - 2*np.sqrt(eta*mu)/(B0*l)*np.tanh(B0*l/\
        (2*np.sqrt(eta*mu)))) + dp0*l*mu0/(2*B0)*(2*np.sqrt(eta*mu)/(B0**2*l)*np.tanh(\
        B0*l/(2*np.sqrt(eta*mu))) - B0**-1*sech(B0*l/(2*np.sqrt(eta*mu)))**2)

    dBdmu = dBdmu[:,None]; dBdrho = dBdrho[:,None]; dBddp0 = dBddp0[:,None]
    dBdeta = dBdeta[:,None]; dBdB0 = dBdB0[:,None]
    return np.hstack((dBdmu, dBdrho, dBddp0, dBdeta, dBdB0))*x*(ub - lb).reshape((1, 5))/2.
# --- end of the transcription -------------------------------------------------------------------


def check_2016():
    t, _ = roots_legendre(11)
    T = np.stack(np.meshgrid(*[t]*5, indexing='ij'), axis=-1).reshape(-1, 5)
    lo, hi = np.log(hm.LOWER), np.log(hm.UPPER)
    assert np.allclose(lo, lb) and np.allclose(hi, ub)
    Xh = lo + (hi - lo)*(T + 1)/2                 # the 11^5 nodes as log-inputs
    half = (ub - lb)/2
    pairs = [('uavg', hm.uavg(Xh), uavg_2016(Xh)),
             ('Bind', hm.Bind(Xh), Bind_2016(Xh)),
             ('uavg_grad', hm.uavg_grad(Xh), uavg_grad_2016(Xh)/half),
             ('Bind_grad', hm.Bind_grad(Xh), Bind_grad_2016(Xh)/half)]
    print(f'1. hartmann_model.py against the 2016 gradients.py (transcribed), {len(Xh):,} nodes; '
          'max |new - 2016| / max |2016|:')
    worst = 0.0
    for name, new, old in pairs:
        r = np.max(np.abs(new - old))/np.max(np.abs(old))
        worst = max(worst, r)
        print(f'   {name:10s} {r:.1e}')
    print(f'   largest {worst:.1e} (DATA.md: about 1e-14)')


def check_label_73():
    D = np.loadtxt(os.path.join(HERE, 'MHD_Generator_Data_Uavg.txt'), delimiter=',', skiprows=1)
    lab, L, y = D[:, 0].astype(int), np.log(D[:, 1:6]), np.log(D[:, 6])
    A = np.column_stack([np.ones(len(L))] + [L[:, i] for i in range(5)]
                        + [L[:, i]*L[:, j] for i in range(5) for j in range(i, 5)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = np.abs(y - A @ c)/np.median(np.abs(y - A @ c))
    i73 = np.flatnonzero(lab == 73)[0]
    print(f'2. quadratic in the log-inputs ({A.shape[1]} terms) fitted to log u_avg on all {len(y)} rows: '
          f'label 73 lies {r[i73]:.1f} times the median absolute residual off the fit; 95th percentile '
          f'of that ratio {np.percentile(r, 95):.1f} (DATA.md: 9 and 5)')


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='\n'.join(__doc__.split('\n')[1:])).parse_args()
    t0 = time.perf_counter()
    check_2016()
    check_label_73()
    print(f'run time {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
