"""Hartmann problem: average velocity and induced magnetic field, and their gradients.

The closed-form quantities of interest are Eqs. (36) and (37) of A. Glaws,
P. G. Constantine, J. N. Shadid and T. M. Wildey, "Dimension reduction in
magnetohydrodynamics power generation models: Dimensional analysis and active
subspaces", Stat. Anal. Data Min. 10(5) (2017) 312-325, doi:10.1002/sam.11355.
Equation and table numbers below refer to that paper.

The inputs are the logs of five parameters, in this order:

    log mu, log rho, log dp0/dx, log eta, log B0

(fluid viscosity, fluid density, applied pressure gradient, magnetic
resistivity, applied magnetic field). The half-width l of the channel and the
vacuum permeability mu0 are fixed keyword arguments, both 1 by default: the
paper sets l = 1 m (p. 318) and does not state mu0 for the Hartmann problem.
mu0 only scales B_ind. Neither quantity depends on rho.

The model is closed form, so there are no data files.
"""

import numpy as np

NAMES = ('log mu', 'log rho', 'log dp0/dx', 'log eta', 'log B0')
LABELS = (r'$\log(\mu)$', r'$\log(\rho)$', r'$\log(\frac{\partial p_0}{\partial x})$',
          r'$\log(\eta)$', r'$\log(B_0)$')

# Ranges of the physical parameters for the Hartmann problem (Table 2), in the order of NAMES.
# The inputs are uniform on [log(LOWER), log(UPPER)].
LOWER = np.array([.05, 1., .5, .5, .1])
UPPER = np.array([.2, 5., 3., 3., 1.])


def _params(logx):
    x = np.exp(np.asarray(logx, dtype=float))
    if x.shape[-1] != 5:
        raise ValueError(f'expected 5 log-inputs in the last axis, got shape {x.shape}')
    return np.moveaxis(x, -1, 0)      # mu, rho, dp0, eta, B0


def uavg(logx, l=1.0):
    """Average flow velocity across the channel, Eq. (36).

    Parameters
    ----------
    logx : array_like, shape (..., 5)
        Log-inputs in the order of NAMES.
    l : float, default 1.0
        Half-width of the channel.

    Returns
    -------
    numpy.ndarray, shape (...)
    """
    mu, rho, dp0, eta, B0 = _params(logx)
    a = B0*l/np.sqrt(eta*mu)                       # the Hartmann number
    return -dp0*eta/B0**2*(1 - a/np.tanh(a))


def Bind(logx, l=1.0, mu0=1.0):
    """Induced magnetic field, Eq. (37).

    Parameters
    ----------
    logx : array_like, shape (..., 5)
        Log-inputs in the order of NAMES.
    l : float, default 1.0
        Half-width of the channel.
    mu0 : float, default 1.0
        Vacuum permeability.

    Returns
    -------
    numpy.ndarray, shape (...)
    """
    mu, rho, dp0, eta, B0 = _params(logx)
    b = B0*l/(2*np.sqrt(eta*mu))                   # half the Hartmann number
    return dp0*l*mu0/(2*B0)*(1 - np.tanh(b)/b)


def uavg_grad(logx, l=1.0):
    """Gradient of `uavg` with respect to the log-inputs.

    Parameters
    ----------
    logx : array_like, shape (..., 5)
        Log-inputs in the order of NAMES.
    l : float, default 1.0
        Half-width of the channel.

    Returns
    -------
    numpy.ndarray, shape (..., 5)
        d uavg / d log x_i = x_i * d uavg / d x_i, in the order of NAMES.
    """
    mu, rho, dp0, eta, B0 = _params(logx)
    a = B0*l/np.sqrt(eta*mu)
    P = dp0*eta/B0**2
    u = -P*(1 - a/np.tanh(a))
    dh = -1/np.tanh(a) + a/np.sinh(a)**2           # d/da of (1 - a coth a)
    # d a / d log(mu, eta, B0) = (-a/2, -a/2, a)
    d_mu = P*dh*a/2
    return np.stack([
        d_mu,                                       # log mu
        np.zeros_like(u),                           # log rho
        u,                                          # log dp0/dx
        u + d_mu,                                   # log eta
        -2*u - P*dh*a,                              # log B0
    ], axis=-1)


def Bind_grad(logx, l=1.0, mu0=1.0):
    """Gradient of `Bind` with respect to the log-inputs.

    Parameters
    ----------
    logx : array_like, shape (..., 5)
        Log-inputs in the order of NAMES.
    l : float, default 1.0
        Half-width of the channel.
    mu0 : float, default 1.0
        Vacuum permeability.

    Returns
    -------
    numpy.ndarray, shape (..., 5)
        d Bind / d log x_i = x_i * d Bind / d x_i, in the order of NAMES.
    """
    mu, rho, dp0, eta, B0 = _params(logx)
    b = B0*l/(2*np.sqrt(eta*mu))
    P = dp0*l*mu0/(2*B0)
    B = P*(1 - np.tanh(b)/b)
    dk = -1/(b*np.cosh(b)**2) + np.tanh(b)/b**2    # d/db of (1 - tanh(b)/b)
    # d b / d log(mu, eta, B0) = (-b/2, -b/2, b)
    d_mu = -P*dk*b/2
    return np.stack([
        d_mu,                                       # log mu
        np.zeros_like(B),                           # log rho
        B,                                          # log dp0/dx
        d_mu,                                       # log eta
        -B + P*dk*b,                                # log B0
    ], axis=-1)
