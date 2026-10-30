"""Basic reproduction number of the modified SEIR Ebola model and its gradient.

The model is from P. Diaz, P. Constantine, K. Kalmbach, E. Jones and
S. Pankavich, "A modified SEIR model for the spread of Ebola in Western Africa
and metrics for resource allocation", Appl. Math. Comput. 324 (2018) 141-155,
doi:10.1016/j.amc.2017.11.039. Equation and table numbers below refer to that
paper.

R0 (Eq. (4)) depends on eight of the model's parameters, in this order:

    beta1, beta2, beta3, rho1, gamma1, gamma2, omega, psi

The incubation rate delta (fixed at 1/9) and the burial fraction rho2 do not
enter R0. The model is closed form, so there are no data files.
"""

import numpy as np

NAMES = ('beta1', 'beta2', 'beta3', 'rho1', 'gamma1', 'gamma2', 'omega', 'psi')
LABELS = (r'$\beta_1$', r'$\beta_2$', r'$\beta_3$', r'$\rho_1$', r'$\gamma_1$',
          r'$\gamma_2$', r'$\omega$', r'$\psi$')
COUNTRIES = ('Liberia', 'Sierra Leone')

# Fitted values (Table 2), in the order of NAMES.
FITTED = {
    'Liberia': np.array([.376, .135, .163, .98, .0542, .174, .325, .5]),
    'Sierra Leone': np.array([.251, .395, .079, .76, .051, .0833, .370, .442]),
}

# Uniform ranges for the global analysis (Table 7): (lower, upper), in the order of NAMES.
# The fitted Sierra Leone gamma2 = 0.0833 lies outside its range [0.1236, 0.384];
# the paper has the same disagreement between Tables 2 and 7.
BOUNDS = {
    'Liberia': (np.array([.1, .1, .05, .41, .0276, .081, .25, .0833]),
                np.array([.4, .4, .2, 1., .1702, .21, .5, .7])),
    'Sierra Leone': (np.array([.1, .1, .05, .41, .0275, .1236, .25, .0833]),
                     np.array([.4, .4, .2, 1., .1569, .384, .5, .7])),
}


def R0(x):
    """Basic reproduction number, Eq. (4).

    Parameters
    ----------
    x : array_like, shape (..., 8)
        Parameters in physical units, in the order of NAMES.

    Returns
    -------
    numpy.ndarray, shape (...)
    """
    b1, b2, b3, r1, g1, g2, om, p = np.moveaxis(np.asarray(x, dtype=float), -1, 0)
    return (b1 + b2*r1*g1/om + b3*p/g2) / (g1 + p)


def R0_grad(x):
    """Gradient of R0 with respect to the physical parameters.

    Parameters
    ----------
    x : array_like, shape (..., 8)
        Parameters in physical units, in the order of NAMES.

    Returns
    -------
    numpy.ndarray, shape (..., 8)
        Partial derivatives in the order of NAMES.
    """
    b1, b2, b3, r1, g1, g2, om, p = np.moveaxis(np.asarray(x, dtype=float), -1, 0)
    d = g1 + p
    R = (b1 + b2*r1*g1/om + b3*p/g2) / d
    return np.stack([
        1/d,                        # beta1
        r1*g1/om/d,                 # beta2
        p/g2/d,                     # beta3
        b2*g1/om/d,                 # rho1
        (b2*r1/om - R)/d,           # gamma1
        -b3*p/g2**2/d,              # gamma2
        -b2*r1*g1/om**2/d,          # omega
        (b3/g2 - R)/d,              # psi
    ], axis=-1)
