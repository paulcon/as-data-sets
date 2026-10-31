"""Direct radiative forcing by anthropogenic sulfate aerosol: a global-mean box model.

Charlson et al., Science 255(5043):423-430, 1992, Eqs. 1-5 (p. 426), combined as

    dF_R = -1/2 S0 T^2 (1 - A_c) (1 - R_s)^2 beta Psi_e f_Psi B,
    B    = 3 * 1e12 * Q * Y * L / (365 * A)        (g of sulfate per m^2),

with the nine uncertain inputs and the lognormal distributions of the UQ
benchmark version of the model (Penner et al., BAMS 75(3):375-400, 1994, as
tabulated by UQTestFuns, "sulfur" function). The factor 3 is the mass ratio of
sulfate to sulfur (96/32), 1e12 converts Tg to g, and 365 converts the lifetime
L from days to years. The transmittance T and the surface coalbedo 1 - R_s enter
as the squared quantities T^2 and (1 - R_s)^2, so dF_R is a constant times the
product of the nine inputs, each to the power 1.

Inputs x (physical units, column order of NAMES) are independent lognormals,
ln x_i ~ N(MU_i, SIGMA_i^2). The active-subspace analysis works in the log
inputs z = ln x, whose distribution is Gaussian, N(MU, diag(SIGMA^2)).
`grad_log(x)` returns the exact gradient d(dF_R)/dz_i = x_i d(dF_R)/dx_i, which
for this model equals dF_R for every i.

Only numpy is imported.
"""
import numpy as np

#: Input names, in column order.
NAMES = ('Q', 'Y', 'L', 'Psi_e', 'beta', 'f_Psi', 'T2', 'one_minus_Ac', 'one_minus_Rs_sq')

#: Units of the physical inputs.
UNITS = ('Tg S/yr', None, 'days', 'm^2/g', None, None, None, None, None)

#: Descriptions (Charlson et al. 1992, p. 426 and Table 1; UQTestFuns).
DESCRIPTIONS = (
    'source strength of anthropogenic SO2',
    'fraction of emitted SO2 that reacts to form sulfate aerosol',
    'atmospheric lifetime of sulfate',
    'mass scattering efficiency of sulfate (alpha in Charlson et al.)',
    'fraction of scattered light scattered upward (beta bar)',
    'relative increase in scattering from hygroscopic growth (f(RH))',
    'square of the transmittance T of the atmosphere above the aerosol layer',
    'fraction of the earth not covered by cloud, 1 - A_c',
    'square of the surface coalbedo, (1 - R_s)^2',
)

#: Log-scale means and standard deviations of the lognormal inputs (UQTestFuns,
#: attributed there to Penner et al. 1994). exp(MU) is the median.
MU = np.array([4.26268, -0.693147, 1.70475, 1.60944, -1.20397, 0.530628, -0.544727,
               -0.941609, -0.328504])
SIGMA = np.array([0.139762, 0.405465, 0.405465, 0.336472, 0.262364, 0.182322, 0.336472,
                  0.0953102, 0.182322])

#: Fixed constants of the benchmark version (UQTestFuns): solar constant in W/m^2
#: (Kopp and Lean 2011) and the surface area of the earth in m^2.
S0 = 1361.0
AREA = 5.1e14

#: Charlson et al. 1992, Table 1 (p. 426): solar constant F_T, area A, and the
#: nominal input values mapped to the columns of NAMES (tau = 0.02 yr = 7.3 days,
#: T = 0.76 squared, 1 - R_s = 0.85 squared) with their relative uncertainties
#: (T and 1 - R_s as given for the unsquared quantities).
S0_CHARLSON = 1370.0
AREA_CHARLSON = 5e14
NOMINAL_CHARLSON = np.array([90., .4, .02*365, 5., .29, 1.7, .76**2, .4, .85**2])
REL_UNC_CHARLSON = np.array([.15, .5, .5, .4, .25, .2, .2, .1, .1])

SULFATE_PER_SULFUR = 3.0  # g SO4(2-) per g S, 96/32
DAYS_PER_YEAR = 365.0


def constant(s0=S0, area=AREA):
    """The factor K in dF_R = K * prod(x_i), in W/m^2 per unit product of inputs."""
    return -0.5 * s0 * SULFATE_PER_SULFUR * 1e12 / (DAYS_PER_YEAR * area)


def burden(x, area=AREA):
    """Column burden B of anthropogenic sulfate, g SO4(2-)/m^2 (Charlson Eq. 5)."""
    x = np.asarray(x, dtype=float)
    return SULFATE_PER_SULFUR * 1e12 * x[..., 0] * x[..., 1] * x[..., 2] / (DAYS_PER_YEAR * area)


def forcing(x, s0=S0, area=AREA):
    """Direct radiative forcing dF_R in W/m^2 at physical inputs x, shape (..., 9)."""
    x = np.asarray(x, dtype=float)
    return constant(s0, area) * np.prod(x, axis=-1)


def grad_log(x, s0=S0, area=AREA):
    """Exact gradient of dF_R with respect to z = ln x, shape (..., 9).

    d(dF_R)/dz_i = x_i d(dF_R)/dx_i = dF_R, because every input enters with power 1.
    """
    x = np.asarray(x, dtype=float)
    return np.repeat(forcing(x, s0, area)[..., None], x.shape[-1], axis=-1)


def grad_physical(x, s0=S0, area=AREA):
    """Exact gradient of dF_R with respect to the physical inputs x, shape (..., 9)."""
    x = np.asarray(x, dtype=float)
    return grad_log(x, s0, area) / x


def log_forcing(x, s0=S0, area=AREA):
    """ln(-dF_R) = ln(-K) + sum_i ln x_i: exactly linear in the log inputs."""
    x = np.asarray(x, dtype=float)
    return np.log(-constant(s0, area)) + np.sum(np.log(x), axis=-1)


def moments(s0=S0, area=AREA):
    """Exact mean and variance of dF_R under the lognormal inputs."""
    K = constant(s0, area)
    s2 = np.sum(SIGMA**2)
    mean = K * np.exp(MU.sum() + s2 / 2)
    var = K**2 * np.exp(2 * MU.sum() + s2) * np.expm1(s2)
    return mean, var
