"""Analytic cathode-coupling and plume components of the coupled Hall thruster model of
Marks, Eckels, Mora & Gorodetsky, J. Appl. Phys. (2025), doi:10.1063/5.0283796.

Written from the equations in the paper (Eqs. 2, 5-7), the cathode model of Jorns & Byrne (2021,
doi:10.1088/1361-6595/abd3b6) and the semi-empirical plume model the paper cites (its Refs. 10, 13:
Eckels et al. 2024, doi:10.1007/s44205-024-00079-w; Allen et al., IEPC 2022). numpy/scipy only.
Part of the as-data-sets repository (MIT licence for code).
"""
import numpy as np

TORR_TO_PA = 101325.0 / 760.0

# Gauss-Legendre nodes on [0, pi/2] for the angular integrals
_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)
PHI = 0.25 * np.pi * (_GL_X + 1.0)
W_PHI = 0.25 * np.pi * _GL_W


def cathode_coupling_voltage(PB_Torr, Tec, Vvac, PT_uTorr, Pstar_uTorr, Vd):
    """Eq. (2): Vcc = Vvac + Tec*log(1 + PB/PT) - Tec*PB/(PT + P*), clipped to [0, Vd].

    Returns (Vcc, clipped) where clipped flags values outside [0, Vd] before clipping.
    """
    PB = np.asarray(PB_Torr) * 1.0e6  # uTorr, same units as PT and P*
    raw = Vvac + Tec * np.log1p(PB / PT_uTorr) - Tec * PB / (PT_uTorr + Pstar_uTorr)
    clipped = (raw < 0) | (raw > Vd)
    return np.clip(raw, 0.0, Vd), clipped


def _gauss_norm(alpha):
    """2*pi * int_0^{pi/2} exp(-(phi/alpha)^2) sin(phi) dphi  (hemisphere normalisation)."""
    alpha = np.atleast_1d(alpha)[:, None]
    return 2 * np.pi * np.sum(W_PHI * np.exp(-(PHI / alpha) ** 2) * np.sin(PHI), axis=1)


def plume(PB_Torr, IB, T, c0, c1, c2, c3, c4, c5, sigma_cex=55.0e-20, r=1.0):
    """Semi-empirical plume model (Eqs. 5-7).

    j_ion(r, phi) = j_beam + j_scatter + j_cex, with Gaussian angular profiles for the main beam
    (divergence alpha1 = c2*PB + c3, capped at pi/2) and the scattered population (alpha2 = alpha1/c1),
    a uniform hemisphere of charge-exchange ions with fraction 1 - exp(-r*n*sigma_cex), and neutral
    density n = c4*PB + c5 (PB in Pa).

    The divergence angle is computed from the beam + scattered ions (charge-exchange ions excluded):
        cos(phi_d) = int j cos(phi) sin(phi) dphi / int j sin(phi) dphi,
    and Tc = T cos(phi_d) (Eq. 7).

    Returns dict with Tc [N], phi_d [rad], j_peak [A/m^2] (centreline current density at r) and
    alpha1 [rad]. Vectorised over the inputs (1-D arrays of equal length).
    """
    PB = np.atleast_1d(PB_Torr) * TORR_TO_PA
    IB, T, c0, c1, c2, c3, c4, c5 = (np.atleast_1d(v).astype(float) for v in (IB, T, c0, c1, c2, c3, c4, c5))
    PB = np.broadcast_to(PB, IB.shape)
    n = c4 * PB + c5
    alpha1 = np.minimum(c2 * PB + c3, np.pi / 2)
    if np.any(alpha1 <= 0):
        raise ValueError("non-positive main-beam divergence angle alpha1 = c2*PB + c3")
    alpha2 = alpha1 / c1
    A1 = (1 - c0) / _gauss_norm(alpha1)
    A2 = c0 / _gauss_norm(alpha2)
    decay = np.exp(-r * n * sigma_cex)
    base = (IB * decay / r**2)[:, None]
    jb = base * A1[:, None] * np.exp(-(PHI / alpha1[:, None]) ** 2)
    js = base * A2[:, None] * np.exp(-(PHI / alpha2[:, None]) ** 2)
    jnc = jb + js
    num = np.sum(W_PHI * jnc * np.cos(PHI) * np.sin(PHI), axis=1)
    den = np.sum(W_PHI * jnc * np.sin(PHI), axis=1)
    cosd = num / den
    jcex = IB * (1 - decay) / (2 * np.pi * r**2)
    j_peak = IB * decay / r**2 * (A1 + A2) + jcex
    return {"Tc": T * cosd, "phi_d": np.arccos(cosd), "j_peak": j_peak, "alpha1": alpha1}
