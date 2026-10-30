"""Single-diode solar cell model: maximum power P_max, numpy/scipy only.

A Python re-implementation of the model in Constantine, Zaharatos and
Campanelli (2015), eqs. (9)-(11) with N_S = 1. The shipped data
SingleDiodePV-Pmax.txt came from Campanelli's MATLAB code, not from this
module; the notebook uses it only to check the stored values.

Inputs p = (I_SC [A], I_S [A], n [-], R_S [ohm], R_P [ohm]). The current
I(V) is solved explicitly with the Lambert W function, and P_max = max_V V I(V)
is found by a bounded scalar search on [0, V_oc].
"""
import numpy as np
from scipy.constants import Boltzmann, elementary_charge
from scipy.optimize import brentq, minimize_scalar
from scipy.special import lambertw

# Table 1 bounds in the coordinates (I_SC, log I_S, n, R_S, R_P).
XL = np.array([0.05989, np.log(2.2e-11), 1.0, 0.16625, 93.75])
XU = np.array([0.23958, np.log(2.2e-7), 2.0, 0.665, 375.0])


def thermal_voltage(T=25.0):
    """V_th = k_B T / q [V] at temperature T [deg C]."""
    return Boltzmann * (273.15 + T) / elementary_charge


def _w_of_exp(arg):
    """W(exp(arg)) without overflow for large arg (Newton on w + log w = arg)."""
    if arg < 500.0:
        return lambertw(np.exp(arg)).real
    w = arg - np.log(arg)
    for _ in range(4):
        w -= (w + np.log(w) - arg) / (1.0 + 1.0 / w)
    return w


def current(V, p, T=25.0):
    """Current I [A] at voltage V [V], from eq. (9) with I_L from eq. (10)."""
    Isc, Is, n, Rs, Rp = p
    a = n * thermal_voltage(T)
    IL = Isc + Is * np.expm1(Isc * Rs / a) + Isc * Rs / Rp
    arg = np.log(Rs * Rp * Is / (a * (Rs + Rp))) + Rp * (Rs * (IL + Is) + V) / (a * (Rs + Rp))
    return (Rp * (IL + Is) - V) / (Rs + Rp) - (a / Rs) * _w_of_exp(arg)


def pmax(p, T=25.0):
    """Maximum power P_max = max_V V I(V) [W] for physical inputs p."""
    Voc = brentq(lambda V: current(V, p, T), 0.0, 5.0, xtol=1e-15)
    r = minimize_scalar(lambda V: -V * current(V, p, T), bounds=(0.0, Voc),
                        method='bounded', options={'xatol': 1e-12})
    return -r.fun


def to_physical(x):
    """Map normalized x in [-1, 1]^5 to physical p (I_S from log I_S)."""
    q = XL + (np.asarray(x, dtype=float) + 1.0) * (XU - XL) / 2.0
    q[..., 1] = np.exp(q[..., 1])
    return q


def to_normalized(p):
    """Map physical p to normalized x in [-1, 1]^5 (I_S through its log)."""
    q = np.array(p, dtype=float)
    q[..., 1] = np.log(q[..., 1])
    return 2.0 * (q - XL) / (XU - XL) - 1.0
