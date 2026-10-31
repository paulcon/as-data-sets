"""FLORIS 2.0.1 Gauss-Curl Hybrid (GCH) wake model of the Lillgrund wind farm, ported to numpy.

This is a line-by-line numpy port of the parts of FLORIS 2.0.1 (NREL, Apache-2.0,
https://github.com/NREL/floris, tag v2.0.1) that van Beek, Viré and Andersen (Energies
14(5):1293, 2021) used: the `gauss_legacy` velocity deficit with near wake (their Eqs. 1-2,
7-9), the `gauss` deflection model with the natural deflection ad + bd x (Eqs. 3-6, 10),
secondary steering and yaw-added recovery from the wake-rotation vortices (Sec. 2.2.3, Eq. 11),
Crespo-Hernandez added turbulence and sum-of-squares (sosfs) wake combination, all on the
FLORIS 2.0.1 turbine grid (5 x 5 points per rotor, of which the 9 inside the rotor disk are
used for the rotor-averaged speed). Only the zero-yaw (baseline) case is ported.

The port reproduces FLORIS 2.0.1 to rounding error (see DATA.md, "Accuracy"); the arithmetic
follows the FLORIS source line by line, including its quirks:
- `ad` enters the deflection in metres (FLORIS adds `ad + bd*(x - x_turbine)`), not as ad*D
  as the paper's Eq. 10 writes it;
- the secondary-steering effective yaw angle is chosen from the integers -30..30 degrees, and
  the Crespo-Hernandez area overlap counts rotor points with a deficit <= 0.05 m/s, so farm
  power is piecewise smooth with small jumps in the six parameters (DATA.md, "Smoothness");
- rotor points where the field is NaN are dropped from the rotor average, a NaN or infinite
  average becomes 0, and a failed effective-yaw search gives 0 (as in FLORIS 2.0.1).

`farm_power(x, wd, ws)` returns the farm power in W for the six wake parameters
x = (alpha, beta, ka, kb, ad, bd). `farm_power_and_grad` adds the gradient by central
differences with the discrete choices (effective yaw angles, overlap counts) frozen at x.

Only numpy is imported.

Licence: this file is a derivative of FLORIS 2.0.1, Copyright 2020 NREL, licensed under the
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); changes: ported to a
single numpy function for the zero-yaw case on a fixed layout.
"""
import numpy as np

#: Parameter names (Table 2 of the paper), in column order.
NAMES = ('alpha', 'beta', 'ka', 'kb', 'ad', 'bd')
#: Units of the parameters ('m' for ad: FLORIS adds ad in metres).
UNITS = ('-', '-', '-', '-', 'm', '-')
DESCRIPTIONS = ('near-wake length: turbulence weight (Eq. 1)',
                'near-wake length: thrust weight (Eq. 1)',
                'wake expansion: turbulence-intensity weight, k = ka*I + kb (Eq. 9)',
                'wake expansion: base rate (Eq. 9)',
                'natural wake deflection: offset (Eq. 10; metres in FLORIS)',
                'natural wake deflection: slope per metre downstream (Eq. 10)')
#: Bounds of the paper's Table 2 (optimizer bounds, reused as uniform Sobol ranges).
LOWER = np.array([0.125, 0.015, 0.05, 0.0, -1.0, -0.1])
UPPER = np.array([2.5, 0.3, 1.5, 0.02, 1.0, 0.1])
#: FLORIS 2.0.1 default ("tuned") values, the paper's Psi_org.
NOMINAL = np.array([0.58, 0.077, 0.38, 0.004, 0.0, 0.0])

# ---- turbine: Siemens SWT-2.3-93 -------------------------------------------------------------
D = 92.6            # rotor diameter, m (paper Sec. 2.1)
HUB_HEIGHT = 65.0   # m (paper Sec. 2.1)
TSR = 8.0           # tip-speed ratio (FLORIS 2.0.1 example input)
PP = 1.88           # pP = pT (FLORIS 2.0.1 example input; no effect at zero yaw)
#: Generic SWT-2.3-93 power (kW) and thrust curves of PyWake 2.6.20 (MIT,
#: py_wake/examples/data/lillgrund.py). The paper's empirical SCADA curve is under NDA.
_WS = np.arange(3.0, 26.0)
_POWER_KW = np.array([0, 65, 180, 352, 590, 906, 1308, 1767, 2085, 2234, 2283, 2296, 2299,
                      2300, 2300, 2300, 2300, 2300, 2300, 2300, 2300, 2300, 2300], float)
_CT = np.array([0.0, 0.81, 0.84, 0.83, 0.85, 0.86, 0.87, 0.79, 0.67, 0.45, 0.34, 0.26, 0.21,
                0.17, 0.14, 0.12, 0.10, 0.09, 0.07, 0.07, 0.06, 0.05, 0.05])

# ---- ambient flow and held model settings ---------------------------------------------------
RHO = 1.225         # air density, kg/m^3 (FLORIS 2.0.1 example input)
TI = 0.06           # ambient turbulence intensity (FLORIS 2.0.1 example input)
SHEAR = 0.12        # power-law shear exponent (FLORIS 2.0.1 example input)
EPS_GAIN = 0.2      # vortex core size e = 0.2 D (paper Sec. 2.2.3)
YAW_RECOVERY_ALPHA = 0.03   # alpha_r (paper Sec. 2.2.3)
#: Crespo-Hernandez (initial, constant, ai, downstream): FLORIS 2.0.1 example input.
CRESPO_HERNANDEZ = (0.5, 0.9, 0.75, -0.325)

# ---- layout: the 48 Lillgrund turbines -------------------------------------------------------
#: UTM easting/northing (m) of PyWake 2.6.20's Lillgrund example, shifted by X0, Y0.
X0, Y0 = 358805.0, 6152606.0
_UTM_X = [361469, 361203, 360936, 360670, 360404, 360137, 359871, 361203, 360936, 360670,
          360404, 360137, 359871, 359604, 359338, 360936, 360670, 360404, 360137, 359871,
          359604, 359338, 359071, 360670, 360404, 360137, 359871, 359338, 359071, 358805,
          360390, 360137, 359871, 359604, 359071, 358805, 359871, 359604, 359338, 359071,
          358805, 359604, 359338, 359071, 358805, 359338, 359071, 358805]
_UTM_Y = [6154543, 6154244, 6153946, 6153648, 6153349, 6153051, 6152753, 6154695, 6154396,
          6154098, 6153800, 6153501, 6153203, 6152905, 6152606, 6154847, 6154548, 6154250,
          6153952, 6153653, 6153355, 6153057, 6152758, 6154999, 6154701, 6154402, 6154104,
          6153507, 6153209, 6152910, 6155136, 6154853, 6154554, 6154256, 6153659, 6153361,
          6155005, 6154706, 6154408, 6154110, 6153811, 6155157, 6154858, 6154560, 6154262,
          6155309, 6155010, 6154712]
LAYOUT_X = np.array(_UTM_X, float) - X0
LAYOUT_Y = np.array(_UTM_Y, float) - Y0
N_TURBINES = len(LAYOUT_X)

#: The paper's flow case of Fig. 6 (4.8D spacing): wind from 180 deg at 8 m/s.
WD, WS = 180.0, 8.0
#: Turbine spacings (in D) of the paper's Table 1 and one alignment wind direction (deg) each:
#: the directions of the outputs, in column order. 180 deg (4.8 D) is the main case.
SPACINGS = (3.3, 4.3, 4.8, 5.9, 7.1, 8.6)
DIRECTIONS = (120.0, 42.0, 180.0, 75.0, 156.0, 90.0)

_NG = 5                     # FLORIS 2.0.1 rotor grid: 5 x 5 points, 9 inside the disk
_GRID_POINT_COUNT = _NG*_NG


def turbine_table():
    """Wind speeds (m/s), Cp and Ct tables in the form FLORIS 2.0.1 takes them."""
    cp = _POWER_KW*1e3/(0.5*RHO*np.pi*(D/2)**2*_WS**3)
    return _WS.copy(), cp, _CT.copy()


_WS_TAB, _CP_TAB, _CT_TAB = turbine_table()


def _interp(xt, yt, x):
    """scipy.interpolate.interp1d(kind='linear', fill_value='extrapolate') at a scalar x."""
    i = np.clip(np.searchsorted(xt, x, side='left'), 1, len(xt) - 1)
    lo, hi = i - 1, i
    slope = (yt[hi] - yt[lo])/(xt[hi] - xt[lo])
    return float(slope*(x - xt[lo]) + yt[lo])


def _cp(u):
    return 0.0 if u < _WS_TAB[0] else _interp(_WS_TAB, _CP_TAB, u)


def _ct(u):
    if u < _WS_TAB[0]:
        return 0.99
    c = _interp(_WS_TAB, _CT_TAB, u)
    return 0.9999 if c > 1.0 else c


def _cosd(a):
    return np.cos(np.radians(a))


def _sind(a):
    return np.sin(np.radians(a))


def _tand(a):
    return np.tan(np.radians(a))


class _Geometry:
    """Flow grid, rotated coordinates and turbine order for one wind direction (FLORIS 2.0.1)."""

    def __init__(self, wd):
        R = D/2
        theta = ((wd - 270) % 360 + 360) % 360
        off = np.linspace(-R, R, _NG)
        # FlowField._discretize_turbine_domain
        x = np.zeros((N_TURBINES, _NG, _NG))
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        for i in range(N_TURBINES):
            xt, yt0 = LAYOUT_X[i], LAYOUT_Y[i]
            yt = np.linspace(yt0 - R, yt0 + R, _NG)
            zt = np.linspace(HUB_HEIGHT - R, HUB_HEIGHT + R, _NG)
            for j in range(_NG):
                for k in range(_NG):
                    xg, yg = xt, yt[j]
                    xo, yo = xg - xt, yg - yt0
                    x[i, j, k] = xo*_cosd(-1*theta) - yo*_sind(-1*theta) + xt
                    y[i, j, k] = yo*_cosd(-1*theta) + xo*_sind(-1*theta) + yt0
                    z[i, j, k] = zt[k]
        # FlowField.calculate_wake: rotation about the centre of the grid
        cx = np.mean([np.min(x), np.max(x)])
        cy = np.mean([np.min(y), np.max(y)])
        xo, yo = LAYOUT_X - cx, LAYOUT_Y - cy
        self.tx = xo*_cosd(theta) - yo*_sind(theta) + cx      # rotated turbine coordinates
        self.ty = yo*_cosd(theta) + xo*_sind(theta) + cy
        xo, yo = x - cx, y - cy
        ang = np.full(x.shape, theta)
        self.x = xo*_cosd(ang) - yo*_sind(ang) + cx              # rotated flow grid
        self.y = xo*_sind(ang) + yo*_cosd(ang) + cy
        self.z = z
        self.order = sorted(range(N_TURBINES), key=lambda i: self.tx[i])   # sorted_in_x_as_list
        # rotor points inside the disk (Turbine._create_swept_area_grid), in FLORIS order;
        # each coincides with its own turbine's flow-grid point (nearest-neighbour lookup)
        self.rotor = [(j, k) for k in range(_NG) for j in range(_NG)
                      if np.hypot(off[j], off[k]) < R]
        self.rj = np.array([j for j, _ in self.rotor])
        self.rk = np.array([k for _, k in self.rotor])
        zl = np.linspace(np.min(self.z), np.max(self.z), _NG)
        self.zlin = zl


_GEOM = {}


def _geometry(wd):
    key = float(wd)
    if key not in _GEOM:
        _GEOM[key] = _Geometry(key)
    return _GEOM[key]


def _check(x, ti=TI):
    x = np.asarray(x, float)
    if x.shape != (6,):
        raise ValueError('expected the six parameters (alpha, beta, ka, kb, ad, bd)')
    if not np.all(np.isfinite(x)):
        raise ValueError('parameters must be finite')
    alpha, beta, ka, kb, ad, bd = x
    if alpha <= 0 or beta <= 0:
        raise ValueError('alpha and beta must be positive (near-wake length, Eq. 1)')
    if ka*ti + kb <= 0:
        raise ValueError('the wake expansion rate ka*TI + kb must be positive (Eq. 9)')
    return x


def turbine_powers(x, wd=WD, ws=WS, ti=TI, shear=SHEAR, frozen=None, return_state=False):
    """Powers (W) of the 48 turbines, in layout order, for parameters x at wind (wd deg, ws m/s).

    ti, shear: ambient turbulence intensity and power-law shear exponent (held settings).

    frozen: a state returned by an earlier call with return_state=True; its discrete choices
    (effective yaw angles and area-overlap counts) are reused instead of recomputed.
    """
    alpha, beta, ka, kb, ad, bd = _check(x, ti)
    if not (np.isfinite(wd) and np.isfinite(ws) and ws > 0):
        raise ValueError('wind direction must be finite and wind speed positive')
    if not (np.isfinite(ti) and ti > 0 and np.isfinite(shear) and shear >= 0):
        raise ValueError('turbulence intensity must be positive and shear exponent >= 0')
    g = _geometry(wd)
    X, Y, Z = g.x, g.y, g.z
    R = D/2
    u_initial = ws*(Z/HUB_HEIGHT)**shear
    grid_ws = np.full(X.shape, float(ws))
    dudz = np.gradient(u_initial, g.zlin, axis=2)
    kappa, lmda = 0.41, D/8
    lm = kappa*Z/(1 + kappa*Z/lmda)
    nu = lm**2*np.abs(dudz[0, :, :])
    eps = EPS_GAIN*D
    ch_init, ch_const, ch_ai, ch_down = CRESPO_HERNANDEZ

    TI0 = ti
    ti = np.full(N_TURBINES, TI0)
    u_wake = np.zeros(X.shape)
    v = np.zeros(X.shape)
    avg = np.zeros(N_TURBINES)
    state = {'yaw': {}, 'count': {}} if frozen is None else frozen
    yaw = 0.0
    n_nan = 0
    gX, gY, gZ = X, Y, Z
    test_gamma = np.linspace(-30, 30, 61)
    trig = _sind(test_gamma)*_cosd(test_gamma)**2
    with np.errstate(divide='ignore', invalid='ignore'):
        for pos, i in enumerate(g.order):
            x1, x2 = g.tx[i], g.ty[i]
            # Turbine.update_velocities / average_velocity / Ct / aI
            vel = (u_initial[i] - u_wake[i])[g.rj, g.rk]
            data = vel[np.isnan(vel) == False]          # noqa: E712 (FLORIS drops NaN points)
            n_nan += vel.size - data.size
            avg[i] = np.cbrt(np.mean(data**3)) if data.size else np.nan
            if np.isnan(avg[i]) or np.isinf(avg[i]):    # FLORIS: average_velocity = 0
                avg[i] = 0.0
            rest = np.array(g.order[pos + 1:], dtype=int)
            if rest.size == 0:
                break
            U = avg[i]
            Ct = _ct(U)*_cosd(yaw)
            aI = 0.5/_cosd(yaw)*(1 - np.sqrt(1 - Ct*_cosd(yaw)))
            TIi = ti[i]

            # ---- secondary steering: effective yaw (base_velocity_deflection.py) ---------
            # Only the turbine's own rotor points satisfy the selection (turbines are >= 3.3 D
            # apart), so the search is done on them.
            if frozen is None:
                yL = gY[i] - x2
                zL = gZ[i] - HUB_HEIGHT
                dist = np.sqrt(yL**2 + zL**2)
                xL = np.abs(gX[i] - x1)
                m = (dist < D/2) & (xL < D/4) & (np.abs(yL) > 0.1)
                Gamma = v[i][m]*((2*np.pi)*(yL[m]**2 + zL[m]**2)) \
                    / (yL[m]*(1 - np.exp(-(yL[m]**2 + zL[m]**2)/(eps**2))))
                Gamma0 = np.mean(np.abs(Gamma))
                tmp1 = 8*Gamma0/(np.pi*RHO*D*U*Ct)
                tmp = np.abs(trig - tmp1)
                if np.all(np.isfinite(tmp)):
                    state['yaw'][i] = test_gamma[int(np.argmin(tmp))] + yaw   # first minimum, as FLORIS
                else:   # FLORIS: no candidate beats 10000, the lookup fails and it uses 0
                    state['yaw'][i] = 0.0 + yaw

            # From here on, fields are computed only at the flow points of the turbines later in
            # the sweep order; FLORIS computes them everywhere, but no other value is read again.
            X, Y, Z = gX[rest], gY[rest], gZ[rest]
            uI = u_initial[rest]

            # ---- deflection (wake_deflection/gauss.py) -----------------------------------
            yaw_d = -1*state['yaw'][i]
            uR = uI*Ct*_cosd(0.0)*_cosd(yaw_d)/(2.*(1 - np.sqrt(1 - (Ct*_cosd(0.0)*_cosd(yaw_d)))))
            u0 = uI*np.sqrt(1 - Ct)
            x0 = D*(_cosd(yaw_d)*(1 + np.sqrt(1 - Ct*_cosd(yaw_d))))/(
                np.sqrt(2)*(4*alpha*TIi + 2*beta*(1 - np.sqrt(1 - Ct)))) + x1
            ky = ka*TIi + kb
            kz = ka*TIi + kb
            C0 = 1 - u0/grid_ws[rest]
            M0 = C0*(2 - C0)
            E0 = C0**2 - 3*np.exp(1./12.)*C0 + 3*np.exp(1./3.)
            sigma_z0 = D*0.5*np.sqrt(uR/(uI + u0))
            sigma_y0 = sigma_z0*_cosd(yaw_d)*_cosd(0.0)
            yR = Y - x2
            xR = yR*_tand(yaw_d) + x1
            theta_c0 = 1.0*(0.3*np.radians(yaw_d)/_cosd(yaw_d))*(1 - np.sqrt(1 - Ct*_cosd(yaw_d)))
            delta0 = np.tan(theta_c0)*(x0 - x1)
            dnw = ((X - xR)/(x0 - xR))*delta0 + (ad + bd*(X - x1))
            dnw[X < xR] = 0.0
            dnw[X > x0] = 0.0
            sy = ky*(X - x0) + sigma_y0
            sz = kz*(X - x0) + sigma_z0
            sy[X < x0] = sigma_y0[X < x0]
            sz[X < x0] = sigma_z0[X < x0]
            lnN = (1.6 + np.sqrt(M0))*(1.6*np.sqrt(sy*sz/(sigma_y0*sigma_z0)) - np.sqrt(M0))
            lnD = (1.6 - np.sqrt(M0))*(1.6*np.sqrt(sy*sz/(sigma_y0*sigma_z0)) + np.sqrt(M0))
            dfw = delta0 + (theta_c0*E0/5.2)*np.sqrt(sigma_y0*sigma_z0/(ky*kz*M0)) \
                * np.log(lnN/lnD) + (ad + bd*(X - x1))
            dfw[X <= x0] = 0.0
            delta = dnw + dfw

            # ---- velocity deficit (wake_velocity/gaussianModels/gauss_legacy.py) ---------
            yaw_v = -1*yaw
            xRv = (Y - x2)*_tand(yaw_v) + x1
            uRv = uI*Ct/(2.0*(1 - np.sqrt(1 - Ct)))
            u0v = uI*np.sqrt(1 - Ct)
            yaw_e = -1*yaw
            s_z0 = D*0.5*np.sqrt(uRv/(uI + u0v))
            s_y0 = s_z0*_cosd(yaw_e)*_cosd(0.0)
            x0v = D*(_cosd(yaw_v)*(1 + np.sqrt(1 - Ct)))/(np.sqrt(2)*(
                4*alpha*TIi + 2*beta*(1 - np.sqrt(1 - Ct)))) + x1
            s_y = (((x0v - xRv) - (X - xRv))/(x0v - xRv))*0.501*D*np.sqrt(Ct/2.0) \
                + ((X - xRv)/(x0v - xRv))*s_y0
            s_z = (((x0v - xRv) - (X - xRv))/(x0v - xRv))*0.501*D*np.sqrt(Ct/2.0) \
                + ((X - xRv)/(x0v - xRv))*s_z0
            s_y[X < xRv] = 0.5*D
            s_z[X < xRv] = 0.5*D
            a = _cosd(0.0)**2/(2*s_y**2) + _sind(0.0)**2/(2*s_z**2)
            b = -_sind(2*0.0)/(4*s_y**2) + _sind(2*0.0)/(4*s_z**2)
            c = _sind(0.0)**2/(2*s_y**2) + _cosd(0.0)**2/(2*s_z**2)
            r = a*((Y - x2) - delta)**2 - 2*b*((Y - x2) - delta)*(Z - HUB_HEIGHT) \
                + c*(Z - HUB_HEIGHT)**2
            C = 1 - np.sqrt(1 - (Ct*_cosd(yaw_v)/(8.0*s_y*s_z/D**2)))
            velDef = uI*C*np.exp(-1*r**1/(2*np.sqrt(0.5)**2))
            velDef[X < xRv] = 0
            velDef[X > x0v] = 0
            s_y = ky*(X - x0v) + s_y0
            s_z = kz*(X - x0v) + s_z0
            s_y[X < x0v] = s_y0[X < x0v]
            s_z[X < x0v] = s_z0[X < x0v]
            a = _cosd(0.0)**2/(2*s_y**2) + _sind(0.0)**2/(2*s_z**2)
            b = -_sind(2*0.0)/(4*s_y**2) + _sind(2*0.0)/(4*s_z**2)
            c = _sind(0.0)**2/(2*s_y**2) + _cosd(0.0)**2/(2*s_z**2)
            r = a*(Y - x2 - delta)**2 - 2*b*(Y - x2 - delta)*(Z - HUB_HEIGHT) \
                + c*(Z - HUB_HEIGHT)**2
            C = 1 - np.sqrt(1 - (Ct*_cosd(yaw_v)/(8.0*s_y*s_z/D**2)))
            velDef1 = uI*C*np.exp(-1*r**1/(2*np.sqrt(0.5)**2))
            velDef1[X < x0v] = 0
            turb_u = np.sqrt(velDef**2 + velDef1**2)

            # ---- wake-rotation vortices (gaussian_model_base.calc_VW), zero yaw ----------
            Uinf = float(ws)
            Gwr = 0.5*2*np.pi*D*(aI - aI**2)*U/TSR
            yL = Y + 0.01 - x2
            zL = Z + 0.01 - HUB_HEIGHT
            fac = eps**2/(4*nu[rest]*(X - x1)/Uinf + eps**2)
            rr = yL**2 + zL**2
            V5 = (((yL*Gwr)/(2*np.pi*rr))*(1 - np.exp(-rr/(eps**2))) + 0.0)*fac
            W5 = ((zL*Gwr)/(2*np.pi*rr))*(1 - np.exp(-rr/(eps**2)))*fac
            zL = Z + 0.01 + HUB_HEIGHT
            rr = yL**2 + zL**2
            V6 = (((yL*Gwr)/(2*np.pi*rr))*(1 - np.exp(-rr/(eps**2))) + 0.0)*fac
            W6 = ((zL*Gwr)/(2*np.pi*rr))*(1 - np.exp(-rr/(eps**2)))*fac
            Vt = V5 + V6          # the yaw vortices V1-V4, W1-W4 are exactly 0 at zero yaw
            Wt = W5 + W6
            Vt[X < x1 + 10] = 0.0
            Wt[X < x1 + 10] = 0.0
            Vt[np.abs(Y - x2) > D] = 0.0
            Wt[np.abs(Y - x2) > D] = 0.0

            # ---- yaw-added recovery (gaussian_model_base.yaw_added_recovery_correction) --
            U1 = uI - turb_u
            xLocs = X - x1
            yLocs = Y - x2
            U2 = (-1*Wt*xLocs*np.abs(yLocs))/(np.pi*(YAW_RECOVERY_ALPHA*xLocs + D/2)**2)
            U_total = U1 + np.nan_to_num(U2)
            turb_u = uI - U_total
            turb_u[X < x1] = 0

            # ---- Crespo-Hernandez added turbulence (flow_field.calculate_wake) -----------
            for q, i2 in enumerate(rest):
                if g.tx[i2] > x1 and np.abs(x2 - g.ty[i2]) < 2*D:
                    if frozen is None:
                        free = uI[q][g.rj, g.rk]
                        wake = (uI[q] - turb_u[q])[g.rj, g.rk]
                        state['count'][(i, i2)] = int(np.sum(free - wake <= 0.05))
                    overlap = (_GRID_POINT_COUNT - state['count'][(i, i2)])/_GRID_POINT_COUNT
                    if overlap > 0.0 and g.tx[i2] <= 15*D + x1:
                        ti_calc = ch_const*aI**ch_ai*TI0**ch_init*((g.tx[i2] - x1)/D)**ch_down
                        ti[i2] = np.sqrt((overlap*ti_calc)**2 + TIi**2)

            # ---- sosfs combination --------------------------------------------------------
            u_wake[rest] = np.hypot(turb_u, u_wake[rest])
            v[rest] = np.hypot(Vt, v[rest])

    P = np.array([0.5*RHO*(np.pi*R**2)*_cp(u)*1.0*1.0*u**3 for u in avg])
    if not np.all(np.isfinite(P)):
        raise RuntimeError('non-finite turbine power')
    if return_state:
        return P, {'yaw': dict(state['yaw']), 'count': dict(state['count']), 'avg': avg, 'ti': ti,
                   'n_nan': n_nan}
    return P


def farm_power(x, wd=WD, ws=WS, ti=TI, shear=SHEAR):
    """Total power of the 48 turbines, W."""
    return float(np.sum(turbine_powers(x, wd, ws, ti, shear)))


def free_power(ws=WS, shear=SHEAR):
    """Power (W) of one turbine in the undisturbed sheared inflow (FLORIS rotor average)."""
    z = np.linspace(HUB_HEIGHT - D/2, HUB_HEIGHT + D/2, _NG)
    u = ws*(z/HUB_HEIGHT)**shear
    off = np.linspace(-D/2, D/2, _NG)
    vel = np.array([u[k] for k in range(_NG) for j in range(_NG) if np.hypot(off[j], off[k]) < D/2])
    U = np.cbrt(np.mean(vel**3))
    return 0.5*RHO*(np.pi*(D/2)**2)*_cp(U)*U**3


def farm_power_and_grad(x, wd=WD, ws=WS, ti=TI, shear=SHEAR, h=1e-4):
    """Farm power (W) and its gradient w.r.t. the physical parameters (W per unit).

    Central differences with step h*(UPPER - LOWER)/2 in each parameter, with the discrete
    choices of FLORIS (secondary-steering yaw angles, Crespo-Hernandez overlap counts)
    frozen at x, so the stencil never straddles a jump.
    """
    x = _check(x, ti)
    P, st = turbine_powers(x, wd, ws, ti, shear, return_state=True)
    half = (UPPER - LOWER)/2
    g = np.empty(6)
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = h*half[j]
        fp = np.sum(turbine_powers(x + dx, wd, ws, ti, shear, frozen=st))
        fm = np.sum(turbine_powers(x - dx, wd, ws, ti, shear, frozen=st))
        g[j] = (fp - fm)/(2*dx[j])
    return float(np.sum(P)), g
