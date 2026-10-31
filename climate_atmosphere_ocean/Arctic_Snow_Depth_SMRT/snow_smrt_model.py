"""Single-layer dry snow on first-year sea ice, AMSR2 brightness temperatures from SMRT.

The configuration of Yan et al. (2026, The Cryosphere 20:6157-6180, doi:10.5194/tc-20-6157-2026),
Sect. 2.1, 3.3 and 3.4, Tables 1 and 4, and Supplement Table S1: one dry snow layer (sticky hard
spheres microstructure, improved Born approximation with the dense-snow correction) on a two-layer
first-year ice column over a water substrate, flat (Fresnel) interfaces, DORT solver, SMRT 1.5.1,
upwelling brightness temperature at the snow-air interface (no atmosphere) for the 12 AMSR2
channels at 55 deg.

Settings the paper does not state are chosen here and marked "assumed" in CONFIG; DATA.md lists
them. This module needs SMRT (https://github.com/smrt-model/smrt, LGPL-3.0; the shipped data were
made with smrt 1.5.1 from PyPI, the paper's version) and numpy. It contains no SMRT code. The
notebook does not import SMRT.

Inputs (physical units, Table 1 of the paper):
    depth [m], density [kg m^-3], temperature [K], radius [mm], stickiness [-]
Outputs: TB [K] at CHANNELS, in that order.

Two discontinuities of this configuration (DATA.md, "Accuracy"): depth = 0 gives bare ice (SMRT
drops a zero-thickness layer), and the dense-snow correction switches IBA to the inverted medium
(air bubbles in ice) for density > DENSE_THRESHOLD = 916.7/2 kg m^-3, a step in TB of up to tens of K.
"""
import numpy as np

NAMES = ['depth', 'density', 'temperature', 'radius', 'stickiness']
UNITS = ['m', 'kg m^-3', 'K', 'mm', '-']
LOWER = np.array([0.0, 150.0, 240.0, 0.1, 0.1])
UPPER = np.array([0.6, 500.0, 270.0, 0.4, 0.3])

#: AMSR2 channels, SMRT channel codes: 6.925, 10.65, 18.7, 23.8, 36.5, 89.0 GHz, V and H.
FREQ_CODES = ['06', '10', '19', '23', '37', '89']
FREQ_GHZ = [6.925, 10.65, 18.7, 23.8, 36.5, 89.0]
CHANNELS = [f'{c}{p}' for c in ['6', '10', '18', '23', '36', '89'] for p in ['V', 'H']]

#: SMRT's density of ice (kg m^-3); the dense-snow correction applies above half of it.
DENSITY_OF_ICE = 916.7
DENSE_THRESHOLD = 0.5 * DENSITY_OF_ICE

CONFIG = {
    'smrt_version': '1.5.1',                            # Supplement Tab. S1
    'emmodel': 'iba',                                   # paper Sect. 2.1; Tab. S1
    'emmodel_options': {'dense_snow_correction': 'auto'},   # Tab. S1 "with dense snow correction"
    'rtsolver': 'dort',                                 # Tab. S1
    'rtsolver_options': {},                             # assumed: SMRT defaults (n_max_stream=32, m_max=2)
    'theta_deg': 55.0,                                  # paper Sect. 2.1, 3.1
    'snow_microstructure': 'sticky_hard_spheres',       # paper Sect. 2.1; Tab. S1
    'snow_salinity': 0.0,                               # assumed: fresh dry snow (paper Sect. 3.3, Fig. 2f)
    'permittivity_models': 'SMRT 1.5.1 defaults',       # Tab. S1 lists SMRT's defaults
    'interfaces': 'flat',                               # paper Sect. 4.3, p. 6175; Tab. S1 (flat/Fresnel)
    'atmosphere': None,                                 # paper Sect. 2.1: TB at the snow-air interface
    'ice_type': 'firstyear',                            # paper Sect. 3.4 (dry snow on FYI)
    'ice_thickness_m': [0.5, 1.0],                      # Table 4
    'ice_temperature_K': 265.0,                         # Table 4
    'ice_density_kg_m3': 910.0,                         # Table 4 (not used by SMRT for first-year ice)
    'ice_brine_volume_fraction': 0.05,                  # Table 4 (5 %)
    'ice_scatterer_radius_m': 1.0e-4,                   # Table 4 (0.1 mm brine inclusions)
    'ice_microstructure': 'sticky_hard_spheres',        # assumed
    'ice_stickiness': 0.3,                              # assumed (paper's superimposed-ice value, Table 3)
    'brine_inclusion_shape': 'spheres',                 # assumed (SMRT default)
    'water_substrate': 'SMRT default sea water below the ice (271.15 K)',   # Tab. S1 (properties: defaults, assumed)
}


def _smrt():
    import smrt  # noqa: F401  (imported lazily so that the constants above load without SMRT)
    from smrt import make_ice_column, make_model, make_snowpack, sensor_list
    return make_ice_column, make_model, make_snowpack, sensor_list


class SnowOnFYI:
    """Callable SMRT model: X (N, 5) physical inputs -> TB (N, 12) in K."""

    def __init__(self, n_max_stream=32, theta_deg=None, ice_temperature=None, ice_bvf=None,
                 ice_radius_mm=None, ice_stickiness=None):
        """Defaults (None) are the CONFIG values used for the shipped data; the keywords exist so
        that the EMA entry point can expose the held settings as parameters."""
        make_ice_column, make_model, make_snowpack, sensor_list = _smrt()
        self._make_snowpack = make_snowpack
        c = CONFIG
        pick = lambda v, k: c[k] if v is None else float(v)  # noqa: E731
        self.ice = make_ice_column(c['ice_type'], thickness=c['ice_thickness_m'],
                                   temperature=pick(ice_temperature, 'ice_temperature_K'),
                                   microstructure_model=c['ice_microstructure'],
                                   brine_inclusion_shape=c['brine_inclusion_shape'],
                                   brine_volume_fraction=pick(ice_bvf, 'ice_brine_volume_fraction'),
                                   radius=(c['ice_scatterer_radius_m'] if ice_radius_mm is None
                                           else float(ice_radius_mm) * 1e-3),
                                   stickiness=pick(ice_stickiness, 'ice_stickiness'),
                                   density=c['ice_density_kg_m3'], add_water_substrate=True)
        self.sensor = sensor_list.amsr2(channel=FREQ_CODES, theta=pick(theta_deg, 'theta_deg'))
        self.model = make_model(c['emmodel'], c['rtsolver'],
                                emmodel_options=dict(c['emmodel_options']),
                                rtsolver_options=dict(n_max_stream=n_max_stream))

    def snowpack(self, x):
        d, rho, T, r, tau = (float(v) for v in x)
        if not np.all(np.isfinite([d, rho, T, r, tau])):
            raise ValueError(f'non-finite input {x!r}')
        if d < 0 or rho <= 0 or rho >= 916.7 or T <= 0 or T >= 273.15 or r <= 0 or tau <= 0:
            raise ValueError(f'input outside the model limits: {dict(zip(NAMES, x))}')
        # a zero-thickness layer is dropped by SMRT (bare ice); see DATA.md on the d -> 0 limit
        return self._make_snowpack([d], CONFIG['snow_microstructure'], density=rho, temperature=T,
                                   radius=r * 1e-3, stickiness=tau,
                                   salinity=CONFIG['snow_salinity']) + self.ice

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        res = self.model.run(self.sensor, [self.snowpack(x) for x in X])
        tbv = np.asarray(res.TbV()).reshape(len(FREQ_CODES), -1)   # (6, N)
        tbh = np.asarray(res.TbH()).reshape(len(FREQ_CODES), -1)
        out = np.empty((len(X), 12))
        out[:, 0::2] = tbv.T
        out[:, 1::2] = tbh.T
        if not np.all(np.isfinite(out)) or np.any(out <= 0) or np.any(out > 300):
            raise RuntimeError(f'SMRT returned non-physical TB for inputs {X.tolist()}')
        return out


def gr(tb, hi='36V', lo='18V'):
    """Gradient ratio GR(hi/lo) = (TB_hi - TB_lo)/(TB_hi + TB_lo), paper Eq. (12)."""
    a, b = tb[..., CHANNELS.index(hi)], tb[..., CHANNELS.index(lo)]
    return (a - b) / (a + b)


def fd_stencil(x, h):
    """Points for second-order finite differences in normalized step h (fraction of the half-range).

    Central (x +- delta e_j) where both points are in [LOWER, UPPER]; otherwise the second-order
    one-sided stencil (x + s delta, x + 2 s delta), s pointing into the box. Returns the (2m, 5)
    points and the list of (kind, delta) per input, kind 'c' or +-1.
    """
    half = (UPPER - LOWER) / 2
    pts, kinds = [], []
    for j in range(len(x)):
        dlt = h * half[j]
        e = np.zeros(len(x)); e[j] = dlt
        if x[j] - dlt >= LOWER[j] and x[j] + dlt <= UPPER[j]:
            pts += [x + e, x - e]; kinds.append(('c', dlt))
        else:
            s = 1.0 if x[j] - dlt < LOWER[j] else -1.0
            pts += [x + s * e, x + 2 * s * e]; kinds.append((s, dlt))
    return np.array(pts), kinds


def fd_gradient(f0, fpts, kinds):
    """Gradient (outputs, 5) in physical units from the values at fd_stencil points."""
    G = np.empty((f0.shape[-1], len(kinds)))
    for j, (k, dlt) in enumerate(kinds):
        a, b = fpts[2 * j], fpts[2 * j + 1]
        if k == 'c':
            G[:, j] = (a - b) / (2 * dlt)
        else:   # f'(x) ~ s (-3 f0 + 4 f(x + s d) - f(x + 2 s d)) / (2 d)
            G[:, j] = k * (-3 * f0 + 4 * a - b) / (2 * dlt)
    return G
