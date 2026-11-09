"""PyGEM v0.1.0 glacier mass balance for one glacier, ported to numpy.

A re-implementation of `runmassbalance` (with `massredistributionHuss`,
`massredistributioncurveHuss`, `surfacetypebinsinitial`, `surfacetypebinsannual` and
`surfacetypeDDFdict`) in pygemfxns_massbalance.py of PyGEM v0.1 (D. R. Rounce, MIT licence,
https://github.com/drounce/PyGEM, tag v0.1, commit 4d710538dae04fc178a383cafaae04198e3e88dd), with
the model options of that version's pygem_input.py that Rounce et al. (2020, J. Glaciol. 66, 175-187)
describe in their Sec. 2: monthly time step on 10 m elevation bins; temperature downscaled with lapse
rates plus T_bias (Eq. 4); precipitation times k_p with a precipitation gradient (Eq. 7) and the
Huss and Hock (2015) limit on the uppermost 25 % of a glacier with more than 1000 m of relief;
snow/rain split linear within T_snow +/- 1 degC (Eq. 6); degree-day melt of snow, firn and ice
(Eq. 3) with f_ice = f_snow / 0.7 and f_firn their mean (Sec. 2.1); Woodward et al. (1997) refreeze
reset every October (Eq. 8); annual surface-type update from the 5-year running mean of each bin's
climatic mass balance; annual Huss and Hock (2015) mass redistribution with a parabolic cross-section
(retreat and advance). Frontal ablation is omitted (the glacier here is land-terminating), and so is
the off-glacier bookkeeping that does not enter the glacier-wide mass balance. The licence and
copyright notice of PyGEM are in LICENSE-PyGEM.txt beside this file.

The quantity of interest is the glacier-wide specific mass balance over the 18 calendar years
2000-2017, B (m w.e. a-1): total mass change divided by the initial area and the number of years,
as in PyGEM v0.1's run_calibration.py.

Inputs are physical: x = (k_p, T_bias, f_snow) or x = (k_p, T_bias, f_snow, d_prec, T_snow) with
k_p (-), T_bias (degC), f_snow (mm w.e. d-1 degC-1), d_prec (m-1, i.e. 1e-4 m-1 = 0.01 % m-1) and
T_snow (degC). Gradients use the complex step: every branch (melt only above 0 degC, melt limited by
the snowpack, refreeze limits, surface type, retreat and advance) is decided on the real part, so the
derivative is that of the branch the model takes at x, exact to rounding. B jumps by small amounts where
a branch changes (for example when a bin's surface type flips); the gradient ignores those jumps.

Only numpy is imported. Files are read relative to this module.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


_BINS = _load('glacier_bins.csv')
_CLIM = _load('climate_monthly.csv')
_SITE = _load('site.csv')[0]

Z_BINS = _BINS[:, 0]                    # m, bin centres
AREA0 = _BINS[:, 1]                     # km2
THICK0 = _BINS[:, 2]                    # m
YEARS = _CLIM[:, 0].astype(int)
MONTHS = _CLIM[:, 1].astype(int)
DAYS = _CLIM[:, 2].astype(int)
T_GCM = _CLIM[:, 3] - 273.15            # degC
P_GCM = _CLIM[:, 4] * DAYS              # m per month
Z_GCM = float(_SITE[2])                 # m, ERA5 grid-cell height
Z_MED = float(_SITE[3])                 # m, glacier median elevation (reference elevation of Eqs. 4, 7)
NYEARS = len(T_GCM) // 12

# Settings of PyGEM v0.1 (pygem_input.py) and of the paper (Sec. 2.1, p. 177)
LAPSE_RATE = -0.0065        # K m-1 (lrgcm = lrglac; pygem_input.py default; the paper used ERA-Interim levels)
DDF_ICE_RATIO = 0.7         # f_snow = 0.7 f_ice (p. 177)
DENSITY_ICE = 900.0         # kg m-3
DENSITY_WATER = 1000.0      # kg m-3
RF_MONTH = 10               # refreeze potential reset in October (p. 176)
TOLERANCE = 1e-12
ADVANCE_THRESHOLD = 5.0     # m
TERMINUS_PERCENTAGE = 20.0  # %
PRECGRAD_FIXED = 1e-4       # m-1 (0.01 % m-1, p. 177)
TSNOW_FIXED = 1.0           # degC (p. 177)

NAMES3 = ('k_p', 'T_bias', 'f_snow')
NAMES5 = NAMES3 + ('d_prec', 'T_snow')
UNITS = {'k_p': '-', 'T_bias': 'degC', 'f_snow': 'mm w.e. d-1 degC-1', 'd_prec': 'm-1', 'T_snow': 'degC'}
NOMINAL = {'k_p': 1.0, 'T_bias': 0.0, 'f_snow': 4.1, 'd_prec': PRECGRAD_FIXED, 'T_snow': TSNOW_FIXED}


def _pos(a):
    """Real part > 0, as a boolean mask."""
    return np.real(a) > 0


def _nz(a):
    return np.nonzero(np.real(a))[0]


def _ppow(a, p):
    """a**p where Re(a) > 0, else 0 (keeps the complex step clean at zero thickness)."""
    out = np.zeros_like(a)
    m = _pos(a)
    out[m] = a[m] ** p
    return out


def _curve(h0, a0, g, dv, mbclim):
    """massredistributioncurveHuss with the parabolic glacier shape (option_glaciershape = 1)."""
    h1 = np.zeros_like(a0)
    a1 = np.zeros_like(a0)
    if g.shape[0] > 3:
        atot = np.real(a0.sum())
        if atot > 20:
            gamma, a, b, c = 6, -0.02, 0.12, 0
        elif atot > 5:
            gamma, a, b, c = 4, -0.05, 0.19, 0.01
        else:
            gamma, a, b, c = 2, -0.30, 0.60, 0.09
        on = _pos(a0)
        er = np.zeros(a0.shape)
        er[on] = (g[-1] - g) / (g[-1] - g[0])
        dhn = np.zeros(a0.shape)
        dhn[on] = (er[on] + a) ** gamma + b * (er[on] + a) + c
        dhn[dhn > 1] = 1
        dhn[dhn < 0] = 0
        fs = dv / (a0 * dhn).sum() * 1000
        bin_dv = dhn * fs / 1000 * a0
    else:
        bin_dv = mbclim / 1000 * a0
    h1[g] = (h0[g] / 1000) ** 1.5 + (h0[g] / 1000) ** 0.5 * bin_dv[g] / a0[g]
    h1[np.real(h1) < 0] = 0
    h1[g] = _ppow(h1[g], 2 / 3) * 1000
    a1[g] = a0[g] * _ppow(h1[g] / h0[g], 0.5)
    dh = h1 - h0
    rem = bin_dv - (a1 * h1 - a0 * h0) / 1000
    rem[np.abs(np.real(rem)) < TOLERANCE] = 0
    return h1, a1, dh, rem.sum()


def _redistribute(a0, h0, mbclim_year, g_initial, area_initial):
    """massredistributionHuss: retreat and advance of the glacier after one year."""
    a1 = np.zeros_like(a0)
    h1 = np.zeros_like(a0)
    dv = (mbclim_year / 1000 * DENSITY_WATER / DENSITY_ICE * a0).sum()         # km3 ice
    if -np.real(dv) < np.real((h0 / 1000 * a0).sum()):
        g = _nz(a0)
        h1, a1, dh, rem = _curve(h0, a0, g, dv, mbclim_year)
        # retreat: spread the volume change left over where bins melted out
        while np.real(rem) < 0:
            ar, hr = a1.copy(), h1.copy()
            gr = _nz(ar)
            mb_r = np.zeros_like(ar)
            mb_r[gr] = rem / ar.sum() * 1000
            h1, a1, dh, rem = _curve(hr, ar, gr, rem, mb_r)
        # advance: thickness gains above the threshold become new area at the terminus
        while (np.real(dh) > ADVANCE_THRESHOLD).any():
            a1_raw, h1_raw = a1.copy(), h1.copy()
            dh[np.real(dh) <= ADVANCE_THRESHOLD] = 0
            adv = _nz(dh)
            h1[adv] = h1[adv] - (dh[adv] - ADVANCE_THRESHOLD)
            a1[adv] = a1_raw[adv] * (h1[adv] / h1_raw[adv]) ** 0.5
            vol = (a1_raw[adv] * h1_raw[adv] / 1000).sum() - (a1[adv] * h1[adv] / 1000).sum()
            term = g[(g - g[0] + 1) / g.shape[0] * 100 < TERMINUS_PERCENTAGE]
            if term.shape[0] <= 1:
                term = g.copy()
            if term.shape[0] >= 2:
                tavg = a0[term[1]:term[-1] + 1].mean()
            else:
                ti = g_initial[(g_initial - g_initial[0] + 1) / g_initial.shape[0] * 100 < TERMINUS_PERCENTAGE]
                if ti.shape[0] <= 1:
                    ti = g_initial.copy()
                tavg = area_initial[ti[1]:ti[-1] + 1].mean()
            t0 = term[0]
            if np.real(a1[t0]) < np.real(tavg) and np.real(h0[t0]) < np.real(h0[g].mean()):
                fill = h1[t0] / 1000 * (tavg - a1[t0])
                if np.real(vol) < np.real(fill):
                    a1[t0] = a1[t0] + vol / (h1[t0] / 1000)
                    vol = 0 * vol
                else:
                    a1[t0] = a1[t0] + fill / (h1[t0] / 1000)
                    vol = vol - fill
            if np.real(vol) > 0:
                b2a = t0 - 1
                while b2a > g_initial.min() and not (b2a == g_initial).any():
                    b2a -= 1
                if b2a < 0:
                    a1, h1 = a1_raw, h1_raw
                    vol = 0 * vol
                else:
                    h1[b2a] = h1[t0]
                    fill = h1[b2a] / 1000 * tavg
                    if np.real(vol) < np.real(fill):
                        a1[b2a] = vol / (h1[b2a] / 1000)
                        vol = 0 * vol
                    else:
                        a1[b2a] = tavg
                        vol = vol - fill
            g = _nz(a1)
            a1_raw, h1_raw = a1.copy(), h1.copy()
            if np.real(vol) > 0:
                h1, a1, dh, rem = _curve(h1, a1, g, vol, np.zeros_like(a1))
            dh = h1 - h1_raw
    return a1, h1


def _surface_type_annual(st, mbclim_annual, year):
    lo = 0 if year < 5 else year - 4
    avg = np.real(mbclim_annual[:, lo:year + 1].mean(axis=1))
    st[(st != 0) & (avg <= 0)] = 1
    st[(st != 0) & (avg > 0)] = 2
    st[st == 2] = 3                     # firn (option_surfacetype_firn = 1)
    return st


def simulate(k_p, T_bias, f_snow, d_prec=PRECGRAD_FIXED, T_snow=TSNOW_FIXED, area_constant=False):
    """Run the model for 2000-2017. f_snow in mm w.e. d-1 degC-1. Arguments may be complex
    (complex step); branches are decided on the real parts.

    Returns a dict with B (m w.e. a-1), the monthly glacier-wide climatic mass balance (m w.e.),
    and the annual bin areas (km2) and thicknesses (m), shape (nbins, NYEARS + 1).
    """
    dtype = np.result_type(k_p, T_bias, f_snow, d_prec, T_snow, float)
    ddfsnow = f_snow / 1000.0
    ddfice = ddfsnow / DDF_ICE_RATIO
    ddf = {0: ddfsnow, 1: ddfice, 2: ddfsnow, 3: (ddfsnow + ddfice) / 2}
    nb, nt = Z_BINS.shape[0], T_GCM.shape[0]
    z = Z_BINS
    a0 = AREA0.astype(dtype)
    h0 = THICK0.astype(dtype)
    g_initial = _nz(a0)
    area_initial = a0.copy()
    h_initial = h0.copy()
    h_initial[:g_initial[0]] = h_initial[g_initial[0]]
    st = np.zeros(nb)
    st[(z < Z_MED) & _pos(a0)] = 1
    st[(z >= Z_MED) & _pos(a0)] = 3     # snow above the median elevation, then firn
    acc = np.zeros((nb, nt), dtype)
    melt = np.zeros((nb, nt), dtype)
    refr = np.zeros((nb, nt), dtype)
    mbclim = np.zeros((nb, nt), dtype)
    mbclim_annual = np.zeros((nb, NYEARS), dtype)
    area_annual = np.zeros((nb, NYEARS + 1), dtype)
    thick_annual = np.zeros((nb, NYEARS + 1), dtype)
    rf_pot = np.zeros(nb, dtype)
    snow_left = np.zeros(nb, dtype)
    for year in range(NYEARS):
        sl = slice(12 * year, 12 * year + 12)
        if np.real(h0).max() > 0:
            g = _nz(a0)
            if g.shape[0] != 0:
                temp = (T_GCM[sl] + LAPSE_RATE * (Z_MED - Z_GCM) + LAPSE_RATE * (z - Z_MED)[:, None] + T_bias
                        + LAPSE_RATE * (h0 - h_initial)[:, None])
                precsnow = P_GCM[sl] * k_p * (1 + d_prec * (z - Z_MED))[:, None]
                if z[g[-1]] - z[g[0]] > 1000:
                    up = g[(g - g[0] + 1) / g.shape[0] * 100 > 75]
                    precsnow[up] = precsnow[up[0]] * np.exp(-(z[up] - z[up[0]]) / (z[up[-1]] - z[up[0]]))[:, None]
                    if year == 0:
                        # The 87.5 % floor is applied to the columns of the first model year only, as in
                        # PyGEM v0.1, whose loop indexes the months of year 0 in every year.
                        for mo in range(12):
                            cap = 0.875 * precsnow[g, mo][np.argmax(np.real(precsnow[g, mo]))]
                            sel = up[(np.real(precsnow[up, mo]) < np.real(cap)) & (np.real(precsnow[up, mo]) != 0)]
                            precsnow[sel, mo] = cap
                rain = (0.5 + (temp - T_snow) / 2) * precsnow
                snow = precsnow - rain
                hot = np.real(temp) > np.real(T_snow) + 1
                cold = np.real(temp) <= np.real(T_snow) - 1
                rain[hot] = precsnow[hot]
                snow[hot] = 0
                snow[cold] = precsnow[cold]
                rain[cold] = 0
                ddf_bins = np.zeros(nb, dtype)
                for k, v in ddf.items():
                    ddf_bins[st == k] = v
                for mo in range(12):
                    step = 12 * year + mo
                    pack = snow_left + snow[:, mo]
                    energy = temp[:, mo] * DAYS[step]
                    energy[np.real(energy) < 0] = 0
                    msnow = ddfsnow * energy
                    lim = np.real(msnow) > np.real(pack)
                    msnow[lim] = pack[lim]
                    energy = energy - msnow / ddfsnow
                    energy[np.abs(np.real(energy)) < TOLERANCE] = 0
                    mglac = np.zeros(nb, dtype)
                    mglac[g] = ddf_bins[g] * energy[g]
                    mtot = mglac + msnow
                    if MONTHS[step] == RF_MONTH:
                        w = DAYS[sl] / DAYS[sl].sum()
                        tann = (temp * w).sum(axis=1)
                        pot = (-0.69 * tann + 0.0096) / 100
                        pot[np.real(pot) < 0] = 0
                        if np.real(pot).max() > 0:
                            rf_pot = pot
                    rf = msnow + rain[:, mo]
                    lim = np.real(rf) > np.real(pack)
                    rf[lim] = pack[lim]
                    lim = np.real(rf) > np.real(rf_pot)
                    rf[lim] = rf_pot[lim]
                    rf[np.abs(np.real(rf)) < TOLERANCE] = 0
                    rf_pot = rf_pot - rf
                    rf_pot[np.abs(np.real(rf_pot)) < TOLERANCE] = 0
                    snow_left = pack - msnow
                    snow_left[np.abs(np.real(snow_left)) < TOLERANCE] = 0
                    acc[:, step] = snow[:, mo]
                    melt[g, step] = mtot[g]
                    refr[g, step] = rf[g]
                    mbclim[g, step] = snow[g, mo] + rf[g] - mtot[g]
                # the glacier cannot lose more than its mass in one year
                max_loss = -(a0 * h0 * DENSITY_ICE / DENSITY_WATER).sum() / a0.sum()
                mb_year = (a0 * mbclim[:, sl].sum(axis=1)).sum() / a0.sum()
                if np.real(mb_year) < np.real(max_loss):
                    dif = max_loss - mb_year
                    wide_melt = (melt[:, sl] * a0[:, None]).sum() / a0.sum()
                    melt[:, sl] = melt[:, sl] * (1 + TOLERANCE - dif / wide_melt)
                    mbclim[:, sl] = acc[:, sl] + refr[:, sl] - melt[:, sl]
                mbclim_annual[:, year] = mbclim[:, sl].sum(axis=1)
                st = _surface_type_annual(st, mbclim_annual, year)
                if area_constant:
                    a1, h1 = a0, h0
                elif np.real(a0).max() > 0:
                    a1, h1 = _redistribute(a0.copy(), h0.copy(), mbclim_annual[:, year], g_initial, area_initial)
                    st[np.real(a1) == 0] = 0
                    st[(st == 0) & (np.real(a1) != 0)] = st[_nz(a0)[0]]
                else:
                    a1, h1 = np.zeros_like(a0), np.zeros_like(a0)
                    st = np.zeros(nb)
                if np.real(a1[0]) > 0:
                    raise RuntimeError('the glacier advanced to the lowest elevation bin of glacier_bins.csv')
                if year == 0:
                    area_annual[:, 0] = a0
                    thick_annual[:, 0] = h0
                area_annual[:, year + 1] = a1
                thick_annual[:, year + 1] = h1
                h0, a0 = h1.copy(), a1.copy()
    # glacier-wide values (calc_glacwide: divide only where the area-weighted sum is positive)
    area_m = np.repeat(area_annual[:, :NYEARS], 12, axis=1)
    wide = area_m.sum(axis=0)

    def glacwide(v):
        s = (v * area_m).sum(axis=0)
        out = np.zeros(nt, dtype)
        p = np.real(s) > 0
        out[p] = s[p] / wide[p]
        return out

    mb_wide = glacwide(acc) + glacwide(refr) - glacwide(melt)
    B = (mb_wide * wide).sum() / wide[0] / NYEARS
    if not np.isfinite(np.real(B)):
        raise FloatingPointError('non-finite mass balance')
    return {'B': B, 'mb_monthly': mb_wide, 'area_annual': area_annual, 'thick_annual': thick_annual}


def _args(x):
    x = list(x)
    if len(x) not in (3, 5):
        raise ValueError('x must have 3 entries (k_p, T_bias, f_snow) or 5 (+ d_prec, T_snow)')
    return x


def mass_balance(x, grad=False, area_constant=False):
    """B (m w.e. a-1) at physical inputs x; with grad=True also dB/dx (physical units) by the complex step."""
    x = _args(x)
    B = float(np.real(simulate(*x, area_constant=area_constant)['B']))
    if not grad:
        return B
    g = np.empty(len(x))
    for i in range(len(x)):
        h = 1e-30 * max(abs(x[i]), 1e-4)
        xc = [complex(v) for v in x]
        xc[i] += 1j * h
        g[i] = np.imag(simulate(*xc, area_constant=area_constant)['B']) / h
    return B, g


def mass_balances(X, grad=False, area_constant=False):
    """Row-wise mass_balance over an (M, 3) or (M, 5) array."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f = np.empty(len(X))
    G = np.empty(X.shape) if grad else None
    for i, x in enumerate(X):
        if grad:
            f[i], G[i] = mass_balance(x, grad=True, area_constant=area_constant)
        else:
            f[i] = mass_balance(x, area_constant=area_constant)
    return (f, G) if grad else f
