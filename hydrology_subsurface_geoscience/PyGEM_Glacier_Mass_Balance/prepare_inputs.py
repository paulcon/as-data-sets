"""Write the stored model inputs (glacier_bins.csv, climate_monthly.csv, site.csv) from OGGM sample data.

    python prepare_inputs.py --src DIR --out DIR            # DIR = a checkout of OGGM/oggm-sample-data
    python prepare_inputs.py --src DIR --out DIR --check    # also compare with the files here

The three CSVs read by `pygem_mb_model.py` are reductions of five files of the OGGM sample-data
repository (https://github.com/OGGM/oggm-sample-data, commit ae1dfa73a34bd31ba977945056d5ad34b58060ef,
BSD-3-Clause), checked by sha256 before use:

- test-workflow/climate/sel_era5_monthly_t2m_1979-2018.nc, sel_era5_monthly_prcp_1979-2018.nc,
  sel_era5_invariant.nc: ERA5 monthly 2 m temperature (K), total precipitation (m per day, monthly
  mean of daily totals) and geopotential on a 4 x 4 grid of 0.25 deg around the Oetztal Alps.
  The grid cell nearest to Hintereisferner (RGI60-11.00897, 46.80 N, 10.76 E) is (46.75 N, 10.75 E).
  Its values for January 2000 to December 2017 are written unchanged (as unpacked by netCDF4, with
  17 significant digits): this extraction is lossless.
- test-files/consensus_composite/RGI60-11.00897_thickness.tif: ice thickness (m) on a 25 m UTM 32N
  grid, the consensus estimate of Farinotti et al. (2019, Nat. Geosci. 12, 168-173). Pixels with
  thickness > 0 are taken as the glacier (8.0325 km2; the RGI area is 8.036 km2).
- test-files/hef_srtm.tif: SRTM surface elevation (m, February 2000), resampled bilinearly onto the
  thickness grid.

The glacier pixels are grouped into 10 m elevation bins (bin k holds 10k <= z < 10(k+1), centre
10k + 5 m); each bin's area is the pixel count times 625 m2 and its thickness the mean pixel
thickness. This binning is a reduction, not a lossless conversion. Empty bins extend the table 500 m
below the lowest glacier bin so that the glacier can advance.

Needs numpy, netCDF4 and rasterio (tested with netCDF4 1.7.5, rasterio 1.5.2). Without them it exits
with an install message. The model and the notebook do not need this script or those packages.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

COMMIT = 'ae1dfa73a34bd31ba977945056d5ad34b58060ef'
SOURCES = {
    't2m': ('test-workflow/climate/sel_era5_monthly_t2m_1979-2018.nc',
            '464ffa8902b5df06796a0d0753be5f3b8a4dfc947dac4ddce7b53ea1414d005b'),
    'tp': ('test-workflow/climate/sel_era5_monthly_prcp_1979-2018.nc',
           'b922abc4b96fed8c8ba96d8d50e83a482da835e560fc23e3c7ed683ebb1b89af'),
    'inv': ('test-workflow/climate/sel_era5_invariant.nc',
            '4d4d41fd0ee3cb2c5dacd5a1c6e9d672ce9ccf6fd0c6132ab3b5b0ecf179ef4f'),
    'thick': ('test-files/consensus_composite/RGI60-11.00897_thickness.tif',
              '92debf889c7a7c54e0206f9feb679850d7f9760d13abb0d3cccf35e5da9eb7d5'),
    'dem': ('test-files/hef_srtm.tif',
            '079f80987a4f309a2210e86399ad05c994d2bd9b2eec8acb0f66f06f5b746bf0'),
}
GLACIER_LAT, GLACIER_LON = 46.8003, 10.7584   # RGI centre point of Hintereisferner
YEAR0, YEAR1 = 2000, 2017                      # calendar years kept (inclusive)
G = 9.80665                                    # m s-2, to convert geopotential to height
BIN = 10.0                                     # m
EXTEND_BELOW = 500.0                           # m of empty bins below the terminus
FILES = ('glacier_bins.csv', 'climate_monthly.csv', 'site.csv')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def climate(src):
    import netCDF4
    ds_t = netCDF4.Dataset(os.path.join(src, SOURCES['t2m'][0]))
    ds_p = netCDF4.Dataset(os.path.join(src, SOURCES['tp'][0]))
    ds_z = netCDF4.Dataset(os.path.join(src, SOURCES['inv'][0]))
    lat = np.asarray(ds_t['latitude'][:], dtype=float)
    lon = np.asarray(ds_t['longitude'][:], dtype=float)
    assert np.array_equal(lat, np.asarray(ds_p['latitude'][:])) and np.array_equal(lon, np.asarray(ds_p['longitude'][:]))
    i = int(np.argmin(np.abs(lat - GLACIER_LAT)))
    j = int(np.argmin(np.abs(lon - GLACIER_LON)))
    dates = netCDF4.num2date(ds_t['time'][:], ds_t['time'].units)
    dates_p = netCDF4.num2date(ds_p['time'][:], ds_p['time'].units)
    assert all(a == b for a, b in zip(dates, dates_p))
    keep = np.array([YEAR0 <= d.year <= YEAR1 for d in dates])
    years = np.array([d.year for d in dates])[keep]
    months = np.array([d.month for d in dates])[keep]
    days = np.array([(np.datetime64(f'{y + (m == 12):04d}-{(m % 12) + 1:02d}-01') -
                      np.datetime64(f'{y:04d}-{m:02d}-01')).astype(int) for y, m in zip(years, months)])
    t2m = np.asarray(ds_t['t2m'][:, i, j], dtype=float)[keep]
    tp = np.asarray(ds_p['tp'][:, i, j], dtype=float)[keep]
    zgcm = float(np.asarray(ds_z['z'][0, i, j], dtype=float)) / G
    assert len(years) == 12 * (YEAR1 - YEAR0 + 1) and months[0] == 1 and months[-1] == 12
    return np.column_stack([years, months, days, t2m, tp]), lat[i], lon[j], zgcm


def glacier(src):
    import rasterio
    from rasterio.warp import Resampling, reproject
    with rasterio.open(os.path.join(src, SOURCES['thick'][0])) as r:
        thick = r.read(1).astype(float)
        transform, crs = r.transform, r.crs
    with rasterio.open(os.path.join(src, SOURCES['dem'][0])) as r:
        dem = np.zeros(thick.shape, dtype='float64')
        reproject(source=r.read(1).astype('float64'), destination=dem, src_transform=r.transform,
                  src_crs=r.crs, dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear)
    pix = abs(transform.a * transform.e)                 # m2 per pixel (625)
    mask = thick > 0
    z = dem[mask]
    h = thick[mask]
    k = np.floor(z / BIN).astype(int)
    k0 = int(np.floor((z.min() - EXTEND_BELOW) / BIN))
    k1 = int(k.max())
    ks = np.arange(k0, k1 + 1)
    area = np.array([(k == kk).sum() for kk in ks]) * pix / 1e6           # km2
    hbar = np.array([h[k == kk].mean() if (k == kk).any() else 0.0 for kk in ks])
    zc = ks * BIN + BIN / 2
    zmed = float(np.median(z))
    return np.column_stack([zc, area, hbar]), zmed, float(z.min()), float(z.max()), mask.sum() * pix / 1e6


def write(out, bins, clim, site):
    src = f'OGGM/oggm-sample-data@{COMMIT[:8]}'
    np.savetxt(os.path.join(out, 'glacier_bins.csv'), bins, fmt='%.17g', delimiter=',',
               header=(f'Hintereisferner (RGI60-11.00897) on 10 m elevation bins: glacier pixels (consensus '
                       f'thickness > 0, Farinotti et al. 2019) binned on SRTM elevation; {src}; prepare_inputs.py\n'
                       'z_bin (m a.s.l., bin centre),area (km2),ice_thickness (m, bin mean)'))
    np.savetxt(os.path.join(out, 'climate_monthly.csv'), clim, fmt='%.17g', delimiter=',',
               header=(f'ERA5 monthly means at grid cell ({site[0]:.2f} N, {site[1]:.2f} E), Jan {YEAR0}-Dec {YEAR1}, '
                       f'values unchanged; {src}; contains modified Copernicus Climate Change Service information; '
                       'prepare_inputs.py\n'
                       'year,month,days_in_month,t2m (K),tp (m per day; monthly mean of daily totals)'))
    np.savetxt(os.path.join(out, 'site.csv'), np.atleast_2d(site), fmt='%.17g', delimiter=',',
               header=('ERA5 cell and glacier summary (glacier from the consensus thickness grid and SRTM); '
                       'prepare_inputs.py\n'
                       'era5_lat (deg N),era5_lon (deg E),z_era5 (m, geopotential / 9.80665),'
                       'z_med (m, median glacier pixel elevation),z_min (m),z_max (m),area (km2)'))


def read(path):
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', required=True, help='checkout of OGGM/oggm-sample-data at commit ' + COMMIT[:8])
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--check', action='store_true', help='compare with the CSVs in this directory')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here:
        ap.error('--out must be outside this data set\'s directory (not the directory or a folder in it), so that '
                 'the shipped files are not overwritten')
    try:
        import netCDF4  # noqa: F401
        import rasterio  # noqa: F401
    except ImportError:
        sys.exit('prepare_inputs.py needs netCDF4 and rasterio: pip install netCDF4==1.7.5.1.post1 rasterio==1.5.2, '
                 'then rerun it.')
    for key, (rel, digest) in SOURCES.items():
        path = os.path.join(args.src, rel)
        if not os.path.exists(path):
            sys.exit(f'missing {rel}: check out https://github.com/OGGM/oggm-sample-data at {COMMIT} and rerun.')
        if sha256(path) != digest:
            sys.exit(f'sha256 mismatch for {rel}: expected {digest}')
    clim, lat, lon, zgcm = climate(args.src)
    bins, zmed, zmin, zmax, area = glacier(args.src)
    site = np.array([lat, lon, zgcm, zmed, zmin, zmax, area])
    os.makedirs(out, exist_ok=True)
    write(out, bins, clim, site)
    print(f'wrote {out}: {len(bins)} bins ({(bins[:, 1] > 0).sum()} with ice), area {bins[:, 1].sum():.4f} km2, '
          f'volume {(bins[:, 1] * bins[:, 2]).sum() / 1e3:.5f} km3; z_med {zmed:.1f} m, range {zmin:.1f}-{zmax:.1f} m; '
          f'ERA5 cell ({lat:.2f}, {lon:.2f}) at {zgcm:.1f} m; {len(clim)} months')
    if args.check:
        for n in FILES:
            a, b = read(os.path.join(out, n)), read(os.path.join(HERE, n))
            same = a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=0)
            print(f'{n}: shape {a.shape}, max abs difference '
                  f'{np.max(np.abs(a - b)) if a.shape == b.shape else float("nan"):.1e} '
                  f'(tolerance rtol 1e-12): {"OK" if same else "DIFFERENT"}')


if __name__ == '__main__':
    main()
