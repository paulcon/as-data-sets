"""Convert the outside input files of FaIR_Climate_Response_Calibration losslessly (numpy only).

    python convert_data.py --zenodo ZDIR --fair-calibrate FCDIR --out DIR           # write the files
    python convert_data.py --zenodo ZDIR --fair-calibrate FCDIR --out DIR --check   # and compare with the shipped ones

ZDIR    the files of the Zenodo record 10.5281/zenodo.10566813 (FaIR calibration data v1.4.1):
        calibrated_constrained_parameters.csv, CH4_lifetime.csv, landuse_scale_factor.csv,
        lapsi_scale_factor.csv (their sha256 are checked).
FCDIR   a clone of github.com/chrisroadmap/fair-calibrate at tag v1.4.1
        (commit 5a31144b58d6b9e2b23a845f68f73c7eda98701c); data/forcing/solar_erf_timebounds.csv and
        data/forcing/volcanic_ERF_1750-2101_timebounds.csv are read.

Writes
  posterior_parameters.csv               841 x 47: prior-ensemble index, the 45 parameters, seed
  fixed_calibrations.csv                 1 x 9: methane lifetime calibration, land-use and LAPSI factors
  solar_volcanic_forcing_1750-2101.csv   352 x 3: time bound, solar ERF, volcanic ERF (W m-2)

Values are written with 17 significant digits (repr-exact for float64) and are compared on reading
with the source values; --check also compares DIR with the shipped files (relative tolerance 1e-15).
--out must be outside this directory; it may be this directory itself only with --overwrite (no --check).
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fair_driver as D  # noqa: E402  (numpy-only part)

SHA256 = {
    "calibrated_constrained_parameters.csv": "76e2b9156ed26aa2730aa7023f1e40025a3637fe479df18e822120eff001848c",
    "CH4_lifetime.csv": "32e517971f03cd325b3042b5c933f4d3fde9f0d56699b0da5bf928331d1a4848",
    "landuse_scale_factor.csv": "b730032c575e7ad4a6c274c38216fa58c289bab284a6eab62bd7f646284a3c3c",
    "lapsi_scale_factor.csv": "7dbc6a521375c93265d1c2c2cbf01decc38d27fcf5c9186b40931453086c720d",
    "solar_erf_timebounds.csv": "129e6d00d505eea4559e54f16f4b3fc19ceaad715ac4308108829ac5ad1cb5b4",
    "volcanic_ERF_1750-2101_timebounds.csv": "a051cea4531b1fd4a662e3dd17ef022f97a54cac8ba5d8e48599ec19b8b82f91",
}
TOL = 1e-15


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_table(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def verify(path):
    name = os.path.basename(path)
    got = sha(path)
    if got != SHA256[name]:
        sys.exit(f"{name}: sha256 {got} differs from the expected {SHA256[name]}")


def save(path, A, desc, cols, fmt="%.17g"):
    np.savetxt(path, A, delimiter=",", fmt=fmt, header=desc + "\n" + ", ".join(cols))


def max_rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def convert(zdir, fcdir, out):
    written = {}
    # posterior
    p = os.path.join(zdir, "calibrated_constrained_parameters.csv")
    verify(p)
    hdr, rows = read_table(p)
    assert hdr[1:46] == D.PARAM_NAMES and hdr[46] == "seed", "unexpected columns"
    A = np.array([[float(v) for v in r] for r in rows])
    cols = (["config_id [index in the 1.6 million prior ensemble]"]
            + [f"{n} [{u}]" for n, u in zip(D.PARAM_NAMES, D.UNITS)] + ["seed [-]"])
    f = os.path.join(out, D.F_POSTERIOR)
    save(f, A, "fair-calibrate v1.4.1 posterior (841 members), Zenodo 10.5281/zenodo.10566813, "
               "calibrated_constrained_parameters.csv; physical units", cols)
    written[f] = A
    # fixed calibrations
    vals = []
    for fn, keys in (("CH4_lifetime.csv", ["base", "CH4", "NOx", "VOC", "HC", "N2O", "temp"]),
                     ("landuse_scale_factor.csv", ["CO2_AFOLU"]), ("lapsi_scale_factor.csv", ["BC"])):
        p = os.path.join(zdir, fn)
        verify(p)
        hdr, rows = read_table(p)
        row = [r for r in rows if r[0] == "historical_best"][0]
        vals += [float(row[hdr.index(k)]) for k in keys]
    units = ["yr", "ppb-1", "(Mt NO2 yr-1)-1", "(Mt VOC yr-1)-1", "ppt-1", "ppb-1", "K-1",
             "W m-2 (Gt CO2)-1", "W m-2 (Mt BC yr-1)-1"]
    f = os.path.join(out, D.F_FIXED)
    V = np.array(vals)[None, :]
    save(f, V, "fixed v1.4.1 calibrations (row historical_best of CH4_lifetime.csv, "
               "landuse_scale_factor.csv, lapsi_scale_factor.csv; Zenodo 10.5281/zenodo.10566813)",
         [f"{n} [{u}]" for n, u in zip(D.FIXED_NAMES, units)])
    written[f] = V
    # solar and volcanic forcing, time bounds 1750-2101
    ps = os.path.join(fcdir, "data", "forcing", "solar_erf_timebounds.csv")
    pv = os.path.join(fcdir, "data", "forcing", "volcanic_ERF_1750-2101_timebounds.csv")
    verify(ps)
    verify(pv)
    _, rs = read_table(ps)
    _, rv = read_table(pv)
    sol = {int(r[0]): float(r[1]) for r in rs}
    vol = {int(r[0]): float(r[1]) for r in rv}
    years = np.arange(D.START, D.END + 1)
    F = np.c_[years, [sol[y] for y in years], [vol[y] for y in years]]
    f = os.path.join(out, D.F_FORCING)
    save(f, F, "solar and volcanic effective radiative forcing on time bounds (fair-calibrate v1.4.1, "
               "data/forcing/solar_erf_timebounds.csv and volcanic_ERF_1750-2101_timebounds.csv)",
         ["timebound [yr]", "solar ERF [W m-2]", "volcanic ERF [W m-2]"])
    written[f] = F
    # read back and compare with the source values
    for f, A in written.items():
        B = np.loadtxt(f, delimiter=",", ndmin=2)
        r = max_rel(B, A)
        print(f"{os.path.basename(f)}: {B.shape}, max relative difference from the source values {r:.1e}")
        if r > TOL:
            sys.exit(f"{f}: conversion is not lossless")
    return [os.path.basename(f) for f in written]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", required=True)
    ap.add_argument("--fair-calibrate", required=True)
    ap.add_argument("--out", required=True, help="output directory, outside this one (this one itself only with --overwrite)")
    ap.add_argument("--overwrite", action="store_true", help="allow --out to be this directory")
    ap.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = ap.parse_args()
    out = os.path.realpath(a.out)
    here = os.path.realpath(HERE)
    same = out == here
    inside = os.path.commonpath([out, here]) == here
    if inside and not same:
        sys.exit("--out is inside this directory: use a directory outside it")
    if same and (a.check or not a.overwrite):
        sys.exit("--out is this directory: use another directory (or --overwrite without --check "
                 "to replace the shipped files)")
    os.makedirs(a.out, exist_ok=True)
    names = convert(a.zenodo, getattr(a, "fair_calibrate"), a.out)
    if a.check:
        for n in names:
            r = max_rel(np.loadtxt(os.path.join(a.out, n), delimiter=",", ndmin=2),
                        np.loadtxt(os.path.join(HERE, n), delimiter=",", ndmin=2))
            print(f"check {n}: max relative difference from the shipped file {r:.1e} "
                  f"({'ok' if r <= TOL else 'FAIL'}, tolerance {TOL:g})")


if __name__ == "__main__":
    main()
