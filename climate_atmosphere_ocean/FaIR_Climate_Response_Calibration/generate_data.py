"""Regenerate the FaIR files of FaIR_Climate_Response_Calibration (needs FaIR 2.1.3; see DATA.md).

    python generate_data.py --stage runs --out DIR                      # qoi*.csv and gradients_*.csv
    python generate_data.py --stage runs --rows 0:20 --out DIR --check  # a subset, compared with the shipped files
    python generate_data.py --stage emissions --fair-example EXT --rcmip RCMIP --out DIR [--check]

Stages
  emissions  ssp245_emissions_1750-2100.csv: historical emissions 1750-2021 of fair-calibrate v1.4.1
             (taken from EXT = examples/data/calibrated_constrained_ensemble/extensions_1750-2500.csv of
             github.com/OMS-NetZero/FAIR at tag v2.2.0, rows of any scenario, which agree up to 2022) and
             SSP2-4.5 for 2022-2100 from RCMIP = data/emissions/rcmip-5-1-0-corrected-nox.csv of
             fair-calibrate v1.4.1, harmonised to 2022 with aneris exactly as in fair-calibrate's
             calibration/12_harmonize-emissions.py. Needs aneris-iamc 0.3.1 and scmdata 0.16.1.
  runs       qoi.csv (deterministic runs: internal variability off), qoi_internal_variability.csv (each
             member's own seed, internal variability on, as in the paper), and gradients_<qoi>.csv
             (central differences in the analysis coordinates u = fair_driver.to_analysis(x), step
             h_j = DELTA * sd(u_j), sd over the 841 posterior members). Needs FaIR 2.1.3.

Run stage "runs" resumes: the gradient of each parameter is cached in DIR/_cache_<rows>/ and a log is
appended to DIR/generate_data.log. --out is required, may not be inside this directory, and may be
this directory itself only with --overwrite (which replaces the shipped files); --check and --rows need an
--out outside this directory. --check compares with the shipped
files at a relative tolerance of 1e-9 (QoIs) and 1e-6 (gradients) of each file's largest value.
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fair_driver as D  # noqa: E402

DELTA = 1e-3              # finite-difference step, in posterior standard deviations of u_j
TOL = {"qoi": 1e-9, "grad": 1e-6, "emis": 1e-10}
INSTALL_FAIR = (f"FaIR is not installed. Install it (pip install fair=={D.FAIR_VERSION}, the version "
                "that wrote the shipped files; DATA.md gives the full environment) and rerun generate_data.py.")
INSTALL_ANERIS = ("aneris or scmdata is not installed. Install them (pip install aneris-iamc==0.3.1 "
                  "scmdata==0.16.1 in the FaIR environment of DATA.md) and rerun generate_data.py.")


def logger(out):
    path = os.path.join(out, "generate_data.log")

    def log(msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return log


# ---------------------------------------------------------------- emissions stage
OVERRIDES = (  # fair-calibrate v1.4.1, calibration/12_harmonize-emissions.py
    [("reduce_ratio_2150_cov", "Emissions|C2F6"), ("reduce_ratio_2150_cov", "Emissions|C6F14"),
     ("reduce_ratio_2150_cov", "Emissions|CF4"), ("reduce_ratio_2150_cov", "Emissions|CO"),
     ("reduce_ratio_2080", "Emissions|CO2"), ("reduce_offset_2150_cov", "Emissions|CO2|AFOLU"),
     ("reduce_ratio_2080", "Emissions|CO2|Energy and Industrial Processes")]
    + [("constant_ratio", f"Emissions|{g}") for g in
       ("HFC-125", "HFC-134a", "HFC-143a", "HFC-227ea", "HFC-23", "HFC-32", "HFC-4310mee")]
    + [("reduce_ratio_2150_cov", "Emissions|OC"), ("constant_ratio", "Emissions|SF6"),
       ("reduce_ratio_2150_cov", "Emissions|VOC")]
)
CO2_NAMES = {"CO2 FFI": "Emissions|CO2|Energy and Industrial Processes",
             "CO2 AFOLU": "Emissions|CO2|AFOLU"}


def make_emissions(ext_file, rcmip_file, out, log):
    try:
        import warnings
        import pandas as pd
        import scmdata
        import aneris.convenience
    except ImportError:
        sys.exit(INSTALL_ANERIS)
    df = pd.read_csv(ext_file)
    df = df[df["scenario"] == "medium-extension"]          # any scenario: all agree up to 2022
    hist_cols = [f"{y}.5" for y in range(1750, 2023)]
    variables = [CO2_NAMES.get(v, "Emissions|" + v) for v in df["variable"]]
    H = pd.DataFrame(df[hist_cols].values, columns=list(range(1750, 2023)))
    H.index = pd.MultiIndex.from_arrays(
        [["history"] * len(df), ["GCP+CEDS+PRIMAP+GFED"] * len(df), ["World"] * len(df), variables,
         list(df["unit"])], names=["model", "scenario", "region", "variable", "unit"])
    var_units = dict(zip(variables, df["unit"]))
    times_future = [datetime.datetime(y, 1, 1) for y in range(2022, 2501)]
    fut = (scmdata.ScmRun(rcmip_file, lowercase_cols=True)
           .filter(scenario=[D.SCENARIO], variable=variables, region="World")
           .interpolate(times_future).timeseries(time_axis="year"))
    sel = fut.index.get_level_values("variable").isin(
        ("Emissions|CO2|Energy and Industrial Processes", "Emissions|CO2|AFOLU", "Emissions|N2O"))
    fut.iloc[sel] = fut.iloc[sel] / 1000                     # Mt CO2 -> Gt CO2, kt N2O -> Mt N2O
    names = list(fut.index.names)
    rows = [list(i) for i in fut.index]
    for r in rows:
        r[names.index("unit")] = var_units[r[names.index("variable")]]
    fut.index = pd.MultiIndex.from_tuples([tuple(r) for r in rows], names=names)
    fut = fut.reorder_levels(["model", "scenario", "region", "variable", "unit"]).sort_index()
    overrides = pd.DataFrame([{"method": m, "variable": v} for m, v in OVERRIDES])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        harm = [aneris.convenience.harmonise_all(g, history=H.sort_index(), harmonisation_year=2022,
                                                 overrides=overrides).reset_index(level=(5, 6, 7, 8, 9), drop=True)
                for _, g in fut.groupby(["model", "scenario"])]
    harm = pd.concat(harm).reset_index()
    fair_names, cols, E = [], [], []
    inv = {v: k for k, v in CO2_NAMES.items()}
    for v, u in zip(df["variable"], df["unit"]):
        long = CO2_NAMES.get(v, "Emissions|" + v)
        h = df.loc[df["variable"] == v, [f"{y}.5" for y in range(1750, 2022)]].values.squeeze()
        f = harm.loc[harm["variable"] == long, list(range(2022, 2101))].values.squeeze()
        E.append(np.concatenate([h, f]).astype(float))
        fair_names.append(inv.get(long, v))
        cols.append(f"{fair_names[-1]} [{u}]")
    E = np.array(E).T
    tp = np.arange(1750, 2101) + 0.5
    path = os.path.join(out, D.F_EMIS)
    np.savetxt(path, np.c_[tp, E], delimiter=",", fmt="%.12e", header=(
        "SSP2-4.5 emissions for FaIR 2.1.3 (fair-calibrate v1.4.1): history 1750-2021, aneris-harmonised "
        "(2022) RCMIP v5.1.0 SSP2-4.5 from 2022; annual means at mid-year time points\n"
        + ", ".join(["time [yr]"] + cols)))
    log(f"wrote {path}: {E.shape[0]} years x {E.shape[1]} species")
    return [D.F_EMIS]


# ---------------------------------------------------------------- runs stage
def run_stage(out, rows, log):
    if D.FAIR is None:
        sys.exit(INSTALL_FAIR)
    ids, X, seeds = D.load_posterior()
    sd_u = D.to_analysis(X).std(axis=0, ddof=1)            # step scale: all 841 members, whatever --rows
    X, ids, seeds = X[rows], ids[rows], seeds[rows]
    ens = D.FairEnsemble()
    tag = f"{rows.start}-{rows.stop}"
    t0 = time.time()
    det = ens.run(X)
    log(f"deterministic run of {len(X)} members: {time.time() - t0:.1f} s")
    t1 = time.time()
    sto = ens.run(X, seeds=seeds, stochastic=True)
    log(f"run with internal variability: {time.time() - t1:.1f} s")
    cache = os.path.join(out, f"_cache_{tag}")
    os.makedirs(cache, exist_ok=True)
    t2 = time.time()
    G = D.fd_gradients(ens, X, DELTA, sd_u, cache=cache, log=log)
    log(f"gradients (90 runs): {time.time() - t2:.1f} s")
    qcols = ["config_id [index in the prior ensemble]"] + [f"{q} [K]" for q in D.QOI_NAMES] + ["CO2_2022 [ppm]"]
    hdr = ("FaIR {v} runs of the fair-calibrate v1.4.1 posterior, SSP2-4.5, {how}; rows {a}:{b}"
           .format(v=D.FAIR_VERSION, a=rows.start, b=rows.stop, how="{how}"))
    for fn, r, how in ((D.F_QOI, det, "internal variability off (deterministic)"),
                       (D.F_QOI_IV, sto, "internal variability on, each member's own seed")):
        A = np.c_[ids, np.column_stack([r[q] for q in D.QOI_NAMES]), r["CO2_2022"]]
        np.savetxt(os.path.join(out, fn), A, delimiter=",", fmt=["%d"] + ["%.15e"] * 5,
                   header=hdr.format(how=how) + "\n" + ", ".join(qcols))
    for q in D.QOI_NAMES:
        cols = [f"d{q}/d{s} [K per unit of u_j = {s}]" for s in D.SHORT]
        np.savetxt(os.path.join(out, D.GRAD_FILES[q]), G[q], delimiter=",", fmt="%.12e", header=(
            f"gradient of {q} (deterministic run) with respect to the analysis coordinates u "
            f"(fair_driver.to_analysis), central differences, step {DELTA:g} posterior sd of u_j; "
            f"rows {rows.start}:{rows.stop}\n" + ", ".join(cols)))
    log(f"total {time.time() - t0:.1f} s")
    return [D.F_QOI, D.F_QOI_IV] + list(D.GRAD_FILES.values())


def check(out, files, rows):
    for fn in files:
        kind = "grad" if fn.startswith("gradients") else ("emis" if fn == D.F_EMIS else "qoi")
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)
        if kind != "emis":
            old = old[rows]
        rel = np.max(np.abs(new - old)) / np.max(np.abs(old))
        print(f"  {fn}: max |new - shipped| / max |shipped| = {rel:.1e} "
              f"({'ok' if rel <= TOL[kind] else 'FAIL'}, tolerance {TOL[kind]:g})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["emissions", "runs"])
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true", help="allow --out to be this directory")
    p.add_argument("--rows", default=None, help="a:b, a subset of the 841 members (stage runs)")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    p.add_argument("--fair-example", help="extensions_1750-2500.csv (FaIR v2.2.0 examples)")
    p.add_argument("--rcmip", help="rcmip-5-1-0-corrected-nox.csv (fair-calibrate v1.4.1 data/emissions)")
    a = p.parse_args()
    out = os.path.realpath(a.out)
    here = os.path.realpath(HERE)
    same = out == here
    inside = os.path.commonpath([out, here]) == here
    if inside and not same:
        sys.exit("--out is inside this directory: use a directory outside it")
    if same and (a.check or a.rows or not a.overwrite):
        sys.exit("--out is this directory: use another directory; --check and --rows always need one "
                 "outside it (--overwrite replaces the shipped files)")
    os.makedirs(a.out, exist_ok=True)
    log = logger(a.out)
    if a.stage == "emissions":
        if not (a.fair_example and a.rcmip):
            sys.exit("stage emissions needs --fair-example and --rcmip")
        files = make_emissions(a.fair_example, a.rcmip, a.out, log)
        rows = None
    else:
        if a.rows:
            i, j = (int(s) for s in a.rows.split(":"))
            rows = slice(i, j)
        else:
            rows = slice(0, 841)
        files = run_stage(a.out, rows, log)
    if a.check:
        check(a.out, files, rows)


if __name__ == "__main__":
    main()
