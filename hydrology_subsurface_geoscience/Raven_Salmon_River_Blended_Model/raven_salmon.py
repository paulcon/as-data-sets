"""Raven blended (weighted-process) model of the Salmon River (WSC 08KC001), BC.

Runs Raven at one point of the 43-input study of Mai, Craig and Tolson (2020)
and returns scalar summaries of the simulated daily streamflow. Only numpy and
the standard library are used.

The Raven input deck is NOT part of this file. It comes from the authors' xSSA
repository (github.com/julemai/xSSA, file examples/scripts/raven_templates.py,
LGPL-3.0, Copyright 2019-2020 Juliane Mai), which is downloaded at run time from
the pinned commit below, checked against its SHA-256, read with `ast` (never
imported or executed) and filled in here. This module holds only our own code:
the input table, the weight transform, the values put into the templates'
placeholders, and the output reader. The template file is not redistributed.

Inputs (in this order, physical units, Table C2 of the paper):
  x01 ... x35   the 35 model parameters (x04 and x11 are log10 of 1/d rates)
  r1  ... r8    U[0, 1] numbers that set the 13 process-option weights

Outside this repository: the Raven executable (DATA.md, "Raven build"), the
xSSA forcing and streamflow files (examples/data_in/data_obs/08KC001) and the
xSSA template file (fetched by `fetch_templates`).

MIT licence (this repository). Author: Claude (Anthropic) for P. Constantine, 2026.
"""
import ast
import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np

# --------------------------------------------------------------------------
# Where the deck templates come from (fetched, not redistributed)
# --------------------------------------------------------------------------
XSSA_REPO = "julemai/xSSA"
XSSA_COMMIT = "034e7189618b377a8b25ac95fde0ea11656cfa8c"  # master on 2026-11-01 (2021-07-27)
XSSA_TEMPLATE_FILE = "examples/scripts/raven_templates.py"
XSSA_TEMPLATE_SHA256 = "8109faaeb01ed6ad7e3d835e6a08a1e3a328d8911199f2e0c0bfca500aa060d0"
XSSA_TEMPLATE_URL = (f"https://raw.githubusercontent.com/{XSSA_REPO}/{XSSA_COMMIT}/"
                     f"{XSSA_TEMPLATE_FILE}")
XSSA_LICENSE = "LGPL-3.0-or-later"
# module-level string constants of the template file -> deck file extension
TEMPLATE_KEYS = {"RVI": "rvi", "RVP": "rvp", "RVC": "rvc", "RVT": "rvt", "RVH": "rvh"}

# --------------------------------------------------------------------------
# The 43 inputs: name, lower, upper, units, Raven target (Table C2, p. 5855)
# --------------------------------------------------------------------------
_TABLE = [
    ("x01", 0.0, 1.0, "-", "HMETS_RUNOFF_COEFF (infiltration M1)"),
    ("x02", 0.1, 3.0, "-", "B_EXP TOPSOIL (infiltration M2, VIC/ARNO)"),
    ("x03", 0.5, 3.0, "-", "HBV_BETA TOPSOIL (infiltration M3)"),
    ("x04", -5.0, -1.0, "log10(1/d)", "BASEFLOW_COEFF TOPSOIL = 10^x04 (quickflow N1)"),
    ("x05", 0.0, 100.0, "mm/d", "MAX_BASEFLOW_RATE TOPSOIL (quickflow N2, N3)"),
    ("x06", 0.5, 2.0, "-", "BASEFLOW_N TOPSOIL (quickflow N2, N3)"),
    ("x07", 5.0, 10.0, "m", "TOPMODEL_LAMBDA (quickflow N3)"),
    ("x08", 0.0, 3.0, "-", "PET_CORRECTION TOPSOIL (evaporation O1, O2)"),
    ("x09", 0.0, 0.05, "frac", "SAT_WILT TOPSOIL (evaporation O2)"),
    ("x10", 0.0, 0.45, "frac", "FIELD_CAPACITY TOPSOIL = x09 + x10 (evaporation O2)"),
    ("x11", -5.0, -2.0, "log10(1/d)", "BASEFLOW_COEFF PHREATIC = 10^x11 (baseflow P1, P2)"),
    ("x12", 0.5, 2.0, "-", "BASEFLOW_N PHREATIC (baseflow P2)"),
    ("x13", 0.0, 0.1, "frac", "SNOW_SWI_MIN (snow balance Q1)"),
    ("x14", 0.01, 0.3, "frac", "SNOW_SWI_MAX = x13 + x14 (snow balance Q1)"),
    ("x15", 0.005, 0.1, "1/mm", "SWI_REDUCT_COEFF (snow balance Q1)"),
    ("x16", -5.0, 2.0, "degC", "DD_REFREEZE_TEMP (snow balance Q1)"),
    ("x17", 0.0, 1.0, "-", "REFREEZE_EXP (snow balance Q1)"),
    ("x18", 0.0, 5.0, "mm/d/degC", "REFREEZE_FACTOR (snow balance Q1, Q3)"),
    ("x19", 0.0, 0.4, "frac", "SNOW_SWI (snow balance Q3)"),
    ("x20", 0.3, 20.0, "-", "GAMMA_SHAPE (convolution, surface runoff R1)"),
    ("x21", 0.01, 5.0, "-", "GAMMA_SCALE (convolution, surface runoff R1)"),
    ("x22", 0.5, 13.0, "-", "GAMMA_SHAPE2 (convolution, delayed runoff S1)"),
    ("x23", 0.15, 1.5, "-", "GAMMA_SCALE2 (convolution, delayed runoff S1)"),
    ("x24", 1.5, 3.0, "mm/d/degC", "MIN_MELT_FACTOR (potential melt T1)"),
    ("x25", 0.0, 5.0, "mm/d/degC", "MAX_MELT_FACTOR = x24 + x25 (potential melt T1)"),
    ("x26", -1.0, 1.0, "degC", "DD_MELT_TEMP (potential melt T1)"),
    ("x27", 0.01, 0.2, "1/mm", "DD_AGGRADATION (potential melt T1)"),
    ("x28", 0.00001, 0.02, "1/d", "PERC_COEFF TOPSOIL (percolation U1)"),
    ("x29", 0.0, 0.5, "m", "thickness TOPSOIL (soil model)"),
    ("x30", 0.0, 2.0, "m", "thickness PHREATIC (soil model)"),
    ("x31", -3.0, 3.0, "degC", "RAINSNOW_TEMP (rain-snow partitioning V1)"),
    ("x32", 0.5, 4.0, "degC", "RAINSNOW_DELTA (rain-snow partitioning V1)"),
    ("x33", 0.8, 1.2, "-", "RAINCORRECTION (precipitation correction W1)"),
    ("x34", 0.8, 1.2, "-", "SNOWCORRECTION (precipitation correction W1)"),
    ("x35", 0.0, 0.02, "1/d", "PERC_COEFF PHREATIC (percolation U1)"),
    ("r1", 0.0, 1.0, "-", "infiltration weights (with r2)"),
    ("r2", 0.0, 1.0, "-", "infiltration weights (with r1)"),
    ("r3", 0.0, 1.0, "-", "quickflow weights (with r4)"),
    ("r4", 0.0, 1.0, "-", "quickflow weights (with r3)"),
    ("r5", 0.0, 1.0, "-", "soil evaporation weights"),
    ("r6", 0.0, 1.0, "-", "baseflow weights"),
    ("r7", 0.0, 1.0, "-", "snow balance weights (with r8)"),
    ("r8", 0.0, 1.0, "-", "snow balance weights (with r7)"),
]
NAMES = [t[0] for t in _TABLE]
LOWER = np.array([t[1] for t in _TABLE])
UPPER = np.array([t[2] for t in _TABLE])
UNITS = [t[3] for t in _TABLE]
DESCRIPTIONS = [t[4] for t in _TABLE]
M_INPUTS = len(NAMES)  # 43

# Process groups with blended options: (name, Raven options in the template's
# order, slice of r1..r8 that sets their weights). Table C1 and paper Sect. 2.1.2.
PROCESS_GROUPS = [
    ("infiltration", ["INF_HMETS", "INF_VIC_ARNO", "INF_HBV"], slice(0, 2)),
    ("quickflow", ["BASE_LINEAR_ANALYTIC", "BASE_VIC", "BASE_TOPMODEL"], slice(2, 4)),
    ("evaporation", ["SOILEVAP_ALL", "SOILEVAP_TOPMODEL"], slice(4, 5)),
    ("baseflow", ["BASE_LINEAR_ANALYTIC", "BASE_POWER_LAW"], slice(5, 6)),
    ("snow balance", ["SNOBAL_HMETS", "SNOBAL_SIMPLE_MELT", "SNOBAL_HBV"], slice(6, 8)),
]
WEIGHT_NAMES = [f"w_{g[0].replace(' ', '_')}_{o}" for g in PROCESS_GROUPS for o in g[1]]

# Basin properties for 08KC001 (xSSA examples/data_in/basin_metadata/
# basin_physical_characteristics.txt)
BASIN = dict(id="08KC001", name="SALMON RIVER NEAR PRINCE GEORGE",
             lat_deg=54.45653168, lon_deg=-123.35145290, area_km2=4230.0,
             elevation_m=842.92539620, slope_deg=3.35760243, forest_frac=0.47636629)

# The run window is fixed by the templates; the output reader checks against it.
START, END, EVAL_START = "1989-01-01", "2010-12-31", "1991-01-01"
WARMUP_DAYS = 730  # 1989-01-01 ... 1990-12-31


# --------------------------------------------------------------------------
# Templates: download once, verify, parse without executing
# --------------------------------------------------------------------------
def template_path(folder):
    """Local path of the cached xSSA template file inside `folder`."""
    return os.path.join(folder, f"xSSA-{XSSA_COMMIT[:7]}", os.path.basename(XSSA_TEMPLATE_FILE))


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fetch_templates(folder, download=True):
    """Return the path of the verified xSSA template file under `folder`.

    Downloads it from XSSA_TEMPLATE_URL if it is not there yet (and `download`
    is true). Raises RuntimeError if the file is missing or its SHA-256 differs
    from XSSA_TEMPLATE_SHA256.
    """
    path = template_path(folder)
    if not os.path.exists(path):
        if not download:
            raise RuntimeError(f"xSSA template file not found at {path}; fetch it with "
                               f"raven_salmon.fetch_templates({folder!r}) (needs network)")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(XSSA_TEMPLATE_URL, timeout=60) as resp:
            data = resp.read()
        if hashlib.sha256(data).hexdigest() != XSSA_TEMPLATE_SHA256:
            raise RuntimeError(f"downloaded {XSSA_TEMPLATE_URL} has an unexpected SHA-256")
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
    if _sha256(path) != XSSA_TEMPLATE_SHA256:
        raise RuntimeError(f"{path} does not match the pinned xSSA commit {XSSA_COMMIT} (SHA-256)")
    return path


def load_templates(folder, download=True):
    """The five deck templates {extension: text}, read from the verified file.

    The file is parsed with `ast` and only its module-level string constants
    named in TEMPLATE_KEYS are taken; nothing in it is executed.
    """
    path = fetch_templates(folder, download=download)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in TEMPLATE_KEYS
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            out[TEMPLATE_KEYS[node.targets[0].id]] = node.value.value
    missing = set(TEMPLATE_KEYS.values()) - set(out)
    if missing:
        raise RuntimeError(f"template file lacks {sorted(missing)}")
    return out


# --------------------------------------------------------------------------
# Weights (paper Appendix A)
# --------------------------------------------------------------------------
def stick_breaking_weights(r):
    """Moeini et al. (2011) transform, paper Eq. (A1)-(A2).

    For N options and N-1 numbers r_1..r_{N-1} in [0, 1]:
      w_j = (1 - sum_{i<j} w_i) * (1 - (1 - r_j)^(1/(N-j))),  j = 1..N-1
      w_N = 1 - sum_{i<N} w_i
    Each w_j then has CDF 1 - (1 - w)^(N-1) (Eq. A3).
    """
    r = np.asarray(r, dtype=float)
    n = r.size + 1
    w = np.empty(n)
    s = 0.0
    for j in range(n - 1):
        w[j] = (1.0 - s) * (1.0 - (1.0 - r[j]) ** (1.0 / (n - 1 - j)))
        s += w[j]
    w[n - 1] = 1.0 - s
    return w


def weights(x):
    """The 13 process-option weights (order of WEIGHT_NAMES) for one input row."""
    r = np.asarray(x, dtype=float)[35:43]
    return np.concatenate([stick_breaking_weights(r[g[2]]) for g in PROCESS_GROUPS])


# --------------------------------------------------------------------------
# Filling the templates
# --------------------------------------------------------------------------
def _num(v):
    """Shortest text that reads back as the same double."""
    return repr(float(v))


def placeholder_values(x):
    """Values for the four placeholder groups of the xSSA templates at row x.

    par:     x01..x35 as given;
    dpar:    the derived Raven values of Table C2's caption (two sums of a lower
             value and an increment, plus the melt-factor sum; 10^x for the two
             log10 rates; initial soil stores at half of each layer, in mm);
    weights: process01..process05, the weights of each blended process written
             one after another (full precision);
    props:   the basin constants of BASIN.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (M_INPUTS,):
        raise ValueError(f"x must have {M_INPUTS} entries")
    v = dict(zip(NAMES, x))
    par = {n: _num(v[n]) for n in NAMES[:35]}
    dpar = {
        "sum_x09_x10": _num(v["x09"] + v["x10"]),
        "sum_x13_x14": _num(v["x13"] + v["x14"]),
        "sum_x24_x25": _num(v["x24"] + v["x25"]),
        "pow_x04": _num(10.0 ** v["x04"]),
        "pow_x11": _num(10.0 ** v["x11"]),
        "half_x29": _num(v["x29"] * 0.5 * 1000.0),
        "half_x30": _num(v["x30"] * 0.5 * 1000.0),
    }
    w = weights(x)
    sizes = [len(g[1]) for g in PROCESS_GROUPS]
    ends = np.cumsum(sizes)
    wtxt = {f"process{k + 1:02d}": " ".join(_num(c) for c in w[e - s:e])
            for k, (s, e) in enumerate(zip(sizes, ends))}
    props = {k: (val if isinstance(val, str) else _num(val)) for k, val in BASIN.items()}
    return dict(par=par, dpar=dpar, weights=wtxt, props=props)


def write_deck(x, folder, templates, stem="salmon"):
    """Write the five Raven input files for row x into `folder`; return the stem path.

    `templates` is the dict returned by `load_templates`.
    """
    vals = placeholder_values(x)
    os.makedirs(folder, exist_ok=True)
    for ext in TEMPLATE_KEYS.values():
        with open(os.path.join(folder, f"{stem}.{ext}"), "w") as fh:
            fh.write(templates[ext].format(**vals))
    return os.path.join(folder, stem)


# --------------------------------------------------------------------------
# Reading Raven's output
# --------------------------------------------------------------------------
def read_hydrograph(path):
    """Simulated and observed daily flow (m^3/s) from Raven's Hydrographs.csv.

    Raven reports period-ending values: the row dated d holds the mean flow of
    the day that ends at 00:00 on d (the observation of calendar day d-1).
    """
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    dates = [row[1] for row in rows]
    qsim = np.array([float(row[4]) if row[4].strip() else np.nan for row in rows])
    qobs = np.array([float(row[5]) if len(row) > 5 and row[5].strip() else np.nan for row in rows])
    return header, dates, qsim, qobs


def summaries(dates, qsim, qobs):
    """Scalar outputs over the rows dated 1991-01-01 ... 2010-12-31."""
    i0 = dates.index(EVAL_START)
    i1 = dates.index(END) + 1
    qs, qo = qsim[i0:i1], qobs[i0:i1]
    if qs.size != 7305 or not np.all(np.isfinite(qs)):
        raise RuntimeError("simulated flow missing or not finite in 1991-2010")
    ok = np.isfinite(qo) & (qo >= 0)
    e = qs[ok] - qo[ok]
    nse = 1.0 - np.sum(e ** 2) / np.sum((qo[ok] - qo[ok].mean()) ** 2)
    r = np.corrcoef(qs[ok], qo[ok])[0, 1]
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (qs[ok].std() / qo[ok].std() - 1) ** 2
                        + (qs[ok].mean() / qo[ok].mean() - 1) ** 2)
    return dict(mean_Q=float(qs.mean()), NSE=float(nse), KGE=float(kge),
                n_obs=int(ok.sum()))


def read_diagnostics(path):
    """Raven's Diagnostics.csv as {column: float}."""
    with open(path) as fh:
        head = fh.readline().strip().split(",")
        vals = fh.readline().strip().split(",")
    out = {}
    for h, v in zip(head, vals):
        try:
            out[h.strip()] = float(v)
        except ValueError:
            pass
    return out


def run(x, raven_exe, obs_dir, templates, workdir=None, keep=False):
    """Run Raven at one input row.

    Returns a dict with mean_Q, NSE, KGE (computed here from Hydrographs.csv),
    n_obs and Raven's own DIAG_NSE, DIAG_KGE, DIAG_RMSE. `templates` is the
    dict returned by `load_templates`.
    """
    raven_exe = os.path.abspath(raven_exe)
    obs_dir = os.path.abspath(obs_dir)
    tmp = tempfile.mkdtemp(prefix="raven_", dir=workdir)
    try:
        stem = write_deck(x, tmp, templates)
        os.symlink(obs_dir, os.path.join(tmp, BASIN["id"]))
        out = os.path.join(tmp, "output")
        os.makedirs(out)
        proc = subprocess.run([raven_exe, stem, "-o", out + "/"], cwd=tmp,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        hyd = os.path.join(out, "Hydrographs.csv")
        dia = os.path.join(out, "Diagnostics.csv")
        if proc.returncode != 0 or not os.path.exists(hyd) or not os.path.exists(dia):
            err = os.path.join(out, "Raven_errors.txt")
            msg = open(err).read() if os.path.exists(err) else proc.stdout.decode()[-2000:]
            raise RuntimeError("Raven failed:\n" + msg)
        _, dates, qsim, qobs = read_hydrograph(hyd)
        res = summaries(dates, qsim, qobs)
        diag = read_diagnostics(dia)
        res["DIAG_NSE"] = diag.get("DIAG_NASH_SUTCLIFFE", np.nan)
        res["DIAG_KGE"] = diag.get("DIAG_KLING_GUPTA", np.nan)
        res["DIAG_RMSE"] = diag.get("DIAG_RMSE", np.nan)
        return res
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
