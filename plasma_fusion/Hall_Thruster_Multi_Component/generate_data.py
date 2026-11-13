"""Generate the SPT-100 coupled cathode-thruster-plume data set.

Model: Marks, Eckels, Mora & Gorodetsky, "Uncertainty quantification of a multi-component Hall
thruster model at varying facility pressures", J. Appl. Phys. (2025), doi:10.1063/5.0283796.

Pipeline per sample (all at one fixed operating condition, see OPERATING below):
  1. cathode coupling voltage Vcc from Eq. (2)            (spt100_components.py, numpy)
  2. HallThruster.jl v0.18.7 1-D discharge -> ID, IB, T, u_ion(exit), time-averaged over the
     last 500 us of a 1 ms run                            (run_hallthruster.jl, Julia)
  3. semi-empirical plume -> divergence angle, Tc = T cos(phi_d), centreline j_ion at 1 m
                                                          (spt100_components.py, numpy)

Usage (DIR is a working directory outside this one; every file the stages write goes there):
  python generate_data.py design --out DIR     # the Latin hypercube design and the Julia input file
  python generate_data.py run --out DIR        # run HallThruster.jl on all rows not yet done (resumable)
  python generate_data.py assemble --out DIR   # combine into DIR/hall_thruster_spt100.csv
  python generate_data.py --out DIR            # all three
  python generate_data.py check                # no Julia: redraw the design and recompute the cathode and
                                               # plume columns and the oscillation flag from the stored
                                               # discharge outputs; compare with the shipped CSV (read only)
  python generate_data.py --check              # `check`, then design, run and assemble in a temporary
                                               # directory (all 440 rows; needs Julia) and compare the new
                                               # CSV with the shipped one

Both checks print the largest differences from the shipped CSV (inputs relative to each column's largest
magnitude, the other columns elementwise relative) and fail (exit status 1) only on structural
mismatches (shape, NaN pattern, status column, oscillation flag, inputs differing by more than
INPUT_RTOL). --check exits with status 3 after the `check` stage when Julia is missing. --check takes no stages; with --out DIR
it keeps the files in DIR.

The shipped CSV is the record of the runs: the stages never write into this directory, and
`assemble` writes a new file in DIR for comparison. The `run` stage needs Julia and HallThruster.jl
v0.18.7; without them the script exits with status 3 and says what to install (DATA.md has the
versions of the shipped runs).

Oscillation flag (last CSV column, `oscillating`): 1 if the standard deviation over the averaging
window of the discharge current I_D(t), divided by its mean, exceeds OSC_THRESHOLD = 1 %; else 0
(NaN for failed runs). The flag is derived from the stored columns ID_std and ID; no value is dropped.

Environment for the `run` stage: JULIA (path to julia >= 1.10; default: julia on the PATH),
JULIA_PROJECT (a Julia environment with HallThruster v0.18.7) and BFIELD (path to the SPT-100
magnetic-field file bfield_spt100.csv from HallThrusterPEM, see DATA.md).
"""
import argparse
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.stats import qmc

from spt100_components import cathode_coupling_voltage, plume

SEED = 662
M = 440
OSC_THRESHOLD = 0.01   # std/mean of I_D(t) over the averaging window above which a run is "oscillating"
INPUT_RTOL = 1e-8   # inputs are deterministic draws: a larger difference means a different seed or range

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "hall_thruster_spt100.csv"               # the shipped file (in HERE); `assemble` writes one in --out
HALLTHRUSTER_VERSION = "0.18.7"
INSTALL = ("HallThruster.jl v" + HALLTHRUSTER_VERSION + " is needed to run the thruster model (the `run` stage "
           "of generate_data.py, checks 5-8 of check_accuracy.py). Install Julia (>= 1.10; "
           "the shipped runs used Julia 1.13.3) and, in a Julia environment ENV, HallThruster.jl v"
           + HALLTHRUSTER_VERSION + ": julia --project=ENV -e 'using Pkg; Pkg.add(name=\"HallThruster\", version=\""
           + HALLTHRUSTER_VERSION + "\")'. Then set JULIA (the julia executable, if not on the PATH), "
           "JULIA_PROJECT=ENV and BFIELD (the SPT-100 magnetic-field file; DATA.md says where to get it), "
           "and rerun this script.")


def paths(out):
    return {"design": os.path.join(out, "design_unit.csv"), "jl_in": os.path.join(out, "julia_inputs.csv"),
            "jl_out": os.path.join(out, "julia_outputs.csv"), "csv": os.path.join(out, CSV)}


def require_julia():
    """Exit with status 3 and an install message unless Julia, HallThruster.jl and the B-field file are there.
    Returns the julia executable."""
    julia = os.environ.get("JULIA") or "julia"
    exe = julia if os.path.isfile(julia) else shutil.which(julia)
    problem = None
    if exe is None:
        problem = f"Julia not found ({julia!r})."
    elif not os.path.isfile(os.environ.get("BFIELD", "")):
        problem = "BFIELD is not set to an existing file."
    else:
        r = subprocess.run([exe, "--startup-file=no", "-e", "using HallThruster; print(pkgversion(HallThruster))"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            problem = "HallThruster.jl cannot be loaded in JULIA_PROJECT."
        elif r.stdout.strip() != HALLTHRUSTER_VERSION:
            print(f"warning: HallThruster.jl {r.stdout.strip()} found; the shipped runs used v{HALLTHRUSTER_VERSION}",
                  file=sys.stderr)
    if problem:
        print(f"{problem} {INSTALL}", file=sys.stderr)
        sys.exit(3)
    return exe

# Fixed operating condition: Diamant et al. (2014) L3 facility, condition 4 (300 V, 5.665 mg/s total
# flow with anode:cathode flow ratio 12, background pressure 12.3 uTorr-Xe).
OPERATING = {"Vd": 300.0, "PB_Torr": 1.23e-5, "mdot_a": 5.665e-6 * 12.0 / 13.0}

# 18 model parameters, Table I of the paper. (name, units, lower, upper, scale); 'log10' means the
# parameter is 10**x with x ~ U(lower, upper); bounds of log10 parameters are exponents.
PARAMS = [
    ("Tec", "eV", 1.0, 6.0, "lin"),
    ("Vvac", "V", 0.0, 60.0, "lin"),
    ("PT", "uTorr", 20.0, 200.0, "lin"),
    ("Pstar", "uTorr", 1.0, 100.0, "lin"),
    ("alpha_anom", "-", 0.0, 1.0, "lin"),
    ("beta_anom", "-", 0.0, 1.0, "lin"),
    ("z_anom", "-", 0.75, 1.5, "lin"),
    ("L_anom", "-", 0.0, 0.5, "lin"),
    ("dz_anom", "-", 0.0, 0.5, "lin"),
    ("u_n", "m/s", 100.0, 500.0, "lin"),
    ("c_w", "-", 0.5, 1.5, "lin"),
    ("f_n", "-", 1.0, 10.0, "lin"),
    ("c0", "-", 0.0, 1.0, "lin"),
    ("c1", "-", 0.1, 0.9, "lin"),
    ("c2", "rad/Pa", -15.0, 15.0, "lin"),
    ("c3", "rad", 0.2, np.pi / 2, "lin"),
    ("c4", "m^-3/Pa", 18.0, 22.0, "log10"),
    ("c5", "m^-3", 14.0, 18.0, "log10"),
]
NAMES = [p[0] for p in PARAMS]


def unit_to_physical(U):
    X = np.empty_like(U)
    for j, (_, _, lo, hi, sc) in enumerate(PARAMS):
        x = lo + (hi - lo) * U[:, j]
        X[:, j] = 10.0**x if sc == "log10" else x
    return X


def design(P):
    U = qmc.LatinHypercube(d=len(PARAMS), seed=SEED).random(M)
    np.savetxt(P["design"], U, delimiter=",", fmt="%.12f")
    X = unit_to_physical(U)
    p = dict(zip(NAMES, X.T))
    Vcc, clipped = cathode_coupling_voltage(OPERATING["PB_Torr"], p["Tec"], p["Vvac"], p["PT"], p["Pstar"],
                                            OPERATING["Vd"])
    rows = np.column_stack([
        np.arange(M), np.full(M, OPERATING["Vd"]), np.full(M, OPERATING["mdot_a"]),
        np.full(M, OPERATING["PB_Torr"]), Vcc, p["Tec"], p["alpha_anom"], p["beta_anom"], p["z_anom"],
        p["L_anom"], p["dz_anom"], p["u_n"], p["c_w"], p["f_n"],
    ])
    np.savetxt(P["jl_in"], rows, delimiter=",", fmt="%.12e")
    print(f"design: M = {M}, Vcc clipped to [0, Vd] in {int(clipped.sum())} samples")


def run(P):
    julia = require_julia()
    bfield = os.environ["BFIELD"]
    cmd = [julia, "--startup-file=no", os.path.join(HERE, "run_hallthruster.jl"), bfield, P["jl_in"], P["jl_out"]]
    t0 = time.time()
    subprocess.run(cmd, check=True, env={**os.environ, "JULIA_NUM_THREADS": "1"})
    print(f"run stage wall time {time.time() - t0:.0f} s")


def oscillation_flag(status, ID, ID_std):
    """1.0 where std(I_D)/mean(I_D) over the averaging window exceeds OSC_THRESHOLD, 0.0 if not, NaN if failed."""
    flag = np.full(len(status), np.nan)
    ok = status == 0
    flag[ok] = (ID_std[ok] / ID[ok] > OSC_THRESHOLD).astype(float)
    return flag


def assemble(P):
    U = np.loadtxt(P["design"], delimiter=",")
    X = unit_to_physical(U)
    p = dict(zip(NAMES, X.T))
    Vcc, _ = cathode_coupling_voltage(OPERATING["PB_Torr"], p["Tec"], p["Vvac"], p["PT"], p["Pstar"],
                                      OPERATING["Vd"])
    if not os.path.isfile(P["jl_out"]):
        sys.exit(f"{P['jl_out']} not found: run the `run` stage first")
    raw = np.loadtxt(P["jl_out"], delimiter=",", ndmin=2)
    raw = raw[np.argsort(raw[:, 0])]
    if raw.shape[0] != M or np.any(raw[:, 0] != np.arange(M)):
        raise RuntimeError("julia_outputs.csv is incomplete; rerun the `run` stage")
    status = raw[:, 1].astype(int)
    ID, IB, T, uexit, ID_std, T_std, ID_a, ID_b, T_a, T_b, wall = raw[:, 2:13].T
    Tc = np.full(M, np.nan); phid = np.full(M, np.nan); jpk = np.full(M, np.nan)
    ok = status == 0
    pl = plume(OPERATING["PB_Torr"], IB[ok], T[ok], p["c0"][ok], p["c1"][ok], p["c2"][ok], p["c3"][ok],
               p["c4"][ok], p["c5"][ok])
    Tc[ok], phid[ok], jpk[ok] = pl["Tc"], pl["phi_d"], pl["j_peak"]
    cols = NAMES + ["status", "Vcc", "ID", "IB", "T", "Tc", "u_exit", "j_peak", "phi_d",
                    "ID_std", "T_std", "ID_a", "ID_b", "T_a", "T_b", "wall_s", "oscillating"]
    units = [f"{n} [{u}]" for n, u, *_ in PARAMS] + [
        "status [0 ok; 1 solver retcode; 2 exception; 3 non-finite]", "Vcc [V]", "ID [A]", "IB [A]", "T [N]",
        "Tc [N]", "u_exit [m/s]", "j_peak [A/m^2]", "phi_d [rad]", "ID_std [A]", "T_std [N]", "ID_a [A]",
        "ID_b [A]", "T_a [N]", "T_b [N]", "wall_s [s]", "oscillating [1 if ID_std/ID > 0.01; 0 steady]"]
    data = np.column_stack([X, status, Vcc, ID, IB, T, Tc, uexit, jpk, phid, ID_std, T_std, ID_a, ID_b,
                            T_a, T_b, wall, oscillation_flag(status, ID, ID_std)])
    header = ("SPT-100 coupled cathode/HallThruster.jl v0.18.7/plume model (Marks et al. 2025, "
              f"doi:10.1063/5.0283796); LHS M={M} seed={SEED}; Vd=300 V, PB=1.23e-5 Torr, "
              "mdot_a=5.229e-6 kg/s; NaN = failed run; oscillating = std/mean of ID(t) > 1 %\n" + ", ".join(units))
    np.savetxt(P["csv"], data, delimiter=",", fmt="%.10e", header=header)
    print(f"assembled {P['csv']}: {ok.sum()} ok, {(~ok).sum()} failed "
          f"(status counts {np.bincount(status, minlength=4).tolist()}); "
          f"HallThruster.jl wall time {np.nansum(wall):.0f} s; oscillating (ID_std/ID > {OSC_THRESHOLD:g}): "
          f"{int(np.nansum(data[:, -1]))}")


def summary(diffs):
    """Print the one summary line: the largest of the (difference, quantity) pairs."""
    d, name = max(diffs, key=lambda t: t[0])
    print(f"largest difference from the shipped files: {d:.1e} (relative, {name})")


def check(ship=HERE):
    """Redraw the inputs and recompute the numpy columns; (False on a structural mismatch, differences)."""
    data = np.loadtxt(os.path.join(ship, CSV), delimiter=",")
    m = len(PARAMS)
    if data.shape != (M, m + 17):
        print(f"FAIL: {CSV}: shape {data.shape}, expected {(M, m + 17)}")
        return False, []
    U = qmc.LatinHypercube(d=m, seed=SEED).random(M)
    X = unit_to_physical(U)
    p = dict(zip(NAMES, X.T))
    dx = np.max(np.abs(X - data[:, :m]) / np.abs(data[:, :m]).max(axis=0).clip(1e-300))   # relative to column max
    Vcc, _ = cathode_coupling_voltage(OPERATING["PB_Torr"], p["Tec"], p["Vvac"], p["PT"], p["Pstar"],
                                      OPERATING["Vd"])
    ok = data[:, m] == 0
    IB, T = data[ok, m + 3], data[ok, m + 4]
    pl = plume(OPERATING["PB_Torr"], IB, T, p["c0"][ok], p["c1"][ok], p["c2"][ok], p["c3"][ok],
               p["c4"][ok], p["c5"][ok])
    rel = lambda a, b: np.max(np.abs(a - b) / np.abs(b))
    flag = oscillation_flag(data[:, m].astype(int), data[:, m + 2], data[:, m + 9])
    same = np.array_equal(flag, data[:, -1], equal_nan=True)
    d = [rel(Vcc, data[:, m + 1]), rel(pl['Tc'], data[ok, m + 5]), rel(pl['j_peak'], data[ok, m + 7]),
         rel(pl['phi_d'], data[ok, m + 8])]
    print(f"check: largest relative difference: inputs (to the column maximum) {dx:.1e}; elementwise: "
          f"Vcc {d[0]:.1e}; Tc {d[1]:.1e}; "
          f"j_peak {d[2]:.1e}; phi_d {d[3]:.1e}; oscillating flag reproduced: {same} "
          f"({int(np.nansum(data[:, -1]))} runs)")
    if not same:
        print("FAIL: check: the oscillation flag differs from the shipped column")
    if dx > INPUT_RTOL:
        print(f"FAIL: check: the redrawn inputs differ from the shipped ones by more than {INPUT_RTOL:g} (relative)")
    diffs = [(dx, f"{CSV} inputs")] + [(v, f"{CSV} {c}") for v, c in zip(d, ("Vcc", "Tc", "j_peak", "phi_d"))]
    return bool(same and dx <= INPUT_RTOL), diffs


def compare_rerun(P, ship=HERE):
    """Compare the CSV that `assemble` wrote in --out with the shipped one, column by column.

    Returns (False on a structural mismatch, list of (difference, quantity))."""
    new = np.loadtxt(P["csv"], delimiter=",")
    old = np.loadtxt(os.path.join(ship, CSV), delimiter=",")
    if new.shape != old.shape:
        print(f"FAIL: rerun: shape {new.shape} differs from the shipped {old.shape}")
        return False, []
    m, n = len(PARAMS), old.shape[1]
    same_nan = np.array_equal(np.isnan(new), np.isnan(old))
    exact = [m, n - 1]                                                   # status and the oscillation flag
    same_flags = np.array_equal(new[:, exact], old[:, exact], equal_nan=True)

    def rel(cols):
        a, b = new[:, cols], old[:, cols]
        fin = ~np.isnan(b) & ~np.isnan(a)
        return float(np.max(np.abs(a[fin] - b[fin]) / np.maximum(np.abs(b[fin]), 1e-300))) if fin.any() else 0.0
    dx = float(np.max(np.abs(new[:, :m] - old[:, :m]) / np.abs(old[:, :m]).max(axis=0).clip(1e-300)))  # column max
    d = rel([j for j in range(m, n) if j not in exact and j != n - 2])  # all other columns but wall_s
    print(f"rerun: largest relative difference from the shipped CSV: inputs (to the column maximum) {dx:.1e}, "
          f"other columns but wall_s (elementwise) {d:.1e}; NaN pattern equal: {same_nan}; status and oscillation flag equal: {same_flags}")
    if not (same_nan and same_flags):
        print("FAIL: rerun: the NaN pattern, the status column or the oscillation flag differs from the shipped CSV")
    if dx > INPUT_RTOL:
        print(f"FAIL: rerun: the inputs differ from the shipped ones by more than {INPUT_RTOL:g} (relative)")
    passed = bool(same_nan and same_flags and dx <= INPUT_RTOL)
    return passed, [(dx, f"rerun {CSV} inputs"), (d, f"rerun {CSV} other columns but wall_s")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="*", help="design, run, assemble or check (default: design run assemble)")
    ap.add_argument("--out", help="working directory for design, run and assemble (not this directory; "
                                  "with --check, default: a temporary directory)")
    ap.add_argument("--check", action="store_true",
                    help="the check stage, then design, run and assemble in --out, compared with the shipped CSV")
    args = ap.parse_args()
    if args.check:
        if args.stages:
            ap.error("--check takes no stages")
        if not args.out:
            args.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, args.out, ignore_errors=True)
        if os.path.abspath(args.out) == HERE:
            ap.error("--out must not be this directory: the shipped CSV is the record of the runs")
        ok, diffs = check()
        if not ok:
            if diffs:
                summary(diffs)
            sys.exit(1)
        os.makedirs(args.out, exist_ok=True)
        P = paths(args.out)
        try:
            require_julia()
        except SystemExit:
            summary(diffs)              # the `check` stage ran; the rerun needs Julia (exit status 3)
            raise
        for s in ("design", "run", "assemble"):
            {"design": design, "run": run, "assemble": assemble}[s](P)
        ok, more = compare_rerun(P)
        summary(diffs + more)
        if not ok:
            sys.exit(1)
        return
    stages = args.stages or ["design", "run", "assemble"]
    bad = [s for s in stages if s not in ("design", "run", "assemble", "check")]
    if bad:
        ap.error(f"unknown stage(s) {bad}")
    if any(s != "check" for s in stages):
        if not args.out:
            ap.error("design, run and assemble need --out DIR (a directory outside this one)")
        if os.path.abspath(args.out) == HERE:
            ap.error("--out must not be this directory: the shipped CSV is the record of the runs")
        os.makedirs(args.out, exist_ok=True)
    if "run" in stages:
        require_julia()                 # before any stage writes, so a missing Julia stops the script at once
    P = paths(args.out) if args.out else None
    for s in stages:
        if s == "check":
            ok, diffs = check()
            if diffs:
                summary(diffs)
            if not ok:
                sys.exit(1)
        else:
            {"design": design, "run": run, "assemble": assemble}[s](P)


if __name__ == "__main__":
    main()
