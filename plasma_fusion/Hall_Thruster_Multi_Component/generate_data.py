"""Generate the SPT-100 coupled cathode-thruster-plume data set.

Model: Marks, Eckels, Mora & Gorodetsky, "Uncertainty quantification of a multi-component Hall
thruster model at varying facility pressures", J. Appl. Phys. (2025), doi:10.1063/5.0283796.

Pipeline per sample (all at one fixed operating condition, see OPERATING below):
  1. cathode coupling voltage Vcc from Eq. (2)            (spt100_components.py, numpy)
  2. HallThruster.jl v0.18.7 1-D discharge -> ID, IB, T, u_ion(exit), time-averaged over the
     last 500 us of a 1 ms run                            (run_hallthruster.jl, Julia)
  3. semi-empirical plume -> divergence angle, Tc = T cos(phi_d), centreline j_ion at 1 m
                                                          (spt100_components.py, numpy)

Usage:
  python generate_data.py design     # write the Latin hypercube design and the Julia input file
  python generate_data.py run        # run HallThruster.jl on all rows not yet done (resumable)
  python generate_data.py assemble   # combine into hall_thruster_spt100.csv
  python generate_data.py            # all three
  python generate_data.py check      # no Julia: redraw the design and recompute the cathode and plume
                                     # columns and the oscillation flag from the stored discharge
                                     # outputs; compare with the CSV

Oscillation flag (last CSV column, `oscillating`): 1 if the standard deviation over the averaging
window of the discharge current I_D(t), divided by its mean, exceeds OSC_THRESHOLD = 1 %; else 0
(NaN for failed runs). The flag is derived from the stored columns ID_std and ID; no value is dropped.

Environment: JULIA (path to julia >= 1.10) and JULIA_PROJECT (an environment with HallThruster
v0.18.7) must be set for the `run` stage; BFIELD is the path to the SPT-100 magnetic-field file
(bfield_spt100.csv from HallThrusterPEM, see DATA.md).
"""
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import qmc

from spt100_components import cathode_coupling_voltage, plume

SEED = 662
M = 440
OSC_THRESHOLD = 0.01   # std/mean of I_D(t) over the averaging window above which a run is "oscillating"

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(HERE, "design_unit.csv")
JL_IN = os.path.join(HERE, "julia_inputs.csv")
JL_OUT = os.path.join(HERE, "julia_outputs.csv")
OUT = os.path.join(HERE, "hall_thruster_spt100.csv")

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


def design():
    U = qmc.LatinHypercube(d=len(PARAMS), seed=SEED).random(M)
    np.savetxt(DESIGN, U, delimiter=",", fmt="%.12f")
    X = unit_to_physical(U)
    p = dict(zip(NAMES, X.T))
    Vcc, clipped = cathode_coupling_voltage(OPERATING["PB_Torr"], p["Tec"], p["Vvac"], p["PT"], p["Pstar"],
                                            OPERATING["Vd"])
    rows = np.column_stack([
        np.arange(M), np.full(M, OPERATING["Vd"]), np.full(M, OPERATING["mdot_a"]),
        np.full(M, OPERATING["PB_Torr"]), Vcc, p["Tec"], p["alpha_anom"], p["beta_anom"], p["z_anom"],
        p["L_anom"], p["dz_anom"], p["u_n"], p["c_w"], p["f_n"],
    ])
    np.savetxt(JL_IN, rows, delimiter=",", fmt="%.12e")
    print(f"design: M = {M}, Vcc clipped to [0, Vd] in {int(clipped.sum())} samples")


def run():
    julia = os.environ.get("JULIA", "julia")
    bfield = os.environ["BFIELD"]
    cmd = [julia, "--startup-file=no", os.path.join(HERE, "run_hallthruster.jl"), bfield, JL_IN, JL_OUT]
    t0 = time.time()
    subprocess.run(cmd, check=True, env={**os.environ, "JULIA_NUM_THREADS": "1"})
    print(f"run stage wall time {time.time() - t0:.0f} s")


def oscillation_flag(status, ID, ID_std):
    """1.0 where std(I_D)/mean(I_D) over the averaging window exceeds OSC_THRESHOLD, 0.0 if not, NaN if failed."""
    flag = np.full(len(status), np.nan)
    ok = status == 0
    flag[ok] = (ID_std[ok] / ID[ok] > OSC_THRESHOLD).astype(float)
    return flag


def assemble():
    U = np.loadtxt(DESIGN, delimiter=",")
    X = unit_to_physical(U)
    p = dict(zip(NAMES, X.T))
    Vcc, _ = cathode_coupling_voltage(OPERATING["PB_Torr"], p["Tec"], p["Vvac"], p["PT"], p["Pstar"],
                                      OPERATING["Vd"])
    raw = np.loadtxt(JL_OUT, delimiter=",", ndmin=2)
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
    np.savetxt(OUT, data, delimiter=",", fmt="%.10e", header=header)
    print(f"assembled {OUT}: {ok.sum()} ok, {(~ok).sum()} failed "
          f"(status counts {np.bincount(status, minlength=4).tolist()}); "
          f"HallThruster.jl wall time {np.nansum(wall):.0f} s; oscillating (ID_std/ID > {OSC_THRESHOLD:g}): "
          f"{int(np.nansum(data[:, -1]))}")


def check():
    data = np.loadtxt(OUT, delimiter=",")
    m = len(PARAMS)
    U = qmc.LatinHypercube(d=m, seed=SEED).random(M)
    X = unit_to_physical(U)
    p = dict(zip(NAMES, X.T))
    dx = np.max(np.abs(X - data[:, :m]) / np.abs(X).clip(1e-300))
    Vcc, _ = cathode_coupling_voltage(OPERATING["PB_Torr"], p["Tec"], p["Vvac"], p["PT"], p["Pstar"],
                                      OPERATING["Vd"])
    ok = data[:, m] == 0
    IB, T = data[ok, m + 3], data[ok, m + 4]
    pl = plume(OPERATING["PB_Torr"], IB, T, p["c0"][ok], p["c1"][ok], p["c2"][ok], p["c3"][ok],
               p["c4"][ok], p["c5"][ok])
    rel = lambda a, b: np.max(np.abs(a - b) / np.abs(b))
    flag = oscillation_flag(data[:, m].astype(int), data[:, m + 2], data[:, m + 9])
    same = np.array_equal(flag, data[:, -1], equal_nan=True)
    print(f"check: inputs max rel diff {dx:.1e}; Vcc {rel(Vcc, data[:, m + 1]):.1e}; "
          f"Tc {rel(pl['Tc'], data[ok, m + 5]):.1e}; j_peak {rel(pl['j_peak'], data[ok, m + 7]):.1e}; "
          f"phi_d {rel(pl['phi_d'], data[ok, m + 8]):.1e}; oscillating flag reproduced: {same} "
          f"({int(np.nansum(data[:, -1]))} runs)")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["design", "run", "assemble"]
    for s in stages:
        {"design": design, "run": run, "assemble": assemble, "check": check}[s]()
