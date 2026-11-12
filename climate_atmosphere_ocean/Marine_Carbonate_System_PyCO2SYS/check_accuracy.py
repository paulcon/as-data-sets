"""Recompute the numbers of the Accuracy section of DATA.md (Marine_Carbonate_System_PyCO2SYS).

    python check_accuracy.py              all checks (needs PyCO2SYS 1.8.3; about a minute)
    python check_accuracy.py --inputs-only   check 0 only (numpy; no PyCO2SYS needed)

Run it from any directory; it reads the shipped files next to this file and writes nothing. It does
not regenerate the data: checks 3 and 4 recompute a few stored values and compare. Each line prints
the recomputed value; DATA.md quotes these values.

Checks:
 0. The shipped inputs of both studies against NOMINAL + SD * default_rng(seed).standard_normal((1000, 14))
    (seeds 20220104 and 20220105): largest absolute difference.
 1. The override identity at the nominal point (both pressures): PyCO2SYS with the constants passed back
    as overrides (offsets 0, borate multiplier 1) against PyCO2SYS without overrides.
 2. PyCO2SYS 1.8.3 with its own default options at the Table 4 conditions (no hypothetical alpha, beta
    systems): pH and pCO2, to compare with the v1.8.0 defaults the data use.
 3. nominal_reference.csv recomputed (the five rows): largest relative difference per row.
 4. The stored outputs and gradients of the first 20 samples of each study recomputed: largest
    relative difference (gradients: per row, 2-norm).
 5. Step sizes: gradients of the first 20 samples of each study with steps 10 times larger and 10
    times smaller than the shipped ones: largest relative difference (row 2-norm) per output.
 6. PyCO2SYS's built-in derivatives (grads_of / grads_wrt: forward differences with the paper's
    Appendix H steps) at the first 20 samples of each study with the constants at their parameterised
    values (offsets 0, borate multiplier 1), where the two definitions coincide: largest relative
    difference (row 2-norm) per output, and per input in sd-scaled units relative to the gradient norm.

Needs numpy and PyCO2SYS 1.8.3 (pip install PyCO2SYS==1.8.3 pandas), imported through
carbonate_driver.py; without PyCO2SYS the script stops with an install message after check 0.
"""
import argparse
import os
import platform
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carbonate_driver as cm  # noqa: E402
import generate_data as gd  # noqa: E402

GRAD_WRT = ("par1", "par2", "temperature", "salinity", "total_silicate", "total_phosphate",
            "pk_CO2", "pk_carbonic_1", "pk_carbonic_2", "pk_borate", "pk_water",
            "pk_aragonite", "pk_calcite", "total_borate")
N_SPOT = 20


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=",", ndmin=2)


def rel_rows(A, B):
    return np.linalg.norm(A - B, axis=-1) / np.linalg.norm(B, axis=-1)


def check_inputs():
    for study in gd.STUDIES:
        d = np.max(np.abs(load(f"inputs_{study}.csv") - gd.draw(study)))
        print(f"0. inputs_{study}.csv vs NOMINAL + SD * default_rng({gd.STUDIES[study]['seed']}): "
              f"largest absolute difference {d:.1e}")


def check_override():
    pyco2 = cm.pyco2
    for study in gd.STUDIES:
        held = gd.STUDIES[study]["held"]
        _, kw = cm._base_kwargs(cm.NOMINAL[None, :], held)
        r0 = pyco2.sys(**kw)
        plain = np.array([np.ravel(r0[o])[0] for o in cm.OUTPUTS])
        over = cm.evaluate(cm.NOMINAL, held=held)[0]
        print(f"1. override identity, {study} study: largest relative difference "
              f"{np.max(np.abs(over - plain) / np.abs(plain)):.1e}")


def check_v183_defaults():
    r = cm.pyco2.sys(par1=2300.0, par2=2100.0, par1_type=1, par2_type=2, salinity=33.0, temperature=22.0,
                     pressure=1234.0, total_silicate=10.0, total_phosphate=1.0, total_ammonia=2.0,
                     total_sulfide=3.0)
    print(f"2. PyCO2SYS {cm.pyco2.__version__} own defaults at the Table 4 conditions: "
          f"pH {float(np.ravel(r['pH_total'])[0]):.4f}, pCO2 {float(np.ravel(r['pCO2'])[0]):.2f} uatm")


def check_reference():
    stored, new = load("nominal_reference.csv"), gd.reference_rows()
    d = np.max(np.abs(new - stored) / np.abs(stored), axis=1)
    print("3. nominal_reference.csv recomputed: largest relative difference per row "
          + ", ".join(f"{v:.1e}" for v in d))


def check_spot():
    for study in gd.STUDIES:
        held = gd.STUDIES[study]["held"]
        X = load(f"inputs_{study}.csv")[:N_SPOT]
        Y = load(f"outputs_{study}.csv")[:N_SPOT]
        G = load(f"gradients_{study}.csv")[:N_SPOT].reshape(N_SPOT, 4, 14)
        dy = np.max(np.abs(cm.evaluate(X, held=held) - Y) / np.abs(Y))
        dg = np.max(rel_rows(cm.gradients(X, held=held), G))
        print(f"4. {study} study, first {N_SPOT} samples recomputed: outputs {dy:.1e}, gradients {dg:.1e} "
              "(largest relative difference)")


def check_steps():
    for study in gd.STUDIES:
        held = gd.STUDIES[study]["held"]
        X = gd.draw(study)[:N_SPOT]
        G = cm.gradients(X, held=held)
        for fac in (10.0, 0.1):
            Gf = cm.gradients(X, steps=cm.STEPS * fac, held=held)
            e = np.max(rel_rows(Gf, G), axis=0)
            print(f"5. {study} study, steps x{fac:g}: largest relative difference per output "
                  + ", ".join(f"{o} {v:.1e}" for o, v in zip(cm.OUTPUTS, e)))


def check_builtin_derivatives():
    for study in gd.STUDIES:
        X = gd.draw(study)[:N_SPOT]
        X[:, 6:13] = 0.0
        X[:, 13] = 1.0
        held = gd.STUDIES[study]["held"]
        G = cm.gradients(X, held=held)
        _, kw = cm._base_kwargs(X, held)
        r = cm.pyco2.sys(**kw, grads_of=list(cm.OUTPUTS), grads_wrt=list(GRAD_WRT))
        Gp = np.empty_like(G)
        for q, o in enumerate(cm.OUTPUTS):
            for j, w in enumerate(GRAD_WRT):
                Gp[:, q, j] = np.broadcast_to(r[f"d_{o}__d_{w}"], (X.shape[0],))
        Gp[:, :, 13] *= np.asarray(r["total_borate"])[:, None]        # d/df_B = B_T d/dB_T
        e = np.max(rel_rows(Gp, G), axis=0)
        print(f"6. {study} study vs PyCO2SYS forward differences: largest relative difference per output "
              + ", ".join(f"{o} {v:.1e}" for o, v in zip(cm.OUTPUTS, e)))
        sd = gd.STUDIES[study]["sd"]
        ej = np.max(np.abs(G - Gp) * sd / np.linalg.norm(G * sd, axis=2, keepdims=True), axis=(0, 1))
        print("   per input (sd-scaled, relative to the gradient norm): "
              + ", ".join(f"{n} {v:.1e}" for n, v in zip(cm.NAMES, ej)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs-only", action="store_true", help="check 0 only (no PyCO2SYS needed)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    print(f"numpy {np.__version__}, Python {platform.python_version()}, {platform.platform()}")
    check_inputs()
    if a.inputs_only:
        return
    if cm.pyco2 is None:
        print(cm.INSTALL.replace("and rerun.", "and rerun check_accuracy.py (or use --inputs-only)."), file=sys.stderr)
        sys.exit(3)
    print(f"PyCO2SYS {cm.pyco2.__version__}")
    warnings.simplefilter("ignore")
    check_override()
    check_v183_defaults()
    check_reference()
    check_spot()
    check_steps()
    check_builtin_derivatives()
    print(f"run time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
