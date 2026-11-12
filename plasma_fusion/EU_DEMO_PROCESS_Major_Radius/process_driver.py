"""Run one PROCESS v3.4.2 optimisation for one input vector and return scalar outputs.

Part of the EU_DEMO_PROCESS_Major_Radius data set (code MIT). Used by generate_data.py and
check_accuracy.py. Runs in-process (process.main.SingleRun) so the ~6 s import and numba start-up
are paid once per Python process. Needs PROCESS v3.4.2 (imported only inside run_one).
"""
import contextlib, gzip, importlib.util, io, os, re, shutil, sys, time
import numpy as np

NAMES = ["hfact", "psepbqar_max", "core_radius", "f_w", "kappa", "f_he"]

# Outputs read from the MFILE (first occurrence of "(name)").
OUTPUTS = ["ifail", "rmajor", "sqsumsq", "n_solver_iterations", "convergence_parameter",
           "norm_objf", "p_plant_electric_net_mw", "t_plant_pulse_burn", "p_fusion_total_mw",
           "b_plasma_toroidal_on_axis", "q95", "p_div_bt_q_aspect_rmajor_mw",
           "f_nd_impurity_electrons(13)", "capcost", "kappa", "hfact"]

# Iteration variables whose initial values a warm start replaces (17 in the template).
ITERVARS = ["b_plasma_toroidal_on_axis", "temp_plasma_electron_vol_avg_kev",
            "beta_total_vol_avg", "nd_plasma_electrons_vol_avg", "dr_cs", "q95", "dr_bore",
            "f_c_plasma_non_inductive", "t_tf_superconductor_quench", "dr_tf_nose_case",
            "dx_tf_turn_steel", "f_a_tf_turn_cable_copper", "c_tf_turn", "f_a_cs_turn_steel",
            "f_nd_impurity_electrons(13)", "dr_tf_wp_with_insulation"]  # + rmajor via {rmajor0}


def fill_deck(template, x, rmajor0=9.0, maxcal=40, warm=None):
    vals = dict(zip(NAMES, [float(v) for v in x]))
    vals.update(rmajor0=rmajor0, maxcal=int(maxcal))
    txt = template
    for k, v in vals.items():
        txt = txt.replace("{" + k + "}", repr(v) if isinstance(v, float) else str(v))
    if re.search(r"\{(" + "|".join(list(vals)) + r")\}", txt):
        raise ValueError("unfilled placeholder")
    if warm:
        for name, v in warm.items():
            pat = re.compile(r"^(\s*)" + re.escape(name) + r"(\s*=\s*)([^*\n]+)", re.M)
            txt, n = pat.subn(lambda m: f"{m.group(1)}{name}{m.group(2)}{v!r} ", txt)
            if n == 0:
                raise ValueError(f"warm-start variable {name} not in deck")
    return txt


def parse_mfile(path, names=OUTPUTS):
    out = {}
    txt = open(path, encoding="utf-8", errors="replace").read()
    for n in names:
        m = re.search(r"\(" + re.escape(n) + r"\)_*\s+(\S+)", txt)
        out[n] = float(m.group(1)) if m else float("nan")
    return out


def run_one(template, x, workdir, tag, rmajor0=9.0, maxcal=40, warm=None, keep_mfile_gz=None):
    from process.main import SingleRun
    os.makedirs(workdir, exist_ok=True)
    indat = os.path.join(workdir, f"{tag}_IN.DAT")
    open(indat, "w").write(fill_deck(template, x, rmajor0, maxcal, warm))
    t0 = time.process_time(); w0 = time.time()
    err = ""
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            r = SingleRun(indat)
            r.run()
    except Exception as e:  # recorded, never hidden
        err = f"{type(e).__name__}: {e}"[:300]
    cpu = time.process_time() - t0; wall = time.time() - w0
    mf = os.path.join(workdir, f"{tag}_MFILE.DAT")
    res = parse_mfile(mf) if os.path.exists(mf) else {k: float("nan") for k in OUTPUTS}
    itv = {}
    if os.path.exists(mf):
        itv = parse_mfile(mf, ITERVARS + ["rmajor"])
        if keep_mfile_gz:
            with open(mf, "rb") as f, gzip.open(keep_mfile_gz, "wb") as g:
                shutil.copyfileobj(f, g)
    for f in os.listdir(workdir):
        if f.startswith(tag + "_") or f.startswith(tag + "process"):
            os.remove(os.path.join(workdir, f))
    res.update(cpu_s=cpu, wall_s=wall, error=err)
    return res, itv


if __name__ == "__main__":
    # One run from the command line, e.g. the smoke test of DATA.md ("Regenerating"):
    #   python process_driver.py <template IN.DAT> <out.json> hfact psepbqar_max core_radius f_w kappa f_he
    #       [warm.json]
    import json
    if importlib.util.find_spec("process") is None:
        print("PROCESS is not installed. Install PROCESS v3.4.2 (https://github.com/ukaea/PROCESS, tag v3.4.2; "
              "the recipe is in DATA.md, \"Regenerating\") and rerun process_driver.py.", file=sys.stderr)
        sys.exit(3)
    tpl_path, out_json = sys.argv[1], sys.argv[2]
    x = [float(v) for v in sys.argv[3:9]]
    warm, r0 = None, 9.0
    if len(sys.argv) > 9:
        b = json.load(open(sys.argv[9]))
        warm = {k: v for k, v in b["itv"].items() if k != "rmajor"}
        r0 = b["itv"]["rmajor"]
    res, _ = run_one(open(tpl_path, encoding="utf-8").read(), x, os.getcwd(), "point",
                     rmajor0=r0, maxcal=50, warm=warm)
    json.dump(res, open(out_json, "w"))
