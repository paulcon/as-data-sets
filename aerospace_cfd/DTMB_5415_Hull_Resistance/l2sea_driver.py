"""Run the L2-Sea potential-flow code once and read its outputs.

L2-Sea (A. Serani and M. Diez, CNR-INM; GPL-3.0) is a compiled Fortran program. It is
an outside tool here: this module only writes its two input files, starts the program,
and reads its output files. No L2-Sea code is copied. Build instructions are in DATA.md.

The mapping of the 16 inputs follows the UM-Bridge 'forward' model
(github.com/UM-Bridge/benchmarks, models/l2-sea/umbridge-server.py):

    x[0]  Froude number Fr            -> &FREE_WARP fr
    x[1]  draft T (full scale, m)     -> &PANCA_PARAMETERS ztrasla
    x[2:] shape coefficients s1..s14  -> variables.inp, one per line
    fidelity (1 finest .. 7 coarsest) -> &MAIN_PARAMETERS igrid
    sinkoff, trimoff ('y' or 'n')     -> &FREE_WARP sinkoff, trimoff

Every other setting is that of the example case examples/DTMB-5415 shipped with the code.
Each call works in a fresh temporary copy of the case, so calls are independent. Unlike the
UM-Bridge server, which returns zeros when the program fails, a failure raises RuntimeError.

Only the standard library is imported.
"""
import os
import re
import shutil
import subprocess
import tempfile

CASE_FILES = ("5415_90_25_fs.x", "SBDF.nml", "alfak.out", "eigenvecto.out")

#: Names of the values `run` returns, in order.
OUTPUT_NAMES = ("R_T", "g1", "g2", "g3", "g4", "R_w", "R_f", "S_wet", "displacement", "cpu_s")
#: Extra diagnostics `run` also returns (force-dyn.out): the wave-cut wave resistance and the
#: code's ITTC-1957 frictional resistance, both N. They are used by `failure_check`.
DIAG_NAMES = ("Rw_wc", "Rf_ittc")


#: Drafts T (m) at which runs fail silently, by fidelity (DATA.md, Accuracy). G7: failures at the
#: parent hull for T in [-5.655, -5.61] (Fr 0.28) and [-5.62, -5.61] (Fr 0.36), and the 7 flagged
#: rows of the design at T in [-5.676, -5.618]; G6: failures for T in [-5.63, -5.61] and design
#: row 305 (T -5.623). Widened by a margin. Not mapped for G1-G5. `failure_reason` reports every
#: run with T in the band as failed, whatever numbers it returned.
BAD_T = {7: (-5.70, -5.60), 6: (-5.65, -5.60)}
#: Friction check: R_f / Rf_ittc (both as the code prints them) of valid runs follows
#: FRICTION_LINE[0] + FRICTION_LINE[1] * Fr to within FRICTION_TOL. Fitted to the 34 valid G7
#: reruns of design_diagnostics.csv (residuals -0.0034 .. 0.0070); the shape study stays within
#: 0.0071, the flagged design rows are 0.0216 or more away (check_accuracy.py).
FRICTION_LINE = (1.0160, 0.0635)
FRICTION_TOL = 0.015
#: Wave-cut check: valid runs have Rw_wc <= WAVECUT_MAX_RATIO * R_w (valid runs in the shipped
#: files: 0.27 .. 1.22, and 1.60 for one run inside the draft band; flagged design rows 4.8 and
#: more; check_accuracy.py).
WAVECUT_MAX_RATIO = 2.0


def failure_reason(x, out, fidelity, band=True):
    """Return None if a run passes the per-run checks, else a short reason it is not trusted.

    x is the 16-vector of inputs (only Fr and T are used), out a dict with at least R_w, R_f,
    Rw_wc and Rf_ittc (as `run` returns it). With band=False the draft-band test is skipped, so
    only the returned numbers are judged. The checks:
    the draft band BAD_T of the fidelity; friction R_f / Rf_ittc off the line FRICTION_LINE by more
    than FRICTION_TOL; the wave-cut wave resistance larger than WAVECUT_MAX_RATIO times R_w or
    negative resistances. They were calibrated on G7 and checked on G6 only; see DATA.md.
    generate_data.py applies it to no run: the shipped values are stored as the code printed
    them, and the notebook applies the rule when it reads them.
    """
    fidelity = int(fidelity)
    if band and fidelity in BAD_T:
        lo, hi = BAD_T[fidelity]
        if lo <= float(x[1]) <= hi:
            return "draft T = %.4f m lies in the band [%g, %g] m of silent failures on G%d" % (
                float(x[1]), lo, hi, fidelity)
    if min(out["R_w"], out["R_f"], out["Rw_wc"], out["Rf_ittc"]) < 0:
        return "negative resistance component"
    fr = float(x[0])
    ratio = out["R_f"] / out["Rf_ittc"]
    if abs(ratio - FRICTION_LINE[0] - FRICTION_LINE[1] * fr) > FRICTION_TOL:
        return "friction R_f / R_f,ITTC = %.4f, expected %.4f +- %g" % (
            ratio, FRICTION_LINE[0] + FRICTION_LINE[1] * fr, FRICTION_TOL)
    if out["Rw_wc"] > WAVECUT_MAX_RATIO * out["R_w"]:
        return "wave-cut wave resistance %.4g N > %g x R_w = %.4g N" % (
            out["Rw_wc"], WAVECUT_MAX_RATIO, WAVECUT_MAX_RATIO * out["R_w"])
    return None


def _set_nml(text, group, key, value):
    """Set `key = value` inside the Fortran namelist group `&group ... /`."""
    m = re.search(r"&%s\b(.*?)\n/" % group, text, flags=re.S)
    if m is None:
        raise RuntimeError("namelist group %s not found" % group)
    body = m.group(1)
    pat = re.compile(r"(^\s*%s\s*=\s*)([^!\n]*?)(\s*(!.*)?$)" % key, flags=re.M)
    if not pat.search(body):
        raise RuntimeError("key %s not found in group %s" % (key, group))
    body = pat.sub(lambda mm: mm.group(1) + value + mm.group(3), body, count=1)
    return text[:m.start(1)] + body + text[m.end(1):]


def _floats(line):
    return [float(v) for v in line.split()]


def run(x, l2sea_home, fidelity=7, sinkoff="y", trimoff="y", timeout=1800, keep_dir=None):
    """Run L2-Sea at x = [Fr, T, s1, ..., s14]; return a dict keyed by OUTPUT_NAMES.

    R_T     total resistance, model scale, N (objective.out; = R_w + R_f to print rounding)
    g1..g4  geometric constraints (beam, draft, sonar dome x2), satisfied when <= 0
    R_w     wave resistance from hull-pressure integration, N (force-dyn.out Rw_pi)
    R_f     frictional resistance, flat-plate (local Reynolds number), N (force-dyn.out Rf)
    S_wet   wetted hull surface, model scale, m^2 (panca.out)
    displacement  displaced volume, model scale, m^3 (panca.out)
    cpu_s   the program's own time report, s (elapsed_CPUtime.out). It is not CPU time: the
            program reads SYSTEM_CLOCK (wall time) and divides the count by 10000, which under
            gfortran (1000 counts per second) gives a tenth of the wall time; see DATA.md
    Rw_wc, Rf_ittc  diagnostics: wave-cut wave resistance and ITTC-1957 friction, N
    """
    x = [float(v) for v in x]
    if len(x) != 16:
        raise ValueError("x must have 16 entries: Fr, T, s1..s14")
    exe = os.path.join(l2sea_home, "bin", "L2-Sea")
    case = os.path.join(l2sea_home, "examples", "DTMB-5415")
    tmp_root = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    work = tempfile.mkdtemp(prefix="l2sea_", dir=tmp_root)
    try:
        for f in CASE_FILES:
            shutil.copy(os.path.join(case, f), work)
        with open(os.path.join(work, "variables.inp"), "w") as fh:
            for v in x[2:]:
                fh.write(repr(v) + "\n")
        nml = os.path.join(work, "SBDF.nml")
        with open(nml) as fh:
            txt = fh.read()
        txt = _set_nml(txt, "MAIN_PARAMETERS", "igrid", str(int(fidelity)))
        txt = _set_nml(txt, "FREE_WARP", "fr", repr(x[0]))
        txt = _set_nml(txt, "FREE_WARP", "sinkoff", "'%s'" % sinkoff)
        txt = _set_nml(txt, "FREE_WARP", "trimoff", "'%s'" % trimoff)
        txt = _set_nml(txt, "PANCA_PARAMETERS", "ztrasla", repr(x[1]))
        with open(nml, "w") as fh:
            fh.write(txt)
        env = dict(os.environ, OMP_NUM_THREADS="1")
        proc = subprocess.run("ulimit -s unlimited; exec '%s'" % exe, shell=True, cwd=work,
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
        d = os.path.join(work, "CPU000")
        obj = os.path.join(d, "objective.out")
        if proc.returncode != 0 or not os.path.exists(obj):
            raise RuntimeError("L2-Sea failed (exit %d): %s" % (
                proc.returncode, proc.stdout.decode(errors="replace")[-400:]))
        with open(obj) as fh:
            lines = fh.read().splitlines()
        o = _floats(lines[1])
        if int(o[0]) != int(fidelity):
            raise RuntimeError("objective.out reports grid %d, expected %d" % (o[0], fidelity))
        with open(os.path.join(d, "force-dyn.out")) as fh:
            fd = _floats(fh.read().splitlines()[1])   # Fr Rw_pi Rw_wc Rh Rf Rf_ittc Rt_pi ...
        with open(os.path.join(d, "panca.out")) as fh:
            pan = fh.read()
        disp = float(re.search(r"Displacement \(m\^3\)\s*=\s*(\S+)", pan).group(1))
        surf = float(re.search(r"Hull surface \(m\^2\)\s*=\s*(\S+)", pan).group(1))
        with open(os.path.join(d, "elapsed_CPUtime.out")) as fh:
            cpu = float(re.search(r"=\s*(\S+)", fh.read()).group(1))
        out = dict(zip(OUTPUT_NAMES, o[-5:] + [fd[1], fd[4], surf, disp, cpu]))
        out.update(zip(DIAG_NAMES, [fd[2], fd[5]]))
        bad = [k for k, v in out.items() if not (v == v and abs(v) < 1e30)]
        if bad or out["R_T"] <= 0:
            raise RuntimeError("L2-Sea returned invalid output %r" % out)
        # force-dyn.out also has a hydrostatic term Rh; it is not part of R_T and is not
        # stored, because identical runs gave Rh = 0 or 2.5e-3 N (not reproducible).
        # (R_T and R_w + R_f differ by up to a few 1e-6 relative in the printed digits.)
        if abs(fd[1] + fd[4] - out["R_T"]) > 1e-4 * out["R_T"]:
            raise RuntimeError("R_w + R_f != R_T: %r" % out)
        return out
    finally:
        if keep_dir:
            shutil.copytree(work, keep_dir, dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
