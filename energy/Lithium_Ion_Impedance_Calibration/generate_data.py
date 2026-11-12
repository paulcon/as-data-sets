"""Generate the stored data: impedance misfits of the grouped SPMe and their gradients.

This script needs PyBaMM and PyBOP (the shipped files were written with Python 3.13,
PyBaMM 26.10.1.1, pybammsolvers 0.11.1, PyBOP 26.3, CasADi 3.8.1, numpy 2.3.5, scipy 1.18.1;
install recipe in DATA.md); the notebook does not. Without PyBaMM or PyBOP the script stops
with an error that says what to install; it also stops if other versions are installed,
unless --any-version is given.

    python generate_data.py --rows 0:400   --partial /tmp/eis_partial    # resumable chunks
    python generate_data.py --rows 400:800 --partial /tmp/eis_partial
    python generate_data.py --rows 800:1000 --partial /tmp/eis_partial
    python generate_data.py --assemble     --partial /tmp/eis_partial    # writes the CSVs here
    python generate_data.py --check 0,517,999                            # recompute rows, compare
    python generate_data.py --reference --jacobian                       # small files at the truth
    python generate_data.py --gradcheck 3,400,871                        # adjoint vs central FD
    python generate_data.py --meshcheck 3,400,871                        # paper mesh vs 2x mesh
    python generate_data.py --checkpoints                                # Table 3 best fit, H directions

Design: N = 1000 independent Monte Carlo samples, uniform on the optimisation bounds of
Table 3 (p. 12) in physical units, numpy default_rng(SEED). Each sample gives the misfit
Phi (Eq. 7) and its exact gradient (adjoint, see spme_eis.py) against synthetic reference
spectra at the true parameters of Table 3, plus the nine per-SOC terms Phi_m and their
gradients. inputs.csv is the source of truth for the inputs (written with %.17g, checked
to round-trip exactly). Rows already in --partial are skipped (resume). One process.
--rows and --assemble need --partial, a directory outside this one. --check, --gradcheck and
--meshcheck print and write nothing; --reference, --jacobian, --assemble and --checkpoints
rewrite shipped files in this directory.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spme_eis_driver as se  # noqa: E402

SEED = 20261101
VERSIONS = {"pybamm": "26.10.1.1", "pybop": "26.3", "casadi": "3.8.1"}
INSTALL = 'pip install "pybop==26.3" "pybamm==26.10.1.1" "pybammsolvers==0.11.1" "casadi==3.8.1"'


def require_packages(any_version=False):
    """Stop with an install message (exit status 3) unless PyBaMM, PyBOP and CasADi of the
    versions that wrote the shipped files are importable."""
    have = {}
    for name in VERSIONS:
        try:
            have[name] = __import__(name).__version__
        except ImportError:
            print(f"{name} is not installed. This script needs PyBaMM 26.10.1.1, pybammsolvers 0.11.1, "
                  f"PyBOP 26.3 and CasADi 3.8.1 (Python 3.13: {INSTALL}; recipe in DATA.md). "
                  "Install them and rerun this script.", file=sys.stderr)
            sys.exit(3)
    wrong = {k: v for k, v in have.items() if v != VERSIONS[k]}
    if wrong and not any_version:
        print(f"installed {wrong}; the shipped data use {VERSIONS} ({INSTALL}). "
              "Install those versions and rerun this script, or pass --any-version.", file=sys.stderr)
        sys.exit(3)
N = 1000
FMT = "%.17g"


def design(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    return se.LOWER + (se.UPPER - se.LOWER) * rng.random((n, se.M_INPUTS))


def read_csv(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=",", comments="#", ndmin=2)


def inputs():
    path = os.path.join(HERE, "inputs.csv")
    X = design()
    if os.path.exists(path):
        Xf = read_csv("inputs.csv")
        if not np.allclose(Xf, X, rtol=1e-15, atol=0):
            raise RuntimeError("inputs.csv differs from the seeded design")
        return Xf
    return X


def run_rows(rows, partial):
    os.makedirs(partial, exist_ok=True)
    X = inputs()
    t0 = time.time()
    mf = se.ImpedanceMisfit()
    print(f"set-up {time.time() - t0:.1f} s; states {mf.eis.n_states}; {mf.eis.versions}",
          flush=True)
    for i in rows:
        out = os.path.join(partial, f"row{i:04d}.npz")
        if os.path.exists(out):
            continue
        t, c = time.time(), time.process_time()
        phi, phim, g, gm = mf.misfit(X[i], gradient=True)
        dt, dc = time.time() - t, time.process_time() - c
        np.savez(out, x=X[i], phi=phi, phim=phim, g=g, gm=gm, seconds=dt, cpu_seconds=dc)
        print(f"row {i}: Phi = {phi:.6e} Ohm^2, {dt:.2f} s wall, {dc:.2f} s CPU", flush=True)
    print(f"process CPU time {time.process_time():.0f} s", flush=True)


def header_cols():
    socs = [f"{int(round(100 * s))}" for s in se.SOCS]
    return socs


def assemble(partial):
    X = inputs()
    rows = [np.load(os.path.join(partial, f"row{i:04d}.npz")) for i in range(N)]
    for i, r in enumerate(rows):
        if not np.array_equal(r["x"], X[i]):
            raise RuntimeError(f"row {i}: inputs differ")
    phi = np.array([r["phi"] for r in rows])
    phim = np.array([r["phim"] for r in rows])
    g = np.array([r["g"] for r in rows])
    gm = np.array([r["gm"] for r in rows])          # (N, 9, 18)
    secs = np.array([r["seconds"] for r in rows])
    cpu = np.array([r["cpu_seconds"] if "cpu_seconds" in r.files else np.nan for r in rows])
    units = [p[3] for p in se.PARAMETERS]
    names = [f"{n} [{u}]" for n, u in zip(se.NAMES, units)]
    np.savetxt(os.path.join(HERE, "inputs.csv"), X, fmt=FMT, delimiter=",",
               header=f"{N} Monte Carlo samples (numpy default_rng({SEED})), uniform on the "
                      "Table 3 optimisation bounds of the 18 grouped SPMe parameters, physical units\n"
                      + ",".join(names))
    if not np.array_equal(read_csv("inputs.csv"), X):
        raise RuntimeError("inputs.csv did not round-trip exactly")
    socs = header_cols()
    np.savetxt(os.path.join(HERE, "outputs.csv"), np.column_stack([phi, phim]), fmt=FMT,
               delimiter=",",
               header="Impedance misfit Phi = sum over 9 SOCs and 60 frequencies of |Z_ref - Z(theta)|^2 "
                      "(Eq. 7), then the nine per-SOC terms Phi_m; Ohm^2\n"
                      + ",".join(["Phi [Ohm^2]"] + [f"Phi_SOC{s} [Ohm^2]" for s in socs]))
    np.savetxt(os.path.join(HERE, "gradients.csv"), g, fmt=FMT, delimiter=",",
               header="Exact (adjoint) gradient of Phi w.r.t. the 18 physical inputs; "
                      "Ohm^2 per input unit\n" + ",".join(f"dPhi/d{n} [Ohm^2/{u}]"
                                                         for n, u in zip(se.NAMES, units)))
    cols = [f"dPhi_SOC{s}/d{n}" for s in socs for n in se.NAMES]
    np.savetxt(os.path.join(HERE, "gradients_soc.csv"), gm.reshape(N, -1), fmt=FMT,
               delimiter=",",
               header="Exact gradients of the nine per-SOC terms Phi_m (SOC-major: 18 columns per "
                      "SOC, inputs in the order of inputs.csv); Ohm^2 per input unit\n" + ",".join(cols))
    print(f"assembled {N} rows; wall time per sample: median {np.median(secs):.2f} s, "
          f"total {secs.sum():.0f} s; CPU time per sample (rows with a record, "
          f"{np.isfinite(cpu).sum()}): median {np.nanmedian(cpu):.2f} s, total {np.nansum(cpu):.0f} s")


def write_reference():
    eis = se.SPMeImpedance()
    th = se.true_theta()
    Z = np.array([eis.impedance(th, s) for s in se.SOCS])      # (9, 60)
    cols = ["frequency [Hz]"]
    data = [se.FREQUENCIES]
    for s, z in zip(header_cols(), Z):
        cols += [f"ReZ_SOC{s} [Ohm]", f"ImZ_SOC{s} [Ohm]"]
        data += [z.real, z.imag]
    np.savetxt(os.path.join(HERE, "reference_impedance.csv"), np.column_stack(data), fmt=FMT,
               delimiter=",",
               header="Synthetic reference impedance Z_m(w_k) of the grouped SPMe at the true "
                      "parameters (Table 3, R0 = 0.01 Ohm), 9 SOCs x 60 frequencies\n" + ",".join(cols))
    np.savetxt(os.path.join(HERE, "true_parameters.csv"), th[None, :], fmt=FMT, delimiter=",",
               header="True grouped parameters (Chen2020 via PyBOP GroupedSPMe, R0 = 0.01 Ohm; "
                      "Table 3 'theta'), physical units\n"
                      + ",".join(f"{n} [{p[3]}]" for n, p in zip(se.NAMES, se.PARAMETERS)))


def check(rows):
    X = read_csv("inputs.csv")
    f = read_csv("outputs.csv")
    g = read_csv("gradients.csv")
    gm = read_csv("gradients_soc.csv")
    zref = read_csv("reference_impedance.csv")
    mf = se.ImpedanceMisfit()
    zr = np.column_stack([zref[:, 1::2] + 1j * zref[:, 2::2]]).T
    print("reference spectra: max rel. diff", np.max(np.abs(mf.Zref - zr) / np.abs(zr)))
    for i in rows:
        t = time.time()
        phi, phim, gi, gmi = mf.misfit(X[i], gradient=True)
        dt = time.time() - t
        rel = lambda a, b: np.max(np.abs(a - b)) / np.max(np.abs(b))
        print(f"row {i}: Phi rel {rel(np.r_[phi, phim], f[i]):.2e}, grad rel {rel(gi, g[i]):.2e}, "
              f"per-SOC grad rel {rel(gmi.ravel(), gm[i]):.2e} ({dt:.1f} s)")


def jacobian_fd(steps=(1e-5, 1e-3)):
    """dZ_m(w_k)/dtheta at the true parameters by central differences (step h*(upper-lower)),
    one (9, 60, 18) complex array per step. Writes nothing."""
    eis = se.SPMeImpedance()
    th = se.true_theta()
    Js = []
    for h in steps:
        J = np.empty((len(se.SOCS), len(se.FREQUENCIES), se.M_INPUTS), complex)
        for j in range(se.M_INPUTS):
            d = h * (se.UPPER[j] - se.LOWER[j])
            e = np.zeros(se.M_INPUTS)
            e[j] = d
            for m, s in enumerate(se.SOCS):
                J[m, :, j] = (eis.impedance(th + e, s) - eis.impedance(th - e, s)) / (2 * d)
        Js.append(J)
    return Js


def jacobian(steps=(1e-5, 1e-3)):
    """Write impedance_jacobian.csv from the first step of jacobian_fd and print the largest
    difference from the second step (relative to the largest entry, in normalized coordinates)."""
    half = (se.UPPER - se.LOWER) / 2
    Js = jacobian_fd(steps)
    a, b = Js[0] * half, Js[1] * half
    print(f"impedance Jacobian at theta_true: steps {steps}, max |difference| / max |entry| "
          f"(normalized coordinates) = {np.max(np.abs(a - b)) / np.max(np.abs(a)):.2e}")
    J = Js[0].reshape(-1, se.M_INPUTS)
    socs = np.repeat(100 * se.SOCS, len(se.FREQUENCIES))
    fr = np.tile(se.FREQUENCIES, len(se.SOCS))
    units = [p[3] for p in se.PARAMETERS]
    cols = (["SOC [%]", "frequency [Hz]"] + [f"dReZ/d{n} [Ohm/{u}]" for n, u in zip(se.NAMES, units)]
            + [f"dImZ/d{n} [Ohm/{u}]" for n, u in zip(se.NAMES, units)])
    np.savetxt(os.path.join(HERE, "impedance_jacobian.csv"),
               np.column_stack([socs, fr, J.real, J.imag]), fmt=FMT, delimiter=",",
               header="Sensitivity of the model impedance to the 18 physical inputs at the true "
                      f"parameters (central differences, step {steps[0]:g} x range), 9 SOCs x 60 "
                      "frequencies (SOC-major)\n" + ",".join(cols))


def gradcheck(rows, steps=(1e-4, 1e-6)):
    """Adjoint gradient vs central finite differences of Phi, step h * (upper - lower)."""
    X = read_csv("inputs.csv")
    g = read_csv("gradients.csv")
    mf = se.ImpedanceMisfit()
    half = (se.UPPER - se.LOWER) / 2
    for i in rows:
        for h in steps:
            fd = np.empty(se.M_INPUTS)
            for j in range(se.M_INPUTS):
                d = h * (se.UPPER[j] - se.LOWER[j])
                e = np.zeros(se.M_INPUTS)
                e[j] = d
                fd[j] = (mf.misfit(X[i] + e, check=False)[0]
                         - mf.misfit(X[i] - e, check=False)[0]) / (2 * d)
            err = np.max(np.abs(fd - g[i]) * half) / np.max(np.abs(g[i]) * half)
            print(f"row {i}, h = {h:g}: max |FD - adjoint| / max |adjoint| "
                  f"(normalized coordinates) = {err:.2e}")


def meshcheck(rows, factor=2):
    """Phi on the paper's mesh vs a mesh refined by `factor` in every domain."""
    X = read_csv("inputs.csv")
    f = read_csv("outputs.csv")
    fine = {k: factor * v for k, v in se.MESH.items()}
    mf = se.ImpedanceMisfit(mesh=fine)
    print(f"refined mesh {fine}: {mf.eis.n_states} states")
    for i in rows:
        t = time.time()
        phi = mf.misfit(X[i])[0]
        print(f"row {i}: Phi paper mesh {f[i, 0]:.6e}, refined {phi:.6e}, log10 difference "
              f"{np.log10(phi) - np.log10(f[i, 0]):+.2e} ({time.time() - t:.0f} s)")


# Table 3, p. 12: the paper's best impedance fit theta-hat (lowest cost of ten PSO runs), as printed
TABLE3_HAT = np.array([6682, 1020, 200.0, 403.5, 200.0, 0.500, 1.500, 500.0, 5158, 27746, 0.6178,
                       0.6953, 0.8548, 0.02634, 0.2642, 0.9092, 0.2647, 0.0101])


def checkpoints():
    """Spectra at six extra parameter points (one model run each, no gradients).

    case 0: the Table 3 best fit theta-hat, as printed (for the fitting errors, Eq. 17);
    case 1: theta-hat with R0 = 0.01005 Ohm, the lower end of its print-rounding interval;
    cases 2-5: theta_true + t * half * v, t = +1e-3, -1e-3, +1e-2, -1e-2 (normalized
    coordinates), v the weakest eigenvector of the Gauss-Newton matrix H = Re(J^H J) built
    from impedance_jacobian.csv (largest-magnitude entry positive): how far Phi is quadratic.
    Writes check_parameters.csv (6 x 18) and check_spectra.csv (6 x 540 rows, long format)."""
    th = se.true_theta()
    half = (se.UPPER - se.LOWER) / 2
    JZ = read_csv("impedance_jacobian.csv")
    Jc = (JZ[:, 2:20] + 1j * JZ[:, 20:38]) * half
    hev, hW = np.linalg.eigh(np.real(Jc.conj().T @ Jc))
    v = hW[:, 0] * np.sign(hW[np.argmax(np.abs(hW[:, 0])), 0])
    print(f"smallest eigenvalue of H {hev[0]:.3e}; v = " + ", ".join(f"{x:+.3f}" for x in v))
    hat_lo = TABLE3_HAT.copy()
    hat_lo[17] = 0.01005
    cases = [TABLE3_HAT, hat_lo] + [th + t * half * v for t in (1e-3, -1e-3, 1e-2, -1e-2)]
    labels = ["Table 3 theta-hat", "theta-hat, R0 = 0.01005", "truth + 1e-3 v_min(H)",
              "truth - 1e-3 v_min(H)", "truth + 1e-2 v_min(H)", "truth - 1e-2 v_min(H)"]
    eis = se.SPMeImpedance()
    zref = read_csv("reference_impedance.csv")
    Zr = (zref[:, 1::2] + 1j * zref[:, 2::2]).T                      # (9, 60)
    rows = []
    for c, (theta, lab) in enumerate(zip(cases, labels)):
        t = time.time()
        Z = np.array([eis.impedance(theta, s) for s in se.SOCS])   # (9, 60)
        phi = np.sum(np.abs(Z - Zr) ** 2)
        fe = 100 * np.mean(np.abs(Z - Zr) / np.abs(Zr), axis=1)
        print(f"case {c} ({lab}): Phi = {phi:.4e} Ohm^2; FE_m [%] = "
              + " ".join(f"{x:.2f}" for x in fe) + f" ({time.time() - t:.1f} s)", flush=True)
        for m, s in enumerate(se.SOCS):
            rows.append(np.column_stack([np.full(len(se.FREQUENCIES), c),
                                         np.full(len(se.FREQUENCIES), 100 * s),
                                         se.FREQUENCIES, Z[m].real, Z[m].imag]))
    units = [p[3] for p in se.PARAMETERS]
    np.savetxt(os.path.join(HERE, "check_parameters.csv"),
               np.column_stack([np.arange(len(cases)), np.array(cases)]), fmt=FMT, delimiter=",",
               header="Extra parameter points: 0 Table 3 theta-hat (printed); 1 theta-hat with "
                      "R0 = 0.01005 Ohm; 2-5 theta_true + t*half-range*v_min(H), t = +1e-3, -1e-3, "
                      "+1e-2, -1e-2; physical units\n" + ",".join(
                          ["case"] + [f"{n} [{u}]" for n, u in zip(se.NAMES, units)]))
    np.savetxt(os.path.join(HERE, "check_spectra.csv"), np.vstack(rows), fmt=FMT, delimiter=",",
               header="Model impedance at the points of check_parameters.csv, 9 SOCs x 60 "
                      "frequencies per case (case-major, then SOC-major)\n"
                      "case,SOC [%],frequency [Hz],ReZ [Ohm],ImZ [Ohm]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", help="a:b, rows to run into --partial")
    ap.add_argument("--partial", help="directory for the per-row files of --rows and --assemble "
                                      "(required with them; outside this directory)")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--reference", action="store_true",
                    help="write reference_impedance.csv and true_parameters.csv")
    ap.add_argument("--check", help="comma-separated rows to recompute and compare")
    ap.add_argument("--jacobian", action="store_true",
                    help="write impedance_jacobian.csv (dZ/dtheta at the true parameters)")
    ap.add_argument("--gradcheck", help="comma-separated rows: adjoint vs central differences")
    ap.add_argument("--meshcheck", help="comma-separated rows: paper mesh vs 2x refined mesh")
    ap.add_argument("--checkpoints", action="store_true",
                    help="write check_parameters.csv and check_spectra.csv (Table 3 fit, H directions)")
    ap.add_argument("--any-version", action="store_true",
                    help="run with PyBaMM/PyBOP/CasADi versions other than those of the shipped files")
    a = ap.parse_args()
    if (a.rows or a.assemble) and (a.partial is None
                                   or os.path.abspath(a.partial).startswith(HERE + os.sep)
                                   or os.path.abspath(a.partial) == HERE):
        ap.error("--rows and --assemble need --partial DIR outside this directory")
    require_packages(a.any_version)
    if a.reference:
        write_reference()
    if a.rows:
        lo, hi = (int(v) for v in a.rows.split(":"))
        run_rows(range(lo, hi), a.partial)
    if a.assemble:
        assemble(a.partial)
    if a.check:
        check([int(v) for v in a.check.split(",")])
    if a.jacobian:
        jacobian()
    if a.gradcheck:
        gradcheck([int(v) for v in a.gradcheck.split(",")])
    if a.meshcheck:
        meshcheck([int(v) for v in a.meshcheck.split(",")])
    if a.checkpoints:
        checkpoints()
