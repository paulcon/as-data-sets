"""Accuracy checks for Hector_Matilda_Warming.

    python check_accuracy.py            # parts 1-3 and 5 (needs R, hector 3.1.1, matilda v1.0.0), about 11 min wall (measured)
    python check_accuracy.py --paper    # also part 4 (same needs), about 27 min more (measured; 2000 Hector runs)
    python check_accuracy.py --weights  # part 5 only: numpy only, reads the shipped files, a few seconds

Prints
  1. Matilda path: warming from matilda::iterate_model (run to Hector's end year 2300) and
     the 1850-1900 normalisation, against outputs.csv, at the first 5 samples.
  2. Step size: central differences with steps 0.125 and 0.5 against the shipped step 0.25,
     at the first 10 gradient rows (relative difference of each gradient vector); then steps
     0.1 and 0.5 at rows 376, 249 and 273, and the NPP0 component of the CO2 gradient at row 249.
  3. Roughness: f on 41 points spanning +-0.5 in z_j (spacing 0.025) along each z_j at rows 376,
     249 and 273: the largest deviation from a quadratic fit, the largest single-step anomaly
     (a step's change minus the quadratic fit's change over the same step), and the slope of the
     quadratic fit at the sample, compared with the shipped gradient.
  4. (--paper) The paper's own parameter sets: set.seed(245); generate_params(SSP3-7.0 core, 1000)
     as in 01_running_matilda/02_running_matilda_case_study.R of github.com/JGCRI/Brown-etal_2024-PLOSClimate
     (commit c9a9b49). (a) Run under SSP3-7.0 with iterate_model: the sets that fail. (b) Run under
     SSP2-4.5, weighted with score_runs(criterion_gmst_obs(), score_bayesian) on the SSP2-4.5 series,
     and the weighted probability of 1.7 < metric <= 3.8 degC for three metrics: raw Hector gmst
     2081-2100 mean (what metric_calc computes in that code), that minus the ensemble mean of gmst
     1850-1900 (that code's normalize_temperature), and minus each run's own 1850-1900 mean.
     (c) The raw-metric probability again with the misalignment of part 5, using the sets that
     failed in (a).
  5. (--weights, or with the other parts) The likely cause of the paper's 92 %: emulation of the
     authors' Fig. 6B pipeline on the shipped 999 finite runs. In that pipeline the sets that fail
     under SSP3-7.0 (114 and 930 with the paper's draws) get weight NA; na.omit() drops them;
     metric_calc() then renumbers the remaining runs 1..n (R/metric_calc.R, run_number <- 1:nrow(df)),
     and merge(metrics, weights, by = "run_number") pairs each run from 115 on with the weight of a
     neighbouring run. The emulation numbers the 999 finite runs 1..999, removes runs 114 and 930,
     and pairs metrics and weights the same way. Printed: P(1.7 < T <= 3.8) and the 0.5 degC bins of
     Fig. 6B (raw-GMST metric) for correct weights, no weights and misaligned weights, beside the
     SSP2-4.5 bar of the published Fig. 6B (values read from the figure, about +-0.01). Also the
     probability with the score_bayesian likelihood of Matilda before 2024-02-07, exp(-rmse^2/2).
Writes nothing.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hector_driver as D  # noqa: E402

H, SCEN = 0.25, "ssp245"
NEED = ("R with the packages hector 3.1.1 (git tag v3.1.1 of github.com/JGCRI/hector) and "
        "matilda v1.0.0 (git tag v1.0.0 of github.com/JGCRI/matilda); see DATA.md, Regenerating. "
        "Install them and rerun check_accuracy.py.")
ROWS_WIDE = (376, 249, 273)        # rows of gradients.csv for the step and roughness checks
FAIL_PAPER = (114, 930)            # paper's sets (1-based) that fail under SSP3-7.0, part 4(a)
BINS = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, np.inf])
# SSP2-4.5 bar of the published Fig. 6B (p. 19), read from the figure (about +-0.01), bins (1,1.5] .. (4,4.5], >4.5
FIG6B_SSP245 = (0.02, 0.185, 0.39, 0.275, 0.095, 0.03, 0.005, 0.0)

R_PAPER = r"""
suppressPackageStartupMessages({library(hector); library(matilda)})
a <- commandArgs(TRUE)
set.seed(245)
p370 <- newcore(system.file("input/hector_ssp370.ini", package = "hector"), suppresslogging = TRUE)
params <- generate_params(p370, 1000)
r370 <- suppressMessages(iterate_model(p370, params, save_vars = GMST(), save_years = 2081:2100))
fail370 <- sapply(split(r370$value, r370$run_number), function(v) any(is.na(v)))
core <- newcore(system.file("input/hector_ssp245.ini", package = "hector"), suppresslogging = TRUE)
res <- suppressMessages(iterate_model(core, params, save_vars = GMST(), save_years = 1850:2100))
w <- score_runs(res, criterion_gmst_obs(), score_bayesian)
sp <- split(res, res$run_number)
raw <- sapply(sp, function(d) mean(d$value[d$year >= 2081]))
ref <- sapply(sp, function(d) mean(d$value[d$year <= 1900]))
out <- data.frame(raw = raw, ref = ref, weight = w$weights[match(as.integer(names(sp)), w$run_number)],
                  fail370 = as.integer(fail370[match(names(sp), names(fail370))]),
                  npp0 = params$NPP_FLUX0)
write.csv(format(out, digits = 17), a[1], row.names = FALSE, quote = FALSE)
cat(nrow(params), "parameter sets;", sum(is.na(raw)), "failed under SSP2-4.5;", sum(fail370),
    "failed under SSP3-7.0 (sets", paste(which(fail370), collapse = ", "), ")\n")
"""

R_MATILDA = r"""
suppressPackageStartupMessages({library(hector); library(matilda)})
a <- commandArgs(TRUE)
p <- read.csv(a[1])
core <- newcore(system.file("input/hector_ssp245.ini", package = "hector"), suppresslogging = TRUE)
res <- suppressMessages(iterate_model(core, p, save_vars = GMST(), save_years = 1850:2100))
w <- sapply(split(res, res$run_number), function(d)
  mean(d$value[d$year >= 2081]) - mean(d$value[d$year <= 1900]))
writeLines(sprintf("%.17e", w), a[2])
"""


def load(fn):
    return np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)


def interval_and_bins(m, w):
    w = w / w.sum()
    p = np.sum(w * ((m > 1.7) & (m <= 3.8)))
    b = [np.sum(w[(m > BINS[i]) & (m <= BINS[i + 1])]) for i in range(len(BINS) - 1)]
    return p, np.array(b)


def misaligned(m, w, fail):
    """Pair metrics and weights as Fig_06B_plot_probability_bar.R does after na.omit().

    m, w: metric and weight of runs numbered 1..N (array order). fail: run numbers whose weight is NA.
    Returns the metrics and the weights merge(metric_calc(...), use_weights, by = "run_number") pairs."""
    runs = np.arange(1, len(m) + 1)
    keep = ~np.isin(runs, fail) & np.isfinite(m) & np.isfinite(w)
    mk = m[keep]                               # na.omit(matilda_weighted)
    new = np.arange(1, len(mk) + 1)            # metric_calc: run_number <- 1:nrow(df)
    wt = dict(zip(runs[keep], w[keep]))        # na.omit(use_weights), original run numbers
    has = np.array([r in wt for r in new])
    return mk[has], np.array([wt[r] for r in new[has]])


def paper_ensemble():
    with tempfile.TemporaryDirectory() as d:
        rs, pout = os.path.join(d, "p.R"), os.path.join(d, "p.csv")
        with open(rs, "w") as f:
            f.write(R_PAPER)
        r = subprocess.run([D.rscript(), rs, pout], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("paper ensemble failed: " + r.stderr[-400:])
        print("4. paper ensemble (seed 245, 1000 sets):", r.stdout.strip())
        raw, ref, w, f370, npp = np.genfromtxt(pout, delimiter=",", skip_header=1, unpack=True)
    fail = [int(k) for k in np.flatnonzero(f370 > 0) + 1]
    print("   (a) NPP0 of the sets failing under SSP3-7.0 [Pg C/yr]: "
          + ", ".join(f"set {k} {npp[k - 1]:.1f}" for k in fail)
          + f"; smallest NPP0 among the sets that run under SSP3-7.0: {npp[f370 == 0].min():.1f}")
    ok = np.isfinite(raw) & np.isfinite(w)
    rw, rf, wn = raw[ok], ref[ok], w[ok] / w[ok].sum()
    print("   (b) SSP2-4.5, weights from the SSP2-4.5 series")
    for name, m in [("raw gmst 2081-2100 mean", rw), ("minus ensemble-mean 1850-1900", rw - rf.mean()),
                    ("minus own 1850-1900 mean", rw - rf)]:
        pw = np.sum(wn * ((m > 1.7) & (m <= 3.8)))
        pu = np.mean((m > 1.7) & (m <= 3.8))
        o = np.argsort(m)
        cw = np.cumsum(wn[o])
        q = [m[o][np.searchsorted(cw, p)] for p in (0.05, 0.5, 0.95)]
        print(f"   {name:30s}: weighted P(1.7 < T <= 3.8) = {100 * pw:.1f} % (unweighted {100 * pu:.1f} %); "
              f"weighted 5-50-95 % {q[0]:.2f} {q[1]:.2f} {q[2]:.2f} degC")
    print(f"   ensemble mean of gmst 1850-1900: {rf.mean():.3f} degC (sd across runs {rf.std():.3f}); "
          f"effective sample size of the weights {1 / np.sum(wn**2):.0f}")
    mm, ww = misaligned(raw, w, fail)
    p, b = interval_and_bins(mm, ww)
    print(f"   (c) raw metric, weights misaligned as in part 5 with the failing sets {list(fail)}: "
          f"P(1.7 < T <= 3.8) = {100 * p:.1f} %; bins (1,1.5] .. (4,4.5] " + " ".join(f"{x:.3f}" for x in b[:7]))


def weights_misalignment():
    OUT = load("outputs.csv")
    raw = OUT[:, 0] + OUT[:, 2]                 # raw Hector gmst 2081-2100 mean (metric_calc's metric)
    ok = np.isfinite(raw) & np.isfinite(OUT[:, 4])
    m, w = raw[ok], OUT[ok, 4]
    print(f"5. Fig. 6B pipeline emulated on the {ok.sum()} finite runs (numbered 1..{ok.sum()}), SSP2-4.5, raw-GMST metric")
    labels = "(1,1.5] (1.5,2] (2,2.5] (2.5,3] (3,3.5] (3.5,4] (4,4.5]"
    print(f"   {'weights':34s} P(1.7<T<=3.8)  bins {labels}")
    cases = [("correct (Matilda v1.0.0)", m, w), ("none (unweighted)", m, np.ones_like(m)),
             (f"misaligned (runs {FAIL_PAPER[0]}, {FAIL_PAPER[1]} NA)",) + misaligned(m, w, FAIL_PAPER)]
    for name, mm, ww in cases:
        p, b = interval_and_bins(mm, ww)
        print(f"   {name:34s} {100 * p:5.1f} %        " + "  ".join(f"{x:.3f}" for x in b[:7]))
    print(f"   {'published Fig. 6B (read, +-0.01)':34s}   92 % (text)   " + "  ".join(f"{x:.3f}" for x in FIG6B_SSP245[:7]))
    # alternative candidate: score_bayesian before matilda commit 2024-02-07, likelihood exp(-rmse^2 / 2) with the
    # stored rmse (RMSE / sd(obs)), i.e. without the division by sd(RMSE)
    r = OUT[ok, 3]
    wold = np.exp(-0.5 * r**2)
    wold /= wold.sum()
    own = OUT[ok, 0]
    print(f"   pre-2024-02-07 score_bayesian, exp(-rmse^2/2): P(1.7 < T <= 3.8) = "
          f"{100 * np.sum(wold * ((own > 1.7) & (own <= 3.8))):.1f} % (per-run baseline), "
          f"{100 * interval_and_bins(m, wold)[0]:.1f} % (raw metric); effective sample size {1 / np.sum(wold**2):.0f} "
          f"(current weights {1 / np.sum((w / w.sum())**2):.0f})")
    for name, mm, ww in cases:
        p, b = interval_and_bins(mm, ww)
        print(f"   max |bin - Fig. 6B| for {name:34s}: {np.max(np.abs(b[:7] - np.array(FIG6B_SSP245[:7]))):.3f}")


def main():
    if "--weights" in sys.argv[1:]:
        weights_misalignment()
        return
    try:
        D.tool_versions()
    except RuntimeError as e:
        sys.exit(f"{e}\nThis script needs {NEED}")
    IN, OUT, GR = load("inputs.csv"), load("outputs.csv"), load("gradients.csv")
    Z, X = IN[:, :6], IN[:, 6:]

    # 1. Matilda's own iterate_model path
    with tempfile.TemporaryDirectory() as d:
        pin, rs, pout = (os.path.join(d, n) for n in ("p.csv", "m.R", "w.txt"))
        with open(pin, "w") as f:
            f.write(",".join(D.NAMES) + "\n")
            for row in X[:5]:
                f.write(",".join(repr(float(v)) for v in row) + "\n")
        with open(rs, "w") as f:
            f.write(R_MATILDA)
        r = subprocess.run([D.rscript(), rs, pin, pout], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("matilda iterate_model failed: " + r.stderr[-400:])
        wm = np.loadtxt(pout)
    diff = np.max(np.abs(wm - OUT[:5, 0]))
    print(f"1. iterate_model (to 2300) vs outputs.csv warming, rows 0-4: max |diff| = {diff:.1e} degC")

    # 2. step-size check
    rows = slice(0, 10)
    G = {H: [GR[rows, :6], GR[rows, 6:]]}
    for h in (H / 2, 2 * H):
        F = D.run_z(D.fd_stencil(Z[rows], h), SCEN)
        G[h] = list(D.fd_gradients(F, h))[:2]
    print(f"2. step size: relative difference |g_h - g_{H}| / |g_{H}| over rows 0-9")
    for q, name in enumerate(D.OUTPUTS[:2]):
        for h in (H / 2, 2 * H):
            rel = np.linalg.norm(G[h][q] - G[H][q], axis=1) / np.linalg.norm(G[H][q], axis=1)
            print(f"   {name:9s} h = {h:5.3f}: median {np.median(rel):.3f}, max {np.max(rel):.3f}")
        e = np.abs(G[H / 2][q] - G[2 * H][q]) / np.linalg.norm(G[H][q], axis=1)[:, None]
        print(f"   {name:9s} largest component difference (h=0.125 vs 0.5) per input, "
              f"as a fraction of |g|: " + ", ".join(f"{D.PARAMS[j][1]} {e[:, j].max():.3f}" for j in range(6)))

    # 2b. steps 0.1 and 0.5 at the rows of part 3
    R3 = list(ROWS_WIDE)
    G3 = {}
    for h in (0.1, 0.5):
        F = D.run_z(D.fd_stencil(Z[R3], h), SCEN)
        G3[h] = list(D.fd_gradients(F, h))[:2]
    print(f"   rows {R3}: relative difference from the shipped gradient, per row")
    for q, name in enumerate(D.OUTPUTS[:2]):
        g0 = GR[R3, 6 * q:6 * q + 6]
        for h in (0.1, 0.5):
            rel = np.linalg.norm(G3[h][q] - g0, axis=1) / np.linalg.norm(g0, axis=1)
            print(f"   {name:9s} h = {h:5.3f}: " + " / ".join(f"{v:.3f}" for v in rel))

    # 3. roughness: 41 points on [-0.5, 0.5] (spacing 0.025) along each z_j at ROWS_WIDE
    d = np.linspace(-0.5, 0.5, 41)
    pts = [Z[i] + t * np.eye(6)[j] for i in R3 for j in range(6) for t in d]
    F = D.run_z(np.array(pts), SCEN).reshape(len(R3), 6, len(d), -1)
    print(f"3. roughness: 41 points spanning +-0.5 in z_j (spacing 0.025) at rows {R3}")
    print("   resid = max |f - quadratic fit|; step = largest |f(t+0.025) - f(t) - (q(t+0.025) - q(t))|;")
    print("   slope = derivative of the quadratic fit at the sample")
    j_npp = 4
    for q, name in enumerate(D.OUTPUTS[:2]):
        res = np.zeros((len(R3), 6))
        stp = np.zeros((len(R3), 6))
        rel = np.zeros((len(R3), 6))
        slope = np.zeros((len(R3), 6))
        for a, i in enumerate(R3):
            for j in range(6):
                y = F[a, j, :, q]
                c = np.polyfit(d, y, 2)
                fit = np.polyval(c, d)
                res[a, j] = np.max(np.abs(y - fit))
                stp[a, j] = np.max(np.abs(np.diff(y) - np.diff(fit)))
                slope[a, j] = c[1]
                rel[a, j] = abs(c[1] - GR[i, 6 * q + j]) / np.linalg.norm(GR[i, 6 * q:6 * q + 6])
        for a, i in enumerate(R3):
            print(f"   {name:9s} row {i}: resid " + ", ".join(f"{D.PARAMS[j][1]} {res[a, j]:.1e}" for j in range(6)))
            print(f"   {name:9s} row {i}: step  " + ", ".join(f"{D.PARAMS[j][1]} {stp[a, j]:.1e}" for j in range(6)))
        print(f"   {name:9s}: |slope - shipped gradient| / |g|: median {np.median(rel):.3f}, max {rel.max():.3f}")
        if q == 1:
            a = R3.index(249)
            print(f"   co2_2100  row 249, NPP0 component: h=0.25 (shipped) {GR[249, 6 + j_npp]:.1f}, "
                  f"h=0.5 {G3[0.5][1][a, j_npp]:.1f}, h=0.1 {G3[0.1][1][a, j_npp]:.1f}, "
                  f"quadratic-fit slope on +-0.5 {slope[a, j_npp]:.1f} ppmv per unit z")
    if "--paper" in sys.argv[1:]:
        paper_ensemble()
    weights_misalignment()


if __name__ == "__main__":
    main()
