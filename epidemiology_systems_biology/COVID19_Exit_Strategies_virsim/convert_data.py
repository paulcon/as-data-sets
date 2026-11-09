"""Merge the per-chunk virsim result files into the shipped CSVs (numpy only).

    python convert_data.py RESULTS_DIR --out DIR            # write runs_S.csv, ic_prev_S.csv into DIR
    python convert_data.py RESULTS_DIR --out DIR --check    # ... and compare them with the shipped files

RESULTS_DIR holds the chunk files that R/run_chunk.R writes, S_rows_AAAA-BBBB.csv
(QoIs, status, run time, echoed seed and inputs) and S_rows_AAAA-BBBB_ts.csv (daily
IC prevalence), for S in FC, CT, IL, PO: the folder `results/` of the archive of the
external run (DATA.md, "Regenerating"), or the folder generate_data.py writes.

The conversion is lossless: every value is copied from the chunk files (the row's
kind, replicate group and source row follow from its row number, see ROW LAYOUT
below). Before writing, the script checks that every row 0..571 of each strategy is
present once with status 0, that the replicate rows repeat the inputs of their group,
that the whole days used by IL and PO are the truncated inputs, and that both QoIs
recomputed from the stored daily series equal the stored QoIs (relative tolerance
1e-8). It prints the run-time totals. With --check it compares the new files with the
shipped ones and passes when every value agrees to a relative 1e-12 (the conversion
copies the decimal text, so the expected difference is 0).

ROW LAYOUT (each strategy): rows 0..511 are the main design (kind 0, group -1,
source_row -1); rows 512..571 are 6 replicate groups of 10 rows (kind 1); group 0
repeats the Table 1 mean point (source_row -1), group g = 1..5 repeats main row g - 1
(source_row g - 1) with new seeds.

--out is required and may not be this directory or inside it, so the shipped files
are never overwritten.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

STRATEGIES = ["FC", "CT", "IL", "PO"]
COMMON = ["avg_duration_infectiousness", "R0", "intervention_effect_var_inv",
          "shape_exposed_time"]
POLICY = {
    "FC": ["intervention_effect", "uptake"],
    "CT": ["trace_rate_I", "trace_prob_E", "trace_contact_reduction"],
    "IL": ["uptake", "lock_length", "lift_length", "lock_effect"],
    "PO": ["uptake", "intervention_lift_interval", "pl_intervention_effect_hi"],
}
UNITS = {
    "intervention_effect": "-", "uptake": "-", "trace_rate_I": "1/d", "trace_prob_E": "-",
    "trace_contact_reduction": "-", "lock_length": "d", "lift_length": "d", "lock_effect": "-",
    "intervention_lift_interval": "d", "pl_intervention_effect_hi": "-",
    "avg_duration_infectiousness": "d", "R0": "-", "intervention_effect_var_inv": "-",
    "shape_exposed_time": "-",
}
# whole days that virsim used (the inputs truncated), written by the driver
APPLIED = {"FC": [], "CT": [], "IL": [("lock_days", "lock_length"), ("lift_days", "lift_length")],
           "PO": [("lift_interval_days", "intervention_lift_interval")]}
N_MAIN, GROUPS, REPS = 512, 6, 10
N_ROWS = N_MAIN + GROUPS * REPS
N_DAYS = 551            # days 0..550
IC_CAPACITY = 109.0     # IC beds per million (paper p. 6)
WINDOW = 30             # moving-average window in days (paper p. 6)
QOI_TOL = 1e-8
CHECK_TOL = 1e-12


def inputs(s):
    return POLICY[s] + COMMON


def read_text_rows(path):
    """Return (column names, rows as lists of decimal strings) of a chunk file."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    cols = [re.sub(r"\s*\[.*\]$", "", c.strip()) for c in lines[1].lstrip("#").split(",")]
    rows = [ln.split(",") for ln in lines[2:] if ln.strip() and not ln.startswith("#")]
    for r in rows:
        if len(r) != len(cols):
            sys.exit(f"{os.path.basename(path)}: a row has {len(r)} fields, header {len(cols)}")
    return cols, rows


def centred_ma(ts):
    """Centred moving average of even order WINDOW, weights (1/2, 1, ..., 1, 1/2)/WINDOW."""
    w = np.r_[0.5, np.ones(WINDOW - 1), 0.5] / WINDOW
    return np.convolve(ts, w, mode="valid")


def layout(row):
    if row < N_MAIN:
        return 0, -1, -1
    g = (row - N_MAIN) // REPS
    return 1, g, (-1 if g == 0 else g - 1)


def out_columns(s):
    cols = ["row", "kind", "group", "source_row", "seed"] + inputs(s) + [
        "status", "runtime_s", "IC_prev_avg_max", "IC_excess_days"] + [a for a, _ in APPLIED[s]]
    units = ["-"] * 5 + [UNITS[n] for n in inputs(s)] + [
        "-", "s", "IC patients per million", "IC patient-days per million"] + ["d"] * len(APPLIED[s])
    return cols, units


def merge(results, s):
    """Collect, check and return the text table (runs) and the daily series of strategy s."""
    files = sorted(f for f in glob.glob(os.path.join(results, f"{s}_rows_*.csv"))
                   if not f.endswith("_ts.csv"))
    if not files:
        sys.exit(f"{s}: no chunk files in {results}")
    if glob.glob(os.path.join(results, f"{s}_rows_*.partial")):
        sys.exit(f"{s}: unfinished chunks (.partial files) in {results}")
    runs, ts = {}, {}
    for f in files:
        cols, rows = read_text_rows(f)
        ci = {c: i for i, c in enumerate(cols)}
        for r in rows:
            k = int(r[ci["row"]])
            if k in runs:
                sys.exit(f"{s}: row {k} appears twice")
            runs[k] = {c: r[ci[c]] for c in cols}
        tcols, trows = read_text_rows(f[:-4] + "_ts.csv")
        for r in trows:
            k = int(r[0])
            if k in ts:
                sys.exit(f"{s}: daily series of row {k} appears twice")
            if len(r) != N_DAYS + 1:
                sys.exit(f"{s}: daily series of row {k} has {len(r) - 1} days")
            ts[k] = r[1:]
    missing = sorted(set(range(N_ROWS)) - set(runs))
    if missing or len(runs) != N_ROWS:
        sys.exit(f"{s}: {len(missing)} of {N_ROWS} rows missing, e.g. {missing[:3]}")
    bad = [k for k in range(N_ROWS) if runs[k]["status"] != "0" or k not in ts]
    if bad:
        sys.exit(f"{s}: {len(bad)} rows without status 0 or without a daily series, e.g. {bad[:3]}")

    cols, _ = out_columns(s)
    table, worst = [], 0.0
    for k in range(N_ROWS):
        r = runs[k]
        kind, g, src = layout(k)
        line = [str(k), str(kind), str(g), str(src)] + [r[c] for c in cols[4:]]
        table.append(line)
        series = np.array(ts[k], dtype=float)
        q1 = centred_ma(series).max()
        q2 = np.maximum(0.0, series - IC_CAPACITY).sum()
        for new, old in ((q1, float(r["IC_prev_avg_max"])), (q2, float(r["IC_excess_days"]))):
            worst = max(worst, abs(new - old) / max(1.0, abs(old)))
        for a, src_name in APPLIED[s]:
            if int(r[a]) != int(np.floor(float(r[src_name]))):
                sys.exit(f"{s}: row {k}: {a} = {r[a]} is not the truncated {src_name}")
    if worst > QOI_TOL:
        sys.exit(f"{s}: QoIs recomputed from the daily series differ by {worst:.1e} (relative)")
    # replicate rows repeat the inputs of their group
    names = inputs(s)
    for g in range(GROUPS):
        rows_g = [N_MAIN + g * REPS + j for j in range(REPS)] + ([g - 1] if g > 0 else [])
        ref = [table[rows_g[0]][5 + j] for j in range(len(names))]
        for k in rows_g:
            if [table[k][5 + j] for j in range(len(names))] != ref:
                sys.exit(f"{s}: replicate group {g}: row {k} has other inputs")
    seeds = [int(t[4]) for t in table]
    if len(set(seeds)) != N_ROWS:
        sys.exit(f"{s}: repeated seeds")
    rt = np.array([float(t[cols.index("runtime_s")]) for t in table])
    print(f"{s}: {N_ROWS} rows, all status 0; QoIs recomputed from the daily series: max relative "
          f"difference {worst:.1e}; run time median {np.median(rt):.0f} s, max {rt.max():.0f} s, "
          f"{rt.sum() / 3600:.1f} process-hours")
    return table, [[str(k)] + ts[k] for k in range(N_ROWS)], rt


def write(out, s, table, series):
    cols, units = out_columns(s)
    h1 = (f"# virsim v1.0.5 runs, strategy {s} (Gugole et al. 2021, PLOS Comput Biol 17:e1009355); "
          f"rows 0-511 main design (scrambled Sobol' points through the Table 1 inverse CDFs), "
          f"rows 512-571 seed replicates (kind 1; group 0 = Table 1 mean point, group g = main row "
          f"g-1); status 0 = ok")
    h2 = "# " + ", ".join(f"{c} [{u}]" for c, u in zip(cols, units))
    with open(os.path.join(out, f"runs_{s}.csv"), "w") as fh:
        fh.write(h1 + "\n" + h2 + "\n")
        for line in table:
            fh.write(",".join(line) + "\n")
    h1 = (f"# virsim v1.0.5 daily IC prevalence, strategy {s}: number of prevalent IC patients per "
          f"million on days 0-550, one line per row of runs_{s}.csv")
    h2 = "# row [-], " + ", ".join(f"day_{d} [IC patients per million]" for d in range(N_DAYS))
    with open(os.path.join(out, f"ic_prev_{s}.csv"), "w") as fh:
        fh.write(h1 + "\n" + h2 + "\n")
        for line in series:
            fh.write(",".join(line) + "\n")


def check(out, s):
    ok = True
    for fn in (f"runs_{s}.csv", f"ic_prev_{s}.csv"):
        new = np.loadtxt(os.path.join(out, fn), delimiter=",")
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",")
        if new.shape != old.shape:
            print(f"  {fn}: shape {new.shape} differs from the shipped {old.shape}")
            ok = False
            continue
        rel = np.max(np.abs(new - old) / np.maximum(1.0, np.abs(old)))
        flag = "ok" if rel <= CHECK_TOL else "DIFFERS"
        print(f"  {fn}: max |new - shipped| / max(1, |shipped|) = {rel:.1e} ({flag}, tolerance {CHECK_TOL:g})")
        ok &= rel <= CHECK_TOL
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True, help="output folder, outside this directory")
    ap.add_argument("--check", action="store_true", help="compare with the shipped files")
    ap.add_argument("--strategy", choices=STRATEGIES, nargs="*", default=STRATEGIES)
    a = ap.parse_args()
    out = os.path.realpath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([out, here]) == here:
        sys.exit("--out must be outside this data set's directory (not the directory or a folder in it); "
                 "the shipped files are never overwritten")
    os.makedirs(out, exist_ok=True)
    ok, total = True, []
    for s in a.strategy:
        table, series, rt = merge(a.results_dir, s)
        write(out, s, table, series)
        total.append(rt)
        if a.check:
            ok &= check(out, s)
    rt = np.concatenate(total)
    print(f"all: {rt.size} runs, {rt.sum() / 3600:.1f} process-hours, mean {rt.mean():.0f} s per run")
    if a.check:
        print("check passed" if ok else "check FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
