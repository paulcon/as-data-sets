"""Convert the stored GlaDS-GP runs of Hill et al. (2025) into this data set's CSV files.

Stored data: the values were computed by the authors with GlaDS (ISSM v4.24) and deposited in the
Zenodo record "timghill/GladsGP: v1.0.0", doi:10.5281/zenodo.15053542 (MIT licence; archive
``GladsGP.tar.gz``, 5,157,897,717 bytes). This script only changes the layout. It reads eight files of
the extracted archive, checks their SHA-256, and writes

    glads_train_inputs.csv, glads_test_inputs.csv  run_id + 8 GlaDS parameters (physical units); the
                                                   number tokens are copied character for character
                                                   from expdesign/synthetic_{train,test}_physical.csv
    glads_train_qoi.csv,    glads_test_qoi.csv     run_id + 17 scalar outputs per run, read from
                                                   issm/{train,test}/synthetic_{channel_frac,
                                                   log_transit_time,channel_length}.npy

Every float32 value is written with 9 significant digits and every float64 value with ``repr``; both
reload bit for bit, which the script checks. The files have two ``#`` header lines (a description;
column names with units) and load with ``np.loadtxt(path, delimiter=",")``.

    python convert_data.py SRC --out DIR            # write the four CSV files into DIR
    python convert_data.py SRC --out DIR --check    # also compare them with the shipped files

SRC is the directory that holds ``experiments/synthetic/`` of the extracted archive. DIR must lie
outside this data set's directory. ``--check`` compares every number with the shipped file to a
relative tolerance of 1e-12 (NaN equal to NaN) and also prints whether the text is identical.
It also prints whether every stored value is exactly representable in float32.
Exit status 0 = pass, 1 = fail (a source file differs from v1.0.0, or the check fails), 2 = usage error
(including a missing source file). Needs numpy; no pickle is ever loaded.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join("experiments", "synthetic")
RTOL = 1e-12
ARCHIVE = ("GladsGP.tar.gz of the Zenodo record doi:10.5281/zenodo.15053542 "
           "(https://zenodo.org/records/15053542, 5,157,897,717 bytes)")


class UsageError(Exception):
    """A source file is missing: SRC is not the extracted archive."""

# SHA-256 of the archive members read (archive GladsGP.tar.gz, SHA-256
# f17b96da1a469cb5f2be2663ed8174f6aef8f426585a5098e0b05c5d3561c450)
SOURCES = {
    "expdesign/synthetic_train_physical.csv":
        "92821f8a0219d4391b02b768d71af63e3b4b8bd8a9833d2979d3440d63517dec",
    "expdesign/synthetic_test_physical.csv":
        "51677a66f66fd36b2b272b86b2ecf82a3eec9442230d3145f16a34327a5d249b",
    "issm/train/synthetic_channel_frac.npy":
        "4b35ce2127a4e5b0119c1f99691e13ba56b56bf4559e74c3da5289a11f39cb30",
    "issm/train/synthetic_log_transit_time.npy":
        "2b13a8e88958ab31c0afe1122cd327c10bc83a4ffa76052acb1d41a21c23b835",
    "issm/train/synthetic_channel_length.npy":
        "689f10491bc8e7c9b8417636c56cf7975df562220c8b6f14e7ec45a278ef6a0b",
    "issm/test/synthetic_channel_frac.npy":
        "bbb35ae7b8fabbb4053f7eed213c9e2ecc27628b1b51d176e48538407c43315e",
    "issm/test/synthetic_log_transit_time.npy":
        "b55238e10d4ea34e7135b6d6ee282416710df11cbbd811f58338ebacb05b3a27",
    "issm/test/synthetic_channel_length.npy":
        "7398775b65a19b7fa26111d71c3d18004e80e0ca2c350261ab74345c8481c98c",
}

# Column order of the authors' design files (l_c comes before A, unlike the rows of Table 2).
HEADER = [r"$k_{\rm{s}}$", r"$k_{\rm{c}}$", r"$h_{\rm{b}}$", r"$r_{\rm{b}}$",
          r"$l_{\rm{c}}$", r"$A$", r"$\omega$", r"$e_{\rm{v}}$"]
INPUTS = [("k_s", "Pa s^-1 as printed in Table 2"), ("k_c", "m^(3/2) s^-1 as printed in Table 2"),
          ("h_b", "m"), ("r_b", "-"), ("l_c", "m"), ("A", "s^-1 Pa^-3"), ("omega", "-"), ("e_v", "-")]
N = {"train": 512, "test": 100}
GATES = [f"x{g:02d}km" for g in (5, 10, 15, 20, 25, 30)]
QOIS = [  # (file stem, columns, units); each .npy is (k, N): one row per column
    ("channel_frac", [f"fQ_{g}" for g in GATES] + ["fQ_mean"], "-"),
    ("log_transit_time", [f"log10Ts_{g}" for g in GATES] + ["log10Ts_mean"], "log10(a)"),
    ("channel_length", ["Lc_R0.5m", "Lc_R1.0m"], "km"),
]
DESC_IN = ("GlaDS-GP {split} design ({design}), physical units, values copied verbatim from "
           "expdesign/synthetic_{split}_physical.csv; run_id = job number (1-based); Hill et al. 2025 "
           "GMD doi:10.5194/gmd-18-4045-2025, data doi:10.5281/zenodo.15053542 (MIT)")
DESIGN = {"train": "512-point Sobol", "test": "100-point CD-minimised LHS"}
DESC_Q = ("GlaDS-GP {split} scalar QoIs per run, copied from issm/{split}/synthetic_*.npy (authors' "
          "aggregate_outputs.py); fQ = channel/total discharge at fluxgate x and mean over 6 gates "
          "(paper fQ = fQ_mean); log10Ts = log10 sheet transit time in years (paper log Ts = "
          "log10Ts_mean); Lc = channel network length for radius >= R (paper Lc = Lc_R0.5m); same "
          "run_id as the inputs file")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(x, dtype):
    if not np.isfinite(x):
        return "nan"
    return format(x, ".9g") if dtype == np.float32 else repr(float(x))


def build(src):
    """Return {file name: text} for the four CSV files, after checking every source file."""
    exp = os.path.join(src, EXP)
    for rel, digest in SOURCES.items():
        p = os.path.join(exp, rel)
        if not os.path.isfile(p):
            raise UsageError(f"missing source file {p}; SRC must be the directory that holds experiments/ "
                             f"after extracting {ARCHIVE}")
        if sha256(p) != digest:
            raise SystemExit(f"{p}: SHA-256 differs from the recorded value; not the v1.0.0 file")
    texts, arrays, f32 = {}, {}, []
    for split, n in N.items():
        with open(os.path.join(exp, "expdesign", f"synthetic_{split}_physical.csv"),
                  encoding="utf-8") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
        if [h.strip().lstrip("#").strip() for h in lines[0].split(",")] != HEADER:
            raise SystemExit(f"{split} design: unexpected header {lines[0]!r}")
        tokens = [[t.strip() for t in ln.split(",")] for ln in lines[1:]]
        if len(tokens) != n or any(len(r) != 8 for r in tokens):
            raise SystemExit(f"{split} design: expected {n} rows of 8 fields")
        cols = ["run_id"] + [f"{a} [{u}]" for a, u in INPUTS]
        rows = [[str(i + 1)] + r for i, r in enumerate(tokens)]
        name = f"glads_{split}_inputs.csv"
        texts[name] = ("# " + DESC_IN.format(split=split, design=DESIGN[split]) + "\n# "
                       + ",".join(cols) + "\n" + "".join(",".join(r) + "\n" for r in rows))
        arrays[name] = np.array([[float(t) for t in r] for r in rows])

        blocks, qcols = [], []
        for stem, names, units in QOIS:
            Y = np.load(os.path.join(exp, "issm", split, f"synthetic_{stem}.npy"), allow_pickle=False)
            if Y.shape != (len(names), n):
                raise SystemExit(f"{split} {stem}: shape {Y.shape}, expected {(len(names), n)}")
            blocks.append(Y.T)
            f32.append(np.array_equal(Y.astype(np.float32).astype(Y.dtype), Y, equal_nan=True))
            qcols += [f"{c} [{units}]" for c in names]
        rows = [[str(i + 1)] + [fmt(x, b.dtype) for b in blocks for x in b[i]] for i in range(n)]
        name = f"glads_{split}_qoi.csv"
        texts[name] = ("# " + DESC_Q.format(split=split) + "\n# " + ",".join(["run_id"] + qcols)
                       + "\n" + "".join(",".join(r) + "\n" for r in rows))
        arrays[name] = np.column_stack([np.arange(1, n + 1, dtype=float)]
                                       + [b.astype(np.float64) for b in blocks])
        # bit-for-bit round trip of every stored value through its text
        back = np.array([[float(t) for t in r] for r in rows])
        start = 1
        for b in blocks:
            k = b.shape[1]
            if not np.array_equal(back[:, start:start + k].astype(b.dtype), b, equal_nan=True):
                raise SystemExit(f"{name}: the text does not reload to the stored {b.dtype} values")
            start += k
    print("every stored output value exactly representable in float32:", "yes" if all(f32) else "no")
    return texts, arrays


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="directory holding experiments/synthetic/ of the extracted archive")
    ap.add_argument("--out", required=True, help="output directory, outside this data set's directory")
    ap.add_argument("--check", action="store_true", help="compare with the shipped CSV files")
    a = ap.parse_args(argv)
    if inside(a.out, HERE):
        print("--out must be outside the data set directory; the shipped files are never overwritten")
        return 2
    try:
        texts, arrays = build(a.src)
    except UsageError as err:
        print(err, file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"wrote {name}: {arrays[name].shape[0]} rows x {arrays[name].shape[1]} columns")
    if not a.check:
        return 0
    bad = 0
    for name, text in texts.items():
        shipped = os.path.join(HERE, name)
        S = np.loadtxt(shipped, delimiter=",")
        B = np.loadtxt(os.path.join(a.out, name), delimiter=",")
        ok = S.shape == B.shape and np.allclose(B, S, rtol=RTOL, atol=0.0, equal_nan=True)
        if ok:
            dev = np.abs(B - S)[np.isfinite(S)]
            scale = np.abs(S)[np.isfinite(S)]
            rel = float(np.max(dev / np.where(scale > 0, scale, 1.0)))
        with open(shipped, encoding="utf-8") as fh:
            same_text = fh.read() == text
        print(f"{name}: {'ok' if ok else 'DIFFERENT'} (shape {S.shape}"
              + (f", max relative difference {rel:.1e}" if ok else "")
              + f", tolerance {RTOL:g}; text {'identical' if same_text else 'differs'})")
        bad += not ok
    print("CHECK PASSED" if not bad else "CHECK FAILED")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
