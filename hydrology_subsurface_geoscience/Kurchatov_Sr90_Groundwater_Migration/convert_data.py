"""Convert the VLSE MARTHE data file to the repository's CSV format, losslessly.

marthedata.txt (tab-separated, one header line, 300 rows, 20 input then 10 output
columns) is copied verbatim from https://www.sfu.ca/~ssurjano/Code/marthedata.txt
(downloaded 2026-11-01). This script writes inputs.csv and outputs.csv: two '#' header
lines, then the original number strings, unchanged, separated by commas. No value is
re-formatted, rounded or dropped.

    python convert_data.py           # write inputs.csv and outputs.csv
    python convert_data.py --check   # verify the shipped CSVs against marthedata.txt
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "marthedata.txt"
SHA256 = "2c0981a73bf4b2641db19d1b6727113918569f90d5ac51488e869d5a263a39c7"
N_IN = 20

# Units: [n.r.] = not reported in the paper or on the VLSE page. kd in m3/m3 (paper, Fig. 5);
# concentrations in Bq/l (paper, Fig. 6).
UNITS = {"kd1": "m3/m3", "kd2": "m3/m3", "kd3": "m3/m3", "poros": "-"}
DESC_IN = ("# MARTHE Sr-90 migration model of the RRC Kurchatov Institute radwaste site (Volkova, Iooss "
           "and Van Dorpe 2008, doi:10.1007/s00477-006-0093-y): the 20 inputs of the 300 runs, physical "
           "units, values copied verbatim from marthedata.txt (VLSE); [n.r.] = unit not reported")
DESC_OUT = ("# MARTHE Sr-90 migration model (Volkova, Iooss and Van Dorpe 2008): predicted Sr-90 "
            "concentration at 10 observation wells at the end of 2010, values copied verbatim from "
            "marthedata.txt (VLSE); rows match inputs.csv")


def read_source():
    raw = SRC.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"marthedata.txt SHA-256 {digest} differs from {SHA256}")
    lines = raw.decode("ascii").splitlines()
    names = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    if len(names) != 30 or any(len(r) != 30 for r in rows) or len(rows) != 300:
        raise SystemExit("unexpected layout of marthedata.txt")
    return names, rows


def csv_text(desc, cols, rows, sl):
    head = desc + "\n# " + ", ".join(f"{c} [{u}]" for c, u in cols) + "\n"
    return head + "".join(",".join(r[sl]) + "\n" for r in rows)


def build():
    names, rows = read_source()
    cols_in = [(n, UNITS.get(n, "n.r.")) for n in names[:N_IN]]
    cols_out = [(n, "Bq/l") for n in names[N_IN:]]
    return (csv_text(DESC_IN, cols_in, rows, slice(0, N_IN)),
            csv_text(DESC_OUT, cols_out, rows, slice(N_IN, None)), names, rows)


def check():
    t_in, t_out, names, rows = build()
    ok = True
    for fn, txt in (("inputs.csv", t_in), ("outputs.csv", t_out)):
        same = (HERE / fn).read_text() == txt
        print(f"{fn}: byte-identical to a fresh conversion: {same}")
        ok &= same
    src = np.loadtxt(SRC, skiprows=1)
    csv = np.hstack([np.loadtxt(HERE / "inputs.csv", delimiter=","),
                     np.loadtxt(HERE / "outputs.csv", delimiter=",")])
    tok = all(a == b for r, ln in zip(rows, zip(*(open(HERE / f).read().splitlines()[2:]
                                                  for f in ("inputs.csv", "outputs.csv"))))
              for a, b in zip(r, ln[0].split(",") + ln[1].split(",")))
    print(f"parsed values identical (np.array_equal): {np.array_equal(src, csv)}; shape {csv.shape}; "
          f"number strings identical: {tok}")
    ok &= np.array_equal(src, csv) and tok
    print("OK" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    t_in, t_out, _, _ = build()
    (HERE / "inputs.csv").write_text(t_in)
    (HERE / "outputs.csv").write_text(t_out)
    print("wrote inputs.csv and outputs.csv")
