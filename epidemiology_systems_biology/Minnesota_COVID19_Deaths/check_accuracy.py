"""Accuracy checks for Minnesota_COVID19_Deaths (needs R and the article's supplement folder 'code').

    python check_accuracy.py --code CODE_DIR [--rscript PATH]       # about 60 model runs

Prints the numbers of the Accuracy section of DATA.md:
  1. the driver against the article's own simulator runs: covid_sim at the first 5 design points of
     the first stored sequential-design run in code/data/covid60.RData (mapped to [0.01, 0.99] as in
     code/covid_sim.R), compared with the deaths stored there;
  2. the shipped forward-difference gradients against central differences (step 1e-6) at rows
     0, 1, 2 of the shipped files (relative 2-norm error);
  3. that inputs 4 and 5 do not change the output (row 0 with each set to 0.01 and to 0.99);
  4. the article's reference matrix C (code/data/covid_Ch.RData, object Cbmat: the GP estimate
     from 2,000 uniform samples on [0, 1]^8 behind the article's Fig. 5, right), its eigenvalues,
     their natural logarithms, and the first three eigenvectors.
Without R or the supplement folder the script stops with an error that says what to install.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mn_covid_driver as D  # noqa: E402

READ_R = r"""
args <- commandArgs(trailingOnly = TRUE)
setwd(args[1])
load('data/covid60.RData')
m <- res[[1]]$models[[1]]
w17 <- function(A, file) writeLines(apply(A, 1, function(r) paste(sprintf('%.17g', r), collapse = ',')), file)
w17(cbind(m$X0[1:5, ], m$Z0[1:5]), args[2])      # 17 significant digits: the stored doubles exactly
load('data/covid_Ch.RData')
w17(Cbmat, args[3])
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", default=None)
    p.add_argument("--rscript", default=None)
    a = p.parse_args()
    try:
        code = D.find_code(a.code)
        rscript = D.find_rscript(a.rscript)
    except RuntimeError as e:
        sys.exit(str(e))
    rv, pv = D.versions(rscript)
    print(f"{rv}, plyr {pv}")
    with tempfile.TemporaryDirectory() as tmp:
        rfile = os.path.join(tmp, "read.R")
        with open(rfile, "w") as fh:
            fh.write(READ_R)
        ref_csv, c_csv = os.path.join(tmp, "ref.csv"), os.path.join(tmp, "C.csv")
        subprocess.run([rscript, rfile, code, ref_csv, c_csv], check=True)
        ref = np.loadtxt(ref_csv, delimiter=",")
        C = np.loadtxt(c_csv, delimiter=",")

        X = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
        G = np.loadtxt(os.path.join(HERE, "gradients.csv"), delimiter=",")
        f = np.loadtxt(os.path.join(HERE, "outputs.csv"), delimiter=",")
        h = 1e-6
        rows = [0, 1, 2]
        pts = [D.from_unit(ref[:, :8])]
        for i in rows:
            for j in range(8):
                for s in (1, -1):
                    x = X[i].copy()
                    x[j] += s * h
                    pts.append(x[None])
        x = X[0].copy()
        for v in (0.01, 0.99):
            x[3] = x[4] = v
            pts.append(x[None].copy())
        pts.append(X[rows])
        P = np.vstack(pts)
        y = D.run_batch(P, os.path.join(tmp, "y.csv"), code, rscript)

    k = 0
    y_ref = y[k:k + 5]
    k += 5
    rel = np.abs(y_ref - ref[:, 8]) / ref[:, 8]
    print("1. covid_sim at 5 stored design points of covid60.RData (res[[1]], first model):")
    for r, yr, z in zip(rel, y_ref, ref[:, 8]):
        print(f"   driver {yr:.10g}, stored {z:.10g}, relative difference {r:.1e}")
    print(f"   max relative difference {rel.max():.1e}")
    half = 0.5 * (D.UPPER - D.LOWER)
    print("2. shipped forward differences (step 1e-7) vs central differences (step 1e-6), "
          "relative 2-norm error in normalized coordinates:")
    for i in rows:
        yc = y[k:k + 16].reshape(8, 2)
        k += 16
        gc = (yc[:, 0] - yc[:, 1]) / (2 * h)
        e = np.linalg.norm((G[i] - gc) * half) / np.linalg.norm(gc * half)
        print(f"   row {i}: {e:.1e}; largest component error / |grad| per input "
              + np.array2string(np.abs(G[i] - gc) * half / np.linalg.norm(gc * half), precision=1))
    y45 = y[k:k + 2]
    k += 2
    y_rows = y[k:k + 3]
    print(f"3. row 0 with inputs 4 and 5 at 0.01 and 0.99: deaths {y45[0]:.10g} and {y45[1]:.10g}; "
          f"shipped value {f[0]:.10g}; differences {y45[0] - f[0]:.1e}, {y45[1] - f[0]:.1e}")
    print(f"   rows {rows} rerun: max relative difference from outputs.csv "
          f"{np.max(np.abs(y_rows - f[rows]) / f[rows]):.1e}")
    lam, W = np.linalg.eigh(C)
    lam, W = lam[::-1], W[:, ::-1]
    W = W * np.sign(W[np.argmax(np.abs(W), axis=0), np.arange(8)])   # sign rule: largest |entry| positive
    np.set_printoptions(linewidth=120)
    print("4. article's reference matrix Cbmat (covid_Ch.RData; inputs on [0, 1]):")
    print("   eigenvalues " + np.array2string(lam, precision=4))
    print("   ln eigenvalues " + np.array2string(np.log(lam), precision=3))
    print("   ratios lambda_k/lambda_k+1 " + np.array2string(lam[:-1] / lam[1:], precision=3))
    for c in range(3):
        print(f"   w{c + 1} = " + np.array2string(W[:, c], precision=4, suppress_small=False))


if __name__ == "__main__":
    main()
