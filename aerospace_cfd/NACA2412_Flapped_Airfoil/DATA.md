# NACA2412_Flapped_Airfoil: data

Kind: **stored data**. XFOIL 6.99 (M. Drela and H. Youngren, MIT, GPL) wrote every
aerodynamic number here. It is an outside Fortran program, so its outputs are the record.
`generate_data.py` and `xfoil_driver.py` (MIT, written for this data set, standard library and numpy)
drive XFOIL with the command sequence of the paper's code and write the CSVs. They reproduced
the shipped files exactly with the same XFOIL build (see Regenerating); without an XFOIL executable
`generate_data.py` stops with an error that says what to build. XFOIL is rerun only if a problem is
found in the shipped data. `davis_training_98.csv` holds the authors' own XFOIL runs, copied as text.

## Files

| file | contents | role |
|---|---|---|
| `NACA2412_Flapped_Airfoil.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2412; outputs committed |
| `inputs.csv` | 1000 × 3: alpha (deg), flap (deg), Re; Latin hypercube, seed 2412 | challenge study inputs (physical units) |
| `outputs.csv` | 1000 × 7: CL, CD, CDp, CM, transition x/c upper and lower, status | XFOIL outputs, row by row |
| `extension_inputs.csv` | 1000 × 6: as above plus Ncrit, trip x/c upper, trip x/c lower; seed 24120 | our extension (not in the paper) |
| `extension_outputs.csv` | 1000 × 7, as `outputs.csv` | XFOIL outputs for the extension |
| `reference_points.csv` | 11 × 11: set (0 = calibration, 1 = prediction), alpha, flap, Re, then the 7 output columns | XFOIL at the challenge's 7 calibration points (Table 1) and 4 prediction points (Fig. 1) |
| `davis_training_98.csv` | 98 × 6: Re, alpha, flap, CL, CD, CM | the authors' `training/training_100.csv`, values copied as text |
| `davis_rerun_outputs.csv` | 98 × 7, as `outputs.csv` | our XFOIL at the 98 inputs of the authors' file |
| `re_sensitivity.csv` | 35 × 10: alpha, flap, Re, then the 7 output columns | XFOIL at the 7 calibration points for Re = 665,000, 696,500, 700,000, 703,500 and 735,000: a labelled sensitivity check of the paper's printed Re range (p. 3), not a design |
| `convergence_ladder.csv` | 47 × 8: file, row, status, then the continuation path (side, start angle, smallest step, steps, XFOIL processes) | one row for every point of the XFOIL files that needed the retry ladder (status 3 or 4) or failed every attempt (−1); its first header line names the driver module `xfoil_run.py`; the driver in this directory is `xfoil_driver.py` |
| `generate_data.py` | writes all of the above; `--check N` reruns the first N rows of each design plus every retry-ladder row, and compares (writes nothing) | regenerates the data (needs XFOIL; exits with status 1 and a build message without it) |
| `xfoil_driver.py` | `run(alpha, flap, re, ncrit=0.01, xtr_top=1, xtr_bot=1, ...)`: one XFOIL analysis with the convergence protocol below | the XFOIL driver |
| `check_accuracy.py` | recomputes the numbers below marked with its name, from the shipped files (numpy only; does not run XFOIL; under 1 s); `--davis-src FILE` also compares `davis_training_98.csv` with the authors' file | check |

Every CSV is plain comma-separated numbers after two `#` lines (a description and the column
names with units), so `np.loadtxt(path, delimiter=",")` reads it. Status: 0 = converged
directly, 1 = converged in the authors' sweep, 2 = converged in our negative-alpha sweep,
3 = converged directly with `ITER 100`, 4 = converged in an `ITER 100` continuation
(3 and 4 are the retry ladder), −1 = no convergence. A status −1 row is kept, with `nan` in
all six output columns.

## Model

**Paper.** G. Davis and A. Renganathan, "A Bayesian latent Gaussian process framework for
aerodynamic uncertainty quantification", arXiv:2606.28871v1 [stat.ML], 27 June 2026,
doi:10.48550/arXiv.2606.28871. This is a **preprint**, the only published version. The
authors' repository holds an AIAA Journal submission package dated 2026-06-26
(`paper/aiaa_journal_submission/README_submission.txt`): submitted to AIAA Journal, not yet
published. Section, page and table numbers below refer to it. The paper addresses the Second AIAA
UQ Challenge Problem for Aerodynamics (Cary, Rumpfkeil, Hristov and Schaefer, AIAA SciTech 2026,
AIAA 2026-0091, doi:10.2514/6.2026-0091). It fits a GP surrogate to XFOIL data and calibrates
it against the challenge's truth data with a latent-input GP and a Kennedy–O'Hagan discrepancy.
It contains no active-subspace analysis.

**Low-fidelity model** (Sec. 1.1, p. 2; Sec. 1.2.1, p. 3). The model is XFOIL v6.99 "with a
100-panel baseline setup that accompanied the challenge problem", for the NACA 2412 airfoil with
a 30 % chord flap at a nominal Re = 700,000, sea level, incompressible. The full command sequence
is in the authors' code (github.com/csdlpsu/uqchallenge2, commit `73eaec9`, `uqlib.py`, class
`XFOIL`), and `xfoil_driver.py` uses it with two departures that change no number (below):
`PLOP` `G F` (graphics off); `NACA 2412`; `GDES` `FLAP` with hinge x/c = 0.70, y given as 0.5 of the local thickness
(`999`, `0.5`), deflection β (deg, + down), `EXEC`; `PPAR` `N` 100; `OPER`, `MACH 0`,
`VISC Re`, `VPAR` `N` 0.01; `PACC`; `ALFA α`. XFOIL's defaults apply otherwise: 20 viscous
iterations (`xfoil.f`, ITMAX = 20; raised to 100 only in our retry ladder), no trip (x_tr = 1).
The departures: `xfoil_driver.py` writes the trip setting out (`XTR 1 1`, XFOIL's default) so that the
extension can vary it, and it switches `PACC` on just before the last `ALFA`, where the authors switch
it on before the first (see the reader, below). The paper does not state N_crit = 0.01 (XFOIL's default is 9);
it says only that "the internal XFOIL settings are fixed" (p. 5). With N_crit = 0.01, transition
falls at x/c ≤ 0.0419 in every converged run, so the boundary layer is in effect turbulent.

**Convergence protocol.** The authors' `xfoil_training.py` is followed, each attempt in a fresh
XFOIL process:
1. `ALFA α` directly (status 0).
2. If that fails, `ALFA 0, 1, …, ceil(α)−1` and then α, i.e. `numpy.arange(0, α)` then α
   (status 1). This is the driver of p. 3 that "first accumulates converged solutions at lower
   angles of attack … and then continues toward the desired condition".
3. **Our addition**, used only if 2 fails and α < 0: the same unit steps from 0 down to α
   (status 2). The authors' sweep is empty for α ≤ 0, so it cannot help there.
4. **Retry ladder (ours)**, used only if 1–3 all fail: `ITER 100` (100 viscous iterations),
   then `ALFA α` directly (status 3).
5. **Retry ladder**, if 4 fails: an `ITER 100` continuation (status 4). It starts at the integer
   angle nearest to α that converges directly with `ITER 100`, trying first the integers on
   the side of α towards 0 (below α when α > 0, as on the authors' 0, 1, 2, … path; nearest
   first, at most 5), then those on the other side. From there it steps to α in steps of 0.5°
   (0.25° when α > 7° and β > 10°). If a step fails, the path is cut back to its last
   converged angle and the rest is redone in a new XFOIL process with half the step, down to
   1/16°; XFOIL is deterministic, so repeating the converged prefix restores the same
   converged state. Every angle of the path must appear in the polar file (XFOIL writes only
   converged points), in order, ending at α. `convergence_ladder.csv` records each path.

Attempts 4–5 run only after 1–3 fail, so every point that converges in attempts 1–3 has the
numbers it would have without the ladder: in the original run, running the four XFOIL files with
attempts 1–3 only gave identical text for every such row (961 of `outputs.csv`, 994 of
`extension_outputs.csv`, 11 of `reference_points.csv`, 96 of `davis_rerun_outputs.csv`; the counts by
`check_accuracy.py`, the comparison an external run, not reproducible from the repository).

**Why the ladder.** The points that fail attempts 1–3 are mostly not stalled. With N_crit =
0.01 transition sits at the leading edge (x/c ≤ 0.0419), and XFOIL's Newton iteration there can
settle into a limit cycle that 20 iterations do not leave. In XFOIL's trace of the failed point
(α, β) = (−2.456°, −2.943°) the largest boundary-layer residual is at station 3 (next to the
leading edge) in all 20 iterations, and the rms residual repeats a four-iteration cycle
between 0.008 and 0.056 over the last 10 iterations instead of falling (external run, not
reproducible from the repository). Point (7.286°, 10.573°), one of the high-lift failures, converges
directly with `ITER 100` (row 15 of `outputs.csv`, status 3: CL 1.4251, CD 0.02848, CM −0.0994),
well below the file's largest CL of 1.6405 (`check_accuracy.py`).
In the extension, where N_crit ∈ [0.01, 9] moves transition back, only 6 of 1000 points fail
attempts 1–3.

Only the target angle is read as the answer (attempts 1–4 switch `PACC` on just before the last
`ALFA`). The authors' reader accepts the last polar row whose α matches within 1e-3; in a sweep
it could therefore take a converged intermediate angle within 1e-3 of the target, which our
protocol excludes. A point that fails every attempt is written as `nan` with status −1, never
dropped from a file.

**Inputs.** The challenge study uses α ∈ [−5°, 10°] and β ∈ [−5°, 15°] (p. 3), and Re ∈
[696,500, 703,500] = 7e5 × [0.995, 1.005] (`xfoil_training.py` line 11, `re_range = [7e5*.995,
7e5 * 1.005]`; `uqlib.py` lines 25–26, `DEFAULT_BOUNDS`; read at commit 73eaec9 on 2026-11-02).
**The paper prints 665,000–735,000** (±5 %, p. 3: "and Reynolds number from 665000 to 735000"; the
same text is in the authors' AIAA Journal source, `paper/aiaa_journal_submission/main_aiaa_journal.tex`,
line 168). We use the code's range. The independent evidence is the
code and the authors' data: their 98 training rows fall in 98 different Re strata of a 100-point Latin
hypercube on ±0.5 % (as they do in α and β), but in only 10 strata on ±5 % (notebook, comparison
section), and their Re values lie between 696,565 and 703,451 (`check_accuracy.py`), so their design was
drawn on ±0.5 %. This is consistent with the truth data's input uncertainty, Re ± 3500 in Table 1
(p. 2), which the paper's Tables 3–5 (p. 23) also use to bound the latent Re (696,500–703,500); the
challenge truth files in the authors' repository (`given_truths.csv`,
`uq_challenge_ground_truth_epistemic_bounds.xlsx`) give no other Re interval. The printed range is
probably a factor-10 misprint; that is unconfirmed and remains an open question for the authors. `re_sensitivity.csv` runs XFOIL at the printed range too (Accuracy, below). The inputs are independent
uniforms on the box, the distribution a Latin hypercube samples.

| name | symbol | units | nominal | range | source |
|---|---|---|---|---|---|
| alpha | α | deg | none (operating condition) | [−5, 10] | Sec. 1.2.1, p. 3; Table 1 |
| flap | β_flap | deg | none (operating condition) | [−5, 15] | Sec. 1.2.1, p. 3; Table 1 |
| Re | Re | – | 7.0e5 | [696,500, 703,500] | code (`xfoil_training.py`); paper p. 3 prints [665,000, 735,000] |

**Our extension** (not in the paper or the challenge). These are three XFOIL settings that the
challenge holds fixed, each independent and uniform: N_crit ∈ [0.01, 9] (the challenge's value up
to XFOIL's default), and trip locations x_tr/c ∈ [0.05, 1] on the upper and lower surfaces
(1 = no trip, the challenge's setting). The ranges are our choice.

**Quantities of interest** (Sec. 1.1, Table 1). The lift coefficient CL, the drag coefficient
CD and the quarter-chord pitching-moment coefficient CM, all dimensionless scalars, with no
transformation. XFOIL's polar file prints CL and CM to 4 decimals and CD and CDp to 5; those
are the digits stored. CDp (pressure drag, XFOIL's CD − CDf) and the transition locations are
stored for diagnosis only.

**Gradients.** None; XFOIL does not provide them. Finite differences were judged unreliable
here. Run directly, XFOIL's CL changes by only +0.0004 over the whole ±0.5 % Re range at
(5°, 5°) (−0.0002 to +0.0006 at the 7 Table 1 points; `re_sensitivity.csv`), so a
finite-difference step of 1 % of that range is far below the printed precision. Failed points
would also break stencils. The notebook uses the global
quadratic estimator (`qphd`) on values, checked against local linear gradients (OPG) and OLS.

**Sampling design.** Latin hypercube, `generate_data.latin_hypercube` (numpy `default_rng`; one
point per stratum in each input, strata permuted independently): 1000 points, seed 2412
(challenge study), and 1000 points, seed 24120 (extension). The paper uses a 100-point Latin
hypercube (pyDOE `lhs`, no seed), plus a held-out set of 100 XFOIL points for validation (Sec. 3.1)
that is not in the authors' repository.

**Published results.** No active-subspace results. The paper gives the truth data (Table 1),
GP validation plots (Fig. 4), predictive distributions at the 4 prediction points (Figs. 5–8),
interval coverage of 94.2–95.8 % (abstract; Table 2) and learned latent inputs (Tables 3–5). None of
these is an XFOIL output that could be compared number by number. The notebook compares XFOIL
with Table 1 instead (CL outside the ±0.009 truth interval at 6 of 7 points, CM inside ±0.008 at
all 7).

**Run outcome.** Challenge study: 816 runs converged directly, 133 in the authors' sweep,
12 in ours, 35 directly with `ITER 100`, 1 in an `ITER 100` continuation (row 12, from 5° to
5.142° at β = −4.868°), and 3 (0.3 %) not at all. Of the 39 points that fail attempts 1–3, 21
lie in the corner α > 5°, β > 10° (80 design points); the largest CL among the 36 the ladder
converges is 1.6200, below the file's largest, 1.6405, so they are not post-stall. With attempts
1–3 alone the failures are missing not at random: 40.5 % (17 of 42) of the points with
y1 = w1(CL)ᵀx ≥ 1.0 failed, against 0 % for y1 ∈ [−0.2, 0.2) (notebook). The 3 remaining
failures are rows 370 (−2.456°, −2.943°, Re 699,921), 552 (3.835°, 7.906°, 700,460) and
893 (−2.048°, −4.254°, 697,595), all at moderate lift: isolated XFOIL non-convergences, not
stall. Extension: 990 direct, 3 authors' sweep, 1 ours, 6 continuation, 0 failures. No
converged run has CD ≤ 0 or a transition location outside [0, 1]. One challenge run has
CDp = −0.00108 (XFOIL bookkeeping; kept, not used). The notebook drops status −1 rows only,
and shows that neither the sweep-converged nor the ladder rows change the subspace
(|w1·w1| ≥ 0.99984 with only the directly converged rows, ≥ 0.999994 without the ladder rows).
The eigenvalue ratios depend on the rule for failed runs by up to 4 % (CL 143.7–151.2 around
147.4), so they are quoted as about 147 (CL), 67 (CD) and 13 (CM).

## Regenerating

```bash
# 1. build XFOIL 6.99 (see "XFOIL build" below), then
python generate_data.py --xfoil /path/to/xfoil                       # all files except the copy
python generate_data.py --davis-src /path/to/uqchallenge2/training/training_100.csv
python generate_data.py --xfoil /path/to/xfoil --check 25            # 25 rows of each design + ladder rows
```

Without an XFOIL executable (`--xfoil PATH` or `XFOIL_EXE`) the script stops with exit status 3 and a
message that says what to build and to rerun it (checked in `.venv`); only the `--davis-src` copy runs
without XFOIL. `--check` writes nothing. XFOIL is not installed in `.venv`, so the script was not run
there.

**Run time of `generate_data.py`** (the original run that wrote the shipped files, 2026-11-01, in a
Linux container with Python 3.13, Ubuntu 24.04.5 x86-64, one process under `nice -n 10` on a heavily
shared 2-CPU machine): 717 s of wall time and 150 s of user CPU: 425 s for the 1000 challenge runs,
190 s for the extension, 2.5 s for the reference points, 93 s for the 98 reruns and 6 s for the
35 Re-sensitivity runs. A run takes 0.1–0.2 s on average and a sweep up to about 1 s; a ladder
point a few seconds; a point that fails every attempt 30–60 s (about 35 s on an idle CPU).
In that run `--check 25` reproduced the first 25 rows of each design and all 42 ladder rows (36 + 6)
exactly, status included, and both input files from their seeds, in 63 s. The failed rows are not
rerun by `--check`.

**In `.venv`** (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, active_subspaces
1.0.0.dev0): the notebook checks that both input files are the stated Latin hypercubes
(`generate_data.latin_hypercube` from the seeds; exact) and runs in about 35 s (35.1 and 32.9 s in two runs on 2026-11-02, other jobs sharing the machine; their text outputs are identical); `check_accuracy.py` runs in
under 1 s.

**XFOIL build** (the executable is not shipped; SHA-256 of the one used:
`eaed247a63d802556bddcd11243d29ea5397d1a5a1178b8dd0bd2b9836903907`):

- Source: Ubuntu's source package from `archive.ubuntu.com/ubuntu/pool/universe/x/xfoil/`:
  `xfoil_6.99.dfsg+1.orig.tar.gz` (SHA-256 `f5ece350981f35f7b5b4cef540c921ffad877b0942591efb88bad76cc08b597a`)
  and `xfoil_6.99.dfsg+1-3.debian.tar.xz` (`b6688dc60a8437fbabb0da80fdceb623bcdd6c4f35ca8afbe130f817c6140837`).
  The 8 patches of `debian/patches/series` were applied in order with `patch -p1`.
- Compiler flags changed (no source change): in `plotlib/config.make`, `-ffpe-trap=invalid,zero`
  and `-finit-real=inf` were removed. Then `make -C plotlib` and, in `bin/`,
  `make xfoil FFLAGS="-O2 -fbounds-check -fallow-argument-mismatch -fdefault-real-8" FFLOPT="<same>"`,
  with gfortran 13.3.0 and libX11 (needed to link plotlib; graphics are switched off at run time).
  This is double precision, as in Debian's package.
- Why not the packaged binary, and why these flags (external runs, not reproducible from the
  repository): Ubuntu's `xfoil` package (`apt-get install xfoil`, 6.99.dfsg+1-3) is built with
  `-ffpe-trap=invalid,zero` and dies with SIGFPE on entering `GDES`, so it cannot deflect a flap. A
  build that kept `-finit-real=inf` ran but returned NaN in the boundary layer at easy points (e.g.
  α = −0.787°, β = −1.264°) and had failed at 24 of the authors' inputs, all of which the authors
  converged, when the test was stopped. A single-precision build (no `-fdefault-real-8`) failed 5 of
  the 98 instead of 2 and matched the authors' rounded values less often (CL 37 % against 49 % of
  converged rows), so double precision was kept.

## Accuracy

- **Repeatability** (original run; external, not reproducible from the repository without XFOIL).
  XFOIL is deterministic: `--check 25`, three repeats of single points, and running the files with
  attempts 1–3 only (for every row those attempts converge) gave identical values.
- **Against the authors' runs** (notebook, comparison section). At the 98 inputs of
  `davis_training_98.csv` this build converges at 97. It fails at (7.94°, 10.42°), which the
  authors converged. (5.04°, 11.20°) fails attempts 1–3 and converges in the ladder's
  continuation, within +0.0003 (CL), −0.00002 (CD) and 0.0000 (CM) of the authors' values, a
  check of the ladder. The median absolute differences are 5.1e-5
  (CL), 3.6e-6 (CD) and 3.8e-5 (CM), i.e. at the printed digits. The largest is 2.3e-3 in CL,
  at (9.71°, 14.95°), a sweep-converged point at the high-lift corner. Rounded to XFOIL's
  printed digits, the authors' values equal ours in 48 % (CL), 71 % (CD) and 69 % (CM) of the
  rows, so the two XFOIL builds differ in the last printed digit. The active subspaces from
  the two data sets agree to |w1·w1| ≥ 0.99990.
- **Precision.** CL and CM are stored to 1e-4 and CD to 1e-5 (XFOIL's polar format). That is
  ample for the global fits used (quadratic LOO R² 0.99921, 0.98865 and 0.99810), but too coarse
  for finite-difference gradients in Re.
- **Reynolds number** (`re_sensitivity.csv`; notebook). The data cannot resolve Re's effect
  over ±0.5 %: the quadratic's change in CL across the range is −1.05e-3 ± 1.5e-3 (1 SE) at
  mid-range α and β and −2.10e-3 ± 1.7e-3 at (5°, 5°), zero within 1.2 SE and of the wrong
  sign. XFOIL run directly gives +0.0004 at (5°, 5°) over ±0.5 % and +0.0043 over the
  paper's printed ±5 %; over ±5 % CL rises by 0.0008–0.0055 at the six Table 1 points with
  α, β ≥ 0 and falls by 0.0016 at (−5°, −5°), and CD falls by 0.00024–0.00046. Even the largest
  ±5 % change gives Re an eigenvalue weight of order 1.3e-5 of λ1, so Re is inactive for
  either range.
- **Physical fidelity.** XFOIL is the challenge's low-fidelity model. At the 7 truth points of
  Table 1 its CL differs from the truth centroid by up to 0.0290 and lies outside the ±0.009
  interval at 6 of 7 points (notebook). At the 4 prediction points, against the truth intervals
  in the authors' `uq_challenge_ground_truth_epistemic_bounds.xlsx` (digitized from Figs. 2–5
  of Cary et al.), XFOIL is inside all three intervals at (0°, 0°), (2°, 2°) and (−3°, −3°);
  at (7°, 15°) CL is 1.5155 against 1.5335 ± 0.0088 and CD 0.03808 against 0.03716 ± 0.00090,
  both outside. This model-form error is the subject of the paper.

## How these files relate to the paper

- The paper's XFOIL data are the authors' 100-point design, of which `training_100.csv` (98
  rows, shipped here as `davis_training_98.csv`) is in their repository. Their held-out 100
  validation points are not. Our 1000-point designs are new runs of the same XFOIL setup with
  this build. They are not the paper's data, and they agree with it to the last printed digit
  where both converge (above).
- The authors' file holds 13 to 15 significant digits per value (`check_accuracy.py`), while XFOIL
  6.99's polar file prints 4 or 5 decimals and the authors' reader parses that file. How their file
  got its extra digits is unknown.
- `davis_training_98.csv` is the authors' `training/training_100.csv` at commit 73eaec9 (SHA-256
  `c0bf52ce10b4c0ef555f71ad292e6d571ea629d2f74e7b4bcd5de987b94e6216`) with its header line replaced by
  two `#` lines: its 98 data lines are identical text (`check_accuracy.py --davis-src`, against a copy
  downloaded from GitHub on 2026-11-02).
- The extension design, the negative-α sweep, the retry ladder (status 3 and 4), the
  Re-sensitivity file and the 1000-point designs are ours. The authors' driver uses XFOIL's
  default 20 iterations only.

## Unknowns

- The Re range: the paper prints 665,000–735,000 (p. 3), the authors' code and data use 696,500–703,500.
  The printed range is probably a factor-10 misprint; that is unconfirmed and remains an open question for
  the authors (Model, Inputs).
- The paper does not state N_crit = 0.01; it says only that "the internal XFOIL settings are fixed" (p. 5)
  (Model).
- The paper's held-out set of 100 XFOIL validation points is not in the authors' repository, and its
  100-point Latin hypercube has no seed (Model, Sampling design).
- How the authors' file got its 13 to 15 significant digits, when XFOIL 6.99's polar file prints 4 or 5
  decimals, is unknown (How these files relate to the paper).
- Whether the authors will release `davis_training_98.csv` under CC BY 4.0 is an open question (License
  and credit).

## License and credit

Code (`generate_data.py`, `xfoil_driver.py`, `check_accuracy.py`, the notebook): MIT (`LICENSE`). Data
written here (all CSVs except `davis_training_98.csv`): CC BY 4.0 (`LICENSE-DATA`). XFOIL is © Mark Drela
and Harold Youngren, MIT, GPL; it is run as an outside program and none of its code is in this
directory. `davis_training_98.csv` is from github.com/csdlpsu/uqchallenge2 (G. Davis and
A. Renganathan, Penn State), commit `73eaec9`. **That repository states no licence**, so the file
is not under CC BY 4.0: it is hosted here with the source credited and is
listed in the exceptions list of `LICENSE-DATA`. Whether the authors will release it under CC BY 4.0
is an open question; if they attach a licence or object, that takes precedence. Cite
G. Davis and A. Renganathan, arXiv:2606.28871 (2026), doi:10.48550/arXiv.2606.28871; the
challenge problem, A. W. Cary et al., AIAA 2026-0091, doi:10.2514/6.2026-0091; and XFOIL,
M. Drela, in *Low Reynolds Number Aerodynamics*, Springer, 1989, pp. 1–12,
doi:10.1007/978-3-642-84010-4_1. The paper's work was funded by the Penn State ICDS Junior
Researcher Collaboration grant.
