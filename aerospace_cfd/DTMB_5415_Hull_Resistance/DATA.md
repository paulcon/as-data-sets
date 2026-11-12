# DTMB_5415_Hull_Resistance: data

Kind: **stored data**. The files were generated for this data set on 2026-10-31 and 2026-11-01 by
running the L2-Sea Fortran code of A. Serani and M. Diez (CNR-INM; GPL-3.0) as an outside program,
through `l2sea_driver.py` and `generate_data.py` (both MIT, written for this data set; no L2-Sea code
is copied). The values are as the code printed them and are not altered. The shipped files are the
record: `generate_data.py` regenerates them where a build of the same code is available (see
Regenerating), and without one it stops with an error that says what to build. L2-Sea is rerun only if
a problem is found in the shipped data.

## Files

| file | contents | role |
|---|---|---|
| `DTMB_5415_Hull_Resistance.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 5415; outputs committed |
| `inputs.csv` | 384 × 16: Fr, T (m, full scale), s1..s14, physical units | stored inputs: the first 384 points of a scrambled Sobol' sequence |
| `outputs.csv` | 384 × 9: R_T (N), g1..g4, R_w (N), R_f (N), S_wet (m²), displacement (m³) | L2-Sea output at fidelity 7 |
| `fidelity_check.csv` | 141 × 12: row of `inputs.csv`, fidelity, the nine outputs at that fidelity, R_T at fidelity 7 | grid check: rows 0–5 on G3; row 63 on G7 and G5; rows 0–127 (a full 128-point Sobol' net) and rows 189, 207, 305, 323, 343 on G6 (row 305 failed there: NaN) |
| `draft_sweep.csv` | 99 × 14: fidelity, Fr, T, the nine outputs, Rw_wc (N), Rf_ittc (N) | parent hull at closely spaced drafts T ∈ [−5.70, −5.58] m: G7 at Fr 0.28 (49 drafts) and 0.36 (25), G6 at Fr 0.28 (25); maps the failure band (runs that raised: NaN) |
| `shape_inputs.csv` | 256 × 14: s1..s14 | **second study (shape only)**: first 256 points of a 14-D scrambled Sobol' sequence (seed 14) on [−1, 1]¹⁴; Fr = 0.28, T = −6.16 m, G7 held |
| `shape_outputs.csv` | 256 × 11: the nine outputs, Rw_wc (N), Rf_ittc (N) | L2-Sea output of the shape-only study |
| `design_diagnostics.csv` | 41 × 3: row of `inputs.csv`, Rw_wc (N), Rf_ittc (N) | G7 reruns of every 12th row, the 7 flagged rows and rows 71, 101, recording two diagnostics `outputs.csv` lacks (their nine outputs equal `outputs.csv` exactly) |
| `reference_runs.csv` | 8 × 29: hull (0 parent, 1 the paper's optimum), fidelity, 16 inputs, nine outputs, the code's own time report (s; a tenth of the wall time, see Fidelity), wall time (s) | reference points of the paper |
| `uq_uniform_quadrature.csv` | 18 × 4: Fr, T, weight, R_T | parent hull, 6 × 3 Gauss–Legendre grid in (Fr, T), fidelity 6 |
| `generate_data.py` | the design, the reference points, the run loop, `assemble` | regenerates the files (needs a build of L2-Sea; exits with status 3 and a build message without one) |
| `l2sea_driver.py` | writes the two L2-Sea input files, runs the program, reads its outputs; `failure_reason`, the per-run failure rule | the interface to L2-Sea (standard library only) |
| `check_accuracy.py` | recomputes the numbers of Accuracy below that the notebook does not print, from the shipped files (numpy only; does not run L2-Sea; under 1 s) | check |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column
names with units), so `np.loadtxt(path, delimiter=',')` reads them.

## Model

**Credit.** The model and code are A. Serani and M. Diez's L2-Sea (CNR-INM); the benchmark that
publishes it is Seelinger et al. (2025), a paper of 25 authors.

**Citation.** L. Seelinger, A. Reinarz, M. B. Lykkegaard, R. Akers, A. M. A. Alghamdi, D. Aristoff,
W. Bangerth, J. Bénézech, M. Diez, K. Frey, J. D. Jakeman, J. S. Jørgensen, K.-T. Kim, B. M. Kent,
M. Martinelli, M. Parno, R. Pellegrini, N. Petra, N. A. B. Riis, K. Rosenfeld, A. Serani,
L. Tamellini, U. Villa, T. J. Dodwell, R. Scheichl, "Democratizing uncertainty quantification",
*J. Comput. Phys.* 521:113542 (2025), doi:10.1016/j.jcp.2024.113542 (open access, CC BY 4.0;
publisher PDF read; the arXiv preprint 2402.13768 was not used). The model is App. A.1.2
(pp. 18–19); the UQ benchmark Sec. 5.1 (pp. 11–13) and App. A.3.3 (p. 32); the optimization
benchmark App. A.4.1 (pp. 32–33). Grid details: R. Pellegrini et al., *Mathematics* 10(3):481
(2022), doi:10.3390/math10030481, Table 2.

**Physics.** Linear potential flow (Laplace's equation, A.2) around the DTMB 5415 hull at model
scale (Lpp = 5.720 m), solved by a boundary element method with the Dawson linearization of the
free-surface condition. Wave resistance R_w integrates the hull pressure from Bernoulli's theorem
(A.3); frictional resistance R_f uses a flat-plate formula with the local Reynolds number; R_T =
R_w + R_f. Sinkage and trim frozen (`sinkoff = trimoff = 'y'`, as in both of the paper's
benchmarks). Hull shape: g(ξ, s) = g0(ξ) + Σ s_k ψ_k(ξ) (A.4), hierarchical orthonormal modes; in
the code mode k is scaled by √(3 a_k), with a_k the KLE variance (`alfak.out`: 5.44, 2.86, 1.88,
1.50, 0.978, 0.666, 0.509, 0.398, 0.376, 0.228, 0.136, 0.115, 0.0966, 0.0716).

**Inputs** (all independent uniform in this data set):

| name | symbol | units | nominal | range | source |
|---|---|---|---|---|---|
| Froude number | Fr | – | 0.28 (optimization benchmark) | [0.25, 0.41] | App. A.1.2, p. 19; Sec. 5.1.1, p. 11 |
| Draft (vertical translation `ztrasla` of the full-scale grid) | T | m, full scale | −6.16 | [−6.776, −5.544] (±10 %) | App. A.1.2, p. 19; Eq. 2, p. 11 |
| Shape coefficients | s1 … s14 | – | 0 (parent hull) | [−1, 1] each | App. A.1.2, p. 19 |

The units of T are not stated in the paper. The code adds `ztrasla` to the vertical coordinate of the
full-scale grid file (Lpp = 142 m, `dist_pp`), and the design value 6.16 m at full scale is the
0.248 m model-scale draft of Pellegrini et al., Table 1 (scale 142/5.72 = 24.8); so T is a full-scale
length in metres, negative below the waterline.

The paper's UQ benchmark (Sec. 5.1, Eqs. 1–2; App. A.3.3) varies only (Fr, T): Fr triangular on
[0.25, 0.41] with density 2(Fr_b − t)/(Fr_b − Fr_a)² (mode at 0.25) and T beta(α = β = 10) on
[−6.776, −5.544] (in the paper's parametrization of Eq. 2, p. 11, density ∝ (t − T_a)^α (T_b − t)^β;
Beta(11, 11) in the usual one), with s = 0 and fidelity 3. Its optimization benchmark (App. A.4.1) varies only s,
at Fr = 0.28 and T = −6.16. Neither is an active-subspace study; this data set varies all 16 inputs
uniformly on the paper's box, as the model page allows.

**Quantity of interest.** R_T, total resistance at model scale, N (UM-Bridge `output[0]`). Also
stored: g1 (beam), g2 (draft), g3, g4 (sonar-dome diameter and height), satisfied when ≤ 0
(App. A.4.1); R_w and R_f (`force-dyn.out`: Rw_pi and Rf); wetted surface S_wet and displaced volume
(`panca.out`), model scale.

**Fidelity.** Seven grids, G1 (finest) … G7 (coarsest), refinement ratio 2^0.25. The paper says the
default is 1 (p. 19); the UM-Bridge server (`config.get("fidelity", 7)`) and its documentation page
say 7, and that is what the UM-Bridge container runs. **The design uses fidelity 7** (the UM-Bridge default)
for cost. Wall times on the shared 2-CPU machine that wrote the files (`reference_runs.csv`): the parent
hull took 2.87 s on G7, 5.06 s on G6, 9.81 s on G5, 30.81 s on G4, 62.20 s on G3 and 432.87 s on G1, so G1
costs **151 times** G7 (`check_accuracy.py`). The paper's normalized costs (1.00 for G1, 0.03 for G7, App.
A.4.1) give about 33; they were measured differently. The paper's "30–35 s per evaluation" (p. 13) is for its
Sec. 5.1 study at fidelity 3, on one physical CPU core of a cloud node, against 62.2 s of wall time here on
a shared machine. The design runs took 2.7–16.1 s of wall time each (median 8.1 s), 3,251 s in all
(original run; the per-run times are in the raw run log, which is not shipped).

The time the code writes to `elapsed_CPUtime.out` (column `cpu_s` of `reference_runs.csv`) is not used:
it is a tenth of the wall time in every run (wall / report = 10.00–10.10, `check_accuracy.py`). L2-Sea's
`src/main.f90` (commit 100da62, lines 23–24 and 73–78) reads `SYSTEM_CLOCK` with a default integer, a wall
clock despite the label "Elapsed CPU time", and divides the count difference by 10000; gfortran's default
clock counts 1000 per second, so the printed figure is wall time / 10. (A compiler whose default clock
counts 10000 per second would print seconds.) How much G7 differs from finer grids is measured in
`fidelity_check.csv` and `reference_runs.csv` (Accuracy).

**Gradients.** None: the code computes none (UM-Bridge: "Gradient … not supported"), and
finite differences would cost 17 runs per gradient (forward; 32 central). The notebook uses the
global quadratic estimator (QPHD), which fits 153 coefficients for m = 16; the 377 runs it uses give
2.46 per coefficient.

**Sampling design (main study).** The first 384 points of a 16-dimensional scrambled Sobol' sequence
(`scipy.stats.qmc.Sobol(d=16, scramble=True, seed=5415)`, of 512 generated),
mapped linearly to the box. The 384 points are three consecutive 128-point blocks of the sequence, i.e. a
union of three 128-point Sobol' nets (rows 0–127, 128–255, 256–383); the first 256 also form a 256-point net. The design was run in five
chunks (2026-10-31 and 2026-11-01; the last extended it from 320 to 384 rows); each run is
deterministic, so the chunking does not affect the values.

**Second study (shape only).** The optimization benchmark's setting
(App. A.4.1, p. 32: "fixed Froude number, namely Fr = 0.28, and nominal draft T = −6.16"; 0.28 and −6.16
are also the nominal values of the model page) with the 14 shape coefficients uniform on [−1, 1]¹⁴: the
first 256 points of a 14-dimensional scrambled Sobol' sequence (`qmc.Sobol(d=14, scramble=True,
seed=14)`, a 256-point net), on G7. QPHD fits 120 coefficients for m = 14, so 256 runs give 2.13 per
coefficient. This is a separate study with its own input space; it is not part of the
16-input design.

**Draft sweeps.** The parent hull (s = 0) at T = −5.70, −5.6975, …, −5.58 m on G7 at
Fr = 0.28 (49 drafts), and at T = −5.70, −5.695, …, −5.58 m on G7 at Fr = 0.36 and on G6 at Fr = 0.28
(25 each), to map the band of silent failures (Accuracy).

**Published results.** No active subspace or sensitivity analysis. Reference values the paper
gives, used in the notebook's comparison: R_T at the parent hull, fidelity 7: 48.9337769 N (UM-Bridge
`test_output.py`, not in the paper); E[R] ≈ 51.664730931098383 N (triangular/beta inputs, fidelity 3,
256-point sparse grid) and ≈ 68.398655639653271 N (Fr and T uniform, fidelity 6, 181 points), App.
A.3.3; the optimized hull of App. A.4.1, R = 36.3116 N at fidelity 1, a 12.5 % reduction; the
normalized costs {1.00, 0.46, 0.20, 0.11, 0.07, 0.04, 0.03} of fidelities 1–7; 30–35 s per
evaluation on one core for the Sec. 5.1 study (p. 13, fidelity 3). Pellegrini et al. (2022, Table 2)
give R at the parent hull on G1–G7: 41.5, 41.8, 42.3, 43.5, 44.4, 45.3, 48.9 N.

## Regenerating

Build L2-Sea (Ubuntu 24.04; any Linux with gfortran, LAPACK and BLAS should do):

```bash
apt-get install gfortran liblapack-dev libblas-dev          # checked: GNU Fortran 13.3.0, LAPACK/BLAS 3.12.0
git clone https://github.com/MAORG-CNR-INM/NATO-AVT-331-L2-Sea-Benchmark.git L2SEA
cd L2SEA && git checkout 100da62 && mkdir -p lib && make     # about 75 s; writes bin/L2-Sea
```

Then, from this directory (one process; each step can be stopped and resumed, runs already in RAW are skipped):

```bash
python generate_data.py design     --l2sea-home L2SEA --raw RAW --stop 384 --max-minutes 19   # repeat until done
python generate_data.py check      --l2sea-home L2SEA --raw RAW
python generate_data.py quadrature --l2sea-home L2SEA --raw RAW
python generate_data.py reference  --l2sea-home L2SEA --raw RAW
python generate_data.py sweep      --l2sea-home L2SEA --raw RAW --max-minutes 17   # repeat until done
python generate_data.py shape      --l2sea-home L2SEA --raw RAW --max-minutes 17   # repeat until done
python generate_data.py g6         --l2sea-home L2SEA --raw RAW --max-minutes 17   # repeat until done
python generate_data.py diag       --l2sea-home L2SEA --raw RAW --max-minutes 17
python generate_data.py assemble   --raw RAW --n 384 --out .
```

`--l2sea-home` may be replaced by the environment variable `L2SEA_HOME`. Without a build of L2-Sea every
step but `assemble` stops with exit status 3 and a message that says what to build and to rerun the script
(checked in `.venv`). `sweep`, `shape`, `g6` and `diag` with `--check K` rerun K evenly spaced stored runs
instead and print the largest relative difference from the stored values; they write nothing to this
directory. `assemble` writes the CSVs into `--out` (default: this directory) and checks that the 41 `diag`
reruns equal `outputs.csv` exactly. L2-Sea is not installed in `.venv`, so none of these steps was run there.

**Provenance of the shipped files** (original run, in a Linux container with Python 3.13; not repeatable
without a build of L2-Sea). L2-Sea commit `100da62ea5ee4647b3312ffb999b2ef22649110a` (2026-03-31), built
with the repository's Makefile (GNU block: `-O3 -fopenmp`, `-llapack -lblas`; GNU Fortran 13.3.0,
LAPACK/BLAS 3.12.0), binary SHA-256 `2fe87cd05df3ad809ccacfe04b3387f44515f52c28a7057ab46fb8346f1537a4`;
example case `examples/DTMB-5415` unchanged except for the five namelist entries set by `l2sea_driver.py`
(igrid, fr, sinkoff, trimoff, ztrasla) and `variables.inp`; `OMP_NUM_THREADS=1`, `ulimit -s unlimited`;
Sobol' seed 5415; runs dated 2026-10-31 and 2026-11-01. **Run time of `generate_data.py`** (wall time, one
process on a shared 2-CPU machine with other jobs running): design 3,251 s, checks 385 s, reference points
1,002 s (two G1 runs of 433 s and 453 s), quadrature 123 s (79 min); draft sweeps 957 s (99 runs), shape
study 1,283 s (256 runs, median 4.5 s), G6 reruns 568 s (120 runs), diagnostic reruns 487 s (41 runs)
(55 min, `nice -n 10`, chunks of at most 17 min); about 2 h 15 min in all. In that run, `--check 3` on
`sweep`, `shape` and `g6` gave 0 (exact reproduction), and rerunning rows 0, 100 and 383 reproduced
`outputs.csv` exactly (relative difference 0).

**In `.venv`** (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, active_subspaces 1.0.0.dev0):
the notebook checks that `inputs.csv` and `shape_inputs.csv` are the stated Sobol' points
(`generate_data.design_points()`, `shape_points()`; exact), and runs in about 50 s (45.0 and 52.3 s in two runs on 2026-11-02, other jobs sharing the machine; their text outputs are identical); `check_accuracy.py` runs in
under 1 s.

## Accuracy

There are no gradients to check; the accuracy questions are the grid and failed solves. The notebook
prints the numbers of this section, except those marked `check_accuracy.py` (recomputed from the shipped
files in `.venv`, under 1 s) and those marked as the original run.

- **Reference values.** The build reproduces UM-Bridge's test value at the parent hull on G7
  (48.9337769 N, all digits), Pellegrini et al.'s Table 2 on G7, G6, G5, G4, G3 and G1 (48.93, 45.33,
  44.41, 43.47, 42.30, 41.50 N against 48.9, 45.3, 44.4, 43.5, 42.3, 41.5 N), the paper's optimized hull
  on G1 (36.3116074 N against 36.3116 N, App. A.4.1) and, within 0.067 %, its E[R] for uniform (Fr, T) on
  G6 (68.4442 N by a 6 × 3 Gauss–Legendre rule against 68.3987 N by a 181-point sparse grid; the error of
  neither rule was estimated). The 18 quadrature runs store only R_T, so the per-run failure rule below was
  not applied to them; their shallowest draft node, −5.683 m, lies 0.033 m below the G6 band, which was
  mapped at Fr 0.28 only (`check_accuracy.py`).
- **Grid.** On five ordinary rows G3 gives 4.7–10.4 % less R_T than G7 (ratio 0.8956–0.9531), almost all
  in R_w (ratio 0.762–0.890); R_f changes by under 1 %. On rows 0–127 (126 valid on both grids) G6 gives on
  average 3.5 % less R_T than G7 (ratio 0.9133–1.0228, mean 0.9650, correlation 0.99818), again mostly in
  R_w. **The active subspace transfers from G7 to G6:** the least-squares directions of the two grids on
  the same 126 rows differ by 1.80°, and a linear model of the grid difference (‖∇d‖ = 1.56 N against
  √λ1 = 50.7 N) changes C by ‖ΔC‖₂ = 91.1, against λ1 − λ2 = 2450, which turns w1 by 1.61° (λ1/λ2 21.2
  instead of 21.6). The check was not repeated on G3 or G1, where the bias is larger.
- **Silent failures in a band of draft (7 of 384 design runs, 1.8 %).** Rows 3, 63, 189, 207, 305, 323,
  343. They were detected by the friction coefficient C_f = R_f / (½ ρ U² S_wet), U = Fr √(g Lpp), against the
  ITTC-1957 line 0.075 / (log10 Re − 2)²: the ratio is a smooth function of the inputs (1.027–1.054 in the
  kept runs), and these seven are 12 to 6,463 robust standard deviations from a linear fit in the 16 inputs
  (the next largest, row 71, is at 6.5; the threshold 8 was set after seeing the data, and thresholds 7–10
  flag the same rows, 4–6 add row 71, 13 drops row 189; |w1 · w1| = 1.0000 in all cases). Row 63 has
  R_f = 354 N, 14 times the flat-plate value; rerun on G7 it gives the same value (the code is
  deterministic), and on G6 and G5 32.8 and 33.1 N. On G6 the six other flagged rows that run change by
  13–91 % in R_T (rows 323 and 343 upward, with R_w ×1.6 and ×2.0), and row 305 fails there too.
  **The failures come from the draft, not from the hull shape:** all seven have T between −5.676 and
  −5.618 m (4.7 % of the T range; probability 6.9e-8 for seven uniform draws), while their shape
  coefficients are typical. The draft sweeps at the parent hull (`draft_sweep.csv`) confirm it: on G7 at
  Fr 0.28, 18 of 49 drafts fail, all with T ∈ [−5.655, −5.610] m (5 raise, 13 return numbers, from
  plausible-looking 49.5 N to 96,838 N or a negative R_f; one valid draft inside, −5.6375 m); at Fr 0.36 two
  drafts fail (−5.620, −5.615 m) and the next (−5.610 m) returns 77.88 N against 72.86 and 69.92 N on either
  side while passing the numerical checks; on G6 at Fr 0.28, 4 of 25 fail (−5.630 to −5.610 m). Across the
  band the wetted surface jumps (−9.9 m²/m against a median of −0.67 m²/m on G7): the waterline crosses a
  line of the coarse hull grid. The band shifts with Fr, grid and hull shape (to −5.676 m among the design
  rows). 30 design rows have T in [−5.70, −5.60] m, so the data have almost no trusted values in that slab;
  leaving out all 30 gives λ1/λ2 = 20.96 and |w1 · w1| = 0.9999. The exclusion of the 7 does not distort
  the design as a whole (centred L2 discrepancy 0.0401 for the kept 377 rows, against 0.0394–0.0400 for
  random 377-row subsets; largest shift of a normalized input mean 0.016). A residual test on R_T would
  have caught only some of them (robust z of rows 189, 323, 343 against the quadratic of the kept rows:
  0.6, −0.3, −2.3).
- **Failure-detection rule** (`l2sea_driver.failure_reason`; the notebook applies it to the shipped
  values, and `generate_data.py` stores every run as printed). A run is not trusted if (1) its draft lies in
  the band of its grid, T ∈ [−5.70, −5.60] m on G7 or [−5.65, −5.60] m on G6 (not mapped for G1–G5); or
  (2) R_f / Rf_ittc (both printed by the code in `force-dyn.out`) differs from 1.0160 + 0.0635 Fr by more
  than 0.015; or (3) the wave-cut wave resistance Rw_wc exceeds 2 R_w; or (4) a resistance component is
  negative. Calibration (`check_accuracy.py`, from the shipped files): a straight-line fit to the 34 valid G7
  reruns of `design_diagnostics.csv` gives 1.01602 + 0.06351 Fr, with residuals from the rule's line of
  −0.0034 to 0.0070; the shape study stays within −0.0058 to 0.0071, and the flagged rows are 0.0216 or more
  away. Rw_wc / R_w is 0.27–1.22 in every run of the shipped files that passes the numerical checks, except
  the Fr 0.36, T −5.610 m sweep run (1.60), which lies inside the draft band; the flagged rows have 4.8 and
  more. The G6 reruns of `fidelity_check.csv` do not store Rw_wc and Rf_ittc; in the original run their
  friction residuals stayed within 0.009 and their Rw_wc / R_w up to 1.79 (external run, not reproducible
  from the repository). On the 41 rerun rows the numerical checks (2)–(4) flag exactly the seven rows of the
  notebook's friction rule (notebook); on the draft sweeps they flag every failed run that returned numbers
  except the Fr 0.36, T −5.610 m run above, which rule (1) catches, and every failed draft lies inside the
  band of its grid (`check_accuracy.py`).
  The rule is a calibrated heuristic for G7 and G6 near the box of this data set, not a guarantee. The values
  are kept in `outputs.csv` and `draft_sweep.csv` unaltered; the notebook leaves the 7 rows out of the main
  study by the friction rule, and applies both rules to the shape study (no run flagged).
- **Hydrostatic term.** `force-dyn.out` also has a term Rh that is not part of R_T and gave 0 or
  2.5e-3 N in identical runs; it is not stored.
- **Geometric constraints.** 70 of 384 hulls violate at least one constraint of the optimization
  benchmark (g1 42, g2 6, g3 31, g4 0); in the shape-only study 52 of 256. They are valid inputs of the model (the paper's box is
  s ∈ [−1, 1]¹⁴) and are kept.

## How these files relate to the paper

- **Not the paper's computation.** The paper's studies vary two inputs (UQ, Sec. 5.1 and A.3.3) or
  fourteen (optimization, A.4.1) and use G3, G6 or adaptive fidelity; none is an active-subspace study and
  none publishes run-level data. These files are a new computation with the same code, inputs and box. The
  shape-only study uses the optimization benchmark's fixed Fr and T and its box for s, with uniform s
  (on G7, where the benchmark lets the optimizer choose the fidelity).
- **Same model as the UM-Bridge container.** `l2sea_driver.py` sets the inputs exactly as UM-Bridge's
  `umbridge-server.py` does (commit `7c6e151` of github.com/UM-Bridge/benchmarks), runs the binary
  directly instead of through Docker (Docker Hub was not reachable), and raises on a failed run where the
  server returns zeros. The reference values above confirm the mapping.
- **Fidelity default.** The paper (p. 19) says the default is 1; the UM-Bridge server and docs say 7.
  These files use 7.
- **Distributions.** The paper's UQ benchmark uses a triangular Fr and a beta draft with α = β = 10 in its
  Eq. 2 (Beta(11, 11) in the usual parametrization); here all
  inputs are uniform (all normalized the same way, to [−1, 1]).

## Unknowns

- The units of T are not stated in the paper; that T is a full-scale length in metres is inferred from the
  code and Pellegrini et al., Table 1 (Model).
- The per-run wall times of the design are in the raw run log, which is not shipped (Model, Fidelity).
- The error of the 6 × 3 Gauss–Legendre rule and of the paper's 181-point sparse grid for E[R] was not
  estimated (Accuracy, Reference values).
- The draft band of silent failures was not mapped for G1–G5 (Accuracy, Failure-detection rule), and the
  transfer of the active subspace was not checked on G3 or G1 (Accuracy, Grid).
- The G6 reruns of `fidelity_check.csv` do not store Rw_wc and Rf_ittc; their values in the original run
  are an external run, not reproducible from the repository (Accuracy).

## License and credit

Code (`generate_data.py`, `l2sea_driver.py`, `check_accuracy.py`, the notebook): MIT (`LICENSE`). Data (the CSVs): CC BY 4.0
(`LICENSE-DATA`); they are outputs of the L2-Sea code, which is GPL-3.0-only (A. Serani and M. Diez,
CNR-INM; github.com/MAORG-CNR-INM/NATO-AVT-331-L2-Sea-Benchmark, doi:10.5281/zenodo.19351757). The
GPL covers the program, not data it computes; no part of the program is redistributed here. The
article is CC BY 4.0. Cite Seelinger et al. (2025), doi:10.1016/j.jcp.2024.113542, and the L2-Sea
benchmark (Pellegrini et al. 2022, doi:10.3390/math10030481; Serani and Diez, zenodo 19351757).
