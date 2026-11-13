# Freyberg_MODFLOW_Forecast: data

Kind: **stored data (generated with MODFLOW-2005 1.12.00), with a regeneration script.** The model runs in
MODFLOW-2005, an outside code (Fortran), driven by `freyberg_driver.py`. The data were generated on
2026-11-10 in a Linux x86-64 container (Python 3.13, numpy 2.5.4, MODFLOW-2005 1.12.00 double-precision build
`mf2005dbl`). The shipped CSVs are the record. **Regenerating the data, or rerunning the accuracy checks,
needs MODFLOW-2005 1.12.00** (`mf2005dbl`); `generate_data.py` and `check_accuracy.py` stop with an error that
says what to install when it is missing. The notebook needs only numpy (it imports the driver's constants,
which read the files in `modflow_files/`, and runs no model).

## Files

| file | contents | role |
|---|---|---|
| `Freyberg_MODFLOW_Forecast.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2020; outputs committed |
| `freyberg_driver.py` | writes the MODFLOW input files from a point, runs `mf2005dbl`, reads heads; `kfield` (kriging, replaces PEST's `fac2real`), `interp` (replaces `targpest`), `run`, `sample`; constants `NAMES`, `LOWER_LOG10`, `UPPER_LOG10`, `X_BEST_LOG10`, `FD_STEP` | the model driver (not a numpy/scipy model); its constants import without MODFLOW |
| `modflow_files/` | nine files copied unchanged from the USGS release, directory `model/Figure5b_pilot_points_flat_bottom`: `freyberg_revisit_pp.bas`, `.dis`, `.wel`, `.riv`, `.oc`, `factors1.dat`, `points1.dat`, `targpest.dat`, `figure5b_from_model_of_table6a_pilot_points_heads_too_highBEST.pst` | model input (public domain) |
| `generate_data.py` | writes the four CSVs (`--out`, `--rows`, `--check`) | regenerates the data (needs MODFLOW-2005) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" (`--quick`: the calibrated model only; `--only 5`, `--only 6`: one check) | accuracy check (needs MODFLOW-2005) |
| `inputs.csv` | 500 × 71: log10 R, log10 K at pilot points KP775–KP844 (log10 of m/d) | Monte Carlo inputs, seed 602988 |
| `outputs.csv` | 500 × 24: forecast heads at fr9c16, fr11c13, fr20c14, fr26c10, fr29c6, fr34c12 (m); calibration heads at the 16 targets (m); `n_dry`, `n_dry_stencil_max` (cells) | model output |
| `gradients.csv` | 500 × 426: ∂h/∂log10 x_j (m per log10 unit) for the six forecasts, 71 columns per forecast in that order | model output |
| `calibrated.csv` | 23 × 5: the release's calibrated model (16 calibration heads, river flux, 6 forecasts): observed or true value, PEST weight, model value, linear prior sd, linear posterior sd | model output at the calibrated point |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Missing values are `nan`: a head next to a dry cell, and every output of a
sample where MODFLOW did not converge (its two dry-cell counts are then −1). Inputs are rounded to the written
digits before the model runs, so each output belongs exactly to the stored input row.

## Model

### Citation

R. J. Hunt, M. N. Fienen and J. T. White, "Revisiting 'An Exercise in Groundwater Model Calibration and
Prediction' After 30 Years: Insights and New Directions", *Groundwater* 58(2) (2020) 168–182,
[doi:10.1111/gwat.12907](https://doi.org/10.1111/gwat.12907) (publisher's open-access version read). Model files: R. J. Hunt, M. J. Haserodt,
M. N. Fienen and J. T. White, MODFLOW-2005 models for revisiting "An Exercise in Groundwater Model
Calibration and Prediction", U.S. Geological Survey data release, doi:10.5066/P96A7ZC9, `model.zip`
(137 689 166 bytes, sha256 9009ae19a801405c889c3793d50df17861d95c15fd44d2a394fb1791dace660b; not shipped:
the nine files above are copied from it unchanged). The exercise itself is D. L. Freyberg, An exercise in
ground-water model calibration and prediction, *Ground Water* 26(3) (1988) 350–360,
doi:10.1111/j.1745-6584.1988.tb00399.x.

### Physics and solver

- The Freyberg (1988) synthetic aquifer as reconstructed by the paper: one layer, steady state, water table,
  40 × 20 cells of 250 m (705 active), no-flow north, east and west, specified head on the south, a river
  in column 15, six pumping wells, uniform recharge R (p. 169, Figure 1). The release's model
  `Figure5b_pilot_points_flat_bottom` is the pilot-point model of Table 6a (p. 174) and Figure 5b (p. 179):
  base elevation 3.3409 m, except at the 22 well and target cells, where the given elevations are kept
  (the paper: "Base elevation = 3.34 fixed", Table 6; the base is the fixed PEST parameter `bottom_elev`,
  control file line 21).
- K: 70 pilot points one every three cells (p. 174, Figure 2), kriged in log10 K with the release's
  `factors1.dat` (every cell's factors sum to 1 within 5e-8; the release ran PEST's `fac2real`, bounds 1e-10
  and 2e7 m/d from its `fac2real1.in`). `freyberg_driver.kfield` applies the factors.
- LPF with LAYTYP = 1 (convertible): transmissivity K(h − b) depends on the head, so the heads are **not**
  linear in the stresses, as they would be for confined flow. No rewetting: a cell that goes dry stays
  inactive, and a dry well cell stops pumping. Check 2 below measures the nonlinearity in R.
- **Slope kinks from the river package.** RIV leakage is C (stage − h) while a river cell's head is above
  the cell's river bottom and C (stage − bottom) below it, so the heads are continuous but their slopes
  jump wherever a river cell's head crosses its bottom (bottoms 20.0 m in row 1 down to 10.25 m in row 40,
  the release's `.riv`). The layer top is 200 m, so there is no confined/unconfined switch. The kinks lie on
  hypersurfaces (measure zero for C), but a finite-difference step that straddles one gives a one-sided mix
  of two slopes (Accuracy, check 5). The dry-cell region (below) is a further, larger discontinuity: there
  a well stops pumping and the head jumps.
- Two stress periods, both steady state: period 1 the calibration state, period 2 the forecast state with the
  river conductance reduced from 4320 to 43.2 m²/d (the release's river file; "reducing the river conductance
  by 2 orders of magnitude", p. 169).
- **Settings changed from the release**, for finite-difference gradients: PCG HCLOSE 1e-10 m (release 1e-5),
  RCLOSE 1e-6 m³/d (0.1), MXITER 200 (100), ITER1 50 (25); the double-precision build, so the head file is
  double precision; R and K written in free format at full precision (the release writes K as E14.6 through
  `fac2real`). One MODFLOW run takes about 0.05–0.1 s.
- **Quantity of interest.** Heads at the six pumping wells in period 2 (the exercise's forecasts, p. 169;
  names give row and column, Figure 5). The well coordinates are those of the release's `targpest.dat`;
  heads are interpolated bilinearly between the four surrounding cell centres, ignoring cells of zero weight
  (`freyberg_driver.interp`, written for this data set in place of PEST's `targpest`). A head with a dry or
  inactive cell of nonzero weight is `nan`. The notebook studies fr20c14 (row 20, column 14, beside the river).

### Inputs

| inputs | units | nominal (calibrated) | range | source |
|---|---|---|---|---|
| R (PEST `r1`) | m/d | 1.3722522e-4 | 5e-5 to 5e-4, log-transformed | control file `figure5b_from_model_of_table6a_pilot_points_heads_too_highBEST.pst`, "* parameter data", line 20 |
| K at KP775–KP844 (PEST `kpkp775`–`kpkp844`) | m/d | 2.698 to 8.571 (`points1.dat`) | 0.5 to 50, log-transformed | same file, lines 22–91 |

The paper prints no bounds; they are the control file's. The calibrated values are the control file's
(Table 6a, p. 174: R = 1.372 × 10⁻⁴ m/d; mean, median, sd, maximum and minimum of the 70 pilot-point K 5.52,
5.45, 1.35, 8.57, 2.70 m/d, which the control file's values reproduce exactly; `check_accuracy.py`).
The inputs are stored as log10 of m/d; the notebook normalizes each by its own half-range in log10
(`ac.Uniform(LOWER_LOG10, UPPER_LOG10)`).

### Gradients

Forward differences in log10 of each input, step 1e-4 (`freyberg_driver.FD_STEP`), 72 MODFLOW runs per
sample. `gradients.csv` holds ∂h/∂log10 x_j in m per log10 unit (the gradient with respect to ln x_j is that
divided by ln 10). No adjoint is available for this MODFLOW-2005 model.

### Sampling design

Plain Monte Carlo: 500 independent draws, uniform in log10 of each of the 71 inputs between the PEST bounds
(numpy `default_rng`, seed 602988). This design is this data set's choice: it is the prior box of the
calibration. The notebook's analysis is conditional on part of this box (below). The paper samples nothing; its runs are PEST calibration runs (1305 and 1019 for Table 6a and
6b, p. 174) and a linear uncertainty analysis.

Of the 500 samples, MODFLOW did not converge in 3 (rows 152, 387, 459; stored as `nan`, counts −1) at the
tightened closure. In 164 a cell is dry at the sample (`n_dry > 0`; no sample has dry cells only in its
stencil). A forecast is `nan` when a cell next to its well is dry: 120 times at fr9c16, 44 at fr34c12, 13 at
fr11c13, 10 at fr20c14, never at fr26c10 and fr29c6 (notebook). The notebook keeps the 333 rows with
`n_dry_stencil_max == 0`, i.e. no active cell is dry at the sample or at any of its 71 forward-difference
points, a stated rule; no row is removed from the files. Every one of the 164 dropped rows has a `nan`
forecast at one or more wells, so in each a cell of nonzero interpolation weight next to a well (the well
cell or a neighbour) is dry; the rule itself tests every active cell.

**Which part of the box is dropped** (notebook, 497 converged rows). The drop flag correlates with the
normalized inputs log10 K at KP790 (r = −0.54; the pilot point 235 m from fr9c16), log10 R (−0.25), log10 K
at KP833 (−0.23; 260 m from fr34c12) and KP791 (−0.21; 515 m from fr9c16); every other input has |r| ≤ 0.09.
The dropped region is low conductivity next to fr9c16 and fr34c12 and low recharge. Because recharge
dominates w1, the selection tilts the kept density along the active variable: the kept fraction rises from
0.48 for y1 in [−1.2, −0.8) to 0.83 for y1 in [0.8, 1.2), and from 0.45 in the lowest fifth of the log10 R
range to 0.82 in the highest. The 154 dropped rows with a finite fr20c14 head have a lower mean head
(18.02 m) than the kept rows (19.31 m). The notebook's subspace therefore describes the conditional design:
the log-uniform density restricted to the region with no dry cell, which is weighted towards higher
recharge. The eigenvectors barely depend on this (adding back the 154 dropped rows whose dry cells do not
change over the stencil, 487 rows in all, moves the n = 2 subspace by 0.011), but the head range, the
sufficient summary plots, the leave-one-out R² and the Monte Carlo prior sd describe the conditional
distribution, not the full box. The dropped region is a different model (a dry well stops pumping, so the
full-box function jumps there).

### Published results (paper)

- Table 6a/6b (p. 174): pilot-point calibration RMSE 0.000 m, forecast RMSE 0.30 m, R = 1.372 × 10⁻⁴ m/d.
- Figure 5b (p. 179) and text p. 178: linear (FOSM) forecast densities for the six wells from the
  pilot-point flat-bottom model; all six true forecasts within the posterior 95 % interval. No numbers are
  printed; the densities' peaks can be read from the figure (by eye, about 0.25, 0.55, 0.5, 0.55, 0.55 and
  0.2 for fr9c16 … fr34c12).
- No active-subspace or global sensitivity results.

### What the notebook finds (from these files)

- fr20c14 head, 333 kept rows: eigenvalues 7.996e+01, 4.696e+00, 1.252e-01; λ1/λ2 = 17.0, λ2/λ3 = 37.5, then
  2.79 and 1.18. Left-out trace share 6.00e-02 (n = 1) and 4.81e-03 (n = 2); mean bootstrap subspace distances
  0.013, 0.015, 0.135 for n = 1, 2, 3 (500 replicates). The notebook chooses **n = 2**.
- w1 is recharge (component 0.972), w2 the conductivity at the four pilot points around the well (KP810 0.736,
  KP805 0.435, KP809 0.397, KP804 0.264) against recharge (−0.196).
- Polynomial surrogates, leave-one-out R²: 0.9591 and 0.9589 (n = 1, degree 2 and 3), 0.9926 and 0.9959 (n = 2).
- The other wells: λ1/λ2 from 11.6 (fr34c12) to 517.1 (fr26c10), recharge component of w1 0.917 to 0.982.
  Adding back the rows whose dry cells do not change over the stencil (487 rows) moves the fr20c14 subspace by
  0.014 (n = 1) and 0.011 (n = 2).
- The analysis is of the conditional design (no dry cell; 333 of 497 converged rows; see "Sampling
  design"): fr20c14 head 10.69–29.61 m, Monte Carlo prior sd 4.18 m over the kept rows.
- `calibrated.csv`: calibration RMSE 4.0e-05 m, forecast RMSE 0.305 m (Table 6a: 0.000 and 0.30 m); linear
  posterior sd of the six forecasts 1.56, 0.72, 0.77, 0.74, 0.71, 1.85 m (peak densities 0.26, 0.55, 0.52,
  0.54, 0.56, 0.22, close to Figure 5b by eye), all six true forecasts inside the 95 % interval, as the paper
  reports (p. 178). This match depends on the assumed prior (Accuracy, check 6): with a prior sd of a half
  or a sixth of the log10 range instead of a quarter the peak densities are 0.12–0.39 or 0.31–0.76, and with
  a head observation sd of 0.1 m those of fr26c10 and fr29c6 become 0.84 and 0.93. The agreement is a
  consistency check of the assumed prior: it supports, but does not establish, that the paper used the
  bound-based default.

## Regenerating

External run (MODFLOW-2005 is not in the repository's `.venv`):

```bash
python3 -m venv venv && venv/bin/pip install numpy
curl -LO https://github.com/MODFLOW-ORG/executables/releases/latest/download/linux.zip   # mac.zip, win64.zip for other platforms
unzip linux.zip mf2005dbl && export MF2005DBL=$PWD/mf2005dbl
venv/bin/python generate_data.py --out /tmp/freyberg            # all files, resumable by chunks of 25 rows
venv/bin/python generate_data.py --check                          # all rows, in a temporary directory; prints the largest differences
venv/bin/python generate_data.py --out /tmp/freyberg_chk --rows 0:5 --check
venv/bin/python check_accuracy.py
```

The shipped files were written with `linux.zip` of 45 010 855 bytes (sha256
ab17192a3531dec61d4f4eab3da10f618e85d1f792d5edc71306a3e06e0e22c3, dated 2026-05-26; its `code.json` names
MF2005.1_12u, version 1.12.00, the version the paper cites, Harbaugh et al. 2017 in its references, and the
release's readme names). `--out` must be
outside this directory; `--check` without `--out` uses a temporary directory. Without `mf2005dbl` both scripts exit
with status 3 and say what to install. `--check` prints the largest differences from the shipped rows (heads absolute in m, gradients in relative
2-norm per row, `calibrated.csv` relative) and does not judge them; it exits with status 1 only on a structural
mismatch: different shapes, NaN pattern or dry-cell counts, or inputs that differ by more than 1e-8 of their
largest magnitude (they are deterministic draws from the seed). MODFLOW is not installed in `.venv`, so no
differences have been measured on another platform. The full run took about 75 min wall time on one process in a 2-CPU container shared with other jobs
(225–250 s per 25 rows, 72 MODFLOW runs per row; the run was resumed once from its chunks after a container
restart). `--rows 0:5 --check` (with `calibrated.csv`) took 65 s and printed every difference below the
printed precision (0.00e+00) on the platform that wrote the files.

## Accuracy

`check_accuracy.py` printed every number in this section (MODFLOW-2005 1.12.00 `mf2005dbl`, Linux x86-64,
2026-11-10; checks 1–4 4 min 36 s on one core, check 5 about 1.5 min, check 6 about 12 s).

- **The calibrated model** (check 1). Pilot-point K of the control file: mean 5.52, median 5.45, sd 1.35,
  max 8.57, min 2.70 m/d, as Table 6a; the kriged field over the 705 active cells: mean 5.55, median 5.34,
  sd 1.30. RMSE of the 16 calibration heads 4.0e-05 m (Table 6a: 0.000 m), of the six forecasts 0.305 m
  (0.30 m). River flux −3710.30 m³/d against the control file's flux observation −3710.51 m³/d. This checks
  the replacements of `fac2real` and `targpest` and the solver settings together.
- **Linearity** (check 2). At the calibrated point the increments h(2R) − h(R) and h(R) − h(R/2) have ratios
  1.474 (fr34c12) to 1.802 (fr20c14), not 2: the heads are not affine in recharge, because the layer is
  unconfined (transmissivity grows with the head). A confined version would be affine in R and in the
  pumping rates (but not in K).
- **Step** (check 3, fr20c14 at row 1, the first row without a dry cell, a row with no kink near the
  sample; relative 2-norm difference from
  central differences with step 1e-4): central differences with steps 1e-2, 1e-3, 1e-5 differ by 4.8e-03,
  3.8e-04, 1.9e-08; forward differences by 1.0e-03, 2.7e-04 and 1.0e-05, and by about 1e-4 at the shipped
  step 1e-4 (check 4).
- **Shipped gradients** (check 4, rows 1, 2, 3, 7, 9, 10, 11, 14, 15, 16, the first ten with no dry cell):
  against central differences with step 1e-4, the stored forward differences differ by at most 1.1e-04
  (median 9.7e-05) for fr20c14 and at most 8.7e-05 for all six forecasts together. That error, of the order
  of the step, is far below the eigenvalue gaps the notebook uses, at rows where no step straddles a kink.
- **Kinks** (check 5). At rows 30 and 316 the stored forward differences agree with central differences of
  step 1e-3 and 1e-4 alike (6.3e-05 to 1.0e-04 relative 2-norm, all six forecasts). At row 389 they agree
  with step 1e-4 (7.7e-05 to 1.1e-04) but differ from step 1e-3 by 4.1e-02 (fr9c16), 3.5e-02 (fr11c13),
  1.2e-02 (fr20c14), 3.7e-03, 2.9e-03, 2.3e-03 (fr26c10, fr29c6, fr34c12). A line scan in log10 R at row 389
  (21 points over ±2e-3) shows ∂h_fr9c16/∂log10 R at 12.37–12.41 m per log10 unit below an offset of 2e-4
  and 10.87–10.89 above 4e-4: a kink, not curvature. Inside that interval the head of the river cell in
  row 6, column 15 (bottom 18.75 m) crosses its river bottom (head − bottom −1.62e-03 m at 2e-4, 3.63e-04 m
  at 4e-4). The shipped gradient at row 389 is not affected (its forward step 1e-4 stops short of the
  kink). Over the 333 kept rows, a forward step of 1e-4 in log10 R moves a river-cell head across its
  bottom at 1 row (row 99), whose stored R derivative is then a one-sided mix of two slopes; at the kept
  samples the closest river-cell head lies within 1e-3 m of its bottom at 2 rows and within 1e-2 m at 11.
  Crossings caused by the steps in the 70 K inputs were not measured.
- **Prior of the linear analysis** (check 6). Posterior peak densities of fr9c16 … fr34c12 at the calibrated
  point: 0.13, 0.30, 0.28, 0.34, 0.39, 0.12 with a prior sd of log10 range/2; 0.26, 0.55, 0.52, 0.54, 0.56,
  0.22 with /4 (`calibrated.csv`); 0.37, 0.76, 0.72, 0.68, 0.66, 0.31 with /6; 0.27, 0.59, 0.58, 0.84, 0.93,
  0.25 with /4 and a head observation sd of 0.1 m.

## How these files relate to the paper

- The model is the release's own (MODFLOW-2005 1.12.00 and its input files), rerun here; only PEST's
  pre- and post-processors were replaced, and the rerun reproduces Table 6a's calibration and forecast RMSE.
- The paper's question is calibration and linear forecast uncertainty; the 500-sample prior study and its
  active subspace are this data set's own, as is the choice of fr20c14 as the quantity studied.
- `calibrated.csv` repeats the paper's kind of linear analysis with stated assumptions (prior sd of a quarter
  of the log10 range, PEST++'s default, assumed; observation sd 1/weight; regularisation left out); the
  paper's PEST++ settings are not in the release's control file. The Figure 5b comparison therefore checks
  the assumed prior for consistency; it is not an independent match (check 6).

## Open questions (for the authors)

1. Which prior did the PEST++ linear analysis of Figure 5 use (bound-based sd, and with which range
   multiple), and did it include the Tikhonov prior information? Was it done in PEST++ (p. 178) or PEST: the
   release's `run_model_for_Figure5b.bat` runs `pest.exe` on the BEST control file?
2. The release's control file has the observation `flux_obs` = −3710.507 m³/d; is it the net river leakage
   of period 1 (the model here gives −3710.30 m³/d at the calibrated point)?

## Unknowns

- PEST's `targpest` rule for a target next to a dry or inactive cell (here: `nan`); its interpolation for
  interior points was matched (calibration RMSE 4.0e-5 m at the calibrated point).
- The prior of the paper's linear analysis (above).
- The share of forward steps in the 70 K inputs that straddle a river-bottom kink (only the R step was
  counted: 1 of 333 kept rows).

## License and credit

Code (`*.py`, the notebook): MIT. Data (the four CSVs): CC BY 4.0. The files in `modflow_files/` are from a
U.S. Geological Survey data release (public domain, U.S. Government work), copied unchanged; credit
R. J. Hunt, M. J. Haserodt, M. N. Fienen, J. T. White (doi:10.5066/P96A7ZC9). MODFLOW-2005 (A. W. Harbaugh
et al., U.S. Geological Survey) is public domain and is not shipped. Cite the paper (above) and Freyberg
(1988).
