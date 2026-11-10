# Pulmonary_Pulse_Wave_Network: data

Kind: **stored data (generated for this repository with the authors' C++ solver `sor06`), with a
regeneration script.** The model is a 1-D pulse-wave PDE solved by an outside code that is not a
numpy/scipy model, so the shipped CSVs are the record. The solver's repository is licensed
**CC BY-NC 4.0**, so neither its code nor its data files (inflow waveform, stored runs) are part of this data
set: `pulmonary_wk_driver.py` clones the repository at a pinned commit, compiles the solver with g++ in a
directory outside this one, and runs it as a command-line program. The authors' own stored runs (also
CC BY-NC 4.0) were not used; every value here comes from new runs of their solver. The notebook needs only
numpy and active_subspaces, because it reads the shipped files and the driver's constants.

## Files

| file | contents | role |
|---|---|---|
| `Pulmonary_Pulse_Wave_Network.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2025; outputs committed |
| `pulmonary_wk_driver.py` | `build` (git clone at `COMMIT` + g++), `waveforms`, `qois`, `forward_gradient`, `fd_gradient`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `NOMINAL`, unit factors `R_SCALE`, `C_SCALE`, `K_TO_MMHG` | drives the outside solver; its constants import with numpy only |
| `generate_data.py` | writes the three CSVs (`--work`, `--out`, `--rows`, `--check`, `--source`) | regenerates the data (needs git and g++) |
| `check_accuracy.py` | the numbers of "Accuracy" below | accuracy check (needs git and g++) |
| `inputs.csv` | 160 × 9: ln k1, k2, ln k3, Rp1, Rp2, Rd1, Rd2, CT1, CT2 in the solver's units (below) | Latin hypercube, seed 2025130 |
| `outputs.csv` | 160 × 4: p_mean, p_sys, p_dia in mmHg; status | solver output |
| `gradients.csv` | 160 × 27: ∂p_mean/∂x_j, ∂p_sys/∂x_j, ∂p_dia/∂x_j (j in the input order) in mmHg per input unit | forward differences of solver runs |

The notebook's main QoI, the pulse pressure pp = p_sys − p_dia, is not a separate column: it and its gradient
(∂p_sys/∂x_j − ∂p_dia/∂x_j) are computed from the stored columns.

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the 15 digits written before the solver runs, so each
output belongs exactly to the stored input row. `status`: 0 = all forward differences; 1 = at least one
backward difference (forward point outside the box); 2 = the base run failed (outputs and gradients NaN);
3 = a perturbed run failed (gradients NaN). The notebook drops rows with status 2 or 3.

Of the 160 rows, 158 have status 0 and 2 have status 2 (rows 38 and 55: the solver stopped with
"TOO MANY CYCLES": the 100-cycle cap was reached before the cycle-to-cycle tolerance; both have ln k3 within
about 0.25 of its lower bound and large distal resistances). No row needed a backward difference and no perturbed
run failed. Dropping the two rows by the status flag censors the slowest-converging corner of the box.

## Model

### Citation

M. J. Colebank, "Assessing parameter identifiability of a hemodynamics PDE model using spectral surrogates
and dimension reduction", *PLOS Computational Biology* 21(10): e1013553 (2025),
doi:10.1371/journal.pcbi.1013553 (publisher's version read; CC BY). Code: github.com/mjcolebank/Colebank_Identifiability_PDE,
commit `9b2741f89faf1c12127c2a05a5c229be32c9905a` (2025-06-04), directory `WK_Simulator/` (CC BY-NC 4.0,
© M. J. Colebank 2025; solver by M. S. Olufsen, M. U. Qureshi and M. J. Colebank).

### Physics and solver

- 1-D mass and momentum balance (paper Eqs. 1-2, p. 3) in three vessels, MPA, LPA and RPA (p. 4), with a
  power-law velocity profile, γ = 9 (Eq. 3), and the wall law p = (4/3)(Eh/r0)(√(A/A0) − 1) (Eq. 4).
  Vessel lengths 4.30, 2.50, 5.75 cm and radii 1.35, 0.90, 1.10 cm (taper-free), 10 grid points per cm,
  8192 time steps per 0.9 s cycle, ρ = 1.055 g/cm³, μ = 0.049 g/(cm s) (all in the solver, `sor06.c`,
  `sor06.h`). Inflow: the repository's `Qin.dat` (mean 97.22 mL/s; printed by `check_accuracy.py`).
- Outlets: three-element Windkessel on the LPA (1) and RPA (2). The paper's Eq. 5 (p. 4) prints the middle
  term as q (Rp + Rd)/(Rp Rd), which is not dimensionally consistent; Fig. 1 on the same page prints
  q (Rp + Rd)/(Rd CT), the standard form that the solver's boundary routine implements.
- Two-step Lax-Wendroff scheme (p. 4). The solver repeats cycles until the squared difference of the MPA inlet
  pressure between two successive cycles, summed over the 512 printed times, is below 1e-5 (at most 100
  cycles; otherwise it stops with "TOO MANY CYCLES"), then prints one more cycle.
- **Sign of k2.** Eq. 4 prints Eh/r0 = k1 e^(−k2 r0) + k3; the code computes k1 e^(k2 r0) + k3 with
  k2 ∈ [−30, −10] 1/cm. The data use the code's k2.
- **Units.** The paper gives k1, k3 in mmHg, R in mmHg s/mL and CT in mL/mmHg (p. 4). In the code k1 and k3
  are in g/(cm s²) (divided by ρ g Lr to nondimensionalize), and Rp, Rd, CT are passed in the solver's
  nondimensional units: 1 unit of R = ρ g Lr/(1333.22 q) = 0.077628 mmHg s/mL, 1 unit of CT =
  1.28819 mL/mmHg (q = 10 mL/s, Lr = 1 cm). The resistance unit is confirmed by the output: p_mean equals the
  mean inflow times the parallel Windkessel resistance in these units (Accuracy below). The compliance unit
  rests on reading the code (`sor06.c`, `arteries.c`); the mean pressure does not test it. As a cross-check
  of the scale of the code-unit bounds, the stored MPA pressures span 0.67 mmHg (lowest p_dia) to 146.77 mmHg
  (highest p_sys; both printed by the notebook), and the paper's seven Windkessel training waveforms
  (Fig. 3(a), p. 12) span about 20 to 120 mmHg (read from the plot). The shipped inputs are in the
  code's units; `pulmonary_wk_driver.py` holds the factors.
- **Parameter set.** Eq. 6 (p. 4) lists Rp,2 twice; the ninth parameter is Rd,2.

### Inputs

The main text prints no ranges. The bounds are those of the authors' sampling script
`WK_Simulator/sample_3ves.m` (`low`, `upp`), which draws a Latin hypercube with k1 and k3 log-uniform. The
paper says all priors are uniform (p. 5); this data set follows the code, so its inputs are uniform in
(ln k1, k2, ln k3, Rp1, Rp2, Rd1, Rd2, CT1, CT2).

| input | units | lower | upper | physical | source |
|---|---|---|---|---|---|
| ln k1 | ln(g/(cm s²)) | ln 1e6 | ln 1e8 | k1 = 1e6 to 1e8 g/(cm s²) | Eq. 4 (p. 3); bounds `sample_3ves.m` |
| k2 | 1/cm | −30 | −10 | code sign | Eq. 4; `sample_3ves.m` |
| ln k3 | ln(g/(cm s²)) | ln 3e4 | ln 5e5 | k3 = 22.5 to 375 mmHg | Eq. 4; `sample_3ves.m` |
| Rp1, Rp2 | solver units | 0.5 | 10 | 0.0388 to 0.776 mmHg s/mL | Eqs. 5-6 (p. 4); `sample_3ves.m` |
| Rd1, Rd2 | solver units | 0.8 | 20 | 0.0621 to 1.553 mmHg s/mL | Eqs. 5-6; `sample_3ves.m` |
| CT1, CT2 | solver units | 1e-5 | 1.2 | 1.29e-5 to 1.546 mL/mmHg | Eqs. 5-6; `sample_3ves.m` |

### Quantities of interest (this data set's choice)

The paper uses time series (PCA scores of the midpoint signals, p. 6) and defines no scalar QoI. This data
set takes the MPA midpoint pressure over the printed cycle (512 points) and records its time average
p_mean, its maximum p_sys and its minimum p_dia, in mmHg. The notebook's main QoI is the pulse pressure
pp = p_sys − p_dia, computed from the stored values; p_sys is the second QoI, and p_mean is a structural
control (it equals the mean inflow times the parallel outlet resistance, so it does not depend on the wave
dynamics). p_sys and p_dia are the maximum and minimum over the 512 printed points, so they are piecewise
smooth in the inputs where the time of the maximum or minimum switches between printed points; the
finite-difference checks below show no sign of this at the step used.

### Gradients

Forward differences in the code-unit inputs (ln k1 and ln k3 for the stiffness coefficients), step 1e-3 of
each input's half-range, backward where the forward point leaves the box; 9 extra solver runs per row. The
step was chosen from a central-difference study at steps 1e-2, 1e-3 and 1e-4 (see Accuracy). The gradient of
the pulse pressure is the difference of the stored p_sys and p_dia gradients.

### Sampling design

A 160-point Latin hypercube on the box above (one point per stratum in each column; numpy `default_rng(2025130)`,
a random permutation per column plus a uniform jitter in each stratum). The paper used 2500 Latin hypercube
samples for its Windkessel surrogate, of which 2470 ran (p. 10; the sampler is named only in the code); its
seed is not reported. These are not the paper's samples.

### Published results

No active subspace analysis. Sobol' indices of the PCA scores (Fig. 5, p. 14) and point-wise Sobol' indices
(Fig. 6, p. 15) with text on pp. 11-12; univariate profile-likelihood identifiability for designs D1-D3
(Fig. 8, text p. 19). Table 1 (p. 13): PC1 of MPA pressure holds 82.2 % of its variance.

### Results in the notebook

The pulse pressure p_sys − p_dia has a one-dimensional active subspace (λ1/λ2 = 9.4, then 1.7; LOO R² 0.9147
for a quadratic in one variable) led by the wall-stiffness offset ln k3, with smaller weights on the proximal
resistances and the compliances. The systolic pressure is close to one-dimensional (λ1/λ2 = 6.6, then 2.0)
along a direction that mixes ln k3 with all four resistances; the diastolic pressure needs two directions
(ratios 4.7, 6.1). The mean pressure is two-dimensional to within 1e-6 of λ1 (λ3/λ1 = 7.1e-7, the solver's
noise level): it equals the mean inflow times the parallel outlet resistance, so it depends only on Rp1 + Rd1
and Rp2 + Rd2. k1 and k2 have no visible effect on any QoI in this box. All numbers are printed in the
notebook.

## Regenerating

```
python generate_data.py --work /tmp/pulmwk --out /tmp/pulmwk/out            # all rows
python generate_data.py --work /tmp/pulmwk --rows 0:5 --out /tmp/pulmwk/chk --check
python check_accuracy.py --work /tmp/pulmwk --rows 0:5
```

This is an external run: it needs git and g++ (the shipped files were written with g++ 13.3.0, Ubuntu 24.04,
Linux x86-64, on 2026-11-10) and network access to GitHub. Recipe: (1) `git clone
https://github.com/mjcolebank/Colebank_Identifiability_PDE.git` into the work directory and check out commit
`9b2741f89faf1c12127c2a05a5c229be32c9905a` (or pass `--source` with an existing clone); (2) copy
`WK_Simulator/*.c, *.h, Qin.dat, connectivity.txt` into `<work>/bin` and compile each source with
`g++ -c -O2 -D_REENTRANT`, then link `g++ -O2 -o sor06 tools.o junction.o sor06.o arteries.o -lm` (the
repository's Makefile names a Windows compiler path, so the driver compiles directly); (3) run the rows,
each `sor06 k1 k2 k3 Rp1 Rp2 Rd1 Rd2 CT1 CT2 ID` in its own temporary directory, which writes
`output_ID.2d` (512 times × 9 locations: proximal, midpoint, distal of MPA, LPA, RPA; columns t, x, p, q, A,
wave speed). Without g++ or git the script stops with a message that names them. `--out` is required;
`--check` and `--rows` refuse an `--out` that is this directory or inside it, and `--work` must be outside
it (symbolic links are resolved). Finished rows are cached in `<work>/rows.jsonl`, so an interrupted run
resumes.

Run time: the full design (160 rows, 1600 solver runs) took 4057 s of solver CPU (2.5 CPU-s per run,
25 CPU-s per row on average, 95 CPU-s for the slowest row) and 8267 s of wall time on a shared 2-CPU
container, as printed by `generate_data.py`. Tolerance of `--check`: inputs and outputs 1e-9 relative
(max |new − shipped| / max |shipped|), gradients 1e-6 relative per QoI block, and the same failed rows.
`generate_data.py --rows 0:3 --check` on the generating machine printed differences of 0.00e+00 for all three
files (below the written digits; 90 CPU-s, 112 s wall). Another compiler or platform can change the last
digits of each run, which the differences amplify.

## Accuracy

All numbers in this section are printed by `check_accuracy.py --rows 0:5` (about 8 CPU-minutes), except
where another script is named.

- **Gradients.** On rows 0-4, the shipped forward differences agree with central differences at the same step
  (1e-3 of each half-range) to 2.4e-4 to 2.9e-3 relative (‖g_fwd − g_c‖/‖g_c‖ in normalized coordinates, per
  QoI; worst 2.5e-3 for p_mean, 2.4e-3 for p_sys, 2.9e-3 for p_dia, all on row 2). For the pulse pressure
  (gradient ∂p_sys − ∂p_dia) the agreement is 2.3e-4 to 1.4e-3 (worst on row 2). Central differences at
  steps 1e-4 and 1e-3 agree to 7.4e-8 to 1.1e-6 on rows 0, 1 and 3, and to 5.7e-4 (p_sys, row 4) and
  2.3e-3 to 2.5e-3 (row 2, which has a small LPA compliance, CT1 = 0.0019, and converges slowly): below a step
  of about 1e-4 the solver's cycle-convergence tolerance adds noise of order 1e-3 relative, so the step 1e-3
  is used. A step study at one random point before the data were generated (not a shipped row; external run,
  not reproducible from the repository) gave the same picture: central differences at 1e-2, 1e-3, 1e-4 within 7e-4 of each other,
  forward at 1e-3 within 3e-4 to 4e-4 of central.
- **Base runs.** Rerunning the base point of rows 0-4 reproduces the stored outputs to 4.2e-10 mmHg (the written
  digits).
- **Units and mean pressure.** The mean inflow of `Qin.dat` is 97.2222 mL/s. Over the 158 good rows, p_mean
  differs from Qbar × (R_LPA ∥ R_RPA), with R = (Rp + Rd) × 0.077628 mmHg s/mL, by −0.0009 to +0.0009 relative
  (median −0.0001). This confirms the solver's nondimensional resistance unit. The notebook prints the
  matching regression (slope 97.23 mL/s, correlation 1.0000).
- **Estimator check.** The notebook compares the gradient-based subspaces with the quadratic-model (QPHD)
  estimates from the values alone: for the pulse pressure, first eigenvalue ratio 21.6 and distance 0.118 at
  n = 1; for p_mean, distances 0.024 (n = 1) and 0.054 (n = 2).

## How these files relate to the paper

The files are new runs of the paper's Windkessel model, with the authors' solver at the commit above, on the
authors' parameter bounds, but with a different (smaller) Latin hypercube, a scalar QoI chosen here, and
finite-difference gradients the paper does not compute. The paper's 2470 Windkessel training runs are not in
its repository, and its repository's stored runs are CC BY-NC 4.0, so none of them is used or shipped.

## License and credit

The CSVs are CC BY 4.0 and the scripts MIT, like the rest of the repository. They contain only new solver
output and the input design; no file from the authors' repository is shipped. The solver itself
(CC BY-NC 4.0, © M. J. Colebank 2025; Olufsen, Qureshi, Colebank) is fetched and run as an outside tool and
is not redistributed; anyone who regenerates the data accepts its licence, which allows non-commercial use
with attribution to the paper above. Cite the paper when using this data set.

## Unknowns and open questions

- **Ranges.** The paper's main text prints no parameter ranges; they come from the authors' code
  (`sample_3ves.m`), assumed to be those of the paper's 2500-sample design. The S1 File was not read.
- **Prior on k1, k3.** The paper says uniform priors (p. 5); the code samples k1 and k3 log-uniformly. The
  data follow the code.
- **Units.** The paper's units for k (mmHg), R (mmHg s/mL) and CT (mL/mmHg) (p. 4) disagree with the code, where
  k is in g/(cm s²) and R, CT are nondimensional; the code was followed. The R conversion is confirmed by the
  mean-pressure identity (Accuracy); the CT conversion rests on reading the code.
  Whether the paper's figures use the code's units is not stated.
- **Eq. 5 and Eq. 6.** Eq. 5 prints q (Rp + Rd)/(Rp Rd) for a term that Fig. 1 prints as q (Rp + Rd)/(Rd CT)
  (the dimensionally consistent form); Eq. 6 lists Rp,2 twice and omits Rd,2. Sign of k2: Eq. 4 has e^(−k2 r0),
  the code e^(k2 r0) with negative k2.
- **QoI.** Mean, maximum and minimum MPA midpoint pressure are this data set's choice; the paper analyses
  time series through PCA.
- **Failures.** The paper reports 30 failed Windkessel runs of 2500 (p. 10) without saying which; here 2 of 160
  failed (the 100-cycle cap), both with soft walls and large distal resistances.
- Open questions for the author (not contacted): the ranges and log-scales used for the published design and
  whether they are in S1 File; the intended units of Eq. 4-6 parameters; which 30 runs failed and why.
