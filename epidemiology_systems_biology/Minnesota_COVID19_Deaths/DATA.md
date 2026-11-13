# Minnesota_COVID19_Deaths: data

Kind: **stored data (generated with R 4.4.3 running the article's R model), with a regeneration script.** The model
is the University of Minnesota COVID-19 model in the version shipped with the supplement of Wycoff, Binois and Wild
(2021), an R code under the GNU GPL 3.0. It is not part of this repository and was not ported: a numpy translation
of GPL code could not be shipped under the repository's MIT licence. `mn_covid_eval.R` and `mn_covid_driver.py`
(both written for this data set, MIT) only call the model's own function `covid_sim()`. The data were generated
in a Linux container with R installed, and the shipped CSVs are the record. **Regenerating the data, or rerunning the
accuracy checks, needs R (4.4.3 was used) with the package plyr (1.8.9), and the folder `code` of the article's
supplement `code.zip`** (below); `generate_data.py` and `check_accuracy.py` stop with an error that says what to
install or download when either is missing. The notebook needs only numpy, matplotlib and active_subspaces, because
it reads the shipped files and the driver's constants.

## Files

| file | contents | role |
|---|---|---|
| `Minnesota_COVID19_Deaths.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2021; outputs committed |
| `mn_covid_driver.py` | `NAMES`, `SHORT`, `DESCRIPTIONS`, `UNITS`, `NOMINAL`, `LOWER`, `UPPER`, `INACTIVE`, `FD_STEP`; `find_code`, `find_rscript`, `run_batch`, `deaths`, `fd_points`, `fd_assemble`, `deaths_and_grad` | driver of the R model (not a numpy/scipy model); its constants import without R |
| `mn_covid_eval.R` | evaluates `covid_sim(x)` at the rows of a CSV file and appends `row,deaths` lines (resumable) | the only R code in the repository; calls the supplement's code unchanged |
| `generate_data.py` | writes the three CSVs (`--code`, `--out` (required, outside this directory), `--rows`, `--chunk`, `--check`, `--rscript`) | regenerates the data (needs R and the supplement) |
| `check_accuracy.py` | prints the numbers of "Accuracy" below and the article's reference matrix | accuracy check (needs R and the supplement) |
| `inputs.csv` | 100 × 8: the inputs x1..x8 of `covid_sim`, fractions | Monte Carlo inputs, seed 2021 |
| `outputs.csv` | 100 × 1: cumulative deaths at day 365 (persons) | model output |
| `gradients.csv` | 100 × 8: ∂deaths/∂x_j (persons per unit of x_j), forward differences, step 1e-7 | model output |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the written digits (16 significant digits) before the
model runs, so each output belongs exactly to its stored input row.

## Model

### Citation

N. Wycoff, M. Binois and S. M. Wild, "Sequential learning of active subspaces", *Journal of Computational and
Graphical Statistics* 30(4) (2021) 1224–1237, [doi:10.1080/10618600.2021.1874962](https://doi.org/10.1080/10618600.2021.1874962) (Crossref-checked). The PDF read is
the author manuscript (HAL hal-02367750, the arXiv:1907.11572v2 text of 20 September 2020, CC BY 4.0); page numbers
below are that manuscript's printed page numbers. The example is Sec. 4.3, p. 24, with Fig. 5 (right, p. 25) and
Fig. 6 (p. 26). Supplement: appendices and `code.zip`, figshare, doi:10.6084/m9.figshare.13611284; `code.zip`
(15,250,535 bytes, sha256 585279ae0eaf56012b526b106b6897901eeefc836d6088a8c803e1b171f8e081, downloaded
2026-11-09).

The model: E. A. Enns, M. Kirkeide, A. Mehta, R. MacLehose, G. S. Knowlton, M. K. Smith, K. M. Searle, R. Zhao,
S. Gildemeister, A. Simon, E. Sanstead and S. Kulasingam, "Modeling the impact of social distancing measures on the
spread of SARS-CoV-2 in Minnesota", Tech. Rep. 1148-427724, University of Minnesota (2020) (the article's reference).
The version used is `code/helper/covid_allinone.R` of the supplement (sha256
81b4d921a7796e90b5417ddd0e4fba27d5005d8a13ba45f6d9cc60ef2f245c39; header "© 2020 Regents of the University of
Minnesota. This software is licensed via the GNU General Public License 3.0."), a single-file adaptation of the
consortium's model with the function `covid_sim()` added; the article's own scripts (`covid_sim.R`,
`covid_est_C.R`, …) carry a BSD-3-Clause notice by Wycoff, Binois and Wild. Which release of the consortium's
model it was adapted from is not stated (open question 1).

### Physics and solver

- An age- (9 groups) and comorbidity- (2 groups) structured compartmental model with compartments S, E1–E2,
  A1–A3 (asymptomatic or mild), I1–I3 (symptomatic), H, ICU, R, D (and the counters HD, home deaths, and CH,
  cumulative hospitalizations), advanced by explicit steps of 0.05 day from t = 1 to t = 365 (7280 steps) with
  per-step transition probabilities 1 − exp(−r Δt). The notebook's "The model" section gives the equations as the
  code computes them; constants (β = 0.0295, incubation 5.2 days, infectious period 7.8 days, 11 days in hospital,
  8 in ICU, 2,200 ICU beds, the Minnesota contact matrix and population) are the code's.
- The simulator of the article is the code's "scenario 2" (`covid_sim`): social distancing for 1 ≤ t < 41 and
  shelter in place for 6 ≤ t < 20 (comments in the code: day 1 = 22 March 2020). Targeted 60+ distancing and lasting
  behaviour change have start time ∞, so inputs x4 and x5 are never read: the output does not depend on them, and
  their gradient entries are exactly 0 at every sample.
- ICU overflow (more than 2,200 occupied beds) makes the dynamics piecewise smooth: the fraction of ICU patients
  without a bed is max(0, 1 − 2200/ICU), whose derivative jumps where the demand crosses capacity. The QoI is
  continuous, and its derivative is continuous except for small jumps where a time step's ICU demand crosses
  capacity.
- **Quantity of interest:** cumulative deaths at t = 365, the sum of the D compartments (D includes the deaths at
  home), as returned by `covid_sim`. The article calls it "the total covid-19 deaths" (p. 24).
- One run takes about 3 s of CPU in R 4.4.3 (about 3 min of CPU for the 58 runs of `check_accuracy.py`); the article's script `covid_est_C.R` states 7 s per run.

### Inputs

All inputs are independent and uniform on [0.01, 0.99] and are stored in the code's units (fractions). The notebook
normalizes once, through `ac.Uniform(LOWER, UPPER)`, which scales every input by its half-range 0.49.

| x | code name | meaning | code default | range | source |
|---|---|---|---|---|---|
| 1 | `init_cases_detected` | fraction of cases detected at day 0; undetected initial infections 233 (1 − x1)/x1 | 0.02143008 | [0.01, 0.99] | `covid_allinone.R`; footnote 4, p. 24 ("the number of initial cases") |
| 2 | `sip_contact_reduction` | contact reduction, shelter in place | 0.5511887 | [0.01, 0.99] | as above ("contact reduction of shelter in place and social distancing") |
| 3 | `social_distancing_contact_reduction` | contact reduction, social distancing | 0.3755247 | [0.01, 0.99] | as above |
| 4 | `sixty_plus_contact_reduction` | contact reduction of people aged 60+ (not active in scenario 2) | 0.5 | [0.01, 0.99] | as above ("the proportion by which contact is reduced in persons aged 60 or more") |
| 5 | `behavior_change_contact_reduction` | contact reduction from behaviour change (not active in scenario 2) | 0.1 | [0.01, 0.99] | as above |
| 6 | `prop_asymptomatic` | proportion of infections asymptomatic or mild | 0.4116304 | [0.01, 0.99] | as above |
| 7 | `p_h_80` | P(hospitalized) for symptomatic people aged 80+; scales every age group | 0.1032184 | [0.01, 0.99] | as above |
| 8 | `p_dying_home_80` | P(death at home) for symptomatic, non-hospitalized people aged 70+ (the code's meaning; footnote 4 says "over 80") | 0.1392806 | [0.01, 0.99] | as above |

- **Ranges.** The article states none. Its study script `code/covid_sim.R` (the runs of Fig. 6) passes
  x = 0.98 u + 0.01 for u in the unit cube, so the box is [0.01, 0.99]^8, which this data set uses. Its reference
  run `code/covid_est_C.R` (the 2,000 runs behind Fig. 5 and the "ground truth" of Fig. 6) draws x uniformly on
  [0, 1]^8 without that map. The box is wide: it is the whole probability scale of every input, not a range
  around the calibrated values (x1's default, 0.0214, is near its lower end).
- **Input 8.** Footnote 4 (p. 24) calls it "the probability of a person over 80 dying at home"; the code applies
  it to the age groups 70–79 and 80+ (`covid_allinone.R`, line 613, whose comment says so). This data set uses the
  code's meaning.
- **Seven or eight inputs.** Footnote 4 (p. 24) lists seven items, naming the shelter-in-place and
  social-distancing reductions in one item; the code has eight inputs, and Fig. 5 (right) has eight eigenvalues.
- Checks in the notebook: normalized inputs fill [−1, 1] with kurtosis near 1.8 (uniform), no duplicate rows, and
  every value is a valid fraction by construction.

### Gradients

Forward differences in the code's units with step h = 1e-7 (`FD_STEP`): 8 extra runs per sample, 900 runs in all.
`gradients.csv` holds ∂deaths/∂x_j in persons per unit of x_j (physical coordinates); the normalized gradient is
0.49 ∂deaths/∂x_j. The model is a deterministic double-precision recursion, so the differences are not noisy at this
step; a step study at one point (not shipped; x = (0.3, 0.5, 0.4, 0.6, 0.2, 0.45, 0.35, 0.25)) showed the forward
differences converging as h decreases from 1e-3 to 1e-7. At h = 1e-7 they agree with a central difference with
h = 1e-4 to 2e-8 relative in the large components (x6, x7, x8, about 2e5 persons per unit) and to 6e-6 relative in
the small ones (x1, x2, x3, about 80 to 400 persons per unit), while forward differences with h = 1e-4 were off by up
to 3.2 % in the small components (x3) (external run, not reproducible from the repository). The article states that the output "is not
sufficiently precise so as to allow finite differencing" (p. 24); with this code and these steps we did not find that
(Accuracy below). The gradient is one-sided where the ICU-overflow kink lies within h of a sample.

### Sampling design

Plain Monte Carlo: M = 100 independent uniform draws on [0.01, 0.99]^8 with numpy's `default_rng(2021)`. The article
used 2,000 random uniform samples on [0, 1]^8 for its reference GP (`covid_est_C.R`, `runif`, no seed in that script)
and, for the sequential experiments of Fig. 6, a Latin hypercube of 5m = 40 points followed by 60 sequentially chosen
points, 40 replicates (`covid_sim.R` sets `set.seed(123)`; the stored results in `code/data/covid60.RData` hold 40
replicates).

### Published results (paper)

- Fig. 5 (right, p. 25): the log eigenvalues of the GP-based C from 2,000 runs; the caption says "We chose active
  subspace dimensions of 1 and 3, respectively" (wing weight and Covid-19), and the text (p. 24) says Fig. 5 "reveals
  that a three dimensional active subspace is reasonable". The supplement holds the matrix (`covid_Ch.RData`,
  object `Cbmat`); `check_accuracy.py` reads it with R: eigenvalues 1.6171e11, 1.7106e10, 3.9263e9, 4.7998e7, 8.0015e6, 7.2493e6, 6.9984e6, 5.2562e6, natural logs 25.809, 23.563, 22.091, 17.687, 15.895, 15.796, 15.761, 15.475 (the points of Fig. 5, right), ratios
  λ1/λ2 = 9.45, λ2/λ3 = 4.36, λ3/λ4 = 81.8. Its first three eigenvectors lie in the span of x6, x7, x8 (entries on
  x1–x5 below 0.01).
- Fig. 6 (p. 26): subspace error (Eq. (7), p. 21: the sine of the largest principal angle to the 3-D subspace of the
  2,000-run GP) against the number of runs for the Trace, Var1, Var2 and random acquisition rules (left, mean of 40
  replicates; right, box plots of the final error after 100 runs). The two panels use different logarithms: the
  left plots `log10` of the mean error (`helper/plot_func.R`, `plot_errs`, line 34; it ends at about −0.85 to −1.0,
  i.e. errors of 0.10–0.14), the right the natural `log` (`covid_boxplot.R`, line 28; medians about −2.0 to −2.3,
  i.e. 0.10–0.13, whiskers roughly −3 to −1.5). The notebook compares its natural-log distance with the right
  panel. No eigenvector values are reported in the article.

### What the notebook finds (from these files)

- Eigenvalues (normalized coordinates): 3.955e10, 4.376e9, 1.310e9, 3.72e6, 4.45e4, 1.35e4; C has a
  two-dimensional null space (inputs 4 and 5, whose gradient columns are zero), and the two computed eigenvalues
  there are at rounding level; λ1/λ2 = 9.04 [7.08, 12], λ2/λ3 = 3.34 [1.78, 6.65], λ3/λ4 = 352 [117, 3840] (95 % bootstrap, 500
  replicates). The library's gap rule picks n = 1; the notebook uses **n = 3** (largest ratio λ3/λ4, bootstrap subspace
  distance 0.00616 for n = 3 against 0.0385 and 0.0542 for n = 1 and 2, leave-one-out R² 0.99989 for a cubic in three
  active variables against 0.98633 for a quadratic in two).
- The three leading eigenvectors lie in the span of x6, x7, x8 (entries on x1–x3 at most 0.005); normalized
  activity scores 0.6911 (x6), 0.1624 (x7), 0.1464 (x8). w4 is almost x1 alone.
- Against the article's reference matrix: subspace distance 0.0152 for n = 3; |w_k · w_k,paper| = 0.9994, 0.9985,
  0.9985; λ1..λ4 equal to 1.019, 1.065, 1.389 and 0.323 times the article's eigenvalues scaled by 0.49².
- Signs: `ac.orient(by="reference")` to the article's reference eigenvectors, each with its largest entry positive;
  deaths decrease along w1 (correlation −0.9359).

## Regenerating

```bash
# R environment (conda-forge; any R >= 4.0 with plyr should do; 4.4.3 and plyr 1.8.9 wrote the shipped files)
micromamba create -p ./renv -c conda-forge r-base=4.4.3 r-plyr=1.8.9
# the model: download code.zip from https://doi.org/10.6084/m9.figshare.13611284 and unzip it (folder "code")
python generate_data.py --code path/to/code --rscript ./renv/bin/Rscript --out /tmp/mn            # all files
python generate_data.py --code path/to/code --rscript ./renv/bin/Rscript --out /tmp/mn --chunk 60 # resumable chunks
python generate_data.py --code path/to/code --rscript ./renv/bin/Rscript --rows 0:3 --check     # 3 samples, in a temporary directory
python generate_data.py --code path/to/code --rscript ./renv/bin/Rscript --check                # all 100 samples (900 runs)
python check_accuracy.py --code path/to/code --rscript ./renv/bin/Rscript                         # Accuracy numbers
```

`--out` is required without `--check` (`--check` without `--out` writes into a temporary directory and removes it) and
must lie outside this directory: in every mode (full run, `--chunk`, `--rows`, `--check`)
`generate_data.py` refuses this directory and any folder inside it (symlinks resolved), so the shipped files are
never overwritten. `--check` prints, for each column, the largest |new − shipped| over the largest |shipped| value of the same column among the checked rows (absolute for the all-zero gradient columns of inputs 4 and 5), so the small gradient components of x1–x3 are reported too, and does not judge it; it exits with status 1 only for a structural mismatch: a different shape or NaN pattern, regenerated inputs that differ from the shipped ones by more than a relative 1e-8, or a run (`--chunk`) that stops before the rows it was asked to check are done. `--rows 0:1 --check` (9 runs, 50 s) printed difference 0 in every column on the platform below.

Provenance of the shipped files: generated 2026-11-10 in a Linux container (x86-64, 2 cores shared with other jobs),
R 4.4.3 and plyr 1.8.9 from conda-forge, Python 3.11.18, numpy 2.4.6, with `generate_data.py` as shipped, run in
chunks of 54 model runs (`--chunk 54`) in one R process at a time. The run was interrupted once by a container restart after 318 of the 900 model runs and resumed from its run log (`evals.csv`, which `generate_data.py` keeps in `--out`). The 16 completed chunks took 6506 s of wall time; the interrupted chunk is not counted, so the total was about 2 h of wall time on the loaded machine (load average 5 to 7), or about 45 min of CPU at about 3 s of CPU per run. There is no randomness after the seeded
draw, and the R model is deterministic: `generate_data.py --rows 0:2 --check` (18 runs, 164 s of wall time and 57 s of CPU) reproduced the first two rows of all three files with maximum relative difference 0 on the same platform. Other platforms or R versions may differ at the level of rounding, which the step 1e-7 amplifies in the gradients by up to about 1e7.

## Accuracy

`check_accuracy.py` prints every number in this list (it needs R and the supplement; in this container it ran 58 model runs in 7 min 11 s of wall time, about 3 min of CPU).

- **The driver against the article's own runs** (`check_accuracy.py`, item 1). At the first 5 design points of the
  first stored sequential run in `code/data/covid60.RData`, mapped to [0.01, 0.99] as in `covid_sim.R`, the driver
  returns the stored deaths exactly (maximum relative difference 0).
- **Gradients** (item 2): relative 2-norm error of the shipped forward differences against central differences
  with step 1e-6, in normalized coordinates: 2.9e-8, 6.1e-8 and 1.5e-8 at rows 0, 1, 2; the largest component
  error is 5.7e-8 of the gradient norm.
- **Inputs 4 and 5** (item 3): setting both to 0.01 and to 0.99 at row 0 changes the deaths by exactly 0; rerunning
  rows 0–2 reproduces `outputs.csv` exactly.
- **The article's reference matrix** (item 4): the eigenvalues, logs, ratios and eigenvectors quoted under
  "Published results"; the notebook uses them as constants.
- In the notebook, the least-squares slope of the deaths on the normalized inputs and the mean normalized gradient
  have cosine 0.9905, which confirms the gradient coordinates.

## How these files relate to the paper

- **The article's model, run anew.** The deaths are those of the article's own simulator: the driver reproduces the
  article's stored runs in `covid60.RData` exactly at the 5 points checked (Accuracy). The data set's design differs from the article's: 100 Monte
  Carlo samples with gradients on [0.01, 0.99]^8, against the article's 2,000 gradient-free samples on [0, 1]^8 and
  a GP estimate of C. The notebook compares eigenvalue ratios and subspaces with the article's reference matrix.
- The article did not use gradients ("not sufficiently precise so as to allow finite differencing", p. 24); this
  data set does, after the checks above.

## Open questions (for the authors)

1. Which release of the Minnesota COVID-19 model (the consortium's versions 2 or 3, or another) `covid_allinone.R`
   was adapted from, and whether "scenario 2" with its fixed policy dates is the scenario of the consortium's report.
2. What led to the statement that the output is too imprecise for finite differencing: in this code, forward
   differences with steps 1e-5 to 1e-7 agree with central differences (Accuracy).
3. Whether the reference GP was meant to use the unit cube (`covid_est_C.R`) while the sequential runs used
   [0.01, 0.99]^8 (`covid_sim.R`).
4. Fig. 6 says 40 runs; `covid_sim.R` sets `times <- 35`, while `covid60.RData` holds 40 replicates. The same
   script sets `n <- 5 * nvar` (40 initial points) and `f_evals <- 80` (lines 29, 43), i.e. 120 runs per replicate,
   while Fig. 6 ends at 100 runs and the stored file is named `covid60`: the stored results may come from other
   settings than the shipped script.
5. In `get_initial_conditions` (`covid_allinone.R`), the undetected symptomatic and exposed cases of the comorbidity
   group are added to the compartments of the group without comorbidities (the second assignment uses `ic == 0`
   again). The data use the code as written.

## License and credit

Code (`*.py`, `mn_covid_eval.R`, the notebook): MIT. Data (the CSVs): CC BY 4.0. No file of the article's supplement
is shipped. The notebook quotes, as constants, the eigenvalues and first three eigenvectors of the article's
reference matrix `Cbmat` (supplement `code.zip`, `code/data/covid_Ch.RData`; figshare record licensed CC BY 4.0;
credit: N. Wycoff, M. Binois, S. M. Wild), as printed by `check_accuracy.py`. The model (`covid_allinone.R`, GPL-3.0,
© 2020 Regents of the University of Minnesota; credit: the University of Minnesota COVID-19 Modeling Consortium,
E. A. Enns et al.) is not shipped; it is downloaded with the supplement and called unchanged. Study design:
N. Wycoff, M. Binois and S. M. Wild; cite their article (above).

## Unknowns

- The input ranges are not stated in the article; the box [0.01, 0.99]^8 is taken from the article's study script
  `covid_sim.R` (its reference run used [0, 1]^8).
- The model release and the meaning of "scenario 2" (open question 1).
- The article's reference GP (hyperparameters, nugget) is not re-fitted here; only its stored C is compared.
