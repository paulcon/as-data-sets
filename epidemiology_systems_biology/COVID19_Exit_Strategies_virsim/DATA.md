# COVID19_Exit_Strategies_virsim: data

Kind: **stored data (generated for this repository with virsim 1.0.5, an outside R code).** virsim is an R
package with compiled C++ (Rcpp), not a numpy/scipy model, and its licence (CC BY-NC-ND 4.0) does not allow a
modified copy, so the repository holds no virsim source and no port. The runs were made once, on a rented Linux
machine, with the R driver in `R/`, and the shipped CSVs are the record. `generate_data.py` documents and
repeats the computation where R and virsim 1.0.5 are installed; without them it writes and checks the design
and stops with a message that says what to install. `convert_data.py` rebuilds the shipped CSVs from the
per-chunk files of the run and checks them. The notebook needs only numpy, scipy and matplotlib.

## Files

| file | contents | role |
|---|---|---|
| `COVID19_Exit_Strategies_virsim.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 568; outputs committed |
| `runs_FC.csv`, `runs_CT.csv`, `runs_IL.csv`, `runs_PO.csv` | 572 rows each (layout below) | virsim runs: design, seed, status, run time, the two QoIs |
| `ic_prev_FC.csv`, ... `ic_prev_PO.csv` | 572 × 552 each: row, then IC prevalence per million on days 0-550 | daily series behind the QoIs |
| `R/strategies.R` | settings of the four strategies, `run_one()` (one virsim run and its QoIs) | driver (new code; calls virsim, contains none of it) |
| `R/run_chunk.R` | runs a chunk of design rows, writes one CSV per chunk; resumable | driver |
| `install_ubuntu2404.sh` | installs R 4.3.3 and the pinned R packages from the Ubuntu 24.04 archive and builds virsim 1.0.5 from its tarball | install step of the recipe below |
| `generate_data.py` | writes the design (numpy + scipy) and checks it against the shipped inputs; runs `R/run_chunk.R`; merges | regenerates the runs (needs R and virsim 1.0.5) |
| `convert_data.py` | merges the per-chunk result files into the shipped CSVs (`--out`, `--check`) | lossless conversion with checks (numpy only) |

`R/strategies.R` and `R/run_chunk.R` are the files that ran, unchanged (sha256 `e95f6ebc…5d60` and `3386818a…ace78`). `install_ubuntu2404.sh` is the file that ran (sha256 `31dc1fa9…5152`, at commit 3602190) with one later change: it installs into an `INSTALL_DIR` argument (default the current directory) and refuses this directory and any folder in it, where it used to install into its own folder. On the rented machine they sat in one folder with the design files and a GNU-parallel runner, `run_all.sh`; their comments call that folder "the kit", and `generate_data.py` now takes the runner's place. `run_all.sh` is not shipped: where the last message of `install_ubuntu2404.sh` says that `run_all.sh` sets `R_LIBS`, set it yourself as in step 1 of the recipe below.

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`.

**Columns of `runs_S.csv`:** `row`, `kind` (0 main design, 1 seed replicate), `group` (replicate group, −1 for
main rows), `source_row` (the main row a replicate group repeats; −1 for main rows and for group 0), `seed`
(virsim's `set.seed` value), the strategy's inputs in physical units (table below), `status` (0 = ok; all rows
are 0), `runtime_s` (s, elapsed time of the run), `IC_prev_avg_max` (QoI 1, IC patients per million),
`IC_excess_days` (QoI 2, IC patient-days per million), and for IL `lock_days`, `lift_days`, for PO
`lift_interval_days` (the whole days virsim used, the inputs truncated).

**Rows (each strategy):** 0-511 main design; 512-571 six replicate groups of 10 runs with new seeds, group 0 at
the Table 1 means, group g = 1..5 at main row g − 1 (whose own run is an eleventh member of the group).

## Model

### Citation

F. Gugole, L. E. Coffeng, W. Edeling, B. Sanderse, S. J. de Vlas, D. Crommelin, "Uncertainty quantification and
sensitivity analysis of COVID-19 exit strategies in an individual-based transmission model", *PLOS Computational
Biology* 17(9): e1009355 (2021), doi:10.1371/journal.pcbi.1009355 (Crossref-checked). The PDF read is the
publisher's version (CC BY). The model: L. E. Coffeng, virsim, version 1.0.5, https://gitlab.com/luccoffeng/virsim/-/tree/v1.0.5
(the paper's ref. [6]); described in S. J. de Vlas, L. E. Coffeng, *Sci. Rep.* 11: 4445 (2021),
doi:10.1038/s41598-021-83492-7. The paper's study scripts: github.com/FGugole/UQ_covid19 (ref. [28]; LGPL-3.0),
files `EasyVVUQ/*_UQ_bio.r`, read at commit dc6715f when the driver was written.

### Physics and settings

- Individual-based stochastic SEIR model with life-long immunity, in clusters (towns) and superclusters
  (provinces) (paper pp. 3-4). One million agents, 550 days (pp. 4-5), one time step per day. Held settings
  from the study scripts: 1000 clusters, 20 superclusters, cluster size sd 0.95, supercluster size sd 0,
  50 initial infections seeded in the superclusters as in the scripts, contact shape 3.4, contact assortativity
  0.26, population and supercluster mixing 0.05, `inc_cum_cond` 9500.
- Contact rate = R0 / `avg_duration_infectiousness` (p. 9). Latent period Weibull with mean 4.6 d and shape
  `shape_exposed_time`; infectious period exponential with mean `avg_duration_infectiousness` (scripts).
  Inter-individual variation of the intervention effect `intervention_effect_var` = 1/x with x =
  `intervention_effect_var_inv` from Table 1 (p. 9).
- Lockdown before every strategy except PO (pp. 4-5; scripts): transmission 100 % until day 10, 30 % to day 17,
  15 % to day 70, 25 % to day 100. FC and CT also have an external force of infection of 50 per year per
  million (scripts).
- Strategies (pp. 7-9; schedules from the scripts): **FC** transmission `intervention_effect` from day 100,
  `uptake` throughout; **CT** no general measures from day 100 (uptake 1), tracing with `trace_rate_I`,
  `trace_prob_E`, `trace_contact_reduction`; **IL** measures lifted on day 100 for 25 days, then lockdowns of
  `lock_length` days at `lock_effect` alternating with openings of `lift_length` days (both truncated to whole
  days, `as.integer`, as in the scripts), `uptake` throughout; **PO** virsim's `gen_phased_lift` with start
  day 15, initial lockdown 60 days at effect 0.2, 20 phases (one per supercluster) every
  `intervention_lift_interval` days (truncated), effect `pl_intervention_effect_hi` in regions still closed,
  `sc_isolation_effect` 0.5.
- IC patients: virsim's `gen_derived_outcome` defaults (held): each infection needs IC with probability 1/182,
  after a Weibull delay (mean 14 d, shape 10), for a Weibull stay (mean 19.8 d, shape 1.24). This step draws its
  own random numbers after the transmission run, from the same seeded stream.

### Inputs

All inputs are independent, with the distributions of Table 1 (p. 4). The design stores physical values;
the notebook maps each through its CDF to a standard normal score (`z = Φ⁻¹(F(x))`), the same scaling for all.

| input | strategy | units | distribution (Table 1) | mean | 95 % interval |
|---|---|---|---|---|---|
| `intervention_effect` | FC | – | Beta(38, 70) | 0.35 | [0.27, 0.44] |
| `uptake` | FC, IL, PO | – | Beta(16, 2) | 0.89 | [0.71, 0.99] |
| `trace_rate_I` | CT | 1/d | Gamma(shape 2, scale 0.2) | 0.40 | [0.05, 1.11] |
| `trace_prob_E` | CT | – | Beta(2, 6) | 0.25 | [0.04, 0.58] |
| `trace_contact_reduction` | CT | – | Beta(10, 2) | 0.83 | [0.59, 0.98] |
| `lock_length` | IL | d | Gamma(20, 2) | 40.0 | [24.4, 59.3] |
| `lift_length` | IL | d | Gamma(15, 1) | 15.0 | [8.4, 23.5] |
| `lock_effect` | IL | – | Beta(14, 42) | 0.25 | [0.15, 0.37] |
| `intervention_lift_interval` | PO | d | Gamma(25, 2) | 50.0 | [32.4, 71.4] |
| `pl_intervention_effect_hi` | PO | – | Beta(14, 42) | 0.25 | [0.15, 0.37] |
| `avg_duration_infectiousness` | all | d | Gamma(25, 0.2) | 5.00 | [3.24, 7.14] |
| `R0` | all | – | Gamma(100, 0.025) | 2.50 | [2.03, 3.01] |
| `intervention_effect_var_inv` | all | – | Gamma(2, 0.05) | 0.10 | [0.01, 0.28] |
| `shape_exposed_time` | all | – | Gamma(17.5, 1) | 17.5 | [10.3, 26.6] |

m = 6 (FC), 7 (CT), 8 (IL), 7 (PO). Table 1 gives no units; days for the time inputs and per day for
`trace_rate_I` follow from virsim's one-day time step (virsim turns the rate into the per-step probability
1 − exp(−rate)). The paper also treats the seed as an input, drawn uniformly from 16384 to 65536 (p. 10); here
each run's seed is drawn from that set without replacement, and the seed is not a coordinate of the subspace.

### Quantities of interest (p. 6)

QoI 1: the maximum over days of the 30-day moving average of the daily IC prevalence p(t) per million. The paper
says "averaging window = 30 days"; the study scripts use `forecast::ma(order = 30)`, a centred moving average of
even order (weights 1/60, 1/30, ..., 1/30, 1/60 over 31 days, days without a full window dropped), and the
driver computes the same average in base R. QoI 2: Σ_{t=0}^{550} max(0, p(t) − 109), IC patient-days in excess
of 109 beds per million. Both are recomputed from `ic_prev_S.csv` by `convert_data.py` and the notebook.

### Gradients

None. virsim is stochastic and its output is integer counts; a finite difference of two runs differences two
realizations. The notebook uses the QPHD estimator (quadratic least-squares fit, C = g gᵀ + H², in the normal
scores) and checks it against the OLS direction.

### Sampling design

Per strategy: 512 points of a scrambled Sobol' sequence in [0, 1]^m (`scipy.stats.qmc.Sobol`, `random_base2(9)`,
seed 568 + strategy index, FC 0, CT 1, IL 2, PO 3) mapped through the Table 1 inverse CDFs, and 60 replicate
runs (above). Seeds: a permutation of 16384..65536 with `numpy.random.default_rng(668 + strategy index)`, first
572 entries. It is not the paper's design: the paper used Monte Carlo with 2000 runs for the uncertainty
analysis and Saltelli's scheme with 2000(p + 2) runs per strategy for the Sobol' indices, varying only the policy
inputs and the seed there (p. 10). Inputs outside physical meaning: none (all inside the supports; notebook).

### Published results (paper)

- Fig. 2 (pp. 11-12): CDFs of both QoIs per strategy, with and without the non-policy inputs; "only Contact
  Tracing gets close to a probability of 50%" of staying below the IC capacity (p. 11); the CDFs are "only weakly
  affected by non-policy-related parameters" (p. 12).
- Fig. 3 (p. 13): first-order Sobol' indices of QoI 1 for the policy inputs and the seed. `uptake` is
  "responsible for at least 30% of the QoI variance" in FC, IL and PO and "about 70%" in IL; FC's
  `intervention_effect` index exceeds `uptake`'s; `trace_rate_I` carries "circa 50%" in CT; `uptake` and
  `pl_intervention_effect_hi` are of similar importance in PO (p. 13). The seed's index has a 95 % interval that
  "always includes 0 and does not take values above 5%" (pp. 13-14). Total indices in S3 Fig.
- S2 Fig (pp. 9-10; caption p. 21): daily IC prevalence of FC at fixed inputs, varying seed, with the band
  between the 5th and 95th percentiles of 100 realizations; the text gives peaks "between 150 and 200 leading to
  differences up to 25%". The inputs of the figure are not stated.
- No active-subspace analysis.

### What the notebook finds (from these files)

- QPHD, M = 512 per strategy: λ1/λ2 = 13.9 (FC), 15.5 (CT), 16.0 (IL), 11.7 (PO) for QoI 1 and 14.4, 13.0, 13.4,
  10.4 for QoI 2. The dimension follows one rule for all strategies: the largest n with λn/λn+1 ≥ 8 and a stable
  n-dimensional subspace (mean bootstrap distance below 0.3; it is 0.253 or less at the chosen n and 0.487 or more
  one dimension higher). This gives n = 2 for FC (λ2/λ3 = 45.1 and 74.4) and IL (13.3 and 13.4), and n = 3 for
  CT (λ2/λ3 = 3.1 and 3.7, then λ3/λ4 = 16.2 and 17.8) and PO (5.2 and 3.9, then 8.2 and 14.7). For CT a
  quartic in three active variables reaches a leave-one-out R² of 0.977 (QoI 1), against 0.917 for the best
  one-dimensional fit and a noise ceiling of 0.998; its three directions are close to the three tracing inputs.
  For PO the third variable adds little to the fits (n = 2 is a close alternative). Cubic leave-one-out R² on the
  chosen subspaces: 0.972, 0.930, 0.925, 0.946 (QoI 1) and 0.964, 0.903, 0.973, 0.911 (QoI 2).
- w1 is the OLS direction (cosine ≥ 0.988). It is led by `intervention_effect`, R0 and `uptake` (FC);
  `trace_rate_I`, `trace_contact_reduction`, `trace_prob_E` (CT); `uptake` against R0 (IL); `uptake`,
  `pl_intervention_effect_hi`, R0 (PO).
- Seed noise (pooled replicate variance / QoI variance): 0.0018 to 0.0387, most known to about a factor of 2 (CT QoI 2 more loosely, [0.0003, 0.0039])
  (bootstrap over the six replicate groups, e.g. PO QoI 1 0.0387 [0.0217, 0.0619]); the noise varies with the
  input point (group variances 0.00 to 2.66 times the pooled value). A noise null puts the eigenvalues that seed
  noise alone produces at 1.4e-4 to 2.8e-3 of λ1 (QoI 1), with the same relative uncertainty.
- The empirical CDFs of both QoIs agree with values read from the paper's Fig. 2 (top row) to within 0.048, inside
  the 95 % DKW band of ±0.060 for M = 512.
- Treating the IL and PO durations as continuous changes the subspaces by a distance of 0.011 or less.

## Regenerating

The runs are an **external run** made once for this repository; the recipe below repeats it. It needs Linux
x86-64 with R; the shipped run used Ubuntu 24.04 on an AWS EC2 c7a.4xlarge (16 AMD EPYC Zen 4 cores, 32 GiB),
16 R processes. virsim's source tarball is not in the repository (licence CC BY-NC-ND 4.0): take tag v1.0.5 from
https://gitlab.com/luccoffeng/virsim/-/tree/v1.0.5 (the file used, `virsim-v1.0.5.tar.gz`, 34 971 bytes, sha256
`e6c629b537adf3ae34fdeb3e9af00fba30aa9b470fdcf7bbaec315fb375edaa6`).

```bash
# 1. install R 4.3.3, the pinned R packages and virsim 1.0.5 (Ubuntu 24.04 x86-64; about 3-5 min; needs sudo).
#    It installs into INSTALL_DIR/Rlib and writes INSTALL_DIR/install.log; INSTALL_DIR must be outside this directory.
mkdir -p ~/virsim_run
bash install_ubuntu2404.sh /path/to/virsim-v1.0.5.tar.gz ~/virsim_run
export R_LIBS=~/virsim_run/Rlib OMP_NUM_THREADS=1

# 2. design only (numpy + scipy; any machine): writes DIR/design and checks it against the shipped inputs
python generate_data.py --out /tmp/virsim --design-only

# 3. one run, compared with the shipped row (about 1-2 min)
python generate_data.py --strategy FC --rows 0:0 --check

# 4. all 2288 runs, 16 processes, then merged into /tmp/virsim/runs_S.csv, ic_prev_S.csv
python generate_data.py --out /tmp/virsim --jobs 16 --check

# 5. rebuild the shipped CSVs from the per-chunk files of a run, and compare
python convert_data.py /tmp/virsim/results --out /tmp/merged --check
```

Pinned versions (from the install log of the shipped run): R 4.3.3 (Ubuntu package 4.3.3-2build2),
data.table 1.14.10, foreach 1.5.2, iterators 1.0.14, doParallel 1.0.17, matrixStats 1.2.0, Rcpp 1.0.12,
RcppArmadillo 0.12.8.1.0, usethis 2.2.2 (virsim's build calls `usethis::use_data`), virsim 1.0.5 built from the
tarball with g++ 13 (`-O2`). `install_ubuntu2404.sh` stops at the stage that fails (OS, tarball sha256, apt,
build, version check and a short virsim run).

**How the shipped files were made (2026-11-08).** The design files were written by the same code as
`generate_data.py` step 1 (`--design-only` writes identical design files to `DIR/design`; in the Linux container its check printed a
relative difference of 0 for the inputs, with equal seeds. In the macOS `.venv` (scipy 1.17.1) the inputs differ by up to 1.1e-15
relative, at most 10 ulp, in the beta inverse CDFs; the check fails if they differ by more than a relative 1e-8, or if a seed differs). The 572 chunks of 4 rows were run with
`R/run_chunk.R` under GNU parallel, 16 processes, three phases (main rows 0-255, the replicates, main rows
256-511). Before the full run, FC row 0 was run alone on the instance and gave the same QoIs as the same row
run earlier in a Linux container with the same R and package versions (QoI 1 = 51.4833..., QoI 2 = 0; external
run, not reproducible from the repository). The per-chunk files were checked and merged, then converted to the
shipped CSVs by `convert_data.py`; `convert_data.py --check` on those per-chunk files reproduces every value
of the shipped files (it copies the decimal text; its check prints a difference of 0, tolerance 1e-12). The per-chunk files are
not in the repository; they are deposited on Zenodo as the archive `virsim_runs.tar.gz`, doi:[10.5281/zenodo.24088176](https://doi.org/10.5281/zenodo.24088176) (CC BY 4.0)
(2 613 650 bytes, sha256 `06591ad66f547b054d91512144c7798a5bacce4330c33219019383f0730ddd85`). To rerun the check,
download and unpack it and run `python convert_data.py results --out DIR --check` (DIR outside this folder). The
archive has 1163 entries: `results/` with the 1144 chunk files (a design-row file and a daily-series file per
chunk), the GNU-parallel job log `results/joblog.tsv` and `results/run.log`; `merged/` with the 8 merged CSVs;
`pilot/` with the single-run pilot (FC row 0); and `install.log`, `pilot_time.txt`, `pilot_diff.txt` and
`check.log`.

**Cost.** `convert_data.py` prints the run times stored in the files: 2288 runs, 63.9 process-hours, mean 101 s
per run with 16 runs at a time (medians 98 s FC and CT, 104 s IL and PO; longest 116 s); memory about 0.5 GB per
process (measured in the container). One run alone on the same instance took 71 s (external run). Wall time of the full run: 4 h 02 min
(19:47 to 23:49 UTC, from the run log; external run); with install and checks about 4.5 h of instance time, about
US$4 at the on-demand price of a c7a.4xlarge in us-east-1 at the time (an estimate, not a billed figure). The
paper reports 1-2 minutes per run on a laptop (p. 20).

`generate_data.py` without R or virsim stops after the design check with the install message above. It never
writes into this directory: `--out` is required without `--check` and must lie outside it (not the directory or a
folder in it), and `--check` without `--out` writes into a temporary directory and removes it; `convert_data.py`'s
`--out` must also lie outside it. `generate_data.py --check` prints the largest difference of the rerun QoIs from the shipped
rows (|new - shipped| / max(1, |shipped|)) and does not judge it (the pilot rerun of FC row 0 above gave the same QoIs);
it exits with status 1 only for a structural mismatch: a missing chunk or row, a nonzero run status, a different
seed, or regenerated inputs that differ from the shipped ones by more than a relative 1e-8.

## Accuracy

- **QoIs vs daily series** (`convert_data.py`, and the notebook): both QoIs recomputed from `ic_prev_S.csv`
  agree with the stored QoIs to a relative 1e-15 (the rounding level) in every row.
- **Design vs shipped inputs** (`generate_data.py --design-only`): seeds agree exactly and inputs to the rounding level
  (relative difference 0 in the Linux container, up to 1.1e-15 in the macOS `.venv`; tolerance 1e-14); IL and PO whole days equal the truncated inputs in every row (notebook).
- **Reproducibility of runs:** with the same seed, R and package build, a run is repeatable; the pilot on the
  rented machine matched the container's run of FC row 0 in both QoIs (external run, not reproducible from the
  repository; `generate_data.py --check` repeats the comparison where virsim is installed). Another build of R,
  Rcpp or the C++ compiler may draw a different realization for the same seed; the check prints the difference.
- **Seed noise** (notebook): pooled replicate standard deviation 13.2 (FC), 11.5 (CT), 28.2 (IL), 17.6 (PO) per
  million for QoI 1, 0.0018 to 0.0387 of the QoI variance over the design. The pooled variance rests on six input
  points (six replicate groups of 10 or 11 runs) and the noise varies with the point: the group variances are 0.00
  to 2.66 times the pooled value (0.00 for QoI 2 where no run of a group has any overload). 95 % intervals from a
  bootstrap over the six groups: QoI 1 FC [0.0063, 0.0250], CT [0.0011, 0.0027], IL [0.0086, 0.0220], PO
  [0.0217, 0.0619]; QoI 2 FC [0.0030, 0.0221], CT [0.0003, 0.0039], IL [0.0075, 0.0186], PO [0.0146, 0.0435].

## How these files relate to the paper

- **Not the paper's runs.** The paper's outputs were not deposited ("generated by the aforementioned scripts").
  These are new runs of the same model version (virsim 1.0.5), with the same input distributions, fixed
  settings and QoIs as the paper's study scripts, on a different design (Sobol' plus replicates instead of Monte
  Carlo and Saltelli), and with all inputs of a strategy varied together (the paper's sensitivity analysis held
  the non-policy inputs fixed).
- The seeds come from the paper's seed set, but each run's seed was drawn here; the study scripts' seeds are not
  known. Expect the same distributions and sensitivities, not the same individual runs.
- **CDFs (Fig. 2, top row, p. 12).** The paper's uncertainty analysis varies all Table 1 inputs and the seed, as
  this design does. Seven points per strategy read from the published figure (reading uncertainty about ±0.02)
  agree with the empirical CDFs here to within 0.048 (FC), 0.033 (CT), 0.012 (IL) and 0.022 (PO), inside the 95 %
  DKW band of ±0.060 for M = 512; e.g. P(QoI 1 ≤ 109) = 0.299, 0.461, 0.059, 0.166 here against about 0.33, 0.43,
  0.06, 0.17 read from the figure.
- **Sobol' indices (Fig. 3, p. 13).** The rankings of the policy inputs agree with the paper's text (notebook
  table). The levels differ by construction (the paper's sensitivity analysis varied the policy inputs and the
  seed only, p. 10); rescaled by one minus the sum of the non-policy first-order effects (an approximation, exact
  for an additive function) they come close to the paper's: IL `uptake` 0.74 ("about 70%"), CT `trace_rate_I`
  0.58 ("circa 50%"), `uptake` 0.37 in FC and PO ("at least 30%").
- **Seed noise.** The noise share here (0.2-3.9 % of the variance) is the analogue of the seed's total Sobol'
  index. The paper reports the seed's first-order index (95 % interval includes 0, never above 5 %, pp. 13-14),
  which is never larger than the total index, so the two are consistent but the paper's value does not bound the
  share here. The seed's total index is in S3 Fig and not given in the text.
- **Spread at fixed inputs (S2 Fig).** At the Table 1 means, ten seeds give FC daily-prevalence peaks of 159 to 223
  per million (sd 20.5, 95 % χ² interval 14.1 to 37.3; peaks of the 30-day average 148.5 to 209.4). S2 Fig shows a
  5th-95th percentile band of 100 realizations of the daily prevalence (caption, p. 21), with peaks "between 150
  and 200" (pp. 9-10), which for a normal distribution means an sd of about 15.2, inside that interval: the two
  spreads are consistent. The inputs of S2 Fig are not stated (the paper's fixed non-policy runs in Fig. 2 use
  `shape_exposed_time` = 20, p. 11, not the Table 1 mean 17.5).
- **R0.** R0 is an uncertain input of the paper (Table 1, p. 4, "non-policy-related"), varied in its uncertainty
  analysis and held fixed in its sensitivity analysis (p. 10). Here it is a leading input of FC, IL and PO
  (binned first-order effect 0.28, 0.39, 0.16). The paper judged the CDFs "only weakly affected by
  non-policy-related parameters" (p. 12); removing R0's first-order effect here would lower the sd of QoI 1 by
  0.15, 0.22 and 0.09 relative, a modest shift on the log axis of Fig. 2. The weight of R0 goes beyond the scope
  of the paper's sensitivity analysis rather than contradicting its uncertainty analysis.

## Open questions (for the authors)

1. The inputs and settings of S2 Fig (FC at fixed parameters, varying seed), and whether its band of 150-200 was
   computed with `shape_exposed_time` = 20.
2. Whether the IL and PO durations were meant to be truncated to whole days (the scripts' `as.integer`) or
   rounded.
3. The seeds used in the paper's runs.
4. The seed's total Sobol' indices (S3 Fig): were they also near zero? They, not the first-order index, measure
   the intrinsic-noise share.
5. Which virsim commit the runs used: the scripts install virsim from its GitLab repository without a tag; the
   Data Availability statement (p. 1) names version 1.0.5.

## License and credit

Code (`*.py`, `R/*.R`, `install_ubuntu2404.sh`, the notebook): MIT, written for this repository. Data (the
CSVs): CC BY 4.0. They are outputs of runs made here with virsim, whose own licence is CC BY-NC-ND 4.0 (stated
in its `DESCRIPTION` and `LICENSE`). The repository's position is that this
licence covers the virsim code and not outputs computed with it. No virsim code is included, and neither is
virsim's bundled data (`param_sim.rda`). The driver contains no code from the paper's study scripts (LGPL-3.0); it uses the parameter values
they pass to virsim. Model: L. E. Coffeng (virsim). Study design, distributions and QoIs: F. Gugole,
L. E. Coffeng, W. Edeling, B. Sanderse, S. J. de Vlas, D. Crommelin; cite their paper (above) and virsim.

## Unknowns

- The paper's own seeds and runs (not deposited); the commit of the study scripts used for the paper.
- Units of Table 1 (inferred: days, per day).
- The inputs behind S2 Fig.
- Run-to-run reproducibility across R/Rcpp/compiler builds was not tested beyond one machine pair with the same
  versions.
- The virsim commit the paper used (the study scripts install it without a tag).
- How the seed noise varies over the input space: it is measured at six points only.
