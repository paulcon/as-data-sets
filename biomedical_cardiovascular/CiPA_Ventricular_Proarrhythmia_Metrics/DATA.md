# CiPA_Ventricular_Proarrhythmia_Metrics: data

Kind: **stored data**. The model needs Myokit (a Python package that compiles the model to C and
integrates it with SUNDIALS CVODES), so it is not a numpy/scipy model. `generate_data.py` and
`cipaord_driver.py` ran it in a Linux container, where the shipped `cipaord_virtual_drugs.csv` was
written; the CSV is the record. Myokit is not installed in this repository's `.venv`, so the scripts that
need it (`generate_data.py` except `--merge`, `check_accuracy.py`) stop there with an error that says what
to install. The notebook needs only the CSV.

## Files

| file | contents | role |
|---|---|---|
| `CiPA_Ventricular_Proarrhythmia_Metrics.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` (Python 3.11) with seed 1054; outputs committed |
| `cipaord_virtual_drugs.csv` | 224 rows × 27 columns (below) | stored model runs |
| `cipaord_driver.py` | `run(x, ...)`, `beat_metrics(...)`, `response_class(r)`; `INPUTS`, `NAMES`, `LOWER`, `UPPER`, `QOI_NAMES` | Myokit wrapper; the constants need only numpy |
| `generate_data.py` | the design and the runs; `--chunk`, `--merge`, `--check` | regenerates the CSV (needs Myokit) |
| `check_accuracy.py` | reruns three rows from the exact design and from the CSV's inputs | the accuracy numbers not printed elsewhere (needs Myokit) |
| `cipaord_endo.mmt` | the CiPAORdv1.0 endocardial model as a Myokit file (ORd core after the O'Hara–Rudy reference code, BSD-3 notice in its header; hERG section new) | the model |
| `LICENSE-ORd` | the BSD 3-clause notice and disclaimer of the O'Hara–Rudy code, as in the FDA C file's header | licence of the ORd-derived parts of `cipaord_endo.mmt` |

Columns of `cipaord_virtual_drugs.csv` (two `#` lines, then numbers; `np.loadtxt(path, delimiter=",")`):

- 0–8: the inputs `bINa, bINaL, bIto, bIKs, bICaL, bIK1` (%), `sbIKr` (-), `Vhalf` (mV), `Ku` (1/ms), physical values;
- 9–16: features of beat 1000: `qNet` (nC/uF), `APD90`, `APD50` (ms), `peakVm` (mV), `diastolicCa`,
  `peakCa` (nM), `CaTD50`, `CaTD90` (ms);
- 17–18: `max_dv`, the largest dV/dt between 30 % repolarisation and APD90 of beat 1000 (mV/ms; > 0 is
  an EAD, the FDA code's test), and `dVdt_max`, the largest dV/dt of beat 1000 (mV/ms), the upstroke
  velocity except in 4 rows (bINa 74–80 %), where the INa upstroke is slower than the stimulus and
  `dVdt_max` is the stimulus slope, 80 mV/ms (printed by the notebook); there APD90 and APD50 are timed
  from the stimulus onset, as the FDA definition (time of max dV/dt) gives;
- 19–20: `n_EAD`, `n_norepol`: beats among 996–1000 with an EAD, or without full repolarisation;
- 21–22: `qNet_spread`, `APD90_spread`: max − min over beats 996–1000;
- 23–24: `qNet_b500`, `APD90_b500`: the same features of beat 500 (convergence check);
- 25: `response`: 0 normal, 1 EAD in beats 996–1000, 2 a beat without full repolarisation or AP;
- 26: `cpu_s`, CPU time of the run (s).

## Model

**Citation.** J. Parikh, P. Di Achille, J. Kozloski, V. Gurev, Global sensitivity analysis of ventricular
myocyte model-derived metrics for proarrhythmic risk assessment, *Frontiers in Pharmacology* 10:1054,
2019, doi:10.3389/fphar.2019.01054 (CC BY 4.0). Paper PDF read: the publisher's version.

**Model.** CiPAORdv1.0, endocardial cell (paper p. 3): the O'Hara-Rudy human ventricular myocyte
(O'Hara et al., PLoS Comput. Biol. 7(5):e1002061, 2011, doi:10.1371/journal.pcbi.1002061) with IKr
replaced by the hERG Markov model with dynamic drug binding and trapping (Li et al., Circ. Arrhythm.
Electrophysiol. 10(2):e004628, 2017, doi:10.1161/CIRCEP.116.004628) and five conductances rescaled by Dutta
et al. (Front. Physiol. 8:616, 2017, doi:10.3389/fphys.2017.00616): GKr ×1.114/1.1, GNaL ×2.661, GKs
×1.870, GK1 ×1.698, PCa ×1.007. 48 physiological ODE states plus the qNet integral.

**How the model file was made.** The paper used the FDA code (https://github.com/FDA/CiPA, cited as Chang
et al., Front. Physiol. 8:917, 2017, doi:10.3389/fphys.2017.00917) "with minor modifications" (p. 4). That
repository is licensed **GPL-3.0**, but its model file `AP_simulation/models/newordherg_qNet.c` (commit
539ff80, cloned 2026-11-01) has two layers: the O'Hara–Rudy (ORd) reference code, whose header is the
**BSD 3-clause** notice of T. O'Hara and Y. Rudy (Washington University in St. Louis, 2011–2015), and the
FDA's additions (the hERG Markov model with drug binding and trapping, the qNet integral, the R parameter
interface). `cipaord_endo.mmt` is a Myokit re-expression of CiPAORdv1.0 built accordingly:
- **ORd core** (CaMKII, reversal potentials, INa, INaL, Ito, ICaL, IKs, IK1, INaCa, INaK, background
  currents, SR release and uptake, fluxes, buffers, concentrations, geometry): it **follows the O'Hara–Rudy
  reference implementation**, not the typeset equations of the ORd paper. It keeps that code's section
  order, many of its variable names and idioms (`vffrt`/`vfrt`, `dti_develop`/`dti_recover`, `a_rel`,
  `btp`, `Jupnp`/`Jupp`, `Bcai`/`Bcass`/`Bcajsr`, the 0.001 clamp of `tau_rel`), and many right-hand sides
  are the same as the C file's after normalising spaces and component prefixes (88 of 286 non-constant
  assignments, a lower bound, counted against the C file at commit 539ff80; an external run, not
  reproducible from the repository). These parts are derived from
  BSD-3 code, so the notice and disclaimer are reproduced, exactly as in the C file's header, at the top of
  `cipaord_endo.mmt` and in `LICENSE-ORd`.
- **hERG/drug-binding section** (`ikr`): written in flux form from Li et al. (2017), with named rates
  `k1`…`k63`, eleven named fluxes `J_*`, `kon = Ku·sbIKr`, `koff_IO` and `ftrap`. The C file writes each of
  its nine derivatives as one inline expression; none of the 40 non-constant assignments of `ikr` equals a
  C right-hand side, and no comment is copied. The parameter values (`newordherg_pars.txt`,
  `AP_simulation/funs/get_pars.R`) are numbers, used as published.
- **New:** the virtual-drug scaling (`drug`, Parikh et al. Eqs 1–3), the qNet state (the sum of six
  currents that defines qNet, Dutta et al. 2017), the stimulus and the protocol.

So no GPL-only code of the FDA repository is included: of the FDA's additions only numbers (parameter
values, the initial state) and the published model equations are used. The initial state is the published control steady
state at CL 2000 ms (`newordherg_states_CL2000.txt`, 49 numbers). **Equivalence check** (an external
run in the container that wrote the data, not reproducible from the repository: the script was not
kept, and it needs the GPL-3.0 C file): the FDA C file was compiled with its R headers removed and its `derivs`
evaluated through ctypes at 300 random states (voltages in [−95, 45] mV, other states ±30 % of the steady
state, random bound-drug states) and random virtual drugs, with the drug set as `get_pars.R` does
(fold changes 1/((100 − b)/100) times the Dutta factors; Kmax = 2·sbIKr with D = n = halfmax = 1, so that
Kmax·D^n/(D^n + halfmax) = sbIKr). The right-hand sides of the Myokit model agreed to a maximum relative
difference of 1.5e-12 over all 49 states (FDA repository at commit
539ff80d6616895de5d943e7bfb77b346d195db4).

**Inputs** (paper Table 2, p. 3; uniform on a linear scale, as the paper's Saltelli sample of Virtual
Drug Population I):

| name | units | meaning | range | source |
|---|---|---|---|---|
| bINa | % | block of fast Na current INa (GNa × (100 − b)/100, Eq 3) | [0, 80] | Table 2, p. 3 |
| bINaL | % | block of late Na current INaL | [0, 80] | Table 2, p. 3 |
| bIto | % | block of transient outward current Ito | [0, 80] | Table 2, p. 3 |
| bIKs | % | block of slow delayed rectifier IKs | [0, 80] | Table 2, p. 3 |
| bICaL | % | block of L-type Ca current (PCa; ICaL, ICaNa, ICaK) | [0, 80] | Table 2, p. 3 |
| bIK1 | % | block of inward rectifier IK1 | [0, 80] | Table 2, p. 3 |
| sbIKr | - | static hERG block component Kmax·C^h/(IC50^h + C^h) (Eq 1); drug binding rate Ku·sbIKr | [0, 4] | Table 2, p. 3 |
| Vhalf | mV | hERG drug trapping: half-voltage of the bound-closed to bound-open step | [−200, −1] | Table 2, p. 3 |
| Ku | 1/ms | hERG drug unbinding rate | [1e-5, 1] | Table 2 prints 0 to 1; text p. 3 gives 1e-5 to 1 |

We use the text's lower bound 1e-5 for Ku (with Ku = 0 no drug binds, whatever sbIKr). Kt = 3.5e-5 /ms,
the trapping rate the FDA code uses for every drug, is held. Because Ku is sampled on a linear scale, only
3 of the 224 rows have Ku < 0.01 /ms (22 below 0.1): the slow-binding, slow-unbinding end of the range,
where the paper's Fig. 1 places the smallest IKr reduction at a given sbIKr, is barely sampled, and the
small Ku weights found here say little about it.

**Protocol** (paper p. 4): 1000 beats at CL 2000 ms (bradycardia), stimulus −80 µA/µF for 0.5 ms at the
start of each beat, from the control steady state. CVODES (Myokit) with abs/rel tolerances 1e-6/1e-4; the
paper used deSolve lsoda with rtol = atol = 1e-6 (effect checked below). Beats 500 and 996–1000 are logged
every 0.01 ms for the first 20 ms (upstroke) and every 0.25 ms after; the other beats are not logged.

**Quantities of interest** (paper Table 3, p. 5; definitions of the FDA `AP_simulation.R` and
`metric_funs.R`, with crossing times linearly interpolated on the logging grid instead of read off the FDA
code's 1 ms output grid): qNet = ∫ (INaL + ICaL + Ito + IKr + IKs + IK1) dt over the beat (nC/µF; an ODE
state reset to 0 at the start of the beat); APDx = time from max dV/dt to the first crossing after the peak of
Vrest + (1 − x)(Vmax − Vrest), Vrest = min V before the peak; peakVm = max V; diastolicCa and peakCa = min
[Ca]i before the peak and max [Ca]i (nM); CaTDx = time from the first [Ca]i above 90 % of its rise to the
crossing of Ca_rest + (1 − x)(Ca_max − Ca_rest) after the peak. The CaTD start is the FDA code's:
`AP_simulation/AP_simulation.R` (commit 539ff80, lines 265–266) sets `caup <- carest + (camax - carest)*0.9`
and `t_up <- t[which(ca > caup)[1]]`, the first output time above 90 % of the rise; here that time is
interpolated. Whether the paper's "minor modifications" changed it is not known. The paper also studies an EAD threshold
ThEAD,hERG (an extra hERG reduction scan); it is not computed here.

**The paper's unpublished "minor modifications."** Not known. Assumed here: (1) the virtual drug acts
through Eq 3 on the six conductances and through the binding rate Ku·sbIKr of the Li et al. model;
(2) Kt = 3.5e-5 /ms; (3) the features are those of the last beat. The FDA code instead keeps, of its last 250
beats, the beat with the largest plateau dV/dt; the paper says "the last five beats were analyzed" (p. 4).
Only beats 996–1000 and 500 are logged here; the file stores the EAD count of beats 996–1000 (`n_EAD`)
and the plateau dV/dt of beat 1000 only (`max_dv`). With no EADs in the design, the last five beats within
0.11 nC/µF and 0.67 ms of each other, and beat 500 within 0.50 nC/µF and 0.44 ms of beat 1000 (below), the
choice of beat does not matter here.

**Gradients:** none. The notebook uses the global quadratic estimator (QPHD) on values, cross-checked
with the linear least-squares estimator (OLS).

**Sampling design.** The first 224 points of one scrambled Sobol' sequence of 2^8 = 256 points in nine
dimensions (`scipy.stats.qmc.Sobol(d=9, scramble=True, seed=1054)`), mapped linearly to the box. A prefix
of 224 points is not a power of 2, so it lacks the full balance of a Sobol' net; the effect on these
estimates is small. QPHD fits (m+1)(m+2)/2 = 55 coefficients; 224 rows are about 4× that. The paper used
SALib's Saltelli design with n = 500 (10,000 runs, p. 5) and a second population (10,000 kernel-density
samples around the 28 CiPA drugs). Its outputs are not published: the Data Availability statement (p. 17)
says "The datasets analyzed for this study can be found in the Supplemental Material", but the
Supplementary Material (Data Sheet 1, 21 pages) holds the drug parameters of the 28 CiPA drugs (Tables
S1–S3), plots of the two virtual-drug populations' input distributions (Figs S2, S3), Sobol' indices of the
other five features (Fig S5), the regression-vs-Sobol' comparison (Fig S6), the feature correlation heatmap
(Fig S7), the classification thresholds and counts (Tables S4, S5), but no table of the 10,000 runs.

**Abnormal runs.** None: all 224 runs have response 0 (no EAD in beats 996–1000, `n_EAD` = 0, and every
beat repolarises); the largest plateau dV/dt of beat 1000 over all runs is −0.088 mV/ms (notebook). The design is therefore used whole, without
exclusion skew. Two rows have qNet slightly below 0 (minimum −1.75 nC/µF); qNet is a signed charge, so
they are physical and kept.

**Published results.** No active subspace. Linear-regression R² (in-sample) of eight features (Table 4, p. 8); Sobol'
indices of APD90, qNet and peakCa (Fig. 4, with numbers in the text on p. 8); MDA and Monte Carlo filtering
results; classification of the 28 CiPA drugs (Table 5). The notebook compares Table 4 and Fig. 4.

## Regenerating

```bash
pip install myokit numpy scipy        # plus SUNDIALS (e.g. apt install libsundials-dev) and a C compiler
nice -n 10 python generate_data.py   # 14 chunks of 16 points (finished chunks skipped), then merge
python generate_data.py --chunk 3    # one chunk -> parts/chunk_3.npz (progress saved after every point)
python generate_data.py --merge      # parts -> cipaord_virtual_drugs.csv
python generate_data.py --check 3    # rerun 3 rows; compare with the CSV and with rel_tol 1e-6
```

Without Myokit, SUNDIALS and a C compiler, `generate_data.py` (except `--merge`) stops with an error that
says what to install and to rerun it. The shipped file is the record; it is regenerated only if a problem
is found in it.

Provenance of the shipped file: 2026-11-01, written in an Ubuntu 24.04 container, Python 3.13.18, Myokit 1.39.2,
SUNDIALS 6.4.1 (Ubuntu `libsundials-dev`), numpy 2.5.4, scipy 1.18.1; one process at `nice -n 10` on 2
shared CPUs; 14 chunks, each under 20 min wall time (longest 1180 s); the last chunk was interrupted by a
container restart and resumed from its saved progress (12 of 16 points kept). **Run time of
`generate_data.py`:** 24.4 s CPU per run of 1000 beats (19.8–31.5 s, printed by the notebook), 91 min CPU
for the 224 runs, 7284 s (121 min) wall time in all (original run in the container).

## Accuracy

The checks that rerun the model need Myokit, which is not installed in `.venv`; their numbers below come from
the original runs in the container that wrote the data (Python 3.13.18, Myokit 1.39.2), and the scripts
named recompute them where Myokit is installed.

- **Model equations:** right-hand sides equal to the FDA C code to 1.5e-12 (relative), see above (external
  run, not reproducible from the repository).
- **Control steady state.** Pacing the control model (no drug) 1000 beats from the distributed CL 2000 ms
  steady state changes the states by a median relative 2.0e-5 and at most 8.2e-3 (`ical.nca`, a state of
  size 1.5e-3), so the initial state is the steady state of this implementation too (`generate_data.py
  --check`, original run).
- **Convergence of the 1000-beat train** (printed by the notebook): between beats 500 and 1000, qNet changes
  by a median 0.076 nC/µF (max 0.50, which is 2.1 % of its standard deviation over the rows; max relative
  2.2 %) and APD90 by a median 0.047 ms (max 0.44 ms, relative 6.9e-4). Over beats 996–1000 the spread is
  at most 0.11 nC/µF and 0.67 ms.
- **Reproducibility and solver tolerance** (original runs, 2026-11-01). Rerunning rows 0 and 223 at the exact
  design inputs reproduced the eight features to a relative 2.0e-10 (the 10 significant digits written;
  `generate_data.py --check` and `check_accuracy.py`). Rerunning from the CSV's 10-digit inputs instead
  changed them by up to 1.9e-4 (rows 0, 112, 223; `check_accuracy.py`), the level at which CVODES's adaptive
  steps at rel_tol 1e-4 respond to a 1e-10 perturbation of the inputs; this is the numerical noise of the
  data. Tightening rel_tol to 1e-6 (`generate_data.py --check`) (the paper's lsoda tolerance) changed
  the features by at most 3.7e-4 relative (row 0: qNet 48.8162 vs 48.8342 nC/µF, APD90 429.107 vs 429.097
  ms; row 223: 6.1e-5), against standard deviations over the rows of about 24 nC/µF and 110 ms. Cost at
  rel_tol 1e-6: about 20 % more CPU.
- **Gradients:** none, so no gradient check.

## How these files relate to the paper

- **Not the paper's runs.** The authors' simulation outputs are not public; these are a new computation
  with a Myokit re-expression of the FDA CiPAORdv1.0 code (ORd core after the O'Hara–Rudy reference code,
  hERG section written from Li et al. 2017; right-hand sides equal to the C code's to 1.5e-12, an external
  run), the paper's Table 2 box,
  protocol and features, on a smaller Sobol' design.
- **Same as the paper:** model (CiPAORdv1.0 endo), nine inputs and ranges, Eq 3 conductance scaling,
  CL 2000 ms, 1000 beats from the control steady state, features of Table 3 with the FDA definitions,
  uniform linear sampling.
- **Different:** 224 scrambled Sobol' points instead of 10,000 Saltelli points; Myokit/CVODES (rel 1e-4)
  instead of R/deSolve lsoda (1e-6); crossing times interpolated; Ku from 1e-5; the authors' code
  modifications unknown; no EAD threshold metric and no Virtual Drug Population II.
- **Results.** In-sample linear R² (as in Table 4) of six of eight features within 0.01 of Table 4
  (leave-one-out: qNet 0.970, APD90 0.945, peakCa 0.894, CaTD50 0.725, CaTD90 0.856). The two exceptions,
  CaTD50 0.751 vs 0.83 and CaTD90 0.870 vs 0.92, are **outside** the bootstrap 95 % intervals over the
  rows ([0.701, 0.816] and [0.842, 0.904]), while the other six intervals meet the paper's values: a model
  or feature-definition difference, not sampling noise. Neither the solver tolerance (≤ 3.7e-4 relative)
  nor the FDA code's 1 ms output grid (≤ 1 ms against standard deviations of 71.5 and 81.6 ms) can explain
  it. The misfit is concentrated at high ICaL block and low INa block (bICaL > 60 %, bINa < 20 %: 14 rows,
  Ca transient 43.4–76.9 nM, 0.23 of the CaTD50 residual sum of squares, residuals down to −113 ms); adding
  bINa·bICaL and the two squares raises R² to 0.857 (CaTD50) and 0.928 (CaTD90). The CaTD definition here
  is the FDA code's (start at 90 % of the rise, above), so plausible causes are the authors' "minor
  modifications" of that code (which may include the CaTD definition) or the beat they analysed. The paper
  is itself inconsistent here: its p. 8 text says "CaTD90 and peakCa were the only features that had R²
  values of less than 0.91" (p. 8), while Table 4 (p. 8) gives CaTD90 0.92, CaTD50 0.83 and peakVm 0.88;
  the same paragraph says that "qNet, APD50, APD90, and diastolicCa exhibited R2 values greater than 0.94",
  while Table 4 gives 0.94 for APD90 and APD50; and the text on p. 3 gives the range of Vhalf as "(−1 to 200)",
  against −200 to −1 in Table 2. The
  quadratic-surrogate Sobol' indices of qNet (bINaL 0.39, sbIKr 0.26, bICaL 0.15, bINa 0.13) match the
  paper's 40, 26, 16 and 13 %. Active subspaces (not in the paper): n = 1 for seven features by the largest
  ratio λk/λk+1 among k = 1–3 (CaTD90 and peakVm borderline, λ1/λ2 8.3 vs λ2/λ3 6.4 and 8.8 vs 7.3;
  diastolicCa has a second gap of 11.9 after a first of 24.9); n = 2 for CaTD50 (λ2/λ3 12.7 > λ1/λ2 4.9). See the notebook's comparison table.
- **Not a duplicate of `Rabbit_Ventricular_Action_Potential`:** that data set is the Shannon 2004 rabbit
  model with 15 current scaling factors at 500 ms; this one is the human ORd-based CiPA model with
  drug-block inputs (including dynamic hERG binding) at 2000 ms and the CiPA metric qNet.

## License and credit

Code (the notebook, `cipaord_driver.py`, `generate_data.py`, `check_accuracy.py`): MIT (`LICENSE`).
`cipaord_endo.mmt`: **MIT for the new parts; the ORd-derived parts BSD 3-clause (copyright T. O'Hara,
Y. Rudy, Washington University in St. Louis)**; the notice and disclaimer are kept in the file's header
and in `LICENSE-ORd`, as BSD-3 requires for redistributed source. Because they are not under CC BY 4.0,
`cipaord_endo.mmt` and `LICENSE-ORd` are listed in the exceptions list of `LICENSE-DATA`. Its parameter
values and initial state are numbers from the FDA CiPA repository (GPL-3.0), used as published facts and
credited here. Data (`cipaord_virtual_drugs.csv`): CC BY 4.0. The paper is CC BY 4.0. No GPL-only
code of the FDA CiPA repository (GPL-3.0) is included: its hERG/qNet section was re-expressed in flux form
from Li et al. 2017, and its parameter values and published initial state (numbers) were used to check and
initialise the model file. Credit: model, T. O'Hara, L. Virág, A. Varró and Y. Rudy (ORd), Z. Li et al. (hERG binding
model), S. Dutta et al. (CiPAORdv1.0); parameter study, J. Parikh, P. Di Achille, J. Kozloski and
V. Gurev. Cite Parikh et al. 2019 (doi:10.3389/fphar.2019.01054) and the model papers above. The paper
states no funding.
