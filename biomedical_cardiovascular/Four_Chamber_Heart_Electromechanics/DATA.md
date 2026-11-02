# Four_Chamber_Heart_Electromechanics: data

Kind: **stored data**. The values were computed by Strocchi et al. (2023) with their four-chamber heart
electromechanics simulator (CARP coupled to CircAdapt) and with the ToR-ORd cell model, and deposited at
Zenodo, [doi:10.5281/zenodo.7405335](https://doi.org/10.5281/zenodo.7405335) (CC BY 4.0; record and archives
under "Regenerating"). They were not
regenerated (one four-chamber run costs about 3,250 core-hours on a cluster) and are not altered:
`convert_data.py` copies every number character for character into the CSV layout of this repository.

## Files

| file | contents | role |
|---|---|---|
| `Four_Chamber_Heart_Electromechanics.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 1257; outputs committed |
| `fourchamber_inputs.csv` | 407 × 45: the 45 simulator parameters, physical units (column names and units in the second `#` line) | Zenodo `fourchamber/parameters.csv` |
| `fourchamber_outputs.csv` | 407 × 20: LV, RV, LA, RA pressure-volume features (mL, mmHg, mmHg/s); row i matches row i of the inputs | Zenodo `fourchamber/outputs.csv` |
| `torord_ionic_inputs.csv` | 2175 × 29: ToR-ORd conductances and calcium-handling parameters (model units) | Zenodo `ventricular_cell_model/ionic/parameters.csv` (second study) |
| `torord_ionic_outputs.csv` | 2175 × 4: Ca_diast (µM), Ca_ampl (µM), TTP_Ca (ms), RT90_Ca (ms) | Zenodo `ventricular_cell_model/ionic/outputs.csv` |
| `convert_data.py` | reads the deposited `.tar.xz` archives without extracting them, checks their SHA-256, writes the four CSVs; `--check` compares | the conversion (standard library only) |
| `check_accuracy.py` | recomputes the numbers of this file that neither the notebook nor `convert_data.py` prints (stored precision, CV range, strata of the ToR-ORd design) | check |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Deposited label → column name: `[TRPN]` → `TRPN_v`,
`Gncx` → `GNCX`, `Tref,v` → `Tref_v` (and likewise `,v`/`,a` → `_v`/`_a`), `Iup(max)` → `Iup_max_a`,
`[Trpm]max` (a typo in the deposit for the Courtemanche $[Trpn]_{max}$) → `Trpn_max_a`, `gCaL` → `gCaL_a`,
`a,v` → `a_v`, `EDPshift,LV` → `EDPshift_LV`; outputs `pMax` → `pmax`, `dp/dtMax` → `dpdt_max`,
`VMax,la` → `EDVvwave_LA` (the paper's "maximum volume during the v-wave"), `pMax,La` → `pmax_LA`; ToR-ORd
`Ca,diast,v` → `Ca_diast`, `RT90,v` → `RT90_Ca`, `[CMDN]` → `CMDN`, `tau,tr` → `tau_tr`. The full maps are in
`convert_data.py`.

The deposit also holds data of the other sub-models (atrial Courtemanche-Land cell model, ToR-ORd-Land
contraction, tissue electrophysiology, passive mechanics, CircAdapt). They are not used here and not shipped.

## Model

**Citation.** M. Strocchi, S. Longobardi, C. M. Augustin, M. A. F. Gsell, A. Petras, C. A. Rinaldi,
E. J. Vigmond, G. Plank, C. J. Oates, R. D. Wilkinson, S. A. Niederer, "Cell to whole organ global
sensitivity analysis on a four-chamber heart electromechanics model using Gaussian processes emulators",
*PLOS Computational Biology* 19(6):e1011257 (2023), doi:10.1371/journal.pcbi.1011257 (publisher PDF read,
CC BY; checked on Crossref). Code: https://github.com/MarinaStrocchi/Strocchi_etal_2023_GSA (commit
ec8d492, 2024-11-28): Python for the emulators, Sobol' analysis and history matching; **no licence file**,
so nothing of it is copied here. Its `example/ToRORd_ionic/` CSVs are byte-identical (SHA-256) to the
deposited ToR-ORd ionic files.

**Simulator** (Methods, pp. 3–12). Patient-specific four-chamber mesh from CT (a CRT patient). Reaction-Eikonal
electrophysiology without diffusion (Eq. 1) with conduction velocities along the fibres ($CV_{f,v}$,
$CV_{f,a}$), anisotropy ratio 0.4, a fast endocardial layer (factor $k_{FEC}$), a Bachmann-bundle region
($k_{BB}$) and an atrioventricular delay; ToR-ORd (ventricles, with dynamic chloride) and Courtemanche (atria)
action potentials driving the Land active-tension model; transversely isotropic Guccione passive law (Eq. 2,
$\kappa$ = 1000 kPa); pericardium as normal springs of stiffness $k_{peri}$; CircAdapt 0-D circulation. Cell
models pre-paced 500 beats at 854 ms. Each run: unloading to a reference geometry, then 5 beats (Newton
iterations limited to 1 in beats 1–3 and 2 in beats 4–5, linear tolerance $10^{-4}$, time steps 0.02 ms ionic
and 1.0 ms mechanics; these settings change the pressure-volume features by < 3 %, S1 File), with CARP on 512
cores: mean 6 h 20 min, 3247 core-hours per run, 1,321,420 core-hours for the 407 runs (p. 17).

**Inputs of the four-chamber study** (m = 45; Tables 1–3). 117 candidate parameters were screened to 45 by
emulator-based Sobol' analyses of the sub-models (ToR-ORd 29 → 10 for the calcium transient, then with the 17
Land parameters to 13; Courtemanche-Land 3 of 18 plus 10 of 17; tissue electrophysiology; passive mechanics;
CircAdapt), pp. 14–15.

| name(s) | group | units | range | source |
|---|---|---|---|---|
| PCa, TRPN_v, GNCX | ToR-ORd | model units (not stated) | HM: plausible region of the last HM wave | Table 1, p. 4; S2 File |
| Tref_v, TRPN50_v, nTm_v, nTRPN_v, rs_v, rw_v, Aeff_v, ku_v, ca50_v, mu_v | Land, ventricles | model units | HM | Table 2, p. 6; S2 File |
| Iup_max_a, Trpn_max_a, gCaL_a | Courtemanche | model units | HM | Table 1, p. 5; S3 File |
| Tref_a, TRPN50_a, nTm_a, nTRPN_a, rs_a, rw_a, Aeff_a, phi_a, ca50_a, mu_a | Land, atria | model units | HM | Table 2, p. 6; S3 File |
| CVf_v, kFEC, CVf_a, kBB | Eikonal EP | m/s (CV, inferred), 1 | HM | Table 1, p. 4; S4 File |
| AVdelay | Eikonal EP | ms | [100, 200] | Table 1, p. 4 |
| Rsys, Rpulm | CircAdapt | 1 (scaling) | [1.0, 4.0] each | Table 3, p. 8 |
| lAo | CircAdapt | mm | [300, 500] | Table 3 |
| kAo | CircAdapt | 1 | [6.0, 10.0] | Table 3 |
| a_v, bt_v | Guccione, ventricles | kPa, 1 | [0.5, 1.5], [1.5, 4.5] | Table 2 |
| a_a, bf_a, bt_a | Guccione, atria | kPa, 1, 1 | [1.5, 2.5], [4.0, 12.0], [1.5, 4.5] | Table 2 |
| kperi | pericardium | stored kPa/µm (inferred) | paper [0.5, 2.0] kPa/mm; stored values 0.0005–0.002 | Table 3 |
| EDPshift_LV, EDPunload_RV | boundary conditions | mmHg | [3.0, 9.0], [3.0, 8.0] | Table 3 |
| aLvRv, TrefLvRv | LV-RV scaling | 1 | [1.0, 2.0], [0.5, 1.0] | Table 2, p. 7 |

Held values of the other 72 parameters are in Tables 1–3 (e.g. BCL 854 ms, $b_{f,v}$ = 8.0, $b_{ft}$ = 4.0).
**Units.** The paper gives units only for the entries so marked; "model units" are those of the ToR-ORd,
Courtemanche and Land implementations in CARP, not stated. The CV unit (m/s) is inferred from the values
(0.38–1.03; `check_accuracy.py`). The stored $k_{peri}$ equals the paper's kPa/mm value times $10^{-3}$, consistent with kPa/µm
(CARP's length unit is µm); this is an inference.

**Quantities of interest** (p. 16, Table 4). 20 features of the last beat: for LV and RV, EDV, EDP, ESV,
peak pressure, max and min dp/dt; for LA and RA, EDV before atrial contraction, ESV at the end of atrial
contraction, maximum volume during the v-wave, peak pressure during atrial contraction. Volumes mL, pressures
mmHg, dp/dt mmHg/s: inferred (the paper uses mmHg for the EDP inputs and the values are clinical-range
mmHg; Table 4 gives no units). Stored precision: volumes and pressures 2 decimals, dp/dt whole mmHg/s.

**Second study: ToR-ORd ionic model** (S2 File; not read). 29 inputs (all ToR-ORd conductances and
calcium-handling parameters of Table 1), 2175 runs, 4 features of the calcium transient: diastolic calcium,
amplitude, time to peak, 90 % relaxation time (deposit labels `Ca,diast,v`, `Ca,ampl,v`, `TTP,ca,v`,
`RT90,v`; units µM and ms inferred from the values; TTP and RT90 stored in whole ms). The box, inferred from
the data, is the default × [0.75, 1.25] for all 29 inputs: for the 26 inputs with a Table 1 value, sample
min/value is 0.728–0.750 and max/value 1.2475–1.2501; PCa, GNCX and [TRPN] (Table 1: "HM") span 0.750–1.250
of their ToR-ORd defaults (8.3757e-5, 0.0034, 0.07; Tomek et al. 2019). **The two studies' cell regions
differ:** in the four-chamber data GNCX spans 0.275–0.715 of its default (0.000934–0.00243), disjoint from the
ionic box (0.00255–0.00425), and PCa and TRPN_v reach 1.838 and 1.754 of their defaults (against 1.25), so
results of the ionic study do not describe the cell parameters of the four-chamber runs. $P_{Nab}$ and $P_{Cab}$ are stored rounded to two significant digits
(11 and 297 distinct values); whether the runs used the rounded values is not known.

**Gradients.** None: the codes compute none, and finite differences of the four-chamber model would cost 46
runs of 3,250 core-hours per gradient.

**Sampling design.** Four-chamber (pp. 15–17): Latin hypercube samples (N = 90,906) of the 15 interval
parameters were combined with the plausible samples of the last HM waves of the ToR-ORd-Land (N = 90,906),
Courtemanche-Land (148,527) and tissue-EP (99,723) sub-models; from the plausible region the function
`psa_select` of the python library diversipy picked 500 space-filling, "uniformly distributed" points. 42 runs
failed during unloading and 51 during the coupled simulation; the 407 successful runs are the data (the failed
points are not deposited). In the data the 15 interval inputs cover 98.9–99.9 % of the paper ranges with
kurtosis 1.79–1.99, but the failures left a mark: four of them are not uniform (Kolmogorov–Smirnov p = 0.005
for AVdelay, 0.008 for a_v, 0.003 for a_a, 0.011 for bf_a; the other 11 have p ≥ 0.201), with the lowest
quartile under-filled (80, 73, 83 and 72 rows against about 102). The failed runs therefore seem to
concentrate at low passive stiffness and short AV delay, and any analysis of these data describes the model
conditional on a run succeeding. The 30 HM inputs are not uniform (kurtosis up to 3.88)
and are correlated (CVf_v–kFEC r = −0.86, Iup_max_a–Trpn_max_a +0.76; 13 of 990 pairs with |r| > 0.3).
AVdelay takes whole ms (97 values). ToR-ORd ionic: uniform marginals (kurtosis 1.80–1.82); the 2175 rows
are not one Latin hypercube of 2175 points (at most 2019 of 2175 equal-width strata per input are occupied;
`check_accuracy.py`),
consistent with a Latin hypercube with failed or filtered runs or a union of designs (S2 File not read).

**Published results.** No active subspace. Emulator accuracy (Table 4: GPE 5-fold R² 0.7627–0.9852, ISE
82–91 %); total-order Sobol' indices from Saltelli sampling of the emulators with 94,000 evaluations
(Fig. 7 heatmap; Figs 8–10 ranked, signed bars; only bar charts, no printed values). The bar signs are the
signs of the coefficients βi of each GPE's linear prior mean (p. 18; the paper cites Eq (4), but the linear
mean is Eq (3), p. 12), fitted to the same 407 runs; a linear fit to these data therefore reproduces them
nearly by construction. Key statements: the
systemic and pulmonary resistances dominate ventricular systolic outputs and atrial filling; unloading
pressures dominate ventricular EDV; cell active-tension parameters matter for dp/dt.

## Regenerating

The data cannot be regenerated from the repository: they are the authors' runs of a cluster model (one
four-chamber run took 3247 core-hours on 512 cores, p. 17). The CSV files are regenerated from the deposited
archives, with a SHA-256 check of each archive:

```
python convert_data.py <folder with fourchamber.tar.xz and ventricular_cell_model.tar.xz>
python convert_data.py <same folder> --check      # prints "identical" for the four files
```

**Source.** Zenodo record 7405335, "Cell to Whole Organ Global Sensitivity Analysis on a Four-chamber Heart
Electromechanics Model Using Gaussian Processes Emulators - Training Datasets", published 2022-12-06, CC BY 4.0:
DOI [10.5281/zenodo.7405335](https://doi.org/10.5281/zenodo.7405335), the record's only version (concept DOI
10.5281/zenodo.7405334, which resolves to the latest version). This is the DOI the paper's Data Availability
statement gives. Both archives used here are files of that record: `fourchamber.tar.xz` (80,364 bytes, SHA-256
`d6c710dbdfdb7ddcd4ea86775fd024bfe3287f0d39262cf12ea77086e553a87d`, MD5 `665a1dc951ca21c3f0415f3e7893f955`)
and `ventricular_cell_model.tar.xz` (544,168 bytes, SHA-256
`53396893097e6ddc58353b5b960c5df1b293d2864e6f0c1fcb2fda5532140358`, MD5 `3fb2b6ee5112ce9f846e259b62ea7073`);
the sizes and MD5 sums equal those listed by the record (Zenodo API, checked 2026-11-02). The archives
themselves carry no DOI (their members are the CSV files and their folders). The record has four more
archives (atrial cell model, CircAdapt, passive mechanics, tissue electrophysiology), not used here.

In the repo's `.venv` (2026-11-02) `convert_data.py <folder> --check` runs in under 1 s and prints "identical"
for all four files: the shipped CSVs equal the text the script builds from the archives, so every value is
the deposited one. The notebook takes about 3 min in `.venv` (173 s on a quiet machine on 2026-11-02; 301–433 s with other jobs sharing it);
most of it is the QPHD bootstrap of the second study, the null replicates and subsamples of the penalized
quadratic, and the leave-one-out refits of the OLS direction.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name, in `.venv` in under 1 s.

- Conversion: lossless (`convert_data.py --check`, above).
- Simulator settings: the reduced Newton iterations and tolerance change the pressure and volume features by
  < 3 % (paper, S1 File).
- Stored precision (`check_accuracy.py`): 0.01 mL / 0.01 mmHg (at most 2 decimals written); dp/dt in whole
  mmHg/s (1 mmHg/s against standard deviations of 28–292 mmHg/s); ToR-ORd TTP and RT90 in whole ms (against sd
  19.7 and 68.4 ms).
- No gradients, so no gradient check. The notebook's linear fits have leave-one-out R² 0.731–0.963 for the
  four-chamber outputs (each fold refits the direction), close to (and mostly below) the paper's emulator R²,
  and a ridge-penalized quadratic 0.788–0.975 at its LOO-best penalty. With 407 rows and 1081 quadratic
  coefficients, eigenvalue ratios beyond the first direction are set by the penalty, not by the data.
  **The four-chamber study has no eigenvalue gap that can be quoted**: OLS gives one direction per output and
  no second eigenvalue, and the penalized quadratic's λ₁/λ₂ follows the penalty (for LV peak pressure
  140.6–684.7 at equally good LOO error). The notebook reports the directions, not a gap; its eigenvalue
  ratios are for the ToR-ORd study only.
- Physical plausibility: ESV < EDV in all rows and chambers; 48 rows have LV peak pressure below 80 mmHg
  (min 58.0) and 35 rows an LA maximum v-wave volume below the LA EDV. The paper kept such runs (p. 17); so
  does the notebook (no rows dropped).

## How these files relate to the paper

The four-chamber files are exactly the 407 runs used to train the paper's emulators (Table 4, Figs 7–10);
the ToR-ORd ionic files are the training data of its S2 File stage (and of the code repository's example).
The paper's analysis is a Sobol' analysis of emulators; the notebook analyses the runs directly, with OLS
(and a penalized quadratic as a check of the direction and an illustration of its penalty dependence) for the
four-chamber outputs and QPHD for the cell model, and compares the leading inputs with Figs 8–10 and the fit
quality with Table 4. Because the 30 history-matched inputs are correlated, the notebook reports the share of
the linear fit's variance carried by each group of inputs beside the OLS weights (for LV peak pressure the 26
cell-model inputs hold 0.503 of ‖w1‖² but carry 0.160 of the variance). Two inconsistencies in the paper are
noted in the notebook: the figure captions of Figs 8–10 swap the colours of positive and negative bars
relative to the bars and text, and p. 19 states both that $R_{sys}$ lowers and that it raises LV EDV.
The p. 19 text gives the colours consistently with the bars (blue positive, red negative).
Reference 31 of the paper (Arts et al. 2005, CircAdapt) is printed with the wrong volume and pages
("57(4):H1943–H1943"); Crossref gives 288(4):H1943–H1954.

**Overlap with other cardiac data sets of this repository.** None of the runs is shared with another data
set. The ToR-ORd model (Tomek et al. 2019) of the second study derives from the O'Hara–Rudy model, whose CiPA
variant (CiPAORdv1) is the model of `CiPA_Ventricular_Proarrhythmia_Metrics`; that data set varies channel
block of a different model with its own runs and outputs (action-potential and proarrhythmia metrics), not
the ToR-ORd calcium transient. The Courtemanche atrial model, three of whose parameters enter the
four-chamber study, is also used in `Atrial_Cell_Models_Courtemanche_Maleckar` (single-cell runs, different
inputs and outputs). `Luo_Rudy_Cardiac_Action_Potential` (LR1991), `Rabbit_Ventricular_Action_Potential`
(Shannon rabbit model) and `Hodgkin_Huxley_Action_Potential` are single-cell models with their own data.
This is the only organ-scale (electromechanics plus circulation) data set among them.

## License and credit

Data: © Strocchi et al., Zenodo 10.5281/zenodo.7405335, CC BY 4.0 (as stated for the record); article CC BY.
Cite the paper and the Zenodo record when using these files. `convert_data.py`, `check_accuracy.py` and the notebook: MIT
(this repository). No code of the authors' repository (no licence stated) is included.
