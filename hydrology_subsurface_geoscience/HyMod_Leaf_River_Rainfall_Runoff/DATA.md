# HyMod_Leaf_River_Rainfall_Runoff: data

Kind: **Python model**. The model is `hymod_model.py` (numpy only), and `generate_data.py` writes `inputs.csv`, `outputs.csv` and `gradients.csv` from it with a saved seed. The forcing, `forcing.csv`, is stored data: the first 365 rows of SAFE's `LeafCatch.txt`, values unchanged. `forcing_other_years.csv` (three other water years of the same file, values unchanged) is used only by the notebook's robustness section.

## Files

| file | contents | role |
|---|---|---|
| `HyMod_Leaf_River_Rainfall_Runoff.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2015`; outputs committed |
| `hymod_model.py` | `simulate(P, rain, pet, grad=False)`, `qoi(P)`, `qoi_and_grad(P)`, `load_forcing()`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `NOMINAL`, `WARMUP`, `QOI_NAMES` | the model |
| `generate_data.py` | writes the three data CSVs; `--check` compares with the shipped ones; `--forcing-from LeafCatch.txt` writes `forcing.csv` and `forcing_other_years.csv` if missing, else checks them | regenerates the data |
| `forcing.csv` | 365 × 4: day, rain, potential evapotranspiration, observed flow (all mm/day) | stored forcing (Leaf River, water year 1949) |
| `forcing_other_years.csv` | 1095 × 5: `LeafCatch.txt` row, water year, rain, potential evapotranspiration, observed flow (mm/day); water years 1950, 1951, 1959 (rows 366–730, 731–1095, 3653–4017) | stored forcing for the notebook's robustness check only |
| `inputs.csv` | 2000 × 5: Sm (mm), beta (-), alfa (-), Rs (1/day), Rf (1/day) | model inputs, physical units |
| `outputs.csv` | 2000 × 3: NSE (-), Qmean (mm/day), Qmax (mm/day) | model output |
| `gradients.csv` | 2000 × 15: column 5 k + j is ∂(QoI k)/∂(parameter j), physical units | model output |

The CSVs are comma-separated numbers after two `#` lines (a description and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. `gradients.csv` reshapes to (2000, 3, 5).

## Model

**Citation.** F. Pianosi, F. Sarrazin, and T. Wagener, "A Matlab toolbox for Global Sensitivity Analysis", *Environmental Modelling & Software* 70:80–85, 2015, doi:10.1016/j.envsoft.2015.04.009 (open access, CC BY 4.0; publisher PDF read). The paper presents the SAFE toolbox and illustrates it on HyMod (Fig. 3, p. 83; Fig. 4, p. 84).

**HyMod** (Boyle 2001, PhD thesis, Univ. of Arizona; Wagener et al., HESS 5:13–26, 2001, doi:10.5194/hess-5-13-2001): a daily conceptual rainfall-runoff model with a soil-moisture store of capacity Sm whose excess-rainfall fraction is 1 − (1 − s/Sm)^beta, evapotranspiration min(s/Sm, 1) × PET, a split alfa of the excess to three quick linear reservoirs (coefficient Rf) and 1 − alfa to one slow linear reservoir (coefficient Rs). The update rules are those of SAFE's `hymod_sim` (SAFEpython 0.2.0 `HyMod.hymod_sim`; SAFE-matlab `example/hymod/hymod_sim.m`): Sm floored at machine epsilon (`np.spacing(1)` / `eps`); soil routine before evaporation, with the soil store clipped to [0, Sm] before evaporation is taken; evaporation W·PET with W = min(|s/Sm|, 1) from the start-of-day store; overflow above Sm and any deficit below 0 added to the excess and the evaporation, so water balances; explicit daily steps with outflows from start-of-day storages; all stores empty at day 1. These rules are not given in a published model description: they were taken from reading SAFE's code (GPL-3.0), and `hymod_model.py` was written fresh from them (vectorized over samples, with its own forward-mode derivative recursion); no SAFE code or comments were copied. The equations are written out in the notebook and in the module docstring. Note that this soil routine is SAFE's version; Boyle's original formulation, with a Pareto distribution of point storage capacities and a critical capacity, was not read and may differ.

**Forcing and output window** (not stated in the paper's text). The paper's Fig. 4 (p. 84) is "part of the Matlab script for the application of the Elementary Effects Test" to HyMod. Its lines for the ranges (`xmin`, `xmax`), `X_labels`, `myfun = 'hymod_nse'`, `r = 100`, `SampStrategy = 'lhs'`, `design_type = 'radial'` and `Nboot = 100` are all in SAFE-matlab's `workflow_eet_hymod.m` (<https://github.com/SAFEtoolbox/SAFE-matlab>, GPL-3.0; "prepared by Francesca Pianosi and Fanny Sarrazin, University of Bristol, 2014"; read at commit dc157e1, 2026-02-23). That script's Step 2, not shown in Fig. 4, reads `rain = LeafCatch(1:365,1)`, `evap = LeafCatch(1:365,2)`, `flow = LeafCatch(1:365,3)` and `warmup = 30`, and `hymod_sim.m` starts all stores at zero. We use exactly these settings: the first 365 rows of `LeafCatch.txt`, a 30-day warm-up and empty stores, so the QoIs use days 31–365 (335 days). SAFE-matlab's `example/hymod/LeafCatch.txt` holds the same 14 610 × 3 values as SAFE-python's `examples/data/LeafCatch.txt` (checked: rows 1–365 equal `forcing.csv`), and SAFE-python's `examples/workflows/workflow_eet_hymod.py` uses the same rows and warm-up (its comment "# 2-year simulation" on the `data[0:365, 0]` line is stale: the slice is one year).

Two caveats on this provenance. (1) The SAFE-matlab history on GitHub starts with an upload on 2022-11-01 (commit 00e8f38, which already has `LeafCatch(1:365,:)` and `warmup = 30`); the 2015 release is not available, so the paper's run cannot be confirmed to have used these settings. (2) The Fig. 4 excerpt is not identical to the current script: its Step 4 reads `Y = model_evaluation(myfun,X)`, where the current script reads `Y = model_execution(myfun,X,rain,evap,flow,warmup)`, and the excerpt has no `warmup = 30` line after `myfun`. The 2015 version therefore passed the forcing differently and may have used no warm-up. The notebook's robustness section shows that scoring all 365 days changes little (NSE on Sm ≥ 10 mm: λ1/λ2 = 12.01 against 12.81; full ranges still n = 2, λ1/λ2 = 1.84; Qmax unchanged at 18.19).

**Warm-up and initial state.** A 30-day warm-up from empty stores is short against the slow store's time constant 1/Rs, which ranges from 10 days to unbounded on Rs ∈ [0, 0.1] /day. The results therefore carry a start-up transient, mostly through the slow store. The notebook's robustness section recomputes the QoIs and gradients at the same 2000 samples with a one-year spin-up (water year 1949 run twice, scored on days 31–365 of the second pass) and in three other water years (1950, 1951, 1959; empty start, 30-day warm-up). Its printed results:

| QoI, range | paper setting | no warm-up | 1-yr spin-up | WY1950 | WY1951 | WY1959 |
|---|---|---|---|---|---|---|
| NSE, Sm ≥ 10 mm: λ1/λ2 (\|w1·w1_headline\|) | 12.81 | 12.01 (1.000) | 5.31 (0.999) | 5.62 (0.997) | 5.43 (0.984) | 3.26 (0.833) |
| NSE, full ranges: n, λ1/λ2 | 2, 1.79 | 2, 1.84 | 2, 1.64 | 1, 2.19 | 1, 4.70 | 1, 15.82 |
| Qmean, full ranges: w1 leader | Sm | Sm | Rs (\|w1·w1_headline\| = 0.185) | Sm | Sm | Sm |
| Qmax, full ranges: λ1/λ2 (\|w1·w1_headline\|) | 18.19 | 18.19 (1.000) | 17.06 (1.000) | 15.48 (0.997) | 7.28 (0.991) | 2.02 (0.844) |

The direction findings are robust: Rf and alfa lead w1 for the NSE on Sm ≥ 10 mm in every setting (in 1959 the Sm weight grows to 0.684), and for Qmax in all but 1959 (where Sm enters w1 at −0.577). The strength of the one-dimensional structure is not: the headline λ1/λ2 = 12.81 and the full-range n = 2 are specific to the first year of the record started from empty stores. In the other three years the full-range NSE is one-dimensional along Sm (weights 0.950–0.995). In the notebook's Qmean discussion, the steep Rs edge on Sm ≥ 10 mm is partly this transient, and under the spin-up the full-range Qmean w1 moves from Sm to Rs.

The `LeafCatch.txt` header says "Leaf Catchment (Mississippi, USA) 1948/10/10 – 1988/09/30" and cites Sorooshian et al. (1983). Its 14 610 rows match 1 Oct 1948 – 30 Sep 1988 (40 water years), not a 10 October start, so the 365 days are probably water year 1949 (1 Oct 1948 – 30 Sep 1949), and the other years above are 1 Oct – 30 Sep windows on the same reckoning.

**Inputs** (m = 5). Independent uniform, "Parameter ranges (from literature)" (Fig. 4, p. 84, `xmin`, `xmax`). The paper gives no units; those below are SAFE's (`hymod_sim` docstring). SAFE's Python workflow attributes the ranges to Kollat et al., WRR 48(3), 2012, doi:10.1029/2011WR011534 (not read).

| name | symbol | units | range | source |
|---|---|---|---|---|
| maximum soil moisture storage | `Sm` | mm | U[0, 400] | Fig. 4, p. 84 |
| exponent of the soil-moisture curve | `beta` | - | U[0, 2] | Fig. 4, p. 84 |
| fraction of excess to the quick reservoirs | `alfa` | - | U[0, 1] | Fig. 4, p. 84 |
| slow reservoir outflow coefficient | `Rs` | 1/day | U[0, 0.1] | Fig. 4, p. 84 |
| quick reservoir outflow coefficient | `Rf` | 1/day | U[0.1, 1] | Fig. 4, p. 84 |

The paper gives no nominal point; `NOMINAL` in the module is the range midpoints (200, 1, 0.5, 0.05, 0.55), used only for the EMA spec and the notebook's illustration.

**Quantities of interest** (over days 31–365):

- `NSE`: Nash-Sutcliffe efficiency of simulated vs observed flow, dimensionless, ≤ 1. The paper's output (Fig. 3 caption; Fig. 4 `myfun = 'hymod_nse'`).
- `Qmax`: peak simulated flow, mm/day (SAFE `hymod_max`, used in SAFE's PAWN workflow; not in the paper).
- `Qmean`: mean simulated flow, mm/day (added here; not in the paper or SAFE).

**Gradients.** Exact derivatives of each smooth piece of the model, propagated forward through the 365-day recursion (`qoi_and_grad`); the clips (store full or empty) make the QoIs continuous and piecewise smooth, so these are the gradients almost everywhere. For Qmax, the derivative of the flow on the day of the peak. Stored with respect to the physical parameters.

**Sampling design.** M = 2000 independent uniform samples (plain Monte Carlo, `numpy.random.default_rng(2015)`, not a Latin hypercube). The paper's own design is an Elementary Effects Test with r = 100 radial one-at-a-time designs on Latin hypercube points, r (m + 1) = 600 runs (Fig. 4); its seed is not reported.

**Edges of the box.** Every sample is inside the paper's ranges and is a valid model input; no sample is outside physical meaning. Two edges are degenerate, and the samples there are kept in the files: Sm < 10 mm (43 samples, 7 below 1 mm), where the soil store holds almost nothing, all rain runs off and almost nothing evaporates; and Rs < 0.001/day (12 samples), where the slow store hardly drains within the year (Rs = 0 exactly would give a slow store that never drains, and Qmean = 0 when also alfa = 0). The 43 samples with Sm < 10 mm carry 52.1 % of tr Ĉ for the NSE and 62.5 % for Qmean, so the notebook repeats the analysis on Sm ∈ [10, 400] mm (a rule applied in the notebook only; no rows are deleted).

**Published results.** The paper reports no active subspace. Fig. 3a (EET for the NSE, read from the figure to about ±0.02): mean |EE| Rf 1.10, alfa 0.90, Rs 0.30, Sm 0.24, beta 0.15; sd(EE) Rf 2.05, alfa 1.09, Rs 0.32, Sm 0.31, beta 0.19. Fig. 3c gives means at 600 runs of about 1.0 (Rf) and 0.88 (alfa) and about 0.2–0.27 for the other three, so Figs. 3a and 3c differ slightly (3a is presumably the bootstrap mean); the notebook compares with 3a.

## Regenerating

```bash
python generate_data.py                                     # writes the three CSVs here
python generate_data.py --out /tmp/hymod --check            # regenerate elsewhere and compare
python generate_data.py --forcing-from <SAFE-python>/examples/data/LeafCatch.txt   # check (or write) the two forcing files
```

The run takes about 1.2 s (2000 samples, vectorized over samples, one process; written 2026-10-31 with numpy 2.5.4, Python 3.13, in a Linux container). `--check` reproduces the files: inputs bit for bit, outputs to 5.3e-16 and gradients to 4.9e-16 (the digits written). `--forcing-from` confirmed that `forcing.csv` equals rows 1–365, and `forcing_other_years.csv` rows 366–730, 731–1095 and 3653–4017, of `LeafCatch.txt` exactly (both the SAFE-python copy, commit 72c64e5, 2024-09-28, and the SAFE-matlab copy, commit dc157e1). On 2026-10-31 the two forcing files were rewritten by `--forcing-from` to add a licence note to their headers; their data lines are unchanged, and `inputs.csv`, `outputs.csv` and `gradients.csv` are unchanged (`--check`: inputs equal, outputs to 5.3e-16, gradients to 4.9e-16). These results are from the container; before merge they are to be re-run in the repository's `.venv` (CLAUDE.md) and replaced by the `.venv` results.

## Accuracy

- **Against SAFE.** SAFEpython 0.2.0 (`pip install safepython`, in a separate scratch venv; the as-data-sets environment does not need it) was run on 2000 uniform samples plus six edge points (Sm = 0, beta = 0, Rs = alfa = 0, Sm = 1e-3, all upper bounds, Sm = 0.5 with beta = 0.01). The daily flows agree to 1.1e-13 mm/day (largest flow 95.4 mm/day), NSE to 2.7e-15 and Qmax to 8.5e-14. `hymod_model.py` runs the 2006 points in 0.17 s; SAFE takes 11 s (a Python loop per sample).
- **Gradients against central differences** (notebook check cell, 200 samples, step h as a fraction of each half-range): with h = 1e-6 the relative error of the gradient vector has median 2–7e-10 and maximum 7.0e-5 (NSE), 5.5e-5 (Qmean), 3.5e-5 (Qmax). With h = 1e-3 the median is 1–3e-6, but 18, 10 and 7 of the 200 samples have errors above 1e-3 (up to 0.14 for NSE and 0.17 for Qmax): the stencil crosses a kink. A scratch check on 500 other samples gave a maximum of 2.5e-8 at h = 1e-6. Finite differences with a moderate step are therefore unreliable at a few percent of the samples; the forward-mode gradients are used instead.
- **The EET with SAFE itself.** SAFE's own EET workflow (`OAT_sampling` with 'lhs' and 'radial', r = 100, `EET_indices`), rerun with `np.random.seed(0..4)`, gives mean |EE| Sm 0.31–0.34, beta 0.17–0.25, alfa 0.82–1.04, Rs 0.19–0.26, Rf 0.70–1.31; with r = 1000: 0.30, 0.22, 0.88, 0.25, 0.91 (sd 0.68, 0.44, 1.27, 0.47, 1.83). The paper's Fig. 3a values lie in these ranges or near their ends, which supports (but does not prove) that the 2015 Matlab run used the same forcing window.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from the 2015 Matlab SAFE toolbox on an unreported forcing window and an EET design with an unreported seed. The data here are a Python computation of the same model, inputs and ranges, on the forcing window and warm-up of SAFE-matlab's current `workflow_eet_hymod.m` (see the caveats under "Forcing and output window"), with a Monte Carlo design and gradients for the active subspace. The notebook also recomputes the EET with its own radial design for the comparison.
- **The model code is a reimplementation.** SAFE-python is GPL-3.0 (its `LICENSE`; the module headers also say "for non-commercial use only", and the paper says the Matlab toolbox is "freely available from the authors for noncommercial research and educational purposes"). `hymod_model.py` re-implements SAFE's `hymod_sim` algorithm: its update rules (listed under "Model") were taken from reading SAFE's code, since no published description gives them, and the module was written fresh in numpy; no SAFE code or comments were copied. It was checked against SAFE numerically (above).

## License and credit

Code (`hymod_model.py`, `generate_data.py`, the notebook): MIT (`LICENSE`). Data written by `generate_data.py` (`inputs.csv`, `outputs.csv`, `gradients.csv`): CC BY 4.0 (`LICENSE-DATA`). Model and example: F. Pianosi, F. Sarrazin and T. Wagener (SAFE), and D. P. Boyle and T. Wagener et al. (HyMod).

`forcing.csv` and `forcing_other_years.csv` are extracts (rows 1–365, and rows 366–1095 and 3653–4017, of 14 610; values unchanged) of `examples/data/LeafCatch.txt` from SAFE-python (<https://github.com/SAFEtoolbox/SAFE-python>, F. Pianosi, F. Sarrazin, T. Wagener and contributors; the same file is in SAFE-matlab). The file itself states no licence (both CSV headers now say so, so the caveat travels with the data); it sits in a repository whose `LICENSE` is GPL-3.0. Its header credits Sorooshian, S., Gupta, V. K., and Fulton, J. L., "Evaluation of maximum likelihood parameter estimation techniques for conceptual rainfall-runoff models", *Water Resources Research* 19(1):251–259, 1983, doi:10.1029/WR019i001p00251; the Leaf River record is usually described as US National Weather Service data (not verified here). It is hosted here with that credit under the author decision of 2026-10-30 on data with no stated licence; whether the repository's GPL-3.0 covers the data file is an open question for the repository owner. If it may not be hosted, delete the two forcing files and obtain them with `git clone https://github.com/SAFEtoolbox/SAFE-python` and `python generate_data.py --forcing-from SAFE-python/examples/data/LeafCatch.txt`, which writes them when they are missing.

Cite F. Pianosi, F. Sarrazin, and T. Wagener, "A Matlab toolbox for Global Sensitivity Analysis", *Environ. Model. Softw.* 70:80–85, 2015, doi:10.1016/j.envsoft.2015.04.009.

## Open questions

- Which forcing window, warm-up and initial state the 2015 Matlab run used. The current SAFE-matlab `workflow_eet_hymod.m` (and SAFE-python's) uses `LeafCatch(1:365,:)`, `warmup = 30` and empty stores; the GitHub history starts in 2022, and the paper's Fig. 4 excerpt has a different Step 4 (`model_evaluation(myfun,X)`, no warm-up argument). Which random seed did the 2015 run use?
- Whether Fig. 3a's marker is the bootstrap mean or the point estimate of the EE statistics (Fig. 3c at 600 runs, Rf ≈ 1.0 and alfa ≈ 0.88, differs from Fig. 3a, 1.10 and 0.90).
- Whether Sm values near 0 (no soil store) were meant to be in the study range: the 43 samples with Sm < 10 mm carry 52.1 % of the NSE's gradient mass.
- Whether `LeafCatch.txt` starts on 1 or 10 October 1948 (header vs row count).
- The licence of `LeafCatch.txt` (no statement in the file; repository GPL-3.0; the underlying observations are US federal data).
