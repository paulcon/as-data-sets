# EU_DEMO_PROCESS_Major_Radius: data

Kind: **stored data**. The outputs come from an outside code, the UKAEA fusion systems code PROCESS
v3.4.2 (MIT licence), run here on 2026-11-01 over the six input intervals of Miralles-Dolz et al.
(2022). The paper's runs and its input deck are not public (no data or code availability statement),
so these are new runs of a related model at the published input ranges, not the paper's data. No value
in the CSV was changed after it was written; the two runs that PROCESS did not converge are kept with
`converged = 0` and left out in the notebook by that rule.

## Files

| file | contents | role |
|---|---|---|
| `EU_DEMO_PROCESS_Major_Radius.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 675; outputs committed |
| `demo_runs.csv` | 64 × 21: `index`, the 6 inputs (physical units), `converged`, `ifail`, `R0`, `P_net`, `t_burn`, `P_fusion`, `B_T`, `q95`, `PsepBt_qAR`, `f_Xe`, `capcost`, `sqsumsq`, `n_iter`, `cpu` (names and units in the second header line) | stored data |
| `generate_data.py` | the design (scrambled Sobol', seed 675), the PROCESS runs (resumable), the assembly of the CSV, and a `check` mode that reruns stored rows; PROCESS's scratch files go to the required `--work` directory | regenerates the file (needs PROCESS; see below) |
| `check_accuracy.py` | recomputes the numbers of this file that the notebook does not print: from `demo_runs.csv` (`--csv-only`), and with PROCESS the reruns of "Accuracy" | checks |
| `process_driver.py` | fills the input-deck template, runs one PROCESS optimisation in-process, reads the MFILE | driver of the outside code PROCESS (not a numpy/scipy model; needs PROCESS v3.4.2) |
| `demo_template_IN.DAT` | the PROCESS input deck with placeholders `{hfact}` … `{f_he}`, `{rmajor0}`, `{maxcal}` | model file (derived from PROCESS's example deck; MIT, not CC BY: see License) |
| `baseline_warmstart.json` | the warm start of every row: PROCESS's optimum at the Table I baseline point (`res`: its outputs, R0 = 8.356952988650034 m; `itv`: the 17 iteration-variable values) | model file; read by `generate_data.py` (output of PROCESS v3.4.2; MIT, not CC BY: see License) |
| `make_template.py` | builds `demo_template_IN.DAT` from PROCESS v3.4.2 `examples/data/large_tokamak_IN.DAT` by a listed set of exact line edits | provenance of the deck |
| `PROCESS_LICENSE.txt` | PROCESS's MIT licence (UKAEA) | licence of the deck it derives from |

`demo_runs.csv` is plain comma-separated numbers after two `#` lines, so `np.loadtxt(path,
delimiter=',')` reads it. Column meanings: `converged` = 1 if PROCESS returned `ifail = 1` (all
constraints met at a VMCON optimum; the paper's "feasible"), else 0; `ifail` is PROCESS's flag (1
converged, 5 "QSP failed to solve"; −1 would mark a Python exception, none occurred); `R0` the
optimised major radius [m]; `P_net` net electric power [MW]; `t_burn` burn time [s]; `P_fusion`
fusion power [MW]; `B_T` vacuum toroidal field at R0 [T]; `q95` edge safety factor; `PsepBt_qAR`
the achieved divertor metric P_sep B_T/(q95 A R0) [MW T/m]; `f_Xe` the xenon seeding fraction (an
iteration variable); `capcost` total capital investment [M$, 1990 US$, PROCESS cost model 0];
`sqsumsq` the root sum of squares of the equality-constraint residuals; `n_iter` VMCON iterations of
the last attempt; `cpu` CPU seconds of the run. For the two failed rows the outputs are PROCESS's last
iterate (here the starting point) and have no physical meaning.

## Model

Citation: E. Miralles-Dolz, A. Pearce, J. Morris and E. Patelli, "Toward DEMO Power Plant Concept
Selection Under Epistemic Uncertainty", *IEEE Trans. Plasma Sci.* 50(11):4440–4445, 2022,
doi:10.1109/TPS.2022.3180233 (CC BY 4.0; checked against Crossref 2026-11-01). Read from the
publisher's PDF (Strathprints copy of the IEEE version).

**The paper's set-up** (p. 4442): PROCESS models "the European H-mode DEMO baseline", "set to minimize
the major radius (R0) constrained with at least 400 MW of net electric power and a pulselength of 2 h";
"a single PROCESS run finishes in a few seconds on an ordinary laptop"; 6600 Latin hypercube samples,
uniform on each interval, with "no probabilistic interpretation". Outputs: R0 and the feasibility flag.

**The code here.** PROCESS (Kovari et al., Fusion Eng. Des. 89(12):3054–3069, 2014,
doi:10.1016/j.fusengdes.2014.09.018; 104:9–20, 2016, doi:10.1016/j.fusengdes.2016.01.007),
https://github.com/ukaea/PROCESS, tag v3.4.2 (commit c0ae5b28649f2b20fb7efc7904628b6defe4151c,
2026-08-14), pure Python. Each run solves: minimise R0 over 17 iteration variables (R0, B_T, electron
temperature and density, β, q95, CS thickness, bore, non-inductive current fraction, TF quench time,
TF nose-case, conduit and winding-pack thicknesses, TF copper fraction, TF current per turn, CS steel
fraction, Xe fraction) subject to 3 equality constraints (beta consistency, plasma power balance,
radial build) and 23 inequality constraints (among them P_net ≥ 400 MW, t_burn ≥ 7200 s, P_sep B_T/(q95
A R0) ≤ the input limit, density and beta limits, B_TF ≤ 14 T, coil stress/quench/temperature-margin
limits, neutron wall load ≤ 2 MW/m², P_fusion ≤ 3000 MW, injected power ≤ 200 MW, τ_α/τ_E ≥ 5, L-H
threshold), with VMCON (sequential quadratic programming; PyVMCON 2.4.3, QP by Clarabel), tolerance
`epsvmc = 1e-7`. Confinement: IPB98(y,2) (`i_confinement_time = 34`), as the paper's H-factor
definition requires.

**The input deck.** The paper's deck is not published. `demo_template_IN.DAT` is PROCESS v3.4.2's
`examples/data/large_tokamak_IN.DAT` ("Generic large tokamak", James Morris, UKAEA, 28/06/23; Morris is
a co-author of the paper), the closest DEMO-like deck in the repository: same objective, 400 MW net
electric, 7200 s minimum burn, κ = 1.85, H = 1.1, core radius 0.75, aspect ratio 3, single null,
pulsed. Every difference from the stock deck (`make_template.py`):

| item | stock deck | here | why |
|---|---|---|---|
| H-factor `hfact` | iteration variable (`ixc = 10`, upper bound 1.2), start 1.1 | input, the study variable | the paper's input 1 |
| He-4 fraction `f_nd_alpha_thermal_electron` | iteration variable (`ixc = 109`, upper bound 0.1) | input, the study variable | the paper's input 6 |
| divertor limit `p_div_bt_q_aspect_rmajor_max_mw` | 10.0 | input (baseline 9.2) | input 2 |
| core radius `radius_plasma_core_norm` | 0.75 | input (baseline 0.75) | input 3 |
| W fraction `f_nd_impurity_electrons(14)` | 5e-6 | input (baseline 5e-5) | input 4 |
| elongation `kappa` (`i_plasma_geometry = 0`, separatrix value) | 1.85 | input (baseline 1.85) | input 5 |
| R0 bounds `boundl(3)`, `boundu(3)` | 8.0, 9.0 m | 6.0, 13.0 m | the stock bounds would clip the paper's [7.929, 10.322] m |
| R0 start `rmajor` | 8.0 m | baseline optimum (see design) | warm start |
| other iteration-variable starts | deck values | baseline optimum | warm start |
| `maxcal` | default 200 | 50 per attempt | bound the cost of failing runs; PROCESS still retries twice |
| `runtitle` | Generic large tokamak | names this study | — |

Known differences from the paper that remain: the PROCESS version (the paper's 2022 code, then
partly Fortran, is not named; v3 "does not guarantee backward compatibility"); the deck (the paper's
EU H-mode DEMO baseline vs. the 2023 generic large tokamak; the paper's other settings, constraints
and iteration variables are not known); the paper's "core radius energy confinement time scaling"
input is read here as PROCESS's `radius_plasma_core_norm` (its baseline 0.75 matches the deck);
for this input the paper's table and prose disagree: Table I names it "Core Radius" (baseline 0.75,
the deck's `radius_plasma_core_norm`), while the prose of item 3 describes "the parameter representing
the fraction of radiation that is released from the core", which is PROCESS's
`f_p_plasma_core_rad_reduction` (old `coreradiationfraction`, 0.6 in the deck, held fixed here). We
follow the table. The input is live: with `i_rad_loss = 1` (the default) the core-region radiation,
times 0.6, inside ρ < `radius_plasma_core_norm` is subtracted from the loss power, and the linear fit
gives R0 a change of +0.157 m across the interval; the W baseline is printed "50^-5" in Table I and read as 5e-5.

**Inputs** (Table I, p. 4441; text items 1–6, pp. 4441):

| name (code) | PROCESS variable | units | baseline | range | source |
|---|---|---|---|---|---|
| `hfact` | `hfact` | – | 1.1 | U[1.0, 1.3] | Table I, item 1 |
| `psepbqar_max` | `p_div_bt_q_aspect_rmajor_max_mw` | MW T/m | 9.2 | U[8.7, 9.5] | Table I, item 2 |
| `core_radius` | `radius_plasma_core_norm` | – | 0.75 | U[0.6, 0.8] | Table I, item 3 |
| `f_w` | `f_nd_impurity_electrons(14)` | – | 5e-5 (printed "50^-5") | U[1e-5, 1e-4] | Table I, item 4 |
| `kappa` | `kappa` | – | 1.85 | U[1.75, 1.90] | Table I, item 5 |
| `f_he` | `f_nd_alpha_thermal_electron` | – | 0.069 | U[0.06, 0.12] | Table I, item 6 |

The W fraction is uniform, not log-uniform (p. 4442: "A uniform distribution was chosen for the
uncertain input parameters").

**Quantity of interest.** R0 [m], the optimised major radius (the paper's output). Chosen over net
electric power (a constraint, equal to 400 MW at every converged run: 400.00 MW in all 62) and capital
cost (not used in the paper, and PROCESS's cost model 0 is in 1990 US$).

**Gradients.** None. R0 is the value of a constrained optimum (piecewise smooth, kinks where the
active set changes) and each run costs about a minute; the notebook uses the value-based QPHD
estimator.

**Sampling design.** The first 64 points of one scrambled Sobol' sequence (`scipy.stats.qmc.Sobol(d=6,
scramble=True, seed=675).random_base2(7)`, of which the first 64 are used), mapped linearly onto the
Table I box; not a Latin hypercube. A Sobol' prefix was chosen so the design can be extended (to 128
points) by running more of the same sequence. 64 runs is the smallest power of two that leaves at
least 2 × 28 = 56 converged runs for the 28-coefficient quadratic of QPHD (two runs per coefficient, a
common rule of thumb); more were not made because a run costs 70 s CPU (median) here, not "a few
seconds". Every run is
warm-started: the iteration variables start at the optimum of the paper's baseline point, itself
solved from the deck's own starting values except R0, started at 9.0 m (the stock deck has `rmajor =
8.0`): R0 = 8.356952988650034 m in 18 iterations. That optimum is shipped as
`baseline_warmstart.json` and `generate_data.py` reads it, so every row can be rerun from exactly the
start it was computed from (R0 depends on the start; see "Accuracy"). `generate_data.py baseline`
recomputes it, and `--recompute-baseline` makes `run`/`check` use the recomputed one; in another
numpy/scipy that need not be bitwise identical (in a second environment the baseline gave
8.356952988650008 m).

**Runs and failures.** 62 of 64 converged. 2 failed, both with `ifail = 5` at the first VMCON step:
row 28 (H 1.138, limit 8.789, core 0.760, f_W 6.36e-5, κ 1.835, f_He 0.0601) and row 58 (H 1.250, limit
8.985, core 0.724, f_W 6.83e-5, κ 1.780, f_He 0.0634). Both are at the lowest He-4 fractions: 2 of the 5
runs with f_He < 0.065 failed, none of the 59 others. Rerun from the deck's own starting values with R0
started at 9.0 m (cold start), both fail again with `ifail = 5`, so the failures are not an artefact of the
warm start (original run; `check_accuracy.py` check 6 repeats it where PROCESS is installed). 9 more
runs converged only after PROCESS's built-in retry (its first attempt hit 50 iterations or a failed QP,
logged as CRITICAL): rows 3, 7, 15, 19, 20, 25, 32, 37, 63 (from the PROCESS log of the original run,
which is not shipped; external run, not reproducible from the repository, although `generate_data.py run`
logs the retries again). No input is outside its physical meaning
(all fractions in (0, 1), elongation > 1).

**Published results** (none is an active subspace):
- Table II: R0 ∈ [7.929, 10.322] m, width 2.393 m (6600 runs); the cited EU DEMO value 9.0 m (p. 4443,
  from Siccinio et al. 2020).
- Table III, pinching scores (Eq. (1)): He-4 0.3935, H 0.2766, elongation 0.1647, W 0.1130, core radius
  0.0186, divertor limit 0.0075.
- Figs. 1–3: R0 decreases with H and with elongation and increases with He-4 fraction; Fig. 2: no
  feasible solution at κ ≤ 1.80; Fig. 4 and text: a feasibility threshold near κ ≈ 1.83, and
  feasibility rising with He-4 fraction.

## Regenerating

The shipped file is the record; PROCESS is rerun only if a problem is found in it. PROCESS is not
installed here, so nothing in this section was rerun for the repository; the commands and times are
those of the original run (Linux x86_64, Ubuntu 24.04, Python 3.12.3, 2026-11-01), where the file was
written.

**Install recipe** (as used for the original run; stop at the first step that fails):

1. `python3.12 -m venv procvenv && procvenv/bin/pip install --upgrade pip` (PROCESS needs Python ≥ 3.11).
2. `git clone https://github.com/ukaea/PROCESS.git && git -C PROCESS checkout v3.4.2`
   (`git ls-remote --tags https://github.com/ukaea/PROCESS v3.4.2` should print c0ae5b28…).
3. `procvenv/bin/pip install "numpy==2.1.3" "scipy==1.14.1" "numba==0.61.2" ./PROCESS`, then check that
   `procvenv/bin/python -c "import process.main"` works (about 6 s, mostly numba and cvxpy).
4. Smoke test, from a scratch directory (PROCESS writes `process.log` into the working directory):
   `procvenv/bin/python <data set>/process_driver.py <data set>/demo_template_IN.DAT r.json 1.1 9.2 0.75 5e-5 1.85 0.069 <data set>/baseline_warmstart.json`.
   `r.json` should have `"ifail": 1.0` and `rmajor` 8.355522 (the warm start at the baseline point;
   `check_accuracy.py` check 7). The first call compiles numba functions (about 1 extra minute) and caches
   them in the site-packages tree; set `NUMBA_CACHE_DIR` to a writable folder if that tree is read-only.

Environment of the shipped runs: Python 3.12.3, Linux x86_64 (Ubuntu 24.04), process 3.4.2, numpy 2.1.3,
scipy 1.14.1, numba 0.61.2, cvxpy 1.7.5, PyVMCON 2.4.3, clarabel 0.11.1, CoolProp 8.0.0, pandas 3.1.0.
Without PROCESS, `generate_data.py run`/`check`/`baseline` and `process_driver.py` stop with an install
message (exit status 3).

```
python generate_data.py run --n 64 --cpu-budget 960 --work <scratch>   # repeat until "all 64 done"
python generate_data.py assemble --work <scratch>                      # writes demo_runs.csv
python generate_data.py check --rows 2,62 --work <scratch>             # rerun stored rows
python generate_data.py baseline --work <scratch>                      # optional: recompute the warm start
```

`--work` is required and must lie outside the data-set directory. `run` and `check` warm-start from the
shipped `baseline_warmstart.json` unless `--recompute-baseline` is given.

`run` is resumable (one JSON line per finished run in `<scratch>/results.jsonl`; gzipped MFILEs in
`<scratch>/mfiles/`) and stops after `--cpu-budget` CPU seconds; it uses one process. Run time of the
original run: 95.9 CPU minutes for the 64 runs (median 70 s, maximum 229 s per run; the baseline adds
54–70 s), in 6 chunks of about 16 CPU minutes, about 2 h wall time on 2 shared CPUs (nice 10). `assemble`
reproduced the shipped file byte for byte from `results.jsonl`; its MD5 is
5754e8d197ae40a73faeeee114d9506d (`check_accuracy.py` check 1). Numerics: numpy 2.1.3/scipy 1.14.1 were
used because they ran PROCESS about 25 % faster than numpy 2.5/scipy 1.18; at the baseline both gave
R0 = 8.3569529886500 m (differing in the 14th digit).

## Accuracy

`check_accuracy.py --csv-only` recomputes the numbers below that come from `demo_runs.csv` (in `.venv`,
2026-11-02, about a second); its checks 5–7 rerun PROCESS and need it installed. PROCESS is not installed
here, so the PROCESS numbers below are from the original run (Linux, 2026-11-01), not recomputed.

- **Repeatability** (original run; `check_accuracy.py` check 5). `generate_data.py check --rows 2,62`
  reran two stored rows with the same warm start: R0 = 7.7746534604 and 8.0884532029 m, identical to the
  file (relative difference 0). Four further warm-started runs at the baseline point gave four identical
  values, 8.355522324264236 m (`check_accuracy.py` check 7 repeats one).
- **Start-point dependence** (original run; `check_accuracy.py` checks 6 and 7). R0 is a local optimum.
  Two rows were rerun from the deck's own starting values (R0 started at 9.0 m, as for the baseline)
  instead of the baseline optimum: row 1 gave 7.985381 m against the stored 7.984470 m (+0.9 mm); row 5
  (H 1.246, κ 1.900) gave 7.487684 m against 7.672240 m (−0.185 m). At the baseline itself the warm start
  gives 8.355522 m against the cold-start 8.356953 m (−1.4 mm). So of the two rows rerun from a cold
  start, one agreed to 0.9 mm and one was 0.185 m lower, more than the 0.056 m standard deviation of R0
  about its one-dimensional trend. The other 60 stored values were not rerun, and how much they depend on
  the start is unknown. The iteration cap of those cold-start reruns is not recorded; `check_accuracy.py`
  uses the stored runs' cap of 50 per attempt. Replacing the two stored values by the cold-start ones
  changes λ1/λ2 from 53.2 to 55.4 and leaves w1 the same (|w1·w1'| = 0.9999; the notebook shows this), but
  two rows cannot show that the other 60 values are start-independent.
- **Constraint residuals** (`check_accuracy.py` check 4). At the 62 converged runs `sqsumsq` (equality
  residuals) is at most 1.76e-8, P_net is 399.999998 to 400.000036 MW and t_burn 7199.999996 to
  7200.000047 s: both on their limits. The achieved P_sep B_T/(q95 A R0) equals the input limit to within
  2.1e-7 (relative).
- No gradients, so no gradient check.

## How these files relate to the paper

The paper's own 6600 runs are not available, and its deck and PROCESS version are not stated. These
are 64 new runs of PROCESS v3.4.2 on a deck from the PROCESS repository edited to the paper's six
inputs and intervals. The model therefore differs from the paper's in ways we cannot quantify: R0 at
the baseline is 8.357 m here (the paper cites 9.0 m for EU DEMO); R0 over the runs spans [7.611, 9.037]
m (the paper: [7.929, 10.322] m with 100 × more samples); and 21 of the 22 runs at κ ≤ 1.80 converge,
where the paper found none. The qualitative findings agree: R0 falls with H and κ, rises with f_He,
and depends little on the divertor limit and core radius; infeasibility appears at low f_He. The
ranking differs (He-4 first in the paper's pinching, third or fourth here). The paper reports no
active subspace; in these runs one direction dominates (λ1 holds 96.5 % of the trace). The size of
λ1/λ2 is not a result of this data set: λ2 to λ6 are at the noise floor of the 28-coefficient quadratic
fit (a fit to pure noise with the residual standard deviation 0.0385 m gives a leading eigenvalue with
95th percentile 6.87e-3, against λ2 = 7.26e-3), the bootstrap interval is wide (8.5 to 55.4), and part of
the scatter that sets that floor may be the optimiser's start dependence, which was checked at two rows
only (Accuracy). The direction w1 is stable (leave-one-out |w1·w1'| ≥ 0.99984; local-linear gradients
|w1·w1_LL| ≥ 0.99981; |w1·w_OLS| = 0.9996).

## Open questions (only the authors can settle them)

1. Input 3. Table I (p. 4441) names it "Core Radius" (0.6–0.8, baseline 0.75), but the prose of item 3
   describes "the parameter representing the fraction of radiation that is released from the core",
   a different PROCESS input (`f_p_plasma_core_rad_reduction`, old `coreradiationfraction`). This data
   set follows Table I (`radius_plasma_core_norm`). Which one did the study vary?
2. The W baseline printed "50^-5" in Table I, read here as 5e-5.
3. The paper's input deck and PROCESS version, which are not published.

## License and credit

Notebook and code (`generate_data.py`, `process_driver.py`, `make_template.py`, `check_accuracy.py`): MIT
(`LICENSE`). Data (`demo_runs.csv`): CC BY 4.0 (`LICENSE-DATA`). Two model files are not CC BY:
`demo_template_IN.DAT` is derived from PROCESS's `examples/data/large_tokamak_IN.DAT`, and
`baseline_warmstart.json` is output of PROCESS v3.4.2 run here. Both are under PROCESS's MIT licence
(`PROCESS_LICENSE.txt`, kept beside them, which reads "Copyright (c) [2023] [United Kingdom Atomic Energy
Authority]"), are hosted with PROCESS credited, and are listed in the exceptions of `LICENSE-DATA`.
Please cite the paper (doi:10.1109/TPS.2022.3180233) and PROCESS (Kovari et al. 2014, 2016, and the
software release). The PROCESS README asks users of a release other than the latest to cite that
release's own DOI from PROCESS's Zenodo page (concept DOI doi:10.5281/zenodo.8335291, which resolves to
the latest release); the v3.4.2 record was not looked up, and v3.4.2's `CITATION.cff` still says
version 3.2.1, so please use the v3.4.2 record listed under that concept DOI. The paper's funding: EPSRC and ESRC CDT in Risk
and Uncertainty, University of Liverpool (EP/L015927/1); EUROfusion (Euratom grant 101052200).
