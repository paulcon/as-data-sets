# NH3_H2_Kinetics_Uncertainty: data

Kind: **stored data (generated with Cantera 3.2.0), with a regeneration script.** The model runs in Cantera, an
outside code, so it is not a numpy/scipy "Python model" in the repository's sense (`CONTRIBUTING.md`, "Decide which kind of data
set it is"), even though the Cantera driver is a Python module (`nh3h2_kinetics_driver.py`). The data were generated
for this repository in a Linux container; the shipped CSVs are the record. **Regenerating the data, or rerunning
the accuracy checks, needs Cantera 3.2.0 (`pip install cantera==3.2.0`) and the authors' mechanism file
`NH3_H2.yaml` from the paper's supplement**, which is not distributed here (see "License and credit"). Without
either, `generate_data.py` and `check_accuracy.py` stop with an error that says what to install or where to get
the file. The notebook needs only numpy, matplotlib and active_subspaces: it reads the shipped files and the
driver's constants.

## Files

| file | contents | role |
|---|---|---|
| `NH3_H2_Kinetics_Uncertainty.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 470; outputs committed |
| `nh3h2_kinetics_driver.py` | `KineticsModel` (Cantera driver: `idt`, `idt_grad`, `idt_grad_central`, `lfs`, `lfs_grad`, `lfs_grad_fd`); the constants `TABLE` (the supplement's 52 entries), `NAMES`, `LOWER`, `UPPER`, `F_INI`, `F_RED_L`, `F_RED_U`, `CANTERA_INDEX`, `LIMIT`, `label()`, the conditions and `MECH_SHA256` | the model; its constants import without Cantera |
| `generate_data.py` | writes the CSVs (`--case idt / lfs / nominal`, `--out`, `--rows`, `--check`, `--max-seconds`, `--mech`) | regenerates the data (needs Cantera 3.2.0 and `NH3_H2.yaml`) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" below (`--quick`: checks 1–3; `--flames`: 4–5; `--adjoint`: 6; `--path`: 7) | accuracy check (same needs) |
| `idt_inputs.csv` | 1000 × 52: x_i = log10(A_i/A_i0), columns x01..x52 | Latin hypercube, seed 470001 |
| `idt_outputs.csv` | 1000 × 1: ignition delay τ in s | model output |
| `idt_gradients.csv` | 1000 × 52: ∂τ/∂x_i in s per log10 unit | model output (forward differences) |
| `lfs_inputs.csv` | 200 × 52: x_i, as above | Latin hypercube, seed 470002 |
| `lfs_outputs.csv` | 200 × 3: S_u in cm/s, number of grid points, flag (1 if the flame had to be solved from scratch) | model output |
| `lfs_gradients.csv` | 200 × 52: ∂S_u/∂x_i in cm/s per log10 unit | model output (discrete adjoint) |
| `nominal.csv` | 2 × 53: row 1 τ (s) and ∂τ/∂x_i; row 2 S_u (cm/s) and ∂S_u/∂x_i, at x = 0 | the nominal model |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the 16 written significant digits before the model runs,
so each output belongs exactly to the stored input row. No file needs `NH3_H2.yaml` to be read.

## Model

### Citation

G. Huang, H. Wang, L. Tian, O. Haidn, and N. Slavinskaya, "Uncertainty quantification and reduction for
combustion kinetic Modeling: A case study of NH3/H2 models", *Fuel* 400 (2025) 135810,
doi:10.1016/j.fuel.2025.135810 (Crossref-checked; open access, CC BY 4.0). The PDF read is the publisher's
version. Supplement (Appendix A, p. 12): `mmc1.xlsx` (the 52 reactions with A, n, Ea, temperature range, initial
and reduced uncertainty factors), `mmc2.xlsx` (experimental data and the sensitivity-analysis conditions),
`mmc3.zip` (the mechanism in Cantera and Chemkin formats), `mmc4.docx`.

### What the paper does

The authors identify 52 highly sensitive reactions of NH3/H2 combustion from a sensitivity analysis of 11
published models (Sec. 3.1, pp. 4–5; Fig. 3, p. 6), build a new mechanism with 42 species and 346 reactions
(Sec. 3.2, p. 5), set initial uncertainty bounds for each rate constant from the literature (Sec. 3.3, pp. 8–9),
and then sample "the pre-exponential factor (A) of 52 highly sensitive reactions [...] within their initial
uncertainty bounds" by Latin hypercube, 100,000 samples (Sec. 3.4, p. 9). The modified models are run against
about 2,500 experimental targets (IDT, LFS, JSR/PFR speciation), and the samples whose squared relative error
ψ (Eq. 9, p. 3) lies in [−0.5σψ, 0.5σψ] give reduced bounds A_ave ± 0.5σ_A (p. 10). The paper does not compute
active subspaces or gradients; its sensitivities are brute-force local coefficients (Eq. 1, p. 2).

### Physics and solver (this data set)

- Rate constants k(T) = A T^n exp(−Ea/T) (paper Eq. 2, Ea in K). Mechanism: the supplement's `NH3_H2.yaml`,
  sha256 `23289472b013db8b4095448419c62996035b0f02265e0d5569b88f2a5d87284a` (the file in `mmc3.zip`), loaded
  unchanged in Cantera 3.2.0. Each input multiplies one entry's A; for a falloff reaction only the named limit's
  A (low- or high-pressure) changes, and the Troe rate is rebuilt with the new limit (the other limit and the Troe
  coefficients unchanged).
- **Ignition delay τ.** Adiabatic constant-volume reactor (Cantera `IdealGasReactor`, CVODES, rtol 1e-9,
  atol 1e-15) from 4.375 % NH3 / 3.281 % O2 / 92.344 % Ar (φ = 1) at T5 = 1587.982151 K, p5 = 1013250 Pa
  (10 atm). These are the mixture of the paper's Fig. 10(b) (p. 11) and the exact state of the Chen et al. (2021)
  shock-tube point 15 in the supplement's IDT sheet (`mmc2.xlsx`, measured τ = 0.4 ms); the temperature is this
  data set's choice of one point on the figure's 1000/T5 axis (0.630). τ is the first time T reaches T5 + 400 K,
  located by cubic Hermite interpolation of T between the bracketing solver steps with the exact dT/dt. The paper
  does not state its ignition criterion and mentions pressure-rise corrections "where such data were available"
  (p. 3); none is applied here (choices of this data set).
- **Laminar flame speed S_u.** Cantera `FreeFlame`, mixture-averaged transport, NH3/H2 50/50 by mole with air
  (O2 : N2 = 1 : 3.76), φ = 1, Tu = 298 K, p = 1 atm: the conditions of the paper's Fig. 10(c) (p. 11) at φ = 1.
  Grid refinement: ratio 3, slope 0.1, curve 0.2, initial width 3 cm. The paper used ratio 2, slope 0.01,
  curve 0.02 (p. 4); the coarser grid is a cost choice of this data set, and check 5 below measures its effect.
  Every flame is solved from the same initial guess (the converged nominal flame) and then refined; if that fails,
  the flame is solved from scratch (flag column; never needed in the shipped files). The result still depends on
  the solver history: Cantera's `Sim1D` object keeps state from earlier solves, so the refined grid, and with it
  S_u, can change with the flames the same model instance solved before, by up to about 3 % at slow and mid-range
  flames ("Path dependence" under Accuracy). The shipped rows are reproducible by replaying the runs that wrote
  them ("Regenerating").

### Inputs

52 inputs, independent and uniform: x_i = log10(A_i/A_i0) on [−f_ini,i, f_ini,i], where A_i0 is the A of entry i
in `NH3_H2.yaml` and f_ini,i the initial uncertainty factor of the supplement's table (`mmc1.xlsx`, column
"Initial uncertainty bounds f_l^ini = f_u^ini"; paper Eqs. 5–6, p. 3). f_ini ranges from 0.301 to 2.477, a factor
of 2 to 300 on A. The 25 entries of the paper's Table 4 (p. 7) carry its labels (R1–R23, with R9L/R9H and
R22L/R22H); the 27 others are only in the supplement and are labelled S<row>. The full table (reaction, A0 as
printed, f_ini, f_red_l, f_red_u, temperature range) is in the driver (`TABLE`) and printed by the notebook.

- **Uniform in log10 A.** The paper says A was "randomly and uniformly sampled within [its] initial bounds"
  (p. 3; LHS, p. 9) without naming the scale. It reports A on a log10 scale (Fig. 9, p. 10: "the gray solid line
  denotes log10(A0)"; the R1 reduced bounds 13.897 and 14.216, p. 10) and defines the bounds as factors on k
  (Eqs. 5–6). Sampling uniformly in log10 A is this data set's choice, and the paper's reduced bounds support it
  strongly. They are A_ave ± 0.5σ_A of the filtered samples (p. 10), and for a log-uniform distribution that the
  filtering leaves unchanged, ± 0.5σ spans 1/(2√3) = 0.2887 of the initial width, centred on the nominal value. In
  the supplement's table, 41 of the 52 width ratios (f_red_l + f_red_u)/(2 f_ini) lie between 0.2800 and 0.2881,
  and every reduced interval is centred on the nominal value to within 0.049 f_ini (largest offset R1). Sampling
  uniform in A would put the centres of the log10-A statistics 0.22 f_ini (f_ini = 0.301) to 0.83 f_ini
  (f_ini = 2.477) above the nominal value, with width ratios from 0.28 down to 0.09. These numbers are printed by
  the notebook (the input table).
- **A0 in the file and in the table.** For 7 entries the A in `NH3_H2.yaml` differs from the A printed in the
  supplement's table (and, for R3, R9L, R13 and R18, in Table 4): R3 2.2e8 vs 2.16e8, S13 1458.3 vs 1463, R9L 3.6e22 vs
  3.61e22, R13 4.3e10 vs 4.29e10, S37 3.84e40 vs 3.8e40, S40 1.3e13 vs 1.31e13, and **R18 (NH + NO <=> H + N2O)
  2.25e15 vs 2.70e15** (a factor 0.833, −0.079 in log10). S3 (4.6508e12 vs 4.650944e12) and S12 (5.636 vs 5.64)
  differ only in rounding. The inputs are factors on the file's A, the nominal model.
- **n and Ea in the file and in the tables.** These are held at the file's values (they are not inputs), but they
  are part of the nominal model. Differences between `NH3_H2.yaml` and the supplement's table (and Table 4, p. 7,
  for the R entries): **R2 (H2 + O <=> OH + H) Ea 3166.26 K vs 3133.1 K**, so k is 2.1 % lower at 1588 K and
  10.5 % lower at 298 K than with the tables' value; S13 Ea 27422.10 vs 27400 K (k 1.4 % lower at 1588 K); S37 and
  S38 (N2H3 (+M) <=> N2H2 + H (+M), low and high limits) Ea +20.1 K (k 1.3 % lower at 1588 K); R13 n 0.294 vs 0.29
  and Ea −435.79 vs −438.0 K; R4 and R20 Ea −3.5 K, S46 −2.4 K; R15, R17 and S12 below 1 K. In the supplement's
  row for S36 (N2H2 + NH2 <=> NH3 + NNH) the n and Ea columns are shifted: n = 5.433 (which is log10 of its A,
  2.71e5) and Ea = 2.226 (the file's n); the file has n = 2.226 and Ea = −518.32 K. R1's Ea is in "Unknowns".
- **Units.** A in the table is in cm, mol, s (and K); x_i is dimensionless, so the inputs do not depend on units.
- **Reduced bounds.** `F_RED_L`, `F_RED_U` hold the supplement's reduced factors (the reduced box is
  [−f_red_l, f_red_u]); they are not used to sample, only in the notebook's section on the reduced bounds.

### Quantities of interest and gradients

- `idt_outputs.csv`: τ (s). The notebook analyzes log10(τ/1 s), whose gradient is ∇τ/(τ ln 10).
  `idt_gradients.csv`: ∂τ/∂x_i (s per log10 unit) by forward differences with step 1e-3 in x_i (52 extra solves
  per sample).
- `lfs_outputs.csv`: S_u (cm/s). `lfs_gradients.csv`: ∂S_u/∂x_i (cm/s per log10 unit) from Cantera's discrete
  adjoint on the converged grid (`Sim1D.solve_adjoint`; parameter i scales entry i's A by 1 + p_i, so the adjoint
  gives ∂S_u/∂ln A_i, times ln 10). The grid is fixed in the derivative (the refinement is not differentiated).
  The adjoint gradients are exact in direction but short in size, by 0.5–16 %, most at slow flames (Accuracy,
  check 6); this does not change the subspace.
- Gradients are in the physical inputs x_i. The notebook normalizes once, through `ac.Uniform(lower, upper)`,
  which scales each input by its own half-range f_ini,i (for uniform inputs this is also the CDF scaling).

### Sampling design

One Latin hypercube per quantity on the box above, drawn with numpy only (`generate_data.lhs`: for each column a
random permutation of the N strata plus a uniform jitter in each stratum; `default_rng`), so the design does not
depend on the scipy version. Ignition delay: N = 1000, seed 470001. Flame speed: N = 200, seed 470002. The paper's
own design was a Latin hypercube of 100,000 samples (p. 9) with no seed reported; of these it used 30–50 randomly
chosen models per experimental condition (p. 3).

**Size.** The design was sized to what 2 shared CPUs could do in a few hours: an ignition sample (53 solves)
costs about 1.8 s of CPU and a flame sample (one solve plus one adjoint) about 4 s, so 1000 + 200
samples took about 45 minutes of CPU. 1000 gradient samples of τ resolve the leading eigenvectors of a 52-input
problem well (bootstrap distances in the notebook). The flame is the costlier and harder solve, so it has 200.

### Published results (paper)

- No active subspace, gradient or global sensitivity result. The overall sensitivity coefficient of
  H + O2 <=> O + OH (R1) is the largest in all 11 models (Fig. 3, p. 6), and R1 is "the most sensitive across
  these three conditions" of Fig. 2(a) (p. 4; NUIG2024 model).
- Fig. 10(b), (c) (p. 11): Monte Carlo predictions with the initial and the reduced bounds for the IDT of the
  mixture used here (p = 10 atm, a range of T5) and for the LFS of 50 % NH3 + 50 % H2 / air at 298 K, 1 atm, against
  φ. With the initial bounds, the predictions spread over about two decades in τ and over roughly 0–70 cm/s in
  S_u (read from the figure; the S_u axis ends at 70 cm/s). After showing the samples of R4, R13 and R18 in
  Fig. 10(a), the paper says "for reactions other than these three, the width of the reduced bounds is
  approximately 0.2–0.3 times that of the initial bounds" (p. 10); in the supplement's table the ratio is
  0.221–0.288 for all 52 entries (printed by the notebook).

### What the notebook finds (from these files)

- Checks: no failed solve (no NaN row), no flame solved from scratch; both designs have one point per stratum in
  every column; the least-squares slope and the mean normalized gradient agree in direction (cosine 0.988 for
  log10 τ, 0.979 for S_u). τ runs from 0.0555 to 10.634 ms (median 0.7497 ms; nominal 0.6907 ms), S_u from 4.41 to
  125.98 cm/s (median 33.59; nominal 40.69 cm/s; 113–179 grid points).
- log10 τ: λ1/λ2 = 6.14, then no gap (ratios 1.13, 1.14, 1.15, 1.11). λ1 holds 47.3 % of the trace. Bootstrap mean
  subspace distance 0.041 (k = 1) and 0.378 (k = 2). w1 is led by R1 (−0.444), S13 (+0.411), R22L (−0.372), R10
  (−0.318) and R14 (+0.283); its five largest components hold 0.685 of the squared norm. Leave-one-out R²:
  0.7376, 0.7372, 0.7373 for polynomials of degree 1–3 in y1, 0.7565 for a cubic in (y1, y2), 0.7340 for a linear
  function of all 52 inputs. Chosen n = 1; about a quarter of the variance is spread over many directions.
- S_u: λ1/λ2 = 10.22, λ2/λ3 = 4.21; λ1 holds 82.2 % of the trace, the first two 90.2 %. Bootstrap distances 0.040
  (k = 1) and 0.105 (k = 2). w1 is dominated by R1 (+0.923), then S19 (−0.216), R5, R2, R15; w2 is almost S19 alone
  (−0.944). Leave-one-out R²: 0.8820, 0.9332, 0.9324 (degree 1–3 in y1), 0.9440 (cubic in (y1, y2)), 0.8460 (linear in
  all inputs). Chosen n = 1 (gap rule); n = 2 is also defensible.
- |w1(log10 τ) · w1(S_u)| = 0.501: R1 leads both, but the ignition direction is shared among many NH_x/NO
  reactions.
- The paper's reduced bounds shrink the spread of y1 to 0.268 (log10 τ) and 0.229 (S_u) of its initial standard
  deviation; the one-dimensional surrogates' 95 % ranges shrink by 0.286 and 0.244 (an illustration with the
  surrogates, not a model run). For most entries the reduced box is ± 0.5σ of the unfiltered log-uniform prior
  (width ratio near 0.2887, see "Inputs"), so these factors mostly reflect that rule. The ten entries that the
  filtering narrowed below 0.279 (R1 0.221, R14 0.268, R22L, R15, R7, R10, R6, R5, S30 and R11) include six of
  the ten largest values of w1(log10 τ)² + w1(S_u)²; the correlation of 0.2887 − ratio with that sum is 0.60
  (Spearman) and 0.93 (Pearson; 0.54 without R1). So the reactions the filtering narrowed tend to be those the active
  directions weight most.
- Flame-gradient size: dividing the stored flame gradients by the fitted adjoint shortfall of check 6 changes
  λ1/λ2 to 10.223 and λ2/λ3 to 4.216 and leaves w1 and w2 unchanged (|w·w′| = 1.00000).
- Signs: `ac.orient(by="mean_gradient")` (each quantity increases along w_j on average); the paper shows no
  eigenvectors.

## Regenerating

```bash
# 1. Cantera environment (Python 3.13 was used)
python3 -m venv cantvenv && cantvenv/bin/pip install cantera==3.2.0 numpy scipy
# 2. The mechanism (not in this repository): open https://doi.org/10.1016/j.fuel.2025.135810, go to
#    "Appendix A. Supplementary data" and download the zip file of the mechanism (mmc3.zip); unzip it.
#    It holds NH3_H2.yaml (Cantera), NH3_H2.inp, therm_NH3_H2.dat and trans_NH3_H2.dat (Chemkin).
#    Put NH3_H2.yaml next to nh3h2_kinetics_driver.py (or pass --mech PATH, or set NH3H2_MECH=PATH) and check:
sha256sum NH3_H2.yaml   # 23289472b013db8b4095448419c62996035b0f02265e0d5569b88f2a5d87284a
# 3. Runs
cantvenv/bin/python generate_data.py --case idt --out /tmp/nh3        # 1000 ignition samples
cantvenv/bin/python generate_data.py --case lfs --out /tmp/nh3        # 200 flame samples
cantvenv/bin/python generate_data.py --case nominal --out /tmp/nh3    # nominal.csv
cantvenv/bin/python generate_data.py --check                          # all three cases, in a temporary directory, vs the shipped files
cantvenv/bin/python generate_data.py --case idt --rows 0:10 --check   # a subset vs the shipped files
cantvenv/bin/python generate_data.py --case lfs --rows 4:30 --check   # flames: start at a run start
cantvenv/bin/python check_accuracy.py                                 # the Accuracy numbers
```

`--out` is required and must be outside this directory, so the shipped files are never overwritten; with
`--check` it defaults to a temporary directory (removed at the end), and `--case` to all three cases in turn. Runs are
resumable (`<out>/<case>_rows.txt`; `--max-seconds` stops cleanly between rows). `--check` with `--max-seconds`
needs `--out DIR`; a stopped check exits 1 without comparing. `--check` prints the largest differences from the
shipped rows: for ignition delay and `nominal.csv`, the relative maximum difference of the outputs and the relative
2-norm per gradient row; for the flames, per row, the relative difference of S_u and the relative 2-norm
difference of the gradient, with the number of rows that agree to 1e-6 and 1e-3 (the same solver history). A
different solver history gives up to 3.2 % in S_u and 8.3e-2 in the gradient ("Path dependence" under Accuracy,
row 24), so rows above 1e-6 can be the same flame to the accuracy of its grid, not the same digits. The largest
differences observed are below (Linux container, Python 3.13, Cantera 3.2.0; Cantera is not installed on the macOS
machine, so nothing was measured there). `--check` exits 1 only on a structural mismatch: a different shape or set
of nan entries, or inputs that differ by more than a relative 1e-8.

**Reproducing the flame rows exactly.** The shipped flame rows were written in seven runs, each a new process and
so a new model instance, that started at rows 0, 4, 51, 55, 92, 133 and 166 (rows 0–3 in a timing run, then
`--max-seconds` chunks and container restarts). `generate_data.py` (constant `LFS_SESSIONS`) starts a new model
instance at each of these rows, so a run that starts at one of them and is not stopped by `--max-seconds`
replays the original history. Measured in the generating container: `--rows 4:30` reproduces all 26 rows to the
written digits (the check prints a difference of 0 for S_u and the gradients), and a run of `--rows 0:51`
reproduced rows 0–26 to 3e-13 in S_u and 2e-11 in the gradients (the rounding of the written digits) before a
container restart stopped it; `--rows 166:200` (the last run, which includes rows 187 and 192) reproduces all
34 rows to the written digits (432 s). A run that starts elsewhere solves the same flames after a
different sequence: `--rows 20:30` gives rows 24 and 26 that differ by −3.22 % and +1.73 % in S_u (gradients
3.7e-2 and 1.4e-2), with the other eight within 1.4e-5.

Provenance of the shipped files: generated 2026-11-08 in three sessions separated by container restarts (ignition
rows 0–161, then rows 162–999 and flame rows 0–54, then flame rows 55–199 and `nominal.csv`; the Cantera venv was
recreated with the same versions after the first restart) in a Linux container (x86-64, 2 cores shared with about
six other jobs; load average 3–8), Python 3.13.18, Cantera 3.2.0, numpy 2.5.4, scipy 1.18.1, with
`generate_data.py` as shipped, one process at a time under `nice`, in chunks of at most 9 minutes. Wall time
about 1 h 55 min (heavily shared CPU; the logged times per row sum to 4138 s for the ignition rows and 2598 s for
196 flame rows); CPU time about 45 minutes (measured with `--check` runs: 19.6 s of user time for 10 ignition rows
and 30.8 s for 5 flame rows, the latter including the nominal flame that every flame run solves first). The shipped
driver and `generate_data.py` differ from the ones that wrote the data only outside the generating path: `idt()`
restores the gas state on return, so that a later flame in the same instance keeps its pressure; a `refine`
argument of `lfs()` takes effect; the error messages are clearer; and `generate_data.py` has the new instance at
the run starts above and the per-row flame check. With the shipped scripts, `--check` reproduces ignition rows 0–2 and
`nominal.csv` by 0, and flame rows as described above.

## Accuracy

`check_accuracy.py` recomputes every number in this list (it needs Cantera and the mechanism; in the generating
container, Cantera 3.2.0, numpy 2.5.4, checks 1–5 ran in 1407 s, 21.5 min of CPU; checks 1–3 again with the
inflection criterion added (`--quick`) in 124 s, check 6 (`--adjoint`) in 484 s and check 7 (`--path`) in 202 s;
each printed the values below). The rows are fixed by rules on the stored files: the shortest, median and longest
τ (rows 268, 575, 280), the slowest and fastest flame (rows 149, 168), the flames at ranks 0, 10, …, 190 and 199 of
the stored S_u (check 6), and in check 7 rows 24, 26 and 187 (the rows that differ in a `generate_data.py --check`
run of rows 20:30, and a slow flame) and row 192 (a slow flame).

- **Nominal model** (check 1): τ = 0.69075 ms and S_u = 40.6918 cm/s (110 grid points), equal to `nominal.csv`.
- **Ignition criterion** (check 2). τ for a rise of 200 K or 800 K instead of 400 K changes by −5.5 % and +5.2 % at the
  nominal point, by −9.9 % and +2.7 % at the longest τ (10.63 ms), and by −4.3 % and +79 % at the shortest τ
  (0.0555 ms). The constant-volume equilibrium temperature rise at the nominal point is 876 K, so a rise of 800 K
  sits at the end of the heat release, where the time to reach it is sensitive. The time of the maximum of dT/dt
  (the inflection of T, a common shock-tube criterion) is later than the 400 K rise by 1.7 % (nominal), 1.8 %
  (shortest τ), 4.3 % (median τ, row 575) and 2.6 % (longest τ), and its gradient of log10 τ has cosine 0.99994,
  0.99959, 0.99973 and 0.99984 with the 400 K gradient (norm ratios 0.992–1.011; both by central differences with
  step 1e-2). At the shortest τ the 800 K rise gives cosine 0.18 (norm ratio 0.666): the +79 % there and the
  change of direction are a property of the 800 K criterion, which sits at the end of the heat release. So the
  400 K criterion is a stated choice that does not matter for the subspace: the inflection criterion gives the
  same gradients to a cosine of 0.9996 or better.
- **τ** (check 3). Re-solving at rtol 1e-12, atol 1e-20 changes τ by 2.1e-9, 1.1e-8 and 2.0e-8 relative (rows 268,
  575, 280).
- **∂τ/∂x** (check 3). The stored forward differences against central differences at rtol 1e-12 (relative 2-norm
  error in normalized coordinates): 6.9e-4, 8.8e-4 and 6.0e-4, the size expected of a forward difference with step
  1e-3.
- **∂S_u/∂x** (check 4). The stored adjoint gradients against central differences on the same grid (step 1e-2, each
  perturbed flame solved twice from the unperturbed one at rtol 1e-9): relative error 7.9e-3 (nominal), 3.9e-3
  (fastest flame, 125.98 cm/s) and 1.6e-1 (slowest flame, 4.41 cm/s). The directions agree in all three (cosine
  1.00000, 0.99998, 1.00000); at the slowest flame the adjoint vector is shorter, with norm 0.839 of the
  differences' (0.992 and 0.996 at the other two). During development, central differences with steps of 3e-3
  or less that solved each perturbed flame only once differed from the adjoint by about 15 % in some components at
  the nominal point (the Newton iteration stopped early); solving twice removed this (external run, not
  reproducible from the repository).
- **Size of ∂S_u/∂x across the design** (check 6). At 21 flames spread over the design (ranks 0, 10, …, 190 and
  199 of the stored S_u), the adjoint directional derivative along the adjoint direction is short of central
  differences on the same grid by a factor that grows as the flame slows: 16.1 % at the slowest flame (4.41 cm/s),
  on average 8.0 % below 15 cm/s, 3.1 % for 15–30 cm/s, 1.7 % for 30–50 cm/s and 0.5 % above 50 cm/s (3.2 % at
  row 154, 33.8 cm/s, next to the median 33.59 cm/s). The two difference steps (1e-2, 3e-3) agree to 0.4 %, and
  along a random direction the ratio is about the same (0.839 against 0.839 at the slowest flame; within 3.1 %
  elsewhere), so the adjoint gradient is close to a uniformly scaled copy of the true one: its direction is
  right (cosine 1.00000 with the differences at the three points of check 4) and its length is short. The cause
  was not isolated (the differences are consistent between steps and directions, so the shortfall is in the
  adjoint of Cantera 3.2.0 on these flames, not in the differences). A fit 1 − ratio = 1.382 S_u^−1.281 (S_u in cm/s; it overstates
  the slowest flame, 20.6 % against the measured 16.1 %) puts 67 % of the 200 design rows more than 1 % short,
  26 % more than 3 % and 14 % more than 5 %. **The subspace is unaffected:** dividing every stored gradient by the
  fitted ratio changes λ1/λ2 from 10.218 to 10.223 and λ2/λ3 from 4.209 to 4.216, raises λ1 by 2.6 %, and leaves
  |w1·w1′| and |w2·w2′| at 1.00000 (distance of the two-dimensional subspaces 0.0021). The notebook repeats this
  rescaling from the ratios as printed to four digits, so its fit reads 1.390 S_u^−1.283 (20.7 % at the slowest flame). So the flame gradients are accurate in direction everywhere; their magnitudes are biased low by
  0.5–16 %, most at slow flames, which changes the eigenvalues by a few per cent and the eigenvectors not at all.
- **Flame grid** (check 5). With the paper's refinement criteria (ratio 2, slope 0.01, curve 0.02; p. 4), S_u is
  41.3072 cm/s on 977 points at the nominal point, 1.51 % above the stored 40.6918 cm/s, and the adjoint
  gradients on the two grids have cosine 0.99796. At the slowest flame S_u is 4.7539 cm/s on 908 points, 7.68 %
  above the stored 4.4149 cm/s. The fastest flame finds no solution on the paper's grid within Cantera's default
  limit of 1000 grid points. The coarser grid of the shipped data therefore underestimates S_u, by about 1.5 % near
  the nominal flame and more for slow flames; the gradient direction at the nominal point is barely affected.
- **Path dependence of the flame solve** (check 7 and `generate_data.py --check`). A flame solved after a
  different sequence of flames, in the same model instance or in a fresh one, can differ from the stored value,
  because the refined grid changes. In a fresh instance: row 24 gives 19.1937 cm/s on 138 points against the stored
  19.7492 cm/s on 143 points (−2.81 %), and solving it a second time in the same instance gives 19.1378 cm/s on
  139 points; row 187 gives 11.9563 cm/s against 12.1892 cm/s (−1.91 %); row 26 gives 17.2664 cm/s against
  17.2662 cm/s. In a run of rows 20:30, row 24 gives 19.1139 cm/s (−3.22 %) and row 26 17.5645 cm/s (+1.73 %). Of
  the 21 flames of check 6 (fresh instances), 17 agree with the stored values to 3.5e-5 or better and 4 differ:
  rows 24 (−2.81 %), 34 (+2.34 %), 151 (−2.21 %) and 113 (−0.07 %). So at slow and mid-range flames S_u depends on the
  solver history by up to about 3 %, the same size as the grid error of check 5: it is the numerical accuracy of
  the flame on this grid, and part of the scatter in the notebook's sufficient summary plot. The gradients
  follow the grid (row 24: relative 2-norm difference 8.3e-2 in a fresh instance) but keep their direction
  (cosine with the stored gradient 0.99995 or more in check 6). The shipped rows are reproduced exactly by replaying
  the original runs ("Regenerating").
- **Smoothness of S_u in x.** The adjoint holds the grid fixed. Differences of S_u between flames refined on their
  own grids (each in a fresh instance, step 2e-2 along the normalized adjoint direction) give 46.7129 cm/s per
  normalized unit at the nominal point against the adjoint's 45.8255 (which is 0.8 % short there, check 4), and
  10.9199 against the adjoint's 9.5446 at row 192 (S_u 7.37 cm/s; the fit of check 6 puts the adjoint about 11 % short
  at this speed, which puts the fixed-grid derivative near 10.7). With every flame solved from the same history, the
  refined-grid derivative is within about 2 % of the fixed-grid one at these two points. Grid changes between
  solves still make S_u jump by up to a few per cent at slow and mid-range flames (the path dependence above);
  these jumps are part of the scatter of the stored S_u, which the fixed-grid gradients do not see.

## How these files relate to the paper

- **A new computation with the paper's model.** The paper published its mechanism and uncertainty factors but not
  its 100,000 samples or their predictions ("Data will be made available on request", p. 12). These files come
  from the authors' mechanism and the authors' initial bounds, sampled the way the paper describes (LHS of the A
  factors within the initial bounds), at one ignition and one flame condition of its Fig. 10, with gradients the
  paper does not compute. The settings the paper does not state (ignition criterion, no pressure rise, log-uniform
  sampling) and the coarser flame grid are this data set's choices.
- The comparison table at the end of the notebook lists what can be compared.

## Unknowns

1. Whether the paper samples A uniformly in log10 A (assumed) or in A. The scale is not stated, but the reduced
   bounds point strongly to log10 A: 41 of 52 width ratios lie between 0.2800 and 0.2881, just under the
   1/(2√3) = 0.2887 of ± 0.5σ of a log-uniform distribution, and all reduced intervals are centred on the nominal
   value to within 0.049 f_ini ("Inputs").
2. The paper's ignition criterion and whether a pressure-rise model was used for the Chen et al. (2021) data at
   10 atm (p. 3: "where such data were available"). The choice barely matters for the subspace: the inflection
   (maximum dT/dt) criterion gives gradients of log10 τ with cosine 0.9996 or more against the 400 K rise
   (check 2).
3. R18's A: 2.25e15 in `NH3_H2.yaml` against 2.70e15 in Table 4 and the supplement's table (and six smaller
   differences, listed above); and R2's Ea: 3166.26 K in the file against 3133.1 K in Table 4 (p. 7) and the
   supplement (3133.08 K), with smaller Ea and n differences for S13, S37/S38, R13 and others (listed under
   "Inputs"). The file is used as the nominal model.
4. Table 4 prints R1's Ea as −7692.2 K (p. 7); the supplement's table and the mechanism file have +7692.21 K. The
   positive value (the file's) is used; the minus sign in Table 4 appears to be a misprint.
5. The supplement's row for S36 (N2H2 + NH2 <=> NH3 + NNH) has its n and Ea columns shifted (n = 5.433, Ea =
   2.226); the file's n = 2.226, Ea = −518.32 K are used.
6. For 41 of the 52 entries the reduced bounds equal ± 0.5σ of the unfiltered prior to within 3 %. Were these
   entries effectively unconstrained by the experimental targets?

## License and credit

Code (`*.py`, the notebook): MIT. Data (the CSVs): CC BY 4.0; they are model outputs computed here with the
authors' mechanism and uncertainty table. Model and study design: G. Huang, H. Wang, L. Tian, O. Haidn, and
N. Slavinskaya; cite their paper (above). The uncertainty factors and A0 values in the driver's `TABLE` are
transcribed from the supplement's `mmc1.xlsx` (credit: the authors; the article is CC BY 4.0, and the supplement
states no licence). **The mechanism file `NH3_H2.yaml` is not distributed here**: its header says it "is intended
for use by not-for-profit researchers only. Any copying or distribution of this mechanism, or use (in part or
whole) in any commercial software is strictly forbidden without the written consent of the authors." It is linked, not
copied: get it from the supplementary data at
https://doi.org/10.1016/j.fuel.2025.135810 (Appendix A, `mmc3.zip`) as described under "Regenerating"; the driver
checks its sha256. The article is cited and linked, not redistributed. Cantera: D. G. Goodwin, H. K. Moffat, I. Schoegl, R. L. Speth, B. W. Weber,
https://www.cantera.org (BSD-3-Clause).
