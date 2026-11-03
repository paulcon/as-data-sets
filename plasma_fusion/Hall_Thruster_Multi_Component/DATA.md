# Hall_Thruster_Multi_Component: data

Kind: **stored data**. The discharge outputs come from an outside Julia code, HallThruster.jl v0.18.7
(MIT), run for this repository on 2026-11-01 (an external run; see "Regenerating") in the coupled set-up of Marks et al. (2025); the analytic cathode and
plume components are numpy code in this directory. The paper's own runs are not public (its Data
Availability section: "available from the corresponding author on reasonable request"), so these are
new runs of the published model at the published input ranges. No value in the CSV has been changed
after it was written; failed runs are kept with `NaN` outputs and dropped in the notebook by a
stated rule. Runs whose discharge oscillates are kept too and flagged in the last column,
`oscillating` (rule below); the notebook's main analysis leaves them out and uses them in a
sensitivity check.

## Files

| file | contents | role |
|---|---|---|
| `Hall_Thruster_Multi_Component.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 662; outputs committed |
| `hall_thruster_spt100.csv` | 440 × 35: the 18 inputs (physical units; `c4`, `c5` as values), run status, then `Vcc`, `ID`, `IB`, `T`, `Tc`, `u_exit`, `j_peak`, `phi_d`, the averaging diagnostics `ID_std`, `T_std`, `ID_a`, `ID_b`, `T_a`, `T_b`, `wall_s`, and the flag `oscillating` (columns and units in the second header line) | stored data |
| `generate_data.py` | the Latin hypercube design (seed 662), the cathode model, the call to HallThruster.jl, the plume model, and the assembly of the CSV; `check` (no Julia) recomputes the design and the numpy columns | regenerates the file where Julia and HallThruster.jl are installed (see below) |
| `run_hallthruster.jl` | Julia driver: builds the SPT-100 configuration and runs HallThruster.jl once per row, resumable | called by `generate_data.py` and `check_accuracy.py` |
| `spt100_components.py` | `cathode_coupling_voltage` (Eq. (2)) and `plume` (Eqs. (5)–(7)), numpy only | model components |
| `check_accuracy.py` | recomputes the numbers of the Accuracy section and of "How these files relate to the paper" that neither the notebook nor `generate_data.py check` prints; checks 5–8 rerun HallThruster.jl where it is installed | check |

The CSV is plain comma-separated numbers after two `#` lines (a description and the column names), so
`np.loadtxt(path, delimiter=',')` reads it; failed runs have `nan` outputs.

**Oscillation flag.** `oscillating` = 1 if the standard deviation of the discharge current I_D(t)
over the 500 µs averaging window exceeds 1 % of its mean (`ID_std / ID > 0.01`), else 0 (NaN for a
failed run). It is computed from the stored columns by `generate_data.py` (`OSC_THRESHOLD`), and
`generate_data.py check` recomputes it. 20 of the 440 runs are flagged (29 at a 0.5 % threshold, 18 at
2 %); every run whose two half-window averages of I_D or T differ by more than 1 % (16 runs) is among
them (the notebook prints these counts).

Not shipped: the SPT-100 magnetic-field profile `bfield_spt100.csv`, which belongs to the authors'
HallThrusterPEM repository (GPL-3.0). `generate_data.py` reads it from a path in the `BFIELD`
environment variable; get it from
`https://raw.githubusercontent.com/JANUS-Institute/HallThrusterPEM/125cd2118dae8bfeb77d3cc90b40cdbebdbc5777/src/hallmd/devices/SPT-100/bfield_spt100.csv`
(58 rows of z [m], B [T]; SHA-256 `06e33361321f0dce873cca6ebf68dbb680d30feed7dc6c68016956eb0cbc7085`; unchanged in that
repository from 2025-01-17 until it was removed on 2026-03-12, so it is the file the paper's runs used).

## Model

Citation: T. A. Marks, J. D. Eckels, G. E. Mora and A. A. Gorodetsky, "Uncertainty quantification of a
multi-component Hall thruster model at varying facility pressures", *J. Appl. Phys.* 138(15):153305,
2025, doi:10.1063/5.0283796 (CC BY-NC 4.0; checked against Crossref 2026-11-01); erratum: *J. Appl. Phys.*
139(3):039901, 2026, doi:10.1063/5.0314947, which corrects Eq. (6) only (see Plume). Read from the
authors' manuscript dated 21 October 2025 (open-access preprint version); page numbers below are of
that manuscript.

- **Cathode** (Sec. II A 1, Eq. (2), p. 2; Jorns and Byrne 2021):
  V_cc = V_vac + T_ec ln(1 + P_B/P_T) − T_ec P_B/(P_T + P*), clipped to [0, V_d] as in the authors' code
  (no sample needed clipping).
- **Thruster** (Sec. II A 2, Eqs. (3)–(4), pp. 2–4; grid and wall settings p. 4): HallThruster.jl v0.18.7 (commit 22684d2,
  2025-06-25; the version the authors' HallThrusterPEM sets as default), 1-D quasineutral fluid model,
  singly-charged xenon only, 100 cells on an even grid over 3 channel lengths (0.075 m, as the paper's
  text says, p. 4; the authors' `pem_v1_SPT-100.yml` has `domain: [0, 0.08]`, 3.2 channel lengths), 1 ms
  simulated with adaptive time steps, outputs averaged over the last 500 µs. Settings of the runs, following the
  paper text and the authors' `pem_v1_SPT-100.yml` (HallThrusterPEM commit f4f8460, 2025-04-25):
  - anomalous transport `SimpleLogisticShift(ScaledGaussianBohm(anom_scale = α_anom, barrier_scale =
    β_anom, center = z_anom, width = L_anom), shift_length = Δz_anom, midpoint_pressure = 25 µTorr,
    slope = 2)`, which is Eqs. (3)–(4) except that the code's Gaussian is exp(−½((ẑ − z)/L)²) where
    Eq. (3) prints exp(−((ẑ − z)/L)²); `L_anom` is passed to the code unchanged (see unknowns);
  - `WallSheath(BNSiO2, loss_scale = c_w)`, `ion_wall_losses = true` (the paper: for unshielded
    thrusters "ion wall losses are accounted for", p. 4; the April 2025 yml sets `false` explicitly, see
    unknowns; the effect of the choice is measured under Accuracy),
    `neutral_ingestion_multiplier = f_n`, `neutral_velocity = u_n`, `cathode_Tev = T_ec`,
    `cathode_coupling_voltage = V_cc`, `solve_plume = true`, `apply_thrust_divergence_correction = false`
    (the plume model corrects the thrust instead), background temperature 100 K (code default);
  - SPT-100 geometry from HallThrusterPEM's `devices/SPT-100/device.yml`: channel length 0.025 m, inner
    radius 0.035 m, outer radius 0.05 m, unshielded, with its magnetic-field file (above);
  - time step: adaptive, CFL 0.799 (code default), min 1e-10 s, max 1e-7 s, initial step 0.2 Δz/u with
    u = (2e(V_d − V_cc)/m_Xe)^½ (the authors' rule).
- **Plume** (Sec. II A 3, Eqs. (5)–(7), p. 5; the model of the paper's Refs. 10 and 13): written here from
  those equations. Main beam and scattered ions with Gaussian angular profiles of divergence
  α₁ = min(c₂P_B + c₃, π/2) and α₂ = α₁/c₁, current fractions 1 − c₀ and c₀, each normalized over the
  hemisphere; charge-exchange ions uniform over the hemisphere with fraction 1 − exp(−r n σ_cex), with
  n = c₄P_B + c₅ (P_B in Pa), σ_cex = 55 × 10⁻²⁰ m² (not in Table I; the authors' configuration
  gives it as `Uniform(51e-20, 58e-20)` with nominal 55e-20, and only the nominal value is used here; σ_cex
  enters only the charge-exchange fraction, so it changes j_peak but not φ_d, T_c or I_D), r = 1 m.
  The divergence angle uses the beam and scattered ions only:
  cos φ_d = ∫ j cos φ sin φ dφ / ∫ j sin φ dφ over φ ∈ [0, π/2] (400-point Gauss–Legendre), and
  T_c = T cos φ_d (Eq. (7)). Eq. (6) as printed in the article has cos φ in place of sin φ in the
  denominator and j_ion (all ions) in place of the beam ions. The erratum (J. Appl. Phys. 139(3):039901,
  2026, doi:10.1063/5.0314947, read 2026-11-01) corrects exactly this: cos φ_d = I_B,z / I_B with both
  integrals over φ ∈ [0, π/2] of j_ion − j_cex, weighted by cos φ sin φ and sin φ; the authors state that
  their code and results already used the corrected form. It is the form used here. The erratum changes
  nothing else (no table, figure or prior range).
- **Operating condition, held fixed** (the paper calibrates "assuming the operating conditions take on
  their mean values", p. 7): 300 V / 12.3 µTorr-Xe is Diamant et al. (2014) L3 facility, condition 4,
  one of the paper's SPT-100 *training* conditions (Table III; row 4 of the authors'
  `diamant2014/data_L3.csv`): V_d = 300 V, P_B = 12.3 µTorr-Xe, total flow 5.665 mg/s with anode-to-cathode
  flow ratio 12, so ṁ_a = 5.665 × 12/13 = 5.2292 mg/s. (Values from the authors' tabulation of Diamant et
  al. in HallThrusterPEM, `data/spt100/diamant2014/data_L3.csv`; measured there: I_D = 4.5 A,
  T = 81.44 mN, V_cc = 32.84 V.) The paper's N(·, 2 %), N(·, 5 %) and N(·, 2 %) operating-condition
  distributions are not sampled.
- **Inputs** (m = 18), Table I (p. 4), independent:

| name | symbol | units | distribution | component |
|---|---|---|---|---|
| `Tec` | T_ec | eV | U(1, 6) | cathode, thruster |
| `Vvac` | V_vac | V | U(0, 60) | cathode |
| `PT` | P_T | µTorr | U(20, 200) | cathode |
| `Pstar` | P* | µTorr | U(1, 100) | cathode |
| `alpha_anom` | α_anom | – | U(0, 1) | thruster |
| `beta_anom` | β_anom | – | U(0, 1) | thruster |
| `z_anom` | z_anom | channel lengths | U(0.75, 1.5) | thruster |
| `L_anom` | L_anom | channel lengths | U(0, 0.5) | thruster |
| `dz_anom` | Δz_anom | channel lengths | U(0, 0.5) | thruster |
| `u_n` | u_n | m/s | U(100, 500) | thruster |
| `c_w` | c_w | – | U(0.5, 1.5) | thruster |
| `f_n` | f_n | – | U(1, 10) | thruster |
| `c0` | c₀ | – | U(0, 1) | plume |
| `c1` | c₁ | – | U(0.1, 0.9) | plume |
| `c2` | c₂ | rad/Pa | U(−15, 15) | plume |
| `c3` | c₃ | rad | U(0.2, π/2) | plume |
| `c4` | c₄ | m⁻³/Pa | 10^x, x ~ U(18, 22) | plume |
| `c5` | c₅ | m⁻³ | 10^x, x ~ U(14, 18) | plume |

  The paper gives no nominal values (priors only).

  **P_T and P*: the paper's own SPT-100 runs did not use Table I's ranges.** Table I (p. 4) gives
  P_T ~ U(20, 200) and P* ~ U(1, 100) µTorr. Table IV (p. 9), the SPT-100 posterior, lists the prior as
  P_T ~ U(10, 100) and P* ~ U(10, 200) µTorr, and its posterior samples reach P_T = 10 (minimum) and
  P* = 197.95 (maximum), values impossible under Table I. The authors' April 2025 configuration
  (`pem_v1_SPT-100.yml`) has Table IV's ranges. So Table I's prior for these two inputs is not what the
  SPT-100 runs used, and the paper's Fig. 18a indices were most likely computed with Table IV's ranges.
  (Table V, the H9, lists yet another prior, P_T ~ U(1, 100).) Table I's ranges are kept here: they are
  the paper's stated prior, and the choice matters little. at 12.3 µTorr the activity shares of P_T and P* are below 0.001 for I_D, T_c and
  V_cc on the steady runs (at most 1.1e-3, P* for T_c, with all runs; printed in the notebook's
  comparison cell), so neither subspace depends on the choice. See the open questions.

  **c₀.** Table I describes c₀ as the "ratio of main to scattered currents", but the plume model (Eq. (5)
  and the authors' code) uses c₀ as the scattered fraction of the beam current: the scattered population
  carries c₀ and the main beam 1 − c₀. That is the meaning here.
- **Quantities of interest**: the notebook analyses the corrected thrust T_c and the discharge current
  I_D (Table II, p. 4); the file also has V_cc, I_B, T, the singly-charged ion velocity at the channel
  exit plane z = L_ch (linear interpolation between cell centres), the plume current density on the
  centreline at 1 m, and φ_d.
- **Gradients**: none. Finite differences of 500 µs averages of an oscillating discharge are unreliable
  (see Accuracy), so the notebook uses the global quadratic (QPHD) estimator on values.
- **Sampling design**: one Latin hypercube of 440 points in the 18 inputs (`scipy.stats.qmc.LatinHypercube`,
  seed 662), on the uniform coordinates (the exponents of c₄ and c₅), mapped to physical values. The
  design was drawn with scipy 1.18.1 and reproduces with the scipy 1.17.1 of the repo's `.venv` (see
  "Regenerating"). 440 runs are 2.3 per coefficient of the 190-term quadratic that the notebook fits.
- **Published results**: the paper reports no active subspace. Its Sobol' total-effect indices over the
  prior (Sec. III E, p. 19, and Fig. 18, p. 21; 5000 prior samples, bootstrap error bars) are shown only as bar charts.
  Read off Fig. 18a (SPT-100) at 300 dpi, about ±0.01: V_cc: V_vac 1.01; I_D: α_anom 0.76, β_anom 0.21,
  L_anom 0.06; T_c: c₃ 0.68, c₀ 0.11, α_anom 0.11, c₁ 0.10, u_n 0.03; exit ion velocity: z_anom 0.93,
  β_anom 0.52, L_anom 0.09, α_anom 0.07; peak current density: c₃ 0.87, c₄ 0.30, c₀ 0.14, c₁ 0.04,
  c₂ 0.03. The notebook compares these with activity shares.

## Regenerating

The discharge outputs are an external run of HallThruster.jl that cannot be repeated exactly from the
repository: the Julia environment of the run (a project and manifest pinning HallThruster and 28 other
registered packages) is not shipped. What is known of it:
- HallThruster.jl v0.18.7, git commit 22684d2851e60b264620cff597ecb37a69e19cd8 (tree
  00553b6107b6ea079a90c98b3fc717d0f0843324, which is also the General registry's tree hash for v0.18.7),
  run with Julia 1.13.3 (conda-forge build `julia-1.13.3-h192eb0c_0`, linux-64);
- written on 2026-11-01 in a Linux container (2 shared CPUs, load about 6 from other jobs), with
  `run_hallthruster.jl` as shipped and the B-field file above.

To rerun it, install Julia (≥ 1.10) and HallThruster.jl v0.18.7 in a Julia environment, for example
`julia --project=ENV -e 'using Pkg; Pkg.add(name="HallThruster", version="0.18.7")'`. The other packages are
then resolved afresh, not pinned to the versions of the shipped runs. Then

```bash
export JULIA=/path/to/julia JULIA_PROJECT=ENV BFIELD=/path/to/bfield_spt100.csv
python generate_data.py design --out DIR     # LHS design and DIR/julia_inputs.csv (seconds)
python generate_data.py run --out DIR        # HallThruster.jl, one process; resumable (rerun to continue)
python generate_data.py assemble --out DIR   # cathode + plume components, writes DIR/hall_thruster_spt100.csv
```

`DIR` must be a directory other than this one: the shipped CSV is the record of the runs, and a new
file is compared with it, not copied over it. Without Julia, HallThruster.jl or the B-field file, the
`run` stage exits with status 1 and says what to install. Per the repository's rule for outside codes,
the runs are repeated only if a problem is found in the shipped data.

**Run time of the shipped runs** (external run): the 440 HallThruster.jl runs took 5.0–17.6 s each (median
8.7 s; 3806 s, 63 min, in all; `check_accuracy.py`), done in four resumable chunks of at most 19 min with
`nice -n 10`; the first load of HallThruster.jl precompiled for about 15 min. Every run returned
status 0: no run failed. The simulation is deterministic, so a rerun with the same environment reproduces
the discharge outputs: rows 0, 150 and 300, rerun in that environment, agreed with the file to all 11
digits written (`check_accuracy.py` check 5 repeats this where Julia is installed). The driver's optional
trailing columns (cell count, duration, averaging start, ion wall losses) were not used for the 440 runs,
so all of them used the settings above; `check_accuracy.py` uses them for its reruns.

**Without Julia.** `python generate_data.py check` redraws the design and recomputes the cathode and
plume columns and the oscillation flag from the stored discharge outputs; it only reads the shipped file.
In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-02) it runs in about 3 s
and prints: inputs max rel diff 4.0e-09 (the 11 significant digits written); Vcc 4.5e-11; Tc 1.8e-11;
j_peak 4.4e-11; phi_d 4.7e-11; oscillating flag reproduced: True (20 runs). These are rounding-level
differences (the file holds 11 significant digits), so the design and the numpy components reproduce.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name. Its checks 1–4 use the shipped file and
run in `.venv` in about 3 s (`--skip-julia`). Its checks 5–8 rerun HallThruster.jl on rows 0, 150 and 300
and were not run in `.venv` (no Julia); the numbers they produce are quoted from the original run in the
Linux container that wrote the data (2026-11-01, the environment of "Regenerating"; those reruns were not
shipped), and the script recomputes them where Julia and HallThruster.jl v0.18.7 are installed. There are
no gradients to check. The outputs carry two numerical errors:

- **Time averaging.** Most runs are steady: the averages over the first and second halves of the
  500 µs window differ by a median relative 1.6e-11 (I_D). 20 of 440 runs oscillate (std/mean of
  I_D above 1 %), and for 13 (I_D) and 15 (T) runs the half-window averages differ by more than 1 %,
  by up to 63 % and 48 % (columns `ID_a`, `ID_b`, `T_a`, `T_b`; printed in the notebook). Rerunning
  row 0, an oscillating run, for 2 ms and averaging over the last 1 ms changes I_D from 5.093 to
  5.180 A (1.7 %) and T from 79.60 to 88.43 mN (11 %); rows 150 and 300, steady, are unchanged to
  10 digits (`check_accuracy.py` check 6; original run). So T and T_c of the oscillating runs are
  uncertain by about 10 %; the notebook marks them.
- **Oscillating runs.** The 20 flagged runs (column `oscillating`) carry the averaging error above;
  they are all low-transport runs (α_anom at most 0.261; 17 of the 26 runs with α_anom < 0.06, the
  SPT-100 posterior median of Table IV, oscillate; printed in the notebook), so the data are least
  reliable near the calibrated thruster. Two of them, rows 132 and 149, have a time-averaged ion current
  above the discharge current (I_B/I_D = 1.064 and 1.201; `check_accuracy.py`), which a converged discharge
  cannot have: a further sign that their averages are not converged. (The largest I_B/I_D of the steady
  runs is 0.757.) The notebook's main results use the 420 steady runs and repeat everything on all 440.
- **Ion wall losses.** The three accuracy rows were rerun with `ion_wall_losses = false` (the authors'
  April 2025 setting; `check_accuracy.py` check 7; original run). I_D changes by −3.0 %, −3.4 % and −8.8 %
  and T (so also T_c, since φ_d does not depend on T) by −3.0 %, +1.4 % and +0.3 % at rows 0, 150 and 300
  (row 0: I_D 5.093 → 4.941 A, T 79.60 → 77.24 mN, and still oscillating; row 300: I_D 40.52 → 36.95 A);
  the exit ion velocity rises by 7.4 %, 6.4 % and 23 %. So the setting shifts I_D by several per cent and
  u_exit by more, but T_c by a few per cent at most; the data set keeps `true`, as the paper's text says.
- **Grid.** Doubling the cells to 200 changes I_D by −3.6 %, +1.6 % and +0.9 % and T by −8.2 %,
  +1.6 % and +0.7 % at rows 0, 150 and 300 (row 0: I_D 5.093 → 4.907 A, T 79.60 → 73.08 mN;
  `check_accuracy.py` check 8; original run). The paper reports changes below 0.1 A and 5 mN for doubling
  the cells or the domain at its SPT-100 test condition (p. 4); row 0's 6.5 mN is larger, at an
  oscillating run far from that condition. 100 cells is the paper's resolution and is kept.
- **Plume integrals.** The hemisphere normalisations use 400-point Gauss–Legendre quadrature; they agree
  with the closed form (complex error function) to a relative error of order 1e-14 at α = 0.2, 0.5, 1,
  1.5 and 3 rad (`check_accuracy.py`).
- **Samples outside physical meaning** (`check_accuracy.py`). All V_cc lie in [0, V_d] before clipping
  (0.20 to 60.01 V), all averaged currents and thrusts are positive, and no main-beam divergence angle is
  non-positive. The only physically impossible values are the I_B > I_D of the two oscillating runs above.

## How these files relate to the paper

- **Not the paper's runs.** The paper's computations (MCMC calibration, Monte Carlo, Sobol' indices)
  vary the parameters over the prior or posterior at several operating conditions of two thrusters
  (SPT-100 on xenon and H9 on krypton) with the authors' HallThrusterPEM/amisc workflow. This data set
  is a new design over the same 18-parameter prior for the SPT-100 only, at one of its training
  operating conditions, with the authors' discharge code (HallThruster.jl) and the paper's equations
  for the other two components. The HallThrusterPEM code (GPL-3.0) was read for settings but not
  copied or run.
- **Prior, not posterior.** Over the prior most runs are far from the measured thruster:
  at this condition Diamant et al. measured
  I_D = 4.5 A, T = 81.44 mN and V_cc = 32.84 V. Of the 440 runs, 7 have I_D within 20 % of 4.5 A (25 below
  10 A; median 40.0 A), 67 have T_c within 10 % of 81.44 mN, none has both, and 44 have V_cc within 3 V
  of 32.84 V (`check_accuracy.py`). The wide prior on α_anom (up to 16 times Bohm transport) drives this, as in the paper's
  prior Sobol' analysis.
- **Comparison with the paper.** The paper reports no active subspace. Its Sobol' total-effect indices
  over the prior (Fig. 18a, values read off above) are compared in the notebook with activity shares
  of two-dimensional QPHD subspaces, on the 420 steady runs and on all 440. On the steady runs the
  rankings agree: V_cc: V_vac (1.000); I_D: α_anom 0.712 > β_anom 0.254 > L_anom 0.024 (paper 0.76 >
  0.21 > 0.06); T_c: c₃ 0.730 > c₀ 0.085 > α_anom 0.069 ≈ c₁ 0.068 > u_n 0.021 (paper c₃ 0.68 > c₀ 0.11
  ≈ α_anom 0.11 > c₁ 0.10 > u_n 0.03); peak current density (log): c₃ > c₄ > c₀ > c₁; exit ion velocity:
  z_anom 0.630 > β_anom 0.302 (paper 0.93 > 0.52), while the paper's next two, L_anom 0.09 and α_anom 0.07,
  are here at the noise level of the fit (≤ 0.013). With the 20 oscillating runs included, α_anom's
  share for T_c falls to 0.017 (5th), and c₁'s rises to 0.105: the oscillating runs, not the single
  operating condition or the difference between activity scores and Sobol' indices, move α_anom down. The paper's text names α_anom among the dominant inputs for the
  exit ion velocity, but its own bar (0.07) is smaller than L_anom's (0.09).
- **Active subspaces (main results, steady runs).** T_c: n = 1, λ₁/λ₂ ≈ 18 (17.7; bootstrap 95 % 8.7 to
  18.8), all runs 10.7 (bootstrap 2.9 to 11.8). I_D: n = 2, λ₂/λ₃ ≈ 10 (10.2; bootstrap 3.5 to 9.9), all
  runs 12.8 (bootstrap 4.5 to 11.4). The ratios still rise with the number of runs (subsample medians
  for T_c, steady: 12.1, 14.7, 16.8, 17.5 at 250, 300, 350, 400 runs; for I_D, steady: 5.3, 8.1, 9.1,
  9.8), so they are approximate lower estimates; the dimensions and directions are stable (T_c:
  |w₁ᵀw₁| = 0.992 between the steady and all-runs analyses; I_D: subspace distance 0.047).

## Open questions (choices made here)

- Eq. (3) prints the barrier as exp(−((ẑ − z_anom)/L_anom)²); HallThruster.jl's `ScaledGaussianBohm`,
  which the authors' configuration uses, computes exp(−½((ẑ − z_anom)/L_anom)²). L_anom was passed to the
  code unchanged, so these runs follow the code. Which did the paper's runs use?
- `ion_wall_losses`: the paper says ion wall losses are accounted for in unshielded thrusters (p. 4),
  but the authors' April 2025 SPT-100 configuration sets `false`. `true` was used here (effect measured
  under Accuracy: I_D −3 % to −9 %, T within ±3 %). Which did the paper's SPT-100 runs use?
- P_T and P*: Table I gives U(20, 200) and U(1, 100) µTorr, but Table IV's SPT-100 prior is U(10, 100)
  and U(10, 200) and its posterior lies outside Table I's box. Which prior did the SPT-100 runs and the
  Sobol' indices of Fig. 18a use? (Table I is used here.)
- At which operating condition, or over which distribution of operating conditions, were the prior
  Sobol' indices of Fig. 18 computed?
- How were runs with a non-converged 500 µs time average (oscillating discharges, here 20 of 440, all at
  low α_anom) treated in the paper's Monte Carlo and Sobol' estimates?
- Domain length: the paper's text says three channel lengths (0.075 m, used here); the authors'
  configuration has 0.08 m.
- The HallThruster.jl version of the paper's runs is not stated; v0.18.7 is the default of the authors'
  `hallmd` package (their April 2025 configuration names 0.18.5).
- σ_cex is not in Table I; the authors' configuration makes it Uniform(51e-20, 58e-20) m²; the nominal
  55 × 10⁻²⁰ m² was held fixed here (it affects only j_peak).
- Resolved: Eq. (6). The erratum (doi:10.1063/5.0314947) corrects Eq. (6) to the form used here and
  changes nothing else.

## License and credit

Code (`generate_data.py`, `run_hallthruster.jl`, `spt100_components.py`, `check_accuracy.py`) and the
notebook: MIT (`LICENSE`). Data (the
CSV): CC BY 4.0 (`LICENSE-DATA`). Model: T. A. Marks, J. D. Eckels, G. E. Mora and A. A. Gorodetsky
(University of Michigan); discharge code HallThruster.jl by T. Marks, P. Schedler, B. Jorns and
contributors (MIT, https://github.com/UM-PEPL/HallThruster.jl). The SPT-100 magnetic field and
geometry are from HallThrusterPEM (GPL-3.0, https://github.com/JANUS-Institute/HallThrusterPEM),
used as an input and not redistributed. Cite Marks et al., J. Appl. Phys. 138:153305, 2025,
doi:10.1063/5.0283796, and HallThruster.jl, J. Open Source Softw. 8(86):4672, 2023,
doi:10.21105/joss.04672.
