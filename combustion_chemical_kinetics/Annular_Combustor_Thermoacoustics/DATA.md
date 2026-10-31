# Annular_Combustor_Thermoacoustics: data

Kind: **Python model**. The model is `annular_combustor_model.py` (numpy only), a new implementation of the ATACAMAC dispersion relation for an annular combustor with 19 burners. `generate_data.py` writes every data file from it with saved seeds. This data set replaces the 2016 `Atacamac/` files, which are stored data from the authors' code with an undocumented input distribution (see "How these files relate to the paper and to the 2016 files"); those are not shipped.

## Files

| file | contents | role |
|---|---|---|
| `Annular_Combustor_Thermoacoustics.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2016`; outputs committed |
| `annular_combustor_model.py` | `growth_rate(X, case, grad=)`, `split_roots`, `root_gradient`, `symmetric_root`, `dispersion`, `bounds`; constants `CASES`, `NAMES`, `OMEGA0` | the model |
| `generate_data.py` | writes the eight CSVs; `mc_inputs(case)` redraws the Monte Carlo inputs | regenerates the data |
| `inputs_weak.csv`, `inputs_strong.csv` | 1000 × 38 physical inputs: n01..n19 (dimensionless), tau01..tau19 (s) | model inputs (seeds 20161, 20162) |
| `outputs_weak.csv`, `outputs_strong.csv` | 1000 × 4: `f_Im` [1/s] (the QoI), `f_Im_other` [1/s], `freq_max` [Hz], `freq_other` [Hz] | model output |
| `gradients_weak.csv`, `gradients_strong.csv` | 1000 × 38: ∂f_Im/∂x with respect to the **physical** inputs (1/s per unit n; 1/s² for τ) | model output |
| `montecarlo_weak.csv`, `montecarlo_strong.csv` | 10 000 × 4, the same columns as `outputs_*.csv`, at 10 000 further samples (seeds 20171, 20172; inputs not stored, `generate_data.mc_inputs(case)` redraws them) | Monte Carlo reference for the risk factor |

The CSVs are numbers after two `#` lines (a description with the distribution, seed and provenance; the column names with units), so `np.loadtxt(path, delimiter=",")` reads them. `freq_max` is the frequency of the component whose growth rate is `f_Im`.

## Model

### Paper card

- **Paper.** M. Bauerheim, A. Ndiaye, P. Constantine, S. Moreau, F. Nicoud, "Symmetry breaking of azimuthal thermoacoustic modes: the UQ perspective", *J. Fluid Mech.* 789:534–566 (2016), doi:[10.1017/jfm.2015.730](https://doi.org/10.1017/jfm.2015.730). Read from the HAL open copy (hal-01234985v2, author manuscript; page numbers below are its printed manuscript pages, not the journal's). Called JFM below.
- **Companion.** L. Magri, M. Bauerheim, M. P. Juniper, "Stability analysis of thermo-acoustic nonlinear eigenproblems in annular combustors. Part I. Sensitivity", *J. Comput. Phys.* (2016), doi:[10.1016/j.jcp.2016.07.032](https://doi.org/10.1016/j.jcp.2016.07.032), read from the preprint arXiv:1602.08438v2 (which prints that DOI). Called Magri below. It writes the same network as a nonlinear eigenproblem (eq. 30) and gives the constants missing from JFM (Table 1).
- **Earlier study, same UQ setup.** M. Bauerheim, A. Ndiaye, P. Constantine, G. Iaccarino, S. Moreau, F. Nicoud, "Uncertainty quantification of thermo-acoustic instabilities in annular combustors", *Proc. Summer Program 2014*, Center for Turbulence Research, Stanford, pp. 209–218 (cited by JFM as Bauerheim et al. 2014b; not read for this build).
- **Equations.** JFM eqs. (2.1)–(2.11): rotation matrix R over one sector (eq. 2.2), interaction matrix T_i with coupling factors Γ_i,1..4 (eqs. 2.3–2.7), flame parameter 𝔽_i = (ρ⁰c⁰/ρ_u c_u)(1 + n_i e^{jωτ_i}) (eq. 2.8), dispersion relation det(∏ R T_i − I) = 0 (eqs. 2.10–2.11). Time dependence e^{−jωt}: **Im ω > 0 is unstable** (JFM p. 5).
- **QoI.** f_Im = max(Im ω₁, Im ω₂)/(2π) [s⁻¹], the larger growth rate of the two components into which the first azimuthal mode splits (JFM §4.2, p. 20, and App. A.1, p. 27). Risk factor RF = P(f_Im > 0) (JFM eq. 3.2).
- **Constants.**

  | constant | value | source |
  |---|---|---|
  | N | 19 burners | JFM p. 7 |
  | L_c, L_p (half-perimeters) | 0.54 m, 0.54 m | JFM Table 1, p. 11 |
  | S_c, S_p | 0.00785 m², 0.00785 m² | JFM Table 1 |
  | S_i | 0.00028 m² | JFM Table 1 |
  | L_i | **0.0551 m** | Magri Table 1, p. 13 (JFM Table 1 says L_i⁰ = 0.04 m; see below) |
  | α (flame at z = αL_i) | **0.862** | Magri Table 1 (not given in JFM) |
  | ρ_u, c_u (fresh gas, plenum) | 10.93 kg/m³, 519 m/s | JFM Table 1 |
  | ρ⁰, c⁰ (burnt gas, chamber) | 4.70 kg/m³, 832 m/s | JFM Table 1 |
  | ω⁰ = πc⁰/L_c | 4840.38 rad/s | JFM p. 19 |

- **Inputs** (m = 38), independent uniform, standard deviations 10 % (n) and 5 % (τ) of the mean (JFM p. 13: "uniform PDFs have been used"; Table 2, p. 14), so half-widths √3·σ:

  | case | input | mean | σ | range | source |
  |---|---|---|---|---|---|
  | weak (Wcase) | n_i, i = 1..19 | 0.5 | 0.05 | [0.413397, 0.586603] | JFM Table 2, p. 14 |
  | weak | τ_i | 0.635 ms | 31.75 µs | [0.580007, 0.689993] ms | JFM Table 2 (σ printed as 31.7 µs) |
  | strong (Scase) | n_i | 1.75 | 0.175 | [1.44689, 2.05311] | JFM Table 2 |
  | strong | τ_i | 0.735 ms | 36.75 µs | [0.671347, 0.798653] ms | JFM Table 2 (σ printed as 36.7 µs) |

- **Gradients.** Exact, by the implicit function theorem on the dispersion relation (below).
- **Sampling design.** Per case, M = 1000 samples with gradients (seed 20161 weak, 20162 strong) and 10 000 Monte Carlo samples without (20171, 20172), `numpy.random.default_rng(seed).random((M, 38))` mapped to the ranges. The paper used M = 50, 500, 1000 gradient samples (Figs. 10–14) and 10 000 Monte Carlo runs (App. A).
- **Published results** (JFM): five eigenvalues above 1 % of λ₁ in both cases (pp. 16–17, Fig. 10); λ₁ about two orders of magnitude larger in the strong case (p. 16); w₁ equal weights, gains nearly absent in the weak case (Figs. 11–12); w₂, w₃ with period N/2, tied to F̂(±2) after the change of variables {n_i, τ_i} → {Re F̂, Im F̂} (eq. 4.2), which reduces the dimension "from 5 to only 3" (p. 19, Figs. 13–14); surrogate accuracy R (eq. 4.5) for the weak case at M = 1000: FL 0.80, RL 0.80, RQ 0.96, RC 0.96 (Table 4); strong case FQ1000 0.63 and RQ30 0.75 (p. 24); risk factors, Monte Carlo with 10 000 ATACAMAC runs, 14.77 % (weak) and 39.55 % (strong), and from surrogates FL1000 11.90 / 32.03, RL30 11.54 / 30.83, RL1000 11.94 / 33.17, FQ1000 15.65 / 39.26, RQ30 15.25 / 41.2, RQ1000 14.94 / 39.03 % (Table 5, p. 25). Magri Table 2: symmetric roots 5.059×10³ − 1.392j rad/s (case A: n = 0.5, τ = 0.6223 ms) and 4.272×10³ + 3.600×10⁻³j (case B: n = 1.75, τ = 0.735 ms).

### Assumptions where the papers are ambiguous

1. **L_i and α.** JFM Table 1 gives L_i⁰ = 0.04 m and no α. Magri Table 1 gives L_i = 0.0551 m and α = 0.862 and says "same parameters as" JFM. We use Magri's values. With them the model reproduces Magri Table 2 case A (5059.19 − 1.397j against 5.059×10³ − 1.392j) and the 2016 weak-case data to 1.4×10⁻³ s⁻¹ at all 492 samples; with L_i = 0.04 m and α = 0.5, 0.862 or 1 the case-A root is 5136–5379 rad/s with growth rates −15 to −74 rad/s (scratch check), far from both.
2. **The state vector and product order.** JFM writes X = [p̂_p, û_p, p̂_c, û_c] but its coupling example uses −jρ⁰c⁰û; we use Magri's X = [p̂_p, −jρ_u c_u û_p, p̂_c, −jρ⁰c⁰û_c] (eq. 35), for which the propagation matrix of eq. (2.2) is exact, and the product R T_N ⋯ R T_1 (JFM eq. 2.10). The determinant does not depend on where the cycle starts or on the burners' order.
3. **Which root, and the maximum.** "The first azimuthal mode" is taken as the pair of roots that the double root of the symmetric configuration splits into. The search circle (300 rad/s weak, 450 rad/s strong, around the symmetric root) must contain exactly these two roots; the nearest other roots of the symmetric configuration are the plenum mode (3416 − 53j weak, 3493 − 350j strong) and the second azimuthal mode (6266 + 14j, 6250 + 59j rad/s), 850–2000 rad/s away. In all 2 × 11 000 samples the root count is 2 to 10⁻⁹, and in the gradient sets the roots stay within 37 rad/s (weak) and 189 rad/s (strong) of the symmetric root.
4. **Units.** Im(f) = Im(ω)/(2π) in s⁻¹, as on JFM's axes ("Growth rate (s⁻¹)", Fig. 20).
5. **σ of τ.** Exactly 5 % of the mean (31.75 and 36.75 µs), which JFM Table 2 rounds to 31.7 and 36.7 µs.

### Solver

- The symmetric root solves the Bloch condition det(R T − e^{2πj/N} I) = 0 by the secant method from ω⁰ (weak: 5058.035 − 14.690j rad/s, 805.011 Hz, Im f = −2.338 s⁻¹; strong: 4272.262 + 3.709j rad/s, 679.952 Hz, Im f = +0.590 s⁻¹).
- For each sample, the contour integrals (1/2πj)∮ zᵏ D′/D dω, k = 0, 1, 2 (trapezoid rule, 64 points), give the root count and the two roots (Newton's identities); D′/D = tr(A⁻¹ ∂_ω A) with A = M − I and ∂_ω A from the analytic derivatives of R and Γ. Six Newton steps on D/(ω − ω_other) polish each root (step from the SVD of A, stable at the root). Last step ≤ 2.6×10⁻¹² rad/s and |D(root)| ≤ 10⁻¹⁵ in all samples.
- Gradient: ∂ω/∂p = −(yᴴ ∂_pA x)/(yᴴ ∂_ωA x), x and y the right and left singular vectors of A for its zero singular value (the adjoint formula of Magri eq. 14 for a simple root); ∂_pA = S_i R ∂T_i/∂𝔽 Q_i ∂𝔽_i/∂p with prefix/suffix products Q_i, S_i, ∂𝔽/∂n = κe^{jωτ}, ∂𝔽/∂τ = κ n jω e^{jωτ}. ∇f_Im = Im(∇ω)/(2π) for the component that attains the maximum. The maximum is not differentiable where the two growth rates are equal; this happens on a set of measure zero, and the smallest split in the gradient sets is 1.7 rad/s (weak) and 11.5 rad/s (strong).
- `growth_rate` raises `RuntimeError` if the root count differs from 2 by more than 10⁻³.

## Regenerating

```bash
python generate_data.py                       # both cases, writes the eight CSVs here
python generate_data.py --out /tmp/acx --mc 1000 --nfd 3   # quick run elsewhere
```

Seeds are saved in `generate_data.SEEDS`; a rerun reproduces the files to the digits written (the notebook's check cell reruns five samples per case: 2.5×10⁻¹² and 5.0×10⁻¹² s⁻¹). The run that wrote these files (2026-10-30, Python 3.13.18, numpy 2.5.4) took 211 s on one core of a 2-core container: 7 s per 1000 gradient samples, about 100 s per 10 000 Monte Carlo samples, and the rest for the finite-difference check. The printed log reports the nominal roots, the Magri Table 2 check, the branch checks, the gradient check and the risk factors.

## Accuracy

- **Roots.** Last Newton correction ≤ 2.6×10⁻¹² rad/s, residual |D| ≤ 10⁻¹⁵, root count 2 to 10⁻⁹ in every sample (generate_data.py log).
- **Gradients.** Against central differences of the model with step 10⁻⁵ of each input's range, 10 samples per case, all 38 components: relative error (2-norm) at most 8.0×10⁻¹⁰ (weak) and 1.9×10⁻⁹ (strong).
- **Change of variables.** The notebook checks the chain-rule gradient in Fourier coordinates against central differences through the inverse map: 2.5×10⁻⁹ (weak), 9.9×10⁻⁹ (strong).

## How these files relate to the paper and to the 2016 files

- **Not the paper's computation.** The paper's runs come from the authors' ATACAMAC code, with finite-difference gradients (JFM p. 16). That code is not public. These files come from an independent numpy implementation of the published equations with the constants above.
- **The 2016 files** (`Atacamac/atacamac-weak.txt`, `-strong.txt`, in git history before 9c1b8a7; 492 and 488 rows; reviewed in `specs/Atacamac.md`) were compared sample by sample with this model (scratch scripts, not committed):
  - **Distribution.** Their inputs are Gaussian with σ ≈ 8.7 % (n) and 4.3 % (τ) of the mean, not the papers' uniform 10 % / 5 % (spec §3). Evaluated with the 2016 Gaussian distribution, this model gives P(f_Im > 0) = 9.3 % for the weak case (1500 samples), close to the 2016 data fraction (10.37 %), and 15.33 % with the papers' uniform design. The new files use the papers' design.
  - **Weak case: same model.** At the 492 weak inputs, our f_Im (the larger root) matches the 2016 values with correlation 0.9999999994, maximum difference 1.4×10⁻³ s⁻¹ (median 9.3×10⁻⁴), and the 2016 gradients (which are with respect to standardized inputs) match ours times the sample standard deviations with median cosine 0.9996 and median relative difference 2.9 % (finite-difference error and the scaling).
  - **Strong case: one component, not the maximum.** At each of the 488 strong inputs, the 2016 value equals the growth rate of one of our two roots to 0.025 s⁻¹ (median 0.016), and the 2016 gradient matches that root's gradient (median cosine 0.9996). But it is the larger root at only 152 samples and the smaller at 336, with no rule we could find (the higher-frequency root in 326, the root nearer the symmetric one in 229). The fraction of 2016 values above zero is 37.91 %; the larger root at the same inputs gives 69.47 %. So the 2016 strong data are "the root the solver converged to". The paper's reference risk factor for the strong case, 39.55 %, is close to what one component gives: in our Monte Carlo a random one of the two is unstable in 42.55 %, the larger in 66.49 %. This is likely the cause of the strong-case disagreement, but the 10 000-run reference itself is not available, so only the authors can confirm it.
- **Validation against the paper** (notebook): weak case Monte Carlo risk factor 15.33 % ± 0.36 % against 14.77 %; surrogate risk factors FL 11.75 %, RL 11.66 %, RQ 14.66 % against Table 5's 11.90, 11.94, 14.94 %; surrogate accuracies FL 0.799, RL 0.804, RQ 0.964, RC 0.963 against Table 4's 0.80, 0.80, 0.96, 0.96; Monte Carlo frequencies 801.9–809.2 Hz against Fig. 20's 801–810 Hz axis. Strong case: risk factor 66.49 % ± 0.47 % (larger component) against 39.55 %, for the reason above; frequencies 647.6–706.7 Hz, within Fig. 20's 640–720 Hz axis.
- **Magri Table 2 case B.** The model gives 4272.26 + 3.709j rad/s; Magri prints 4.272×10³ + 3.600×10⁻³j. The frequencies agree, but the growth rate differs by 3.7 rad/s, which rounding of n = 1.75 and τ = 7.35×10⁻⁴ s cannot explain (τ ± 5×10⁻⁸ s moves it by ±0.12). Case A agrees to 0.005 rad/s. Open question.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); they were generated here from the published equations, not copied. Model: M. Bauerheim, A. Ndiaye, P. Constantine, S. Moreau and F. Nicoud (JFM 2016), with the formulation and constants of L. Magri, M. Bauerheim and M. P. Juniper (J. Comput. Phys. 2016); implementation and data: Claude (Anthropic) for P. Constantine, 2026. Cite JFM, doi:10.1017/jfm.2015.730.
