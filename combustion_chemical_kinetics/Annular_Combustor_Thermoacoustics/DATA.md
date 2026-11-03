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

### Sources and published results

- **Paper.** M. Bauerheim, A. Ndiaye, P. Constantine, S. Moreau, F. Nicoud, "Symmetry breaking of azimuthal thermoacoustic modes: the UQ perspective", *J. Fluid Mech.* 789:534–566 (2016), doi:[10.1017/jfm.2015.730](https://doi.org/10.1017/jfm.2015.730). Read from the HAL open copy (hal-01234985v1, the author manuscript, whose body is identical to v2; page numbers below are its printed manuscript pages, not the journal's). Called JFM below.
- **Companion.** L. Magri, M. Bauerheim, M. P. Juniper, "Stability analysis of thermo-acoustic nonlinear eigenproblems in annular combustors. Part I. Sensitivity", *J. Comput. Phys.* (2016), doi:[10.1016/j.jcp.2016.07.032](https://doi.org/10.1016/j.jcp.2016.07.032), read from the preprint arXiv:1602.08438v2 (which prints that DOI). Called Magri below. It writes the same network as a nonlinear eigenproblem (eq. 30) and gives the constants missing from JFM (Table 1).
- **Earlier study, same UQ setup.** M. Bauerheim, A. Ndiaye, P. Constantine, G. Iaccarino, S. Moreau, F. Nicoud, "Uncertainty quantification of thermo-acoustic instabilities in annular combustors", *Proc. Summer Program 2014*, Center for Turbulence Research, Stanford, pp. 209–218 (cited by JFM as Bauerheim et al. 2014b). Called CTR below. Its p. 213 gives the same design (σ = 10 % of n̄ and 5 % of τ̄, "uniform distributions") and the results of the same 10 000-run Monte Carlo (Fig. 4 prints the risk factors 14.77 % and 39.55 % of JFM): mean growth rate −2.36 s⁻¹ (weak) and −3.53 s⁻¹ (strong), standard deviation 1.68 and 6.49 s⁻¹.
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
- **Published results** (JFM): five eigenvalues above 1 % of λ₁ in both cases (pp. 16–17, Fig. 10); λ₁ about two orders of magnitude larger in the strong case (p. 16); w₁ equal weights, gains nearly absent in the weak case (Figs. 11–12, p. 18); in the strong case (Fig. 12, 1000 samples) the gain and delay parts of W₁ have the same sign, about −0.08 and −0.2 (read from the figure); w₂, w₃ with period N/2, tied to F̂(±2) after the change of variables {n_i, τ_i} → {Re F̂, Im F̂} (eq. 4.2), which reduces the dimension "from 5 to only 3" (p. 19, Figs. 13–14); in Fig. 13 (500 samples) λ₄/λ₁ is about 3×10⁻³ (weak) and 6×10⁻³ (strong), read from the figure; in Fig. 14 (weak, 1000 samples) W₁ is on Re F̂(0), and W₂, W₃ span the plane Re F̂(2) = Re F̂(−2), Im F̂(2) = −Im F̂(−2); surrogate accuracy R (eq. 4.5) for the weak case at M = 1000: FL 0.80, RL 0.80, RQ 0.96, RC 0.96 (Table 4); strong case FQ1000 0.63 and RQ30 0.75 (p. 24); risk factors, Monte Carlo with 10 000 ATACAMAC runs, 14.77 % (weak) and 39.55 % (strong), and from surrogates FL1000 11.90 / 32.03, RL30 11.54 / 30.83, RL1000 11.94 / 33.17, FQ1000 15.65 / 39.26, RQ30 15.25 / 41.2, RQ1000 14.94 / 39.03 % (Table 5, p. 25). Magri Table 2: symmetric roots 5.059×10³ − 1.392j rad/s (case A: n = 0.5, τ = 0.6223 ms) and 4.272×10³ + 3.600×10⁻³j (case B: n = 1.75, τ = 0.735 ms).

### Assumptions where the papers are ambiguous

1. **L_i and α.** JFM Table 1 gives L_i⁰ = 0.04 m and no α. Magri Table 1 gives L_i = 0.0551 m and α = 0.862 and says "same parameters as" JFM. We use Magri's values. With them the model reproduces Magri Table 2 case A (5059.19 − 1.397j against 5.059×10³ − 1.392j) and the 2016 weak-case data to 1.4×10⁻³ s⁻¹ at all 492 samples; with L_i = 0.04 m and α = 0.5, 0.862 or 1 the case-A root is 5136–5379 rad/s with growth rates −15 to −74 rad/s (external run, not reproducible from the repository), far from both.
2. **The state vector and product order.** JFM writes X = [p̂_p, û_p, p̂_c, û_c] but its coupling example uses −jρ⁰c⁰û; we use Magri's X = [p̂_p, −jρ_u c_u û_p, p̂_c, −jρ⁰c⁰û_c] (eq. 35), for which the propagation matrix of eq. (2.2) is exact, and the product R T_N ⋯ R T_1 (JFM eq. 2.10). The determinant does not depend on where the cycle starts or on the burners' order.
3. **Which root, and the maximum.** "The first azimuthal mode" is taken as the pair of roots that the double root of the symmetric configuration splits into. The search circle (300 rad/s weak, 450 rad/s strong, around the symmetric root) must contain exactly these two roots; the nearest other roots of the symmetric configuration are the plenum mode (3416 − 53j weak, 3493 − 350j strong) and the second azimuthal mode (6266 + 14j, 6250 + 59j rad/s), 850–2000 rad/s away. In all 2 × 11 000 samples the root count is 2 to 10⁻⁹, and in the gradient sets the roots stay within 37 rad/s (weak) and 189 rad/s (strong) of the symmetric root.
4. **Units.** Im(f) = Im(ω)/(2π) in s⁻¹, as on JFM's axes ("Growth rate (s⁻¹)", Fig. 20).
5. **σ of τ.** Exactly 5 % of the mean (31.75 and 36.75 µs), which JFM Table 2 rounds to 31.7 and 36.7 µs.

### Solver

- The symmetric root solves the Bloch condition det(R T − e^{2πj/N} I) = 0 by the secant method from ω⁰ (weak: 5058.035 − 14.690j rad/s, 805.011 Hz, Im f = −2.338 s⁻¹; strong: 4272.262 + 3.709j rad/s, 679.952 Hz, Im f = +0.590 s⁻¹).
- For each sample, the contour integrals (1/2πj)∮ zᵏ D′/D dω, k = 0, 1, 2 (trapezoid rule, 64 points), give the root count and the two roots (Newton's identities); D′/D = tr(A⁻¹ ∂_ω A) with A = M − I and ∂_ω A from the analytic derivatives of R and Γ. Six Newton steps on D/(ω − ω_other) polish each root (step from the SVD of A, stable at the root). Last step ≤ 3.8×10⁻¹² rad/s and |D(root)| ≤ 10⁻¹⁵ in all samples.
- Gradient: ∂ω/∂p = −(yᴴ ∂_pA x)/(yᴴ ∂_ωA x), x and y the right and left singular vectors of A for its zero singular value (the adjoint formula of Magri eq. 14 for a simple root); ∂_pA = S_i R ∂T_i/∂𝔽 Q_i ∂𝔽_i/∂p with prefix/suffix products Q_i, S_i, ∂𝔽/∂n = κe^{jωτ}, ∂𝔽/∂τ = κ n jω e^{jωτ}. ∇f_Im = Im(∇ω)/(2π) for the component that attains the maximum. The maximum is not differentiable where the two growth rates are equal; this happens on a set of measure zero, and the smallest split in the gradient sets is 1.7 rad/s (weak) and 11.5 rad/s (strong).
- `growth_rate` raises `RuntimeError` if the root count differs from 2 by more than 10⁻³.

## Regenerating

```bash
python generate_data.py                       # both cases, writes the eight CSVs here
python generate_data.py --out /tmp/acx --mc 1000 --nfd 3   # quick run elsewhere
python generate_data.py --out /tmp/acx --check   # full run elsewhere, compared with the files here
```

Seeds are saved in `generate_data.SEEDS`; a rerun reproduces the files to the root-finding error (the notebook's check cell reruns five samples per case: 2.5×10⁻¹² and 5.0×10⁻¹² s⁻¹). The shipped files were written on 2026-10-30 in a Linux container (Python 3.13.18, numpy 2.5.4). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-10-31) a full rerun took 372 s on one core, with other jobs running: 11–17 s per 1000 gradient samples, 160–190 s per 10 000 Monte Carlo samples, and the rest for the finite-difference check. It reproduces the inputs exactly, and every other file to at most 1.2×10⁻¹¹ of each column's largest value (elementwise, a relative 10⁻⁸ in the smallest gradient components). `--check` (it refuses to run without `--out` naming another directory) prints these differences file by file; on 2026-11-03 in `.venv` it took 161 s and printed 0 for the inputs, at most 7.4×10⁻¹³ for the outputs and gradients and 1.2×10⁻¹¹ for the Monte Carlo files. The printed log reports the nominal roots, the Magri Table 2 check, the branch checks, the gradient check and the risk factors.

## Accuracy

- **Roots.** Last Newton correction ≤ 3.8×10⁻¹² rad/s, residual |D| ≤ 10⁻¹⁵, root count 2 to 10⁻⁹ in every sample (generate_data.py log in `.venv`).
- **Gradients.** Against central differences of the model with step 10⁻⁵ of each input's range, 10 samples per case, all 38 components: relative error (2-norm) at most 8.5×10⁻¹⁰ (weak) and 1.9×10⁻⁹ (strong).
- **Change of variables.** The notebook checks the chain-rule gradient in Fourier coordinates against central differences through the inverse map: 2.2×10⁻⁹ (weak), 9.8×10⁻⁹ (strong).

## How these files relate to the paper and to the 2016 files

- **Not the paper's computation.** The paper's runs come from the authors' ATACAMAC code, with finite-difference gradients (JFM p. 16). That code is not public. These files come from an independent numpy implementation of the published equations with the constants above.
- **The 2016 files** (`Atacamac/atacamac-weak.txt`, `-strong.txt`, in git history before 9c1b8a7; 492 and 488 rows; reviewed in `docs/port-2026/Atacamac.md`) were compared sample by sample with this model (external run, not reproducible from the repository):
  - **Distribution.** Their inputs are Gaussian with σ ≈ 8.7 % (n) and 4.3 % (τ) of the mean, not the papers' uniform 10 % / 5 % (spec §3). Evaluated with the 2016 Gaussian distribution, this model gives P(f_Im > 0) = 9.3 % for the weak case (1500 samples), close to the 2016 data fraction (10.37 %), and 15.33 % with the papers' uniform design. The new files use the papers' design.
  - **Weak case: same model.** At the 492 weak inputs, our f_Im (the larger root) matches the 2016 values with correlation 0.9999999994, maximum difference 1.4×10⁻³ s⁻¹ (median 9.3×10⁻⁴), and the 2016 gradients (which are with respect to standardized inputs) match ours times the sample standard deviations with median cosine 0.9996 and median relative difference 2.9 % (finite-difference error and the scaling).
  - **Strong case: one component, not the maximum.** At each of the 488 strong inputs, the 2016 value equals the growth rate of one of our two roots to 0.025 s⁻¹ (median 0.016), and the 2016 gradient matches that root's gradient (median cosine 0.9996). But it is the larger root at only 152 samples and the smaller at 336, with no rule we could find (the higher-frequency root in 326, the root nearer the symmetric one in 229). The fraction of 2016 values above zero is 37.91 %; the larger root at the same inputs gives 69.47 %. So the 2016 strong data are "the root the solver converged to". The paper's reference risk factor for the strong case, 39.55 %, is close to what one component gives: in our Monte Carlo a random one of the two is unstable in 42.55 %, the larger in 66.49 %. This is one hypothesis for the strong-case disagreement (see "Validation against the paper": the standard deviation supports it, the mean does not). The 10 000-run reference itself is not available, so only the authors can confirm it.
  - **Strong-case w₁.** The leading eigenvector of the mean outer product of the 2016 strong-case gradients (columns dn01–dtau19; delay part taken negative) has gain part +0.168, delay part −0.156 and gain share 0.54, as in the model (+0.170, −0.154, 0.552), and unlike JFM Fig. 12 (same sign for both parts). Recipe: `git show 9c1b8a7^:Atacamac/atacamac-strong.txt`, `np.linalg.eigh` of G.T @ G / 488 for the 38 gradient columns G.
- **Validation against the paper** (notebook). The model matches the paper's 10 000-run Monte Carlo reference in the weak-case risk factor only.
  - Weak case, matched: Monte Carlo risk factor 15.33 % ± 0.36 % against 14.77 %; surrogate risk factors FL 11.75 %, RL 11.66 %, RQ 14.66 % against Table 5's 11.90, 11.94, 14.94 %; surrogate accuracies FL 0.799, RL 0.804, RQ 0.964, RC 0.963 against Table 4's 0.80, 0.80, 0.96, 0.96; Monte Carlo frequencies 801.9–809.2 Hz against Fig. 20's 801–810 Hz axis.
  - Weak case, unexplained: CTR's mean and standard deviation, −2.36 and 1.68 s⁻¹. f_Im = max has −1.41 and 1.36 s⁻¹. Both components pooled have −2.39 and 1.69 s⁻¹, close to CTR's, but their risk factor is 8.00 %, and a normal law with CTR's mean and standard deviation also gives 8.0 %, not 14.77 %. So no single quantity of this model gives all three published numbers.
  - Strong case, unexplained: risk factor 66.49 % ± 0.47 % (larger component) against 39.55 %; surrogate risk factors 66.6–68.1 % against 32.0–39.0 %; mean and standard deviation +2.30 and 5.49 s⁻¹ (f_Im) against CTR's −3.53 and 6.49 s⁻¹. Frequencies 647.6–706.7 Hz, within Fig. 20's 640–720 Hz axis. The root-choice hypothesis above is supported by the risk factor (a random component: 42.55 %) and by the standard deviation (both components pooled: 6.89 s⁻¹), but not by the mean (pooled: −1.50 s⁻¹, 2 s⁻¹ above CTR's, against a sampling error of about 0.07 s⁻¹).
  - Strong-case w₁: gain part +0.170 and delay part −0.154 (gains 55 % of ‖w₁‖²) against Fig. 12's same-sign parts (gains roughly 14 %). The notebook orients w₁ with a negative delay part, as in Figs. 11–12, so its gain part cannot have Fig. 12's sign.
  - Fourier coordinates: λ₄/λ₁ = 0.0515 (weak) and 0.0713 (strong) against about 3×10⁻³ and 6×10⁻³ in Fig. 13, so the model does not show the reduction "from 5 to only 3". The model's weak w₁ is on Im F̂(0), not Re F̂(0) as in Fig. 14, and its span(w₂, w₃) lies in the plane Re F̂(2) = −Re F̂(−2), Im F̂(2) = Im F̂(−2) (share 0.001 in Fig. 14's plane); the model's strong case has Fig. 14's pattern (w₁ on Re F̂(0), share 0.993). Multiplying every F_i by j maps the model's weak pattern onto Fig. 14's.
- **Magri Table 2 case B.** The model gives 4272.26 + 3.709j rad/s; Magri prints 4.272×10³ + 3.600×10⁻³j. The frequencies agree, but the growth rate differs by 3.7 rad/s, which rounding of n = 1.75 and τ = 7.35×10⁻⁴ s cannot explain if the printed exponent is right: τ = 7.3495×10⁻⁴ and 7.3505×10⁻⁴ s give +3.829j and +3.588j (notebook, cell 5). The printed +3.600×10⁻³ would agree to rounding if the factor 10⁻³ were a misprint; this is a possibility, not confirmed. Case A agrees to 0.005 rad/s. Also unexplained: Magri (p. 15, under Table 2) says case A uses "a slightly smaller τ" than JFM's Wcase "to have a stable configuration", but in the model the Wcase τ̄ = 0.635 ms is the more stable one (Im ω = −14.690 rad/s, against −1.397 at 0.6223 ms).

## Open questions

Points only the authors can settle:

1. **The Monte Carlo reference.** Which quantity the 10 000-run reference reports: its weak-case risk factor fits f_Im = max, its weak-case mean and standard deviation fit the two components pooled, its strong-case risk factor and standard deviation are nearer to one component (or both pooled) than to the maximum, and its strong-case mean fits neither.
2. **Strong-case w₁ (JFM Fig. 12).** The model and the 2016 gradients give gain and delay parts of opposite sign; Fig. 12 shows the same sign and smaller gain weights.
3. **Fourier coordinates (JFM Figs. 13–14).** Fig. 13's λ₄/λ₁ is more than ten times smaller than the model's in both cases. Fig. 14's weak-case W₁ is on Re F̂(0), which near ω⁰τ̄ ≈ π weights the gains and so does not fit Fig. 11; the model's weak w₁ is on Im F̂(0), and its w₂, w₃ lie in the other plane of F̂(±2). A phase convention for F (a factor j) or Fig. 14 showing another case would explain the pattern, but not Fig. 13's eigenvalues.
4. **Magri Table 2 case B.** A possible misprint of the exponent of Im ω₀, and the sentence on the "slightly smaller τ" of case A (above).

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); they were generated here from the published equations, not copied. Model: M. Bauerheim, A. Ndiaye, P. Constantine, S. Moreau and F. Nicoud (JFM 2016), with the formulation and constants of L. Magri, M. Bauerheim and M. P. Juniper (J. Comput. Phys. 2016); implementation and data: Claude (Anthropic) for P. Constantine, 2026. Cite JFM, doi:10.1017/jfm.2015.730.
