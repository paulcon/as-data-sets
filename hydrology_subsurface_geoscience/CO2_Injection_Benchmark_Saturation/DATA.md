# CO2_Injection_Benchmark_Saturation: data (closed-form analogue of the Köppel et al. benchmark)

**Not the authors' model.** The data are the closed-form (exact entropy, Buckley–Leverett) solution of the paper's Eq. (11) evaluated at the benchmark's 10,000 input samples, **not** the paper's finite-volume reference model. A flux factor C_Q = 0.643 (the paper's equations have 1) and the map from the coded porosity samples to porosity were fitted to the authors' Monte Carlo curves. The closed form is a different model from the authors': at r = 100.4 m its standard deviation of S is 0.271 against the reference's 0.309, and it misses the reference also where its solution is smooth (injection rate only, mean S 0.616 against 0.695 at 60 m); see "Accuracy".

Kind: **Python model**, run on stored inputs. The model is `co2_benchmark_model.py`, the closed-form solution of Eq. (11) with the fitted constants, written in numpy/scipy from the paper. `generate_data.py` writes `inputs.csv`, `outputs.csv` and `gradients.csv` from it. The 10,000 input samples and the authors' Monte Carlo reference curves are stored data from the benchmark's Zenodo archive, converted losslessly. The authors' own solver (executables in the archive) was not run; see "How these files relate to the paper".

## Files

| file | contents | role |
|---|---|---|
| `CO2_Injection_Benchmark_Saturation.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 933827; outputs committed |
| `co2_benchmark_model.py` | `saturation`, `saturation_field`, `front_radius`, `grad_front_radius`, `qoi`, `shock`, `frac_flow`, `coded_to_physical`; constants of Table 1, the fitted `C_Q`, `A_PHI`, `B_PHI`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `RADII` | the model (numpy, scipy) |
| `generate_data.py` | writes the three model CSVs; `--from-archive DIR` also rewrites the two stored CSVs from the unzipped archive; `--check` compares a rerun with the shipped files | regenerates the data |
| `calibrate.py` | refits `C_Q`, `A_PHI`, `B_PHI` and the held values of the single-input reference cases | documents the calibration |
| `benchmark_input_samples.csv` | 10,000 × 3 coded samples `ir`, `kd`, `po` (the archive's `InputParameters.csv`) | stored input data |
| `mc_reference_mean_std.csv` | 250 × 9: radius x (m) and mean/std of S for the joint and the three single-input cases (the archive's `MCrefs_MeanStd.csv`) | stored reference data |
| `inputs.csv` | 10,000 × 3 physical inputs Q (m²/s), n (1), φ (1) | model input |
| `outputs.csv` | 10,000 × 6: S at r = 50, 75, 100, 125, 150 m, and the front radius r_f (m), at t = 100 days | model output |
| `gradients.csv` | 10,000 × 3: ∂r_f/∂(Q, n, φ) with respect to the **physical** inputs | model output |
| `LICENSE-uqBenchmarkData.txt` | the archive's BSD 3-clause licence, unchanged | licence of the stored data |

Every CSV has two `#` lines (a description, then the column names with units) and is read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits; the two stored files round-trip the archive's float values exactly (checked when written).

## Model

**Source.** M. Köppel, F. Franzelin, I. Kröker, S. Oladyshkin, G. Santin, D. Wittwar, A. Barth, B. Haasdonk, W. Nowak, D. Pflüger, C. Rohde, "Comparison of data-driven uncertainty quantification methods for a carbon dioxide storage benchmark scenario", *Computational Geosciences* 23(2):339–354, 2019, doi:10.1007/s10596-018-9785-x. Read from the open-access preprint arXiv:1802.03064v1 (8 Feb 2018); page and equation numbers below are the preprint's. Data and executables: Zenodo, doi:10.5281/zenodo.933827 (paper ref. [28]), file `uqBenchmarkData.zip`, 2,691,045 bytes, sha256 fa88bc303c8a58ce9e0155fa86e88897ad1c6806d0be9bc4972094e4da87001c, fetched 2026-11-01.

**Equations** (Sec. 2–3, pp. 3–5). Radially symmetric, capillarity-free fractional-flow model of CO₂ injected into a homogeneous brine aquifer; incompressible immiscible phases, isothermal, constant fluid properties (a reduction of the Class et al. 2009 benchmark). The pressure is analytic (Eqs. 10, 13), so the total radial flux is Q/r and the CO₂ saturation obeys Eq. (11):

φ ∂S/∂t + (Q/r) ∂f(S; n)/∂r = 0, S(r, 0) = 0, S(0, t) = 0.8 (Table 1: "saturation on the left boundary"),

f = λ_g/(λ_g + λ_w), λ_g = Sⁿ/μ_g, λ_w = (1 − S)ⁿ/μ_w (Eqs. 5–6, 14–15), Q = C_p q_CO2 (1 + θ₁) (Eq. 13).

Eq. (11) (and Eq. 12) carries a volumetric source term −q_CO2 in the domain; it is read here as acting only at the well and represented by the left boundary value. This is an interpretation (listed in the report's unknowns). Eq. (6) as printed, k_r,w = (1 − S_w*)², is read here as (1 − S)ⁿ with S := S_g* (the paper's own abbreviation above Eq. 12); with the printed form the brine would be immobile at full brine saturation.

**Solution.** The paper's code solves Eq. (11) numerically by a semi-discrete central-upwind finite-volume scheme (Kurganov and Petrova 2005) with Δr = 1 m and second-order Runge–Kutta (Sec. 2.2); that is the paper's reference model. `co2_benchmark_model.py` evaluates the closed-form exact entropy solution instead, a different and simpler model (see "Accuracy"). With ξ = φ r²/(2 C_Q Q t): S = 0.8 for ξ ≤ f′(0.8); f′(S) = ξ on the rarefaction (bisection, 80 steps; 60 in the vectorized `saturation_field`); and a shock from the Welge saturation S_s (f′(S_s) S_s = f(S_s), `brentq` to 1e-15) to 0 at ξ_s = f(S_s)/S_s. The front radius is r_f = √(2 C_Q Q t ξ_s(n)/φ). Over n ∈ [2, 4], S_s = 0.367–0.506 and ξ_s = 1.862–1.731.

**Constants** (Table 1, p. 4): μ_g = 3.950·10⁻⁵ Pa s, μ_w = 2.535·10⁻⁴ Pa s, K_A = 2·10⁻¹⁴ m², λ = 1.0·10⁴ (Pa s)⁻¹, p_max = 320 bar, p_min = 300 bar, r_max = 500 m, t = 100 days, S(left) = 0.8. C_p q_CO2 = (p_max − p_min) K_A λ / ln r_max = 6.4364·10⁻⁵ m²/s, consistent with the paper's C_p = 3.48·10⁻³ and q_CO2 = 1600 m³/d (p. 4). Q has units m²/s because C_p has units 1/m; the paper's Fig. 1 labels the injection rate in m³/s. The residual saturations and well radius of Table 1 do not enter Eq. (11) as written and are not used.

**Inputs** (independent, given only as 10,000 samples; Sec. 3, Fig. 1):

| name | symbol | units | from the coded sample | sample min / mean / sd / max | source |
|---|---|---|---|---|---|
| `Q` | injection rate C_p q_CO2 (1 + θ₁) | m²/s | Q₀ (1 + `ir`) | 6.740e-5 / 1.074e-4 / 1.158e-5 / 1.286e-4 | Eq. 13; Fig. 1 bars about 0.680–1.286·10⁻⁴ (labelled m³/s) |
| `n` | relative permeability degree θ₂ | 1 | 2 + 2 `kd` | 2.001 / 3.002 / 0.536 / 3.999 | Eqs. 14–15; Fig. 1 axis 2–4 |
| `phi` | porosity θ₃ | 1 | 0.09151 + 0.19647 `po` (fitted) | 3.41e-4 / 0.2021 / 0.1146 / 0.759 | Sec. 3 (NPC database), Fig. 1 bars about 0.004–0.73 |

The archive's `InputParameters.csv` gives the samples in coded form (`ir` ∈ [0.047, 0.998], `kd` ∈ [0.0004, 0.9996], `po` ∈ [−0.464, 3.398]); the archive README calls them "the distribution of the uncertain parameters" without the maps. The maps for `ir` and `kd` approximately reproduce the end points of the Fig. 1 bars (for `ir`, 0.674 against about 0.680·10⁻⁴ at the low end); the map for `po` is fitted (below). **Fig. 1 does not show these samples under any increasing affine map**: equal-width bin counts (10 bins) do not change under such maps, yet they differ. Porosity, archive `po`: 1257, 2574, 2504, 1826, 1056, 493, 201, 70, 16, 3; Fig. 1 (bar heights read from a 600-dpi render of p. 6 in the review; the first bar, about 1050, is plain in the figure): 1050, 2433, 2469, 1835, 1217, 587, 281, 100, 33, 6. Injection rate, archive: 27, 110, 339, 624, 1102, …; Fig. 1: 25, 112, 363, 718, 1067, …. The review found that exponential or power maps do not reproduce the figure's counts either. So the figure's axis end points are weak evidence for the maps, and whether the authors' Monte Carlo reference was computed on the archived samples is an open question (report `unknowns`). The notebook prints both sets of counts. The 4-column `InputParameters.txt` (the executable's input) has, as its second column, residual-saturation samples that the paper's study does not use; they are not shipped.

**Quantities of interest.** The paper studies the mean and standard deviation of S(r, t = 100 d) over r (Fig. 2). For a scalar QoI this data set takes S at r = 100 m (near the peak of the reference standard deviation) and the front radius r_f; S is also stored at 50, 75, 125 and 150 m.

**Gradients.** `gradients.csv` holds ∂r_f/∂(Q, n, φ): exact for Q and φ (r_f ∝ √(Q/φ)); for n, the shock speed ξ_s(n) is differentiated by a central difference with step 1e-4. S at a fixed radius is discontinuous (it jumps when the front passes), so no gradients of S are stored; the notebook estimates its subspace from values.

**Sampling design.** The benchmark's 10,000 samples, used as given (Monte Carlo samples from the data-driven distributions; Sec. 3, p. 5). The paper uses the same set Θ for its Monte Carlo reference.

**Published results.** No active subspace. Monte Carlo mean and standard deviation of S(r) at 100 days for the joint and single-input cases (Fig. 2, numbers in `mc_reference_mean_std.csv`): joint mean 0.308 and standard deviation 0.309 at r = 100.4 m. Qualitatively (p. 5), the relative permeability degree does not influence the saturation significantly, and the injection rate and the porosity are the main sources of uncertainty. Four surrogate methods (aPC, adaptive sparse grids, kernel greedy, hybrid stochastic Galerkin) are compared against this reference (Secs. 4–5).

## Calibration (what the paper does not state)

The exact solution was fitted to the authors' Monte Carlo reference curves by least squares (`calibrate.py`, Nelder–Mead on the mean and standard deviation curves over the 250 points, every 4th sample, profiles tabulated on a 0.025 grid in n):

- **Porosity map**: φ = A_PHI + B_PHI `po`, A_PHI = 0.09151, B_PHI = 0.19647, from the joint curves. The resulting range 0.00034–0.759 is somewhat wider than the porosity bars of the paper's Fig. 1 (about 0.004–0.73 measured in the review, implying φ ≈ 0.0911 + 0.1886 `po`, a 4 % smaller slope), and Fig. 1's bin counts differ from the archived samples (above), so the figure does not confirm the map.
- **Flux factor** C_Q = 0.64299 multiplying Q. The joint curves alone determine only C_Q/φ; the injection-rate-only curves with φ held at the nominal 0.15 (Table 1) separate the two. With the paper's equations as written (C_Q = 1) every front would lie 1/√0.64299 = 1.25 times farther out than in the authors' reference. The origin of the factor in the authors' code is unknown (`unknowns` in the report). A second start of the fit gave C_Q = 0.6434, A_PHI = 0.0915, B_PHI = 0.1967 (agreement to about 0.1 %).
- **Held values of the single-input reference cases** (used only for the comparison figure): n = 2.531 in the injection-rate-only case; injection factor 1 + θ₁ = 0.6806 (θ₁ = −0.32) and φ = 0.15 in the relative-permeability-only case; 1 + θ₁ = 0.6806 and n = 2.437 in the porosity-only case. These are not natural nominal values, and θ₁ = −0.32 lies outside the benchmark's sample range of θ₁ (0.047–0.998); the paper does not say what was held.
- **C_Q is not one consistent constant.** Fitted alone to each reference case with the other inputs held at natural values (1 + θ₁ = 1, φ = 0.15, n = 2 or 3; `calibrate.py`, `cq_consistency`, all 10,000 samples), C_Q is 0.643 (joint, porosity map fixed) and 0.647 (injection rate only, n = 3), but 0.436 (relative permeability degree only) and 0.462 / 0.446 (porosity only, n = 2 / 3): the ratio joint / degree only is 1.47. The held injection factor 0.6806 above absorbs this 1.47× inconsistency (0.6806 ≈ 1/1.47); one flux factor does not describe all four of the authors' cases.

Root-mean-square differences from the reference (all 10,000 samples, 250 points; notebook): joint 0.026 (mean) and 0.022 (std); injection rate only 0.038 and 0.025; relative permeability degree only 0.037 and 0.012; porosity only 0.024 and 0.016. All four are fits, not checks: the joint and injection-rate-only curves set C_Q and the porosity map, and the held values of the other two cases were fitted to their own curves. In particular the relative-permeability-only front at 76 m (reference and model) holds by construction, since in that case only C_Q·g matters and g was fitted to those curves.

**Samples outside physical meaning.** No porosity is ≤ 0 or > 1 under the fitted map, but 41 samples have φ < 0.01 and 603 have φ < 0.05, which is implausibly low for a storage reservoir; they come from the benchmark's own sample tail (the paper's Fig. 1 shows the same tail). They are kept in the files. 305 fronts lie beyond 250 m (the reference output range) and 30 beyond the authors' domain radius r_max = 500 m (all with φ < 0.05). The exact solution ignores the outer boundary; this does not change S at r ≤ 250 m (all characteristics travel outward), but those 30 values of r_f have no counterpart in the authors' code. The notebook reports the front-radius subspace with and without φ < 0.05.

## Regenerating

```bash
python generate_data.py                                     # the three model CSVs, about 10 s
python generate_data.py --out /tmp/co2 --check              # rerun elsewhere and compare
python generate_data.py --from-archive <dir>/uqBenchmarkData  # also rewrite the two stored CSVs
python calibrate.py                                         # refit the constants, about 1 min
```

There is no randomness. Run times (this container, one core, Python 3.13.18, numpy 2.5.4, scipy 1.18.1, 2026-11-01): `generate_data.py` 8–13 s, `calibrate.py` 34–91 s (52 s with the consistency check added after the review), the notebook 57–108 s depending on the load of the shared container (most of it the four saturation fields of the reference comparison). `--check` reproduces all five CSVs with difference 0. `calibrate.py` prints the constants used in the module (C_Q = 0.64299, A_PHI = 0.09151, B_PHI = 0.19647; held n 2.531; 1 + θ₁ = 0.6806, n = 2.437) and the C_Q consistency check above.

## Accuracy

- **Model vs. files**: the notebook re-solves five samples and finds relative difference 0.
- **Exact solution**: S_s and ξ_s are solved to `brentq` tolerance 1e-15; the rarefaction inversion is bisection on an interval of length < 0.5 with 60–80 halvings (error below 1e-15). `saturation_field` agrees with the scalar `saturation` to 1e-12 (checked at three samples).
- **Gradients**: against an independent central difference of `front_radius` in each input (relative step 1e-6) at 50 random samples, the largest relative error of the gradient (norm, scaled by the input standard deviations) is 3.3e-9, and of the n-component alone 2.1e-7. The n-derivative with steps 1e-3, 1e-4, 1e-5 gives −0.978472, −0.978471, −0.978471 at one sample.
- **Against the authors' solver: a different model.** Not checked sample by sample (the executables were not run, and the archive has no per-sample outputs). Through the Monte Carlo statistics, the closed form misses the reference where it was not fitted: the joint standard deviation of S at 100.4 m is 0.271 against 0.309 (12 % low; peak 0.273 at 105 m against 0.309 at 98 m), and in the injection-rate-only case 0.044 against 0.134 (3.06 times lower). **Front smearing is not the cause**: smearing lowers the standard deviation at a fixed radius. Smoothing every exact joint profile along r with a Gaussian of σ = 1, 2, 4 and 8 cells (1.004 m) lowers the std at 100.4 m from 0.271 to 0.269, 0.266, 0.260 and 0.246, away from the reference. The mismatch also appears where there is no front: in the injection-rate-only case every front lies between 93.9 and 129.7 m, so at 60 and 80 m all samples are on the smooth rarefaction, yet the mean is 0.616 against 0.695 at 60.2 m and 0.548 against 0.608 at 80.3 m, and the std 0.013 against 0.024 and 0.014 against 0.028. This difference is structural: the authors' flux function, source term or held values are not what the closed form assumes. The std deficits are unexplained. All numbers are printed by the notebook ("What the closed form does not reproduce").

## How these files relate to the paper

- **Not the authors' computation.** The paper's numbers come from a central-upwind finite-volume code whose Linux and Windows executables (`prog`, `win_prog.exe`, no source) are in the Zenodo archive. They were not run here (build rule). The data here are an independent closed-form (exact) solution of the paper's Eq. (11), with a flux factor and the porosity map fitted to the authors' published Monte Carlo statistics; it is a different model from the authors' (see "Accuracy"), and its active subspace is that of this closed-form analogue. Per-sample outputs of the authors' code are not in the archive (only the mean/std curves), so a sample-by-sample comparison was not possible.
- **Inputs.** The 10,000 coded samples are the benchmark's own and are unchanged in `benchmark_input_samples.csv`; `inputs.csv` is their physical form under the maps above.
- **Not the same study as `Geological_CO2_Storage_Leakage`.** That data set (Amrhein et al., preprint 2026) is a steady 1-D Darcy leakage model through a layered caprock above a storage site, with 22–30 inputs (layer permeabilities, viscosities, thicknesses) and QoIs leakage time and leakage rate. This data set is the transient two-phase displacement in the storage formation itself near the well (Buckley–Leverett), with three inputs and the CO₂ saturation and plume front as QoIs. The two share no model, inputs or data.

## License and credit

Code (`co2_benchmark_model.py`, `generate_data.py`, `calibrate.py`, the notebook): MIT, written from the paper's equations; no code from the archive was used. Model data (`inputs.csv`, `outputs.csv`, `gradients.csv`): CC BY 4.0. Stored data (`benchmark_input_samples.csv`, `mc_reference_mean_std.csv`): from the Zenodo archive doi:10.5281/zenodo.933827, copyright 2017 Fabian Franzelin, Markus Koeppel, Ilja Kroeker, Sergey Oladyshkin, Gabriele Santin, Dominik Wittwar, under the BSD 3-clause licence in `LICENSE-uqBenchmarkData.txt`, which asks users to cite the benchmark paper (cited there under its working title, "Transfer and intercomparison of data-driven uncertainty quantification methods for two-phase flow in carbon dioxide storage"; published as the paper above). Cite M. Köppel et al., Comput. Geosci. 23(2):339–354, 2019, doi:10.1007/s10596-018-9785-x, and the Zenodo record. The paper's work was funded by the DFG within the Cluster of Excellence in Simulation Technology (EXC 310/2), University of Stuttgart.
