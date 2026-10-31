# STAT5_Dimerization_Signaling: data

Kind: **Python model**. The model is `STAT5_model.py`, and `generate_data.py` draws the 1000 input samples with a saved seed (2019) and writes `inputs.csv`, `outputs.csv` and `gradients.csv` from it.

## Files

| file | contents | role |
|---|---|---|
| `STAT5_Dimerization_Signaling.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2019`; outputs committed |
| `STAT5_model.py` | `qoi(xi)`, `qoi_and_grad(xi)`, `trajectory(xi, times)`, `observables(y)`; `NAMES`, `SHORT`, `UNITS`, `NOMINAL_THETA`, `NOMINAL`, `LOWER`, `UPPER`, `PETAB_LOWER`, `PETAB_UPPER`, `TIMES`, `OBS`, `QOI_NAMES` | the model; inputs `xi` are log10 of the rate constants |
| `generate_data.py` | draws the inputs (`numpy.random.default_rng(2019)`) and writes the three CSVs | regenerates the data |
| `inputs.csv` | 1000 × 6 inputs log10(θ), columns in the order of `NAMES` | model input |
| `outputs.csv` | 1000 × 48: time averages over [0, 240] min of `pSTAT5A_rel`, `pSTAT5B_rel`, `rSTAT5A_rel` (%), then each observable (%) at t = 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240 min | model output |
| `gradients.csv` | 1000 × 288: column 6 i + j is ∂(output i)/∂(log10 θ_j) | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Inputs are written with 17 significant digits (exact), outputs with 13, gradients with 11.

**Inputs are log10 values.** The study distribution is uniform in log10(θ), so the stored inputs are ξ = log10(θ) (θ in the units below) and the gradients are with respect to ξ. Pass both with `ac.Uniform(LOWER, UPPER)` on ξ. For derivatives with respect to θ itself, divide column j by θ_j ln 10.

## Model

**Citation.** H. Hass, C. Loos, E. Raimúndez-Álvarez, J. Timmer, J. Hasenauer and C. Kreutz, "Benchmark problems for dynamic modeling of intracellular processes", Bioinformatics 35(17):3073–3082, 2019, doi:10.1093/bioinformatics/btz020. The PDF read is the bioRxiv preprint (doi:10.1101/404590, posted 30 Aug 2018, CC BY 4.0); page and table numbers below refer to it. The paper is a collection of 20 models. This data set uses one, `Boehm` in Table 1 (p. 12): M. E. Boehm, L. Adlung, M. Schilling, S. Roth, U. Klingmüller and W. D. Lehmann, "Identification of isoform-specific dynamics in phosphorylation-dependent STAT5 dimerization by quantitative mass spectrometry and mathematical modeling", J. Proteome Res. 13(12):5685–5694, 2014, doi:10.1021/pr5006923. Table 1 lists 8 species, 3 observables, 48 data points, 1 experimental condition, 9 parameters, and the features C (several compartments), E(1) (constant error parameters), u(t, θ) (input with an unknown parameter) and x0(θ) (parameter-dependent initial values).

**Source of the equations.** The paper prints no model equations; it points to the collection's files. We rewrote the ODEs by hand from the SBML file of the PEtab version, `problems/Boehm_JProteomeRes2014/v1/model.xml` in github.com/Benchmarking-Initiative/Benchmark-Models-PEtab (commit c44d5ede9bf9, 2026-10-26), with the observation functions from `observables.tsv` and the best-fit values and bounds from `parameters.tsv`. The repository is BSD-3-Clause for its own content and refers to the original publications for models and data. No code was copied.

**Equations.** States in the model's concentration units (the SBML defines only the time unit, minutes): unphosphorylated STAT5A and STAT5B (A, B), phosphorylated dimers in the cytoplasm (pAB, pAA, pBB) and in the nucleus (nAB, nAA, nBB). Compartment volumes V_c = 1.4 and V_n = 0.45.

- dA/dt = −2 w_AA − w_AB + (V_n/V_c)(2 k_exp_homo nAA + k_exp_hetero nAB)
- dB/dt = −2 w_BB − w_AB + (V_n/V_c)(2 k_exp_homo nBB + k_exp_hetero nAB)
- dpAB/dt = w_AB − k_imp_hetero pAB;  dpAA/dt = w_AA − k_imp_homo pAA;  dpBB/dt = w_BB − k_imp_homo pBB
- dnAB/dt = (V_c/V_n) k_imp_hetero pAB − k_exp_hetero nAB;  dnAA/dt = (V_c/V_n) k_imp_homo pAA − k_exp_homo nAA;  dnBB/dt = (V_c/V_n) k_imp_homo pBB − k_exp_homo nBB

with w_AA = k_phos Epo(t) A², w_AB = k_phos Epo(t) A B, w_BB = k_phos Epo(t) B², Epo(t) = 1.25e-7 exp(−Epo_degradation_BaF3 · t), A(0) = 207.6 · ratio, B(0) = 207.6 · (1 − ratio), dimers 0 at t = 0. Observables, in per cent, with s = specC17 and c = 1 − s:

- pSTAT5A_rel = (100 pAB + 200 s pAA)/(pAB + s A + 2 s pAA)
- pSTAT5B_rel = (100 pAB + 200 c pBB)/(pAB + c B + 2 c pBB)
- rSTAT5A_rel = (100 pAB + 100 s A + 200 s pAA)/(2 pAB + s A + 2 s pAA + c B + 2 c pBB)

**Inputs (m = 6).** The benchmark estimates 9 parameters (Table 1). Six are the rate constants below. The other three, `sd_pSTAT5A_rel`, `sd_pSTAT5B_rel` and `sd_rSTAT5A_rel` (best fit 3.85, 6.59, 3.15), are the standard deviations of the Gaussian measurement noise; they enter only the likelihood and do not change any model prediction, so they are not inputs here. Inputs are ξ_j = log10(θ_j), independent uniform on [lower, upper]:

| name | description | units of θ | best fit θ | lower ξ | upper ξ | source |
|---|---|---|---|---|---|---|
| `Epo_degradation_BaF3` | Epo degradation rate | 1/min | 0.026982514033029 | −2.5689 | −0.5689 | PEtab `parameters.tsv`; box ours |
| `k_exp_hetero` | nuclear export, heterodimer | 1/min | 1.00067973851508e-05 | −5.0000 (PEtab bound) | −3.9997 | PEtab `parameters.tsv`; box ours |
| `k_exp_homo` | nuclear export, homodimers | 1/min | 0.006170228086381 | −3.2097 | −1.2097 | PEtab `parameters.tsv`; box ours |
| `k_imp_hetero` | nuclear import, phosphorylated heterodimer | 1/min | 0.0163679184468 | −2.7860 | −0.7860 | PEtab `parameters.tsv`; box ours |
| `k_imp_homo` | nuclear import, phosphorylated homodimers | 1/min | 97749.3794024716 | 3.9901 | 5.0000 (PEtab bound) | PEtab `parameters.tsv`; box ours |
| `k_phos` | Epo-induced phosphorylation and dimerization | 1/(conc² min) | 15766.5070195731 | 3.1977 | 5.0000 (PEtab bound) | PEtab `parameters.tsv`; box ours |

Held at their benchmark values (fixed, not estimated, in `parameters.tsv`): ratio = 0.693, specC17 = 0.107. Fixed in the SBML: total STAT5 207.6, Epo(0) = 1.25e-7, V_c = 1.4, V_n = 0.45.

**The study box: our choice, and why.** The paper gives no input distribution: it says only that "upper- and lower bounds are defined for all parameters. In most cases, these bounds cover eight orders of magnitude or even more" (Sec. 2.3, p. 5), and recommends a log10 scale for estimation (Sec. 4.2, p. 6). The PEtab bounds are [1e-5, 1e5] for all six rates, ten decades. The paper card flagged that such bounds might give stiff or failed solves. We checked this with scratch scripts:

- **Full PEtab box.** All 64 corners and 200 random points solved with Radau at rtol = atol = 1e-10, at a median 0.06 s. In a further 300 random points, all solved, but 79.7 % have a time-averaged pSTAT5A_rel below 1 %, only 15 % lie between 5 and 95 %, and 54 % of all observable values at the 15 times are below 1 % or above 99 %. Over most of that box the model barely responds to Epo, so an active subspace there would describe the on/off switch, not the measured dynamics.
- **Our box.** We take the best fit ± 1 decade in each rate, clipped to the PEtab bounds. Three rates are clipped (k_exp_hetero at the lower bound; k_imp_homo and k_phos at the upper), so their intervals are 1.0, 1.01 and 1.80 decades wide instead of 2. All 64 corners and 200 random points solved (median 1.8 s per plain solve with four processes sharing two cores), and the time-averaged pSTAT5A_rel spans 0.93–99.06 % there.

The box is defensible as "the neighborhood of the fit to the data", but it is not from the paper, and the results depend on it. The authors could replace it with a profile-likelihood or posterior range.

**Quantities of interest.** The benchmark's objective is the likelihood. Following the brief, we use smooth physical outputs instead:

- main QoI `pSTAT5A_rel_mean`: (1/240) ∫₀²⁴⁰ pSTAT5A_rel(t) dt, the time-averaged relative phosphorylation of STAT5A (%);
- the same time averages for pSTAT5B_rel and rSTAT5A_rel;
- each observable at the 15 non-zero measurement times (at t = 0 every observable depends only on ratio and specC17).

The time averages are quadrature states integrated with the ODE. Every output is a smooth function of ξ on the box: the denominators of the observables are positive whenever STAT5 is present.

**Gradients.** From the forward sensitivity equations with respect to ξ, dS/dt = J S + (∂f/∂θ) diag(θ ln 10), S(0) = 0, solved together with the model and the quadratures in one 77-state Radau solve (rtol = atol = 1e-10) with an analytic Jacobian of the model (the Newton matrix omits the second-derivative coupling of S to y, which affects only convergence). Gradients of the observables are ∇_y h · S at the output times; gradients of the time averages are quadrature states.

**Sampling design.** N = 1000 independent uniform samples of ξ, `numpy.random.default_rng(2019).random((1000, 6))` mapped to the box. The brief's minimum for the gradient estimator is max(10 m, 300) = 300. The paper has no forward sampling study; it uses multistart optimization.

**Published results.** The paper reports no sensitivity or active-subspace result for this model. For the Boehm problem it gives the model size (Table 1) and uses it, with the other 19 models, in its studies of log-transformation and optimizer performance, with no number specific to this model that we can reproduce. The collection provides the best-fit parameters (PEtab `nominalValue`), not printed in the paper.

In the notebook, for `pSTAT5A_rel_mean`:
- λ1/λ2 = 13.5, λ2/λ3 = 2.7, λ1 holds 0.895 of the trace; gap rule n = 1;
- w1: k_imp_hetero −0.812, Epo_degradation_BaF3 −0.478, k_phos +0.321, k_exp_homo −0.097; k_exp_hetero and k_imp_homo are inert (activity scores 6.5e-6 and 1.0e-12 relative to the largest);
- sigmoidal sufficient summary plot; LOO R² 0.9668 (cubic in y1), 0.9812 (quintic in y1, y2).

## Regenerating

```bash
python generate_data.py                                    # 1000 samples, 2 processes
python generate_data.py --n 20 --out /tmp/stat5 --check    # the first 20 rows, compared with the files here
```

The seed is fixed and the samples are the first N rows of one stream, so a run reproduces the files. `--workers` changes only the run time. The run that wrote the shipped files (2026-10-30; Python 3.13.18, numpy 2.5.4, scipy 1.18.1; Linux x86_64, 2 cores shared with another job) took 965 s on two processes, about 1–2 s of CPU per sample. `--n 20 --check` printed: inputs identical; maximum relative difference, outputs 4.1e-13 and gradients 4.6e-11 (the digits written). The notebook runs in about 25 s.

## Accuracy

These checks were made with scratch scripts, not committed:

- **Against the benchmark repository.** At the PEtab best fit, the observables at the 16 measurement times agree with the repository's `simulations.tsv` to 2.0e-6 (absolute, per cent), and the Gaussian negative log-likelihood of the 48 measurements with the best-fit noise parameters is 138.222.
- **Outputs.** At 5 samples (rows 0, 200, 400, 600, 800), the shipped outputs agree with the same solve at rtol = atol = 1e-12 to a relative 4.6e-9 at worst.
- **Gradients against tighter sensitivities.** At the same 5 samples, the shipped gradients agree with the sensitivity solve at rtol = atol = 1e-12 to a relative 2.7e-9 (row norm, worst output).
- **Gradients against independent central differences.** Central differences of `qoi` (plain solves at rtol = atol = 1e-12) with step h = 1e-3 in ξ, 12 extra solves per sample, at the same 5 samples: relative error (row norm) 1.0e-7 to 5.6e-7 for `pSTAT5A_rel_mean`, and at most 9.5e-5 over all 48 outputs (the largest values are for outputs whose gradients are tiny). At the best fit the error for `pSTAT5A_rel_mean` falls from 1.5e-4 at h = 1e-2 to 1.3e-5, 1.5e-6 and 1.3e-7 at h = 3e-3, 1e-3, 3e-4, as O(h²), so the differences converge to the sensitivities.
- The notebook re-solves 3 samples with sensitivities and prints the largest absolute differences from the files: about 5e-12 % (outputs) and at most 4.3e-9 % (gradients, which are up to about 120 % per decade).

## How these files relate to the paper

- **Not the paper's computation.** The paper studies parameter estimation and reports no forward sampling, sensitivities or gradients. The data here are an independent Python computation of one of its models, with a study box and outputs that we chose.
- **What is compared.** The model structure (8 states, 3 observables, 48 data points, 1 condition) matches Table 1; the 16 measurement times are from the PEtab measurement table. The input count differs (6 against 9), because the 3 noise parameters do not affect physical outputs. The simulation at the best fit matches the repository's reference simulation and likelihood value.
- **Measurements not shipped.** The 48 measurements of Boehm et al. are in the PEtab repository, which refers to the original publication for their terms. We did not find a licence for them, so we neither ship them nor ship a likelihood computed from them. With the data, a negative log-likelihood QoI (which would bring in the 3 noise parameters, m = 9) is a short addition.
- No data were copied from the paper or the repository.

## Open questions (for the authors)

- What ranges, or prior, should a forward study of the Boehm model use, rather than the 10-decade estimation bounds? (We used the best fit ± 1 decade.)
- Under what terms may the Boehm et al. (2014) measurements in the benchmark collection be redistributed, so that a likelihood QoI could be shipped?
- Three best-fit rates are at or near a bound (k_exp_hetero at 1e-5, k_imp_homo at 9.8e4, k_phos at 1.6e4 with upper bound 1e5). Are these practically non-identifiable, as the inertness of k_exp_hetero and k_imp_homo here suggests?

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). Model: M. E. Boehm et al. (2014), as curated in the benchmark collection of H. Hass et al. (2019) and its PEtab version (BSD-3-Clause repository). Notebook, model code and data: Claude (Anthropic) for P. Constantine, 2026. Cite H. Hass, C. Loos, E. Raimúndez-Álvarez, J. Timmer, J. Hasenauer and C. Kreutz, "Benchmark problems for dynamic modeling of intracellular processes", Bioinformatics 35(17):3073–3082, 2019, doi:10.1093/bioinformatics/btz020, and M. E. Boehm et al., J. Proteome Res. 13(12):5685–5694, 2014, doi:10.1021/pr5006923.
