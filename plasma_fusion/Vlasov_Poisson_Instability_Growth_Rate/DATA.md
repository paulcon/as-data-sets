# Vlasov_Poisson_Instability_Growth_Rate: data

Kind: **Python model**. The model is `plasma_dispersion_model.py` (numpy and scipy only), a reimplementation from the paper's equations, and `generate_data.py` writes all five CSV files from it with a saved seed (2024).

## Files

| file | contents | role |
|---|---|---|
| `Vlasov_Poisson_Instability_Growth_Rate.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2024; outputs committed |
| `plasma_dispersion_model.py` | `dispersion`, `most_unstable_root`, `growth_rate`, `growth_rate_and_grad`, `RootError`; `NAMES`, `NOMINAL_TS/DB/BOT`, `GLOBAL_LOWER/UPPER`, `percent_box` | the model |
| `generate_data.py` | draws the samples and writes the five CSVs | regenerates the data |
| `global_bimaxwellian.csv` | 1000 × 15: k, mu1, mu2, s1, s2, beta, gamma, omega_r, ∂gamma/∂(k, mu1, mu2, s1, s2, beta), gap | main study (global box) |
| `double_beam.csv` | 3072 × 16: pct, then as above | Double Beam, ± 1, 5, 10, 15, 20, 25 % (512 each) |
| `bump_on_tail.csv` | 2560 × 16 | Bump-on-Tail with Table II's baseline, ± 1, 5, 10, 15, 25 % |
| `bump_on_tail_s025.csv` | 2560 × 16 | Bump-on-Tail with sigma1² = sigma2² = 0.25 and mu1 held at 0 (the setup the paper's numbers imply; see below) |
| `two_stream.csv` | 3584 × 10: pct, k, mu, s, gamma, omega_r, ∂gamma/∂(k, mu, s), gap | Two-Stream, ± 1, 5, 10, 15, 20, 25, 50 % |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names), so `np.loadtxt(path, delimiter=',')` reads them. Every quantity is dimensionless. `s` stands for sigma² (the paper's "thermal velocity"). Inputs and gradients are in **physical** coordinates; the gradient with respect to the normalized input x_j ∈ [−1, 1] is ∂gamma/∂p_j times (upper_j − lower_j)/2. `gap` is gamma minus the growth rate of the next root found: small values (< 1e-3) mark samples near a crossing of root branches, where gamma has a kink.

## Model

**Citation.** S. Terrab and S. Pankavich, "Global sensitivity analysis of plasma instabilities via active subspaces", *Communications in Nonlinear Science and Numerical Simulation*, article 107994, 2024, doi:10.1016/j.cnsns.2024.107994. Built from the arXiv preprint 2106.08552v2 (8 April 2022); page, table and figure numbers below are the preprint's, and published numbers should be checked against the journal version. Volume not confirmed.

**Equations.** The dimensionless 1-D Vlasov–Poisson system (paper Eq. (1)) linearized about a homogeneous equilibrium; a mode e^{i(kx − ωt)} grows at gamma = Im ω, where ω is a root of the dispersion function (Eq. (2)). With u = ω/k and A_i = (u − mu_i)/sqrt(2 s_i):

- Bi-Maxwellian (Eq. (5)): k² ε = k² + (beta/s1)[1 + A1 Z(A1)] + ((1 − beta)/s2)[1 + A2 Z(A2)].
- Two-Stream (Eq. (3)): k² ε = k² − [1 − 2A² + 2(A − A³) Z(A)].

Z(z) = i √π w(z) is the plasma dispersion function, with w the Faddeeva function (`scipy.special.wofz`), the Landau-continued Z; the paper used the MATLAB code `zetaf.m` (Zaghloul and Ali 2011). We checked the Z-function form against direct quadrature of Eq. (2) at the Double Beam and Two-Stream baseline roots: |ε| = 3e-15 and 7e-16.

**Quantity of interest.** gamma = k Im u*, where u* is the root of k² ε with the **largest imaginary part** (the most unstable mode, or the least-damped one for a stable equilibrium). The paper does not say which root its solver returns (Sec. IV); see "How these files relate to the paper".

**Root finding** (`most_unstable_root`): (1) scan |k² ε| on a grid in the complex u-plane (spacing 0.05; Re u covering both beams with a margin of 4 sqrt(2 s_max) + 1.5; Im u ∈ [−2.5, 3]); start Newton (analytic derivative, tolerance 1e-13) from every local minimum; deduplicate. (2) Certify: count the zeros in the rectangle from Im u* − 0.05 to a top edge on which |k² ε − k²| < k²/2 (the side edges are tested the same way), by the argument principle with adaptive sampling (phase steps < 0.5 rad, winding number within 1e-3 of an integer), and require the count to equal the number of roots found there. (3) On a mismatch, rescan on a 3× finer grid; a second mismatch, a Newton failure, a non-finite value or an input outside its limits (k > 0, s > 0, 0 < beta < 1) raises `RootError`. No uncertified number is returned.

**Failure rates.** In the shipped run, 12 776 samples: one (in `double_beam`) needed the finer rescan, none failed. In a scratch stress test on much wider boxes (1500 Bi-Maxwellian samples with k ∈ [0.1, 1.5], mu1 ∈ [−1, 1], mu2 ∈ [1, 8], s1, s2 ∈ [0.05, 2], beta ∈ [0.01, 0.99], gamma from −3.30 to 0.34; 1500 Two-Stream samples with k ∈ [0.1, 1.5], mu ∈ [−1, 1], s ∈ [0.1, 3]) every sample was certified on the first scan; 13–42 ms a sample (median), 0.14 s at most.

**Gradients.** Implicit function theorem at the root: ∂u*/∂p_j = −∂_{p_j}(k² ε)/∂_u(k² ε), ∂gamma/∂p_j = k Im(∂u*/∂p_j), plus Im u* for p_j = k. All partial derivatives are analytic (Z′ = −2(1 + zZ)). A vanishing ∂_u(k² ε) (a double root) raises `RootError`. Gamma is not differentiable where two roots exchange the lead; samples near such points have a small `gap` (global study: 2 of 1000 below 1e-3; per-percentage counts are printed in the notebook, up to 57 of 512 in Bump-on-Tail σ² = 0.25 at 25 %).

**Inputs** (all dimensionless; independent uniform; ranges by the paper's Eq. (10)):

| name | symbol | Double Beam baseline | Bump-on-Tail baseline | global box | source |
|---|---|---|---|---|---|
| k | k (perturbation wavenumber) | 0.5 | 0.5 | [0.4, 0.6] | Table II, p. 10; p. 29 |
| mu1 | μ1 (mean velocity, beam 1) | 0 | 0 | [−0.1, 0.1] | Table II; p. 29 |
| mu2 | μ2 (mean velocity, beam 2) | 4 | 4 | [3.5, 4.5] | Table II; p. 29 |
| s1 | σ1² (thermal velocity, beam 1) | 0.5 | 0.5 | [0.25, 0.75] | Table II; p. 29 |
| s2 | σ2² (thermal velocity, beam 2) | 0.5 | 0.5 | [0.25, 0.75] | Table II; p. 29 |
| beta | β (density fraction of beam 1) | 0.5 | 0.8 | [0.5, 0.99] | Table II; p. 29 |

Two-Stream (Table I, p. 9): k = 0.5, mu = 0, s = σ² = 1. Percentage studies: lower/upper = baseline × (1 ∓ pct/100); for a zero baseline, ∓ pct/100 in absolute value (p. 11). The paper's p. 29 writes "p3 = 2(mu2 − 8)", which contradicts its own bounds; we use the bounds (p3 = 2(mu2 − 4), as the paper's p. 31 also writes).

**Sampling design.** Paper: N = 512 uniform samples per study, no seed. Here: 512 per percentage study and 1000 for the global box, drawn with `numpy.random.default_rng([2024, study index, pct])`.

**Published results** (preprint): Fig. 1 (Two-Stream peak (k, gamma) = (0.4241, 0.2649)); Figs. 2–5, Tables III–IV (Two-Stream); Figs. 8–13, Tables V–VI (Double Beam); Figs. 14–17, Tables VII–VIII (Bump-on-Tail); Figs. 18–19, pp. 29–31 (global box: w1, w2, eta1 = 0.7509, eta2 = 0.9358, quadratic h(y1, y2)).

## Regenerating

```bash
python generate_data.py                                    # all five files, 2 processes, writes here
python generate_data.py --rows 40 --out /tmp/vp --check    # first 40 rows of each study, compared with the files here
python generate_data.py --studies global --out /tmp/vp --check
```

The 2026-10-30 run (numpy 2.5.4, scipy 1.18.1, Python 3.13.18, Linux container, 2 processes) took 143 s (243 s of CPU, about 19 ms a sample). A rerun with `--rows 40 --check` reproduced every number bit for bit (maximum relative difference 0). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-10-31) the same check gives maximum relative differences from 2.4e-14 to 1.7e-11, depending on the study. The samples are independent; `--workers` changes only the run time.

## Accuracy

- **Gradients vs finite differences** (scratch script, not committed): central differences of the model in normalized coordinates (step 1e-5) at 60 random rows of each file agree with the stored implicit-function gradients to a relative error (2-norm of the gradient vector) with medians from 8.5e-10 to 9.3e-9 and maxima of 6e-9 (global), 4e-8 (Double Beam), 6e-8 (Bump-on-Tail), 1.1e-7 (Bump-on-Tail σ² = 0.25) and 3e-8 (Two-Stream). The notebook repeats this at 5 rows (3.4e-9).
- **Outputs**: Newton converges to a step of 1e-13 |u| with a residual |k² ε| < 1e-9; recomputing gamma at those rows reproduces the files exactly on the platform that wrote them. In the repo's `.venv` the notebook's 5 re-solved rows still match exactly, but other rows differ by up to a relative 1.7e-11 (see `--check` above).
- **Z function**: checked against quadrature of the dispersion integral (above).
- **Validation against the paper**: the Two-Stream maximum growth rate is 0.2649 (paper Fig. 1: 0.2649), at k = 0.4280 rather than the paper's 0.4241; the curve is flat at its peak, so the k of the maximum is sensitive to the grid the paper used. Baseline growth rates: Two-Stream 0.25925 and Double Beam 0.18486, against the intercepts 0.2592 and 0.1848 of the paper's 1 % fits.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from MATLAB code (github.com/sterrab/GSA_PlasmaInstabilities; no licence file, so not copied or read here beyond the repository's front page) with forward differences of step 1e-6 in normalized coordinates (p. 11). These files are an independent Python computation of the same equations, ranges and sample sizes, with exact gradients and our own random samples. The repository posts `.mat` workspaces for the 1 % Two-Stream and 1 % Double Beam studies only (per the paper card); they were not used.
- **Reproduced**: the global box (eta1 0.7461 / 0.7509, eta2 0.9283 / 0.9358, w1 within 0.013 per entry, w2 within 0.047, 3.3° between the 2-D subspaces), Two-Stream (Tables III–IV) and Double Beam (Tables V–VI); see the notebook's comparison table.
- **Bump-on-Tail differs from Table II as stated.** With sigma1² = sigma2² = 0.5 the 1 % weights differ from Table VII by up to 0.245 and the baseline gamma is 0.1266 (paper 0.1210). Table VII also gives mu1 a weight of exactly 0 (1.1e-16), which with the paper's own rule (± 0.01 for a zero baseline) is impossible, since gamma depends on mu2 − mu1. Setting sigma1² = sigma2² = 0.25 (as if 0.5 were σ) and holding mu1 at 0 (as if its range were 1 % of 0) reproduces the 1 % row to 1e-3, the baseline 0.1210, and the 5–15 % weights to 0.0063. Both versions are shipped: `bump_on_tail.csv` (as stated) and `bump_on_tail_s025.csv` (as implied). The Double Beam study, with the same s and mu1 rules as stated, matches the paper, so the discrepancy is specific to the Bump-on-Tail runs.
- **Stable samples.** Where the equilibrium is stable, gamma here is the least-damped root. From 10 % on, the paper's Bump-on-Tail fits (Table VIII) are concave and fall to about −0.2 at large y, while ours flatten just below 0 (the least-damped roots there are weakly damped waves at omega_r ≈ ±1.1; the next ones, at omega_r ≈ 1.6–1.7 and 3.0, have gamma ≈ −0.13 to −0.17). Continuing the unstable root from the baseline gives the same root as ours at all 512 samples at 10 % (scratch check), so the paper's values there are another damped root; the paper does not say how its solver picks one. In the Double Beam and global studies, the paper's fits agree with ours in the stable region too.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). Model and analysis: S. Terrab and S. Pankavich. Python model, data and notebook: Claude (Anthropic) for P. Constantine, 2026, reimplemented from the paper's equations. Cite S. Terrab and S. Pankavich, Commun. Nonlinear Sci. Numer. Simul. (2024) 107994, doi:10.1016/j.cnsns.2024.107994. The paper's work was supported by NSF grants DMS-1551229 and DMS-1614586.
