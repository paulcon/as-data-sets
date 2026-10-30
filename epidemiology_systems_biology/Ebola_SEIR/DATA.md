# Ebola_SEIR: data

Kind: **Python model**. The model is `ebola_model.py`: the basic reproduction number R0 of the modified SEIR model and its gradient, both closed form. There are no data files, and every value the notebook analyzes is computed by the notebook from that module.

## Files

| file | contents | role |
|---|---|---|
| `Ebola_SEIR.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2018; outputs committed |
| `ebola_model.py` | `R0(x)`, `R0_grad(x)`, `NAMES`, `LABELS`, `COUNTRIES`, `FITTED`, `BOUNDS` | the model and the parameter tables |

## Model

R0 = (β1 + β2ρ1γ1/ω + β3ψ/γ2) / (γ1 + ψ), Eq. (4) of Diaz et al. (2018), from the SEIR system of their Eqs. (1)–(2) by the next-generation matrix.

- **Inputs**: m = 8, in the order β1, β2, β3, ρ1, γ1, γ2, ω, ψ (`NAMES`). All are rates in day⁻¹ except ρ1, which is a fraction. The incubation rate δ = 1/9 and the hospital burial fraction ρ2 do not enter R0.
- **Input arrays**: `R0` and `R0_grad` take physical parameters with shape (..., 8). `R0_grad` returns the gradient with respect to the physical parameters, with shape (..., 8). The notebook maps it to the normalized coordinates [-1, 1]^8 by multiplying by (upper − lower)/2 (`space.scale`).
- **Fitted values** (`FITTED`): Table 2 of the paper, for Liberia and Sierra Leone (Guinea is not in the active-subspace analysis). At these values R0 = 1.5634 (Liberia) and 1.4433 (Sierra Leone). The paper's Table 3 gives 1.563 and 1.445. The Sierra Leone difference is rounding in Table 2: with the less-rounded values printed in the paper's Fig. 2c, R0 = 1.4451.
- **Ranges** (`BOUNDS`): independent uniform distributions from Table 7 of the paper. The fitted Sierra Leone γ2 = 0.0833 (Table 2) lies outside its Table 7 range [0.1236, 0.384]. The paper has the same disagreement, and it is not known which value is wrong (author, 2026-10-30). The notebook uses the range as published.
- **Gradient check**: the notebook compares `R0_grad` with central finite differences at one random point per country (maximum relative error 2.4e-9 and 2.2e-10).

## Quadrature

C = ∫ ∇R0 ∇R0ᵀ ρ dx is computed with the paper's tensor Gauss–Legendre rule, 8 points per dimension (8^8 = 16,777,216 points). The weights are the scipy `roots_legendre(8)` weights halved, so their tensor product sums to 1 and already includes the density ρ = 2⁻⁸. The quadrature has no randomness.

The notebook accumulates the 8 × 8 matrix C = Σ ω_i ∇R0(x_i) ∇R0(x_i)ᵀ (normalized coordinates) in 64 blocks of 8^6 points, and passes it to `ac.subspace_from_matrix` (author, 2026-10-30). The full gradient array (1.07 GB) is never held. The notebook first checks this path on the 5^8 rule against `ac.active_subspace(grads=..., quad_weights=...)` with the full gradient array. The eigenvalues agree to a relative 1.7e-13 and w1, w2 to 1.8e-15 (printed).

The 2016 notebook multiplied the integrand by 2⁻⁸ as well, so its eigenvalue plots were 256 times too small (λ1 ≈ 3.99e-3 for Liberia instead of 1.0217). The notebook prints both values. The eigenvectors and the normalized activity scores do not depend on that factor.

The rule has converged. λ1 for each rule size (the 8^8 values are printed by the notebook; the others are from a scratch run of the same code):

| points per dimension | Liberia | Sierra Leone |
|---|---|---|
| 6 | 1.0194 | 1.1014 |
| 7 | 1.0213 | 1.1042 |
| 8 | 1.0217 | 1.1049 |

**Memory**: the peak resident memory of the notebook is about 0.43 GB on macOS (433 MB in the committed run, printed). Passing the full 8^8 gradient array to `ac.active_subspace` instead would peak at about 3 GB: the array is 1.07 GB, the function keeps a normalized copy, and it hashes the bytes for its provenance record, which makes a third copy.

## Regenerating

There are no shipped outputs and no `generate_data.py`. The notebook computes everything in about 20 seconds (17 s in the committed run; one Python process, about 7 s per country for the quadrature):

- R0 at the fitted values, and the gradient check (one random point per country);
- the 5^8 check of the blocked matrix, and the active subspace of each country from the 8^8 rule;
- 2000 uniform random points per country for the sufficient summary plots (the same normalized points mapped into each country's box).

All draws come from one `np.random.default_rng(2018)`. The seed affects only the gradient-check points and the summary-plot points, not the eigenvalues, eigenvectors or activity scores. Two runs give identical printed output.

## Relation to the paper's runs

The paper computed the same 8^8 rule with its own code (https://github.com/PaulMDiaz/Ebola, not used here). The notebook's w1 matches the paper's Fig. 8, with the same signs: the 2016 sign rule (first component positive) is restored with `ac.orient`. The normalized activity scores for ψ (0.522 Liberia, 0.689 Sierra Leone) match Fig. 9. The paper's Fig. 7 eigenvalues are about 4 times the notebook's, with the same spacing (specs/Ebola.md §10). That factor is what normalizing to [0, 1] instead of [-1, 1] would give, but this is not confirmed. Fig. 7 shows λ1/λ2 ≈ 4 (Liberia) and 7 (Sierra Leone), and the notebook prints 3.9 and 7.1. The paper's text on p. 151 ("nearly two orders of magnitude" between λ1 and λ2) disagrees with its own figure, and the notebook does not repeat it.

## License and credit

Code: MIT (`LICENSE`). There are no data files; the parameter values in `ebola_model.py` are from the paper (Tables 2 and 7). Cite P. Diaz, P. Constantine, K. Kalmbach, E. Jones and S. Pankavich, "A modified SEIR model for the spread of Ebola in Western Africa and metrics for resource allocation", Appl. Math. Comput. 324 (2018) 141–155, doi:10.1016/j.amc.2017.11.039.
