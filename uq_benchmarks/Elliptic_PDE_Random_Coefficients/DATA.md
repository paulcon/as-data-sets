# Elliptic_PDE_Random_Coefficients: data

Kind: **Python model**. The model is `pde_model.py`, and every value the notebook analyzes is computed by the notebook from it. The KL bases `kl-long.npz` and `kl-short.npz` are inputs that cannot be regenerated, so they are treated as stored data.

## Files

| file | contents | role |
|---|---|---|
| `Elliptic_PDE_Random_Coefficients.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2014; outputs committed |
| `pde_model.py` | `PDESolver(corrcase)`: `qoi`, `grad_qoi`, `qoi_and_grad`, `solve`, `residual` | the model; byte-identical to `../PDE_Inverse_Problem_MCMC/pde_model.py` |
| `kl-long.npz` | `K` (10000, 100) float64, `lam` (100,) float64 | KL basis for β = 1 (stored) |
| `kl-short.npz` | same | KL basis for β = .01 (stored) |

## Model

`-∇·(a ∇u) = 1` on [0, 1]², with u = 0 on the left, top and bottom sides and zero flux on the right side. The log-coefficient is `log a = B x`, with `B = K * sqrt(lam)`, `x ~ N(0, I_100)`, and row r of `B` the cell with row index r // 100 (s2) and column index r % 100 (s1).

- **Discretization**: five-point vertex-centred finite volumes on 100 × 100 cells. The face coefficients are averages of the two adjacent cells, and the unknowns are the 99 × 99 interior vertices. The Neumann condition copies the last column (first order). This is the 2016 numpad model's residual, solved with a sparse LU. The papers used a MATLAB P1 finite-element code instead, which is not in the repo.
- **Quantity of interest**: trapezoid rule for u on the vertex column s1 = 0.99 over s2 = 0.01, …, 0.99, with the end nodes at half weight (the 2016 weights). It has no units (the PDE is nondimensional).
- **Gradient**: a discrete adjoint, ∂f/∂x_i = −λᵀ(∂A/∂x_i)u with Aᵀλ = c. Inputs are standard normal, so the physical and normalized coordinates coincide.
- **Check**: `python uq_benchmarks/check_pde_model.py` (residual < 1e-10, adjoint vs finite differences < 1e-6, f(0) = 0.07608, copies identical). About 20 ms per value and gradient.

## KL bases (stored)

- `K` holds orthonormal eigenvectors (KᵀK = I to 1e-14) and `lam` the 100 largest eigenvalues of the discrete covariance matrix exp(−‖s − t‖₁/(2β)), on the 100 × 100 grid of cell centres ((i + ½)/100), with β = 1 (`kl-long`) and β = .01 (`kl-short`). The matrix has unit diagonal, so `lam.sum() / 10000` is the share of field variance the 100 modes carry: 99.2 % (long), 13.3 % (short).
- The kernel was identified from the data, not from a script. `lam` equals the top 100 eigenvalues of that matrix to a relative 3.4e-14 (long) and 8.7e-15 (short) (specs/Elliptic_PDE.md §10). The papers write the kernel with 1/β: Constantine, Dow & Wang 2014, eq. (5.3), which also lacks the minus sign; the book, eq. (3.80); and the MCMC paper, eq. (5.5) with β = 0.02. The shipped `kl-short` ratios λ_i/λ_1 do not match CDW Table 1, but the `kl-long` ratios do.
- Several eigenvalues come in equal pairs (e.g. λ2 = λ3), and within each pair the basis is an arbitrary rotation. The stored `K` therefore defines the coordinates x_i, and it must not be regenerated: a new eigensolver would rotate the pairs and change the meaning of every x_i, and of the MCMC chains, which use `kl-short`.
- The script that generated the files is lost (author, 2026-10-30). The zip timestamps are 2015-07-25.
- `kl-short.npz` is byte-identical to `../PDE_Inverse_Problem_MCMC/kl-short.npz` (md5 079370cfa2411de13b20ab62c7b22879). `kl-long.npz` has md5 a8279052c510fe995b0351fb348d0c98.

## Regenerating

There are no shipped outputs and no `generate_data.py`. The notebook computes everything from `pde_model.py` in about two minutes (one Python process):

- 300 values and gradients per field (seed 2014);
- the kriging designs (5 and 25 points);
- the full-space training sets of 3M + P = 905 and 925 points per field;
- 500 test points.

All draws come from one `np.random.default_rng(2014)`, and two runs give identical printed output.

## License and credit

Code: MIT (`LICENSE`). Data (`kl-*.npz`): CC BY 4.0 (`LICENSE-DATA`). Notebook by Ryan Howard and Paul Constantine. Cite P. G. Constantine, E. Dow and Q. Wang, "Active subspace methods in theory and practice: applications to kriging surfaces", SIAM J. Sci. Comput. 36(4):A1500–A1524, 2014, doi:10.1137/130916138.
