# Shared_Subspace_H2_Ignition_Flame: data

Kind: **stored data (generated with Cantera 3.2.0), with a regeneration script.** The model runs in Cantera, an
outside code; `h2_konnov_driver.py` drives it, `generate_data.py` regenerates the files where Cantera is installed
and otherwise stops with an error that says what to install, and the shipped CSVs are the record. The notebook
needs only numpy, scipy, matplotlib and active_subspaces: it reads the CSVs and the driver's constants (which
import without Cantera).

The Cantera mechanism file is **not shipped**: it is a translation of the publisher's supplementary files of
Konnov (2008), which may not be redistributed. `build_mechanism.py` rebuilds it (`konnov2008_h2.yaml`) from those
files; see Regenerating.

## Files

| file | contents | role |
|---|---|---|
| `Shared_Subspace_H2_Ignition_Flame.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2018; outputs committed |
| `h2_konnov_driver.py` | `H2Model` (Cantera driver: `ignition_delay`, `idt_grad`, `flame_speed`, `sl_grad`, `sl_adjoint`); constants `REACTIONS`, `UF`, `SIGMA`, `IDT_CONDITIONS`, `SL_P`; numpy-only `normalized_gradients`, `shared_directions` | the model and the shared-direction rule; constants import without Cantera |
| `build_mechanism.py` | checks the SHA-256 of Konnov's three supplementary files, removes the non-standard `BASE M=N2` token, runs Cantera's `ck2yaml` | rebuilds `konnov2008_h2.yaml` (not shipped) |
| `generate_data.py` | writes all CSVs (`--out DIR` required, `--case`, `--rows`, `--check`, `--max-seconds`; resumable) | regenerates the data (needs Cantera 3.2.0 and the mechanism) |
| `check_accuracy.py` | prints every number of "Accuracy" below (`--quick`: one row; `--adjoint`: the built-in adjoint chain-rule check only) | accuracy check (needs Cantera 3.2.0 and the mechanism) |
| `inputs.csv` | 350 × 33: ln(k_j/k_j0), j = 1..33 in mechanism order (dimensionless) | gradient-set inputs, seed 2018033 |
| `outputs.csv` | 350 × 21: τ (s) at the 16 ignition conditions, then S_L (m/s) at 5 pressures | model output |
| `gradients.csv` | 350 × 693: ∂f_i/∂ln k_j; output i major, reaction j minor; units of the output (s, m/s) | model output |
| `mc_inputs.csv`, `mc_outputs.csv` | 1000 × 33 and 1000 × 21, as above, no gradients | Monte Carlo set, seed 2018100 |
| `directions.csv` | 22 × 33: first eigenvectors u_1..u_21 (normalized inputs x, x16 component positive), then the shared direction v | computed from `gradients.csv` by `shared_directions` |
| `line_runs.csv` | 49 × 43: ξ, the 21 outputs at x = ξ v, the 21 outputs at x = ξ u_i, ξ = −4, −3.833, …, 4 | model output (Approximations III and II) |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the written digits before the model runs, so each output
belongs to the stored input row. The largest file is `gradients.csv` (4.3 MB).

Output order (columns of `outputs.csv`, `mc_outputs.csv`): τ at (1.59 bar = 1.5692 atm, 1000 K), then
(P, T0) for P = 1, 2, 5, 10, 20 atm and T0 = 1000, 1400, 1800 K (T0 fastest); then S_L at 1, 2, 5, 10, 20 atm.

## Model

### Citation

W. Ji, J. Wang, O. Zahm, Y. M. Marzouk, B. Yang, Z. Ren, C. K. Law, "Shared low-dimensional subspaces for
propagating kinetic uncertainty to multiple outputs", *Combustion and Flame* 190 (2018) 146–157,
doi:10.1016/j.combustflame.2017.11.021. The PDF read is the publisher's version. Only the hydrogen case is built
here (the paper also treats methane and DME).

Mechanism and uncertainty factors: A. A. Konnov, "Remaining uncertainties in the kinetic mechanism of hydrogen
combustion", *Combustion and Flame* 152 (2008) 507–528, doi:10.1016/j.combustflame.2007.10.024 (paper's ref. [29];
publisher's version read). The citations were checked against the articles' own title pages, not on Crossref.

### Physics and solver

- **Mechanism.** Konnov (2008): 10 species (H, H2, O, O2, H2O, OH, H2O2, HO2, Ar, N2), 33 reactions in the
  supplementary mechanism file (paper Table 1, p. 150: "10 species, 33 reactions"), with Konnov's thermodynamic
  and transport files. The mechanism file's `REACTIONS BASE M=N2` line is not standard CHEMKIN; the token is
  dropped, so unlisted third bodies have efficiency 1, which is what Konnov's Table 1 footnote a states. Nothing
  else is changed.
- **Mixture.** Stoichiometric H2/air, H2 : O2 : N2 = 2 : 1 : 3.76 by moles (paper: "stoichiometric mixture of
  hydrogen/air", p. 151; the air composition is this data set's choice).
- **Ignition delay** (paper Table 2 and Sec. 3.2, pp. 151–152). Adiabatic constant-volume reactor (Cantera
  `IdealGasReactor`, CVODES BDF, rtol 1e-10, atol 1e-20). τ is the first time T reaches T0 + 400 K, located by
  cubic Hermite interpolation of T between the bracketing solver steps with the exact dT/dt from the energy
  equation. The paper gives no definition for these runs (for its HCCI runs it uses the maximum heat release rate,
  p. 153); the T0 + 400 K criterion is this data set's choice. Conditions: 1.59 bar and 1000 K (Fig. 1, p. 151),
  and P = 1, 2, 5, 10, 20 atm × T0 = 1000, 1400, 1800 K (Sec. 3.2, p. 152, stated there for methane; Table 2 gives
  the same ranges for H2). The text of p. 151 says 1–100 atm, Table 2 1–20 atm; the Sec. 3.2 grid stops at 20 atm.
  One ignition takes about 10 ms.
- **Flame speed.** Freely propagating adiabatic premixed flame (`FreeFlame`), T_u = 300 K, P = 1, 2, 5, 10, 20 atm,
  mixture-averaged transport, domain width 0.03 m / P[atm]^0.7, refine criteria ratio 2.5, slope 0.04, curve 0.08,
  prune 0.01 (185–201 grid points at the nominal point), Newton tolerances rtol 1e-9, atol 1e-14 (steady and
  transient). Each sample's solve starts from the nominal solution at that pressure (so results do not depend on
  the order of the samples). Cantera's `from_array` does not restore the flame's pressure; the driver sets it.

### Inputs

| input | meaning | distribution | source |
|---|---|---|---|
| ln(k_j/k_j0), j = 1..33 | log rate multiplier of reaction j (mechanism order) | normal, sd σ_j = ln(UF_j)/3, truncated to ±3σ_j (k in [k_j0/UF_j, k_j0·UF_j]) | paper Eq. (1), p. 148; UF_j from Konnov Table 1, pp. 509–510 |

Normalized inputs x_j = ln(k_j/k_j0)/σ_j (paper Eq. 2) are truncated standard normals on [−3, 3]. The multiplier
scales the whole rate (Cantera's rate multiplier: falloff and three-body rates included; reverse rates follow).

Uncertainty factors (mechanism index: Konnov No., UF): 1: 1a, 2; 2: 1b, 2.5; 3: 1c, 3.2; 4: 1d, 3.2; 5: 2, 2;
6: 3, 3; 7: 4a, 2; 8: 4b, 2; 9: 5a, 1.2; 10: 5b, 1.2; 11: 5c, 1.2; 12: 5d, 1.4; 13: 6a, 2.5; 14: 6b, 2.5; 15: 7, 1.3;
16: 8, 1.5; 17: 9, 2; 18: 10, 1.5; 19: 11, 1.2; 20: 12, 2; 21: 13, 3; 22: 14, 2; 23: 15, 3; 24, 25: 16 (two
expressions), 3 and 3; 26, 27: 17a (two expressions), 2.5 and 1.4; 28: 17b, 1.4; 29: 18, 3; 30: 19, 2; 31: 20, 3;
32, 33: 21 (two expressions; one UF printed), 2 and 2. Choices (this data set's, not stated in either paper): each
duplicate expression is a separate input with the UF printed on its line; where Konnov prints a second factor for
a low-pressure limit (5c: 1.3) or for collision efficiencies (1a: 5; 2: 2 and 3), the first, rate-constant factor
is used. Indices follow the mechanism file, which is the paper's Fig. 1c numbering (R16 = H + O2 = OH + O,
R9 = H + O2 (+M) = HO2 (+M)). Input x10 (H + O2 (+Ar)) has no effect: the mixture contains no argon.

### Quantities of interest and gradients

`outputs.csv` stores τ (s) and S_L (m/s). The notebook analyses log10 τ (the paper reports IDT "in log scale",
p. 151) and S_L in m/s. `gradients.csv` stores ∂f/∂ln k_j by forward differences k_j → (1 + δ) k_j, δ = 1e-3,
divided by ln(1 + δ) (paper Eq. 20), for both outputs; ∂f/∂x_j = σ_j ∂f/∂ln k_j (Eq. 19). For S_L each perturbed
solve starts from the sample's converged solution on its grid (no regridding), so the difference is that of one
discrete model. The paper used Cantera's adjoint for S_L (p. 150); here the finite differences are stored, and the adjoint is
checked against them in Accuracy. 34 runs
per output per sample.

### Sampling design

Plain Monte Carlo from the truncated distribution (independent standard normals from numpy's `default_rng`,
entries outside [−3, 3] redrawn): 350 samples with gradients (seed 2018033) and 1000 without (seed 2018100). 350 is
the paper's M = αβ log d with α = β = 10 and d = 33 (Algorithm 2.1, p. 148; α, β p. 151): 100 ln 33 = 349.7. The
paper reports no seed and used 100 bootstrap replicates; the notebook uses 200. `line_runs.csv`: 49 equally
spaced ξ in [−4, 4] along v and along each u_i (2058 runs), for the paper's Approximations II and III (p. 151) and
Algorithm 2.2 (p. 150). At ξ = ±4 along v the largest |x_j| is 4.6, beyond the ±3 truncation (rate constants still
positive and finite): these runs lie outside the input distribution by construction.

### Shared direction

`shared_directions` (numpy only) computes u_i, the first eigenvector of mean(∇f_i ∇f_i^T) in x for each output,
sign chosen so the x16 component is positive (as in the paper's Fig. 1c), and from A = [u_2..u_21] (the 20 outputs
of the paper's Sec. 3.2 grid; the Fig. 1 condition is held out) both the paper's minimum-norm solution
v = A(AᵀA)⁻¹1 (Eq. 10) and a truncated-SVD solution of Aᵀv = 1 that keeps the fewest singular vectors with
max_i |u_iᵀv − 1| ≤ 0.05. The truncation rule is this data set's choice: Eq. (10) is badly conditioned for these
outputs (condition number 5.6e4, ‖v‖ = 13.8; notebook). `directions.csv` and `line_runs.csv` use the
truncated-SVD v (8 of 20 singular vectors, ‖v‖ = 2.505).

### Published results (paper, hydrogen)

- Fig. 1 (p. 151), IDT at 1.59 bar and 1000 K: eigenvalues with bootstrap bounds, λ1 "larger than the second one by
  two orders of magnitude" (λ1 about 1.5e-2 and λ2 about 1.5e-4 read from Fig. 1a); summary plot of log10 IDT against w1ᵀx (about −4.1 to
  −3.0); w1 led by R16 (H + O2 = OH + O, about +0.92) and R9 (H + O2 (+M) = HO2 (+M), about −0.38; values read from
  Fig. 1c).
- The shared subspace (Figs. 2–4, Table 4–5) is shown for methane, DME and natural gas only. No eigenvalues are
  tabulated.

### What the notebook finds (from these files)

- Every output has a one-dimensional active subspace: λ1/λ2 from 51.1 (IDT, 5 atm, 1000 K) to 1124.1 (IDT, 1 atm,
  1000 K); flame speeds 81.3–97.2. Quadratic surrogates in y1: leave-one-out R² 0.9712–0.9990.
- Fig. 1 condition: λ1 = 1.340e-2, λ1/λ2 = 586; w1: x16 +0.962, x9 −0.251, x20 +0.085; log10 τ from −4.095 to −3.466.
- Shared direction: the 20 training eigenvectors have cosines 0.095–0.999; held-out u_1ᵀv = 0.9948. Approximation
  III (on v) has relative RMS error 0.035–0.154, Approximation II (on u_i) 0.035–0.165. The ignition-delay means
  are reproduced within 0.0044 in log10 s; the flame-speed means come out 0.017–0.024 m/s high (III), outside the
  Monte Carlo 95 % interval, with either direction. This is the second-order gap of evaluating on the line:
  ½ tr(H(I − u uᵀ)), with H fitted by least squares to the stored gradients, predicts Monte Carlo means 0.0201,
  0.0239, 0.0270, 0.0270 and 0.0235 m/s below Approximation II at 1, 2, 5, 10, 20 atm (observed 0.0189, 0.0226,
  0.0259, 0.0272, 0.0243 m/s); the paper's Approximation I (conditional mean) would remove it.

## Regenerating

```bash
python3 -m venv cantvenv && cantvenv/bin/pip install cantera==3.2.0      # numpy comes with it
# the three supplementary files of Konnov (2008) (publisher's "Supplementary data",
# doi:10.1016/j.combustflame.2007.10.024) in KONNOV_DIR:
#   1-s2.0-S0010218007003318-mmc1.txt  sha256 3bcd9bf3d5c34a717c31e90bb779210088df37138fc351e23eb815bd860232dd
#   1-s2.0-S0010218007003318-mmc2.txt  sha256 2e8f449b9e90067a94674ecec22a8bf6284a6405efbbdb3ed6ae6256b28b6920
#   1-s2.0-S0010218007003318-mmc3.txt  sha256 91e0b98c2129de80fd92cef17b2234d1a14d058d2c601423716d27efd7ff05cb
cantvenv/bin/python build_mechanism.py KONNOV_DIR          # writes konnov2008_h2.yaml here (do not commit it)
cantvenv/bin/python generate_data.py --out /tmp/h2gen      # all cases; resumable; copy the CSVs in afterwards
cantvenv/bin/python generate_data.py --case grad --rows 0:2 --out /tmp/h2chk --check    # subset vs shipped
cantvenv/bin/python check_accuracy.py                       # the Accuracy numbers (about 4 min)
```

Tolerance of `--check`: 1e-6 of each column's largest value for inputs and outputs, 1e-3 for gradients.

Provenance of the shipped files (an external run, not reproducible from the repository): generated 2026-11-09/10
in a Linux container (x86-64, 2 cores shared with other jobs), Python 3.13.18, Cantera 3.2.0, numpy 2.5.4, with
`generate_data.py` as shipped, one process, in chunks of 9 minutes (the work was interrupted once by a container
restart and resumed from the saved rows). CPU time: 3992 s for the 350 gradient samples (about 11 s each: 16 × 34
ignitions and 5 × 34 flame solves), 622 s for the 1000 Monte Carlo samples (per-row CPU times from the run log), and the line runs (not logged
separately; 2058 of about 273 000 runs): about 78 min in all. `--case grad --rows 0:2 --check` and `--case mc --rows 0:5 --check` on the same machine
reproduced the stored rows with no difference at the written precision. The line case was spot-checked
(external run, not reproducible from the repository): four entries of `line_runs.csv` (ξ = −3.167, +2.667, −2.000,
+1.000; ignition and flame outputs, along v and u_i) recomputed with `H2Model` agree to 2e-11 relative or better.

## Accuracy

`check_accuracy.py` prints every number in this list (Cantera 3.2.0, Linux x86-64, 2026-11-10; about 4 min).
Gradient errors are relative 2-norm errors of the gradient vector in x.

- **τ, solver tolerance.** Re-solving at rtol 1e-13, atol 1e-24 changes τ by at most 3.4e-9 relative over the 16
  conditions at rows 0, 117, 233, 349.
- **τ gradients.** Shipped forward differences against central differences (same δ): median 4.0e-4 to 4.3e-4,
  worst 1.5e-3 to 3.4e-3 (at 2 atm and 1000 K, near the explosion limit), at the same four rows.
- **S_L, discretization.** At the nominal point, a finer grid (ratio 2, slope 0.02, curve 0.04; about 365 points)
  with Newton tolerances 1e-11/1e-16 changes S_L by −2.1e-3, −1.7e-4, +1.1e-3, +3.7e-3 and +1.2e-2 relative at 1, 2,
  5, 10 and 20 atm. This is a bias of the flame-speed level (largest at 20 atm), shared by all samples; it is not
  removed.
- **S_L gradients.** Shipped forward differences against central differences: 1.4e-4 to 1.8e-4 (rows 0, 175, 349,
  all pressures).
- **Cantera's flame-speed adjoint (not used for the stored gradients).** On 13 rows (0, 29, …, 348), against the
  shipped finite differences, the raw output of `FreeFlame.get_flame_speed_reaction_sensitivities()` differs by a
  median of 1.6e-1 and up to 3.2e-1 at every pressure. This is not an error of the routine but a different
  derivative: its perturbation sets multiplier j to 1 + dp, and the residual is linear in each multiplier, so it
  returns ∂ln S_L/∂m_j, the derivative with respect to the rate multiplier m_j = k_j/k_j0. Multiplied by the sample's
  m_j = exp(σ_j x_j) (chain rule) it gives ∂ln S_L/∂ln k_j, the derivative stored here. Section 4c of
  `check_accuracy.py` (rows 328, 218, 239, 1 atm): raw built-in vs finite differences 1.56e-1, 2.22e-1, 2.96e-1;
  built-in × m_j vs the adjoint that perturbs m_j as m_j(1 + dp) (`H2Model.sl_adjoint`) 3.4e-11, 3.9e-11, 4.2e-11,
  and vs the finite differences 1.6e-3, 1.5e-3, 1.8e-3. At x = 0, where every m_j = 1, the two adjoint forms are the
  same (printed difference 0.0e+00). On the 13 rows, `H2Model.sl_adjoint` agrees with the finite differences to a
  median of 8.2e-4 to 4.6e-3 at 1–10 atm and 1.5e-2 (max 2.1e-2) at 20 atm. Forward and central differences agree
  to about 1.5e-4 (above), so the finite differences are the stored gradients.
- In the notebook, the least-squares slope of each output on x and its mean normalized gradient agree in
  direction (cosines 0.9904–0.9999).

## How these files relate to the paper

- **Not the paper's computation.** The paper published no data or code. These files come from an independent
  Cantera computation with the paper's mechanism, uncertainty factors, input distribution, gradient formula,
  sample size and conditions, but with this data set's choices for the ignition criterion, the air composition,
  the flame solver settings, finite-difference flame-speed gradients, and how Konnov's factors map to the 33
  reactions.
- **Agreement.** Fig. 1 is reproduced qualitatively: one dominant direction led by R16 with an opposite R9
  contribution and a smooth summary plot. Differences: λ1/λ2 = 586 against "two orders of magnitude"; the R9
  weight is −0.25 against about −0.38; the paper's Fig. 1b reaches log10 τ ≈ −3.0, ours −3.47; the paper's
  eigenvalues from λ4 on sit near 1e-5 (Fig. 1a), ours below 1e-6.
- **Extension.** The shared-subspace study (Sec. 3.2) is done for hydrogen with the paper's methane grid; the paper
  has no hydrogen result to compare. The truncated-SVD v is this data set's regularization of Eq. (10).

## Open questions (for the authors)

1. The ignition-delay definition and the air composition used for the H2 runs of Fig. 1.
2. How the Konnov uncertainty factors were assigned to duplicate expressions (16, 17a, 21) and to reactions with a
   second printed factor (5c low-pressure limit; collision efficiencies of 1a and 2).
3. Whether the shared direction v for methane (Fig. 2b) came from Eq. (10) exactly or from a regularized solve,
   and how well conditioned A was.
4. The pressure range of Sec. 3.1: 1–100 atm (text, p. 151) or 1–20 atm (Table 2).
5. Whether the S_L adjoint gradients (p. 150) came from Cantera's `get_flame_speed_reaction_sensitivities()` at
   non-unit multipliers. That routine returns ∂ln S_L/∂m_j; without the factor m_j it is not ∂ln S_L/∂ln k_j, which
   would bias C for S_L (Accuracy).
6. The origin of the eigenvalue floor near 1e-5 from λ4 on in Fig. 1a (ours are below 1e-6), and the two samples
   near log10 τ = −3.0 in Fig. 1b (our largest of 350 is −3.466).

## License and credit

Code (`*.py`, the notebook): MIT. Data (the CSVs): CC BY 4.0. The uncertainty factors in `h2_konnov_driver.py` are
facts transcribed from Konnov (2008), Table 1; cite that paper. The Konnov mechanism, thermodynamic and transport
files are not included; `build_mechanism.py` rebuilds the Cantera file from them. Study design: W. Ji, J. Wang,
O. Zahm, Y. M. Marzouk, B. Yang, Z. Ren, C. K. Law; cite their paper (above). Cantera: D. G. Goodwin,
H. K. Moffat, I. Schoegl, R. L. Speth, B. W. Weber, https://www.cantera.org (BSD-3-Clause; not copied here).
