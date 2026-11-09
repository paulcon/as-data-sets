# H2_Flame_Kinetics_Bayesian_Calibration: data

Kind: **stored data (generated with Cantera 3.2.0), with a regeneration script.** The model runs in Cantera, an
outside code, so it is not a numpy/scipy "Python model" in the repository's sense (CLAUDE.md, "Two kinds of data
sets"), even though the driver is a Python module (`h2_flame_driver.py`). A numpy/scipy port (a stiff reactor with
Troe falloff and a 1-D premixed flame with mixture-averaged transport) would be a second, unchecked combustion
code. Cantera was installed in a separate environment (a Linux container), the data were generated there (the Su20
gradients on a 16-core cloud instance), and the provenance is recorded below. The shipped CSVs are the record. **Regenerating the data, or rerunning the accuracy
checks, needs Cantera 3.2.0** (`pip install cantera==3.2.0`); `generate_data.py` and `check_accuracy.py` stop with
an error that says what to install when it is missing. The repo's `.venv` has no Cantera. The notebook needs only
numpy, matplotlib and active_subspaces: it reads the shipped files and the driver's constants.

**Note on flame-speed gradients.** Cantera 3.2.0's adjoint flame-speed sensitivities are wrong for these flames:
they point in the right direction but have the wrong size, too small by a factor of about 2 to 8 for Su20 and off
by a few per cent for Su1, with an error that grows with the grid size ("Model", "Accuracy"). The shipped flame-speed
gradients are therefore central finite differences (`FlameSpeed.fd_grad`), not adjoint sensitivities. Anyone
computing flame-speed gradients with Cantera's adjoint should check them against finite differences.

## Files

| file | contents | role |
|---|---|---|
| `H2_Flame_Kinetics_Bayesian_Calibration.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2019; outputs committed |
| `h2_flame_driver.py` | `PARAMS`, `NAMES`, `NOMINAL`, `LOWER`, `UPPER`, `UNITS`, `PRIOR_MEAN`, `PRIOR_SD`, `sample_prior`, `IGNITION`, `FLAMES`; Cantera classes `IgnitionDelay` (`tau`, `grad`) and `FlameSpeed` (`solve`, `fd_grad`, `adjoint_grad`); `build_mechanism` | the model (Cantera driver, not a numpy/scipy model); its constants import without Cantera |
| `burke2012_x1x6.yaml` | Cantera mechanism: Burke et al. (2012), 11 species, 27 reactions, plus X1 and X6 (29 reactions) | model input (see License) |
| `LICENSE-PelePhysics.txt` | PelePhysics' BSD-3-Clause licence | covers the distribution the mechanism file was derived from |
| `generate_data.py` | writes all CSVs (`--out`, `--rows`, `--qoi`, `--check`, `--nominal`, `--procs`) | regenerates the data (needs Cantera 3.2.0) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" below (`--quick`: checks 1–3; `--checks`: the checks named) | accuracy check (needs Cantera 3.2.0) |
| `inputs.csv` | 500 × 31: the inputs of Table 1 in physical units (A in cm-mol-s units, efficiencies dimensionless), order of `NAMES` | Monte Carlo draws from the prior, seed 2019452 |
| `outputs.csv` | 500 × 6: τ (s), Su1 (m/s), Su20 (m/s), and a status flag for each (0 solved; 1 solved on the retry; 2 failed, value NaN) | model output |
| `gradients_tau.csv` | 500 × 31: ∂τ/∂x_j (s per unit of x_j) | central differences |
| `gradients_Su1.csv` | 500 × 31: ∂Su1/∂x_j (m/s per unit of x_j); NaN where the flame failed | central finite differences on the fixed grid of each flame |
| `gradients_Su20.csv` | 500 × 31: ∂Su20/∂x_j (m/s per unit of x_j); NaN where the flame failed | central finite differences on the fixed grid of each flame |
| `nominal.csv`, `nominal_gradients.csv` | the three QoIs (1 × 3) and their gradients (3 × 31) at the Table 1 values | model output |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the written digits (`%.15e`) before the model runs, so
each output belongs exactly to its stored input row.

## Model

### Citation

J. Bell, M. Day, J. Goodman, R. Grout, and M. Morzfeld, "A Bayesian approach to calibrating hydrogen flame
kinetics using many experiments and parameters", *Combustion and Flame* 205 (2019) 305–315,
doi:10.1016/j.combustflame.2019.04.023 (Crossref-checked). The PDF read is the arXiv preprint (arXiv:1803.03500v1,
9 March 2018); page numbers below refer to it. Published numbers, if they differ, are those of the journal version.

### Mechanism

- Burke, Chaos, Ju, Dryer and Klippenstein, Int. J. Chem. Kinet. 44 (2012) 444–474, doi:10.1002/kin.20603
  (the paper's ref. [16]), "version 6-10-2011", with the N2-bath block of R9 (the block the file enables by
  default; its values are those of Table 1). Source: PelePhysics, github.com/AMReX-Combustion/PelePhysics, commit
  5e697a009eedfd9762f5dcfac60c1817d1b9971d (2026-11-01), `Mechanisms/BurkeDryer/mechanism.yaml` (15 179 bytes,
  sha256 eec9650a87e5940b091d9cb6007a47b9442182f767dcd056c725ac75b5dcd658), a ck2yaml conversion of the
  Princeton Chemkin files.
- `h2_flame_driver.build_mechanism` turns it into `burke2012_x1x6.yaml` (14 743 bytes, sha256
  f2cf079e520d015b3d3feded52a91fa1c2a0e09c37618db3a38bf38fe5a89060) by text edits only: it removes the species CO
  and CO2 and their collision efficiencies (no reaction makes or uses them, and the mixtures contain none, so this
  changes no rate), and appends X1, HO2 + H ⇌ H2O + O (A = 3.97e12, β = 0, Ea = 671 cal/mol), and X6,
  O + OH + M ⇌ HO2 + M (8.0e15, 0, 0; efficiencies H2 2, H2O 12, Ar 0.7, He 0.7), with the values of the paper's
  Table 1 (taken there from You, Packard and Frenklach, Int. J. Chem. Kinet. 44 (2012) 101–116,
  doi:10.1002/kin.20627, its ref. [20]). Rerunning `build_mechanism` on the source gives an identical file (same
  sha256).
- **Checked against Table 1** (`check_accuracy.py`, check 1): A, β and Ea of all 31 Arrhenius expressions, the 32
  printed third-body efficiencies and both Troe Fc values agree with Table 1, except the second R14 expression,
  Ea = −1629.3 cal/mol in the file and −1630 as printed (rounding in the table). The file also lists efficiencies
  that Table 1 does not print: none remain after CO and CO2 are removed.
- Thermodynamic and transport data are Burke et al.'s (the PelePhysics files); the paper used them "in [21]
  format" (Table 1 caption).

### Inputs

The 31 inputs of Table 1 (p. 6; the red and green entries, Sec. 2.2, p. 5), in the order of Table 1. Prior: each
input independent, Gaussian with mean = sd = its Table 1 value, truncated to the bounds L:U of Table 1 (p. 5).
Each A multiplies its own Arrhenius expression only (the high- and low-pressure limits of the falloff
reactions R9 and R15 are separate inputs); an efficiency sets that species' collision efficiency in that
reaction; `X6_eff_ArHe` sets the Ar and He efficiencies of X6 together (the two green entries, "varied
synchronously", Table 1 caption). Units: cm-mol-s with T in K (cm³/(mol s) for bimolecular reactions and
high-pressure limits, cm⁶/(mol² s) for R9's low-pressure limit and X6, s⁻¹ and cm³/(mol s) for R15's high- and
low-pressure limits).

| # | name | parameter | reaction | units | Table 1 value | bounds L:U | prior mean | prior sd | source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `R9_Ainf` | A, high-pressure limit | H + O2 (+M) ⇌ HO2 (+M) | cm^3/mol/s | 4.65084e+12 | 2e+12:1e+13 | 5.7e+12 | 2.186e+12 | Table 1, p. 6 |
| 2 | `R9_A0` | A, low-pressure limit | H + O2 (+M) ⇌ HO2 (+M) | cm^6/mol^2/s | 6.366e+20 | 0:8e+20 | 4.294e+20 | 2.238e+20 | Table 1, p. 6 |
| 3 | `R9_eff_H2` | efficiency of H2 | H + O2 (+M) ⇌ HO2 (+M) | - | 2 | 0:6 | 2.459 | 1.442 | Table 1, p. 6 |
| 4 | `R9_eff_H2O` | efficiency of H2O | H + O2 (+M) ⇌ HO2 (+M) | - | 14 | 0:28 | 14 | 7.554 | Table 1, p. 6 |
| 5 | `R9_eff_O2` | efficiency of O2 | H + O2 (+M) ⇌ HO2 (+M) | - | 0.78 | 0:3 | 0.9985 | 0.609 | Table 1, p. 6 |
| 6 | `R9_eff_Ar` | efficiency of Ar | H + O2 (+M) ⇌ HO2 (+M) | - | 0.67 | 0:3 | 0.862 | 0.5302 | Table 1, p. 6 |
| 7 | `R9_eff_He` | efficiency of He | H + O2 (+M) ⇌ HO2 (+M) | - | 0.8 | 0:3 | 1.022 | 0.6219 | Table 1, p. 6 |
| 8 | `R10_A` | A | H + HO2 ⇌ H2 + O2 | cm^3/mol/s | 2.75e+06 | 1e+06:5e+16 | 3.965e+06 | 1.99e+06 | Table 1, p. 6 |
| 9 | `R11_A` | A | H + HO2 ⇌ 2 OH | cm^3/mol/s | 7.079e+13 | 2e+13:1e+14 | 6.11e+13 | 2.259e+13 | Table 1, p. 6 |
| 10 | `R12_A` | A | HO2 + O ⇌ O2 + OH | cm^3/mol/s | 2.85e+10 | 1e+09:1e+11 | 3.654e+10 | 2.163e+10 | Table 1, p. 6 |
| 11 | `R13_A` | A | HO2 + OH ⇌ H2O + O2 | cm^3/mol/s | 2.89e+13 | 1e+13:6e+13 | 3.363e+13 | 1.369e+13 | Table 1, p. 6 |
| 12 | `R14a_A` | A | 2 HO2 ⇌ H2O2 + O2 (1st) | cm^3/mol/s | 4.2e+14 | 1e+14:2e+15 | 5.812e+14 | 3.139e+14 | Table 1, p. 6 |
| 13 | `R14b_A` | A | 2 HO2 ⇌ H2O2 + O2 (2nd) | cm^3/mol/s | 1.3e+11 | 5e+10:4e+11 | 1.819e+11 | 8.434e+10 | Table 1, p. 6 |
| 14 | `R15_Ainf` | A, high-pressure limit | H2O2 (+M) ⇌ 2 OH (+M) | s^-1 | 2e+12 | 5e+11:1e+12 | 7.565e+11 | 1.441e+11 | Table 1, p. 6 |
| 15 | `R15_A0` | A, low-pressure limit | H2O2 (+M) ⇌ 2 OH (+M) | cm^3/mol/s | 2.49e+24 | 1e+23:1e+25 | 3.232e+24 | 1.936e+24 | Table 1, p. 6 |
| 16 | `R15_eff_H2` | efficiency of H2 | H2O2 (+M) ⇌ 2 OH (+M) | - | 3.7 | 0:15 | 4.749 | 2.909 | Table 1, p. 6 |
| 17 | `R15_eff_H2O` | efficiency of H2O | H2O2 (+M) ⇌ 2 OH (+M) | - | 7.5 | 0:20 | 8.847 | 5.053 | Table 1, p. 6 |
| 18 | `R15_eff_H2O2` | efficiency of H2O2 | H2O2 (+M) ⇌ 2 OH (+M) | - | 7.7 | 0:20 | 8.979 | 5.098 | Table 1, p. 6 |
| 19 | `R15_eff_O2` | efficiency of O2 | H2O2 (+M) ⇌ 2 OH (+M) | - | 1.2 | 0:5 | 1.542 | 0.9457 | Table 1, p. 6 |
| 20 | `R15_eff_N2` | efficiency of N2 | H2O2 (+M) ⇌ 2 OH (+M) | - | 1.5 | 0:5 | 1.889 | 1.131 | Table 1, p. 6 |
| 21 | `R15_eff_He` | efficiency of He | H2O2 (+M) ⇌ 2 OH (+M) | - | 0.65 | 0:4 | 0.8369 | 0.5158 | Table 1, p. 6 |
| 22 | `R16_A` | A | H + H2O2 ⇌ H2O + OH | cm^3/mol/s | 2.41e+13 | 5e+11:1e+14 | 3.114e+13 | 1.889e+13 | Table 1, p. 6 |
| 23 | `R17_A` | A | H + H2O2 ⇌ H2 + HO2 | cm^3/mol/s | 4.82e+13 | 1e+12:9e+13 | 4.618e+13 | 2.425e+13 | Table 1, p. 6 |
| 24 | `R18_A` | A | H2O2 + O ⇌ HO2 + OH | cm^3/mol/s | 9.55e+06 | 1e+05:3e+07 | 1.192e+07 | 7.014e+06 | Table 1, p. 6 |
| 25 | `R19a_A` | A | H2O2 + OH ⇌ H2O + HO2 (1st) | cm^3/mol/s | 1.74e+12 | 5e+10:5e+12 | 2.13e+12 | 1.216e+12 | Table 1, p. 6 |
| 26 | `R19b_A` | A | H2O2 + OH ⇌ H2O + HO2 (2nd) | cm^3/mol/s | 7.59e+13 | 4e+12:4e+14 | 9.924e+13 | 5.947e+13 | Table 1, p. 6 |
| 27 | `X1_A` | A | H + HO2 ⇌ H2O + O | cm^3/mol/s | 3.97e+12 | 1e+12:9e+12 | 4.697e+12 | 2.145e+12 | Table 1, p. 6 |
| 28 | `X6_A` | A | O + OH + M ⇌ HO2 + M | cm^6/mol^2/s | 8e+15 | 2e+15:2e+16 | 9.943e+15 | 4.708e+15 | Table 1, p. 6 |
| 29 | `X6_eff_H2` | efficiency of H2 | O + OH + M ⇌ HO2 + M | - | 2 | 0:6 | 2.459 | 1.442 | Table 1, p. 6 |
| 30 | `X6_eff_H2O` | efficiency of H2O | O + OH + M ⇌ HO2 + M | - | 12 | 0:35 | 14.63 | 8.526 | Table 1, p. 6 |
| 31 | `X6_eff_ArHe` | efficiency of Ar/He | O + OH + M ⇌ HO2 + M | - | 0.7 | 0:3 | 0.8999 | 0.5528 | Table 1, p. 6 |

Prior mean and sd: those of the truncated distribution (closed form, `h2_flame_driver.prior_moments`); the
notebook scales each input by them.

Notes on Table 1 as printed:
- **R15_Ainf:** the Table 1 value 2.00e12 lies outside its own bounds 5e11:1e12. The prior is still well defined (a
  Gaussian of mean and sd 2e12 truncated to [5e11, 1e12], nearly flat there); every sample is below the Table 1
  value, and the nominal runs use the Table 1 value. Whether a bound or the value is misprinted is not known.
- The first H2O2 + OH ⇌ HO2 + H2O row is labelled "R18", like the H2O2 + O row above it, and the second has no
  label; this data set calls them R19a and R19b. Their first bound is printed "5(10:5(12)" and read as 5e10:5e12.
- The paper's Table 2 (p. 9) numbers the R15 N2 efficiency as parameter 19; in the print order of Table 1 it is
  the 20th (after H2, H2O, H2O2, O2), while R16 = 22 agrees. The data set uses the print order.

### Quantities of interest (conditions chosen for this data set)

The paper conditions on 91 experiments (77 premixed flame speeds, 6 flow reactors, 8 ignition delays; p. 5) whose
conditions are only in its refs. [16] and [3]; those were not read. The three QoIs below are of the same kinds,
at conditions chosen here; only the mixture and pressure of Su20 come from the paper.

| QoI | definition | conditions | source |
|---|---|---|---|
| τ (s) | constant-volume adiabatic reactor (as the paper simulates ignition delays, p. 5); first time T reaches T0 + (T_eq − T0)/2, T_eq = 1480 K the constant-volume equilibrium temperature (a kinetics-independent threshold, 1290 K); cubic Hermite interpolation between solver steps | H2:O2:Ar = 2:1:97 (molar), T0 = 1100 K, P0 = 2 atm | this data set's choice |
| Su1 (m/s) | freely propagating, unstrained, adiabatic premixed flame (Cantera `FreeFlame`; "the propagation speed of the steady, unstrained flame", p. 5) | H2/air (O2:N2 = 1:3.76), φ = 1, 1 atm, 298 K | this data set's choice |
| Su20 (m/s) | as Su1 | H2/air, φ = 0.4, 20 atm, 298 K | the paper's 2-D flame-kernel mixture and pressure (p. 11); its unburned temperature is not stated, 298 K is assumed |

Solvers: CVODES (rtol 1e-9, atol 1e-18) for τ; `FreeFlame` with mixture-averaged transport and no Soret term,
grid criteria ratio 2, slope 0.02, curve 0.04, initial width 3 cm (Su1) and 2 cm (Su20), solved from scratch with
`auto=True`. The paper used PREMIX and VODE (p. 5); it does not state its transport model.

**Failures.** A flame whose `auto=True` solve raises, or ends with Su ≤ 0 or a burned temperature below the
unburned one + 300 K, is retried from the converged nominal flame with time stepping and grid refinement (status
1); if that fails too, Su and its gradient are stored as NaN with status 2. An ignition that fails is stored as
NaN with status 2. In the shipped data no retry and no failure occurred: all 500 rows have status 0 for τ, Su1 and Su20
(printed by the notebook). The notebook drops a row only when a status flag is 2.

### Gradients and their coordinates

All gradients are with respect to the physical inputs (per unit of A or of efficiency).
- τ: central differences with step h_j = 1e-3 × prior sd_j (second-order one-sided where x_j − h_j < 0; never
  needed in the shipped rows) (`IgnitionDelay.grad`), 62 extra reactor runs per sample.
- Su1 and Su20: central finite differences on the fixed grid of each flame (`FlameSpeed.fd_grad`). The flame is
  solved from scratch as for the value, a second time with the gas that carries the shadow reactions (below; at
  multiplier 0: same value, identical grid, checked per row), and tightened in place on its grid (steady-state
  rtol 1e-9, atol 1e-14). Each input is then perturbed in place on that one flame object and the flame re-solved
  on the fixed grid, then reset and re-solved back to the base state: an A factor through `modify_reaction`, with
  step h = 1e-3 × prior sd and at most 2 % of its value (the cap matters in the lower tails, e.g. R9_A0 near 0); an
  efficiency through the multipliers of the parent reaction and its first shadow, (0, 1) for +h and (2, −1) for
  −h, h = Δ = 1e-3 × prior sd. Cantera 3.2 cannot change third-body efficiencies of an existing Solution
  (`modify_reaction` replaces only the rate; checked: the rates do not change), so the gas carries "shadow" copies
  of each parent reaction with one efficiency raised by Δ (and 2Δ, used by the adjoint below). For plain
  three-body reactions, which are linear in the efficiency, the efficiency scheme is exact; for the falloff
  reactions R9 and R15 it is a forward difference of the rate in the efficiency, first order in h, with a relative
  error of the rate derivative of at most 7.4e-05 in magnitude (H2 and H2O efficiencies of R9 and R15 at the
  Table 1 values, 900 K and 20 atm and 1500 K and 1 atm; `check_accuracy.py` (10)). A converged solution is
  not restored into a new flame object (`from_array`): the re-solve can re-anchor the flame on a newly inserted grid point and shift Su by about 1e-3, by a path that differed between two
  machines; central differences cancel such a shift, forward differences do not.
- Cantera's discrete adjoint (`Sim1D.solve_adjoint`, wrapped as `FlameSpeed.adjoint_grad`, with a parameter
  perturbation of 1e-5 × prior sd and efficiencies through the shadows with multipliers set to the second-order
  one-sided difference q(ε) + δ(−3q(ε) + 4q(ε+Δ) − q(ε+2Δ))/(2Δ)) is not used for the stored gradients. It points
  in the right direction but has the wrong scale: off by a few per cent, either way, for Su1 and too small by a
  factor of about 2 to 8 for Su20 ("Accuracy"). The driver's adjoint agrees with Cantera's own
  `get_flame_speed_reaction_sensitivities` for the A factors, so the defect is Cantera's.
- The notebook uses ln τ, ln Su1, ln Su20 with gradients ∇q/q, and scales each input by its prior sd.

### Sampling design

Plain Monte Carlo: 500 independent draws from the truncated-Gaussian prior, by the inverse-CDF transform of
`numpy.random.default_rng(2019452).random((500, 31))` (`sample_prior`). The paper's design is different: an
affine-invariant MCMC of the posterior (64 walkers, about 1.5e4 steps, 9.6e5 samples; p. 7) and 576 posterior
samples for its propagation studies (p. 11).

### Published results

- No active subspaces. Posterior: Fig. 1 (autocorrelations, p. 9) and Fig. 2 (triangle plot of 8 parameters,
  p. 10): positive correlation of R10 and R11 A factors (IDs 9, 8), negative correlation of the He and H2O
  efficiencies of R9 (7, 4), and broad, near-uniform posteriors for R9's high-pressure A and, less so, X1's A,
  "not constrained by the data" (p. 11).
- Propagation of 576 posterior samples: products of a 2-D lean kernel (H2/air, 20 atm, φ = 0.4) with sd below
  2 %, and jet ignition times with sd above 10 % of the mean (abstract, pp. 11–12).

### What the notebook finds (from these files)

Logarithms of the QoIs, inputs scaled by their truncated-prior sd, gradient estimator over M = 500 rows (none
dropped), 500 bootstrap replicates (seed 2019). Every number is printed by the notebook.

| | ln τ | ln Su1 | ln Su20 |
|---|---|---|---|
| dimension chosen in the notebook | 1 | 2 | 1 |
| gap that sets it (95 % bootstrap interval) | λ1/λ2 = 18.1 [14.8, 21.9] | λ2/λ3 = 9.6 [8.6, 11.1] | λ1/λ2 = 6.5 [5.5, 7.7] |
| other ratios | λ2/λ3 1.5; λ3/λ4 7.5 | λ1/λ2 44.7 [14.0, 99.9], set by the R9_A0 tail (below) | λ2/λ3 2.0; λ3/λ4 2.2; λ4/λ5 5.8 |
| λ1 (95 % bootstrap interval) | 0.01787 [0.0146, 0.0212] | 0.2063 [0.0592, 0.465] | 1.782 [1.54, 2.03] |
| leading components of w1 | R9_eff_Ar +0.702, R9_A0 +0.687 | R9_A0 +0.999 | R9_A0 −0.926, R9_eff_H2O −0.346 |
| leading components of w2 | R9_A0 −0.712, R9_eff_Ar +0.651 | R11_A +0.722, R10_A −0.510 | R9_eff_H2O −0.846, R9_A0 +0.372 |
| LOO R², quadratic in y1; in (y1, y2) | 0.9182; 0.9471 | 0.6972; 0.9266 | 0.8788; 0.9362 |

- **ln Su1 is two-dimensional.** λ2 and λ3 have separated bootstrap intervals ([0.0041, 0.00488] against
  [0.000399, 0.000537]), the 2-D subspace is stable (mean bootstrap distance 0.026), and λ1 + λ2 carry 0.997 of
  the trace. Its first ratio, λ1/λ2 = 44.7, is driven by the lower tail of the R9_A0 prior (truncated at 0): the
  rows with R9_A0 below 0.01 of its Table 1 value (prior mass 0.0055; 4 of the 500 drawn) carry 0.728 of the
  (R9_A0, R9_A0) entry of the matrix, which is nearly all of λ1, so λ1 rests on a few draws. On the 464 rows with
  R9_A0 ≥ 0.1 × its Table 1 value (exactly a sample of the prior with that lower bound), λ1/λ2 is 5.5, λ2/λ3 is
  8.0, and w1 mixes R9_A0 with R11_A, R10_A and R9_eff_H2O; ln τ (18.2) and ln Su20 (6.8) barely change.
- ln Su1's second direction contrasts R11_A (+0.722) and R10_A (−0.510), consistent with the paper's positive
  posterior correlation of these two A factors (p. 11). R9_Ainf and X1_A have activity scores of at most 4.4e-04
  of the trace, consistent with the paper's finding that its data do not constrain them (p. 11).

## Regenerating

```bash
# Cantera environment (Python 3.13 was used)
python3 -m venv cantvenv && cantvenv/bin/pip install cantera==3.2.0 numpy scipy
cantvenv/bin/python generate_data.py --out /tmp/h2flame                         # all 500 rows, then the CSVs
cantvenv/bin/python generate_data.py --out /tmp/h2flame --rows 0:80             # resumable chunks
cantvenv/bin/python generate_data.py --out /tmp/h2flame --procs 16              # rows in 16 processes
cantvenv/bin/python generate_data.py --out /tmp/h2flame --qoi Su20 --procs 16   # one QoI only
cantvenv/bin/python generate_data.py --out /tmp/h2chk --rows 0:3 --check        # subset vs shipped files
cantvenv/bin/python generate_data.py --out /tmp/h2nom --nominal                 # nominal.csv, nominal_gradients.csv
cantvenv/bin/python check_accuracy.py                                           # the Accuracy numbers (checks 1-7)
cantvenv/bin/python check_accuracy.py --checks 8,9,10                           # the 500-row adjoint comparison and checks 9, 10
# the mechanism file, from a PelePhysics clone at the commit above:
cantvenv/bin/python -c "import h2_flame_driver as D; D.build_mechanism('<PelePhysics>/Mechanisms/BurkeDryer/mechanism.yaml', 'burke2012_x1x6.yaml')"
```

`--out` is required and must lie outside this directory; the scripts never write into it (copy new files in by
hand after comparing them). Each row and QoI is saved as `DIR/rows/<qoi>/row_NNNN.npz` when done and skipped on a
rerun. When every row of a QoI exists, its `gradients_<qoi>.csv` is written to DIR, with a printed summary of the
finite-difference checks for Su1 and Su20 (step check, forward against central, drift, grid identity); when all
three QoIs exist, `inputs.csv` and `outputs.csv` as well. `--check` passes when values agree to 1e-6 relative and
gradients to 3e-4 of the largest sd-scaled entry, and the status flags are equal.

**Provenance of the shipped files.** All files were written with `h2_flame_driver.py` and Cantera 3.2.0, numpy
2.5.4, scipy 1.18.1 on Linux x86-64. The inputs, τ, Su1 and Su20 values and the τ gradients come from a run of
2026-11-03 in chunks of 80 rows (Python 3.13.18, a container with 2 shared CPUs), with `nominal.csv` and the τ row
of `nominal_gradients.csv`. The Su20 column of `outputs.csv` is the from-scratch solve of the 2026-11-09 run, whose
flames also gave the Su20 gradients (below): the 2026-11-03 run gives the same values to 1e-6 in 499 rows, whose
stored digits are kept, and row 412 holds the 2026-11-09 value. This is an external run, not reproducible from the repository without installing Cantera.

**Su1 gradients.** `gradients_Su1.csv` and the Su1 row of `nominal_gradients.csv` were computed on 2026-11-09 in
the container with `generate_data.py --qoi Su1 --procs 2` (3046 s of CPU for the 500 rows, about 6 s per row) and
`--nominal`. The 500 Su1 values recomputed in that run equal `outputs.csv` to 2.0e-11 (all status flags 0), and a
second, separate run of the same method on all 500 rows agrees with the shipped gradients to 5e-11 of the largest
sd-scaled entry of each row. The run's finite-difference summary: step check median 1.3e-05, max 4.7e-03
(row 416); forward against central median 2.8e-04, max 5.0e-03; largest drift back to the base state 2.2e-09;
identical grid in 500 of 500 rows (printed by `generate_data.py` when it writes `gradients_Su1.csv`).

**Su20 gradients.** `gradients_Su20.csv` was computed on 2026-11-09 with the per-row method of
`FlameSpeed.fd_grad` on an AWS c7a.4xlarge (16 cores, Ubuntu 24.04, Python 3.12.3, Cantera 3.2.0, numpy 2.5.4,
scipy 1.18.1) in 16 worker processes: the 500 rows in 31.9 min of wall time and 8.2 CPU-hours (per row: median
48 s, mean 59 s, max 450 s of CPU). The step size is that of Su1 (1e-3 of each prior sd, at most 2 % of an A
factor). Checks in that run: every row solved, with its shadow-gas base flame on the grid of the value's solve;
the 499 values other than row 412 reproduce `outputs.csv` to 1e-6; no forward-only or failed entry. To regenerate
it, run `generate_data.py --out DIR --qoi Su20 --procs N` on a many-core machine; `generate_data.py --check`
reproduces the run's rows here (table below). The Su20 row of `nominal_gradients.csv` was computed with
`generate_data.py --nominal` in the container (2026-11-09).

**Row 412.** The Su20 value of row 412 is that of the 2026-11-09 from-scratch solve, 0.01153983 m/s on a grid of
364 points, obtained in two environments (the container and the c7a), so that the value and its finite-difference
gradient belong to the same flame and `generate_data.py --check` passes. The 2026-11-03 run had given
0.01161286 m/s on 375 points: a grid-refinement decision went the other way, and the two differ by 6.3e-3,
less than Su20's grid sensitivity (1.4 % to 3.3 %, "Accuracy"). See Unknowns.

**Run time.** τ: under 1 s of CPU per row (62 extra reactor runs for the gradient); Su1 with its gradient: about
6 s; Su20 with its gradient: 48 s median, 59 s mean, up to 450 s on the c7a, and 142 s, 225 s and 109 s for rows
0, 1 and 412 in the container (`generate_data.py --check`). The whole data set therefore costs about 9 CPU-hours, almost all of it Su20, which
`--procs` spreads over cores. `--nominal` a few minutes; `check_accuracy.py` checks 1–7 about 2 to 3 h on one core (check 6 about 25 min per Su20 row),
`--quick` about 1 min, check 8 about 40 min, checks 9 and 10 a few seconds.

**`--check` results** (printed by `generate_data.py --check` in the container, 2026-11-09):

| rows | τ value / gradient | Su1 value / gradient | Su20 value / gradient | status flags |
|---|---|---|---|---|
| 0:2 (2026-11-09) | 5.9e-10 / 5.3e-05 | 3.2e-13 / 2.1e-11 | 3.4e-13 / 1.0e-11 | equal; PASS |
| 412 (2026-11-09) | 2.9e-09 / 4.8e-05 | 3.1e-14 / 1.3e-11 | 6.2e-12 / 2.2e-11 | equal; PASS |
| 497:500 (2026-11-09, `--qoi Su1`) | not run | 1.1e-13 / 1.7e-11 | not run | equal; PASS |

Values: largest relative difference; gradients: largest difference relative to the largest sd-scaled stored entry.
The τ gradient differences (up to 5.3e-05) are the noise of central differences of an adaptive ODE
solution, within the stated tolerance 3e-4. Other platforms or Cantera versions may differ by more; the shipped
CSVs are the record.

## Accuracy

All numbers below are printed by `check_accuracy.py` (Cantera 3.2.0, run 2026-11-09; check numbers in
parentheses) unless another source is named. Check rows: 0, 1, 2 and 105 (the smallest Su20 of the 500).

| quantity | value | source |
|---|---|---|
| mechanism file vs Table 1: A, β, Ea of 31 Arrhenius expressions, 32 efficiencies, 2 Troe Fc | largest relative difference 4.3e-04 (R14 2nd expression, Ea −1629.3 vs −1630 printed); every other entry agrees | `check_accuracy.py` (1) |
| QoIs at the Table 1 values, recomputed | τ = 0.832715 ms, Su1 = 2.275584 m/s, Su20 = 0.007107 m/s, equal to `nominal.csv` at the printed digits | `check_accuracy.py` (2) |
| Su1, finer grid (slope 0.01, curve 0.02) | relative change −8.6e-04 to −2.3e-03 (nominal and check rows) | `check_accuracy.py` (3) |
| Su20, finer grid | relative change +1.4e-02 to +3.3e-02 | `check_accuracy.py` (3) |
| τ re-solved at rtol 1e-12, atol 1e-22 | relative difference 3.9e-08 to 8.8e-08 | `check_accuracy.py` (4) |
| τ gradient with a step ten times smaller | relative 2-norm difference 5.2e-04 to 3.6e-03 | `check_accuracy.py` (4) |
| τ with the threshold at 1/4 or 3/4 of the temperature rise (definition, not error) | −9.4 % to −11.9 %; +25.5 % to +34.0 % | `check_accuracy.py` (4) |
| Su1 gradient: stored vs `fd_grad` recomputed | 2.7e-12 to 2.7e-11 | `check_accuracy.py` (5) |
| Su1: step check (central with a 10x step vs central); forward vs central | 1.8e-05 to 3.5e-04; 1.6e-04 to 1.6e-03 | `check_accuracy.py` (5) |
| Su1: largest drift back to the base state after a perturbation | 1.4e-10 to 7.7e-10 | `check_accuracy.py` (5) |
| Su1: stored vs independent central differences (step 1e-2 sd; A factors in place, efficiencies through a new gas and a restored solution) | 1.8e-05 to 3.5e-04 | `check_accuracy.py` (5) |
| Su1 over the 500 rows: step check (10x step); forward vs central; largest drift; base on the identical grid | median 1.3e-05, max 4.7e-03 (row 416); median 2.8e-04, max 5.0e-03; 2.2e-09; 500 of 500 | `generate_data.py --qoi Su1`, the 2026-11-09 run |
| Su20 gradient: stored vs `fd_grad` recomputed | 1.9e-12 to 1.4e-11 | `check_accuracy.py` (6) |
| Su20: step check (10x step); forward vs central | 1.7e-05 to 2.7e-03; 6.6e-04 to 4.8e-03 | `check_accuracy.py` (6) |
| Su20: largest drift back to the base state after a perturbation | 3.9e-09 to 7.6e-09 | `check_accuracy.py` (6) |
| Su20: stored vs independent central differences (as for Su1) | 7.3e-05 to 2.7e-03 | `check_accuracy.py` (6) |
| Su20 over the 500 rows: step check (10x step); largest drift; base on the identical grid | median 5.6e-05, max 1.5e-02 (row 119); 1.9e-08; 500 of 500 | the 2026-11-09 c7a run (external run, not reproducible from the repository without a many-core rerun; `generate_data.py` prints the same summary when it writes `gradients_Su20.csv`) |
| Cantera adjoint at the check rows: \|stored\| / \|adjoint\|; cosine; relative error | Su1: 0.971 to 0.992; 0.99994 to 1.00000; 1.4e-02 to 3.1e-02. Su20: 2.253 to 6.277; 0.99929 to 0.99999; 5.6e-01 to 8.4e-01 | `check_accuracy.py` (5, 6) |
| driver's adjoint vs Cantera's `get_flame_speed_reaction_sensitivities` (13 plain A factors) | Su1 2.7e-10 to 8.1e-10; Su20 8.0e-11 to 1.8e-08 | `check_accuracy.py` (5, 6) |
| Cantera adjoint over the 500 rows, Su1: \|stored\| / \|adjoint\|; cosine; relative error | median 0.979, 5–95 % 0.956 to 0.997 (extremes 0.942 at row 362 and 1.02 at row 200); median 0.99987, min 0.99934; median 0.0278, 5–95 % 0.0104 to 0.0493, max 0.0678 (row 362) | `check_accuracy.py` (8) |
| Cantera adjoint over the 500 rows, Su20: \|stored\| / \|adjoint\|; cosine | median 4.10, 5–95 % 1.86 to 8.14 (extremes 0.919 at row 355 and 21 at row 47); median 0.99984, min 0.99402 | `check_accuracy.py` (8) |
| plain Cantera `FreeFlame` (no driver) at the Table 1 values: R9 elasticity of Su, fixed-grid central differences / adjoint | Su20: 1.36 with coarse grid criteria (ratio 3, slope 0.06, curve 0.12; 132 points), 3.91 with the data's (313 points); Su1: 0.99 (144 points) and 0.96 (375 points) | `check_accuracy.py` (9) |
| efficiency scheme of `fd_grad` for R9 and R15 (forward difference in the efficiency): relative error of the rate derivative | −1.9e-07 to −7.4e-05 | `check_accuracy.py` (10) |
| row 412: Su20 re-solved from scratch vs `outputs.csv` | 0.0115398311 m/s on 364 points, as stored; relative difference 6.2e-12 | `check_accuracy.py` (7) |
| status flags of the 500 rows | all 0 (no retry, no failure) | notebook |

Reading the table:
- **Flame-speed gradients.** The shipped finite differences are reproducible (to 1.4e-11 or better between
  machines and runs), their truncation error is small (a step ten times larger changes them by a median of
  1.3e-05 for Su1 and 5.6e-05 for Su20; the largest Su20 change, 1.5e-02 at row 119, comes from R15_A0, which
  there lies at 0.053 prior sd, so that the larger step is 19 % of its value; the step actually used is ten times
  smaller, and the truncation error falls with its square), and a second finite-difference scheme for the
  efficiencies agrees (to 3.5e-04 or better for Su1, 2.7e-03 or better for Su20). For efficiencies the 10x-step
  check extrapolates the same multipliers linearly, so it tests how linear Su is in the rate perturbation; check 10
  bounds the truncation of the efficiency scheme itself.
- **Cantera's adjoint** points in the right direction (cosine at least 0.9993 for Su1 and 0.994 for Su20
  over the 500 rows) but has the wrong scale: for Su1, off by 1.0 % to 4.9 % (5–95 % of the rows; at most 6.8 %), almost
  purely a scale factor (|FD| / |adjoint| 0.942 to 1.02); for Su20, too small by a row-dependent factor (median 4.10,
  5–95 % 1.86 to 8.14; 0.919 to 21 at the extremes). The driver's adjoint matches Cantera's own `get_flame_speed_reaction_sensitivities`, so
  the defect is Cantera's (its steady-state Jacobian or the adjoint solve), not the driver's. It depends on the
  grid: for a plain `FreeFlame` at the nominal Su20 condition, without the driver, the adjoint's R9 elasticity is
  1.36 times too small on a coarse grid (132 points) and 3.91 times on the data's (313 points), against central
  differences on the same fixed grid; at the Su1 condition it is within 4 % on both grids (check 9).
- **Grid resolution.** Su20 is about 1.4 % to 3.3 % below its finer-grid value, Su1 about 0.1 % to 0.2 % above it.
  These are biases of a few hundredths in ln Su20 and a few thousandths in ln Su1, against prior sds of ln Su of
  0.956 and 0.130 (notebook). The gradients are those of the discrete flame on its grid.
- **Ignition delay.** The solve and its gradient are accurate to the digits that matter; the QoI depends strongly
  on its definition (threshold), which is this data set's choice.

## How these files relate to the paper

- **Not the paper's computation.** The paper samples the *posterior* of the 31 parameters given 91 experiments
  (PREMIX flames, VODE reactors, emcee). These files sample the *prior* the paper starts from and evaluate three
  QoIs of the kinds it calibrates against, at conditions chosen here (only the 20 atm, φ = 0.4 H2/air mixture is
  the paper's, from its flame-kernel case, p. 11). The active subspaces describe which combinations of the 31
  parameters such measurements inform; the paper reports none. Its posterior findings can be compared only
  qualitatively (notebook table).
- The paper says its 576 posterior calibrations are "posted online" (p. 3); the preprint gives no address, and one
  web search (2026-11-03) found none. With them, the same QoIs could be evaluated on the posterior.

## License and credit

Code (`*.py`, the notebook): MIT. Data (the CSVs): CC BY 4.0. `burke2012_x1x6.yaml` is derived from PelePhysics'
`Mechanisms/BurkeDryer/mechanism.yaml` (BSD-3-Clause; `LICENSE-PelePhysics.txt`, kept beside the file; Alliance
for Sustainable Energy, LBNL and others). The mechanism itself carries "(c) Burke, Chaos, Ju, Dryer, and
Klippenstein; Princeton University, 2011" and states no licence; credit M. P. Burke, M. Chaos, Y. Ju, F. L. Dryer,
S. J. Klippenstein (2012). Reactions X1 and X6: X. You, A. Packard, M. Frenklach (2012), as printed in Bell et al.
Table 1. Study design and priors: J. Bell, M. Day, J. Goodman, R. Grout, M. Morzfeld; cite their paper (above).
Cantera: D. G. Goodwin, H. K. Moffat, I. Schoegl, R. L. Speth, B. W. Weber, https://www.cantera.org.

## Unknowns

1. The conditions of the paper's 91 experiments (refs. [16], [3]); the three conditions here are this data set's.
2. The unburned temperature of the paper's 20 atm flame kernel (298 K assumed).
3. The paper's transport model for PREMIX (mixture-averaged without Soret used here) and its ignition-delay
   criterion (here half the adiabatic temperature rise).
4. R15_Ainf: Table 1 value 2.00e12 outside its bounds 5e11:1e12; which is misprinted.
5. The paper's parameter numbering (Table 2 gives the R15 N2 efficiency ID 19, Table 1's order gives 20).
6. The URL of the 576 posted posterior samples.
7. Why Cantera 3.2.0's discrete adjoint has the wrong scale for these flames: too small by a factor of about 2
   to 8 for Su20, off by 1 % to 5 % (5–95 % of the rows) for Su1, with a defect that grows with grid refinement
   (check 9). Not reported upstream; check 9 is a self-contained example.
8. Row 412: the 2026-11-03 run gave Su20 = 0.01161286 m/s on a grid of 375 points; that solve could not be
   reproduced (two environments give 0.01153983 m/s on 364 points), and the shipped value is the reproducible one.
   Rare grid-refinement flips of this kind can occur on other machines.
