# Ammonia_Syngas_Optimized_Kinetics: data

Kind: **stored data (generated with Cantera 3.2.0), with a regeneration script.** The model runs in Cantera, an
outside code, so it is not a numpy/scipy "Python model" in the repository's sense (CLAUDE.md, "Two kinds of data
sets"); the Cantera driver is `ammonia_idt_driver.py`. A numpy/scipy port of a 37-species, 246-entry stiff
mechanism with falloff, duplicate reactions and an excited-state species (OH\*) would be a second, unchecked kinetics
code, so none was written. The shipped CSVs are the record. **Regenerating the data, or rerunning the accuracy
checks, needs Cantera 3.2.0** (`pip install cantera==3.2.0`) **and the paper's mechanism file, which is not part of
this data set**: it is downloaded from the article page and converted with `convert_mechanism.py` ("Regenerating").
`generate_data.py` and `check_accuracy.py` stop with an error that says what to install or download when either is
missing. The notebook needs only numpy, matplotlib and active_subspaces: it reads the shipped files and the driver's
constants and region geometry.

## Files

| file | contents | role |
|---|---|---|
| `Ammonia_Syngas_Optimized_Kinetics.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2023; outputs committed |
| `ammonia_idt_driver.py` | `IgnitionModel` (Cantera driver: `ignition_delay`, `taus`, `log10_tau_and_grad`, `central_grad`); `REACTIONS`, `CANTERA_INDEX`, `F_TABLE`, `f_of_T`, `NAMES`, `LOWER`, `UPPER`, `T_NODES`, `to_theta`, `violation`, `feasible`, `sample_feasible`, `CONDITIONS` | the model; everything but `IgnitionModel` imports without Cantera |
| `convert_mechanism.py` | makes the mechanism file `ammonia_syngas_optimized.yaml` (37 species, 246 reaction entries) from the article's supplementary zip `d3re00160a1_suppl.zip` with Cantera's `ck2yaml` (`--zip`, `--out`); checks the zip's sha256 | prepares the model input (needs Cantera 3.2.0); the mechanism itself is not included (see "License and credit") |
| `generate_data.py` | writes the five CSVs (`--out`, `--rows`, `--max-minutes`, `--check`) | regenerates the data (needs Cantera 3.2.0) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" below (`--quick`: checks 1–3) | accuracy check (needs Cantera 3.2.0) |
| `inputs.csv` | 100 × 123: node coordinates x_(r,T) = log10(k_r(T)/k*_r(T)), reactions R01–R41 in Table 2 order, T = 300, 750, 3000 K | Monte Carlo inputs, uniform on the feasible region, seed 2023462 |
| `outputs.csv` | 100 × 3: ignition delays tau_A, tau_B, tau_C in s | model output |
| `gradients_A.csv`, `gradients_B.csv`, `gradients_C.csv` | 100 × 123 each: d log10(tau)/d x_j (dimensionless) | model output (forward differences) |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the written digits (`%.15e`) before the model runs, so each
output belongs exactly to the stored input row.

## Model

### Citation

W. Li, C. Zou, and H. Yao, "An optimized model for ammonia/syngas combustion", *Reaction Chemistry & Engineering*
8(8) (2023) 2071–2085, doi:10.1039/D3RE00160A (Crossref-checked). The PDF read is the publisher's version; the
article is not open access. The electronic supplementary information (ESI) has two files: `d3re00160a1_suppl.zip`
(the optimized mechanism: `optimized.inp`, `thermo.dat`, `tran.dat`) and `d3re00160a2_suppl.pdf` (Figs. S1–S15,
Tables S1–S3). Neither is included here ("Regenerating" gives the download). sha256: zip ee493371ad68aef892269bc3d19951609caa364d4ccd7d83112c3d0414a86a6b (65 231 bytes), pdf
06ab7f2088be7500baf477e2a7b46c5babb52611a77077b42c2d46e401267de1 (1 677 422 bytes).

### Physics and solver

- The paper optimized the Arrhenius parameters (A, b, E) of 41 influential reactions, 123 parameters, against 200
  shock-tube and 172 RCM ignition delays and 1411 laminar flame speeds (abstract and Sec. 2, pp. 2071, 2073; Table 2,
  p. 2076). For pressure-dependent reactions only the low-pressure limit was optimized (p. 2075; Table 2 caption).
- Region (eqn (4)–(5), p. 2074): |log10(k(T)/k0(T))| <= f(T), i.e. −f(T) ln 10 <= α + β ln T − ε/T <= f(T) ln 10
  with α = ln(A/A0), β = b − b0, ε = (E − E0)/R. The uncertainty factor f(T) of each reaction is in ESI Table S3
  (ESI pp. 9–11): a constant "between 300 K and 3000 K", or values at 3–4 temperatures, piecewise linear between
  them and constant outside them (Table S3 footnote). `F_TABLE` in the driver holds the printed values. Two rows
  use the wrong verb for the change they describe (N2O + H = N2 + OH: "decreasing to 0.70 at 1200 K, increasing to
  0.60 at 2600 K"; NH3 + H = NH2 + H2: "1.2 at 400 K, increasing to 0.9 at 650 K"); the printed numbers are used.
  NH2 + H (+M) = NH3 (+M) is given only "0.4 between 2000 K and 3000 K"; by the footnote it is 0.4 at every T.
- The constraint is enforced on 300–3000 K (1 K grid), the range of the paper's integrated adjustments (p. 2076).
- **Centre (departure from the paper).** The trial-model ("nominal") rate parameters A0, b0, E0 are not printed in
  the paper or the ESI; Table 2 gives only the optimized A*, b*, E* and the integrated adjustment
  ∫log10(k*/k0)dT / ∫f(T)dT in per cent. The region is therefore centred on the optimized rate constants k*:
  k(T) = k*(T) exp(α + β ln T − ε/T), so x = 0 is the paper's optimized model. Because the optimized model sits
  inside the paper's region but not at its centre (its adjustments are 1.7 % to 78.1 % of the uncertainty range,
  Table 2), the region used here is the paper's region shifted by k*/k0 for each reaction.
- **Coordinates (this data set's choice).** x_(r,i) = log10(k_r(T_i)/k*_r(T_i)) at T_i = 300, 750, 3000 K. They fix
  (α, β, ε) of reaction r by a 3 × 3 linear solve (`to_theta`), so the 123 inputs are a linear reparametrization
  of the paper's 123 (α, β, ε). Bounds of the box: |x_(r,i)| <= f_r(T_i). The feasible region is the part of the
  box where eqn (5) holds at every T on the grid; the notebook prints each reaction's feasible share of its
  3-D box (Monte Carlo): 0.285 to 0.953, product 0.012 over the 41 reactions.
- **Branching-ratio constraint, eqn (6), p. 2074, not imposed:** it bounds k(NH2 + NO = NNH + OH)/k(NH2 + NO =
  N2 + H2O) − br_th(T) by 0.2, with br_th from Klippenstein's theory; br_th(T) is not printed.
- **Duplicate entries.** N2O + H = N2 + OH has two Arrhenius entries (Table 2 lists both with one adjustment); one
  (α, β, ε) acts on both. Falloff reactions R03, R04, R06, R22: the low-pressure limit is perturbed, the
  high-pressure limit and the Troe or Lindemann form are kept.
- **Mechanism file vs. Table 2.** `check_accuracy.py` compares the 42 Arrhenius entries of the 41 reactions in the
  converted file with Table 2: A* agree to 3.8e-4 relative (the file has one more digit than the table), b* is the
  same to the printed digits, and E* agrees to 3.4e-4 relative, except one entry: NH2 + H = NH + H2 has
  E* = 3.5639e3 cal/mol in the ESI file and 3.564 × 10^4 in Table 2. The two values give rate constants that
  differ by a factor 1.0e7 at 1000 K and 3.2e3 at 2000 K. The file, which is what the authors distribute as the
  optimized model, is used; which value is intended is open question 1.
- Conversion (`convert_mechanism.py`): the same as `ck2yaml --permissive --input optimized.inp --thermo thermo.dat
  --output ammonia_syngas_optimized.yaml` on the unzipped files (Cantera 3.2.0; `--permissive` lets the converter
  skip a duplicate HOCO entry in `thermo.dat`). 37 species and
  237 reactions, as in the paper (p. 2073); Cantera stores 246 entries because 9 reactions are written as
  DUPLICATE pairs. `tran.dat` is not needed (no transport).
- **Reactor and QoIs.** Adiabatic constant-volume ideal-gas reactor (Cantera `IdealGasReactor`, CVODES BDF,
  rtol 1e-8, atol 1e-20), the paper's model for shock tubes (p. 2073). The delay is the time of the maximum of the
  ignition signal named in ESI Table S1 (ESI p. 8) for the experiment the condition is taken from, located by the
  vertex of a parabola through the three solver steps around the largest value:

  | key | mixture (mole fractions) | P0 | T0 | criterion | source of the condition |
  |---|---|---|---|---|---|
  | A | NH3 0.011429, O2 0.008571, Ar 0.98 (φ = 1) | 1.4 atm | 1850 K | max d[OH\*]/dt | Mathieu and Petersen (2015); ESI Table S1, Fig. S14a |
  | B | NH3/air, φ = 1 (O2:N2 = 21:79) | 20 atm | 1400 K | max dP/dt | Shu et al. (2019); ESI Table S1 (15.9–41.6 atm, 1181–1581 K) |
  | C | NH3/O2/Ar = 0.12/0.18/0.70 | 40 bar | 1100 K | max dP/dt | He et al. (2019); ESI Table S1, Fig. S14b |

  For C the paper used the experimenters' RCM volume profile (p. 2073). Here the reactor is adiabatic and
  constant-volume from the end-of-compression state (a simplification: no heat loss, so the delays are shorter;
  see "How these files relate to the paper"). The integration stops once T has passed 90 % of the rise to the
  constant-volume equilibrium temperature and the signal has fallen below 20 % of its maximum; a run raises an
  error if no ignition occurs within 1 s, or if the maximum is at an end of the record. The signal is evaluated
  only after T has risen by 2 % of the rise to equilibrium.
- **Ignition criterion.** At x = 0, `check_accuracy.py` prints the relative difference between the delay used and
  other criteria: condition A, max dT/dt −3.5e-2, half temperature rise −3.9e-2, max dP/dt −3.5e-2
  (the OH\* peak comes 3.5 % after the steepest temperature rise); B, +2.9e-6 and −9.1e-4; C, +6.7e-8 and
  −7.2e-5.

### Inputs

| inputs | units | nominal | range | source |
|---|---|---|---|---|
| x_(r,i), r = 1..41 (Table 2 order), T_i = 300, 750, 3000 K: 123 inputs named `R01_300K` … `R41_3000K` | log10 of a rate-constant ratio (dimensionless) | 0 (the optimized model) | box |x_(r,i)| <= f_r(T_i), restricted to the feasible region of eqn (5) on 300–3000 K | eqn (4)–(5), p. 2074; f_r(T): ESI Table S3, pp. 9–11; coordinates and centre: this data set |

The notebook prints, for each reaction, f_r at the three nodes, the smallest f_r on 300–3000 K and the
feasible share of the 3-D box.

### Gradients

Forward differences of log10 τ in the node coordinates, step h_j = 1e-3 × f_r(T_i) (the same step, 1e-3, in
normalized coordinates), 123 extra ignitions per condition per sample (`log10_tau_and_grad`). Accuracy below.

### Sampling design

Monte Carlo, uniform on the feasible region: for each sample and each reaction, the three node coordinates are
drawn uniformly in the reaction's box and redrawn until the reaction satisfies eqn (5) on the grid
(`sample_feasible`, numpy `default_rng(2023462)`). The region is a product of the 41 per-reaction regions, so this
is exactly uniform on it. The first n rows of a larger design are the design of size n. 100 samples: chosen to fit
the shared 2-core machine (about 27 s of CPU per sample for 372 ignitions, 45 min of CPU in all),
and enough for the leading direction (the two independent halves of the design give |w1·w1| of 0.997 or more,
in the notebook). The paper ran an optimization (Bayesian optimization, then a genetic algorithm,
Sec. 4.2, p. 2075), not a sampling study, and reports no seeds.

### Published results (paper)

- Table 2 (p. 2076): A*, b*, E* of the 41 reactions and the integrated adjustment relative to the uncertainty
  range; 39 of the 41 are within 50 % (p. 2076); the largest is NH3 (+M) = NH2 + H (+M), 78.1 %.
- Table 3 (p. 2079): average prediction errors over the IDT and LFS targets. ESI Fig. S14 (ESI p. 7): delays of the
  optimized and eleven other models at the Mathieu–Petersen and He et al. conditions.
- No active subspace, sensitivity indices of τ to the 123 parameters, or forward uncertainty propagation.

### What the notebook finds (from these files)

- One active direction at each condition; dimension n = 1, chosen in the notebook from the gap, the bootstrap
  subspace distances, the absence of a later gap and the small gain in leave-one-out R² from a second variable
  (values in this data set's node coordinates, 300, 750 and 3000 K):

  | condition | λ1/λ2 (95 % bootstrap interval) | λ2/λ3 | λ1/trace | subspace distance n = 1 / n = 2 | LOO R², quadratic in y1 / in (y1, y2) |
  |---|---|---|---|---|---|
  | A (1850 K, 1.4 atm) | 32.0 [25.4, 37.1] | 1.16 | 0.918 | 0.028 / 0.433 | 0.9448 / 0.9582 |
  | B (1400 K, 20 atm) | 8.5 [6.9, 10.6] | 2.37 | 0.801 | 0.055 / 0.15 | 0.9138 / 0.9294 |
  | C (1100 K, 40 bar) | 13.7 [10.5, 17.2] | 1.46 | 0.831 | 0.051 / 0.311 | 0.9278 / 0.9390 |

- w1 at A: NH3 + H = NH2 + H2 carries 0.565 of ‖w1‖², NH2 + H (+M) = NH3 (+M) 0.166, NO + H (+M) = HNO (+M) 0.102.
  At B and C the weight is on H2NO + O2 = HNO + HO2 (0.226, 0.245) and the NH2 + NO, NH2 + NO2 and NH2 + O2/HO2
  reactions. |w1(B)·w1(C)| = 0.769; A's direction is nearly orthogonal to both (0.270, 0.064).
- Signs, from the mean sensitivity of log10 τ to multiplying one rate constant by the same factor at every
  temperature (the sum of its three node derivatives, which does not depend on the node choice): at A a faster
  NH3 + H = NH2 + H2 lengthens the delay (+0.225) and a faster NH2 + H (+M) = NH3 (+M) shortens it (−0.277); at B and
  C the terminating channel NH2 + NO = N2 + H2O lengthens it (+0.282, +0.576) and the chain-carrying
  NH2 + NO = NNH + OH shortens it (−0.313, −0.516); at C NH2 + NO2 = N2O + H2O (+0.341) and NH2 + NO2 = H2NO + NO
  (−0.230) also act in opposite directions.
- **Dependence on the node temperatures.** Other nodes give a linear reparametrization of the same region and the
  same uniform distribution, but the inputs are then scaled by f_r at the new nodes, so the active subspace changes.
  The notebook transforms the stored gradients exactly to five other node sets ((300, 500, 3000), (300, 1000, 3000),
  (300, 1200, 3000), (400, 1000, 2000) and (500, 1500, 2500) K). What holds for every set: n = 1 (λ1/λ2 is 23.5–36.6
  at A, 7.9–9.1 at B and 13.5–14.3 at C, against λ2/λ3 of 1.12–1.25, 2.37–2.41 and 1.40–1.47); the first three
  reactions at A (NH3 + H = NH2 + H2, NH2 + H (+M) = NH3 (+M), NO + H (+M) = HNO (+M)); H2NO + O2 = HNO + HO2 first
  at B and C, then NH2 + NO = NNH + OH at B and NH2 + NO2 = N2O + H2O at C; B and C close (|w1(B)·w1(C)|
  0.672–0.796), A nearly orthogonal to both. What does not hold: the shares. NH3 + H = NH2 + H2 carries 0.364 to
  0.639 of ‖w1‖² at A, and the middle node carries 0.025–0.613 (A), 0.505–0.997 (B) and 0.779–0.998 (C) of it (0.385,
  0.691 and 0.907 at 750 K). The per-node shares and the exact ratios are properties of the chosen coordinates.
- The two halves of the design give the same w1 (|w1·w1| 0.999, 0.997, 0.997); λ1/λ2 varies more between halves
  (A 32.1/31.0, B 10.0/7.4, C 10.4/17.1).
- Over the region, τ (2.5–97.5 %) spans 1.154–3.290 ms (A), 0.500–1.413 ms (B) and 5.172–75.800 ms (C); at x = 0 it
  is 1.8703, 0.8175 and 15.1083 ms (computed with the driver when the data were made; quoted in the notebook as
  stored values; `check_accuracy.py` prints them).

## Regenerating

The mechanism file is not part of this data set (see "License and credit"). Get it as follows.

1. Open the article page, https://doi.org/10.1039/D3RE00160A (Reaction Chemistry & Engineering; the article is not
   open access, so the supplementary files may need a subscription), and follow the link "Electronic
   supplementary information". Download `d3re00160a1_suppl.zip`: 65 231 bytes, sha256
   ee493371ad68aef892269bc3d19951609caa364d4ccd7d83112c3d0414a86a6b. It holds `optimized.inp`, `thermo.dat` and
   `tran.dat` (CHEMKIN format). The supplementary information states no licence.
2. Convert it with `convert_mechanism.py` (below). It checks the sha256, reads `optimized.inp` and `thermo.dat` from
   the zip, runs Cantera's `ck2yaml` in permissive mode and checks that the result has 37 species and 246 reaction
   entries.
3. Write the converted file outside this data-set folder (`convert_mechanism.py` refuses the folder and any
   folder in it) and set the environment variable `AMMONIA_SYNGAS_MECH` to its path; the driver reads it from there.

```bash
# Cantera environment (Python 3.13 was used; Cantera has wheels for Linux and macOS)
python3 -m venv cantvenv && cantvenv/bin/pip install cantera==3.2.0
cantvenv/bin/python convert_mechanism.py --zip d3re00160a1_suppl.zip --out /path/to/ammonia_syngas_optimized.yaml
export AMMONIA_SYNGAS_MECH=/path/to/ammonia_syngas_optimized.yaml
cantvenv/bin/python generate_data.py --out /tmp/ammonia                          # all 100 rows
cantvenv/bin/python generate_data.py --out /tmp/ammonia --max-minutes 10         # resumable chunks
cantvenv/bin/python generate_data.py --out /tmp/ammonia_check --rows 0:3 --check # a subset vs. the shipped files
cantvenv/bin/python check_accuracy.py                                             # the Accuracy numbers
```

Without Cantera or without the mechanism file, `generate_data.py`, `check_accuracy.py` and the driver's
`IgnitionModel` stop with a message that gives these steps.

`generate_data.py` writes each finished row to `DIR/rows/row_NNNN.npz` and skips it on the next run, and writes the
CSVs when all requested rows exist. `--out DIR` must be outside this directory (not the directory or a folder in
it), in every mode. `--check` compares τ (relative tolerance 1e-6) and each gradient row (relative
2-norm difference in normalized coordinates) with the shipped files, with gradient tolerances of 5e-2 at condition A
and 2e-3 at B and C. These are set by noise in the forward differences, not by their truncation error: on the same
platform a recomputed row matches the stored one to about 1e-11, but a rounding-level change of the inputs,
x (1 ± 1e-13), which stands in for a change of platform, changes the stored gradients by up to 1.7e-2 at A and
5.6e-4 at B and C and τ by up to 2.2e-7 (`check_accuracy.py`, check 7). The step of 1e-3 amplifies the solver noise
in τ, which is largest at A (check 4). The tolerances leave a margin of about three over these changes; a rerun on
another platform is not expected to match the gradients at A more closely than a few per cent.

Provenance of the shipped files: generated on 2026-11-08 in a Linux container (x86-64, 2 cores shared with other
jobs, `nice`), Python 3.13.18, Cantera 3.2.0, numpy 2.5.4, with `generate_data.py` and `ammonia_idt_driver.py` as
shipped, one process, in resumable 10-minute chunks. CPU time per sample 27 s (372 ignitions), 45
min in all (about 1.6 h of wall time, set by the shared machine). On the same platform, `generate_data.py --out DIR
--rows 0:3 --check` recomputed rows 0–2 (1.5 min of CPU) and passed: inputs, τ and gradients agree within the
printing resolution of the check (it prints 0.0e+00 for every file).

## Accuracy

`check_accuracy.py` recomputes every number in this list (it needs Cantera and the mechanism file; with Cantera 3.2.0 on Linux x86-64,
Python 3.13, 2026-11-08, it used 461 s of CPU and printed the values below).

- **Mechanism file vs. Table 2** (check 1): A* 3.8e-4 relative, b* the same to the printed digits, E* 3.4e-4
  relative, one flagged entry (NH2 + H = NH + H2, E*; k(file)/k(Table 2) = 1.0e7 at 1000 K, 3.2e3 at 2000 K).
- **τ at x = 0** (check 2): 1.8703 ms (A), 0.8175 ms (B), 15.1083 ms (C).
- **Ignition criterion** (check 3): the figures under "Physics and solver".
- **τ** (check 4). Re-solving rows 0, 1, 2 at rtol 1e-11, atol 1e-24 changes τ by 3.0e-7 to 5.3e-7 relative at A
  and by 2.1e-8 to 5.1e-8 at B and C.
- **Gradients** (check 5; relative 2-norm error in normalized coordinates at row 0, against central differences at
  rtol 1e-11 with the same step / a step ten times smaller): A 1.4e-2 / 1.6e-2, B 3.4e-4 / 3.5e-4, C 3.6e-4 /
  3.6e-4. The two central-difference gradients differ from each other by 1.3e-2 at A, 8.2e-5 at B and 1.5e-6 at C,
  so at A the error comes from noise in τ (check 4: τ at A moves most under a change of tolerance), which the small
  difference step amplifies, not from the truncation error of the forward difference.
- **Directional derivatives** (check 6; central differences of log10 τ along four random unit directions at rows 0,
  1, 2, rtol 1e-11, against the stored gradient, largest error over the directions relative to the gradient norm):
  A 1.1e-3 to 3.0e-3, B 3.6e-5 to 4.4e-5, C 5.9e-5 to 1.1e-4.
- **Reproducibility** (check 7; rows 0 and 14 recomputed with the shipped settings, against the stored values): at
  the stored x, τ agrees to the rounding level (below 1e-15) and the gradients to about 1e-11; at x (1 ± 1e-13), τ changes by up to 2.2e-7 (A),
  1.9e-8 (B) and 9.0e-9 (C), and the gradients by up to 1.7e-2 (A), 5.6e-4 (B) and 1.8e-4 (C). These set the
  tolerances of `generate_data.py --check` ("Regenerating").
- The gradient noise at A, about 1 % of the gradient norm, is small next to the eigenvalue gap there
  (λ2/λ1 = 1/32.0 in the notebook). In the notebook, the least-squares slope of log10 τ on the ten most influential
  normalized inputs and the mean normalized gradient agree in direction (cosines 0.990, 0.932, 0.977).

## How these files relate to the paper

- **Not the paper's computation.** The paper optimized the 123 parameters against data; it did not sample them.
  These files are a forward study of the paper's optimized mechanism over the paper's uncertainty region,
  re-centred on the optimized constants (the trial-model constants are not published). Departures, all stated
  above: the centre; the node coordinates and the uniform distribution on the region; no branching-ratio
  constraint; constant-volume adiabatic simulation of the RCM condition C without the volume profile; Cantera
  instead of CHEMKIN Pro.
- **Agreement at x = 0.** Condition A: τ = 1.8703 ms here; the optimized model's curve in ESI Fig. S14a reads
  about 1.9 ms at 1850 K (digitized from the figure image, 1000/T = 0.541). Condition C: τ = 15.1083 ms here;
  the curve in ESI Fig. S14b reads about 19 ms at 1100 K (digitized; the curve is hidden by other models' curves at
  that temperature, so it was interpolated linearly between the visible points at 1098 K and 1107 K). The
  shorter delay here is the expected effect of omitting the RCM volume profile (heat loss after compression). Condition B has no figure.

## Open questions (for the authors)

1. E* of NH2 + H = NH + H2: Table 2 prints 3.564 × 10^4 cal/mol, the ESI mechanism 3.5639e3. Which is intended?
2. The trial-model (nominal) parameters A0, b0, E0 of the 41 reactions, needed to centre the region as in the paper.
3. The theoretical branching ratio br_th(T) of eqn (6).
4. Table S3 rows for N2O + H = N2 + OH and NH3 + H = NH2 + H2: are the printed values (not the verbs) right?

## License and credit

Code (`*.py`, the notebook): MIT. Data (the CSVs): CC BY 4.0. The paper's mechanism (ESI `d3re00160a1_suppl.zip`;
RSC supplementary material, no licence stated; credit W. Li, C. Zou, H. Yao; its `thermo.dat`, and so the converted
file's description, starts with the header of AramcoMech 1.3, W. K. Metcalfe, S. Burke, H. J. Curran) is not
included here, in original or converted form, and neither is the article: both are linked by DOI. "Regenerating"
gives the download and the conversion; the notebook runs without them. Study design and mechanism: W. Li, C. Zou and
H. Yao; cite their paper (above). Cantera: D. G. Goodwin, H. K. Moffat, I. Schoegl, R. L. Speth, B. W. Weber,
https://www.cantera.org (BSD-3-Clause; not shipped).
