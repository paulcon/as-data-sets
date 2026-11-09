# European_LV_Feeder_Probabilistic_Power_Flow: data

Kind: **stored data (generated with OpenDSS through OpenDSSDirect.py 0.9.4), with a regeneration script.** The model
is an unbalanced three-phase power flow solved by OpenDSS, an outside code, so it is not a numpy/scipy "Python
model" in the repository's sense, even though the driver is a Python module (`opendss_driver.py`). A numpy/scipy
port (the OpenDSS fixed-point current-injection solver with its line, transformer, source and load models) was not
attempted: the point of the study is the published feeder as OpenDSS solves it, and the comparison with the paper
below shows that OpenDSS's load-model details matter. The data were generated in a Linux container with the
provenance recorded below. The shipped CSVs are the record. **Regenerating the data, or rerunning the accuracy
checks, needs OpenDSSDirect.py 0.9.4** (`pip install opendssdirect.py==0.9.4`, plus scipy): `generate_data.py` and
`check_accuracy.py` stop with an error that says what to install when it is missing. The notebook needs only numpy,
scipy, matplotlib and active_subspaces, because it reads the shipped files and the driver's constants.

## Files

| file | contents | role |
|---|---|---|
| `European_LV_Feeder_Probabilistic_Power_Flow.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2017; outputs committed |
| `opendss_driver.py` | `FeederModel` (OpenDSS driver, not a numpy/scipy model: `solve`, `qoi55`, `qoi110`, `grad`, `load_voltages_pu`); constants `LOAD_NAMES`, `LOAD_BUSES`, `LOAD_PHASES`, `P_REF`, `SD_P`, `PF_NOM`, `PF_LO`, `PF_HI`, `NAMES55/110`, `MEAN55/110`, `SD55/110`, `STEPS55/110`, `QOI_NAMES` | the model; its constants import without OpenDSS |
| `feeder/` | the IEEE European LV Test Feeder in OpenDSS format, unchanged (63 files, 1.1 MB; see "License and credit") | model input |
| `feeder/LICENSE-electricdss-tst.txt` | BSD 3-clause licence of the repository the feeder files come from | covers `feeder/` |
| `generate_data.py` | writes all CSVs (`--case`, `--rows`, `--out` (required), `--check`) | regenerates the data (needs OpenDSSDirect.py 0.9.4) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" below (`--quick`: the nominal point and the tolerance check only) | accuracy check (needs OpenDSSDirect.py 0.9.4) |
| `inputs_55.csv` | 1000 × 55: P_LOAD1..P_LOAD55 in kW | Monte Carlo inputs, base scenario, seed 2017055 |
| `outputs_55.csv` | 1000 × 3: V899A (V), I78A (A), PLOSS (kW) | model output |
| `grad_V899A_55.csv`, `grad_I78A_55.csv`, `grad_PLOSS_55.csv` | 1000 × 55 each: ∂QoI/∂P_i in V/kW, A/kW, kW/kW | model output |
| `inputs_110.csv` | 1000 × 110: P_LOAD1..P_LOAD55 (kW), pf_LOAD1..pf_LOAD55 (–) | Monte Carlo inputs, 110-input scenario, seed 2017110 |
| `outputs_110.csv` | 1000 × 3, as above | model output |
| `grad_V899A_110.csv`, `grad_I78A_110.csv`, `grad_PLOSS_110.csv` | 1000 × 110 each: ∂QoI/∂x_j in QoI units per kW (P) or per unit pf | model output |
| `lhs_inputs_55.csv` | 10 000 × 55: P_i in kW (7.2 MB, the largest file) | Latin hypercube inputs, base scenario, seed 2017100 |
| `lhs_outputs_55.csv` | 10 000 × 3, as above (no gradients) | model output |
| `nominal.csv` | 2 × 6: pf, V899A, V899B, V899C (V), I78A (A), PLOSS (kW) at P_i = P_Li, pf 0.95 and 0.90 | model output |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the written 10 decimals before the model runs, so each
output belongs exactly to the stored input row.

## Model

### Citation

F. Ni, P. H. Nguyen, and J. F. G. Cobben, "Basis-Adaptive Sparse Polynomial Chaos Expansion for Probabilistic
Power Flow", *IEEE Transactions on Power Systems* 32(1) (2017) 694–704, doi:10.1109/TPWRS.2016.2558622
(Crossref-checked). The PDF read is the publisher's version.

### Feeder and solver

- The IEEE European LV Test Feeder (paper pp. 699–700, Fig. 2): radial, 50 Hz, 906 buses, 55 single-phase
  residential loads, 11/0.416 kV transformer, external MV grid "modelled as a voltage source with an impedance",
  data from the IEEE PES test-feeder files (paper ref. [33]). The paper used OpenDSS driven from MATLAB R2014a
  (p. 699).
- Files used: `feeder/`, from dss-extensions/electricdss-tst, `Version8/Distrib/IEEETestCases/LVTestCase`, commit
  3b208397160213cae4a9e2d0a7d1aa3528ce26e1 (2024-06-24), byte for byte. Only the 55 load profiles that
  `LoadShapes.txt` references are kept (of 100), and `Generators.txt` (not used by `Master.dss`) is left out.
  sha256: `Master.dss` 52a8cc42b34f…c8e4, `Lines.txt` adc097cf570a…311d, `Loads.txt` c07b99a8b8eb…b7f6e,
  `LineCode.txt` 0702d4a8a0b7…4da4, `Transformers.txt` d951d5bd01b1…d1d1, `LoadShapes.txt` 816b2baf9764…5dea.
- As in `Master.dss`: source 11 kV at 1.05 p.u. (ISC3 = 3000 A, ISC1 = 5 A), transformer 800 kVA delta-wye,
  XHL 4 %, 905 three-phase lines given by sequence impedances (`LineCode.txt`), 55 single-phase loads, kV 0.23,
  OpenDSS load model 1 (constant P and Q) with the default Vminpu 0.95 and Vmaxpu 1.05.
- The driver compiles `Master.dss`, switches to a snapshot solution, detaches the load shapes and sets each load's
  kW to the value of its shape at point 566 (0-based index 565), the on-peak minute 09:26 (paper p. 700: "the
  on-peak moment, i.e., t = 09:26:00"). Point 566 is also where the feeder's total demand peaks (57.358 kW). A
  yearly-mode solve of the unchanged `Master.dss` over 566 one-minute steps (ending at hour 9.4333) gives the same
  bus-899 voltages, 250.365, 238.421 and 253.414 V, to 6.0e-4 V, within OpenDSS's default tolerance used in that
  solve (`check_accuracy.py`, check 6).
- Solver: OpenDSSDirect.py 0.9.4 (DSS-Python 0.15.7, DSS C-API 0.14.5, based on OpenDSS SVN 3723), snapshot
  power flow, tolerance tightened from the default 1e-4 to 1e-12, at most 100 iterations (8–10 are used). Each
  solve starts from the previous solution. One solve takes about 1.5 ms (`generate_data.py`: 160.0 s for the 1000 × 111 solves of the 55-input case, below).
- **Load model above 1.05 p.u.** OpenDSS's constant-P load becomes a constant impedance (sized to draw the
  specified power at 1.05 p.u. of 230 V) when its voltage exceeds 1.05 p.u. With the 1.05 p.u. source, 43 of the
  55 loads are above 1.05 p.u. at the nominal point (`check_accuracy.py`). The paper says the loads are "modelled as
  constant power" (p. 699). We keep OpenDSS's default, the setting of the distributed files. This is this data set's
  choice, made among the possible readings by matching the paper's Table VIII; constant power at all voltages does
  not match it, and its nominal V899A (250.626 V) is off the median of the paper's Fig. 5 ("How these files relate to
  the paper"). The switch makes the responses piecewise smooth; see Accuracy for its effect on the finite differences.

### Quantities of interest

- `V899A`: phase-A (node 1) voltage magnitude at bus 899, V, phase to ground (the paper's main QoI, p. 700; its
  Figs. 3–5 are in volts).
- `I78A`: phase-A current magnitude in `Line.LINE78` (bus 72 → 79) at its bus-72 end, A. The paper names the
  "current of line 78" (p. 701, Table V) without a phase or end; phase A and the bus-72 end are this data set's choice.
- `PLOSS`: total active power loss of the circuit (`Circuit.Losses`), lines and transformer, kW. The paper's
  "active power loss of the whole feeder" (p. 701) does not say whether the transformer is included; including it
  is this data set's choice.

### Inputs

| case | inputs | units | nominal | distribution | source |
|---|---|---|---|---|---|
| 55, 110 | P_LOAD1..P_LOAD55, active power of each load | kW | P_Li, load shape at 09:26 (0.028–12.659 kW; table in the notebook) | N(P_Li, 0.1²), independent | paper p. 700, Table I (p. 700) |
| 55 | power factor of every load | – | 0.95 (held) | – | paper p. 700 |
| 110 | pf_LOAD1..pf_LOAD55, power factor of each load (lagging) | – | mean 0.90 | U(0.85, 0.95), independent | paper p. 701, Table I |

- **Standard deviation of P.** The paper prints N(PL_i, 0.1²) (p. 700; the rendered formula, whose text layer reads
  "0.12"). The second argument is a variance in the units of PL_i, which are kW, so the sd is 0.1 kW for every
  load. A 10 % relative sd would be written N(PL_i, (0.1 PL_i)²). The absolute reading was chosen among the alternatives
  by matching Table VIII, and the spread of Fig. 4 supports it independently (below). The same sentence says "i = 1, …, 50"; that is a misprint for 55 (Table I and the text count 55
  loads).
- **110 inputs.** The text (p. 701) and Table I give 55 Gaussian active powers and 55 uniform power factors; Table IV
  labels the 55-, 110- and 140-input rows "Mixed" (30 Gaussian and 25 uniform active powers, p. 701). We follow the
  text and Table I.
- **Not built:** the 140-input case (30 PVs "randomly installed at buses with loads", p. 699; the buses are not
  listed) and the IEEE 123-node feeder cases (distribution parameters not printed, Table I).
- **Samples outside physical meaning.** Many loads draw less than 0.1 kW at 09:26, so N(P_Li, 0.1²) gives negative
  powers: 11.89 % of the 55-input values, 11.83 % of the 110-input P values and 11.97 % of the Latin hypercube
  values (11.97 % expected), and every row has at least one. A negative P is an injection (with pf 0.95 its reactive
  power is negative too). In the Latin hypercube set a row has 6.58 negative loads on average, and the smallest value
  is −0.460 kW (`check_accuracy.py`, check 5). They are kept: they are the paper's distribution, and the paper's runs
  evidently kept them too, because setting them to 0 or truncating the normal at 0 breaks the match with Table VIII
  ("How these files relate to the paper"). The notebook counts them.
- The notebook scales every input by its standard deviation (0.1 kW; 0.1/√12 = 0.02887 for pf), through
  `ac.Gaussian(mean, variances)`. For the 110-input case this object is used only for the scaling; the pf samples
  are uniform. With the half-range scaling of the pf inputs the leading direction in physical units is unchanged
  (|cos| = 1.0000; notebook).

### Gradients

Central differences in the physical inputs, with step 0.01 of each input's standard deviation (`STEPS55`,
`STEPS110`): 1e-3 kW for P and 2.887e-4 for pf. That is 110 extra solves per 55-input sample and 220 per 110-input
sample. Coordinates: physical (V, A or kW per kW, or per unit pf). The notebook passes them to the library, which
multiplies them by the standard deviations.

### Sampling design

- `inputs_55.csv`, `inputs_110.csv`: plain Monte Carlo (independent draws with numpy's `default_rng`, seeds 2017055
  and 2017110), 1000 points each, with gradients. These are this data set's design; the paper builds no gradients.
- `lhs_inputs_55.csv`: Latin hypercube, 10 000 points (scipy `qmc.LatinHypercube` with a generator seeded 2017100,
  mapped through the normal inverse CDF), QoIs only. It mirrors the paper's reference, an LHS PPF with 10^5 trials
  (p. 699), at a tenth of the size.
- The paper's own designs: LHS reference with 10^5 trials; experimental designs of N_ed = 800 (base configuration)
  or 250 ("efficient") runs for the surrogate (p. 700). No seeds are reported.

### Published results (paper)

- Table II (p. 700): relative LOO error e_LOO of the sparse PCE of V899A, 55 Gaussian inputs: 8.81e-7 (q = 0.8,
  p = 1–6, N_ed = 800) to 2.36e-6 (N_ed = 250). Table III: 2.36e-6, 4.11e-6, 3.72e-6 for Gaussian, uniform and mixed
  inputs. Table IV: 3.72e-6 (55, Mixed), 3.67e-3 (110), 3.90e-3 (140). Table V: e_LOO 2.36e-6 (voltage), 2.08e-6
  (current), 3.97e-4 (power loss).
- Fig. 3 (p. 702): histograms of the three phase voltages at bus 899, peaks near 250.4 (A), 238.4 (B) and 253.4 V (C),
  read from the figure. Fig. 4: histogram of V899A on 249.6–251.2 V. Fig. 5 and Table VIII: P(U_A ≤ 250) = 4.06 %,
  P(250.4 < U_A ≤ 250.8) = 41.57 %, P(U_A > 250.8) = 1.94 % (LHS, 10^5). Table VII: relative errors of the moments
  (the moments themselves are not printed). Table IX (p. 703), LHS with N_sam = 10^5: t_pre = 0.17 s,
  t_post = 3561.86 s, T_total = 3562.03 s (about 36 ms per power flow from T_total). The text (p. 699: "In the LHS
  based PPF, t_pre = t_sam + N_sam × t0") puts the power flows in t_pre, so the table and the text disagree about
  which stage holds them.
- Table II prints e_LOO with the unit "(p.u.)"; it is the relative leave-one-out error of Eq. (13) (p. 697), divided
  by Σ(f − f̄)², not a per-unit voltage.
- No active subspace analysis.

### What the notebook finds (from these files)

- Base scenario: one active direction for each QoI, λ1/λ2 = 2.22e6 (V899A), 8.49e6 (I78A) and 4.0e3 (PLOSS); a linear
  function of y1 has leave-one-out R² 0.9999987, 0.9999989 and 0.9996878. These very large ratios reflect
  near-linearity, not nonlinear low-dimensional structure: the gradient changes little over the sample (median
  deviation from the mean gradient 1.2e-3 of its norm for V899A), and w1 is the least-squares slope direction
  (|cos| = 1.0000000, 1.0000000 and 0.9999901 for V899A, I78A, PLOSS; notebook). An ordinary regression on the
  inputs finds the same direction. The 110-input case is the more informative one. The V899A direction is dominated by the phase-A loads near bus 899 (negative), with the opposite sign
  on the phase-B and C loads; I78A weights the 17 downstream phase-A loads almost equally; PLOSS is led by phase B.
- 110 inputs: λ1/λ2 = 668.7, 2531.4 and 549.2; 1 − LOO R² of a linear function of y1 is 1.36e-3, 9.77e-4 and
  1.98e-3. The power factors of the largest loads (LOAD26 on phase B, LOAD29 on phase A) take large weights.
- Eigenvector signs: `ac.orient(..., by="mean_gradient")`, so each QoI increases along w1. The paper has no
  eigenvector figures to follow.

## Regenerating

```bash
# OpenDSS environment (Python 3.13 was used)
python3 -m venv dssvenv && dssvenv/bin/pip install opendssdirect.py==0.9.4 numpy scipy
dssvenv/bin/python generate_data.py --out /tmp/lv_full                       # all files (about 8 min)
dssvenv/bin/python generate_data.py --rows 0:20 --out /tmp/lv --check         # subset vs shipped files
dssvenv/bin/python check_accuracy.py                                          # the Accuracy numbers (about 1 min of CPU)
```

`generate_data.py` writes only into `--out`, which it requires and which must be outside this directory (it
refuses the directory and its subdirectories in every mode). The
tolerance of `--check` is 1e-8 (outputs) and 1e-5 (gradients) of the largest stored value of each file.

Provenance of the shipped files: generated 2026-11-03 in a Linux container (x86-64, 2 CPUs shared with other jobs,
run at `nice 10` in one process), Python 3.13.18, OpenDSSDirect.py 0.9.4 (DSS-Python 0.15.7, dss_python_backend
0.14.5, DSS C-API 0.14.5 based on OpenDSS SVN 3723), numpy 2.5.4, scipy 1.18.1, with `generate_data.py` as shipped.
Run time of `generate_data.py`: 160.0 s (55-input case), 315.9 s (110-input case), 14.6 s (Latin hypercube) and
under 1 s (nominal), 8 min 13 s of wall time and 407 s of CPU in all. There, `--rows 990:1000 --check` reproduced
the last 10 rows of all files of all cases to at most 1.2e-13 of the largest stored value (gradients: difference 0). The other rows were not rerun.

## Accuracy

`check_accuracy.py` prints every number in this list and the numbers marked `check_accuracy.py` elsewhere in this
file (it needs OpenDSSDirect.py; with 0.9.4 in the container above, 2026-11-03, it ran in 66 s of CPU time, 3 min 27 s
of wall time on CPUs shared with other jobs). It uses shipped rows 0, 1 and 2 of each case for the gradient checks and
all 1000 rows of the 55-input case for the load-switch check.

- **Solver tolerance** (`check_accuracy.py`). At the nominal point, a warm start (from 1.2 P_L) and a cold start
  agree to 6.7e-13 relative. With OpenDSS's default tolerance 1e-4, V899A moves by +9.8e-5 V, I78A by +2.5e-6 A and
  PLOSS by +6.4e-5 kW, which is 5e-4 of the sd of V899A (0.21 V).
- **Gradients** (`check_accuracy.py`; relative 2-norm difference of the gradient in sd-scaled coordinates, against the
  shipped step 0.01 sd). Base scenario, rows 0 and 2: 6e-8 to 3e-7 for a step of 0.001 sd, 6e-9 to 5e-8 for 0.1 sd,
  and 6e-9 to 3e-8 for a fourth-order formula with step 0.1 sd. Row 1: 8e-8 to 3e-7 for 0.001 sd, but 2e-7 to 1.8e-5
  for 0.1 sd and 5e-7 to 4.5e-5 for the fourth-order formula, most likely because the wider stencil crosses a load's
  1.05 p.u. switch (the shipped step is ten times smaller). 110 inputs, rows 0–2: 2e-7 to 2.3e-6 for 0.001 sd, 2e-5 to 2.3e-4 for 0.1 sd (curvature in pf), and
  2e-7 to 3.8e-6 for the fourth-order formula. The shipped gradients are therefore accurate to about 1e-7 (base) and
  1e-6 (110 inputs) relative.
- **Load-model switch** (`check_accuracy.py`). At the nominal point 43 of 55 loads are above 1.05 p.u. (load voltages
  1.0366–1.1071 p.u.); over the 1000 base-scenario rows 43–44 are. The smallest distance of any load voltage from
  1.05 p.u. is 2.9e-6 p.u.; 5 rows come within 1e-5 p.u. and 17 within 1e-4 p.u. At those 5 rows the shipped gradients agree with a step of 0.001 sd to 9.3e-8, 6.6e-8 and
  3.3e-7 (V899A, I78A, PLOSS), so at those rows the switch does not show in the shipped gradients beyond the
  finite-difference error.
- In the notebook, the least-squares slope of each QoI on the standardized inputs and the mean standardized gradient
  agree to a cosine of 1.000000 (base-scenario V899A and I78A) and at least 0.999831 (every case), which confirms
  the gradient coordinates.

## How these files relate to the paper

- **A new computation on the same feeder, not the paper's runs.** The paper's samples were not published. These files
  come from the same OpenDSS feeder files and the paper's stated distributions, at the stated time snapshot, with a
  newer OpenDSS (DSS C-API) and a tighter solver tolerance.
- **Agreement.** The nominal three-phase voltages at bus 899, 250.365, 238.421 and 253.414 V (`nominal.csv`), sit where
  the paper's Fig. 3 histograms peak. On the 10 000 Latin hypercube points, P(U_A ≤ 250), P(250.4 < U_A ≤ 250.8) and
  P(U_A > 250.8) are 4.13 % [3.76, 4.54], 41.21 % [40.25, 42.18] and 2.18 % [1.91, 2.49] (binomial 95 % intervals;
  notebook) against the paper's 4.06 %, 41.57 % and 1.94 % (Table VIII); all three paper values are inside the
  intervals. The range, 249.56–251.17 V, matches the span of Fig. 4.
- **How the two open settings were chosen.** The sd of the load powers (0.1 kW or 0.1 P_Li) and the load model
  (OpenDSS's default switch to constant impedance above 1.05 p.u., or constant power at every voltage) were chosen
  among the four combinations by matching Table VIII, so the agreement above is not an independent confirmation.
  `check_accuracy.py` (check 4; 2000 Monte Carlo samples, seed 1) gives P(U_A ≤ 250), P(250.4 < U_A ≤ 250.8),
  P(U_A > 250.8):

  | sd of P | load model | nominal V899A | sd of V899A | probabilities |
  |---|---|---|---|---|
  | 0.1 kW | default switch (Vmaxpu 1.05), used here | 250.365 V | 0.206 V | 4.50 / 40.40 / 1.70 % |
  | 0.1 P_Li | default switch | 250.365 V | 0.460 V | 22.85 / 29.85 / 16.75 % |
  | 0.1 kW | constant power (Vminpu 0.8, Vmaxpu 1.2) | 250.626 V | 0.204 V | 0.05 / 67.15 / 18.45 % |
  | 0.1 P_Li | constant power | 250.626 V | 0.488 V | 10.60 / 32.55 / 35.00 % |

  The chosen readings are consistent with Figs. 4–5, which support them independently of Table VIII: the relative sd
  gives an sd of V899A of 0.460 V, too wide for the span of Fig. 4 (axis 249.6–251.2 V; the chosen reading has
  0.2119 V on the LHS set), and the constant-power reading has a nominal V899A of 250.626 V, while the CDF of Fig. 5
  crosses 0.5 near 250.37 V (read from the figure).
- **Negative load powers** (`check_accuracy.py`, check 5, all 10 000 Latin hypercube rows). As shipped, the three
  probabilities are 4.13 / 41.21 / 2.18 %. With every negative power set to 0 they become 6.07 / 27.88 / 0.27 % (mean
  V899A shifted by −0.0743 V), and with the same stratified uniforms mapped through the normal truncated at 0,
  17.90 / 10.52 / 0.00 % (shift −0.1956 V). Table VIII matches only when the negative powers (injections) are kept, so
  the paper's runs evidently kept them. Table VIII is this sensitive because it pins the mean of V899A to a few
  hundredths of a volt: shifting the LHS voltages by ±0.02 V moves the probabilities outside the binomial intervals
  (notebook).
- **Surrogate accuracy.** The paper's e_LOO is the relative leave-one-out error of its PCE (Eq. 13, p. 697),
  1 − R²_LOO. A linear
  function of the one active variable has 1 − R²_LOO = 1.31e-6 (V899A), 1.09e-6 (I78A), 3.12e-4 (PLOSS) in the base
  scenario (paper Table V, 250 runs: 2.36e-6, 2.08e-6, 3.97e-4) and 1.36e-3 for V899A with 110 inputs (paper Table IV,
  "Mixed": 3.67e-3). The designs and sample sizes differ, so this is a comparison of orders of magnitude.
- The paper's moments (Table VII) are printed only as relative errors, so there is nothing to compare with the
  notebook's mean 250.3657 V and sd 0.2119 V of V899A.

## Unknowns

1. The units of the load sd are not printed; 0.1 kW is read from the printed variance, selected among four
   readings by Table VIII, and consistent with Figs. 4–5 (above).
2. The paper says "constant power"; OpenDSS's default switch to constant impedance above 1.05 p.u., which the paper
   does not mention, was selected by Table VIII and is consistent with Fig. 5.
3. Whether the paper's 10^5 LHS runs kept the negative load powers: Table VIII matches only if they did (above).
4. The phase and end of the "current of line 78", and whether the "active power loss of the whole feeder" includes
   the transformer.
5. The 110-input case: Gaussian active powers (text, Table I) or "Mixed" (Table IV).
6. The PV buses of the 140-input case (not built).
7. The OpenDSS version and solver tolerance the paper used (not stated; the default tolerance changes V899A by
   about 1e-4 V).
8. Table IX (p. 703) lists 3561.86 s for LHS as t_post and 0.17 s as t_pre, while the text (p. 699) puts the
   N_sam power flows in t_pre.

## License and credit

Code (`*.py`, the notebook): MIT. Data (the CSVs): CC BY 4.0. The feeder files in `feeder/` are the IEEE European LV
Test Feeder of the IEEE PES Distribution System Analysis Subcommittee (test feeders described by K. P. Schneider et
al., IEEE Trans. Power Syst. 33(3) (2018) 3181–3188, doi:10.1109/TPWRS.2017.2760011), as distributed by EPRI and the
DSS-Extensions contributors in electricdss-tst under the BSD 3-clause licence (`feeder/LICENSE-electricdss-tst.txt`,
kept beside the files); the IEEE PES pages state no licence for the feeder data. OpenDSS: EPRI (R. C. Dugan and
T. E. McDermott, IEEE PES General Meeting 2011, doi:10.1109/PES.2011.6039829), BSD licence; OpenDSSDirect.py and
DSS C-API: DSS-Extensions, BSD 3-clause. Neither is shipped. Study design: F. Ni, P. H. Nguyen and J. F. G. Cobben;
cite their paper (above).
