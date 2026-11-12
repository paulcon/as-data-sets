# Rabbit_Ventricular_Action_Potential: data

Kind: **stored data**. The model needs Myokit (a Python package that compiles the model to C and
integrates it with SUNDIALS CVODES), so it is not a numpy/scipy model. `generate_data.py` and
`shannon_driver.py` ran it in a Linux container, where the shipped `rabbit_ap_samples.csv` was written;
the CSV is the record. Myokit is not installed in this repository's `.venv`, so the scripts that need it
(`generate_data.py` except `--merge`, `check_accuracy.py`) stop there with an error that says what to
install. The notebook needs only the CSV.

## Files

| file | contents | role |
|---|---|---|
| `Rabbit_Ventricular_Action_Potential.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` (Python 3.11) with seed 328261; outputs committed |
| `rabbit_ap_samples.csv` | 320 rows × 34 columns (below) | stored model runs |
| `shannon_driver.py` | `run(p, n_beats, ...)`, `biomarkers(t, V, Cai)`; `INPUTS`, `NAMES`, `LOWER`, `UPPER` | Myokit wrapper; the constants need only numpy |
| `generate_data.py` | the design and the runs; `--chunk`, `--merge`, `--check` | regenerates the CSV (needs Myokit) |
| `check_accuracy.py` | reruns: row 0, pNaK one at a time, the fixed E_Na_SL, the near-threshold stimulus, the solver tolerance | the accuracy numbers not printed elsewhere (needs Myokit) |
| `shannon_wang_puglisi_weber_bers_2004_a_p.mmt` | the authors' Myokit model file, copied unchanged: Shannon et al. 2004 with E_Na_SL fixed at −15 mV, as distributed by Yang et al. (below) | the model |

Columns of `rabbit_ap_samples.csv` (two `#` lines, then numbers; `np.loadtxt(path, delimiter=",")`):

- 0–14: the factors `pNaK, pK1, pCaL, pCap, pNa, pKr, pNaCa, pKs, pClCa, pNab, ptos, pClb, pCab, pKp, ptof` (dimensionless, physical values);
- 15–22: biomarkers of beat 1000: `Vmax` (mV), `Vrest` (mV), `A90` (ms), `A30` (ms), `Vplt` (mV), `B` (mV ms), `Cai_ED` (mM), `Cai_PSV` (mM);
- 23: `A90_prev`, A90 of beat 999 (ms);
- 24–31: the same 8 biomarkers of beat 500 (`b500_*`), a convergence check;
- 32: `response`: 0 normal 1:1, 1 no AP, 2 alternans or 2:1 (rule below);
- 33: `cpu_s`, CPU time of the run (s).

## Model

**Citation.** Z. Yang, H. Gao, G. L. Smith, R. D. Simitev, Dominant ionic currents in rabbit
ventricular action potential dynamics, *PLOS ONE* 20(7):e0328261, 2025,
doi:10.1371/journal.pone.0328261. Model: T. R. Shannon, F. Wang, J. Puglisi, C. Weber, D. M. Bers,
*Biophysical Journal* 87(5):3351–3371, 2004, doi:10.1529/biophysj.104.047449. PDFs read: the publisher's versions of both.

**Equations.** The Shannon et al. rabbit ventricular myocyte (39 ODE states in the Myokit file: membrane
potential, gates, junctional/subsarcolemmal/bulk Ca, SR Ca, Na and buffers). Each of the 15 currents is
multiplied by a factor p_i wherever it appears (paper Eq 1, p. 5); the `.mmt` file holds them as constants
`<current>.p`. Stimulus: 9.5 A/F at a basic cycle length of 500 ms (paper p. 10); its duration, 5 ms, and
its start, 76 ms into each beat, come from the protocol in the `.mmt` file. The paper notes that the
excitation threshold at baseline is about 9.4 A/F (p. 11).

**The model file is Shannon et al. 2004 with E_Na_SL fixed at −15 mV, as distributed by Yang et al.**
It is Myokit's import of the CellML encoding of the model (header "Myokit CellML 1 API"), with these edits,
none of which the paper mentions:
- `reversal_potentials.E_Na_SL = -15.0` (mV), with the Nernst form
  `R*T/F*log(Nao/Na_SL)` commented out on the next line. In the published model the Na reversal potential of
  each compartment is the Nernst potential with that compartment's [Na] (Shannon et al. 2004, p. 3363:
  "For each compartment C, the Nerst [sic] potential E_C is calculated as", followed by Eq 10,
  E_Na = (RT/F) ln([Na]o/[Na]C); p. 3364:
  "junctional Na channels (11% of the total) were dependent upon extracellular [Na] and junctional [Na]").
  E_Na_SL drives the subsarcolemmal parts of I_Na and I_Nab, which carry 89 % of these currents
  (`Fx_Na_SL` = `Fx_NaBk_SL` = 0.89); the junctional E_Na_jct is still the Nernst form.
- `model_parameters.p1`–`p5` multiply R, T, F, [Cl]i (15 mM) and [Cl]o (150 mM); all are 1, so they change
  nothing.
- The protocol line that starts the stimulus at 20 ms is commented out and replaced by one that starts it
  at 76 ms.

The paper says that the factors p_i "allow the original model formulation to be used without
modification" (p. 5) and that it uses "a machine readable model specification file available from the
CellML model repository" (p. 5). Its Fig 2(d) (p. 12) shows a baseline AP peaking near 26 mV, as this file
does, so the paper's runs used the modified file too. We keep it, because it is what the paper ran. The fixed E_Na_SL matters: see "Accuracy" (check 3 of `check_accuracy.py`). Whether it
was intended is asked of the authors (open).

**Inputs** (paper Table 4, p. 12, "parameter subspace of normal response"; uniform on a linear scale, as
the paper's SALib Saltelli sample; nominal 1 = the model file unscaled):

| name | scales | range | source |
|---|---|---|---|
| pNaK | Na-K pump current I_NaK | [0.0001, 1.07] | Table 4, p. 12 |
| pK1 | inward rectifier K current I_K1 | [0.03, 1.01] | Table 4, p. 12 |
| pCaL | L-type Ca current I_CaL | [0.7, 1.6] | Table 4, p. 12 |
| pCap | sarcolemmal Ca pump I_Cap | [0.0001, 4.5] | Table 4, p. 12 |
| pNa | fast Na current I_Na | [1, 10] | Table 4, p. 12 |
| pKr | fast delayed rectifier I_Kr | [0.0001, 3.2] | Table 4, p. 12 |
| pNaCa | Na/Ca exchanger I_NaCa | [0.6, 3] | Table 4, p. 12 |
| pKs | slow delayed rectifier I_Ks | [0.0001, 10] | Table 4, p. 12 |
| pClCa | Ca-activated Cl current I_ClCa | [0.0001, 10] | Table 4, p. 12 |
| pNab | background Na current I_Nab | [0.0001, 10] | Table 4, p. 12 |
| ptos | slow transient outward I_tos | [0.0001, 2.5] | Table 4, p. 12 |
| pClb | background Cl current I_Clb | [0.0001, 10] | Table 4, p. 12 |
| pCab | background Ca current I_Cab | [0.34, 10] | Table 4, p. 12 |
| pKp | plateau K current I_Kp | [0.0001, 10] | Table 4, p. 12 |
| ptof | fast transient outward I_tof | [0.0001, 4] | Table 4, p. 12 |

The authors' `data_generate.py` uses 1.02 as the upper bound of pK1; we use Table 4's 1.01.

**Quantities of interest** (paper Table 3, p. 6, last beat of the train; t0 = 76 ms, t_BCL = 500 ms):
Vmax = max V; Vrest = V(t_BCL), the last logged point of the beat; A90 (A30) = time from the stimulus
onset to the first crossing of Vmax − 0.9 (0.3)·(Vmax − Vrest) after the peak, linearly interpolated;
Vplt = V(t0 + A90/2); B = trapezoid integral of |V − Vrest| over the beat; plus end-diastolic (at the
stimulus onset) and peak cytosolic [Ca2+]i, two of the paper's four Ca biomarkers (pp. 18–19). V and
[Ca]i are logged every 0.1 ms. The authors' `output_for_AP_features.py` differs in detail: it measures
A90 between the upstroke and downstroke crossings of min V + 0.1 (max V − min V), takes Vrest as the mean
of the last 50 ms, and integrates |V| rather than |V − Vrest|.

**Response classes** (all runs kept in the file; the notebook uses response 0 only): 1 = no AP (Vmax ≤ 0 mV
in beat 999 or 1000, or no repolarization crossing); 2 = alternans or 2:1 (A90 of beats 999 and 1000
differ by more than 2 %). Counts: 288 normal, 16 no AP, 16 alternans or 2:1. The excluded rows have
low pump factors (median pNaK 0.14 against 0.59 for normal rows). Table 4 was found one factor at a
time, so joint corners of the box can leave the normal region.

*Flagged normal rows (printed by the notebook; the CSV is unchanged, the rule is kept, and the rows stay in
the analysis):* the rule admits some pathological cells. 23 of the 288 normal rows have an end-diastolic
[Ca]i above 1 µM (`Cai_ED` > 1e-3 mM; up to 6.5 µM, against a median of 0.114 µM over the normal rows):
Ca overload. 15 have a peak potential below 5 mV (down to 0.11 mV), with A90 of 32–103 ms, the same kind of
response as the no-AP rows just below 0 mV. 10 rows have both, 28 either, and all have pNaK ≤ 0.32 (Na and Ca
loading with a weak pump). The notebook flags them by these two thresholds (check (e)): without the 28 rows,
pNaK still leads w1 of Vmax, A30, Vplt and B, and ties with pK1 for A90 (pK1 −0.53, pNaK +0.52, pClb −0.49);
on the 234 rows with Vmax ≥ 10 mV, pNaK leads A90 (+0.53, pK1 −0.52). So the pNaK headline does not come from
these rows.

*Limitations of the rule (printed by the notebook):*
- **It biases the design.** The 32 excluded runs are 24 of the 64 runs in the lowest fifth of the pNaK
  range (pNaK < 0.216), against 7, 0, 1 and 0 in the other fifths. In the kept rows the normalized pNaK
  has kurtosis 1.93 (Kolmogorov-Smirnov p = 0.071 against uniform); the other 14 inputs have 1.78-1.82
  (uniform: 1.80). QPHD averages the fitted quadratic's gradient outer product over the full box with
  Σ = I/3, so it extrapolates the quadratic into the dropped 10 %, mostly where pNaK acts.
  Sensitivity check: on all 320 rows pNaK still leads w1 of A90 (+0.61) and Vmax (+0.84) and pK1 leads
  Vrest, but A30 and B switch to pClb (−0.62, −0.68), with pNaK second.
- **It reads only beats 999 and 1000**, so it can miss the paper's "intermittent" response (p. 11 and
  Fig 2(e): about 20 normal APs, then an equally long phase without APs) whenever those two beats fall
  in an AP phase. One kept row has a failed beat 500 (`b500_Vmax` < 0).

**Gradients:** none. The notebook uses the global quadratic estimator (QPHD) on values.

**Sampling design.** The first 320 points of one scrambled Sobol' sequence of 2^9 = 512 points in 15
dimensions (`scipy.stats.qmc.Sobol(d=15, scramble=True, seed=20251031)`), mapped linearly to the box.
The paper used SALib's Saltelli design with base size M = 8192 (8192 × 32 = 262,144 runs, from the
authors' `data_generate.py`; the paper states M = 8192, p. 10) and the Jansen estimator; its outputs are
not published.

**Published results.** No active subspace. Sobol' indices: Table 5 (p. 14, top three first-order and
total indices per biomarker), Table 6 (second order), grand total ranking Eq (9) (p. 14) and Fig 5;
reduced models Fig 7 and Table 7 (p. 21; six factors give R² = 0.93 for A90), evaluated on L = 2000
quasi-random full-model runs (Eq 8, p. 9) whose outputs, but not inputs, are in the authors'
repository. The notebook compares.

## Regenerating

```bash
pip install myokit numpy scipy        # plus SUNDIALS, e.g. apt install libsundials-dev, and a C compiler
nice -n 10 python generate_data.py   # 4 chunks of 80 points (finished chunks skipped), then merge
python generate_data.py --chunk 2    # one chunk -> parts/chunk_2.npz (saves progress every 5 points)
python generate_data.py --merge      # parts -> rabbit_ap_samples.csv
python generate_data.py --check 3    # rerun the first 3 points and compare with the CSV
```

Without Myokit, SUNDIALS and a C compiler, `generate_data.py` (except `--merge`) stops with an error that
says what to install and to rerun it. The shipped file is the record; it is regenerated only if a problem
is found in it.

Provenance of the shipped file: 2026-10-31, written in an Ubuntu 24.04 container, Python 3.13.18, Myokit 1.39.2,
SUNDIALS 6.4.1 (Ubuntu `libsundials-dev`), numpy 2.5.4, scipy 1.18.1; CVODES tolerances abs 1e-6,
rel 1e-4 (Myokit defaults; the paper states none); one process at `nice -n 10` on 2 shared CPUs.
**Run time of `generate_data.py`:** 9.8 s CPU per run on average (6.9–13.6 s, printed by the notebook),
52.2 min CPU for the 320 runs, 4473 s (75 min) wall time (original run in the container). **Check:**
`--check 2` (original run in the container, 2026-11-01) reproduced rows 0 and 1 to a relative 4.2e-12 (the
12 significant digits written) in 61 s. It cannot run in `.venv` (no Myokit).

**Pacing sequence.** From the model file's initial state: 499 unlogged beats (Myokit `pre`), beat 500
logged, 498 unlogged beats, beats 999 and 1000 logged. `shannon_driver.run` restores the initial state
explicitly before each run, because Myokit's `pre()` replaces the simulation's default state (without
this, a cached simulation would start each run from the previous run's paced state).

## Accuracy

The checks that rerun the model need Myokit, which is not installed in `.venv`. Numbers marked "CVODES run"
come from `check_accuracy.py` checks 1-5 with Myokit 1.39.2 and SUNDIALS 6.4.1 (Ubuntu 24.04 `libsundials-dev`;
Python 3.12.3, numpy 2.4.6, scipy 1.17.1; an AWS c7a.4xlarge, 2026-11-02; 257 s wall), the solver of the
shipped data. Checks 4 and 5 there reproduced the original run's numbers quoted below exactly. Numbers
marked "scipy run" come from `check_accuracy.py --scipy` (Myokit 1.39.2 parses the model file, scipy
1.17.1's LSODA integrates it, rtol 1e-6, atol 1e-8; Python 3.11 on macOS, a separate environment from
`.venv`, 2026-11-02), an integrator independent of the shipped data's.

- **Convergence of the 1000-beat train** (normal rows; change d = biomarker at beat 1000 − at beat 500;
  printed by the notebook):

  | QoI | median abs(d) | 90th pct | max | sd over rows | median abs(d)/sd | corr(beat 500, beat 1000) |
  |---|---|---|---|---|---|---|
  | Vmax (mV) | 0.576 | 4.00 | 14.7 | 5.57 | 0.103 | 0.844 |
  | Vrest (mV) | 0.113 | 0.961 | 23.5 | 5.37 | 0.021 | 0.940 |
  | A90 (ms) | 3.24 | 16.8 | 99.1 | 32.3 | 0.100 | 0.919 |
  | A30 (ms) | 3.23 | 11.0 | 40.3 | 22.6 | 0.143 | 0.959 |
  | Vplt (mV) | 2.51 | 12.7 | 50.6 | 16.6 | 0.151 | 0.835 |
  | B (mV ms) | 326 | 1103 | 8460 | 2493 | 0.131 | 0.936 |
  | Cai_ED (mM) | 3.5e-6 | 4.6e-5 | 5.2e-3 | 8.0e-4 | 0.004 | 0.736 |
  | Cai_PSV (mM) | 4.3e-5 | 8.5e-4 | 6.5e-3 | 1.5e-3 | 0.029 | 0.927 |

  A90 changes by a median 3.2 % between beats 500 and 1000; 60 % of rows change by more than 2 %
  (the paper's Fig 1 shows changes below 2 % at one parameter point). The drift is concentrated
  where the Na-K pump is weak: correlation of the relative A90 change with pNaK −0.56; median 21.0 %
  for pNaK < 0.3 and 2.2 % for pNaK ≥ 0.3. So the biomarkers are those of the 1000th beat (the paper's
  definition), not of a steady state, for weak-pump rows. The drift goes one way and makes pNaK more
  important: the linear A90 slope on normalized pNaK grows from +19.0 ms (beat 500) to +32.0 ms (beat
  1000), while those on pClb (−28.1, −24.7) and pK1 (−25.1, −25.0) hardly change, and A90's w1 at beat
  500 is led by pClb, pK1 and pKr (notebook, check (a)). A longer train would make pNaK matter more, not
  less. The paper's Fig 1 convergence check is at one point, pNaK 0.5, pNaCa 0.3 and the other factors 1:
  pNaCa is outside the box ([0.6, 3]), and pNa = pNab = 1 while their means over the box are 5.5 and 5.0,
  so the check says little about Na loading in the box.
- **An independent integrator** (scipy run, `check_accuracy.py --scipy`, check 1). Row 0 rerun with scipy's
  LSODA instead of CVODES reproduces the CSV's eight biomarkers to a relative 4.8e-4 at beat 500 and 1.8e-3
  at beat 1000 (A90 86.4 ms, Vmax 17.3 mV, A30 47.5 ms, as in the CSV), 714 s CPU. The CVODES run of check 1
  (the shipped pacing sequence) agrees with the CSV to 5.6e-4 and 1.7e-3, the same size: these are the
  run-to-run differences of a 1000-beat train between platforms. So the CSV is a faithful run of the model
  file, and the biomarker code and pacing do what is stated.
- **pNaK one at a time** (CVODES run, check 2; the other 14 factors at 1). A90 (ms) and [Na]i at the end of the
  beat (mM; 8.81 initially):

  | beat | pNaK 0.1 | pNaK 0.5 | pNaK 1.0 |
  |---|---|---|---|
  | 5 | 308.5 (8.98) | 265.4 (8.93) | 231.1 (8.87) |
  | 20 | 292.1 (9.53) | 255.2 (9.28) | 228.8 (9.02) |
  | 100 | 208.5 (12.06) | 205.0 (10.73) | 207.8 (9.59) |
  | 500 | 118.1 (20.31) | 156.9 (13.33) | 198.7 (9.93) |
  | 1000 | 86.6 (26.94) | 152.1 (13.79) | 199.3 (9.90) |

  The paper's Fig 2(a) (p. 12; read by eye: about 290, 245 and 205 ms at pNaK 0.1, 0.5 and 1.0) falls with
  pNaK, as beats 5–20 do here; at beat 1000 a weak pump has loaded the cell with Na and gives the shortest
  APs, so the positive pNaK effect of the design is present one at a time too. The scipy run (LSODA) of the
  same check agrees to 0.1 ms through beat 100 and at pNaK 0.1, but at beat 1000 it gives a failed beat at
  pNaK 0.5 (Vmax −53.9 mV) and 204.7 ms at pNaK 1.0: near the stimulus threshold the late beats depend on the
  integrator (see "Near-threshold stimulus"). With CVODES every beat of the sweep is normal, as the paper
  reports for pNaK < 1.07.
- **The fixed E_Na_SL** (CVODES run, check 3), beat 1000: at the nominal point the file gives Vmax 25.8 mV,
  A90 199.3 ms and A30 152.7 ms; with the Nernst E_Na_SL of the published model, Vmax 42.9 mV, A90 188.7 ms and A30 108.5 ms
  ([Na]i 10.42 against 9.90 mM; the scipy run gives the same to within 0.4 mV and 5.4 ms). At row 0 the
  file gives Vmax 17.3 mV, A90 86.4 ms and A30 47.5 ms (the CSV's values); with the Nernst form, Vmax 49.2 mV, A90 73.8 ms and A30
  12.4 ms (the same in both runs). With CVODES checks 1-5 took 257 s; the scipy run of checks 1-3 took 66 min (about 6-12 min of CPU per 1000-beat run).
  The paper's Fig 2(d) (baseline AP peak near 26 mV) matches the file.
- **[Na]i loading** (original run; external, not reproducible from the repository, because the 14 rows
  were not recorded): 14 design rows spread over the normal rows (pNaK 0.24-1.03) were rerun with
  `shannon_driver` and logged at beats 1, 2, 5, 10, 20, 50, 100, 200, 500 and 1000 (about 150 s CPU).
  Beat 1000 matched the CSV (A90 to 0.02 ms). [Na]i at the end of beat 1000 was 12.0-25.8 mM (median
  15.9), against 8.81 mM in the initial state; median 9.2 mM at beat 10, 11.8 at beat 100 and 15.2 at
  beat 500. Check 2 below shows the same loading one factor at a time.
- **Near-threshold stimulus** (original run, reproduced exactly by the CVODES run of check 4). At the nominal point (all p = 1) the 9.5 A/F stimulus occasionally fails,
  and which beat fails depends on how the integration is split into `pre`/`run` calls: one 498-beat
  `pre` gives a failed beat 499 (A90 19.4 ms) and recovering beats 152.8 and 178.9 ms; 250 + 249 beats
  give 198.2 ms. With the shipped sequence, beats 999 and 1000 at the nominal point are normal (198.7
  ms) but beat 500 is a recovering beat (152.8 ms). In the box pNa ≥ 1 and the cell is more excitable:
  rerunning the first 10 normal rows with two other splits (one 998-beat `pre`, or a check at beat 250)
  reproduced A90 of beat 1000 to within 0.03 % for 9 rows and 1.9 % for one (row 5: 49.18 vs 50.12 ms).
  The `b500_*` columns can carry such a recovering beat at some rows (1 normal row has `b500_Vmax` < 0).
- **Solver tolerance** (original run, reproduced exactly by the CVODES run of check 5). At the nominal point (one 998-beat `pre`), rel/abs tolerances 1e-3/1e-5 instead
  of 1e-4/1e-6 changed A90 of beat 1000 from 199.19 to 199.79 ms (0.3 %). Tighter tolerances were not
  tried.
- **Gradients:** none, so no gradient check.

## How these files relate to the paper

- **Not the paper's runs.** The authors' repository (https://github.com/ZhechaoYanggla/Sobol_paper,
  commit f11c5e6) holds their scripts, the `.mmt` file and figure data, but not the 262,144 Saltelli
  inputs and outputs: `fig020/sample*.txt` and `fo*.txt` are convergence curves of Sobol' indices
  against sample size (8000 values each), not model runs. It does hold 2000 full-model runs over the
  Table 4 box (the L = 2000 sample of Eq 8), as outputs only, without their inputs:
  `fig060/reduced model.zip`, `*/apd90b.npy`, `apd30b`, `vmaxb`, `vpltb`, `vrestb`, `Bb` (identical in the
  15 folders; `check_accuracy.py --authors`, check 6), and `fig080/apd90_R.npy`. They are enough to compare distributions, not rows. The data
  here are a new computation with the authors' model file, protocol and Table 4 ranges, on a smaller
  Sobol' design.
- **The distributions do not match.** Median (5-95 %) of the normal rows here at beat 1000, at beat
  500, and of the authors' 2000 stored runs: A90 88.6 (45.9-152.3), 95.7, 144.2 (87.5-307.3) ms; A30
  36.6, 39.5, 73.1 ms; Vmax 15.0, 16.2, 21.0 mV; Vplt −18.0, −18.4, −5.3 mV; Vrest −81.7, −81.7, −80.7 mV
  (notebook, check (b); the stored medians from `check_accuracy.py --authors`, check 6). 1.3 % of the
  stored A90 values are 499.9 ms (the whole window above threshold) and the minimum is 60.5 ms (check 6).
  The authors' script times A90 from the upstroke (in the 14-row rerun, about 1 ms shorter than ours; an
  external run), so the definitions do not explain longer APs. In the authors' arrays, their six-factor
  reduced model for A90 (`fig080/apd90_NS.npy`, pNaK fixed at 1) gives R² = 0.837 against the full A90
  (`apd90_R.npy`; 0.841 after `fig080.py`'s filter of values ≥ 499 ms, while the figure is annotated 0.91;
  check 6), so pNaK is not a main A90 driver in their runs. Paper Table 7 (p. 21) lists the six factors of
  this model as pClb, pKr, pNaCa, pK1, pCaL and pKs (R² = 0.93), but the `fig080` folder also holds a file
  named `IKr,IK1,INaCa,IClb,ICaL,IClCa`, which points to pClCa instead of pKs; which set the stored arrays
  used is not stated (open).
  **The cause is unknown.** As stated, the protocol is the same: the paper (p. 10) and
  `output_for_AP_features.py` use the same file (byte-identical), initial state, stimulus, no tolerance
  setting (Myokit defaults) and `pre(999 * 500)` + one logged beat, and the same Table 4 box with the same
  factor-to-current mapping as here. **But the paper's Fig 2(a) matches the first 5–20 beats of this file, not
  the 1000-beat train** (check 2 above): varying pNaK alone, A90 at small pNaK is about 300 ms in Fig 2(a) and
  at beats 5–20 here, and 86.6 ms at beat 1000. The stored biomarkers here are those of beat 1000, the paper's
  stated protocol, and are kept as they are; which train and file produced Fig 2 and the Saltelli runs is
  asked of the authors (open). The evidence points to the authors' cells being less Na-loaded:
  in the 14-row rerun, the median ratio (or difference) of each biomarker at beat k to beat 1000,
  applied to the medians here, gives for beats 5-100 Vmax 18.5-20.6 mV, Vplt −6.0 to +0.1 mV and Vrest
  −80.9 to −80.1 mV, close to the stored 21.0, −5.3 and −80.7, but A90 108-122 ms and A30 52-64 ms,
  still short of 144 and 73. Fewer effective beats, a different model file (`fig010/fig010.py` loads
  `shannonRado.mmt`, not in the repository, and pre-paces 99 beats) or a different Myokit version
  (their paths show Python 3.6; the Myokit changelog says "Myokit is no longer tested on Python 3.6" in
  version 1.34.0, 2023-06-07) could each contribute; the repository does not say which script produced the
  stored runs.
- **Same as the paper:** model file, 15 factors and ranges, stimulus, BCL, 1000-beat train from the
  model's initial state, biomarkers of the last beat, uniform linear sampling.
- **Different:** a 320-point scrambled Sobol' design instead of a 262,144-run Saltelli design; Table 3
  biomarker definitions instead of the authors' script; runs without a normal response flagged and
  excluded in the notebook (the authors' script assigns biomarkers to every run); pK1 upper bound 1.01
  (Table 4) instead of 1.02 (script); Myokit 1.39.2 instead of the authors' (unstated) version.
- **Results.** The notebook finds `pNaK` the leading factor for five of six AP biomarkers in the full box,
  while the paper ranks it seventh (Eq 9). At beat 500, A90 is led by pClb, pK1 and pKr and Vmax by pCaL
  and pClb, close to the paper. On the rows with pNaK ≥ 0.3, A90 is also led by pClb, pK1 and pKr, at
  beat 1000 and at beat 500 alike, while restricting pClb or pK1 to the same share of their ranges
  leaves pNaK first (+0.79, +0.70): that check shows that pNaK's effect is concentrated where the pump
  is weak (a strong nonlinearity), not that it is an artefact of the beat-1000 snapshot. Without the 28
  flagged rows (Ca overload or Vmax < 5 mV) pNaK still leads or ties for the lead (check (e)). See the
  notebook's checks (a)-(e) and comparison table.

## Unknowns

- Whether the fixed E_Na_SL = −15 mV in the authors' model file was intended (asked of the authors; see "Model").
- Why the authors' 2000 stored runs have longer APs than the runs here; which pacing train and model file produced the paper's Fig 2 and its Saltelli runs, and which script produced the stored runs (asked of the authors; see "How these files relate to the paper").
- Which six factors the authors' stored reduced-model arrays used: Table 7 lists pKs, a file name in `fig080` points to pClCa.
- The Myokit version and solver tolerances of the authors' runs (the paper states none).
- The 262,144 Saltelli inputs and outputs of the paper are not published, and the authors' 2000 full-model runs are stored without their inputs.
- Tighter solver tolerances than 1e-4/1e-6 were not tried.

## License and credit

Code (the notebook, `shannon_driver.py`, `generate_data.py`, `check_accuracy.py`): MIT (`LICENSE`). Data
(`rabbit_ap_samples.csv`): CC BY 4.0 (`LICENSE-DATA`). The paper is CC BY 4.0. The model file
`shannon_wang_puglisi_weber_bers_2004_a_p.mmt` is copied unchanged from the authors' repository
(https://github.com/ZhechaoYanggla/Sobol_paper, `Code for Sobol analysis/`, commit
f11c5e627c33b63479baaa4dae821eb342a82e68 of 2025-07-16, the latest; byte-identical, SHA-256
0eebd3bfdbb5e0f3f429ffdd7e5f831c7e2a28db2395cd30b3ca0ba3b3a22202). **That repository states no licence**, so
the file is hosted here with the source credited and is listed in the exceptions list
of `LICENSE-DATA`; if the authors attach a licence or object, that takes precedence. It is Shannon et al. 2004
with E_Na_SL fixed at −15 mV, as distributed by Yang et al. (see "Model"): Myokit's import of the model from the
CellML Model Repository (C. M. Lloyd, J. R. Lawson, P. J. Hunter, P. F. Nielsen, The CellML Model Repository,
*Bioinformatics* 24(18):2122–2123, 2008, doi:10.1093/bioinformatics/btn390; the paper's ref. 28), whose
encoding the file's documentation credits to Martin Fink (Oxford University), with the 15 scaling constants
and the edits listed above added by Yang et al. Credit: model, T. R. Shannon, F. Wang, J. Puglisi, C. Weber and
D. M. Bers; model file and parameter study, Z. Yang, H. Gao, G. L. Smith and R. D. Simitev. Cite Yang et al.,
PLOS ONE 20(7):e0328261, 2025, doi:10.1371/journal.pone.0328261, and Shannon et al., Biophys. J.
87(5):3351–3371, 2004, doi:10.1529/biophysj.104.047449. Paper funding: UK EPSRC, EP/S030875/1 and
EP/T017899/1.
