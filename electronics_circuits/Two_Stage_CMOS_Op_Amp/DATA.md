# Two_Stage_CMOS_Op_Amp: data

Kind: **stored data**, computed for this repository with the circuit simulator ngspice (an outside, non-Python code; BSIM4 cannot reasonably be reimplemented in numpy). `generate_data.py` and `opamp_spice.py` record exactly how: with ngspice-42 installed they reproduce the shipped files bit for bit (checked on 20 rows of each set, below). Without ngspice the files are the record, and the notebook needs only them.

## Files

| file | contents | role |
|---|---|---|
| `Two_Stage_CMOS_Op_Amp.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom, seed 2020; outputs committed |
| `two_stage_opamp.cir.template` | ngspice deck: circuit, testbench, analyses; `$name` fields filled by `string.Template`; circuit lines adapted from AutoCkt (third-party, no licence stated; see License and credit) | the model (circuit description) |
| `45nm_bulk.txt` | BSIM4 model card "BPTM 45nm" (NMOS and PMOS), copied unaltered from AutoCkt (third-party, no licence stated; see License and credit) | the model (device parameters) |
| `opamp_spice.py` | `simulate(design)`, `measure`, `qoi_vector`; `NAMES`, `LOWER`, `UPPER`, `HELD`, `QOI_NAMES` | writes the deck, runs ngspice, measures the four outputs |
| `generate_data.py` | draws the samples and writes the six CSVs; `--check`, `--fd-check` | regenerates the data |
| `inputs.csv` | 1000 × 7: `mp1, mn1, mp3, mn3, mn4, mn5` (multipliers, 1) and `cc_pF` (pF) | set "box" inputs |
| `outputs.csv` | 1000 × 4: gain (V/V), ugbw (Hz), pm (deg), i_supply (A); `nan` for ugbw and pm at the 310 designs with gain ≤ 1 V/V | set "box" outputs |
| `gradients.csv` | 1000 × 28: column 7k + j is ∂q_k/∂x_j with respect to the **physical** inputs (per unit multiplier, per pF); `nan` where q_k is | set "box" gradients |
| `gain200_inputs.csv`, `gain200_outputs.csv`, `gain200_gradients.csv` | the same three layouts for set "gain200" (no `nan`) | set "gain200" |

The CSVs are plain comma-separated numbers (17 significant digits) after two `#` lines, a description and the column names with units, so `np.loadtxt(path, delimiter=',')` reads them; `nan` is read as NaN. Total size 1.7 MB.

## Model

**Paper.** K. Settaluri, A. Haj-Ali, Q. Huang, K. Hakhamaneshi and B. Nikolic, "AutoCkt: Deep reinforcement learning of analog circuit designs," *2020 Design, Automation & Test in Europe Conference & Exhibition (DATE)*, IEEE, 2020, pp. 490–495, doi:10.23919/DATE48585.2020.9116200. Read as the preprint arXiv:2001.01808v2 (20 Jan 2020). The paper is a reinforcement-learning design study, not a sensitivity study; the two-stage op-amp is its Sec. III.B and Fig. 6.

**Circuit** (Fig. 6; AutoCkt netlist `eval_engines/ngspice/ngspice_inputs/netlist/two_stage_opamp.cir`, commit a6c8a61 of 2022-03-17). First stage: NMOS input pair mn1/mn2, PMOS mirror load mp1/mp2, NMOS tail mn3. Second stage: PMOS common source mp3 with NMOS current-source load mn5. Bias: diode-connected mn4 fed by a 30 µA reference, mirrored to mn3 and mn5. Miller capacitor Cc between the stage outputs (no nulling resistor), 10 pF load. Supply 1.2 V; input common mode 0.6 V; unit AC differential input split ±0.5 by two controlled sources. Every device is W = 0.5 µm, L = 90 nm with a parallel multiplier m (mirror pairs share one m). Device model: BSIM4 (level 54), the 45 nm Berkeley Predictive Technology Model card shipped with AutoCkt (Cao et al., CICC 2000, doi:10.1109/CICC.2000.852648). `two_stage_opamp.cir.template` is adapted from the AutoCkt netlist: its element and source lines are AutoCkt's (same node names, elements, connections and values); ours adds braces around parameter references, `.options reltol`, a denser sweep and full-precision output, and writes the outputs to `ac.txt` and `op.txt`. With AutoCkt's sizes it reproduces AutoCkt's netlist output (gain 0.6305531 V/V and supply current 47.15310 µA in both).

**Inputs** (Sec. III.B, p. 3; AutoCkt `two_stage_opamp.yaml`):

| name | symbol | units | range in the paper | here | source |
|---|---|---|---|---|---|
| `mp1` | m of mp1, mp2 (PMOS mirror load) | 1 (× 0.5 µm / 90 nm device) | integers 1…100 (AutoCkt grid 1…99) | real, [1, 100] | Sec. III.B; YAML `[1, 100, 1]` |
| `mn1` | m of mn1, mn2 (NMOS input pair) | 1 | integers 1…100 | real, [1, 100] | same |
| `mp3` | m of mp3 (PMOS common source) | 1 | integers 1…100 | real, [1, 100] | YAML |
| `mn3` | m of mn3 (NMOS tail) | 1 | integers 1…100 | real, [1, 100] | YAML |
| `mn4` | m of mn4 (NMOS bias diode) | 1 | integers 1…100 | real, [1, 100] | YAML |
| `mn5` | m of mn5 (NMOS second-stage load) | 1 | integers 1…100 | real, [1, 100] | YAML |
| `cc_pF` | Cc | pF | 0.1, 0.2, …, 10.0 (AutoCkt grid 0.1…9.9) | real, [0.1, 10] | Sec. III.B; YAML |

The paper writes "the action space for every transistor width … is [1, 100, 1] ∗ 0.5 µm"; the AutoCkt netlist implements this as a multiplier m with W fixed at 0.5 µm. We follow the netlist. The paper's 10¹⁴ designs are an integer grid; AutoCkt builds it with `np.arange(start, stop, step)`, which excludes the stop value, so its grid has 99 values per input (m = 1…99, Cc = 0.1…9.9 pF), 99⁷ = 9.3 × 10¹³ ≈ 10¹⁴, and our box [1, 100], [0.1, 10] pF includes the paper's stated upper ends; **we treat the multipliers and Cc as continuous**: ngspice accepts a real m (it scales the device's currents, charges and capacitances by m), and a real m between grid points changes the outputs smoothly (checked by the step study below).

**Held settings** (`opamp_spice.HELD`): reference current 30 µA, load 10 pF, common mode 0.6 V, supply 1.2 V (all AutoCkt's), ngspice `reltol = 1e-9` (AutoCkt: default 1e-3) and 100 AC points per decade from 1 Hz to 10 GHz (AutoCkt: 10). The two numerical settings were tightened for smooth finite differences; on 10 gain200 designs, AutoCkt's 1e-3 changes the outputs by at most 1.2e-5 relative (PM by 1.9e-4°), and 10 points per decade change ugbw by 3.6e-6 relative and PM by 4.6e-4°.

**Quantities of interest** (Sec. III.B: "gain, unity gain bandwidth, phase margin, and bias current"; definitions as in AutoCkt's `TwoStageClass.py`), with H(f) = v(out)/v(in):

- `gain`: |H(1 Hz)| (V/V). Paper target [200, 400] V/V.
- `ugbw`: f_u with |H(f_u)| = 1 (Hz), from a cubic spline in log f through the sweep (`brentq`, tolerance 1e-14 in log10 f). Target [1e6, 2.5e7] Hz.
- `pm`: 180° + ∠H(f_u), phase unwrapped from 1 Hz, spline as above (deg). Target ≥ 60°.
- `i_supply`: −i(vdd) at the operating point (A), the total supply current, which AutoCkt reports as "ibias" and the paper as "bias current (as a measure of power)". Target [0.1, 10] mA.

**Failure modes.** `opamp_spice.measure_unity_gain` raises `MeasurementError` instead of returning a number when |H(1 Hz)| ≤ 1, when |H| is still ≥ 1 at 10 GHz, when |H| crosses 1 more than once, or when the 1 Hz phase is not within 10° of 0. AutoCkt instead returns the lowest sweep frequency and −180°. `simulate` also raises when ngspice exits nonzero or prints an error, abort, singular-matrix or non-convergence message (the BPTM card's "ckappas/ckappad too small" and "dtox ignored" warnings are expected and ignored). In the shipped data: no simulation failed (also none at the 128 corners of the box); the unity-gain measurement is undefined at 310 of the 1000 box designs, all with gain ≤ 1 V/V (an output stuck at a rail), and is stored as `nan` there (`generate_data.py` calls `measure(strict=False)`); no stencil point of a defined design was undefined. These are not unphysical samples (they are valid designs that do not amplify), so they are kept and counted, not dropped. They do enter the gain analysis on the box: the 310 designs with gain ≤ 1 V/V carry 26.9 % of trace(Ĉ) for gain in dB (the 23 below −40 dB, 6.0 %), so part of the gain subspace describes how fast the gain collapses away from the balance condition (notebook).

**Sampling design.** Monte Carlo, independent log-uniform in every input on the paper's box (each input spans exactly two decades). The paper defines the box but no distribution; log-uniform is our choice, because the circuit depends on ratios of sizes and a log scale treats every input alike.

- Set "box": 1000 draws, `numpy.random.default_rng(2020)`.
- Set "gain200": draws from `default_rng(2021)`, kept when gain ≥ 200 V/V (the paper's lower gain target) until 1000 were kept: 12 367 draws (acceptance 0.0809). These are a sample of the log-uniform distribution conditioned on gain ≥ 200. Bandwidth and phase margin are analysed on this set, because on the box their gradients are dominated by designs whose gain is barely above 1 (notebook). The conditioned inputs are neither uniform nor independent (normalized-input kurtosis 1.76–2.20, means −0.19 to +0.21, largest |correlation| 0.534; the balance variable log10(mp3·mn3/(mp1·mn5)) lies in [0.074, 1.087], sd 0.122, against [−3.52, 3.67], sd 1.161, on the box): the notebook's `ac.Uniform` space is only the normalization for this set, and its active subspaces are those of the conditioned density. They depend on the threshold: on box subsets, the first eigenvector of log10 f_u matches the gain200 one for gain ≥ 50 V/V (|cos| 0.997) but not for gain ≥ 10 (0.836), and that of phase margin only for gain ≥ 100 (0.997; gain ≥ 50: 0.171).

**Gradients.** Central differences in ln x_j with step h = 1e-4 (x_j e^{±h}), 14 extra ngspice runs per sample, converted to ∂q/∂x_j = (q₊ − q₋)/(2h x_j). The notebook converts once to z = log10 x.

**Published results to reproduce.** None for this circuit beyond the design-space size (10¹⁴), the target ranges and the RL statistics (Table II: AutoCkt reaches 963 of 1000 random targets, a random agent 38). The notebook's comparison table lists them beside what the data say.

## Regenerating

```bash
python generate_data.py                                   # both sets, 2 worker processes
python generate_data.py --set box --rows 0:20 --out /tmp/oa --check      # a subset, compared with the files here
python generate_data.py --fd-check 0:6                    # gradient step study (box rows)
python generate_data.py --set gain200 --fd-check 0:6      # the same on gain200 rows
```

Needs numpy, scipy and the program `ngspice` on `PATH` (or `NGSPICE=/path/to/ngspice`). The shipped files were written on 2026-10-30 with ngspice-42 (Ubuntu 24.04 package `ngspice 42+ds-3build1`, installed with `apt-get install ngspice`), Python 3.13, numpy 2.5.4, scipy 1.18.1, on a 2-core container: 879 s in all (box set 313 s; gain200 rejection draws 256 s and its gradients 310 s), 44 000 ngspice runs (the rejection step simulates whole batches of 2000 draws, 14 000 in all), about 40 ms of wall time per run per worker while other work shared the machine (17–20 ms alone). ngspice-42 evaluates BSIM4 devices with OpenMP, and its spinning threads slowed two concurrent runs about tenfold here, so `opamp_spice.py` runs it with `OMP_NUM_THREADS=1` and `OMP_WAIT_POLICY=passive`; this does not change the results (10 samples identical with and without). There is no other randomness: `--rows 0:20 --check` on each set reproduces outputs and gradients exactly (maximum relative difference 0.0, same NaN pattern). Another ngspice version or build may differ in the last digits.

## Accuracy

- **Gradients** (`--fd-check`): relative change of each gradient vector (norm over the 7 inputs) when the step is 1e-3 or 1e-5 instead of 1e-4: at most 1.5e-4 (step 1e-3) and 2.0e-4 (1e-5) on box rows 0–5, and 8.1e-5 and 4.2e-4 on gain200 rows 0–5, over all four outputs. So the shipped gradients are accurate to about 1e-4 relative or better. An independent check (review, 2026-10-30) with central differences in physical units, relative steps 1e-3 and 3e-4, at gain200 rows 59, 634, 785 and box rows 862, 250, 336 agreed with the shipped gradients to between 4.6e-7 and 6.9e-5 relative (vector norm), all four outputs; 121-point sweeps in Cc and mn3 across about two sweep-grid intervals show no kink in log10 f_u (second differences: sd 2e-11 and 7e-10, against first differences 4e-5 and 1.6e-4).
- **What this resolves**: a relative gradient error ε changes the estimated matrix C by about 2ελ₁, so eigenvalues below about 1e-4 λ₁ are at the resolution floor: the sixth and seventh of gain (dB), and the seventh of log10 I, log10 f_u and PM. The notebook does not interpret them quantitatively.
- **Outputs**: on 10 gain200 designs, 1000 points per decade instead of 100 change ugbw by at most 5e-10 relative and PM by 3e-8°. The gain is read at the first sweep point and the current at the operating point, so neither depends on the sweep.
- **Model inputs between grid points**: the step study above runs at real multipliers (x_j e^{±h} with h ≤ 1e-3) and shows smooth central differences, i.e. ngspice does not round m.

## How these files relate to the paper

- **Not the paper's computation.** The paper trains an RL agent; its simulations (ngspice; the AutoCkt README names release 27) are not published as data. These files are an independent ngspice-42 computation of the same netlist and model card over the paper's design box, for an active-subspace study the paper does not make.
- **Choices that are ours**: continuous inputs, log-uniform sampling, the conditioned set gain ≥ 200 (the bandwidth and phase-margin subspaces hold for that conditioning; see Sampling design), `reltol` and sweep density, the spline-based f_u and PM and the errors raised where they are undefined.
- **AutoCkt's own sizes** in the netlist (mp1 10, mn1 38, mp3 4, mn3 9, mn4 20, mn5 60, Cc 3 pF) give gain 0.63 V/V, so no f_u; they are a placeholder, not a design.

## License and credit

Code: MIT (`LICENSE`). Data (the six CSVs): CC BY 4.0 (`LICENSE-DATA`). Circuit, design space and quantities of interest: Settaluri, Haj-Ali, Huang, Hakhamaneshi and Nikolic (DATE 2020; cite the paper above). The AutoCkt repository (github.com/ksettaluri6/AutoCkt) states **no licence**. Following the repository author's decision of 2026-10-30, the model card `45nm_bulk.txt` is shipped unaltered (sha256 `42489e9aab3423fe868c6d249326535892ad87de7f8f5b18d790d0afe8401ef4`) with this credit; it identifies itself as "BPTM 45nm", the Berkeley Predictive Technology Model (Cao et al. 2000), and we found no licence statement for it either. `two_stage_opamp.cir.template` is adapted from AutoCkt's netlist `two_stage_opamp.cir`: its element and source lines are AutoCkt's, with braces around parameters; the header, `.param` substitution fields, options and `.control` block are ours. Like the model card, it is hosted with credit under the 2026-10-30 decision for material without a licence. **Neither `45nm_bulk.txt` nor the circuit lines taken from AutoCkt are covered by this repository's MIT or CC BY 4.0 licences**; they remain under whatever terms their authors set (none stated). `opamp_spice.py` and `generate_data.py` were written for this repository (MIT) from AutoCkt's measurement definitions in `TwoStageClass.py`; no AutoCkt Python code was copied. ngspice is not redistributed here.
