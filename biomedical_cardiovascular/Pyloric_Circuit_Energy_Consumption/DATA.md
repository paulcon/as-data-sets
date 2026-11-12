# Pyloric_Circuit_Energy_Consumption: data

Kind: **stored data**. The files were computed on 2026-10-31 with the authors' simulator
[pyloric](https://github.com/mackelab/pyloric) (MIT licence), commit
89dd27d769fa2652d21bca1dbbc098c85e40244a, an outside code that is not part of this repository
and is not installed in the repository's `.venv` (it needs Cython and a C compiler besides numpy,
scipy and pandas). `generate_data.py` is the script that wrote them, kept for provenance and for
reruns where pyloric is installed; without pyloric it exits with status 3 and says what to install.
The values in the CSVs are the record and are not to be edited; pyloric is rerun only if a problem
is found in them.

## Files

| file | contents | role |
|---|---|---|
| `Pyloric_Circuit_Energy_Consumption.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` with seed 401; outputs committed |
| `generate_data.py` | draws the design, runs pyloric on a block of rows (`--rows`), merges blocks (`--merge`), compares with the files (`--check`) | wrote the CSVs; needs a pyloric clone |
| `check_accuracy.py` | reruns a few rows with other noise seeds and time steps (Accuracy below) | check; needs a pyloric clone |
| `inputs.csv` | 1100 × 31 physical inputs: 24 membrane conductances (mS/cm²; AB/PD, LP, PY × Na, CaT, CaS, A, KCa, Kd, H, leak), then 7 synaptic conductances (nS; AB→LP, PD→LP, AB→PY, PD→PY, LP→PD, LP→PY, PY→LP) | stored inputs |
| `outputs.csv` | 1100 × 9: energy per second of AB/PD, LP, PY and of the circuit (µJ s⁻¹ cm⁻²); spike counts of AB/PD, LP, PY in 10 s; `pyloric_like` (0/1); `finite` (0/1) | stored outputs |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column
names with units), written with 17 significant digits, so `np.loadtxt(path, delimiter=',')`
reads them. The column names follow the notebook: `g_<channel>_<neuron>` and `g_syn_<pre>_<post>`.

## Model

**Citation.** M. Deistler, J. H. Macke, P. J. Gonçalves, "Energy-efficient network activity from
disparate circuit parameters", *Proc. Natl. Acad. Sci. USA* 119(44):e2207632119, 2022,
doi:10.1073/pnas.2207632119 (published 24 October 2022, CC BY-NC-ND 4.0). The text read was the
**bioRxiv preprint**, doi:10.1101/2021.07.30.454484, posted 24 January 2022 (CC BY-NC-ND 4.0); page
numbers below are the preprint's. The model is that of A. A. Prinz, D. Bucher, E. Marder, *Nat.
Neurosci.* 7(12):1345–1352, 2004, doi:10.1038/nn1352. The pyloric README asks users to cite the PNAS
paper.

**Equations** (Methods, "Circuit model of the crustacean stomatogastric ganglion", p. 15).
Three single-compartment neurons (AB/PD, LP, PY), each with I_Na, I_CaT, I_CaS, I_A, I_KCa, I_Kd,
I_H and I_leak, and seven graded synapses I_s = ḡ_s s (V_post − E_s), ds/dt = (s̄(V_pre) − s)/τ_s,
s̄ = 1/(1 + exp((V_th − V_pre)/δ)), τ_s = (1 − s̄)/k₋, with E_s = −70 mV, k₋ = 1/40 ms⁻¹
(glutamatergic) or −80 mV, 1/100 ms⁻¹ (cholinergic), V_th = −35 mV, δ = 5 mV; membrane area
0.628·10⁻³ cm². Gate kinetics and reversal potentials are those coded in pyloric's
`simulator.pyx` (after Prinz et al.); E_Ca follows the Nernst equation with the intracellular
calcium.

**Inputs** (m = 31; p. 15 and pyloric `prior.py`):

| inputs | units | distribution | bounds AB/PD | bounds LP | bounds PY | source |
|---|---|---|---|---|---|---|
| ḡ_Na | mS/cm² | uniform | 0–500 | 0–200 | 0–600 | p. 15; code |
| ḡ_CaT | mS/cm² | uniform | 0–7.5 | 0–2.5 | 0–12.5 | p. 15; code |
| ḡ_CaS | mS/cm² | uniform | 0–8 | 2–12 | 0–4 | p. 15; code |
| ḡ_A | mS/cm² | uniform | 0–60 | 10–60 | 30–60 | p. 15; code |
| ḡ_KCa | mS/cm² | uniform | 0–15 | 0–10 | 0–5 | p. 15; code |
| ḡ_Kd | mS/cm² | uniform | 25–150 | 0–125 | 50–150 | p. 15; code |
| ḡ_H | mS/cm² | uniform | 0–0.02 (paper: 0–0.2) | 0–0.06 | 0–0.06 | code (paper differs for AB/PD) |
| ḡ_leak | mS/cm² | uniform | 0–0.01 | 0.01–0.04 | 0–0.04 | p. 15; code |
| ḡ_s, 6 synapses | nS | log-uniform | 0.01–1000 | | | p. 15; code (ln mS: ln 1e−8 to ln 1e−3) |
| ḡ_s, AB→LP | nS | log-uniform | 0.01–10000 | | | p. 15; code |

The bounds are pyloric's `prior_bounds()`: Prinz's ranges padded by
[100, 2.5, 2, 10, 5, 25, 0.01, 0.01] and clipped at 0. They agree with the paper's Methods except
the AB/PD ḡ_H upper bound: the preprint prints "phigh = [500, 7.5, 8, 60, 15, 150, 0.2, 0.01] mS cm−2
for the maximal membrane conductances of the AB neuron" (p. 15, line 446), while the code's `prior_bounds()`
gives 0.02. The files use the code's value, the one the
authors' simulator samples; the printed 0.2 is taken to be a misprint (open question below).

**Quantity of interest** (Eq. 2, p. 16, after Moujahid et al., Phys. Rev. E 83:031912, 2011): for
each neuron E = ∫ Σ_m g_m (V − E_m)² + Σ_s g_s (V − E_s)² dt over the 10 s after a 1 s burn-in,
with g the gated conductances; the circuit energy is the sum over the three neurons, divided by
10 s, as in the paper. pyloric's energy trace is in nW/cm² (it divides by the membrane area);
pyloric's `summary_stats(..., {"energies": True})` integrates it to nJ/cm², and
`generate_data.py` divides by 10 s and 1000, giving µJ s⁻¹ cm⁻². Beware: pyloric's own docstring
(`interface.py`, line 295) calls the integrated quantity "microJoule"; it is nJ per cm², a factor
1000 smaller and per unit area, which is the easy trap when rebuilding these data. The paper's axes say "µJ/s"
with the same numbers (its posterior circuits span about 3–27, Supp. Fig. 5a). The script also
checks (assert, relative 1e−12) that pyloric's integrated energies equal its own sum of the trace.

**Settings.** dt = 0.025 ms; 11 s simulated (pyloric's `t_max = 11000` ms) and the first 1000 ms
discarded (pyloric's `summary_stats` default), so 10 s are analysed; the paper says only "simulate
for 10 seconds" (p. 15) and does not mention the burn-in. Temperature 283 K (9.85 °C), pyloric's
reference, at which every Q10 factor is 1; the paper's reference temperature is 11 °C (Eq. 1,
p. 16), which makes no difference here because no Q10 factor is used. pyloric's default initial
state. Gaussian **current** noise of SD 0.001 (`noise_std`) at every step: it enters the
exponential-Euler numerator together with the ionic currents (`simulator.pyx`, lines 419–420), so
its units are µA, not mV; pyloric does not rescale it with dt (the paper writes mV·ms⁻⁰·⁵).

**Simulator noise: common random numbers.** pyloric draws the noise from
`numpy.random.RandomState(seed)`. Every run here uses seed 1, so all 1100 circuits see the same
noise realization and the energy is a deterministic function of the 31 inputs.

**Gradients.** None. The simulator has no derivatives, and finite differences would need 32 runs
per sample and are unreliable where spike counts change. The notebook estimates the subspace
from values: a ridge-regularized quadratic (penalty chosen by leave-one-out), with the
unregularized QPHD shown against a permutation null and the linear-model (OLS) direction as the
check.

**Sampling design.** Simple Monte Carlo: 1100 independent samples, `numpy.random.default_rng(401)`,
uniform in the membrane conductances and in ln of the synaptic conductances (pyloric's
coordinates), then stored with the synapses converted to nS (exp(θ)·10⁶). 1100 is about twice the
(m+1)(m+2)/2 = 528 coefficients of a quadratic in m = 31 inputs. All 1100 runs gave finite traces; none were dropped.
Nothing outside physical meaning occurs: every conductance is ≥ 0, every energy > 0
(minimum 0.026).

**Study region.** The whole prior box. The paper's active-subspace analysis is over its posterior
(below), whose samples and simulations are on the Hugging Face Hub (see "How these files relate to
the paper") and are not used here; on the prior, 7 of the 1100 circuits
(0.64 %) are pyloric-like by pyloric's flag, and the rhythm features are undefined for most
circuits, but the energy is defined for every run.

**Published results** (for comparison; Supp. Fig. 5, Methods "Regression neural network", p. 18):
the authors trained a ResNet regression from the 31 parameters to the energy on their database of
35,939 posterior samples that match experimental data, and computed
M = E_{θ∼p(θ|x_o)}[∇f ∇fᵀ] (Eq. 3) from the network's gradients as a Monte Carlo mean over 10,000
posterior samples. Supp. Fig. 5b shows one dominant eigenvalue (values not printed); Supp. Fig. 5c
shows the leading eigenvector, close to the linear-regression weights of Fig. 4c. Read from the
figure's pixels (blue, orange, green = AB/PD, LP, PY, the order of the authors' plotting code): positive
on Na, CaT, CaS of all neurons (largest PY CaT about 0.54 and Na about 0.46), negative on A (about
−0.16 in AB/PD and PY, −0.08 in LP) and on Kd (about −0.08 in all three), PY KCa about −0.12, PY leak
about −0.06; synapses small, with PD→PY about +0.10, PY→LP about +0.11, AB→LP about +0.05. The
authors' plotting code draws every membrane bar below 0.03 (synapse bar below 0.01) at that minimum
height, so those bars give a sign at most. The paper does not print the eigenvalues or the input
scaling; the authors' figure notebook (`paper/fig4/notebooks/03_appendix_active_subspace.ipynb` in
github.com/mackelab/stg_energy, commit 783d134, read but not run) prints in its saved outputs
λ1/λ2 = 35.7893 and λ1/trace = 0.9275, and uses `sbi`'s `ActiveSubspace.find_directions()` with its
defaults, which (sbi 0.16 source) draw 1000 posterior samples (the Methods say 10,000) and multiply
each gradient entry by the prior's standard deviation, i.e. the box scaling used in the notebook up to
a common factor if the attached prior is pyloric's box-uniform one.

## Regenerating

```bash
git clone https://github.com/mackelab/pyloric && git -C pyloric checkout 89dd27d
python3 -m venv pyl && pyl/bin/pip install numpy pandas scipy cython setuptools tqdm matplotlib
# one block of rows per call (each call is one process; pyximport compiles the simulator into ~/.pyxbld on first use)
for r in $(seq 0 100 1000); do
  pyl/bin/python generate_data.py --pyloric pyloric --rows $r:$((r+100)) --out CHUNKDIR
done
pyl/bin/python generate_data.py --merge CHUNKDIR        # writes inputs.csv and outputs.csv here
pyl/bin/python generate_data.py --pyloric pyloric --rows 500:505 --out OTHERDIR --check
pyl/bin/python check_accuracy.py --pyloric pyloric
```

`--out` is required for a run and must be outside this directory (the chunk files are not shipped).
torch and sbi are not needed: pyloric imports them in its package `__init__`, but only
`create_prior()` uses them; `generate_data.py` never calls it and puts empty stand-ins for `torch`
and `sbi.utils` in `sys.modules` before importing pyloric. The design and the noise are seeded, so a
rerun reproduces the files. Without a pyloric clone (or without pandas or Cython), `generate_data.py`
and `check_accuracy.py` exit with status 3 and a message that names pyloric, its commit and the
packages to install (checked in `.venv` on 2026-11-02).

The shipped files were written on 2026-10-31 in a Linux container (Python 3.13.18, numpy 2.5.4,
pandas 3.1.0, scipy 1.18.1, Cython 3.3.0, setuptools 84.2.1, tqdm 4.70.1, matplotlib 3.11.2; gcc).
That run of `generate_data.py` took 1737 s of CPU in 11 blocks of 100 rows (1.58 s per run), one
process at a time, and 3856 s (64 min) of wall time on 2 CPUs shared with other jobs (2.2–8.6 s per
run). In the same environment `--rows 500:505 --check` reproduced those rows exactly (maximum relative
difference 0 for inputs and outputs, 2026-11-01). pyloric is not installed in `.venv`, so neither
script was run there. The design alone (inputs) regenerates from `default_rng(401)` and pyloric's
bounds; the notebook checks this (largest relative difference of order 1e−15, rounding).

## Accuracy

`E_total` equals the sum of the three neurons' energies exactly in every row (checked in the
notebook), and pyloric's integrated energies equal the sum of its trace to a relative 1e−12 (an
assert in `generate_data.py`, which holds for every shipped row). The other numbers below need
pyloric. They come from the original run in the Linux container named above (2026-10-31 and
2026-11-01), and `check_accuracy.py` recomputes them where pyloric is installed (it was not run
here); it also reruns rows 0 and 1, which reproduced the files exactly in the original run.

- **Simulator noise** (`check_accuracy.py`, check 2). Rows 0–7 rerun with noise seeds 2, 3 and 4:
  the relative standard deviation of the circuit energy over the four seeds is 0.07 %–10.2 %
  (median 1.7 %); the largest are for circuits whose spike counts change with the noise (row 6:
  28–40 AB/PD spikes). The mean noise variance is 0.25 % of the variance of the energy over the 1100
  samples, so the noise does not limit the active-subspace estimate; it would matter for
  finite-difference gradients. No pyloric-like flag changed in those runs (all 0).
- **Time step** (`check_accuracy.py`, check 3). Rows 0–2 with the noise switched off, at
  dt = 0.025 ms and 0.0125 ms: the circuit energy differs by 1.2 %, 0.02 % and 1.8 % (relative).
  The paper and the files use 0.025 ms.

## How these files relate to the paper

- **Not the paper's computation.** These files are a new run of the authors' simulator over the
  paper's prior box, with a seeded design and a fixed noise seed. The paper's own outputs are public:
  the stg_energy README (github.com/mackelab/stg_energy) says its `results/` folder (simulation
  outputs, trained density estimators, analysis artifacts) is the public Hugging Face dataset
  [mackelab/STG_energy](https://huggingface.co/datasets/mackelab/STG_energy) (revision a428509,
  last modified 2026-07-28; no licence is stated on the dataset). The revision, the date and the
  file sizes below were read from the Hugging Face API on 2026-11-02; the files were not downloaded. The 35,939-circuit database that
  the Supp. Fig. 5 notebook reads is
  `simulation_data_Tube_MLslurm_cluster/close_to_xo_circuit_parameters.pkl` (4,457,695 bytes),
  `close_to_xo_simulation_outputs.pkl` (9,776,820 bytes) and `close_to_xo_seeds.npy` (287,640 bytes,
  35,939 int64 seeds); the trained regression network is `regression_nets/210717_total_energy.npy`
  with `210717_sensitivity_total_energy.npy`, and the posterior
  `trained_neural_nets/inference/posterior_11deg.pickle`. None of these are used here; they are
  pandas/dill pickles, to be opened only in an isolated environment. Reproducing Supp. Fig. 5 from them is a follow-up.
- **Different region.** The paper's active subspace (Supp. Fig. 5) is over its posterior, where every
  circuit matches the experimental rhythm and the energy is nearly linear in the parameters; here it
  is over the full prior box, where the energy is far from linear (linear leave-one-out R² 0.426).
  The notebook finds the same signs on Na, CaT, CaS (positive) and A (negative) in every neuron,
  different signs on Kd (AB/PD, LP), PY leak and the PD→PY and AB→LP synapses, and a different
  weighting: AB/PD dominates over the box. On the 277 runs in which all three neurons spike, the
  linear-model direction is led by PY, as in the paper, while random subsets of the same size keep
  the box weighting, so the weighting difference is mostly a region effect.
- **Different estimator.** The paper used autodiff gradients of a neural-network regression; the
  notebook uses a ridge-regularized global quadratic fit and the linear-model direction (OLS) from
  the values. The paper's λ1/λ2 (35.79 in the authors' output) is much larger than the notebook's
  4.81 (ridge); the unregularized QPHD ratio (2.76) is against its own noise floor (permutation test
  in the notebook) and is not used.

## Open questions (for the authors)

- AB/PD ḡ_H upper bound: 0.2 mS/cm² (preprint, p. 15) or 0.02 (pyloric `prior_bounds()`, used
  here)? We take 0.2 to be a misprint.
- Supp. Fig. 5: the authors' notebook suggests gradients scaled by the prior SD and 1000 posterior
  samples (Methods: 10,000). Which prior object was attached to the posterior (box or
  classifier-restricted), and which sample count was used for the published figure?
- Does the Hugging Face dataset `mackelab/STG_energy` have a licence (none is stated)?
- The paper says "simulate for 10 seconds" (p. 15); pyloric simulates 11 s and discards the first
  1 s. Was the burn-in used for the paper's energies?
- pyloric's `LICENSE` and README say MIT, but its `setup.py` metadata says AGPLv3; which applies?
  (These files are outputs of the code, and no pyloric code is copied here.)
- Units: is the paper's "µJ/s" the per-cm² quantity that pyloric computes (as the numbers suggest)?

## License and credit

Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); they were generated here by running pyloric, and no
data of the authors are included. Code (`generate_data.py`, `check_accuracy.py`) and the notebook:
MIT (`LICENSE`). The
simulator is pyloric by M. Deistler and the Macke lab (MIT licence, Copyright (c) 2020 mackelab),
which was run, not copied. Model and energy measure: Prinz, Bucher and Marder (2004) and Deistler,
Macke and Gonçalves (2022). Cite M. Deistler, J. H. Macke, P. J. Gonçalves, "Energy-efficient network
activity from disparate circuit parameters", PNAS 119(44):e2207632119, 2022,
doi:10.1073/pnas.2207632119, as the pyloric README asks.
