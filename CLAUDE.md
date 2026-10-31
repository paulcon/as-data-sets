# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of data sets plus Jupyter notebooks that apply **active subspaces** (dimension reduction for simulation models) to science/engineering models. There is no package, build system, test suite, or linter. The top-level directories are 12 application categories in snake_case (`aerospace_cfd/`, `energy/`, `uq_benchmarks/`, …); each has a `README.md`, and some have no data sets yet. Each data set is a self-contained case study one level down, e.g. `epidemiology_systems_biology/HIV_Long_Term_Dynamics/`: one notebook named after its directory, its data files, and sometimes helper `.py` modules. The README has the per-dataset descriptions and publication links, grouped by category. Add a new data set the same way, and add a bullet for it to the README's "Data sets" list and a line to its category's `README.md`.

The directories were renamed and moved on 2026-10-30 (e.g. `HIV/` → `epidemiology_systems_biology/HIV_Long_Term_Dynamics/`, `LM104X/` → `aerospace_cfd/N2_Supersonic_Jet/`). `specs/` still uses the old names; the map is in `specs/README.md`.

## Environment and port status

- **Every 2016 notebook is ported** (2026-10-30 and 2026-10-31, PRs #38–#52): Python 3, the v1 API below, numpy-only loading, seeded randomness, committed outputs, and a `DATA.md`. They run headless in `.venv`. The 15 data sets added in October 2026 (14 in PR #56, Transonic_Fan_Blade_Performance_Maps in PR #60) were written for Python 3 and the v1 API from the start. Every notebook's committed outputs come from `.venv`. 13 of the October data sets were first executed in a Linux container and were re-executed in `.venv` on 2026-10-31 (PR #61). Their `DATA.md` files report only `.venv` results (`generate_data.py --check`, check cells, run times); the container is named only as the place the shipped files were written. Values at the rounding level (null eigenvalues, model-vs-file differences near 1e-16) change between platforms, so the notebook text gives them as orders of magnitude.
- The 2016 versions of all notebooks are on the tag `legacy-master`. They are Python 2 (kernelspec `python2`, `print x`), use pandas, and call the **old** `active_subspaces` API: `ac.subspaces.Subspaces()` with `.compute(df=...)`/`.partition()`/`.W1`/`.eigenvals`, `ac.utils.plotters.*`, `ac.utils.misc.BoundedNormalizer`, `ac.utils.response_surfaces.*`.
- **The local `.venv` (the PyCharm interpreter) is Python 3.11** with `active_subspaces` 1.0.0.dev0, installed editable from `../ActiveSubspacesRevision/upstream-v1/src`. That is a rewritten API: top-level functions such as `ac.active_subspace`, `ac.finite_difference_gradients`, `ac.local_linear_gradients`, `ac.fit_surrogate`, `ac.PolynomialSurrogate`/`ac.RBFSurrogate`, `ac.active_domain`, and `ac.plotting`. **`ac.utils` and `ac.subspaces.Subspaces` do not exist in it.** Check that source tree for signatures rather than guessing.
- `.venv` lacks `pandas` and `numpad` (the autodiff library the legacy PDE code was written for; `pde_model.py` no longer uses it, see below). No ported notebook uses either.

A port is: convert to Python 3, translate the `ac.*` calls to the v1 API, load data with numpy, apply the spec's fixes, and add a `DATA.md` (kind, files, provenance, how to regenerate, license, unknowns). `uq_benchmarks/Elliptic_PDE_Random_Coefficients/` was the first; the PR bodies of #38–#51 show the format.

## Commands

```bash
source .venv/bin/activate
jupyter lab                                   # interactive
# Execute a notebook headless. Run it from inside its directory, because notebooks and helper modules load data by relative path:
cd epidemiology_systems_biology/HIV_Long_Term_Dynamics && jupyter nbconvert --to notebook --execute --inplace HIV_Long_Term_Dynamics.ipynb
```

## Branches

The default branch is `main` (renamed from `master` 2026-10-30). It is protected: every change, the owner's included, lands through a PR merged with "Create a merge commit". Force-pushes and deletion are blocked. The 2016 state is the tag `legacy-master`.

## Update specs

`specs/` holds the October 2026 review: one spec per notebook (known errors, the v1 call map, a pandas-free loader, acceptance criteria), `TODO.md`, `BRANCH_MERGE.md` (master + dev → main), and `numpad_replacement.md`. Read a directory's spec before you change its notebook.

## Two kinds of data sets

Every data set is **stored data** or a **Python model**; MHD_Generator has one part of each. The README tags each entry, and `specs/README.md` ("Two kinds of data sets") has the full rules and the per-data-set table.

- **Stored data**: an outside code (SU2, ParFlow-CLM, Drekar, FAST, MATLAB, …) produced the files, and they are the only record. Never regenerate or "correct" their values. Format conversions must be lossless and scripted. Rows are dropped in the notebook by a stated rule or a flag, never deleted from the file.
- **Python model**: the model is a numpy/scipy `*_model.py` module in the directory that loads its files relative to `__file__` (e.g. `HIV_model.py`, `pde_model.py`); a closed-form model may stay in a notebook cell. Shipped outputs are written by `generate_data.py` with saved seeds, and that script must reproduce them.

## Conventions in the notebooks

- Inputs are declared as a v1 input space and passed in physical units: `ac.Uniform(lower, upper, names=)` (or `ac.Uniform.reference(m)` for data already on `[-1, 1]^m`), `ac.Gaussian`, or `ac.Product` for mixed uniform × Gaussian inputs. v1 normalizes to `[-1, 1]^m` internally, so physical gradients are passed as they are. (The legacy notebooks normalized by hand with `BoundedNormalizer` and rescaled gradients by `.5*(xu - xl)`.)
- Eigenvector signs follow the published figures, set with `ac.orient` (author, 2026-10-30).
- The typical flow is: load samples (inputs X, outputs f, and gradients df when available) → build the subspace from gradients, or from local-linear/global-linear approximations when gradients are missing → plot eigenvalues and eigenvectors → make a sufficient-summary plot of `f` against `X @ W1` → optionally fit a response surface on the active variables.
- Helper modules are imported from the notebook's own directory. Examples: `HIV_model.Tcells`, `ebola_model.R0`, `hartmann_model.uavg`, `single_diode.pmax`, `pde_model.PDESolver` (KL-expansion coefficients loaded from `kl-short.npz`/`kl-long.npz`), and `battery_figs`. The `generate_data.py` scripts regenerate the shipped data files and can be expensive to run (HIV about 33 min; MCMC about 75 min on 16 workers). When several worker processes run, give each one BLAS thread (`OPENBLAS_NUM_THREADS=1`): OpenBLAS otherwise starts one thread per CPU in every worker, which made the MCMC run about 7× slower.
- `uq_benchmarks/Elliptic_PDE_Random_Coefficients/pde_model.py` and `uq_benchmarks/PDE_Inverse_Problem_MCMC/pde_model.py` are byte-identical copies of one numpy/scipy solver (sparse LU plus a discrete adjoint; `specs/numpad_replacement.md`). `PDESolver` has `qoi`/`grad_qoi`/`qoi_and_grad`, a boundary average (Elliptic_PDE). `PDEMisfit` has `observations`/`set_data`/`misfit`/`misfit_grad`, a data misfit at the sensor rows `indz` (MCMC). Edit one copy, copy it over the other, and run `python uq_benchmarks/check_pde_model.py`.
