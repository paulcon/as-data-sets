# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of data sets plus Jupyter notebooks that apply **active subspaces** (dimension reduction for simulation models) to science/engineering models. There is no package, build system, test suite, or linter. The top-level directories are 12 application categories in snake_case (`aerospace_cfd/`, `energy/`, `uq_benchmarks/`, …), each with a `README.md`, plus `docs/` for the guides. Each data set is a self-contained case study one level down, e.g. `epidemiology_systems_biology/HIV_Long_Term_Dynamics/`: one notebook named after its directory, its data files, a `DATA.md`, and sometimes helper `.py` modules. The README has the per-dataset descriptions and publication links, grouped by category.

**`CONTRIBUTING.md` holds the rules for data sets** (layout, the two kinds, notebook conventions, DATA.md, licensing, listing). Follow it when adding or changing a data set. `docs/` holds the user guides: `using-the-notebooks.md`, `notebook-index.md` (written by `docs/make_notebook_index.py`; rerun it after adding or changing a notebook, `--check` to verify), and `templates/DATA.md`.

The public repository is for users: readers of the notebooks and contributors. Keep records of decisions, reviews and work in progress out of it, and out of notebooks, DATA.md files and code comments. Don't cite internal decisions ("author, date"), PRs, or how a notebook used to be written; state the fact or the rule itself.

## Environment

- **The local `.venv` (the PyCharm interpreter) is Python 3.11** with `active_subspaces` 1.0.0.dev0, installed editable from `../ActiveSubspacesRevision/upstream-v1/src`. Its API is top-level functions such as `ac.active_subspace`, `ac.finite_difference_gradients`, `ac.local_linear_gradients`, `ac.fit_surrogate`, `ac.PolynomialSurrogate`/`ac.RBFSurrogate`, `ac.active_domain`, and `ac.plotting`. **`ac.utils` and `ac.subspaces.Subspaces` do not exist in it** (they are the 2016 API, on the tag `legacy-master`). Check that source tree for signatures rather than guessing.
- Every notebook's committed outputs come from `.venv`. Report results from `.venv` only. Values at the rounding level (null eigenvalues, model-vs-file differences near 1e-16) change between platforms, so the notebook text gives them as orders of magnitude.
- `.venv` has no `pandas`; no notebook uses it.

## Commands

```bash
source .venv/bin/activate
jupyter lab                                   # interactive
# Execute a notebook headless. Run it from inside its directory, because notebooks and helper modules load data by relative path:
cd epidemiology_systems_biology/HIV_Long_Term_Dynamics && jupyter nbconvert --to notebook --execute --inplace HIV_Long_Term_Dynamics.ipynb
python docs/make_notebook_index.py            # after adding or changing a notebook
```

Jupyter renders each markdown cell on its own, so a reference link resolves only in the cell that defines it. Write links in a notebook inline, `[[1]](https://doi.org/...)`, unless the definition is in the same cell. Where the local, git-ignored script `check_notebook_links.py` is present at the repository root, run `python check_notebook_links.py` (all notebooks, or pass paths; exits 1 on an unresolved link) before committing a new or edited notebook.

## Branches

The default branch is `main`. It is protected: every change, the owner's included, lands through a PR merged with "Create a merge commit". Force-pushes and deletion are blocked. The 2016 notebooks are on the tag `legacy-master`.

## Two kinds of data sets

Every data set is **stored data** or a **Python model**; MHD_Generator has one part of each. The README tags each entry, and `CONTRIBUTING.md` ("Decide which kind of data set it is") has the rules. In short:

- **Stored data**: an outside code (SU2, ParFlow-CLM, Drekar, FAST, MATLAB, …) produced the files, and they are the only record. Never regenerate or "correct" their values. Format conversions must be lossless and scripted. Rows are dropped in the notebook by a stated rule or a flag, never deleted from the file. Data made for this repository with an outside code (SMRT, Cantera, ngspice, SIPNET, …) are stored data with a `generate_data.py` that exits with an install message when the code is missing; the code is rerun only if a problem is found in the shipped data. Modules that drive an outside code are `*_driver.py`. Accuracy numbers that no notebook or generating script prints come from `check_accuracy.py` (same rule for missing outside codes).
- **Python model**: the model is a numpy/scipy `*_model.py` module in the directory that loads its files relative to `__file__` (e.g. `HIV_model.py`, `pde_model.py`); a closed-form model may stay in a notebook cell. Shipped outputs are written by `generate_data.py` with saved seeds, and that script must reproduce them.

## Conventions in the notebooks

- Inputs are declared as a v1 input space and passed in physical units: `ac.Uniform(lower, upper, names=)` (or `ac.Uniform.reference(m)` for data already on `[-1, 1]^m`), `ac.Gaussian`, or `ac.Product` for mixed uniform × Gaussian inputs. v1 normalizes to `[-1, 1]^m` internally, so physical gradients are passed as they are.
- Eigenvector signs follow the published figures, set with `ac.orient`.
- The typical flow is: load samples (inputs X, outputs f, and gradients df when available) → build the subspace from gradients, or from local-linear/global-linear approximations when gradients are missing → plot eigenvalues and eigenvectors → make a sufficient-summary plot of `f` against `X @ W1` → optionally fit a response surface on the active variables.
- Helper modules are imported from the notebook's own directory. Examples: `HIV_model.Tcells`, `ebola_model.R0`, `hartmann_model.uavg`, `single_diode.pmax`, `pde_model.PDESolver` (KL-expansion coefficients loaded from `kl-short.npz`/`kl-long.npz`), and `battery_figs`. The `generate_data.py` scripts regenerate the shipped data files and can be expensive to run (HIV about 33 min; MCMC about 75 min on 16 workers). When several worker processes run, give each one BLAS thread (`OPENBLAS_NUM_THREADS=1`): OpenBLAS otherwise starts one thread per CPU in every worker, which made the MCMC run about 7× slower.
- `uq_benchmarks/Elliptic_PDE_Random_Coefficients/pde_model.py` and `uq_benchmarks/PDE_Inverse_Problem_MCMC/pde_model.py` are byte-identical copies of one numpy/scipy solver (sparse LU plus a discrete adjoint). `PDESolver` has `qoi`/`grad_qoi`/`qoi_and_grad`, a boundary average (Elliptic_PDE). `PDEMisfit` has `observations`/`set_data`/`misfit`/`misfit_grad`, a data misfit at the sensor rows `indz` (MCMC). Edit one copy, copy it over the other, and run `python uq_benchmarks/check_pde_model.py`.
