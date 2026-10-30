# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of data sets plus Jupyter notebooks that apply **active subspaces** (dimension reduction for simulation models) to science/engineering models. There is no package, build system, test suite, or linter. The top-level directories are 12 application categories in snake_case (`aerospace_cfd/`, `energy/`, `uq_benchmarks/`, …); each has a `README.md`, and some have no data sets yet. Each data set is a self-contained case study one level down, e.g. `epidemiology_systems_biology/HIV_Long_Term_Dynamics/`: one notebook named after its directory, its data files, and sometimes helper `.py` modules. The README has the per-dataset descriptions and publication links, grouped by category. Add a new data set the same way, and add a bullet for it to the README's "Data sets" list and a line to its category's `README.md`.

The directories were renamed and moved on 2026-10-30 (e.g. `HIV/` → `epidemiology_systems_biology/HIV_Long_Term_Dynamics/`, `LM104X/` → `aerospace_cfd/N2_Supersonic_Jet/`). `specs/` still uses the old names; the map is in `specs/README.md`.

## Two incompatible environments

This is the most important thing to know before running or editing anything:

- **The notebooks and scripts in the repo are legacy code**: their kernelspec is `python2` (2.7.11), several use Python 2 `print x` statements (`HyShotII_Scramjet`, `PDE_Inverse_Problem_MCMC`, `Offshore_Wind_Turbine_Fatigue`, `Single_Diode_Solar_Cell`, `Runoff_Permeability_Field` notebooks, plus `Elliptic_PDE_Random_Coefficients/pde_study.py` and `HIV_Long_Term_Dynamics/generate_data.py`), and they call the **old** `active_subspaces` API: `ac.subspaces.Subspaces()` with `.compute(df=...)`/`.partition()`/`.W1`/`.eigenvals`, `ac.utils.plotters.*`, `ac.utils.misc.BoundedNormalizer`, `ac.utils.response_surfaces.*`, `ac.utils.simrunners.*`, `ac.domains.*`.
- **The local `.venv` (the PyCharm interpreter) is Python 3.11** with `active_subspaces` 1.0.0.dev0, installed editable from `../ActiveSubspacesRevision/upstream-v1/src`. That is a rewritten API: top-level functions such as `ac.active_subspace`, `ac.finite_difference_gradients`, `ac.local_linear_gradients`, `ac.fit_surrogate`, `ac.PolynomialSurrogate`/`ac.RBFSurrogate`, `ac.active_domain`, and `ac.plotting`. **`ac.utils` and `ac.subspaces.Subspaces` do not exist in it.** Check that source tree for signatures rather than guessing.
- `.venv` also lacks `pandas` (used by most notebooks, imported as `pn`) and `numpad` (the autodiff library the legacy PDE notebooks and scripts were written for; `pde_model.py` no longer uses it, see below).

So no notebook runs unmodified in `.venv`. If you port a notebook, the job is three things: convert it to Python 3, translate the `ac.*` calls to the v1 API, and install any missing dependencies.

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

## Conventions in the notebooks

- Inputs are normalized to `[-1, 1]^m` before computing subspaces. The usual pattern is `ac.utils.misc.BoundedNormalizer(xl, xu)`, or the hand-written affine map `p = .5*(diag(xu-xl) @ XX.T + xu + xl).T`. Gradients taken with respect to physical inputs are then rescaled by `.5*(xu - xl)`.
- The typical flow is: load samples (inputs X, outputs f, and gradients df when available) → build the subspace from gradients, or from local-linear/global-linear approximations when gradients are missing → plot eigenvalues and eigenvectors → make a sufficient-summary plot of `f` against `X @ W1` → optionally fit a response surface on the active variables.
- Helper modules are imported from the notebook's own directory. Examples: `HIV_model.Tcells`, `pde_model.PDESolver` (KL-expansion coefficients loaded from `kl-short.npz`/`kl-long.npz`), and `battery_figs`. The `generate_data.py` / `*_study.py` scripts regenerate the shipped data files and can be expensive to run.
- `uq_benchmarks/Elliptic_PDE_Random_Coefficients/pde_model.py` and `uq_benchmarks/PDE_Inverse_Problem_MCMC/pde_model.py` are byte-identical copies of one numpy/scipy solver (sparse LU plus a discrete adjoint; `specs/numpad_replacement.md`). `PDESolver` has `qoi`/`grad_qoi`, a boundary average (Elliptic_PDE). `PDEMisfit` has `observations`/`set_data`/`misfit`/`misfit_grad`, a data misfit at the sensor rows `indz` (MCMC). The legacy MCMC callers still call `PDESolver` for the misfit until that notebook is ported. Edit one copy, copy it over the other, and run `python uq_benchmarks/check_pde_model.py`.
