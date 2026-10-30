# Description

*Active subspaces* are part of an emerging set of tools for subspace-based dimension reduction in complex simulation models in science and engineering. This repository contains data sets and associated [Jupyter](http://jupyter.org) notebooks that apply active subspace methods to a wide range of science and engineering models. The Python implementations of the methods are available on [Github](http://github.com/paulcon/active_subspaces/). More information on active subspaces can be found at [activesubspaces.org](http://activesubspaces.org).

Each data set's directory contains a Jupyter notebook (`.ipynb`) and the data or model it analyzes. Data files are mostly text (`.txt`, `.csv`); PDE_Inverse_Problem_MCMC's chains are `.mat` files. Github renders the notebooks when you click an `.ipynb` file, but its rendering of mathematical expressions is not as robust as [nbviewer.org](https://nbviewer.org/). To view a notebook there, paste its Github link into the box at [nbviewer.org](https://nbviewer.org/). Links to notebooks must use the `main` branch (`blob/main/...`): the default branch was renamed from `master` to `main` in October 2026, and nbviewer links that name `master` no longer work. Each notebook also has links to associated publications for more details.

The notebooks were written in 2016 for Python 2 and the pre-1.0 `active_subspaces` API. They are being ported to Python 3 and `active_subspaces` 1.0; until a notebook is ported, it does not run with the packages in `requirements.txt`. Ported so far: Elliptic_PDE_Random_Coefficients. The 2016 version of the repository is the tag [`legacy-master`](https://github.com/paulcon/as-data-sets/tree/legacy-master).

# Data sets

The data sets are grouped by application area, one directory per category. Each data set is one of two kinds, marked at the end of its entry (MHD_Generator has one part of each):

* **Stored data:** an outside simulation code (CFD, hydrology, battery, wind turbine, solar cell) generated the data. The files in the directory are the only record of those runs and cannot be regenerated from this repository.
* **Python model:** the model that generates the data is Python code in the directory (a `*_model.py` module, or a closed-form formula in the notebook). The notebook can call it as needed, and a `generate_data.py` script regenerates any shipped data files.

## Aerospace and CFD (`aerospace_cfd/`)

* **[HyShotII_Scramjet](aerospace_cfd/HyShotII_Scramjet/):** a 7-parameter multiphysics model of a hypersonic scramjet. [[paper]](https://doi.org/10.1016/j.jcp.2015.09.001) _Stored data._
* **[N2_Supersonic_Jet](aerospace_cfd/N2_Supersonic_Jet/):** a 105-parameter shape-design model of a Lockheed Martin N+2 supersonic passenger jet (lift, drag, equivalent area; adjoint gradients). [[dissertation]](https://purl.stanford.edu/xx611nd3190) _Stored data._
* **[NACA0012_Airfoil](aerospace_cfd/NACA0012_Airfoil/):** an 18-parameter model for a two-spatial-dimensional airfoil. [[chapter 5]](https://doi.org/10.1137/1.9781611973860) _Stored data._
* **[ONERA_M6_Wing](aerospace_cfd/ONERA_M6_Wing/):** a 50-parameter model for a three-spatial-dimensional transonic wing. [[paper]](https://doi.org/10.2514/6.2014-1171) _Stored data._

## Hydrology, subsurface and geoscience (`hydrology_subsurface_geoscience/`)

* **[Land_Surface_Energy_Fluxes](hydrology_subsurface_geoscience/Land_Surface_Energy_Fluxes/):** a 19-parameter integrated surface/subsurface hydrologic model. [[paper]](https://doi.org/10.1016/j.cageo.2015.07.001) _Stored data._
* **[Photosynthesis_Stomatal_Resistance](hydrology_subsurface_geoscience/Photosynthesis_Stomatal_Resistance/):** a 20-parameter model of stomatal resistance in an integrated surface/subsurface hydrologic model. [[paper]](https://doi.org/10.1175/JHM-D-16-0053.1) [[thesis]](http://hdl.handle.net/11124/170080) _Stored data._
* **[Runoff_Permeability_Field](hydrology_subsurface_geoscience/Runoff_Permeability_Field/):** a 100-parameter model for subsurface permeability for groundwater flow. [[paper]](https://doi.org/10.1016/j.advwatres.2016.03.020) _Stored data._

## Energy: batteries, PV, wind, nuclear, power grids (`energy/`)

* **[Lithium_Ion_Battery](energy/Lithium_Ion_Battery/):** a 19-parameter model of a lithium-ion battery. [[paper]](https://doi.org/10.1002/sam.11347) [[arXiv]](https://arxiv.org/abs/1606.08770) _Stored data._
* **[Offshore_Wind_Turbine_Fatigue](energy/Offshore_Wind_Turbine_Fatigue/):** a 5-parameter model of offshore wind turbine fatigue. [[model]](https://doi.org/10.1002/we.1870) _Stored data._
* **[Single_Diode_Solar_Cell](energy/Single_Diode_Solar_Cell/):** a 5-parameter model for a lumped-parameter single-diode photovoltaic solar cell. [[paper]](https://doi.org/10.1002/sam.11281) _Stored data._

## Plasma and fusion (`plasma_fusion/`)

* **[MHD_Generator](plasma_fusion/MHD_Generator/):** a 5-parameter magnetohydrodynamics power generation model. [[paper]](https://doi.org/10.1002/sam.11355) [[arXiv]](https://arxiv.org/abs/1609.01255) _Both: Python model for the Hartmann problem; stored data for the 3-D generator._

## Epidemiology and systems biology (`epidemiology_systems_biology/`)

* **[Ebola_SEIR](epidemiology_systems_biology/Ebola_SEIR/):** an 8-parameter model for the geographic spread of Ebola: the basic reproduction number R0 of a 9-parameter SEIR fit. [[paper]](https://doi.org/10.1016/j.amc.2017.11.039) [[arXiv]](https://arxiv.org/abs/1603.04955) _Python model (closed form, in the notebook)._
* **[HIV_Long_Term_Dynamics](epidemiology_systems_biology/HIV_Long_Term_Dynamics/):** a 27-parameter dynamical system model for in-host HIV. [[paper]](https://doi.org/10.3934/mbe.2017040) [[arXiv]](https://arxiv.org/abs/1604.04588) _Python model (`HIV_model.py`)._ The paper's printed nbviewer link points to the old `master` branch and no longer works; use [this link](https://nbviewer.org/github/paulcon/as-data-sets/blob/main/epidemiology_systems_biology/HIV_Long_Term_Dynamics/HIV_Long_Term_Dynamics.ipynb) instead.

## UQ benchmark collections (`uq_benchmarks/`)

* **[Elliptic_PDE_Random_Coefficients](uq_benchmarks/Elliptic_PDE_Random_Coefficients/):** a 100-parameter elliptic PDE whose log-coefficient is a Karhunen–Loève expansion, at two correlation lengths. [[paper]](https://doi.org/10.1137/130916138) _Python model (`pde_model.py`)._ Ported to Python 3 and `active_subspaces` 1.0.
* **[PDE_Inverse_Problem_MCMC](uq_benchmarks/PDE_Inverse_Problem_MCMC/):** exploiting active subspaces to apply MCMC in a 100-parameter PDE model. [[paper]](https://doi.org/10.1137/15M1042127) _Python model (`pde_model.py`)._

Categories with no data sets yet: combustion and chemical kinetics (`combustion_chemical_kinetics/`); climate, atmosphere and ocean (`climate_atmosphere_ocean/`); structural, materials and additive manufacturing (`structural_materials_manufacturing/`); biomedical and cardiovascular (`biomedical_cardiovascular/`); ecology and agriculture (`ecology_agriculture/`); electronics and circuits (`electronics_circuits/`).

# Contributing

If you have a data set from a model, and you find evidence of an active subspace, feel free to contribute a notebook to our repository. Put it in its own directory under the matching category. Contact [Paul Constantine](http://inside.mines.edu/~pconstan) at Colorado School of Mines with questions or comments. Ryan Howard at Colorado School of Mines created the notebooks from the associated publications and data sets over the summer of 2016.

# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run each notebook from its own directory (for example `cd epidemiology_systems_biology/HIV_Long_Term_Dynamics`), since notebooks load their data by relative path.

# License

Code (notebooks and `.py` files) is under the MIT License ([LICENSE](LICENSE)). Data are under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)); credit the data sources named in each notebook.

# Acknowledgments

This material is based upon work supported by the U.S. Department of Energy Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number DE-SC-0011077.

