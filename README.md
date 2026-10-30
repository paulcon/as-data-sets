# Description

*Active subspaces* are part of an emerging set of tools for subspace-based dimension reduction in complex simulation models in science and engineering. This repository contains data sets and associated [Jupyter](http://jupyter.org) notebooks that apply active subspace methods to a wide range of science and engineering models. The Python implementations of the methods are available on [Github](http://github.com/paulcon/active_subspaces/). More information on active subspaces can be found at [activesubspaces.org](http://activesubspaces.org).

Each directory contains a Jupyter notebook (`.ipynb`) and, in most cases, the data set it analyzes, usually as text files (`.txt`, `.csv`; MCMC's chains are `.mat` files). Ebola's model is closed-form and Elliptic_PDE's is a PDE solver in the directory, so those notebooks evaluate their models directly; Elliptic_PDE ships only its KL bases (`.npz`). Github renders the notebooks when you click an `.ipynb` file, but its rendering of mathematical expressions is not as robust as [nbviewer.org](https://nbviewer.org/). To view a notebook there, paste its Github link into the box at [nbviewer.org](https://nbviewer.org/). Links to notebooks must use the `main` branch (`blob/main/...`): the default branch was renamed from `master` to `main` in October 2026, and nbviewer links that name `master` no longer work. Each notebook also has links to associated publications for more details.

The notebooks were written in 2016 for Python 2 and the pre-1.0 `active_subspaces` API. They are being ported to Python 3 and `active_subspaces` 1.0; until a notebook is ported, it does not run with the packages in `requirements.txt`. The 2016 version of the repository is the tag [`legacy-master`](https://github.com/paulcon/as-data-sets/tree/legacy-master).

# Data sets

The repository contains Jupyter notebooks for the following models / data sets.

* **Ebola:** an 8-parameter model for the geographic spread of Ebola: the basic reproduction number R0 of a 9-parameter SEIR fit. [[paper]](https://doi.org/10.1016/j.amc.2017.11.039) [[arXiv]](https://arxiv.org/abs/1603.04955)
* **Elliptic_PDE:** a 100-parameter elliptic PDE whose log-coefficient is a Karhunen–Loève expansion, at two correlation lengths. [[paper]](https://doi.org/10.1137/130916138)
* **HIV:** a 27-parameter dynamical system model for in-host HIV. [[paper]](https://doi.org/10.3934/mbe.2017040) [[arXiv]](https://arxiv.org/abs/1604.04588) The paper's printed nbviewer link points to the old `master` branch and no longer works; use [this link](https://nbviewer.org/github/paulcon/as-data-sets/blob/main/HIV/HIV.ipynb) instead.
* **HyShotII:** a 7-parameter multiphysics model of a hypersonic scramjet. [[paper]](https://doi.org/10.1016/j.jcp.2015.09.001)
* **Hydrology:** a 19-parameter integrated surface/subsurface hydrologic model. [[paper]](https://doi.org/10.1016/j.cageo.2015.07.001)
* **Lithium_Battery:** a 19-parameter model of a lithium-ion battery. [[paper]](https://doi.org/10.1002/sam.11347) [[arXiv]](https://arxiv.org/abs/1606.08770)
* **LM104X:** a 105-parameter shape-design model of a Lockheed Martin N+2 supersonic passenger jet (lift, drag, equivalent area; adjoint gradients). [[dissertation]](https://purl.stanford.edu/xx611nd3190)
* **MCMC:** exploiting active subspaces to apply MCMC in a 100-parameter PDE model. [[paper]](https://doi.org/10.1137/15M1042127)
* **MHD:** a 5-parameter magnetohydrodynamics power generation model. [[paper]](https://doi.org/10.1002/sam.11355) [[arXiv]](https://arxiv.org/abs/1609.01255)
* **NACA0012:** an 18-parameter model for a two-spatial-dimensional airfoil. [[chapter 5]](https://doi.org/10.1137/1.9781611973860)
* **NREL_Wind:** a 5-parameter model of offshore wind turbine fatigue. [[model]](https://doi.org/10.1002/we.1870)
* **ONERA-M6:** a 50-parameter model for a three-spatial-dimensional transonic wing. [[paper]](https://doi.org/10.2514/6.2014-1171)
* **SingleDiodePV:** a 5-parameter model for a lumped-parameter single-diode photovoltaic solar cell. [[paper]](https://doi.org/10.1002/sam.11281)
* **Stomatal_Resistance:** a 20-parameter model of stomatal resistance in an integrated surface/subsurface hydrologic model. [[paper]](https://doi.org/10.1175/JHM-D-16-0053.1) [[thesis]](http://hdl.handle.net/11124/170080)
* **Subsurface_Permeability:** a 100-parameter model for subsurface permeability for groundwater flow. [[paper]](https://doi.org/10.1016/j.advwatres.2016.03.020)

# Contributing

If you have a data set from a model, and you find evidence of an active subspace, feel free to contribute a notebook to our repository. Contact [Paul Constantine](http://inside.mines.edu/~pconstan) at Colorado School of Mines with questions or comments. Ryan Howard at Colorado School of Mines created the notebooks from the associated publications and data sets over the summer of 2016.

# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run each notebook from its own directory, since notebooks load their data by relative path.

# License

Code (notebooks and `.py` files) is under the MIT License ([LICENSE](LICENSE)). Data are under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)); credit the data sources named in each notebook.

# Acknowledgments

This material is based upon work supported by the U.S. Department of Energy Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number DE-SC-0011077.

