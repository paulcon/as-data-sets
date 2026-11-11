"""Write docs/notebook-index.md: every notebook, with its size and the methods it uses.

    python docs/make_notebook_index.py          # rewrite docs/notebook-index.md
    python docs/make_notebook_index.py --check  # exit 1 if the file is out of date

Run it from the repository root after adding or changing a notebook. It reads the code cells of
every */*/*.ipynb for active_subspaces calls, the number of inputs and the subspace dimensions from
SIZES below (read by hand from each notebook's text), and the kind of each data set from the end of its
entry in README.md. It uses only the standard library.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'notebook-index.md'

CATEGORIES = {  # directory: heading, in the README's order
    'aerospace_cfd': 'Aerospace and CFD',
    'hydrology_subsurface_geoscience': 'Hydrology, subsurface and geoscience',
    'climate_atmosphere_ocean': 'Climate, atmosphere and ocean',
    'ecology_agriculture': 'Ecology and agriculture',
    'energy': 'Energy',
    'plasma_fusion': 'Plasma and fusion',
    'epidemiology_systems_biology': 'Epidemiology and systems biology',
    'biomedical_cardiovascular': 'Biomedical and cardiovascular',
    'combustion_chemical_kinetics': 'Combustion and chemical kinetics',
    'structural_materials_manufacturing': 'Structural, materials and additive manufacturing',
    'electronics_circuits': 'Electronics and circuits',
    'uq_benchmarks': 'UQ benchmark collections',
}

# How the subspace is estimated: (label, test on the code)
SOURCES = [
    ('gradients', lambda c, est: 'as' in est or (not est and 'grads=' in c)),
    ('local linear gradients', lambda c, est: 'local_linear_gradients' in c or 'opg' in est),
    ('finite differences', lambda c, est: 'finite_difference_gradients' in c),
    ('linear fit', lambda c, est: 'ols' in est),
    ('quadratic fit', lambda c, est: 'qphd' in est),
    ('SIR', lambda c, est: 'sir' in est),
    ('given matrix', lambda c, est: 'subspace_from_matrix' in c),
    ('given basis', lambda c, est: 'subspace_from_basis' in c),
]
INPUTS = [
    ('uniform', r'ac\.Uniform\b'),
    ('Gaussian', r'ac\.Gaussian\b'),
    ('mixed', r'ac\.Product\b'),
]
USES = [
    ('surrogate', r'ac\.fit_surrogate\b'),
    ('prediction intervals', r'\.predict_interval\b|ac\.predict_interval\b'),
    ('activity scores', r'ac\.activity_scores\b'),
    ('active domain', r'ac\.active_domain\b'),
    ('inverse map', r'ac\.inverse_map\b'),
    ('MCMC', r'(?i)\bmcmc\b|metropolis'),
]

# Read from each notebook's text, not its code: data set -> (number of inputs, active subspace
# dimensions). A list of inputs gives one value for each model or parameterization with a different
# number of inputs. A list of dimensions gives the different dimensions the notebook finds across its
# outputs (or models), and "none" marks an output with no low-dimensional subspace. Add an entry when
# adding a notebook, and update it when a notebook's conclusion changes.
SIZES = {
    # Aerospace and CFD
    'AirfRANS_Airfoil_RANS_Forces': ('5', '1, 2'),
    'DTMB_5415_Hull_Resistance': ('14, 16', '1, 3, none'),
    'HyShotII_Scramjet': ('7', '1'),
    'LS89_Turbine_Blade_Envelopes': ('20', '1'),
    'N2_Supersonic_Jet': ('105', '1, 2'),
    'NACA0012_Airfoil': ('18', '1, 2'),
    'NACA2412_Flapped_Airfoil': ('3, 6', '1, 2'),
    'NACA4412_Shape_Lift_OpenFOAM': ('10', '2'),
    'ONERA_M6_Wing': ('50', '1, 2'),
    'Pipe_Flow_Dimensional_Analysis': ('5', '1, 2'),
    'RAE2822_SHAP_Design_Exploration': ('20', '1, 2'),
    'Stagnation_Temperature_Probe_Design': ('7', '1, 2'),
    'Transonic_Airfoil_Shape_Parameterizations': ('10, 11', '1, 2'),
    'Transonic_Fan_Blade_Performance_Maps': ('25', '2'),
    # Hydrology, subsurface and geoscience
    'Aquifer_Tracer_Advection_Diffusion': ('10', '3'),
    'CMF_Soil_Moisture_SPOTPY': ('4', '1, 2'),
    'CO2_Injection_Benchmark_Saturation': ('3', '1'),
    'Earthquake_Rupture_Seismic_Moment': ('3', '1'),
    'Freyberg_MODFLOW_Forecast': ('71', '2'),
    'Geological_CO2_Storage_Leakage': ('22, 30', '2'),
    'HBV_SASK_Oldman_Streamflow': ('12', '1, 2, 3'),
    'HyMod_Leaf_River_Rainfall_Runoff': ('5', '1, 2, none'),
    'Kaesbach_Catchment_Subsurface_Flow': ('32', '1, 2, none'),
    'Karst_Aquifer_LuKARS_Calibration': ('21', '2'),
    'Kurchatov_Sr90_Groundwater_Migration': ('20', '1'),
    'Land_Surface_Energy_Fluxes': ('19', '1'),
    'Photosynthesis_Stomatal_Resistance': ('20', '1'),
    'PyGEM_Glacier_Mass_Balance': ('3, 5', '1'),
    'Raven_Salmon_River_Blended_Model': ('43', '1'),
    'Runoff_Permeability_Field': ('100', '1'),
    'Subglacial_Drainage_GlaDS': ('8', '1, 2'),
    # Climate, atmosphere and ocean
    'Arctic_Snow_Depth_SMRT': ('5', '1, 2'),
    'Atmospheric_Selenium_Lifetime': ('34', '2'),
    'FaIR_Climate_Response_Calibration': ('45', '1, 4, none'),
    'Hector_Matilda_Warming': ('6', '1'),
    'Ice_Sheet_Basal_Melt_Ensemble': ('15', '1, 2'),
    'Lorenz96_Initial_State_Inference': ('37', '1, none'),
    'Marine_Carbonate_System_PyCO2SYS': ('14', '1, 2'),
    'Parcel_Droplet_Activation': ('8', '1, 2'),
    # Ecology and agriculture
    'SIPNET_Ecosystem_Carbon_Flux': ('8', '1, 4'),
    # Energy
    'BISON_Fuel_Performance_Surrogates': ('13', '1, 2'),
    'European_LV_Feeder_Probabilistic_Power_Flow': ('55, 110', '1'),
    'HTGR_Microreactor_Control_Drums': ('8', '1'),
    'Lillgrund_FLORIS_Wake_Model': ('6', '3'),
    'Lithium_Ion_Battery': ('19', '1'),
    'Lithium_Ion_DFN_Drive_Cycle': ('24', '1, 2'),
    'Lithium_Ion_Impedance_Calibration': ('18', '1'),
    'Offshore_Wind_Turbine_Fatigue': ('5', '1, 2, 3'),
    'PyPSA_Eur_Investment_Ranges': ('5', '1'),
    'Residential_Building_Stock_Energy_Use': ('15', '1, 2'),
    'Single_Diode_Solar_Cell': ('5', '1'),
    # Plasma and fusion
    'EU_DEMO_PROCESS_Major_Radius': ('6', '1'),
    'Hall_Thruster_Multi_Component': ('18', '1, 2'),
    'JAG_ICF_Implosion_Scalars': ('5', '1'),
    'MHD_Generator': ('5', '1, 2'),
    'Vlasov_Poisson_Instability_Growth_Rate': ('3, 5, 6', '1, 2'),
    # Epidemiology and systems biology
    'COVID19_Exit_Strategies_virsim': ('6, 7, 8', '2, 3'),
    'COVID19_SEIR_Epidemic_Peak': ('3', '1'),
    'CityCOVID_Chicago_Calibration': ('4', '1'),
    'CovidSim_Cumulative_Deaths': ('51', '2'),
    'EMT_Switch_ODE': ('11', '1'),
    'Ebola_SEIR': ('8', '2'),
    'HIV_Long_Term_Dynamics': ('27', '1, 2'),
    'HIV_Three_Stage_Reservoir': ('15', '1'),
    'MAPK_Cascade_Bistable_Oscillatory': ('7', '1, none'),
    'Malaria_Bed_Net_Reproduction_Number': ('20', '1'),
    'Minnesota_COVID19_Deaths': ('8', '3'),
    'STAT5_Dimerization_Signaling': ('6', '1'),
    'Zika_SEIR_SEI_Brazil': ('10', '2'),
    # Biomedical and cardiovascular
    'Atrial_Cell_Models_Courtemanche_Maleckar': ('8, 10, 20, 22', '1, 2, 3'),
    'CiPA_Ventricular_Proarrhythmia_Metrics': ('9', '1, 2'),
    'Cole_Cole_Cell_Monolayer_Impedance': ('4', '1, 2'),
    'Four_Chamber_Heart_Electromechanics': ('29, 45', '1'),
    'Hodgkin_Huxley_Action_Potential': ('11', '1, 2'),
    'Luo_Rudy_Cardiac_Action_Potential': ('6', '1, none'),
    'Pulmonary_Pulse_Wave_Network': ('9', '1, 2'),
    'Pyloric_Circuit_Energy_Consumption': ('31', '1'),
    'Rabbit_Ventricular_Action_Potential': ('15', '1'),
    'Tumour_Competition_ODE': ('6', '4, none'),
    # Combustion and chemical kinetics
    'Ammonia_Syngas_Optimized_Kinetics': ('123', '1'),
    'Annular_Combustor_Thermoacoustics': ('38', '3'),
    'H2_Flame_Kinetics_Bayesian_Calibration': ('31', '1, 2'),
    'H2_NH3_Combustor_CFD_NOx': ('3', '1, 2'),
    'H2_O2_Ignition_Delay_Kinetics': ('19, 36', '1'),
    'NH3_H2_Kinetics_Uncertainty': ('52', '1'),
    'Shared_Subspace_H2_Ignition_Flame': ('33', '1'),
    # Structural, materials and additive manufacturing
    'Epoxy_MD_Force_Field_Sensitivity': ('103', '1'),
    'Powder_Bed_Fusion_Residual_Stress': ('6', '1'),
    'Steel_Arch_Inplane_Buckling': ('7', '1, 2'),
    'Two_Material_Thermal_Compliance': ('512', '1'),
    'Wrench_Elasticity_Displacement': ('318', '3'),
    # Electronics and circuits
    'Cavity_Shielding_Effectiveness': ('6', '1, 2'),
    'Two_Stage_CMOS_Op_Amp': ('7', '1, 2, 3'),
    # UQ benchmark collections
    'Elliptic_PDE_Random_Coefficients': ('100', '1'),
    'Fire_Satellite_Coupled_System': ('9', '1, 2'),
    'Golinski_Speed_Reducer': ('6', '1'),
    'Hat_Stiffened_Panel_Shear_Buckling': ('10', '1'),
    'OpenAeroStruct_Wing_Aerostructural': ('12', '1, 2'),
    'PDE_Inverse_Problem_MCMC': ('2, 100', '1, 2'),
    'Point_Heat_Source_Consolidation': ('8', '1, 2'),
    'River_Flood_Dyke_Overflow': ('8', '1'),
    'Simply_Supported_Tube_Deflection': ('6', '2'),
    'Speed_Reducer_Shaft_Limit_State': ('5', '1'),
    'Steel_Column_Reliability': ('9', '1'),
    'Sulfur_Aerosol_Radiative_Forcing': ('9', '1'),
    'Truss_23_Bar_Midspan_Deflection': ('10', '1'),
    'Two_DOF_Equipment_Structure_Oscillator': ('7', '2'),
    'Undamped_Oscillator_Limit_State': ('6', '2'),
}


def kinds():
    """Data set name -> kind, from the italic note that ends each README entry."""
    out = {}
    for line in (ROOT / 'README.md').read_text().splitlines():
        m = re.match(r'\* \*\*\[([^\]]+)\]', line)
        if not m:
            continue
        note = re.findall(r'(?:^|\s)_([A-Z][^_]*?)[._]', line)
        text = note[-1] if note else ''
        if text.startswith('Both'):
            out[m.group(1)] = 'both'
        elif text.startswith('Python model'):
            out[m.group(1)] = 'Python model'
        elif text.startswith('Stored data'):
            out[m.group(1)] = 'stored data'
        else:
            out[m.group(1)] = '?'
    return out


def row(nb_path, kind):
    nb = json.loads(nb_path.read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    est = set(re.findall(r"estimator\s*=\s*['\"]([a-z_]+)", code))
    sources = [label for label, test in SOURCES if test(code, est)]
    inputs = [label for label, pat in INPUTS if re.search(pat, code)]
    uses = [label for label, pat in USES if re.search(pat, code)]
    name = nb_path.parent.name
    if name not in SIZES:
        sys.exit(f'{name}: add its number of inputs and subspace dimensions to SIZES in docs/make_notebook_index.py')
    m, n = SIZES[name]
    link = f'[{name}](../{nb_path.parent.parent.name}/{name}/{nb_path.name})'
    return f'| {link} | {kind} | {m} | {n} | {", ".join(sources)} | {", ".join(inputs)} | {", ".join(uses)} |'


def build():
    kind = kinds()
    lines = [
        '# Notebooks by method',
        '',
        'Every notebook, with its number of inputs, the dimension of the active subspace it finds, how it estimates the subspace, how it models the inputs, and what it does '
        'with the subspace beyond the eigenvalue, eigenvector and sufficient summary plots. '
        '[using-the-notebooks.md](using-the-notebooks.md) explains the terms. '
        'Written by `docs/make_notebook_index.py` from the notebooks\' code, with the inputs and dimensions read from the '
        'notebooks\' text and kept in the script; do not edit by hand.',
        '',
        '- **Inputs (m)**: the number of inputs, the dimension of the input space the subspace is computed in. '
        'A list gives one value for each model, data set or parameterization in the notebook with a different number of inputs.',
        '- **Dimension (n)**: the dimension of the active subspace the notebook settles on, from the eigenvalue gaps '
        'and the fits on the active variables. A list gives the different dimensions found for different outputs or models; '
        '*none* marks an output with no low-dimensional subspace.',
        '- **Subspace from**: *gradients* (the gradient-based estimator, with gradients from files or the model); '
        '*local linear gradients* (gradients estimated from values by local linear fits); '
        '*finite differences*; *linear fit* or *quadratic fit* (a global least-squares model of the values); '
        '*SIR* (sliced inverse regression); *given matrix* or *given basis* (computed in the notebook, e.g. by quadrature).',
        '- **Input distribution**: *uniform* on a box, *Gaussian* (possibly correlated), or *mixed* (a product of uniform and Gaussian blocks).',
        '',
    ]
    nbs = sorted(ROOT.glob('*/*/*.ipynb'))
    for cat, heading in CATEGORIES.items():
        rows = [row(p, kind.get(p.parent.name, '?')) for p in nbs
                if p.parent.parent.name == cat and p.stem == p.parent.name]
        lines += [f'## {heading}', '',
                  '| Notebook | Kind | Inputs (m) | Dimension (n) | Subspace from | Input distribution | Also |',
                  '|---|---|---|---|---|---|---|', *rows, '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if the index is out of date')
    text = build()
    if ap.parse_args().check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f'{OUT.relative_to(ROOT)} is out of date: run python docs/make_notebook_index.py')
        print(f'{OUT.relative_to(ROOT)} is up to date')
        return
    OUT.write_text(text)
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
