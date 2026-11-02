"""Write the data files of CO2_Injection_Benchmark_Saturation.

    python generate_data.py                         # inputs.csv, outputs.csv, gradients.csv
    python generate_data.py --out /tmp/co2 --check  # write elsewhere and compare with the files here
    python generate_data.py --from-archive DIR      # also rewrite the two stored CSVs from the
                                                    # unzipped Zenodo archive DIR/uqBenchmarkData

Stored data (from Zenodo 10.5281/zenodo.933827, uqBenchmarkData.zip, BSD 3-clause):
  benchmark_input_samples.csv  the 10,000 coded samples (ir, kd, po) of InputParameters.csv
  mc_reference_mean_std.csv    the Monte Carlo reference curves of MCrefs_MeanStd.csv
Both are converted losslessly (float values identical; checked on writing).

Model data (written from co2_benchmark_model.py, no randomness):
  inputs.csv     10,000 x 3 physical inputs (Q in m^2/s, n, phi)
  outputs.csv    10,000 x 6: S at r = 50, 75, 100, 125, 150 m and front radius r_f in m, t = 100 d
  gradients.csv  10,000 x 3: d r_f / d(Q, n, phi), physical units

Only numpy, scipy and the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import co2_benchmark_model as cm  # noqa: E402

PROVENANCE = ('co2_benchmark_model.py: exact entropy (Buckley-Leverett) solution of Eq. (11) of '
              'Koeppel et al. 2019, t = 100 d, C_Q = %g, phi = %g + %g po; written by generate_data.py'
              % (cm.C_Q, cm.A_PHI, cm.B_PHI))


def read_csv(path):
    """Read a CSV of this directory: '#' lines, then comma-separated numbers."""
    return np.loadtxt(path, delimiter=',', comments='#', ndmin=2)


def _save(path, A, header):
    np.savetxt(path, A, fmt='%.17g', delimiter=',', header=header)
    if not np.array_equal(read_csv(path), A):
        raise RuntimeError(f'{path} did not round-trip exactly')


def convert_archive(src, out):
    """Lossless conversion of the two stored files of the Zenodo archive."""
    C = np.loadtxt(os.path.join(src, 'InputParameters.csv'), delimiter=',', skiprows=1)
    _save(os.path.join(out, 'benchmark_input_samples.csv'), C,
          'Koeppel et al. (2019) benchmark: 10,000 coded input samples, InputParameters.csv of '
          'Zenodo 10.5281/zenodo.933827 (BSD 3-clause), values unchanged\n'
          'ir (injection-rate perturbation theta1; Q = Cp qCO2 (1 + ir)),'
          'kd (relative permeability degree n = 2 + 2 kd),'
          'po (coded porosity; phi = A_PHI + B_PHI po with A_PHI and B_PHI fitted in DATA.md)')
    with open(os.path.join(src, 'MCrefs_MeanStd.csv')) as fh:
        cols = fh.readline().strip()
    D = np.loadtxt(os.path.join(src, 'MCrefs_MeanStd.csv'), delimiter=',', skiprows=1)
    _save(os.path.join(out, 'mc_reference_mean_std.csv'), D,
          'Koeppel et al. (2019) Monte Carlo reference (10^4 runs of the authors\' finite-volume '
          'code), CO2 saturation mean (e..) and std (s..) at t = 100 d, MCrefs_MeanStd.csv of '
          'Zenodo 10.5281/zenodo.933827 (BSD 3-clause), values unchanged; x in m; al = joint, '
          'ir/kd/po = only that input uncertain\n' + cols)
    return C


def compute(C):
    Q, n, phi = cm.coded_to_physical(*C.T)
    X = np.column_stack([Q, n, phi])
    S = cm.saturation_field(cm.RADII, Q, n, phi)
    rf = np.array([cm.front_radius(*x) for x in X])
    G = np.array([cm.grad_front_radius(*x) for x in X])
    return X, np.column_stack([S, rf]), G


def write(out, X, F, G):
    _save(os.path.join(out, 'inputs.csv'), X,
          '10,000 physical inputs from benchmark_input_samples.csv via co2_benchmark_model.'
          'coded_to_physical\nQ (m^2/s),n (1),phi (1)')
    rad = ','.join(f'S_r{int(r)} (1)' for r in cm.RADII)
    _save(os.path.join(out, 'outputs.csv'), F,
          'CO2 saturation at radii r (m) and front radius at t = 100 d; ' + PROVENANCE +
          '\n' + rad + ',r_front (m)')
    _save(os.path.join(out, 'gradients.csv'), G,
          'gradient of r_front with respect to the physical inputs (Q, phi exact; n by central '
          'difference of the shock speed, h = 1e-4); ' + PROVENANCE +
          '\ndrf_dQ (m per m^2/s),drf_dn (m),drf_dphi (m)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--from-archive', default=None)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if os.path.abspath(a.out) == HERE and a.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    if a.from_archive:
        C = convert_archive(a.from_archive, a.out)
    else:
        C = read_csv(os.path.join(HERE, 'benchmark_input_samples.csv'))
    X, F, G = compute(C)
    write(a.out, X, F, G)
    print(f'wrote {len(X)} samples to {a.out} in {time.perf_counter() - t0:.1f} s')
    if a.check:
        for name in ('inputs.csv', 'outputs.csv', 'gradients.csv', 'benchmark_input_samples.csv',
                     'mc_reference_mean_std.csv'):
            p_new, p_old = os.path.join(a.out, name), os.path.join(HERE, name)
            if not os.path.exists(p_new) or os.path.samefile(p_new, p_old):
                continue
            A, B = read_csv(p_new), read_csv(p_old)
            rel = np.max(np.abs(A - B)) / max(np.max(np.abs(B)), 1e-300)
            print(f'{name}: max |new - shipped| / max |shipped| = {rel:.1e}')


if __name__ == '__main__':
    main()
