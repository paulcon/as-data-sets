"""Write this directory's CSV files from the HESS supplement of Erdal and Cirpka (2019).

The supplement (doi:10.5194/hess-23-3787-2019-supplement, CC BY 4.0) is a ZIP
holding a MATLAB app installer, ``Active Subspace Pilot.mlappinstall``. That
installer is itself a ZIP; its file ``fsroot/activeSubSpPilotData.mat``
(MATLAB v5, 2,987,972 bytes) holds the ensemble:

    Rorg     (10050, 32)  inputs as stored coordinates u in [0, 1]
    obsMTRX  (10050, 9)   outputs
    xlbl, xlblPRINT       input labels

Usage
-----
    python extract_data.py /path/to/activeSubSpPilotData.mat          # write the CSVs
    python extract_data.py /path/to/activeSubSpPilotData.mat --check  # compare only

The conversion is lossless: values are written with 17 significant digits and
``--check`` confirms that the files read back bit for bit (inputs_unit.csv,
outputs.csv) and that inputs.csv maps back to u within 1e-12. Nothing in the
.mat file is executed; scipy.io.loadmat only reads arrays.
"""

import argparse
import os
import sys

import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaesbach_inputs as ki  # noqa: E402

FILES = {
    "unit": os.path.join(HERE, "inputs_unit.csv"),
    "phys": os.path.join(HERE, "inputs.csv"),
    "out": os.path.join(HERE, "outputs.csv"),
}


def load_mat(path):
    d = scipy.io.loadmat(path)
    U, Y = np.asarray(d["Rorg"], float), np.asarray(d["obsMTRX"], float)
    labels = [str(x[0]) for x in d["xlbl"][0]]
    if U.shape != (10050, 32) or Y.shape != (10050, 9) or labels != ki.SUPPLEMENT_LABELS:
        raise SystemExit("unexpected contents of %s" % path)
    return U, Y


def headers():
    unit = ("Erdal & Cirpka (2019) HydroGeoSphere ensemble, 10050 runs: the 32 inputs exactly as "
            "stored in the supplement (coordinate u in [0,1]; see kaesbach_inputs.py for the map "
            "to physical units)",
            ",".join(n + " [u]" for n in ki.NAMES))
    phys = ("Erdal & Cirpka (2019) HydroGeoSphere ensemble, 10050 runs: the 32 inputs in physical "
            "units, computed from inputs_unit.csv by kaesbach_inputs.to_physical (map status per "
            "column in DATA.md)",
            ",".join("%s [%s]" % (n, u) for n, u in zip(ki.NAMES, ki.UNITS)))
    out = ("Erdal & Cirpka (2019) HydroGeoSphere ensemble, 10050 runs: the 9 outputs exactly as "
           "stored in the supplement (row i matches row i of inputs_unit.csv); residence-time unit "
           "not stated by the authors: Fig. 8 years = stored value / 17520 (see DATA.md)",
           ",".join("%s [%s]" % (n, u) for n, u in zip(ki.OUTPUT_NAMES, ki.OUTPUT_UNITS)))
    return unit, phys, out


def write(U, Y):
    hu, hp, ho = headers()
    np.savetxt(FILES["unit"], U, fmt="%.17g", delimiter=",", header="\n".join(hu))
    np.savetxt(FILES["phys"], ki.to_physical(U), fmt="%.17g", delimiter=",",
               header="\n".join(hp))
    np.savetxt(FILES["out"], Y, fmt="%.17g", delimiter=",", header="\n".join(ho))


def check(U, Y):
    Uf = np.loadtxt(FILES["unit"], delimiter=",")
    Yf = np.loadtxt(FILES["out"], delimiter=",")
    Xf = np.loadtxt(FILES["phys"], delimiter=",")
    ok_u = np.array_equal(Uf, U)
    ok_y = np.array_equal(Yf, Y) and np.array_equal(np.signbit(Yf), np.signbit(Y))
    err = np.max(np.abs(ki.to_unit(Xf) - U))
    print("inputs_unit.csv identical to Rorg: %s" % ok_u)
    print("outputs.csv identical to obsMTRX (signs of zeros included): %s" % ok_y)
    print("inputs.csv -> u, max |difference|: %.1e" % err)
    return ok_u and ok_y and err < 1e-12


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mat")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    U, Y = load_mat(a.mat)
    if not a.check:
        write(U, Y)
    sys.exit(0 if check(U, Y) else 1)


if __name__ == "__main__":
    main()
