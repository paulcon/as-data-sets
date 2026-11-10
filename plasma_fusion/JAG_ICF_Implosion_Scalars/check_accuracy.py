"""Recompute the numbers of DATA.md that neither the notebook nor convert_data.py prints.

    python check_accuracy.py                         # checks on the shipped CSVs
    python check_accuracy.py <icf-jag-10k.tar.gz>    # also the tarball's members (names, dtypes, shapes)

Numbers printed: the stored precision of the two files; the agreement of the README example of the
llnl/macc repository (its "sample 0") with the released arrays; the members of the tarball. Requires
numpy. Reads only; writes nothing.
"""

import os
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The example printed in README.md of github.com/llnl/macc (commit a0318c22) as "sample 0".
README_INPUT = [-0.07920084, 0.70821885, 0.377287, 0.12390906, 0.22148967]
README_SCALARS = [0.36831434, 0.36175176, 0.35908509, 0.38851718, 0.45318199, 0.17283457, 0.16303126,
                  0.36568428, 0.17283457, 0.03728897, 0.03728897, 0.12553939, 0.35908509, 0.17283457,
                  0.16303126, 0.35908509, 0.34737663, 0.16303126, 0.36175176, 0.45389942, 0.37021051,
                  0.22734619]


def main(argv):
    X = np.loadtxt(os.path.join(HERE, 'jag_icf_inputs.csv'), delimiter=',')
    S = np.loadtxt(os.path.join(HERE, 'jag_icf_scalars.csv'), delimiter=',')
    print(f'inputs {X.shape}, scalars {S.shape}')
    # Stored precision: inputs are float64 (17 significant digits written), scalars float32.
    s32 = S.astype(np.float32)
    print('scalars representable in float32 (value == float32(value) to 1e-8):',
          bool(np.abs(S - s32.astype(np.float64)).max() < 1e-8))
    print('float32 spacing at the largest scalar value: %.1e' % np.spacing(np.float32(S.max())))
    # README example against the released arrays
    d = np.abs(X - np.array(README_INPUT)).max(axis=1)
    print(f'README sample 0 input vs row 0: max |diff| = {d[0]:.3f}; nearest row {d.argmin()} at '
          f'{d.min():.3f} (max-norm)')
    print(f'README example lists {len(README_SCALARS)} scalars; the file has {S.shape[1]} per row')
    uniq = sorted(set(README_SCALARS))
    hits = sum(bool((np.abs(S - v) < 1e-7).any()) for v in uniq)
    print(f'distinct README example scalars found anywhere in the file (within 1e-7): {hits} of {len(uniq)}')
    row_hits = int(((np.abs(S[:, :, None] - np.array(uniq)) < 1e-7).sum(axis=(1, 2)) >= 3).sum())
    print(f'rows holding at least 3 of them: {row_hits}')
    if len(argv) > 1:
        with tarfile.open(argv[1], 'r:gz') as tf:
            for info in tf:
                fh = tf.extractfile(info)
                version = np.lib.format.read_magic(fh)
                read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                        else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = read(fh)
                print(f'member {info.name}: {info.size} bytes, {dtype}, shape {shape}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
