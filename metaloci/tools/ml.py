"""
Script to "paint" the Kamada-Kawai layaouts using a signal, grouping individuals by type
"""


# pylint: disable=invalid-name, wrong-import-position, wrong-import-order

import pickle
import re
import sys
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from datetime import timedelta
from time import time

import pandas as pd

from metaloci.spatial_stats import lmi

description = (
    "This script adds signal data to a Kamada-Kawai layout and calculates Local Moran's I "
    "for every bin in the layout.\n"
)
description += "It groups the data signal by types/classes specified by the user.\n"
description += "The script will create a structure of folders and subfolders as follows:\n\n"
description += "WORK_DIR\n"
description += " |-datasets\n"
description += "    |-DATASET_NAME\n"
description += "       |-CHROMOSOME\n"
description += "          |-output_file\n"

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=description, add_help=False)

input_arg = parser.add_argument_group(title="Input arguments")

input_arg.add_argument(
    "-w",
    "--work-dir",
    dest="work_dir",
    metavar="PATH",
    type=str,
    required=True,
    help="Path to working directory.",
)

input_arg.add_argument(
    "-s",
    "--signal",
    dest="signal_file",
    metavar="FILE",
    type=str,
    required=True,
    help="Path to the file with the samples/signals to use.",
)

region_input = parser.add_argument_group(title="Region arguments", description="Choose one of the following options.")

region_input.add_argument(
    "-g",
    "--region",
    dest="region_file",
    metavar="PATH",
    type=str,
    help="Region to apply LMI in format chrN:start-end_midpoint.",
)

region_input.add_argument(
    "-G",
    "--region-file",
    dest="region_file",
    metavar="PATH",
    type=str,
    help="Path to the file with the regions of interest.",
)

signal_arg = parser.add_argument_group(title="Signal arguments", description="Choose one of the following options:")

signal_arg.add_argument(
    "-y",
    "--types",
    dest="signals",
    metavar="STR",
    type=str,
    nargs="*",
    action="extend",
    help="Space-separated list of signals to plot.",
)

signal_arg.add_argument(
    "-Y",
    "--types-file",
    dest="signals",
    metavar="PATH",
    type=str,
    help="Path to the file with the list of signals to plot, one per line.",
)

optional_arg = parser.add_argument_group(title="Optional arguments")

optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")

optional_arg.add_argument(
    "-t",
    "--cores",
    dest="num_cores",
    default=1,
    metavar="INT",
    type=int,
    help="Number of cores to use in the Local Moran's " "I calculation (default: %(default)d).",
)

optional_arg.add_argument(
    "-p",
    "--permutations",
    dest="perms",
    default=9999,
    metavar="INT",
    type=int,
    help="Number of permutations to calculate the Local Moran's I p-value " "(default: %(default)d).",
)

optional_arg.add_argument(
    "-v",
    "--pval",
    dest="signipval",
    default=0.05,
    metavar="FLOAT",
    type=float,
    help="P-value significance threshold (default: %(default)4.2f).",
)

optional_arg.add_argument("-u", "--debug", dest="debug", action="store_true", help=SUPPRESS)

args = parser.parse_args(None if sys.argv[1:] else ["-h"])

work_dir = args.work_dir
regions = args.region_file
signal_file = args.signal_file
signals = args.signals
n_cores = args.num_cores
n_permutations = args.perms
signipval = args.signipval
debug = args.debug

if not work_dir.endswith("/"):

    work_dir += "/"

if debug:

    table = [
        ["work_dir", work_dir],
        ["gene_file", regions],
        ["ind_file", signal_file],
        ["types", signals],
        ["num_cores", n_cores],
        ["perms", n_permutations],
        ["debug", debug],
        ["signipval", signipval],
    ]

    print(table)
    sys.exit()

INFLUENCE = 1.5
BFACT = 2

timer = time()

# Read region list. If its a region as parameter, create a dataframe.
# If its a path to a file, read that dataframe.
if re.compile("chr").search(regions):

    df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

else:

    df_regions = pd.read_table(regions)

# Read the signal of the chromosomes corresponding to the regions of interest,
# not to load useless data.
signal_data = lmi.load_signals(df_regions, work_dir=work_dir)

# Iterating through all the genes of the regions file (regions to process).
for i, region_iter in df_regions.iterrows():

    region_timer = time()

    region = region_iter.coords

    print(f"\n------> Working on region {region} [{i + 1}/{len(df_regions)}]\n")

    try:

        with open(
            f"{work_dir}{region.split(':', 1)[0]}/{re.sub(':|-', '_', region)}.mlo",
            "rb",
        ) as mlobject_handler:

            mlobject = pickle.load(mlobject_handler)

    except FileNotFoundError:

        print("\t.mlo file not found for this region.\nSkipping to next region.")

        continue

    # If the pickle file contains the
    if mlobject.kk_nodes is None:

        print("Kamada-Kawai layout has not been calculated for this region. \nSkipping to next region.")

        continue

    # Get average distance between consecutive points to define influence,
    # which should be ~2 particles of radius.
    mean_distance = mlobject.kk_distances.diagonal(1).mean()
    buffer = mean_distance * INFLUENCE
    mlobject.lmi_geometry = lmi.construct_voronoi(mlobject, buffer)

    print(f"\tAverage distance between consecutive particles: {mean_distance:6.4f} [{buffer:6.4f}]")
    print(f"\tGeometry information for region {mlobject.region} saved in: {mlobject.save_path}")

    ########################################################################################################################

    # Load only signal for this specific region.
    mlobject.signals_dict, signal_types = lmi.load_region_signals(mlobject, signal_data, signal_file)

    all_lmi = {}

    for signal_type in signal_types:

        all_lmi[signal_type] = lmi.compute_lmi(mlobject, signal_type, buffer * BFACT, n_permutations, signipval)

    mlobject.lmi_info = all_lmi

    print(f"\n\tRegion done in {timedelta(seconds=round(time() - region_timer))}")
    print(f"\tLMI information for region {mlobject.region} will be saved in: {mlobject.save_path}\n")

    with open(mlobject.save_path, "wb") as hamlo_namendle:

        mlobject.save(hamlo_namendle)

print(f"Total time spent: {timedelta(seconds=round(time() - timer))}.")
print("\nall done.")
