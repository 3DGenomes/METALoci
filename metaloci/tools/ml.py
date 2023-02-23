"""
Script to "paint" the Kamada-Kawai layaouts using a signal, grouping individuals by type
"""


# pylint: disable=invalid-name, wrong-import-position, wrong-import-order

import glob
import os
import pickle
import re
import sys
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict
from datetime import timedelta
from time import time

import geopandas as gpd
import libpysal as lp
import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from scipy.spatial import distance

from metaloci.misc import misc
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

parser = ArgumentParser(
    formatter_class=RawDescriptionHelpFormatter, description=description, add_help=False
)

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
    "-r",
    "--resolution",
    dest="reso",
    metavar="INT",
    type=int,
    required=True,
    help="Resolution of the cooler files, in Kb. Used in case multiple "
    "resolution "
    "Kamada-Kawai layouts exist in the same folder",
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

region_input = parser.add_argument_group(
    title="Region arguments", description="Choose one of the following options."
)

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

signal_arg = parser.add_argument_group(
    title="Signal arguments", description="Choose one of the following options:"
)

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
    help="Number of permutations to calculate the Local Moran's I p-value "
    "(default: %(default)d).",
)

optional_arg.add_argument(
    "-n",
    "--name",
    dest="dataset_name",
    default="",
    metavar="STR",
    type=str,
    help="Name of the dataset. By default corresponds to the name of the "
    "folder where coolers are stored.",
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
resolution = args.reso * 1000
n_cores = args.num_cores
n_permutations = args.perms
dataset_name = args.dataset_name
signipval = args.signipval
debug = args.debug

if not work_dir.endswith("/"):

    work_dir += "/"

if dataset_name == "":

    dataset_name = "WT_G2_20000"

if debug:

    table = [
        ["work_dir", work_dir],
        ["gene_file", regions],
        ["ind_file", signal_file],
        ["types", signals],
        ["reso", resolution],
        ["num_cores", n_cores],
        ["perms", n_permutations],
        ["dataset_name", dataset_name],
        ["debug", debug],
        ["signipval", signipval],
    ]

    print(table)
    sys.exit()

# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

INFLUENCE = 1.5
BFACT = 2
LIMITS = 2  # Create a "box" around the points of the Kamada-Kawai layout.

start_timer = time()

# Read signal file. Will only process the signals present in this list.
with open(signal_file) as signals_handler:

    signal_types = [line.rstrip() for line in signals_handler]

# Read region list. If its a region as parameter, create a dataframe.
# If its a path to a file, read that dataframe.
if re.compile("chr").search(regions):

    df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

else:

    df_regions = pd.read_table(regions)

# Read the chromosomes corresponding to the regions of interest, not to load useless data.
chrom_to_do = list(
    dict.fromkeys(
        re.compile("chr[0-9]*").findall(
            "\n".join([x for y in df_regions["coords"] for x in y.split(":")])
        )
    )
)

# Store in a dictionary the signal data
# Strata
#     CHROMOSOME : pandas.DataFrame(signal)
signal_data = {}

for chrom in chrom_to_do:

    signal_data[chrom] = pd.read_pickle(
        glob.glob(f"{os.path.join(work_dir, 'signal', chrom)}/*_signal.pkl")[0]
    )

# Iterating through all the genes of the regions file (regions to process).
for i, region_iter in df_regions.iterrows():

    region = region_iter.coords

    print(f"------> Working on region {region} [{i + 1}/{len(df_regions)}]\n")

    try:

        with open(
            f"{work_dir}{dataset_name}/{region.split(':', 1)[0]}/{region}_{args.reso}kb.mlo", "rb"
        ) as mlobject_handler:

            mlobject = pickle.load(mlobject_handler)

    except FileNotFoundError:

        print(
            ".mlo file not found for this region. Check dataset name and resolution. \nSkipping to next region."
        )

        continue

    # If the pickle file contains the
    if mlobject.kk_nodes is None:

        print(
            "Kamada-Kawai layout has not been calculated for this region. \nSkipping to next region."
        )

        continue

    mlobject.kk_coords = list(mlobject.kk_nodes.values())
    mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

    # Get average distance between consecutive points to define influence,
    # which should be ~2 particles of radius.
    mean_distance = mlobject.kk_distances.diagonal(1).mean()
    buffer = mean_distance * INFLUENCE
    mlobject.lmi_geometry, coord_id, geometry_data = lmi.construct_voronoi(mlobject, LIMITS, buffer)

    print(f"\tAverage distance between consecutive particles: {mean_distance:6.4f} [{buffer:6.4f}]")
    print(f"\tGeometry information for region {mlobject.region} saved in: {mlobject.save_path}")

    ########################################################################################################################

    # Load only signal for this specific region.
    region_signal = signal_data[mlobject.chrom][
        (signal_data[mlobject.chrom]["start"] >= int(mlobject.start))
        & (signal_data[mlobject.chrom]["end"] <= int(mlobject.end))
    ]

    if len(region_signal) != len(mlobject.kk_coords):

        tmp = len(mlobject.kk_coords) - len(region_signal)
        tmp = np.empty((tmp, len(region_signal.columns)))
        tmp[:] = np.nan

        region_signal = region_signal.append(
            pd.DataFrame(tmp, columns=list(region_signal)), ignore_index=True
        )

    signals_dict = defaultdict(list)

    for i, signal_type in enumerate(signal_types):

        signal_values = region_signal[signal_type]
        signals_dict[signal_type] = misc.signal_normalization(signal_values, 0.01, "01")

    all_lmi = {}

    for signal_type in signal_types:

        signal = []

        res = dict(filter(lambda item: signal_type in item[0], signals_dict.items()))

        for _, x in sorted(coord_id.items(), key=lambda item: item[1]):

            signal.append(np.nanmedian([res[key][x] for key in res]))

        signal_geometry = {"v": [], "geometry": []}

        for i, poly in geometry_data.iterrows():

            signal_geometry["v"].append(signal[coord_id[i]])
            signal_geometry["geometry"].append(poly.geometry)

        gpd_signal = gpd.GeoDataFrame(signal_geometry)  # Stored in Geopandas DataFrame to do LMI

        # Get weights for geometric distance
        # print("\tGetting weights and geometric distance for LM")
        y = gpd_signal["v"].values
        weights = lp.weights.DistanceBand.from_dataframe(gpd_signal, buffer * BFACT)
        weights.transform = "r"

        # Calculate Local Moran's I
        moran_local = Moran_Local(y, weights, permutations=n_permutations, n_jobs=n_cores)
        print(
            f"\tThere are a total of {len(moran_local.p_sim[(moran_local.p_sim < signipval)])} "
            f"significant points in Local Moran's I for signal {signal_type}"
        )
        chrom_number = mlobject.chrom[3:]

        df_lmi = defaultdict(list)

        for i, x in sorted(coord_id.items(), key=lambda item: item[1]):

            bin_start = int(mlobject.start) + (resolution * x) + x
            bin_end = bin_start + resolution

            df_lmi["Class"].append(signal_type)
            df_lmi["ID"].append(signal_type)

            df_lmi["bin_index"].append(x)
            df_lmi["bin_chr"].append(chrom_number)
            df_lmi["bin_start"].append(bin_start)
            df_lmi["bin_end"].append(bin_end)

            df_lmi["signal"].append(signal[x])

            df_lmi["moran_index"].append(i)
            df_lmi["moran_quadrant"].append(moran_local.q[i])
            df_lmi["LMI_score"].append(round(moran_local.Is[i], 9))
            df_lmi["LMI_pvalue"].append(round(moran_local.p_sim[i], 9))
            df_lmi["LMI_inv_pval"].append(round((1 - moran_local.p_sim[i]), 9))

        df_lmi = pd.DataFrame(df_lmi)

        # Changing the data types to the proper ones so the pickle file has a smaller size.
        df_lmi["Class"] = df_lmi["Class"].astype(str)
        df_lmi["ID"] = df_lmi["ID"].astype(str)

        df_lmi["bin_index"] = df_lmi["bin_index"].astype(np.uintc)
        df_lmi["bin_chr"] = df_lmi["bin_chr"].astype(str)
        df_lmi["bin_start"] = df_lmi["bin_start"].astype(np.uintc)
        df_lmi["bin_end"] = df_lmi["bin_end"].astype(np.uintc)

        df_lmi["signal"] = df_lmi["signal"].astype(np.half)

        df_lmi["moran_index"] = df_lmi["moran_index"].astype(np.uintc)
        df_lmi["moran_quadrant"] = df_lmi["moran_quadrant"].astype(np.uintc)
        df_lmi["LMI_score"] = df_lmi["LMI_score"].astype(np.half)
        df_lmi["LMI_pvalue"] = df_lmi["LMI_pvalue"].astype(np.half)
        df_lmi["LMI_inv_pval"] = df_lmi["LMI_inv_pval"].astype(np.half)
        # zSig y Zlag

        all_lmi[signal_type] = df_lmi

    mlobject.lmi_info = all_lmi

    # print(all_lmi)

    print(f"\n\tRegion done in {timedelta(seconds=round(time() - start_timer))}")
    print(
        f"\tLMI information for region {mlobject.region} will be saved in: {mlobject.save_path}\n"
    )

    with open(mlobject.save_path, "wb") as hamlo_namendle:

        mlobject.save(hamlo_namendle)

print(f"Total time spent: {timedelta(seconds=round(time() - start_timer))}.")
print("\nall done.")
