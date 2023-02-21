"""
Script to create the Kamada-Kaway layouts based on HiC data
"""
__author__ = "Iago Maceda Porto and Leo Zuber Ponce"

# pylint: disable=invalid-name, wrong-import-position, wrong-import-order

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter, SUPPRESS
import os
import pickle
import pathlib
import re
import subprocess as sp
from datetime import timedelta
from time import time

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
import networkx as nx
import cooler
from scipy.spatial import distance
from scipy.sparse import csr_matrix
import pyarrow.parquet as pq
from metaloci.misc import misc
from metaloci.graph_layout import kk
from metaloci import mlo
from metaloci.plot import plot
import numpy as np

description = """
This script adds signal data to a Kamada-Kawai layout and calculates Local Moran's I for every 
bin in the layout.
The script will create a structure of folders and subfolders as follows:
WORK_DIR
 |-DATASET_NAME
    |-CHROMOSOME
       |-output_files

The script will save the 'clean' matrix to a pickle file without applying any cutoff to the count 
data.

To find the appropiate cutoff, run the script with a list of cutoffs (space separated) using the 
-x option and the WHATEVER mode activated, -put_thing. For example:

03_KK_layouts.py -w WORK_DIR -l COOLER_DIR -g/-G REGION -r RESOLUTION -o NAME -x 0.5 0.15 put_thing

The script will show the plots of the matrix and the different cutoffs chosen, in descending order.
From the different plots choose the one you think fits better with your data.
If the script runs with only one cutoff, it will save the Kamada-Kawai layout data  to a pickle 
file"""

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
    "-c",
    "--cooler",
    dest="cooler_file",
    metavar="PATH",
    type=str,
    required=True,
    help="Complete path to the cooler file.",
)
input_arg.add_argument(
    "-r",
    "--resolution",
    dest="reso",
    metavar="INT",
    type=int,
    required=True,
    help="Resolution of the cooler files.",
)
region_arg = parser.add_argument_group(
    title="Region arguments", description="Choose one of the following options."
)
region_arg.add_argument(
    "-g",
    "--region",
    dest="regions",
    metavar="PATH",
    type=str,
    help="Path to the file with the regions of interest.",
)
region_arg.add_argument(
    "-G",
    "--region-file",
    dest="regions",
    metavar="PATH",
    type=str,
    help="Path to the file with the regions of interest.",
)

optional_arg = parser.add_argument_group(title="Optional arguments")
optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
optional_arg.add_argument(
    "-o",
    "--cutoff",
    dest="cutoff",
    nargs="*",
    type=float,
    action="extend",
    help="Percentage of top interactions to keep, space separated (default: " "0.2)",
)
optional_arg.add_argument(
    "-n",
    "--name",
    dest="dataset_name",
    metavar="STR",
    default="",
    help="Name of the dataset. By default corresponds to: " "COOLER.NAME_RESOLUTION",
)
optional_arg.add_argument(
    "-f", "--force", dest="force", action="store_true", help="Force rewriting existing data."
)
optional_arg.add_argument(
    "-p",
    "--plot",
    dest="save_plots",
    action="store_true",
    help="Plot the matrix, density and Kamada-Kawai plots, even when a "
    "single cutoff is selected.",
)
# TODO add sort of silent argument?

optional_arg.add_argument("-u", "--debug", dest="debug", action="store_true", help=SUPPRESS)

args = parser.parse_args(None if sys.argv[1:] else ["-h"])

work_dir = args.work_dir
cooler_file = args.cooler_file
regions = args.regions
resolution = args.reso * 1000
cutoffs = args.cutoff
dataset_name = args.dataset_name
force = args.force
save_plots = args.save_plots
debug = args.debug

if not work_dir.endswith("/"):

    work_dir += "/"

if dataset_name == "":

    if cooler_file.endswith(".cool") or cooler_file.endswith(".mcool"):

        dataset_name = cooler_file.rsplit("/", 1)[1].rsplit(".")[0] + "_" + str(resolution)

    else:

        dataset_name = cooler_file.rsplit("/", 1)[1] + "_" + str(resolution)

if cutoffs is None:

    cutoffs = [0.2]

# To sort the floats from the cutoffs
cutoffs.sort(key=float, reverse=True)

if debug:

    table = [
        ["work_dir", work_dir],
        ["cooler_dir", cooler_file],
        ["region_file", regions],
        ["reso", resolution],
        ["cutoffs", cutoffs],
        ["dataset_name", dataset_name],
        ["force", force],
        ["debug", debug],
    ]

    print(table)
    sys.exit()

# Computing

start_timer = time()

# Fancy way of parsing the one-line region into a pandas data-frame
if re.compile("chr").search(regions):

    genes = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

else:

    genes = pd.read_table(regions)

pathlib.Path(os.path.join(work_dir, dataset_name)).mkdir(parents=True, exist_ok=True)

bad_regions = {"region": [], "reason": []}

# Input file with all regions to KK
# regions = pd.read_csv(region_file, sep="\t")

for i, row in genes.iterrows():

    # chr1:36031800-40052000	POU3F1	ENSG00000185668.8

    region_chr, region_start, region_end, mlo.midpoint = re.split(":|-|_", row.coords)

    region_start = int(region_start)
    region_end = int(region_end)

    mlo.midpoint = int(mlo.midpoint)
    mlo.region = f"{region_chr}:{region_start}-{region_end}"
    mlo.resolution = resolution  # check if I can define this outside the loop

    filename = f"{mlo.region}_{mlo.midpoint}_{int(mlo.resolution/1000)}kb"

    pathlib.Path(os.path.join(work_dir, dataset_name, region_chr), "plots").mkdir(
        parents=True, exist_ok=True
    )

    coordsfile = f"{os.path.join(work_dir, dataset_name, region_chr, filename)}_KK.pkl"
    cooler_file = cooler_file + "::/resolutions/" + str(mlo.resolution)

    print(f"\n---> Working on region {mlo.region}.\n")

    # This part has been modified. In some datasets (NeuS) the matrix was completely empty,
    # so the pkl file was not created (gave an error in the MatTransform function:
    # pc = np.min(mat[mat>0]))
    # To bypass this part, the script checks the bad_regions.txt; if the regions
    # is also found in this
    # file, the script skips the calculations (as the script already has checked
    # the region).
    if os.path.isfile(coordsfile):

        print(f"\t{mlo.region} already done.")

        if force:

            print(
                "\tForce option (-f) selected, recalculating "
                "the Kamada-Kawai layout (files will be overwritten)\n"
            )

        else:

            continue

    elif (
        mlo.region in bad_regions["region"]
        and bad_regions["reason"][bad_regions["region"].index(mlo.region)] == "empty"
    ):

        print("\t<--- already done (no data)!")
        continue

    mlo.matrix = cooler.Cooler(cooler_file).matrix(sparse=True).fetch(mlo.region).toarray()
    mlo.matrix = misc.clean_matrix(mlo, bad_regions)

    # This if statement is for detecting empty arrays. If the array is too empty,
    # clean_matrix() returns mlo as None.
    if mlo.matrix is None:

        continue

    for j, cutoff in enumerate(cutoffs):

        time_per_kk = time()

        print(f"\tCutoff to be used is: {int(cutoff * 100)} %")

        # Get submatrix of restraints
        restraints_matrix, mlo.mixed_matrices, mixed_matrices_plot = kk.get_restraints_matrix(
            mlo,
            None,
            cutoff,
            plot_bool=True,
        )

        if save_plots:

            mixed_matrices_plot.savefig(
                os.path.join(
                    work_dir,
                    dataset_name,
                    region_chr,
                    f"plots/{filename}_" f"{cutoff}_mixed-matrices.pdf",
                ),
                dpi=300,
            )

            plt.close()

        print("\tLayouting Kamada-Kawai...")
        mlo.kk_graph = nx.from_scipy_sparse_array(csr_matrix(restraints_matrix))
        mlo.kk_nodes = nx.kamada_kawai_layout(mlo.kk_graph)

        # Get distance matrix  TO CALCULATE IN LMI, NOT HERE
        # coords = list(mlo.kk_nodes.values())
        # dists = distance.cdist(coords, coords, "euclidean")

        if len(cutoffs) == 1:

            # Save mlo.
            mlo_name = os.path.join(work_dir, dataset_name, region_chr, f"{filename}.mlo")
            with open(mlo_name, "wb") as mlo_file:

                pickle.dump(mlo_name, mlo_file)

            print(
                f"\tKamada-Kawai layout of region {mlo.region} saved at {int(cutoff * 100)} % cutoff to file: {mlo_name} "
            )

        if len(cutoffs) > 1 or save_plots:

            plt = plot.plot_kk(mlo)

            fig_name = os.path.join(
                work_dir,
                dataset_name,
                region_chr,
                f"plots/{filename}_" f"{cutoff}_KK.pdf",
            )

            plt.savefig(fig_name, dpi=300)
            plt.close()

        elapsed_time_secs = time() - time_per_kk

        print(f"\tdone in {timedelta(seconds=round(elapsed_time_secs))}.\n")

bad_regions = pd.DataFrame(bad_regions)

if bad_regions.shape[0] > 0:

    bad_regions.to_csv(
        f"{os.path.join(work_dir, dataset_name)}/bad_regions.txt",
        sep="\t",
        index=False,
        header=False,
        mode="a",
    )

print(f"Total time spent: {timedelta(seconds=round(time() - start_timer))}.")
print("\nall done.")
