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
    dest="plot_matKK",
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
plot_matKK = args.plot_matKK
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

print(genes)

pathlib.Path(os.path.join(work_dir, dataset_name)).mkdir(parents=True, exist_ok=True)

bad_regions = {"region": [], "reason": []}

# Input file with all regions to KK
# regions = pd.read_csv(region_file, sep="\t")

for i, row in genes.iterrows():

    # chr1:36031800-40052000	POU3F1	ENSG00000185668.8

    region_chr, region_start, region_end, midpoint = re.split(":|-|_", row.coords)

    region_start = int(region_start)
    region_end = int(region_end)
    midpoint = int(midpoint)

    region_name = f"{region_chr}:{region_start}-{region_end}"
    filename = f"{region_name}_{midpoint}_{resolution}bp"

    pathlib.Path(os.path.join(work_dir, dataset_name, region_chr)).mkdir(
        parents=True, exist_ok=True
    )

    coordsfile = f"{os.path.join(work_dir, dataset_name, region_chr, filename)}_KK.pkl"
    cooler_file = cooler_file + "::/resolutions/" + str(resolution)

    print(f"---> Working on region {region_name}.")

    # This part is modified. In some datasets (NeuS) the matrix was completely empty,
    # so the pkl file was not created (gave an error in the MatTransform function:
    # pc = np.min(mat[mat>0]))
    # To bypass this part, the script checks the bad_regions.txt; if the regions
    # is also found in this
    # file, the script skips the calculations (as the script already has checked
    # the region).
    if os.path.isfile(coordsfile):

        print(f"\t{region_name} already done!")

        if force:

            print(
                "\tForce option (-f) selected, recalculating "
                "the Kamada-Kawai layout (files will be overwritten)"
            )

        else:

            continue

    elif (
        region_name in bad_regions["region"]
        and bad_regions["reason"][bad_regions["region"].index(region_name)] == "empty"
    ):

        print("\t<--- already done (no data)!")
        continue

    # Run the code...
    # See the matrix
    print("\tReading the cooler file and transforming into a matrix...")

    matrix = cooler.Cooler(cooler_file).matrix(sparse=True).fetch(region_name).toarray()
    diagonal = np.array(matrix.diagonal())
    total_zeroes, max_stretch, percentage_zeroes, zero_loc = misc.check_diagonal(diagonal)

    print(f"\tLength of the diagonal is: {len(diagonal)}")
    print(f"\tTotal of 0 in diagonal is: {total_zeroes}")
    print(f"\tMax strech of 0 in diagonal is: {max_stretch}")
    print(f"\t% of 0s in diagonal is: {percentage_zeroes}")

    if total_zeroes == len(diagonal):

        print("\t\tMatrix is empty; passing to the next region")
        bad_regions["region"].append(region_name)
        bad_regions["reason"].append("empty")

        continue

    if int(percentage_zeroes) >= 50:

        bad_regions["region"].append(region_name)
        bad_regions["reason"].append("perc")

    elif int(max_stretch) >= 50:

        bad_regions["region"].append(region_name)
        bad_regions["reason"].append("stretch")

    matrix[zero_loc] = 0
    matrix[:, zero_loc] = 0

    # Pseudocounts if min is zero
    print(np.nanmin(matrix))
    if np.nanmin(matrix) == 0:

        pc = np.min(matrix[matrix > 0])
        print(f"\t\tPseudocounts: {pc}")
        matrix = matrix + pc

    # Scale if all below 1
    print(np.nanmax(matrix))
    if np.nanmax(matrix) <= 1 or np.nanmin(matrix) <= 1:

        sf = 1 / np.nanmin(matrix)
        print(f"\t\tScaling factor: {sf}")
        matrix = matrix * sf

    if np.nanmin(matrix) < 1:

        matrix[matrix < 1] = 1

    matrix = np.log10(matrix)
    pickle_name = os.path.join(work_dir, dataset_name, region_chr, f"{filename}_mat.pkl")

    print(f"\tSaving full log10 matrix of {region_name} to file: ")
    print(f"\t\t{pickle_name}")

    with open(pickle_name, "wb") as output:

        pickle.dump(matrix, output)

    for j, cutoff in enumerate(cutoffs):

        mini_time = time()

        print(f"\tCutoff % to be used is: {int(cutoff * 100)}")

        # Get submatrix of restraints
        plot_save_file = os.path.join(work_dir, dataset_name, region_chr)
        restaints_matrix = kk.get_restraints_matrix(
            plot_save_file,
            matrix,
            None,
            cutoff,
            plot_bool=True,
        )

        # Kamada Kawai Layout
        # Create sparse matrix
        sparse_matrix = csr_matrix(restaints_matrix)
        print(
            f"\t\tSparse matrix contains {np.count_nonzero(sparse_matrix.toarray()):,} restraints"
        )

        # Get the KK layout
        print("\t\tLayouting KK...")
        G = nx.from_scipy_sparse_array(sparse_matrix)
        pos = nx.kamada_kawai_layout(G)

        # Get distance matrix
        coords = list(pos.values())
        dists = distance.cdist(coords, coords, "euclidean")

        if len(cutoffs) == 1:
            # Save KK data
            pickle_name = os.path.join(work_dir, dataset_name, region_chr, f"{filename}_KK.pkl")
            print(
                f"\tSaving KK layout of {region_name} to file with {int(cutoff * 100)}% cutoff: \n\t\t{pickle_name}"
            )

            with open(pickle_name, "wb") as pickle_file:

                pickle.dump(sparse_matrix, pickle_file)
                pickle.dump(pos, pickle_file)
                pickle.dump(dists, pickle_file)
                pickle.dump(coords, pickle_file)
                pickle.dump(G, pickle_file)

            print("\tSaved!")

        if len(cutoffs) > 1 or plot_matKK:

            print("\t\tPlotting KK...\n")

            plt.figure(figsize=(10, 10))
            options = {"node_size": 50, "edge_color": "black", "linewidths": 0.1, "width": 0.05}

            nx.draw(G, pos, node_color=range(len(pos)), cmap=plt.cm.coolwarm, **options)

            if midpoint:

                plt.scatter(
                    pos[midpoint][0], pos[midpoint][1], s=80, facecolors="none", edgecolors="r"
                )

            xs = []
            ys = []

            for _, val in pos.items():

                xs.append(val[0])
                ys.append(val[1])

            sns.lineplot(x=xs, y=ys, sort=False, lw=2, color="black", legend=False, zorder=1)

            fig_name = os.path.join(
                work_dir, dataset_name, region_chr, f"{filename}_" f"{cutoff}_KK.pdf"
            )

            plt.savefig(fig_name, bbox_incehs="tight", dpi=300)
            plt.close()

        elapsed_time_secs = time() - mini_time

        print(f"\tThis KK took {timedelta(seconds=round(elapsed_time_secs))}\n")

    print("done.")

bad_regions = pd.DataFrame(bad_regions)

if bad_regions.shape[0] > 0:

    bad_regions_file = f"{os.path.join(work_dir, dataset_name)}/bad_regions.txt"

    bad_regions.to_csv(bad_regions_file, sep="\t", index=False, header=False, mode="a")

elapsed_time_secs = time() - start_timer

print(f"Execution time: {timedelta(seconds=round(elapsed_time_secs))}")
