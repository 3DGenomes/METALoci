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
from scipy.sparse import csr_matrix
import pyarrow.parquet as pq
from metaloci.misc import misc
from metaloci.graph_layout import kk
from metaloci import mlo
from metaloci.plot import plot
import numpy as np
import dill as pickle

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
    help="path to working directory.",
)
input_arg.add_argument(
    "-c",
    "--cooler",
    dest="cooler_file",
    metavar="PATH",
    type=str,
    required=True,
    help="complete path to the cooler file.",
)
input_arg.add_argument(
    "-r",
    "--resolution",
    dest="reso",
    metavar="INT",
    type=int,
    required=True,
    help="resolution of the cooler files.",
)
region_arg = parser.add_argument_group(
    title="region arguments", description="Choose one of the following options."
)
region_arg.add_argument(
    "-g",
    "--region",
    dest="regions",
    metavar="STR",
    type=str,
    help="region to be computed in chr:start-end format.",
)
region_arg.add_argument(
    "-G",
    "--region-file",
    dest="regions",
    metavar="PATH",
    type=str,
    help="path to the file with the regions of interest.",
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
    help="percentage of top interactions to keep, space separated (default: " "0.2)",
)
optional_arg.add_argument(
    "-n",
    "--name",
    dest="dataset_name",
    metavar="STR",
    default="",
    help="name of the dataset. By default corresponds to: " "COOLER.NAME_RESOLUTION",
)
optional_arg.add_argument(
    "-f", "--force", dest="force", action="store_true", help="force rewriting existing data."
)
optional_arg.add_argument(
    "-p",
    "--plot",
    dest="save_plots",
    action="store_true",
    help="plot the matrix, density and Kamada-Kawai plots, even when a "
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

# Parsing the one-line region into a pandas data-frame
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

    region_chrom, region_start, region_end = re.split(":|-", row.coords)

    mlobject = mlo.mlo(
        f"{region_chrom}:{region_start}-{region_end}",
        region_chrom,
        int(region_start),
        int(region_end),
        resolution,
    )

    filename = f"{mlobject.region}_{int(mlobject.resolution/1000)}kb"

    pathlib.Path(os.path.join(work_dir, dataset_name, region_chrom), "plots", "KK").mkdir(
        parents=True, exist_ok=True
    )

    coordsfile = f"{os.path.join(work_dir, dataset_name, region_chrom, filename)}_KK.pkl"
    cooler_file = cooler_file + "::/resolutions/" + str(mlobject.resolution)

    print(f"\n------> Working on region: {mlobject.region}\n")

    # This part has been modified. In some datasets (NeuS) the matrix was completely empty,
    # so the pkl file was not created (gave an error in the MatTransform function:
    # pc = np.min(mat[mat>0]))
    # To bypass this part, the script checks the bad_regions.txt; if the regions
    # is also found in this
    # file, the script skips the calculations (as the script already has checked
    # the region).
    if os.path.isfile(coordsfile):

        print(f"\t{mlobject.region} already done.")

        if force:

            print(
                "\tForce option (-f) selected, recalculating "
                "the Kamada-Kawai layout (files will be overwritten)\n"
            )

        else:

            continue

    elif (
        mlobject.region in bad_regions["region"]
        and bad_regions["reason"][bad_regions["region"].index(mlobject.region)] == "empty"
    ):

        print("\t<--- already done (no data)!")

        continue

    mlobject.matrix = (
        cooler.Cooler(cooler_file).matrix(sparse=True).fetch(mlobject.region).toarray()
    )
    mlobject.matrix = misc.clean_matrix(mlobject, bad_regions)

    # This if statement is for detecting empty arrays. If the array is too empty,
    # clean_matrix() returns mlo as None.
    if mlobject.matrix is None:

        continue

    for j, cutoff in enumerate(cutoffs):

        time_per_kk = time()

        mlobject.kk_cutoff = cutoff

        print(f"\tCutoff to be used is: {int(mlobject.kk_cutoff * 100)} %")

        # Get submatrix of restraints
        restraints_matrix, mlobject = kk.get_restraints_matrix(
            mlobject,
            None,
            plot_bool=True,
        )

        if save_plots:

            mlobject, mixed_matrices_plot = plot.mixed_matrices_plot(mlobject)

            pathlib.Path(
                os.path.join(work_dir, dataset_name, region_chrom), "plots", "mixed_matrices"
            ).mkdir(parents=True, exist_ok=True)

            mixed_matrices_plot.savefig(
                os.path.join(
                    work_dir,
                    dataset_name,
                    region_chrom,
                    f"plots/mixed_matrices/{filename}_" f"{mlobject.kk_cutoff}_mixed-matrices.pdf",
                ),
                dpi=300,
            )

            plt.close()

        print("\tLayouting Kamada-Kawai...")
        mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(restraints_matrix))
        mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)

        if len(cutoffs) == 1:

            mlobject.save_path = os.path.join(
                work_dir, dataset_name, region_chrom, f"{filename}.mlo"
            )

            # Save mlobject.
            with open(mlobject.save_path, "wb") as hamlo_namendle:

                mlobject.save(hamlo_namendle)

            print(
                f"\tKamada-Kawai layout of region {mlobject.region} saved"
                f" at {int(cutoff * 100)} % cutoff to file: {mlobject.save_path}"
            )

        if len(cutoffs) > 1 or save_plots:

            plt = plot.plot_kk(mlobject)

            fig_name = os.path.join(
                work_dir,
                dataset_name,
                region_chrom,
                f"plots/KK/{filename}_" f"{mlobject.kk_cutoff}_KK.pdf",
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
