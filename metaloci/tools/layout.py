"""
Script to create the Kamada-Kaway layouts based on HiC data
"""
__author__ = "Iago Maceda Porto and Leo Zuber Ponce"

# pylint: disable=invalid-name, wrong-import-position, wrong-import-order

import os
import pathlib
import re
import sys
import warnings
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from datetime import timedelta
from time import time

import cooler
import hicstraw
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import distance

from metaloci import mlo
from metaloci.graph_layout import kk
from metaloci.misc import misc
from metaloci.plot import plot
import pathlib

warnings.filterwarnings("ignore", category=RuntimeWarning)

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

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=description, add_help=False)

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
    required=False,
    help="complete path to the cooler file.",
)

input_arg.add_argument(
    "-i",
    "--hic_file",
    dest="hic_file",
    metavar="PATH",
    type=str,
    required=True,
    help="complete path to the Hi-C file in either .cool/.mcool or .hic format.",
)

input_arg.add_argument(
    "-r",
    "--resolution",
    dest="reso",
    metavar="INT",
    type=int,
    required=True,
    help="Resolution of the Hi-C files to be used (in bp).",
)

region_arg = parser.add_argument_group(title="region arguments", description="Choose one of the following options.")
region_arg.add_argument(
    "-g",
    "--region",
    dest="single_region", # marcius
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

optional_arg.add_argument("-f", "--force", dest="force", action="store_true", help="force rewriting existing data.")

optional_arg.add_argument(
    "-p",
    "--plot",
    dest="save_plots",
    action="store_true",
    help="plot the matrix, density and Kamada-Kawai plots, even when a " "single cutoff is selected.",
)

optional_arg.add_argument(
    "-l",
    "--pl",
    dest="persistence_length",
    type=float, # marcius
    action="store",
    help="set a persistence length for the Kamada-Kawai layout.",
)
# TODO add sort of silent argument?

optional_arg.add_argument("-u", "--debug", dest="debug", action="store_true", help=SUPPRESS)

args = parser.parse_args(None if sys.argv[1:] else ["-h"])

work_dir = args.work_dir
hic_file = args.hic_file
cooler_file = args.cooler_file
sregion = args.single_region # marcius
regions = args.regions
resolution = args.reso
cutoffs = args.cutoff
force = args.force
save_plots = args.save_plots
persistence_length = args.persistence_length
debug = args.debug

if not work_dir.endswith("/"):

    work_dir += "/"


# if dataset_name == "":

#     if cooler_file.endswith(".cool") or cooler_file.endswith(".mcool"):

#         dataset_name = cooler_file.rsplit("/", 1)[1].rsplit(".")[0] + "_" + str(resolution)

#     else:

#         dataset_name = cooler_file.rsplit("/", 1)[1] + "_" + str(resolution)

if cutoffs is None:

    cutoffs = [0.2]

# To sort the floats from the cutoffs
cutoffs.sort(key=float, reverse=True)

if debug:

    table = [
        ["work_dir", work_dir],
        ["hic_fiile", hic_file],
        ["region", sregion], # marcius
        ["region_file", regions],
        ["reso", resolution],
        ["cutoffs", cutoffs],
        ["force", force],
        ["debug", debug],
    ]

    print(table)
    sys.exit()

# Computing

start_timer = time()

# Parsing the one-line region into a pandas data-frame. If the region contains '/', it is a path.
# There is no need to add Windows path as METALoci will not work on Windows either way.
if regions: # marcius

    df_regions = pd.read_table(regions)

if sregion:

    df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

pathlib.Path(os.path.join(work_dir)).mkdir(parents=True, exist_ok=True)

bad_regions = {"region": [], "reason": []}

for i, row in df_regions.iterrows():

    region_chrom, region_start, region_end, poi = re.split(":|-|_", row.coords)

    mlobject = mlo.MetalociObject(
        f"{region_chrom}:{region_start}-{region_end}",
        str(region_chrom),
        int(region_start),
        int(region_end),
        resolution,
        int(poi),
        persistence_length,
    )

    filename = f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}"

    pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "KK").mkdir(parents=True, exist_ok=True)

    coordsfile = f"{os.path.join(work_dir, region_chrom, filename)}_KK.pkl"
    
    if hic_file.endswith(".cool") or hic_file.endswith(".mcool"):
        
        hic_file_str = hic_file + "::/resolutions/" + str(mlobject.resolution)

    if hic_file.endswith(".hic"):
        
        hic_file_str = hic_file

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
                "\tForce option (-f) selected, recalculating " "the Kamada-Kawai layout (files will be overwritten)\n"
            )

        else:

            continue

    elif (
        mlobject.region in bad_regions["region"]
        and bad_regions["reason"][bad_regions["region"].index(mlobject.region)] == "empty"
    ):

        print("\t<--- already done (no data)!")

        continue

    if hic_file.endswith(".cool") or hic_file.endswith(".mcool"):
        mlobject.matrix = cooler.Cooler(hic_file_str).matrix(sparse=True).fetch(mlobject.region).toarray()
        mlobject.matrix = misc.clean_matrix(mlobject, bad_regions)

    if hic_file.endswith(".hic"):
        hic = hicstraw.HiCFile(hic_file_str)
        chrm,start,end = re.split(':|-', mlobject.region)
        mzd = hic.getMatrixZoomData(chrm, chrm, 'observed', 'VC_SQRT', 'BP', resolution)
        mlobject.matrix = mzd.getRecordsAsMatrix(int(start), int(end), int(start), int(end))        
        mlobject.matrix = misc.clean_matrix(mlobject, bad_regions)

    # if "chr" not in mlobject.matrix.chromnames[0]:

    #     cooler.rename_chroms(mlobject.matrix, {chrom: "chr" + str(chrom) for chrom in mlobject.matrix.chromnames})

    # This if statement is for detecting empty arrays. If the array is too empty,
    # clean_matrix() returns mlo as None.
    if mlobject.matrix is None:

        continue

    for j, cutoff in enumerate(cutoffs):

        time_per_kk = time()

        mlobject.kk_cutoff = cutoff

        print(f"\tCutoff to be used is: {int(mlobject.kk_cutoff * 100)} %")

        # Get submatrix of restraints
        restraints_matrix, mlobject = kk.get_restraints_matrix(mlobject)

        if save_plots:

            mixed_matrices_plot = plot.mixed_matrices_plot(mlobject)

            pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "mixed_matrices").mkdir(
                parents=True, exist_ok=True
            )

            mixed_matrices_plot.savefig(
                os.path.join(
                    work_dir,
                    region_chrom,
                    f"plots/mixed_matrices/{filename}_" f"{mlobject.kk_cutoff}_mixed-matrices.pdf",
                ),
                dpi=300,
            )

            plt.close()

        print("\tLayouting Kamada-Kawai...")
        mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(restraints_matrix))
        mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)
        mlobject.kk_coords = list(mlobject.kk_nodes.values())
        mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

        if len(cutoffs) == 1:

            mlobject.save_path = os.path.join(work_dir, region_chrom, f"{filename}.mlo")

            # Save mlobject.
            with open(mlobject.save_path, "wb") as hamlo_namendle:

                mlobject.save(hamlo_namendle)

            print(
                f"\tKamada-Kawai layout of region {mlobject.region} saved"
                f" at {int(cutoff * 100)} % cutoff to file: {mlobject.save_path}"
            )

        if len(cutoffs) > 1 or save_plots:

            kk_plt = plot.get_kk_plot(mlobject)

            fig_name = os.path.join(
                work_dir,
                region_chrom,
                f"plots/KK/{filename}_" f"{mlobject.kk_cutoff}_KK.pdf",
            )

            kk_plt.savefig(fig_name, dpi=300)
            plt.close()

        elapsed_time_secs = time() - time_per_kk

        print(f"\tdone in {timedelta(seconds=round(elapsed_time_secs))}.")

# If there is bad regions, write to a file which is the bad region and why,
# but only if that region-reason pair does not lready exist in the file.
bad_regions = pd.DataFrame(bad_regions)

if bad_regions.shape[0] > 0:

    with open(f"{work_dir}bad_regions.txt", "a+") as handler:

        [
            handler.write(f"{row.region}\t{row.reason}\n")
            for _, row in bad_regions.iterrows()
            if not any(f"{row.region}\t{row.reason}" in line for line in handler)
        ]

print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
print("\nall done.")
