import multiprocessing as mp
import os
import pathlib
import re
import sys
import warnings
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict
from datetime import timedelta
from time import time

import cooler
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from tqdm.contrib.concurrent import process_map

from metaloci import mlo
from metaloci.graph_layout import kk
from metaloci.misc import misc
from metaloci.plot import plot

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
    help="Resolution of the cooler files to be used (in bp).",
)

region_arg = parser.add_argument_group(title="region arguments", description="Choose one of the following options.")
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

optional_arg.add_argument("-f", "--force", dest="force", action="store_true", help="force rewriting existing data.")

optional_arg.add_argument(
    "-p",
    "--plot",
    dest="save_plots",
    action="store_true",
    help="plot the matrix, density and Kamada-Kawai plots, even when a " "single cutoff is selected.",
)

optional_arg.add_argument(
    "-m",
    "--mp",
    dest="multiprocess",
    action="store_true",
    help="Set use of multiprocessing.",
)

optional_arg.add_argument(
    "-t", "--threads", dest="threads", type=int, action="store", help="Number of cores to use in multiprocessing."
)

optional_arg.add_argument(
    "-l",
    "--pl",
    dest="persistence_length",
    action="store",
    help="set a persistence length for the Kamada-Kawai layout.",
)
# TODO add sort of silent argument?

optional_arg.add_argument("-u", "--debug", dest="debug", action="store_true", help=SUPPRESS)

args = parser.parse_args(None if sys.argv[1:] else ["-h"])

work_dir = args.work_dir
cooler_file = args.cooler_file
regions = args.regions
resolution = args.reso
cutoffs = args.cutoff
force = args.force
save_plots = args.save_plots
multiprocess = args.multiprocess
cores = args.threads
persistence_length = args.persistence_length
debug = args.debug

if not work_dir.endswith("/"):

    work_dir += "/"

if multiprocess is None:

    multiprocess = False

if cores is None:

    cores = mp.cpu_count() - 2

if cores > mp.cpu_count():

    cores = mp.cpu_count()

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
        ["force", force],
        ["debug", debug],
    ]

    print(table)
    sys.exit()


def get_region_layout(row, silent: bool = True):

    region_chrom, region_start, region_end, poi = re.split(":|-|_", row.coords)
    region_coords = f"{region_chrom}_{region_start}_{region_end}_{poi}"
    pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "KK").mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(work_dir, region_chrom, f"{region_coords}.mlo")

    if not os.path.isfile(save_path):

        mlobject = mlo.MetalociObject(
            f"{region_chrom}:{region_start}-{region_end}",
            str(region_chrom),
            int(region_start),
            int(region_end),
            resolution,
            int(poi),
            persistence_length,
            save_path,
        )

    elif (
        region_coords in bad_regions["region"]
        and bad_regions["reason"][bad_regions["region"].index(region_coords)] == "empty"
    ):

        if silent == False:

            print(f"\n------> Region {region_coords} already done (no data).")

        return

    else:

        if silent == False:

            print(f"\n------> Region {region_coords} already done.")

        if force:

            if silent == False:

                print(
                    "\tForce option (-f) selected, recalculating "
                    "the Kamada-Kawai layout (files will be overwritten)\n"
                )

            mlobject = mlo.MetalociObject(
                f"{region_chrom}:{region_start}-{region_end}",
                str(region_chrom),
                int(region_start),
                int(region_end),
                resolution,
                int(poi),
                persistence_length,
                save_path,
            )

        else:

            return

    if silent == False:
        print(f"\n------> Working on region: {mlobject.region}\n")

    # This part has been modified. In some datasets (NeuS) the matrix was completely empty,
    # so the pkl file was not created (gave an error in the MatTransform function:
    # pc = np.min(mat[mat>0]))
    # To bypass this part, the script checks the bad_regions.txt; if the regions
    # is also found in this
    # file, the script skips the calculations (as the script already has checked
    # the region).
    cooler_file_str = cooler_file + "::/resolutions/" + str(mlobject.resolution)
    mlobject.matrix = cooler.Cooler(cooler_file_str).matrix(sparse=True).fetch(mlobject.region).toarray()
    mlobject.matrix = misc.clean_matrix(mlobject, bad_regions)

    # if "chr" not in mlobject.matrix.chromnames[0]:

    #     cooler.rename_chroms(mlobject.matrix, {chrom: "chr" + str(chrom) for chrom in mlobject.matrix.chromnames})

    # This if statement is for detecting empty arrays. If the array is too empty,
    # clean_matrix() returns mlo as None.
    if mlobject.matrix is None:

        return

    for cutoff in cutoffs:

        time_per_kk = time()

        mlobject.kk_cutoff = cutoff

        if silent == False:
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
                    f"plots/mixed_matrices/{region_coords}_" f"{mlobject.kk_cutoff}_mixed-matrices.pdf",
                ),
                dpi=300,
            )

            plt.close()

        if silent == False:

            print("\tLayouting Kamada-Kawai...")

        mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(restraints_matrix))
        mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)
        mlobject.kk_coords = list(mlobject.kk_nodes.values())
        mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

        if len(cutoffs) > 1 or save_plots:

            kk_plt = plot.get_kk_plot(mlobject)

            fig_name = os.path.join(
                work_dir,
                region_chrom,
                f"plots/KK/{region_coords}_" f"{mlobject.kk_cutoff}_KK.pdf",
            )

            kk_plt.savefig(fig_name, dpi=300)
            plt.close()

        if silent == False:

            print(f"\tdone in {timedelta(seconds=round(time() - time_per_kk))}.")

        if len(cutoffs) == 1:

            if silent == False:
                print(
                    f"\tKamada-Kawai layout of region {mlobject.region} saved"
                    f" at {int(cutoff * 100)} % cutoff to file: {mlobject.save_path}"
                )

            # Save mlobject.
            with open(mlobject.save_path, "wb") as hamlo_namendle:

                mlobject.save(hamlo_namendle)

        # with open(f"{work_dir}bad_regions.txt", "a+") as handler:

        #     for region, reason in bad_regions.items():

        #         if not any(f"{region}\t{reason[0]}" in line for line in handler):

        #             handler.write(f"{region}\t{reason[0]}\n")
        #             handler.flush()

        # with open(f"{work_dir}bad_regions.txt", "a+") as handler:

        #     lines = [line.rstrip("\n") for line in handler]

        #     for region, reason in bad_regions.items():

        #         if f"{region}\t{reason[0]}" not in lines:
        #             # inserts on top, elsewise use lines.append(name) to append at the end of the file.
        #             lines.insert(0, f"{region}\t{reason[0]}\n")

        #     handler.seek(0)  # move to first position in the file, to overwrite !
        #     handler.write("\n".join(lines))
        #     handler.flush()

        # with open(f"{work_dir}bad_regions.txt", "a+") as handler:

        #     x = handler.readlines()  # reading all the lines in a list, no worry about '\n'

        #     for region, reason in bad_regions.items():

        #         if f"{region}\t{reason[0]}" not in x:
        #             # if word not in file then write the word to the file
        #             handler.write(f"{region}\t{reason[0]}\n")
        #             handler.flush()


# Computing

start_timer = time()

# Parsing the one-line region into a pandas data-frame. If the region contains '/', it is a path.
# There is no need to add Windows path as METALoci will not work on Windows either way.
if "/" in regions:

    df_regions = pd.read_table(regions)

else:

    df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

pathlib.Path(os.path.join(work_dir)).mkdir(parents=True, exist_ok=True)

bad_regions = defaultdict(list)

if multiprocess:

    if __name__ == "__main__":

        print(f"{len(df_regions)} regions will be computed.\n")

        try:

            pool = mp.Pool(processes=cores)
            process_map(get_region_layout, [row for _, row in df_regions.iterrows()], max_workers=cores, chunksize=1)
            pool.close()
            pool.join()

        except KeyboardInterrupt:

            pool.terminate()

else:

    for _, row in df_regions.iterrows():

        get_region_layout(row, silent=False)

# If there is bad regions, write to a file which is the bad region and why,
# but only if that region-reason pair does not lready exist in the file.
# bad_regions = pd.DataFrame(bad_regions)

# if bad_regions.shape[0] > 0:

#     with open(f"{work_dir}bad_regions.txt", "a+") as handler:

#         [
#             handler.write(f"{row.region}\t{row.reason}\n")
#             for _, row in bad_regions.iterrows()
#             if not any(f"{row.region}\t{row.reason}" in line for line in handler)
#         ]

print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
print("\nall done.")
