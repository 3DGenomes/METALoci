import multiprocessing as mp
import os
import pathlib
import re
import warnings
from argparse import SUPPRESS, HelpFormatter
from collections import defaultdict
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

DESCRIPTION = """ Creates a Kamada-Kawai layout from a Hi-C for a given region.
"""


def populate_args(parser):

    parser.formatter_class=lambda prog: HelpFormatter(prog, width=120,
                                                      max_help_position=60)
    
    input_arg = parser.add_argument_group(title="Input arguments")
    optional_arg = parser.add_argument_group(title="Optional arguments")

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
        "--hic",
        dest="hic_file",
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
        help="Resolution of the HI-C files to be used (in bp).",
    )

    input_arg.add_argument(
        "-g",
        "--region",
        dest="regions",
        metavar="PATH",
        type=str,
        required=True,
        help="Region to apply LMI in format chrN:start-end_midpoint or file with the regions of interest. If a file is provided, "
        "it must contain as a header 'coords', 'symbol' and 'id', and one region per line, tab separated.",
    )

    optional_arg.add_argument(
        "-o",
        "--cutoff",
        dest="cutoff",
        nargs="*",
        type=float,
        action="extend",
        help="Percentage of top interactions to keep, space separated (default: 0.2)",
    )

    optional_arg.add_argument("-f", "--force", dest="force", action="store_true", help="force rewriting existing data.")

    optional_arg.add_argument(
        "-p",
        "--plot",
        dest="save_plots",
        action="store_true",
        help="Plot the matrix, density and Kamada-Kawai plots, even when a single cutoff is selected.",
    )

    optional_arg.add_argument(
        "-m",
        "--mp",
        dest="multiprocess",
        action="store_true",
        help="Flag to set use of multiprocessing.",
    )

    optional_arg.add_argument(
        "-t", "--threads", dest="threads", type=int, action="store", help="Number of threads to use in multiprocessing."
    )

    optional_arg.add_argument(
        "-l",
        "--pl",
        dest="persistence_length",
        action="store",
        help="Set a persistence length for the Kamada-Kawai layout.",
    )

def get_region_layout(row, opts, silent: bool = True):

    work_dir = opts.work_dir
    hic_path = opts.hic_file
    resolution = opts.reso
    cutoffs = opts.cutoff
    force = opts.force
    save_plots = opts.save_plots
    persistence_length = opts.persistence_length

    if cutoffs is None:

        cutoffs = [0.2] 

    cutoffs.sort(key=float, reverse=True)

    bad_regions = defaultdict(list) # does not belong here, this is to prevent an error TODO

    region_chrom, region_start, region_end, poi = re.split(":|-|_", row.coords)
    region_coords = f"{region_chrom}_{region_start}_{region_end}_{poi}"
    pathlib.Path(os.path.join(work_dir, region_chrom)).mkdir(parents=True, exist_ok=True)
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

    if hic_path.endswith(".cool") or hic_path.endswith(".mcool"):

        mlobject.matrix = cooler.Cooler(hic_path + "::/resolutions/" + str(mlobject.resolution)).matrix(sparse=True).fetch(mlobject.region).toarray()

    elif hic_path.endswith(".hic"):

        chrm, start, end = re.split(':|-', mlobject.region)
        mlobject.matrix = hicstraw.HiCFile(hic_path).getMatrixZoomData(chrm, chrm, 'observed', 'VC_SQRT', 'BP', mlobject.resolution).getRecordsAsMatrix(int(start), int(end), int(start), int(end))      
    
    mlobject.matrix = misc.clean_matrix(mlobject, bad_regions)

    # This if statement is for detecting empty arrays. If the array is too empty,
    # clean_matrix() would return mlobject.matrix as None.
    if mlobject.matrix is None:

        return

    for cutoff in cutoffs:

        time_per_kk = time()

        mlobject.kk_cutoff = cutoff

        if silent == False:
            print(f"\tCutoff: {int(mlobject.kk_cutoff * 100)} %")

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

            pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "KK").mkdir(parents=True, exist_ok=True)

            kk_plt = plot.get_kk_plot(mlobject)

            fig_name = os.path.join(
                work_dir,
                region_chrom,
                f"plots/KK/{region_coords}_" f"{mlobject.kk_cutoff}_KK.pdf",
            )

            kk_plt.savefig(fig_name, dpi=300)
            plt.close()

        if silent == False:

            print(f"\tdone in {timedelta(seconds=round(time() - time_per_kk))}.\n")

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


def run(opts):

    work_dir = opts.work_dir
    regions = opts.regions
    multiprocess = opts.multiprocess
    cores = opts.threads

    if not work_dir.endswith("/"):

        work_dir += "/"

    if multiprocess is None:

        multiprocess = False

    if cores is None:

        cores = mp.cpu_count() - 2

    elif cores > mp.cpu_count():

        cores = mp.cpu_count()

    start_timer = time()

    if os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    pathlib.Path(os.path.join(work_dir)).mkdir(parents=True, exist_ok=True)

    bad_regions = defaultdict(list)

    if multiprocess:

        print(f"{len(df_regions)} regions will be computed.\n")

        try:

            with mp.Pool(processes=cores) as pool:
            
                pool.starmap(get_region_layout, [(row, opts) for _, row in df_regions.iterrows()])
                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for _, row in df_regions.iterrows():

            get_region_layout(row, opts, silent=False)

    # If there is bad regions, write to a file which is the bad region and why,
    # but only if that region-reason pair does not already exist in the file.
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
