import multiprocessing as mp
import os
import pathlib
import re
import warnings
from argparse import SUPPRESS, HelpFormatter
from collections import defaultdict
from datetime import timedelta
from time import time
import subprocess as sp

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
import pickle
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
        help="Fraction of top interactions to keep, space separated (default: 0.2)",
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

    optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")


def get_region_layout(row, opts, progress=None, silent: bool = True):

    work_dir = opts.work_dir
    regions = opts.regions
    hic_path = opts.hic_file
    resolution = opts.reso
    cutoffs = opts.cutoff
    force = opts.force
    save_plots = opts.save_plots
    persistence_length = opts.persistence_length

    if not work_dir.endswith("/"):

        work_dir += "/"

    if cutoffs is None:

        cutoffs = [0.2] 

    if os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    cutoffs.sort(key=float, reverse=True)

    region_chrom, region_start, region_end, poi = re.split(":|-|_", row.coords)
    region_coords = f"{region_chrom}_{region_start}_{region_end}_{poi}"
    pathlib.Path(os.path.join(work_dir, region_chrom)).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(work_dir, region_chrom, f"{region_coords}.mlo")

    # if not os.path.isfile(save_path):

    #     mlobject = mlo.MetalociObject(
    #         f"{region_chrom}:{region_start}-{region_end}",
    #         str(region_chrom),
    #         int(region_start),
    #         int(region_end),
    #         resolution,
    #         int(poi),
    #         persistence_length,
    #         save_path,
    #     )
        
    # elif mlobject.bad_region == "empty":

    #     if silent == False:

    #         print(f"\n------> Region {region_coords} already done (Hi-C empty in that region).")

    #     if progress is not None: progress["done"] = True

    #     return

    # else:

    #     if silent == False:

    #         print(f"\n------> Region {region_coords} already done.")

    #     if progress is not None: progress["done"] = True

    #     if force:

    #         if silent == False:

    #             print(
    #                 "\tForce option (-f) selected, recalculating "
    #                 "the Kamada-Kawai layout (files will be overwritten)"
    #             )

    #         mlobject = mlo.MetalociObject(
    #             f"{region_chrom}:{region_start}-{region_end}",
    #             str(region_chrom),
    #             int(region_start),
    #             int(region_end),
    #             resolution,
    #             int(poi),
    #             persistence_length,
    #             save_path,
    #         )

    #     else:

    #         return
        
    if os.path.isfile(save_path):

        with open(save_path, "rb") as mlobject_handler:

            mlobject = pickle.load(mlobject_handler)

        if silent == False:

            print(f"\n------> Region {region_coords} already done.")

        if force:

            if silent == False:

                print(
                    "\tForce option (-f) selected, recalculating "
                    "the Kamada-Kawai layout (files will be overwritten)"
                )
            
            os.remove(f"{save_path}")

        else:

            if save_plots:
                
                if silent == False: print(f"\tPlotting Kamada-Kawai...")

                pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "KK").mkdir(parents=True, exist_ok=True)
                pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "mixed_matrices").mkdir(
                    parents=True, exist_ok=True
                )

                plot.get_kk_plot(mlobject).savefig(os.path.join(
                    work_dir,
                    region_chrom,
                    f"plots/KK/{region_coords}_" f"{mlobject.kk_cutoff}_KK.pdf",
                ),
                dpi=300)

                plt.close()

                plot.mixed_matrices_plot(mlobject).savefig(
                    os.path.join(
                        work_dir,
                        region_chrom,
                        f"plots/mixed_matrices/{region_coords}_" f"{mlobject.kk_cutoff}_mixed-matrices.pdf",
                    ),
                    dpi=300,
                )

                plt.close()

                if progress is not None: progress["plots"] = True

            if progress is not None: progress["done"] = True
                        
        if mlobject.bad_region == "empty":

            if silent == False:

                print(f"\n------> Region {region_coords} already done (Hi-C empty in that region).")

            if progress is not None: progress["done"] = True
        
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

        if silent == False:
            print(f"\n------> Working on region: {mlobject.region}\n")

        if hic_path.endswith(".cool") or hic_path.endswith(".mcool"):

            mlobject.matrix = cooler.Cooler(hic_path + "::/resolutions/" + str(mlobject.resolution)).matrix(sparse=True).fetch(mlobject.region).toarray()

        elif hic_path.endswith(".hic"):

            chrm, start, end = re.split(':|-', mlobject.region)
            mlobject.matrix = hicstraw.HiCFile(hic_path).getMatrixZoomData(chrm, chrm, 'observed', 'VC_SQRT', 'BP', mlobject.resolution).getRecordsAsMatrix(int(start), int(end), int(start), int(end))      
        
        mlobject = misc.clean_matrix(mlobject)

        # This if statement is for detecting empty arrays. If the array is too empty,
        # clean_matrix() will return mlobject.matrix as None.
        if mlobject.matrix is None:

            return

        time_per_region = time()

        for cutoff in cutoffs:

            mlobject.kk_cutoff = cutoff

            if silent == False:
                print(f"\tCutoff: {int(mlobject.kk_cutoff * 100)} %")

            # Get submatrix of restraints
            restraints_matrix, mlobject = kk.get_restraints_matrix(mlobject)


            if silent == False:

                print("\tLayouting Kamada-Kawai...")

            mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(restraints_matrix))
            mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)
            mlobject.kk_coords = list(mlobject.kk_nodes.values())
            mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

            if len(cutoffs) > 1 or save_plots:

                if silent == False: print(f"\tPlotting Kamada-Kawai...")

                pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "KK").mkdir(parents=True, exist_ok=True)
                pathlib.Path(os.path.join(work_dir, region_chrom), "plots", "mixed_matrices").mkdir(
                    parents=True, exist_ok=True
                )

                plot.get_kk_plot(mlobject).savefig(os.path.join(
                    work_dir,
                    region_chrom,
                    f"plots/KK/{region_coords}_" f"{mlobject.kk_cutoff}_KK.pdf",
                ),
                dpi=300)

                plt.close()

                plot.mixed_matrices_plot(mlobject).savefig(
                    os.path.join(
                        work_dir,
                        region_chrom,
                        f"plots/mixed_matrices/{region_coords}_" f"{mlobject.kk_cutoff}_mixed-matrices.pdf",
                    ),
                    dpi=300,
                )

                plt.close()

                if progress is not None: progress["plots"] = True

            if silent == False:

                print(f"\tdone in {timedelta(seconds=round(time() - time_per_region))}.\n")

            if len(cutoffs) == 1:

                if silent == False:
                    print(
                        f"\tKamada-Kawai layout of region '{mlobject.region}' "
                        f"at {int(cutoff * 100)} % cutoff saved to file: '{mlobject.save_path}'"
                    )

                # Write to file a list of bad regions, according to the filters defined in clean_matrix().
                with open(f"{work_dir}bad_regions.txt", "a+") as handler:

                    log = f"{mlobject.region}\t{mlobject.bad_region}\n"

                    handler.seek(0)

                    if not any(log in line for line in handler) and mlobject.bad_region != None:

                        handler.write(log)

                # Save mlobject.
                with open(mlobject.save_path, "wb") as hamlo_namendle:

                    mlobject.save(hamlo_namendle)

    if progress is not None:

        progress['value'] += 1

        time_spent = time() - progress['timer']
        time_remaining = int(time_spent / progress['value'] * (len(df_regions) - progress['value']))

        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t[{progress['value']}/{len(df_regions)}] | Time spent: {timedelta(seconds=round(time_spent))} | "
                f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')


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

    if multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:

            manager = mp.Manager()
            progress = manager.dict(value=0, timer=start_timer, done=False, plots=False)
  
            with mp.Pool(processes=cores) as pool:
            
                pool.starmap(get_region_layout, [(row, opts, progress) for _, row in df_regions.iterrows()])

                if progress["done"] == True:

                    print("\tSome regions had already been computed and have been skipped.", end="")

                if progress["plots"] == True:

                    print(f"\n\tKamada-Kawai layout plots saved to {work_dir}chr/plots/KK.", end="")

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for _, row in df_regions.iterrows():

            get_region_layout(row, opts, silent=False)
    
    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nall done.")
