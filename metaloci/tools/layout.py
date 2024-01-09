"""
Creates a Kamada-Kawai layout from a Hi-C for a given region.
"""
import os
import pathlib
import re
import sys
import warnings
import pickle
import glob
import subprocess as sp
import multiprocessing as mp
from argparse import HelpFormatter, SUPPRESS
from time import time
from datetime import timedelta

import h5py
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

DESCRIPTION = """
Creates a Kamada-Kawai layout from a Hi-C for a given region.
"""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller
    """

    ## TODO We do not have the silent argument in this parser, don't know if we have to add it...
    parser.formatter_class=lambda prog: HelpFormatter(prog, width=120,
                                                      max_help_position=60)

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
        "--hic",
        dest="hic_file",
        metavar="PATH",
        type=str,
        required=True,
        help="Complete path to the cool/mcool/hic file.",
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
        help="Region to apply LMI in format chrN:start-end_midpoint or file with the regions "
        "of interest. If a file is provided, it must contain as a header 'coords', 'symbol' and "
        "'id', and one region per line, tab separated. ",
    )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.")

    optional_arg.add_argument(
        "-o",
        "--cutoff",
        dest="cutoff",
        nargs="*",
        type=float,
        action="extend",
        help="Fraction of top interactions to keep, space separated (default: 0.2)",
    )

    optional_arg.add_argument(
        "-a",
        "--absolute",
        dest="absolute",
        action="store_true",
        help="Treat the cutoff as an absolute value instead of a fraction of top interactions "
         "to keep.",
    )

    optional_arg.add_argument(
        "-l",
        "--pl",
        dest="persistence_length",
        metavar="INT",
        action="store",
        help="Set a persistence length for the Kamada-Kawai layout.",
    )

    optional_arg.add_argument(
        "-p",
        "--plot",
        dest="save_plots",
        action="store_true",
        help="Plot the matrix, density and Kamada-Kawai plots, even when a single cutoff "
        "is selected.",
    )

    optional_arg.add_argument(
        "-m",
        "--mp",
        dest="multiprocess",
        action="store_true",
        help="Flag to set use of multiprocessing.",
    )

    ## TODO Should we change this to be a third of the cores, the same way as the ml?
    optional_arg.add_argument(
        "-t",
        "--threads",
        dest="threads",
        default=int(mp.cpu_count() - 2),
        metavar="INT",
        type=int,
        action="store",
        help="Number of threads to use in multiprocessing. (default: %(default)s)"
    )

    optional_arg.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        help="Force METALoci to rewrite existing data.")

    optional_arg.add_argument(
        "-u",
        "--debug",
        dest="debug",
        action="store_true",
        help=SUPPRESS)

def get_region_layout(row : pd.Series, args_2_use : pd.Series,
                      progress = None, counter : int = None, silent: bool = True):
    """
    Function to get the Kamada-Kawai layout for a given region

    Parameters
    ----------
    row : pd.Series
        Data of the region
    args_2_use : pd.Series
        pd.Series with all the needed argument info to run the function
    progress : optional
        Information about if this region has been completed, by default None
    counter : int, optional
        Counter used for when multiprocessing is not active, by default None
    silent : bool, optional
        Verbosity of the function, by default True
    """

    work_dir = args_2_use.work_dir
    hic_path = args_2_use.hic_path
    resolution = args_2_use.resolution
    cutoffs = args_2_use.cutoffs
    persistence_length = args_2_use.persistence_length
    force = args_2_use.force
    save_plots = args_2_use.save_plots
    total_num = args_2_use.total_num

    region_chrom, _, _, _ = re.split(r":|-|_", row.coords)

    pathlib.Path(os.path.join(work_dir, region_chrom)).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(work_dir, region_chrom, f"{re.sub(':|-', '_', row.coords)}.mlo")

    if os.path.isfile(save_path):

        with open(save_path, "rb") as mlobject_handler:

            mlobject = pickle.load(mlobject_handler)

        ## TODO Redo this if chain so we mention the force once (in the run function maybe?) and if force is active
        ## to NOT say Region bla is already done.
        if not silent:

            print(f"\n------> Region {row.coords} already done.")

        if force:

            if not silent:

                print(
                    "\tForce option (-f) selected, recalculating "
                    "the Kamada-Kawai layout (files will be overwritten)."
                )

            os.remove(f"{save_path}")

        else:

            if save_plots:

                if not silent:
                    print("\tPlotting Kamada-Kawai...")

                ## Should the paths be created with the os.path.join?
                ## For example: os.path.join(work_dir, region_chrom, "plots", "KK")
                pathlib.Path(os.path.join(work_dir, region_chrom, "plots", "KK")).mkdir(parents=True, exist_ok=True)

                pathlib.Path(os.path.join(work_dir, region_chrom, "plots", "mixed_matrices")).mkdir(
                    parents=True, exist_ok=True
                )

                plot_name = f"{re.sub(':|-', '_', row.coords)}_" + \
                    f"{mlobject.kk_cutoff['cutoff_type']}_" + \
                    f"{mlobject.kk_cutoff['values']:.4f}_" + \
                    "{}.pdf"

                plot.get_kk_plot(mlobject).savefig(
                    os.path.join(
                        work_dir,
                        region_chrom,
                        "plots",
                        "KK",
                        plot_name.format("KK")), dpi=300)

                plt.close()

                plot.mixed_matrices_plot(mlobject).savefig(
                    os.path.join(
                        work_dir,
                        region_chrom,
                        "plots",
                        "mixed_matrices",
                        plot_name.format("mixed_matrices")), dpi=300,
                )

                plt.close()

                if progress is not None:
                    progress["plots"] = True

            if progress is not None:
                progress["done"] = True

        if mlobject.bad_region == "empty":

            if not silent:

                print(f"\n------> Region {row.coords} already done (Hi-C is empty in that region).")

            if progress is not None:
                progress["done"] = True

    if not os.path.isfile(save_path):

        if not silent:

            print(f"\n------> Working on region: {row.coords} [{counter+1}/{total_num}]\n")

        mlobject = mlo.MetalociObject(
                region = row.coords,
                resolution = resolution,
                persistence_length = persistence_length,
                save_path = save_path,
            )

        mlobject.kk_cutoff["cutoff_type"] = cutoffs["cutoff_type"]

        if hic_path.endswith(".cool"):

            mlobject.matrix = cooler.Cooler(hic_path).matrix(sparse=True).fetch(mlobject.region).toarray()

        if hic_path.endswith(".mcool"):

            mlobject.matrix = cooler.Cooler(f"{hic_path}::/resolutions/{mlobject.resolution}").matrix(sparse=True).fetch(mlobject.region).toarray()

        elif hic_path.endswith(".hic"):

            mlobject.matrix = hicstraw.HiCFile(hic_path).getMatrixZoomData(mlobject.chrom, mlobject.chrom, 'observed', 'VC_SQRT', 'BP', mlobject.resolution).getRecordsAsMatrix(mlobject.start, mlobject.end, mlobject.start, mlobject.end)

        mlobject = misc.clean_matrix(mlobject)

        # This if statement is for detecting empty arrays. If the array is too empty,
        # clean_matrix() will return mlobject.matrix as None.
        if mlobject.matrix is None:

            return

        time_per_region = time()

        for i, cutoff in enumerate(cutoffs["values"]):

            mlobject.kk_cutoff["values"] = cutoffs["values"][i]

            # Get submatrix of restraints
            mlobject = kk.get_restraints_matrix(mlobject, silent)

            if mlobject.kk_restraints_matrix is None:

                return

            if not silent:

                print("\tLayouting Kamada-Kawai...")

            mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject.kk_restraints_matrix))
            mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)
            mlobject.kk_coords = list(mlobject.kk_nodes.values())
            mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

            if len(cutoffs) > 1 or save_plots:

                if not silent:
                    print("\tPlotting Kamada-Kawai...")

                pathlib.Path(os.path.join(work_dir, region_chrom, "plots", "KK")).mkdir(parents=True, exist_ok=True)
                pathlib.Path(os.path.join(work_dir, region_chrom, "plots", "mixed_matrices")).mkdir(
                    parents=True, exist_ok=True
                )

                plot_name = f"{re.sub(':|-', '_', row.coords)}_" + \
                    f"{mlobject.kk_cutoff['cutoff_type']}_" + \
                    f"{mlobject.kk_cutoff['values']:.4f}_" + \
                    "{}.pdf"

                plot.get_kk_plot(mlobject).savefig(
                    os.path.join(
                        work_dir,
                        region_chrom,
                        "plots",
                        "KK",
                        plot_name.format("KK")), dpi=300)

                plt.close()

                plot.mixed_matrices_plot(mlobject).savefig(
                    os.path.join(
                        work_dir,
                        region_chrom,
                        "plots",
                        "mixed_matrices",
                        plot_name.format("mixed_matrices")), dpi=300,
                )

                plt.close()

                if progress is not None:

                    progress["plots"] = True

            if not silent:

                print(f"\tdone in {timedelta(seconds=round(time() - time_per_region))}.\n")

            elif len(cutoffs["values"]) == 1:

                if not silent:
                    print(
                        f"\tKamada-Kawai layout of region '{mlobject.region}' "
                        f"at {int(cutoff * 100)} % cutoff saved to file: '{mlobject.save_path}'"
                    )

                # Write to file a list of bad regions, according to the filters defined
                # in clean_matrix().
                with open(f"{work_dir}bad_regions.txt", "a+", encoding="utf-8") as handler:

                    log = f"{mlobject.region}\t{mlobject.bad_region}\n"

                    handler.seek(0)

                    if not any(log in line for line in handler) and mlobject.bad_region is not None:

                        handler.write(log)

                # Save mlobject.
                with open(mlobject.save_path, "wb") as hamlo_namendle: # :D

                    mlobject.save(hamlo_namendle)

    if progress is not None:

        progress['value'] += 1

        time_spent = time() - progress['timer']
        time_remaining = int(time_spent / progress['value'] * (total_num - progress['value']))

        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t[{progress['value']}/{total_num}] | Time spent: {timedelta(seconds=round(time_spent))} | "
                f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')


def run(opts : list):
    """
    Funtion to run this section of METALoci with the needed arguments

    Parameters
    ----------
    opts : list
        List of arguments
    """

    work_dir = opts.work_dir
    hic_path = opts.hic_file
    resolution = opts.reso
    regions = opts.regions
    cutoffs_opt = opts.cutoff
    absolute = opts.absolute
    persistence_length = opts.persistence_length
    force = opts.force
    save_plots = opts.save_plots
    multiprocess = opts.multiprocess
    threads = opts.threads

    ## Parsing of some of the arguments
    if not work_dir.endswith("/"):

        work_dir += "/"

    if threads > mp.cpu_count():

        threads = mp.cpu_count()

    cutoffs = {}

    if absolute and cutoffs_opt is None or absolute and len(cutoffs_opt) == 0:

        sys.exit("Please provide a cutoff (-o) value when using the absolute flag (-a).")

    elif cutoffs_opt is None or len(cutoffs_opt) == 0:

        cutoffs["cutoff_type"] = "percentage"
        cutoffs["values"] = [0.2]

    elif absolute:

        cutoffs["cutoff_type"] = "absolute"
        cutoffs["values"] = cutoffs_opt

    else:

        cutoffs["cutoff_type"] = "percentage"
        cutoffs["values"] = cutoffs_opt

    if cutoffs["cutoff_type"] == "percentage" and not 0 < cutoffs["values"][0] <= 1:

        sys.exit("Select a cut-off between 0 and 1.")

    cutoffs["values"].sort(key=float, reverse=True)

    if regions is None:

        if len(glob.glob(f"{work_dir}*coords.txt")) > 1:

            sys.exit("More than one region file found. Please provide a region or only one file with regions "
                     "of interest'.")

        try:

            df_regions = pd.read_table(glob.glob(f"{work_dir}*coords.txt")[0])

        except IndexError:

            sys.exit("No regions provided. Please provide a region or a file with regions of interest or run "
                     "'metaloci sniffer'.")

    elif os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    ## Debug 'menu' for testing purposes
    if opts.debug:

        print(f"work_dir ->\n\t{work_dir}")
        print(f"hic_path ->\n\t{hic_path}")
        print(f"resolution ->\n\t{resolution}")
        print(f"regions ->\n\t{regions}")
        print(f"cutoffs ->\n\t{cutoffs}")
        print(f"absolute ->\n\t{absolute}")
        print(f"persistence_length ->\n\t{persistence_length}")
        print(f"force ->\n\t{force}")
        print(f"save_plots ->\n\t{save_plots}")
        print(f"multiprocess ->\n\t{multiprocess}")
        print(f"cores ->\n\t{threads}")

        sys.exit()

    ## Checking if the resolution supplied by the user is within the resolutions of the cool/mcool/hic files
    if hic_path.endswith(".cool"):

        aval_resolutions = cooler.Cooler(hic_path).binsize

        if resolution != aval_resolutions:
            sys.exit("The given resolution is not the same as the provided cooler file. Exiting...")

    elif hic_path.endswith(".mcool"):

        aval_resolutions = [int(x) for x in list(h5py.File(hic_path)["resolutions"].keys())]

        if resolution not in aval_resolutions:

            print("The given resolution is not in the provided mcooler file. Exiting...")
            print("The available resolutions are: " +
                  ", ".join(misc.natural_sort([str(x) for x in aval_resolutions])))
            sys.exit("Exiting...")

    elif hic_path.endswith(".hic"):

        aval_resolutions = hicstraw.HiCFile(hic_path).getResolutions()

        if resolution not in aval_resolutions:

            print("The given resolution is not in the provided Hi-C file.")
            print("The available resolutions are: " +
                  ", ".join(misc.natural_sort([str(x) for x in aval_resolutions])))
            sys.exit("Exiting...")

    ## Added an else statement just in case the user is stupid enough to not use a cool/mcool/hic file.
    else:
        print("HiC file format not supported. Supported formats are: cool, mcool, hic.")
        sys.exit("Exiting...")

    args_2_pass = pd.Series({"work_dir" : work_dir,
                             "hic_path" : hic_path,
                             "resolution" : resolution,
                             "cutoffs" : cutoffs,
                             "persistence_length" : persistence_length,
                             "force" : force,
                             "save_plots" : save_plots,
                             "total_num" : len(df_regions)})

    start_timer = time()

    pathlib.Path(os.path.join(work_dir)).mkdir(parents=True, exist_ok=True)

    if multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:

            progress = mp.Manager().dict(value=0, timer=start_timer, done=False, plots=False)

            with mp.Pool(processes=threads) as pool:

                pool.starmap(get_region_layout, [(row, args_2_pass, progress) for _, row in df_regions.iterrows()])

                if progress["done"]:

                    print("\tSome regions had already been computed and have been skipped.",
                          end="")

                if progress["plots"]:

                    print(f"\n\tKamada-Kawai layout plots saved to '{work_dir}chr/plots/KK'",
                          end="")

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for counter, row in df_regions.iterrows():

            get_region_layout(row, args_2_pass, counter=counter, silent=False)

    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nAll done.")
