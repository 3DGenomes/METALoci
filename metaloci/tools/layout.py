"""
Given a Hi-C and a region, this script creates a Kamada-Kawai layout for said region.
"""
import glob
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import subprocess as sp
import sys
import warnings
from argparse import SUPPRESS, HelpFormatter
from datetime import timedelta
from time import time

import cooler
import h5py
import hicstraw
import networkx as nx
import pandas as pd
from metaloci import mlo
from metaloci.graph_layout import kk
from metaloci.misc import misc
from metaloci.plot import plot
from scipy.sparse import csr_matrix
from scipy.spatial import distance

warnings.filterwarnings("ignore", category=RuntimeWarning)

HELP = "Creates a Kamada-Kawai layout from a Hi-C for a given region."

DESCRIPTION = """
Creates a Kamada-Kawai layout from a Hi-C for a given region.
"""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step.

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller.
    """

    # TODO We do not have the silent argument in this parser, don't know if we have to add it...
    parser.formatter_class = lambda prog: HelpFormatter(prog, width=120, max_help_position=60)

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
        dest="resolution",
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
        help="Region to apply LMI in format chrN:start-end_midpoint or file with the regions of interest. If a file \
        is provided, it must contain as a header 'coords', 'symbol' and 'id', and one region per line, tab separated.",
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
        help="Treat the cutoff as an absolute value instead of a fraction of top interactions to keep.",
    )

    optional_arg.add_argument(
        "-l",
        "--pl",
        dest="persistence_length",
        metavar="FLOAT",
        action="store",
        type=float,
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


def get_region_layout(row: pd.Series, args: pd.Series, progress=None, counter: int = None, silent: bool = True):
    """
    Function to get the Kamada-Kawai layout for a given region

    Parameters
    ----------
    row : pd.Series
        Data of the region
    args : pd.Series
        pd.Series with all the needed argument info to run the function
    progress : optional
        Information about if this region has been completed, by default None
    counter : int, optional
        Counter used for when multiprocessing is not active, by default None
    silent : bool, optional
        Flag to enable or disable the print statements. Useful for multiprocessing.
    """

    region_chrom, _, _, _ = re.split(r":|-|_", row.coords)
    save_path = os.path.join(args.work_dir, region_chrom, 'objects', f"{re.sub(':|-', '_', row.coords)}.mlo")

    pathlib.Path(os.path.join(args.work_dir, region_chrom, 'objects',)).mkdir(parents=True, exist_ok=True)

    if os.path.isfile(save_path):

        # TODO Redo this if chain so we mention the force once (in the run function maybe?) and if force is active
        # to NOT say Region bla is already done.
        if not silent:

            print(f"\n------> Region {row.coords} already done.")

        if args.force:

            if not silent:

                print(
                    "\tForce option (-f) selected, recalculating the Kamada-Kawai layout (files will be overwritten)."
                )

            os.remove(f"{save_path}")

        else:

            with open(save_path, "rb") as mlobject_handler:

                mlobject = pickle.load(mlobject_handler)

            if args.save_plots:

                if not silent:

                    print("\tPlotting Kamada-Kawai...")

                plot.save_mm_kk(mlobject, args.work_dir)

                if progress is not None:

                    progress["plots"] = True

            if progress is not None:

                progress["done"] = True

            if progress is not None:

                progress["done"] = True

    if not os.path.isfile(save_path):

        if not silent:

            print(f"\n------> Working on region: {row.coords} [{counter+1}/{args.total_num}]\n")

        mlobject = mlo.MetalociObject(
            region=row.coords,
            resolution=args.resolution,
            persistence_length=args.persistence_length,
            save_path=save_path,
        )

        mlobject.kk_cutoff["cutoff_type"] = args.cutoffs["cutoff_type"]

        if args.hic_path.endswith(".cool"):

            mlobject.matrix = cooler.Cooler(args.hic_path).matrix(sparse=True).fetch(mlobject.region).toarray()

        if args.hic_path.endswith(".mcool"):

            mlobject.matrix = cooler.Cooler(
                f"{args.hic_path}::/resolutions/{mlobject.resolution}").matrix(sparse=True).fetch(mlobject.region).toarray()

        elif args.hic_path.endswith(".hic"):

            mlobject.matrix = hicstraw.HiCFile(args.hic_path).getMatrixZoomData(
                mlobject.chrom, mlobject.chrom, 'observed', 'VC_SQRT', 'BP', mlobject.resolution).getRecordsAsMatrix(
                mlobject.start, mlobject.end, mlobject.start, mlobject.end)

        mlobject = misc.clean_matrix(mlobject)

        # This if statement is for detecting empty arrays. If the array is too empty,
        # clean_matrix() will return mlobject.matrix as None.
        if mlobject.matrix is None:

            return

        time_per_region = time()

        for i, cutoff in enumerate(args.cutoffs["values"]):

            time_per_cutoff = time()

            mlobject.kk_cutoff["values"] = args.cutoffs["values"][i]  # Select cut-off for this iteration
            mlobject = kk.get_restraints_matrix(mlobject, False, silent)  # Get submatrix of restraints

            if mlobject.kk_restraints_matrix is None:

                misc.write_bad_region(mlobject, args.work_dir)

                # Save mlobject.
                with open(mlobject.save_path, "wb") as hamlo_namendle:

                    mlobject.save(hamlo_namendle)

                if progress is not None:

                    progress['value'] += 1

                    time_spent = time() - progress['timer']
                    time_remaining = int(time_spent / progress['value'] * (args.total_num - progress['value']))

                    print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
                    print(f"\t[{progress['value']}/{args.total_num}] | Time spent: "
                          f"{timedelta(seconds=round(time_spent))} | "
                          f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')

                return

            if not silent:

                print("\tLayouting Kamada-Kawai...")

            mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject.kk_restraints_matrix))
            mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)
            mlobject.kk_coords = list(mlobject.kk_nodes.values())
            mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

            if len(args.cutoffs["values"]) > 1:

                if not silent:
                    
                    print("\tPlotting Kamada-Kawai...")

                plot.save_mm_kk(mlobject, args.work_dir)

                if progress is not None:

                    progress["plots"] = True

            elif len(args.cutoffs["values"]) == 1:

                if not silent:

                    if args.cutoffs["cutoff_type"] == "percentage":

                        print(
                            f"\tKamada-Kawai layout of region '{mlobject.region}' at {int(cutoff * 100)} % cutoff "
                            f"saved to file: '{mlobject.save_path}'"
                        )

                    elif args.cutoffs["cutoff_type"] == "absolute":

                        print(
                            f"\tKamada-Kawai layout of region '{mlobject.region}' at {cutoff} cutoff saved "
                            f" to file: '{mlobject.save_path}'"
                        )

                # Write to file a list of bad regions, according to the filters defined in clean_matrix().
                with open(f"{args.work_dir}bad_regions.txt", "a+", encoding="utf-8") as handler:

                    log = f"{mlobject.region}\t{mlobject.bad_region}\n"

                    handler.seek(0)

                    if not any(log in line for line in handler) and mlobject.bad_region is not None:

                        handler.write(log)

                misc.write_bad_region(mlobject, args.work_dir)

                # Save mlobject.
                with open(mlobject.save_path, "wb") as hamlo_namendle:

                    mlobject.save(hamlo_namendle)

                if args.save_plots:

                    if not silent:

                        print("\tPlotting Kamada-Kawai...")

                    plot.save_mm_kk(mlobject, args.work_dir)

            if not silent:

                print(f"\tRegion '{mlobject.region}' done in "
                      f"{timedelta(seconds=round(time() - time_per_cutoff))}.\n")

        if not silent:

            print(f"\tdone in {timedelta(seconds=round(time() - time_per_region))}.\n")

    if progress is not None:

        progress['value'] += 1
        time_spent = time() - progress['timer']
        time_remaining = int(time_spent / progress['value'] * (args.total_num - progress['value']))

        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t[{progress['value']}/{args.total_num}] | Time spent: {timedelta(seconds=round(time_spent))} | "
              f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')


def run(opts: list):
    """
    Funtion to run this section of METALoci with the needed arguments.

    Parameters
    ----------
    opts : list
        List of arguments.
    """

    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

    if opts.threads > mp.cpu_count():

        opts.threads = mp.cpu_count()

    cutoffs = {}

    if opts.absolute and opts.cutoff is None or opts.absolute and len(opts.cutoff) == 0:

        sys.exit("Please provide a cutoff (-o) value when using the absolute flag (-a).")

    elif opts.cutoff is None or len(opts.cutoff) == 0:

        cutoffs["cutoff_type"] = "percentage"
        cutoffs["values"] = [0.2]

    elif opts.absolute:

        cutoffs["cutoff_type"] = "absolute"
        cutoffs["values"] = [float(i) for i in opts.cutoff]

    else:

        cutoffs["cutoff_type"] = "percentage"
        cutoffs["values"] = [float(i) for i in opts.cutoff]

    if cutoffs["cutoff_type"] == "percentage" and not 0 < cutoffs["values"][0] <= 1:

        sys.exit("Select a cut-off between 0 and 1.")

    cutoffs["values"].sort(key=float, reverse=True)

    if opts.regions is None:

        if len(glob.glob(f"{opts.work_dir}*coords.txt")) > 1:

            sys.exit("More than one region file found. Please, provide a region or one file with regions of interest.")

        try:

            df_regions = pd.read_table(glob.glob(f"{opts.work_dir}*coords.txt")[0])

        except IndexError:

            sys.exit("No regions provided. Please provide a region or a file with regions of interest or run "
                     "'metaloci sniffer'.")

    elif os.path.isfile(opts.regions):

        df_regions = pd.read_table(opts.regions)

    else:

        df_regions = pd.DataFrame({"coords": [opts.regions], "symbol": ["symbol"], "id": ["id"]})

    # Debug 'menu' for testing purposes
    if opts.debug:

        print(f"work_dir ->\n\t{opts.work_dir}")
        print(f"hic_path ->\n\t{opts.hic_file}")
        print(f"resolution ->\n\t{opts.resolution}")
        print(f"regions ->\n\t{opts.regions}")
        print(f"cutoffs ->\n\t{cutoffs}")
        print(f"absolute ->\n\t{opts.absolute}")
        print(f"persistence_length ->\n\t{opts.persistence_length}")
        print(f"force ->\n\t{opts.force}")
        print(f"save_plots ->\n\t{opts.save_plots}")
        print(f"multiprocess ->\n\t{opts.multiprocess}")
        print(f"cores ->\n\t{opts.threads}")

        sys.exit()

    # Checking if the resolution supplied by the user is within the resolutions of the cool/mcool/hic files
    if opts.hic_file.endswith(".cool"):

        available_resolutions = cooler.Cooler(opts.hic_file).binsize

        if opts.resolution != available_resolutions:

            sys.exit("The given resolution is not the same as the provided cooler file. Exiting...")

    elif opts.hic_file.endswith(".mcool"):

        available_resolutions = [int(x) for x in list(h5py.File(opts.hic_file)["resolutions"].keys())]

        if opts.resolution not in available_resolutions:

            print(
                f"The given resolution is not in the provided mcooler file.\nThe available resolutions are: "
                f"{', '.join(misc.natural_sort([str(x) for x in available_resolutions]))}"
            )
            sys.exit("Exiting...")

    elif opts.hic_file.endswith(".hic"):

        available_resolutions = hicstraw.HiCFile(opts.hic_file).getResolutions()

        if opts.resolution not in available_resolutions:

            print(
                f"The given resolution is not in the provided mcooler file.\nThe available resolutions are: "
                f"{', '.join(misc.natural_sort([str(x) for x in available_resolutions]))}"
            )
            sys.exit("Exiting...")

    else:

        print("HiC file format not supported. Supported formats are: cool, mcool, hic.")
        sys.exit("Exiting...")

    parsed_args = pd.Series({"work_dir": opts.work_dir,
                             "hic_path": opts.hic_file,
                             "resolution": opts.resolution,
                             "cutoffs": cutoffs,
                             "persistence_length": opts.persistence_length,
                             "force": opts.force,
                             "save_plots": opts.save_plots,
                             "total_num": len(df_regions)})

    start_timer = time()

    pathlib.Path(os.path.join(opts.work_dir)).mkdir(parents=True, exist_ok=True)

    if opts.multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:

            progress = mp.Manager().dict(value=0, timer=start_timer, done=False, plots=False)

            with mp.Pool(processes=opts.threads) as pool:

                pool.starmap(get_region_layout, [(row, parsed_args, progress) for _, row in df_regions.iterrows()])

                if progress["done"]:

                    print("\tSome regions had already been computed and have been skipped.", end="")

                if progress["plots"]:

                    print(f"\n\tKamada-Kawai layout plots saved to '{opts.work_dir}chr/plots/KK'", end="")

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for counter, row in df_regions.iterrows():

            get_region_layout(row, parsed_args, counter=counter, silent=False)

    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nAll done.")
