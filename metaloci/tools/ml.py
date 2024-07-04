"""
Adds signal data to a Kamada-Kawai layout and calculates Local Moran's I for every bin in the layout.
"""

import glob
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import subprocess as sp
import sys
from argparse import SUPPRESS, HelpFormatter
from collections import defaultdict
from datetime import timedelta
from time import time

import pandas as pd
from metaloci.misc import misc
from metaloci.spatial_stats import lmi

HELP = "Calculates Local Moran's I for every bin in a Kamada-Kawai layout."

DESCRIPTION = """
Adds signal data to a Kamada-Kawai layout and calculates Local Moran's I for every bin in the layout. Outputs
a .mlo file with the LMI data for each signal. It can also output a csv with info for each signal and bed files with
the metalocis found, depending to the flags you set. 
"""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step.

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller.
    """

    parser.formatter_class = lambda prog: HelpFormatter(prog, width=120, max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")

    input_arg.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to working directory."
    )

    input_arg.add_argument(
        "-s",
        "--signal",
        dest="signals",
        metavar="FILE",
        type=str,
        required=True,
        help="Path to the file with the samples/signals to use."
    )

    input_arg.add_argument(
        "-g",
        "--region",
        dest="region_file",
        metavar="PATH",
        type=str,
        help="Region to apply LMI in format chrN:start-end_poi or file containing the regions of interest."
    )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.")

    optional_arg.add_argument(
        "-p",
        "--permutations",
        dest="perms",
        default=9999,
        metavar="INT",
        type=int,
        help="Number of permutations to calculate the Local Moran's I p-value (default: %(default)d)."
    )

    optional_arg.add_argument(
        "-v",
        "--pval",
        dest="signipval",
        default=0.05,
        metavar="FLOAT",
        type=float,
        help="P-value significance threshold (default: %(default)4.2f)."
    )

    optional_arg.add_argument(
        "-a",
        "--aggregated",
        dest="agg",
        metavar="PATH",
        type=str,
        required=False,
        help="Use the file with aggregated signals. This file has 2 columns: first column the name of the original \
        signal, second column the new name of the aggregate, separated by tabs. "
    )

    optional_arg.add_argument(
        "-i",
        "--info",
        dest="moran_info",
        action="store_true",
        help="Flag to unpickle LMI info."
    )

    optional_arg.add_argument(
        "-m",
        "--mp",
        dest="multiprocess",
        action="store_true",
        help="Flag to set use of multiprocessing."
    )

    optional_arg.add_argument(
        "-t",
        "--threads",
        dest="threads",
        default=int(mp.cpu_count() / 3),
        type=int,
        action="store",
        help="Number of threads to use in multiprocessing. Recommended value is one third of your "
        " total cpu count, although increasing this number may improve performance in machines "
        "with few cores. (default: %(default)s)"
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

    optional_arg.add_argument(
        "-b",
        "--bed",
        dest="save_bed",
        action="store_true",
        help="Flag to save the bed file with the metalocis location."
    )

    optional_arg.add_argument(
        "-q",
        "--quadrants",
        dest="quadrants",
        metavar="INT",
        type=int,
        nargs="+",
        default=[1, 3],
        help="Space-separated list with the LMI quadrants to highlight (default: %(default)s). \
        1: High-high (signal in bin is high, signal on neighbours is high). \
        2: High-Low (signal in bin is high, signal on neighbours is low). \
        3: Low-Low (signal in bin is low, signal on neighbours is low). \
        4: Low-High (signal in bin is low, signal on neighbours is high).",
    )

    optional_arg.add_argument(
        "-po",
        "--poi_only",
        dest="poi_only",
        action="store_true",
        help="Flag to only save the point of interest row in the LMI dataframes. Useful for large datasets."
    )


def get_lmi(row: pd.Series, args: pd.Series,
            progress=None, counter: int = None, silent: bool = True):
    """
    This function get the Local Moran's I given the arguments.

    Parameters
    ----------
    row : pd.Series
        Data of the region.
    args : pd.Series
        pd.Series with all the needed argument info to run the function.
    progress : optional
        Information about if this region has been completed, by default None.
    counter : int, optional
        Counter used for when multiprocessing is not active, by default None.
    silent : bool, optional
        Flag to enable or disable the print statements. Useful for multiprocessing.

    Raises
    ------
    Exception
        Exception to handle if the user puts "extra" signals.
    """

    INFLUENCE = 1.5
    BFACT = 2

    region_timer = time()

    if not silent:

        print(f"\n------> Working on region {row.coords} [{counter+1}/{args.total_num}]\n")

    save_path = f"{args.work_dir}{row.coords.split(':', 1)[0]}/objects/{re.sub(':|-', '_', row.coords)}.mlo"

    # The chunk below checks if the region can be processed (has a valid .mlo that has KK information and has signal
    # data). If it can't be processed, it will skip to the next region.
    try:

        with open(save_path, "rb",) as mlobject_handler:

            mlobject = pickle.load(mlobject_handler)
            # This ensures the path is still right even if the user changes the working directory.
            mlobject.save_path = save_path
            
    except FileNotFoundError:

        if not silent:

            print("\t.mlo file not found for this region.\n\tSkipping to next region...")

        return

    if mlobject.kk_nodes is None:

        if not silent:

            print("\tKamada-Kawai layout has not been calculated for this region. \n\tSkipping to next region...")

        return

    try:

        signal_data = pd.read_csv(
            glob.glob(f"{os.path.join(args.work_dir, 'signal', mlobject.chrom)}/*_signal.tsv")[0], sep="\t", header=0)

    except IndexError:

        if not silent:

            print("\tNo signal data found for this region.\n\tSkipping to next region...")

        if progress is not None:

            progress['value'] += 1
            time_spent = time() - progress['timer']
            time_remaining = int(time_spent / progress['value'] * (args.total_num - progress['value']))

            print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
            print(f"\t[{progress['value']}/{args.total_num}] | Time spent: {timedelta(seconds=round(time_spent))} | "
                  f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')

        return

    # Load signals if it has not already been done
    if mlobject.signals_dict is None:

        mlobject.signals_dict = lmi.load_region_signals(mlobject, signal_data, args.signals)

    # Agreggate signals if needed
    if args.aggregate is not None:

        mlobject.agg = args.agg_dict

        lmi.aggregate_signals(mlobject)

    # If the signal is not present in the signals folder (has not been processed with prep) but it is in the list of
    # signal to process, raise an exception and print which signals need to be processed.
    if mlobject.signals_dict is None and progress is not None:

        progress["missing_signal"] = list(mlobject.signals_dict.keys())

        raise Exception()  # This exception is probably a bad idea

    # Get average distance between consecutive points to define influence, which should be ~2 particles of radius.
    neighbourhood = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT

    if args.force:

        mlobject.lmi_info = {}
        mlobject.lmi_geometry = None

    # This checks if every signal you want to process is already computed. If the user works with a
    # few signals but decides to add some more later, the user can use the same working directory and
    # KK, and the LMI will be computed again with the new signals. If everything is already
    # computed, does nothing.
    if mlobject.lmi_info is not None and args.force is False:

        # If the list of signals already processed is equal to the list of signals to process, skip to the next region.
        if [signal for signal in mlobject.lmi_info.keys()] == list(mlobject.signals_dict.keys()):

            if progress is not None:

                progress["done"] = True

            for signal, df in mlobject.lmi_info.items():

                if args.moran_info:

                    misc.write_moran_data(mlobject, args, silent)

                if args.save_bed:

                    misc.write_bed(mlobject, signal, neighbourhood, BFACT, args, silent)

            if not silent:

                print("\tLMI already computed for this region.\n\tSkipping to next region...")

            return

    if mlobject.lmi_geometry is None:

        mlobject.lmi_geometry = lmi.construct_voronoi(mlobject, mlobject.kk_distances.diagonal(1).mean() * INFLUENCE)

    if not silent:

        print(f"\tAverage distance between consecutive particles: {mlobject.kk_distances.diagonal(1).mean():6.4f}"
              f" [{mlobject.kk_distances.diagonal(1).mean() * INFLUENCE:6.4f}]")
        print(f"\tGeometry information for region {mlobject.region} saved to '{mlobject.save_path}'")

    for signal_type in list(mlobject.signals_dict.keys()):

        if signal_type not in mlobject.lmi_info:

            mlobject.lmi_info[signal_type] = lmi.compute_lmi(mlobject, signal_type, neighbourhood,
                                                             args.n_permutations, args.signipval, silent, args.poi_only)

            if args.moran_info:

                misc.write_moran_data(mlobject, args, silent)

            if args.save_bed:

                misc.write_bed(mlobject, signal_type, neighbourhood, BFACT, args, silent)

    if not silent:

        print(f"\n\tRegion done in {timedelta(seconds=round(time() - region_timer))}")
        print(f"\tLMI information for region {mlobject.region} will be saved to '{mlobject.save_path}'")

    with open(mlobject.save_path, "wb") as hamlo_namendle:

        mlobject.save(hamlo_namendle)

    if progress is not None:

        progress['value'] += 1

        time_spent = time() - progress['timer']
        time_remaining = int(time_spent / progress['value'] * (args.total_num - progress['value']))

        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t[{progress['value']}/{args.total_num}] | Time spent: {timedelta(seconds=round(time_spent))} | "
              f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')


def run(opts):
    """
    Funtion to run this section of METALoci with the needed arguments.

    Parameters
    ----------
    opts : list
        List of arguments.
    """
    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

    if opts.region_file is None:

        if len(glob.glob(f"{opts.work_dir}*coords.txt")) > 1:

            sys.exit("More than one region file found. Please provide a region or one file with regions of interest'.")

        try:

            df_regions = pd.read_table(glob.glob(f"{opts.work_dir}*coords.txt")[0])

        except IndexError:

            sys.exit("No regions provided. Please provide a region or a file with regions of interest or run "
                     "'metaloci sniffer'.")

    elif os.path.isfile(opts.region_file):

        df_regions = pd.read_table(opts.region_file)

    else:

        df_regions = pd.DataFrame({"coords": [opts.region_file], "symbol": ["symbol"], "id": ["id"]})

    if opts.debug:

        print(f"work_dir ->\n\t{opts.work_dir}")
        print(f"signals ->\n\t{opts.signals}")
        print(f"regions ->\n\t{opts.region_file}")
        print(f"n_permutations ->\n\t{opts.perms}")
        print(f"signipval ->\n\t{opts.signipval}")
        print(f"aggregate ->\n\t{opts.agg}")
        print(f"moran_info ->\n\t{opts.moran_info}")
        print(f"multiprocess ->\n\t{opts.multiprocess}")
        print(f"cores ->\n\t{opts.threads}")
        print(f"force ->\n\t{opts.force}")
        print(f"poi_only ->\n\t{opts.poi_only}")

        sys.exit()

    agg_dict = defaultdict(list)

    if opts.agg is not None:

        for _, row in pd.read_csv(opts.agg, sep="\t", header=None).iterrows():

            agg_dict[row[1]].append(row[0])

    args = pd.Series({"work_dir": opts.work_dir,
                      "signals": opts.signals,
                      "n_permutations": opts.perms,
                      "signipval": opts.signipval,
                      "aggregate": opts.agg,
                      "moran_info": opts.moran_info,
                      "agg_dict": agg_dict,
                      "force": opts.force,
                      "total_num": len(df_regions),
                      "save_bed": opts.save_bed,
                      "quadrants": opts.quadrants,
                      "poi_only": opts.poi_only})

    start_timer = time()

    if opts.multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:

            progress = mp.Manager().dict(value=0, timer=start_timer, done=False, missing_signal=None)

            with mp.Pool(processes=opts.threads) as pool:

                try:

                    pool.starmap(get_lmi, [(row, args, progress) for _, row in df_regions.iterrows()])

                    if progress["done"]:

                        print("\tSome regions had already been computed and have been skipped.", end="")

                        if opts.moran_info:

                            print(f"\nMoran data saved to: '{opts.work_dir}chr/moran_log/'", end="")

                except Exception:

                    if progress["missing_signal"] is not None:

                        print(f"\tSignal '{progress['missing_signal']}' is in the signal list but has not been"
                                "processed with prep."
                                "\n\tProcess that signal or remove it from the signal list."
                                "\n\tExiting...")

                    pool.close()
                    pool.join()

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for counter, row in df_regions.iterrows():

            get_lmi(row, args, counter=counter, silent=False)

    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nAll done.")
