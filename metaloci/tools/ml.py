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
from metaloci.spatial_stats import lmi

DESCRIPTION = """
Adds signal data to a Kamada-Kawai layout and calculates Local Moran's I for every bin in the layout.
"""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller
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
        help="Region to apply LMI in format chrN:start-end_midpoint or file containing the regions of interest."
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
        dest="info",
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


def get_lmi(row: pd.Series, args_2_use: pd.Series,
            progress=None, counter: int = None, silent: bool = True):
    """
    This function get the Local Moran's I given the arguments.

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

    Raises
    ------
    Exception
        Exception to handle if the user puts "extra" signals
    """

    INFLUENCE = 1.5
    BFACT = 2

    work_dir = args_2_use.work_dir
    signals = args_2_use.signals
    n_permutations = args_2_use.n_permutations
    signipval = args_2_use.signipval
    aggregate = args_2_use.aggr
    moran_info = args_2_use.moran_info
    agg_dict = args_2_use.agg_dict
    force = args_2_use.force
    total_num = args_2_use.total_num

    region_timer = time()

    if not silent:

        print(f"\n------> Working on region {row.coords} [{counter+1}/{total_num}]\n")

    save_path = f"{work_dir}{row.coords.split(':', 1)[0]}/{re.sub(':|-', '_', row.coords)}.mlo"

    try:

        with open(save_path, "rb",) as mlobject_handler:

            mlobject = pickle.load(mlobject_handler)
            # Save path seems to be the same as the one defined in the layout
            mlobject.save_path = f"{work_dir}{row.coords.split(':', 1)[0]}/{re.sub(':|-', '_', row.coords)}.mlo"

    except FileNotFoundError:

        if not silent:

            print("\t.mlo file not found for this region.\n\tSkipping to next region...")

        return

    if mlobject.kk_nodes is None:

        if not silent:

            print("\tKamada-Kawai layout has not been calculated for this region. \n\tSkipping to next region...")

        return

    try:

        signal_data = pd.read_pickle(glob.glob(f"{os.path.join(work_dir, 'signal', mlobject.chrom)}/*_signal.pkl")[0])

    except IndexError:

        if not silent:

            print("\tNo signal data found for this region.\n\tSkipping to next region...")

        if progress is not None:

            progress['value'] += 1
            time_spent = time() - progress['timer']
            time_remaining = int(time_spent / progress['value'] * (total_num - progress['value']))

            print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
            print(f"\t[{progress['value']}/{total_num}] | Time spent: {timedelta(seconds=round(time_spent))} | "
                  f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')

        return

    mlobject.signals_dict = lmi.load_region_signals(mlobject, signal_data, signals)

    if aggregate is not None:

        mlobject.agg = agg_dict

        lmi.aggregate_signals(mlobject)

    # If the signal is not present in the signals folder (has not been processed with prep)
    # but it is in the list of signal to process, raise an exception and print which signals need
    # to be processed.
    if mlobject.signals_dict is None and progress is not None:

        progress["missing_signal"] = list(mlobject.signals_dict.keys())

        raise Exception()

    # This checks if every signal you want to process is already computed. If the user works with a
    # few signals but decides to add some more later, the user can use the same working directory and
    # KK, and the LMI will be computed again with the new signals. If everything is already
    # computed, does nothing.
    if mlobject.lmi_info is not None and force is False:

        # Better version of the code, I think
        if [signal for signal in mlobject.lmi_info.keys()] == list(mlobject.signals_dict.keys()):

            if progress is not None:

                progress["done"] = True

            if moran_info:

                for signal, df in mlobject.lmi_info.items():

                    moran_data_path = os.path.join(work_dir, mlobject.chrom, "moran_data", signal)

                    pathlib.Path(moran_data_path).mkdir(parents=True, exist_ok=True)
                    df.to_csv(
                        os.path.join(
                            moran_data_path,
                            f"{re.sub(':|-', '_', mlobject.region)}_{mlobject.poi}_{signal}.tsv"),
                        sep="\t", index=False, float_format="%.12f")

                    if not silent:

                        print(F"\tMoran data saved to \
                              '{moran_data_path}/{re.sub(':|-', '_', mlobject.region)}_{mlobject.poi}_{signal}.tsv'")

            if not silent:

                print("\tLMI already computed for this region.\n\tSkipping to next region...")

            return

    # Get average distance between consecutive points to define influence, which should be ~2 particles of radius.
    neighbourhood = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT

    if mlobject.lmi_geometry is None:

        mlobject.lmi_geometry = lmi.construct_voronoi(mlobject, mlobject.kk_distances.diagonal(1).mean() * INFLUENCE)

    if not silent:

        print(f"\tAverage distance between consecutive particles: {mlobject.kk_distances.diagonal(1).mean():6.4f}"
              f" [{mlobject.kk_distances.diagonal(1).mean() * INFLUENCE:6.4f}]")
        print(f"\tGeometry information for region {mlobject.region} saved to '{mlobject.save_path}'")

    for signal_type in list(mlobject.signals_dict.keys()):

        if signal_type not in mlobject.lmi_info:

            mlobject.lmi_info[signal_type] = lmi.compute_lmi(mlobject, signal_type, neighbourhood,
                                                             n_permutations, signipval, silent)

    if not silent:

        print(f"\n\tRegion done in {timedelta(seconds=round(time() - region_timer))}")
        print(f"\tLMI information for region {mlobject.region} will be saved to '{mlobject.save_path}'")

    with open(mlobject.save_path, "wb") as hamlo_namendle:

        mlobject.save(hamlo_namendle)

    if moran_info:

        for signal, df in mlobject.lmi_info.items():

            moran_data_path = os.path.join(work_dir, mlobject.chrom, "moran_data", signal)

            pathlib.Path(moran_data_path).mkdir(parents=True, exist_ok=True)
            df.to_csv(
                os.path.join(
                    moran_data_path,
                    f"{re.sub(':|-', '_', mlobject.region)}_{mlobject.poi}_{signal}.tsv"),
                sep="\t", index=False)

            if not silent:

                print(f"\tMoran data saved to "
                      f"'{moran_data_path}/{re.sub(':|-', '_', mlobject.region)}_{mlobject.poi}_{signal}.tsv'")

    if progress is not None:

        progress['value'] += 1

        time_spent = time() - progress['timer']
        time_remaining = int(time_spent / progress['value'] * (total_num - progress['value']))

        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t[{progress['value']}/{total_num}] | Time spent: {timedelta(seconds=round(time_spent))} | "
              f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')


def run(opts):
    """
    Funtion to run this section of METALoci with the needed arguments

    Parameters
    ----------
    opts : list
        List of arguments
    """

    work_dir = opts.work_dir
    signals = opts.signals
    regions = opts.region_file
    n_permutations = opts.perms
    signipval = opts.signipval
    aggregate = opts.agg
    moran_info = opts.info
    multiprocess = opts.multiprocess
    threads = opts.threads
    force = opts.force

    if not work_dir.endswith("/"):

        work_dir += "/"

    if regions is None:

        if len(glob.glob(f"{work_dir}*coords.txt")) > 1:

            sys.exit("More than one region file found. Please provide a region or one file with regions of interest'.")

        try:

            df_regions = pd.read_table(glob.glob(f"{work_dir}*coords.txt")[0])

        except IndexError:

            sys.exit("No regions provided. Please provide a region or a file with regions of interest or run \
                     'metaloci sniffer'.")

    elif os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    if opts.debug:

        print(f"work_dir ->\n\t{work_dir}")
        print(f"signals ->\n\t{signals}")
        print(f"regions ->\n\t{regions}")
        print(f"n_permutations ->\n\t{n_permutations}")
        print(f"signipval ->\n\t{signipval}")
        print(f"aggregate ->\n\t{aggregate}")
        print(f"moran_info ->\n\t{moran_info}")
        print(f"multiprocess ->\n\t{multiprocess}")
        print(f"cores ->\n\t{threads}")
        print(f"force ->\n\t{force}")

        sys.exit()

    agg_dict = defaultdict(list)

    if aggregate is not None:

        for _, row in pd.read_csv(aggregate, sep="\t", header=None).iterrows():

            agg_dict[row[0]].append(row[1])

    parsed_args = pd.Series({"work_dir": work_dir,
                             "signals": signals,
                             "n_permutations": n_permutations,
                             "signipval": signipval,
                             "aggr": aggregate,
                             "moran_info": moran_info,
                             "agg_dict": agg_dict,
                             "force": force,
                             "total_num": len(df_regions)})

    start_timer = time()

    if multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:

            progress = mp.Manager().dict(value=0, timer=start_timer, done=False, missing_signal=None)

            with mp.Pool(processes=threads) as pool:

                try:

                    pool.starmap(get_lmi, [(row, parsed_args, progress) for _, row in df_regions.iterrows()])

                    if progress["done"]:

                        print("\tSome regions had already been computed and have been skipped.", end="")

                        if moran_info:

                            print(f"\nMoran data saved to: '{work_dir}chr/moran_log/'", end="")

                except Exception:

                    if progress["missing_signal"] is not None:

                        print(f"\tSignal '{progress['missing_signal']}' is in the signal list but has not been \
                                processed with prep.\
                                \n\tProcess that signal or remove it from the signal list.\
                                \n\tExiting...")

                    pool.close()
                    pool.join()

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for counter, row in df_regions.iterrows():

            get_lmi(row, parsed_args, counter=counter, silent=False)

    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nAll done.")
