"""
Script to "paint" the Kamada-Kawai layaouts using a signal, grouping individuals by type
"""
import multiprocessing as mp
import os
import pickle
import re
import subprocess as sp
from argparse import HelpFormatter
from datetime import timedelta
from time import time

import pandas as pd

from metaloci.spatial_stats import lmi

DESCRIPTION = (
    "Adds signal data to a Kamada-Kawai layout and calculates Local Moran's I for every bin in the layout."
)


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
        "-s",
        "--signal",
        dest="signals",
        metavar="FILE",
        type=str,
        required=True,
        help="Path to the file with the samples/signals to use.",
    )

    input_arg.add_argument(
        "-g",
        "--region",
        dest="region_file",
        metavar="PATH",
        type=str,
        help="Region to apply LMI in format chrN:start-end_midpoint or file containing the regions of interest.",
    )

    optional_arg.add_argument("-f", "--force", dest="force", action="store_true", help="Force rewriting existing data.")

    optional_arg.add_argument(
        "-p",
        "--permutations",
        dest="perms",
        default=9999,
        metavar="INT",
        type=int,
        help="Number of permutations to calculate the Local Moran's I p-value (default: %(default)d).",
    )

    optional_arg.add_argument(
        "-v",
        "--pval",
        dest="signipval",
        default=0.05,
        metavar="FLOAT",
        type=float,
        help="P-value significance threshold (default: %(default)4.2f).",
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

def get_lmi(region_iter, opts, signal_data, progress=None, i=None, silent: bool = True):

    INFLUENCE = 1.5
    BFACT = 2

    work_dir = opts.work_dir
    signals = opts.signals
    regions = opts.region_file
    n_permutations = opts.perms
    signipval = opts.signipval
    force = opts.force

    if not work_dir.endswith("/"):

        work_dir += "/"

    if os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    region_timer = time()

    region = region_iter.coords

    if silent == False:

        print(f"\n------> Working on region {region} [{i+1}/{len(df_regions)}]\n")

    try:

        with open(
            f"{work_dir}{region.split(':', 1)[0]}/{re.sub(':|-', '_', region)}.mlo",
            "rb",
        ) as mlobject_handler:

            mlobject = pickle.load(mlobject_handler)

    except FileNotFoundError:

        if silent == False:
            print("\t.mlo file not found for this region.\n\tSkipping to next region...")

        return

    if mlobject.kk_nodes is None:

        if silent == False:

            print("Kamada-Kawai layout has not been calculated for this region. \n\tSkipping to next region...")

        return

    # Load only signal for this specific region.
    mlobject.signals_dict, signal_types = lmi.load_region_signals(mlobject, signal_data, signals)

    # This checks if every signal you want to process is already computed. If the user works with a few signals 
    # but decides to add some more later, she can use the same working directory and KK, and the LMI will be
    # computed again with the new signals. If everything is already computed, does nothing.
    if mlobject.lmi_info is not None:

        if [k for k, _ in mlobject.lmi_info.items()] == signal_types:

            if silent == False:

                print("\tLMI already computed for this region. \n\tSkipping to next region...")

            if progress is not None: progress["done"] = True

            return

    # Get average distance between consecutive points to define influence,
    # which should be ~2 particles of radius.
    mean_distance = mlobject.kk_distances.diagonal(1).mean()
    buffer = mean_distance * INFLUENCE
    mlobject.lmi_geometry = lmi.construct_voronoi(mlobject, buffer)

    if silent == False:

        print(f"\tAverage distance between consecutive particles: {mean_distance:6.4f} [{buffer:6.4f}]")
        print(f"\tGeometry information for region {mlobject.region} saved to: {mlobject.save_path}")

    all_lmi = {}

    for signal_type in signal_types:

        all_lmi[signal_type] = lmi.compute_lmi(mlobject, signal_type, buffer * BFACT, n_permutations, signipval, silent)

    mlobject.lmi_info = all_lmi

    if silent == False:

        print(f"\n\tRegion done in {timedelta(seconds=round(time() - region_timer))}")
        print(f"\tLMI information for region {mlobject.region} will be saved to: {mlobject.save_path}")

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
    regions = opts.region_file
    multiprocess = opts.multiprocess
    cores = opts.threads

    if not work_dir.endswith("/"):

        work_dir += "/"

    start_timer = time()

    # Read region list. If its a region as parameter, create a dataframe.
    # If its a path to a file, read that dataframe.
    if os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    # Read the signal of the chromosomes corresponding to the regions of interest,
    # not to load useless data.
    signal_data = lmi.load_signals(df_regions, work_dir=work_dir)

    if multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:
            
            manager = mp.Manager()
            progress = manager.dict(value=0, timer = start_timer, done = False)
        
            with mp.Pool(processes=cores) as pool:

                pool.starmap(get_lmi, [(row, opts, signal_data, progress) for _, row in df_regions.iterrows()])

                if progress["done"] == True:

                    print("\tSome regions had already been computed and have been skipped.", end="")

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for i, row in df_regions.iterrows():

            get_lmi(row, opts, signal_data, i=i, silent=False)

    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nall done.")
