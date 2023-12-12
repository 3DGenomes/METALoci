"""
Script to "paint" the Kamada-Kawai layaouts using a signal, grouping individuals by type
"""
from collections import defaultdict
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import subprocess as sp
from argparse import HelpFormatter
from datetime import timedelta
from time import time

import glob
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
        "-a",
        "--aggregated",
        dest="agg",
        metavar="PATH",
        type=str,
        required=False,
        help="Use the file with aggregated signals",
    )

    optional_arg.add_argument(
        "-i",
        "--info",
        dest="info",
        action="store_true",
        help="Flag to unpickle LMI info.",
    )

    optional_arg.add_argument(
        "-m",
        "--mp",
        dest="multiprocess",
        action="store_true",
        help="Flag to set use of multiprocessing.",
    )

    optional_arg.add_argument(
        "-t", "--threads", dest="threads", default=int(mp.cpu_count() / 3), type=int, action="store", help="Number" 
        " of threads to use in multiprocessing. Recommended value is one third of your total cpu count, although"
        " increasing this number may improve performance in machines with few cores. (default: %(default)s)"
    )

    optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")


def get_lmi(region_iter, opts, progress=None, i=None, silent: bool = True):

    INFLUENCE = 1.5
    BFACT = 2

    work_dir = opts.work_dir
    signals = opts.signals
    regions = opts.region_file
    n_permutations = opts.perms
    signipval = opts.signipval
    moran_info = opts.info
    force = opts.force
    aggregate = opts.agg

    if not work_dir.endswith("/"):

        work_dir += "/"

    if regions is None:

        try:

            df_regions = pd.read_table(glob.glob(f"{work_dir}*coords.txt")[0])
            
            if len(glob.glob(f"{work_dir}*coords.txt")) > 1:
                
                print("More than one region file found. Please provide a region or only one file with regions of interest'.")
                return
        
        except IndexError:
                
            print("No regions provided. Please provide a region or a file with regions of interest or run 'metaloci sniffer'.")
            return
        
    elif os.path.isfile(regions):

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
            mlobject.save_path = f"{work_dir}{region.split(':', 1)[0]}/{re.sub(':|-', '_', region)}.mlo"

    except FileNotFoundError:

        if silent == False:
            print("\t.mlo file not found for this region.\n\tSkipping to next region...")
            
        return

    if mlobject.kk_nodes is None:

        if silent == False:
            print("\tKamada-Kawai layout has not been calculated for this region. \n\tSkipping to next region...")
            
        return

    # Load only signal for this specific region.
    # signal_data = lmi.load_signals(df_regions, work_dir=work_dir)
    
    try:
        
        signal_data = pd.read_pickle(glob.glob(f"{os.path.join(work_dir, 'signal', mlobject.chrom)}/*_signal.pkl")[0])
    
    except IndexError:
    
        if silent == False:
            
            print("\tNo signal data found for this region.\n\tSkipping to next region...")
            
        if progress is not None:

            progress['value'] += 1

            time_spent = time() - progress['timer']
            time_remaining = int(time_spent / progress['value'] * (len(df_regions) - progress['value']))

            print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
            print(f"\t[{progress['value']}/{len(df_regions)}] | Time spent: {timedelta(seconds=round(time_spent))} | "
                    f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')
                
        return
    
    mlobject.signals_dict = lmi.load_region_signals(mlobject, signal_data, signals)

    if aggregate is not None:

        mlobject.agg = defaultdict(list); [mlobject.agg[row[0]].append(row[1]) for _, row in pd.read_csv(aggregate, sep="\t", header=None).iterrows()]

        lmi.aggregate_signals(mlobject)

    # If the signal is not present in the signals folder (has not been processed with prep) but it is in 
    # the list of signal to process, raise an exception and print which signals need to be processed.
    if mlobject.signals_dict == None and progress is not None:

        progress["missing_signal"] = list(mlobject.signals_dict.keys())   
        raise Exception()

    # This checks if every signal you want to process is already computed. If the user works with a few signals 
    # but decides to add some more later, she can use the same working directory and KK, and the LMI will be
    # computed again with the new signals. If everything is already computed, does nothing.
    if mlobject.lmi_info is not None and force is False:

        if [signal for signal, _ in mlobject.lmi_info.items()] == list(mlobject.signals_dict.keys()):

            if progress is not None: progress["done"] = True

            if moran_info:

                for signal, df in mlobject.lmi_info.items():

                    moran_data_path = f"{work_dir}{mlobject.chrom}/moran_data/{signal}"
                    pathlib.Path(moran_data_path).mkdir(parents=True, exist_ok=True)

                    df.to_csv(f"{moran_data_path}/{mlobject.region}_{mlobject.poi}_{signal}.tsv", sep="\t", index=False, float_format="%.12f")

                    if silent == False:
                        print(f"\tMoran data saved to '{moran_data_path}/{mlobject.region}_{mlobject.poi}_{signal}.tsv'")

            if silent == False:
                print("\tLMI already computed for this region. \n\tSkipping to next region...")
                
            return

    # Get average distance between consecutive points to define influence,
    # which should be ~2 particles of radius.
    mean_distance = mlobject.kk_distances.diagonal(1).mean()
    buffer = mean_distance * INFLUENCE
    
    if mlobject.lmi_geometry is None:

        mlobject.lmi_geometry = lmi.construct_voronoi(mlobject, buffer)

    if silent == False:
        print(f"\tAverage distance between consecutive particles: {mean_distance:6.4f} [{buffer:6.4f}]")
        print(f"\tGeometry information for region {mlobject.region} saved to '{mlobject.save_path}'")

    for signal_type in list(mlobject.signals_dict.keys()):

        if signal_type not in mlobject.lmi_info:

            mlobject.lmi_info[signal_type] = lmi.compute_lmi(mlobject, signal_type, buffer * BFACT, n_permutations, signipval, silent)

    if silent == False:
        print(f"\n\tRegion done in {timedelta(seconds=round(time() - region_timer))}")
        print(f"\tLMI information for region {mlobject.region} will be saved to '{mlobject.save_path}'")

    with open(mlobject.save_path, "wb") as hamlo_namendle:

        mlobject.save(hamlo_namendle)

    if moran_info:

        for signal, df in mlobject.lmi_info.items():

            moran_data_path = f"{work_dir}{mlobject.chrom}/moran_data/{signal}"
            pathlib.Path(moran_data_path).mkdir(parents=True, exist_ok=True) 

            df.to_csv(f"{moran_data_path}/{mlobject.region}_{mlobject.poi}_{signal}.tsv", sep="\t", index=False)

            if silent == False:
                print(f"\tMoran data saved to '{moran_data_path}/{mlobject.region}_{mlobject.poi}_{signal}.tsv'")

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
    moran_info = opts.info

    if opts.threads is None:

        cores = int(mp.cpu_count() / 3)

    if not work_dir.endswith("/"):

        work_dir += "/"
    
    if regions is None:

        try:

            df_regions = pd.read_table(glob.glob(f"{work_dir}*coords.txt")[0])
            
            if len(glob.glob(f"{work_dir}*coords.txt")) > 1:
                
                print("More than one region file found. Please provide a region or only one file with regions of interest'.")
                return
        
        except IndexError:
                
            print("No regions provided. Please provide a region or a file with regions of interest or run 'metaloci sniffer'.")
            return
        
    elif os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})
        
    start_timer = time()

    if multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:
            
            progress = mp.Manager().dict(value=0, timer = start_timer, done = False, missing_signal = None)
        
            with mp.Pool(processes=cores) as pool:

                try:

                    pool.starmap(get_lmi, [(row, opts, progress) for _, row in df_regions.iterrows()])

                    if progress["done"] == True:

                        print("\tSome regions had already been computed and have been skipped.", end="")
                    
                        if moran_info:

                            print(f"\n\Moran data saved to: '{work_dir}chr/moran_log/'", end="")

                except Exception:

                    if progress["missing_signal"] is not None:

                        print(f"\tSignal '{progress['missing_signal']}' is in the signal list but has not been processed with prep.\n"
                              "\tProcess that signal or remove it from the signal list.\n\tExiting...")
                
                    pool.close()
                    pool.join()

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for i, row in df_regions.iterrows():

            get_lmi(row, opts, i=i, silent=False)

    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nall done.")
