"""
This script generates METALoci plots from METALoci objects. The object needs to have a Kamakawa-Kawai layout and a 
Local Moran's I already calculated. It will generate the following plots:
    
    * HiC matrix
    * Signal plot
    * Kamada-Kawai layout
    * Local Moran's I scatterplot
    * Gaudí plot for signal
    * Gaudí plot for LMI quadrant
    
and a composite image with all of the above.
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
from datetime import timedelta
from time import time

import fitz  # PyMuPDF
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from metaloci import misc, mlo
from metaloci.misc import misc
from metaloci.plot import plot
from PIL import Image

HELP = "Plots METALoci output."

DESCRIPTION = """Outputs METALoci output.
It creates the following plots: Hi-C matrix, Signal plot, Kamada-Kawai layout, Local Moran's I scatterplot,
Gaudí plot for signal, Gaudí plot for LMI quadrant, and a composite image with all the above."""


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
        "--signals",
        dest="signals",
        required=True,
        metavar="STR",
        type=str,
        nargs="*",
        action="extend",
        help="Space-separated list of signals to plot or path to the file with the list of signals to plot, "
        "one per line."
    )

    input_arg.add_argument(
        "-g",
        "--region",
        dest="regions",
        metavar="PATH",
        type=str,
        help="Region to apply LMI in format chrN:start-end_poi or file with the regions of interest. \
        If a file is provided, it must contain as a header 'coords', 'symbol' and 'id', and one region per line, \
        tab separated."
    )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.")

    optional_arg.add_argument(
        "-e",
        "--preserve",
        dest="rm_types",
        action="store_false",
        default=True,
        help="Preserve temporary .png image files that are used for making the composite figure (default: %(default)s)."
    )

    optional_arg.add_argument(
        "-C",
        "--clean-matrix",
        dest="clean_matrix",
        action="store_true",
        help="Flag to plot the 'clean' Hi-C matrix (the one METALoci uses to calculate the Kamada-Kawai layout) \
        instead of the 'original' Hi-C matrix (default: %(default)s)."
    )

    optional_arg.add_argument(
        "-q",
        "--quarts",
        dest="quart",
        default=[1, 3],
        metavar="INT",
        nargs="*",
        help="Space-separated list with the LMI quadrants to highlight (default: %(default)s). \
        1: High-high, top right (signal in bin is high, signal for neighbours is high). \
        2: Low-High, top left (signal in bin is low, signal for neighbours is high). \
        3: Low-Low, bottom left (signal in bin is low, signal for neighbours is low). \
        4: High-Low, bottom right (signal in bin is high, signal for neighbours is low).",
    )

    optional_arg.add_argument(
        "-v",
        "--pval",
        dest="signipval",
        default=0.05,
        metavar="FLOAT",
        type=float,
        help="P-value significance threshold (default: %(default)s).",
    )

    optional_arg.add_argument(
        "-z",
        "--zscore",
        dest="zscore_signal",
        action="store_true",
        help="Flag to use z-score transformed signal values for the scatter plot (default: %(default)s).",
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
        "-u",
        "--debug",
        dest="debug",
        action="store_true",
        help=SUPPRESS)
    
    style_arg = parser.add_argument_group(title="Style arguments")

    style_arg.add_argument(
        "-M",
        "--metalocis",
        dest="metalocis",
        action="store_true",
        help="Flag to select highlighting of the signal plots. If True, only the neighbouring bins from the point of \
        interest will be highlighted (independently of the quadrant and significance of those bins, but only if the \
        point of interest is significant and in a quadrant of interest). If False, all significant regions that \
        correspond to the quadrant selected with -q will be highlighted (default: False)."
    )

    style_arg.add_argument(
        "-k",
        "--mark_regions",
        dest="mark_regions",
        metavar="PATH",
        type=str,
        help="(experimental) Path to a file to mark certain regions on the gaudí plots. The file must have the " \
        "following columns (tab-separated): region_metaloci chr start end label. The label will be used to mark the " \
        "region on the plot."
    )

    style_arg.add_argument(
        "-n",
        "--neighbourhood",
        dest="neighbourhood_circle",
        action="store_true",
        help="Flag to plot the neighbourhood extension on the Kamada-Kawai and Gaudí plots. This is the influence " \
        "radius around the point of interest that will be considered for the local spatial autocorrelation of the " \
        "point of interest. (default: False)."
    )

    style_arg.add_argument(
        "-r",
        "--restraints",
        dest="restraints",
        action="store_false",
        help="Flag to plot the restraints on the Kamada-Kawai and Gaudí plots. (default: True)."
    )

def get_figures(row: pd.Series, args: pd.Series, progress=None, counter: int = None, silent: bool = True):
    """
    Function to generate the figures for a given region, given that there is a .mlo file associated with it.

    Parameters
    ----------
    row : pd.Series
        Row from the DataFrame containing the regions of interest.
    args : pd.Series
        Arguments parsed from the user input.
    progress : mp.Manager().dict
        Dictionary to keep track of the progress in multiprocessing.
    counter : int
        Counter for the region being processed.
    silent : bool
        Flag to enable or disable the print statements. Useful for multiprocessing.
    """

    if not silent:
        
        print(f"\n------> Working on region {row.coords} [{counter + 1}/{args.total_num}]\n")

    save_path = f"{args.work_dir}{row.coords.split(':', 1)[0]}/objects/{re.sub(':|-', '_', row.coords)}.mlo"

    try:

        with open(save_path, "rb") as mlobject_handler:

            state = pickle.load(mlobject_handler)
            mlobject = mlo.reconstruct(state)
            mlobject.save_path = save_path

    except FileNotFoundError:

        if not silent:
        
            print("\n\t.mlo file not found for this region.\n\tSkipping to the next one.")

        return

    if mlobject.lmi_info is None:

        if not silent:

            print("\n\tLMI not calculated for this region.\n\tSkipping to the next one...")

        return

    if mlobject.kk_distances is None:

        if not silent:

            print("\n\tKamada-Kawai layout not calculated for this region.\n\tSkipping to the next one...")

        return
    
    neighbourhood = mlobject.kk_distances.diagonal(1).mean() * args.INFLUENCE * args.BFACT

    for signal in args.signals:

        if len(signal) == 0:

            continue
        
        if not silent:
            
            print(f"\tPlotting signal: {signal}")

        r_value, p_value = plot.create_composite_figure(mlobject, signal, neighbourhood_circle=args.neighbourhood_circle,
                                    mark_regions=args.mark_regions, signipval=args.signipval, args=args, silent=silent)
        
        if not silent:

            print(f"\t\tFinal composite figure for region '{row.coords}' and signal '{signal}' -> done.")

        for signal_key, df in mlobject.lmi_info.items():

                if signal_key != signal:

                    continue

                sq = [0] * 4
                q = [0] * 4

                for i, lmi_row in df.iterrows():

                    q[lmi_row.moran_quadrant - 1] += 1

                    if lmi_row.LMI_pvalue <= args.signipval:

                        sq[lmi_row.moran_quadrant - 1] += 1

                q_string = "\t".join([f"{sq[i]}\t{q[i]}" for i in range(4)])

                with open(f"{args.work_dir}moran_info.txt", "a+", encoding="utf-8") as handler:

                    log = f"{row.coords}\t{row.name}\t{row.id}\t{signal_key}\t{r_value}\
                        \t{p_value}\t{q_string}\n"

                    handler.seek(0)

                    if os.stat(f"{args.work_dir}moran_info.txt").st_size == 0:

                        handler.write("region\tsymbol\tgene_id\tsignal\tr_value\tp_value\tsq1\tq1\tsq2\tq2\tsq3\tq3\
                                    \tsq4\tq4\n")

                    if not any(log in line for line in handler):

                        handler.write(log)

    if progress is not None:

        progress['value'] += 1
        time_spent = time() - progress['timer']
        time_remaining = int(time_spent / progress['value'] * (args.total_num - progress['value']))

        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t[{progress['value']}/{args.total_num}] | Time spent: {timedelta(seconds=round(time_spent))} | "
              f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')


def run(opts: list):
    """
    Function to run this section of METALoci with the needed arguments

    Parameters
    ----------
    opts : list
        List of arguments
    """
    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

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

    INFLUENCE = 1.5
    BFACT = 2

    # Functions assume this schema of colors if the user says nothing.
    # colors = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}

    # Parse list of signals to plot. Check if there is only one. Then check if it is a file, if so read it and store
    # the signals in a list. If not, store the signal in a list. If there are more than one signal, check if they are
    # files, if so exit. If not, store the signals in a list.
    if len(opts.signals) == 1:
        
        if os.path.isfile(opts.signals[0]) and os.access(opts.signals[0], os.R_OK):

                with open(opts.signals[0], "r", encoding="utf-8") as handler:

                    signals = [line.strip() for line in handler]
        
        else:
                
                signals = [opts.signals[0]]
        
    else:

        if os.path.isfile(opts.signals[0]) and os.path.isfile(opts.signals[1]):

            sys.exit("Please provide only one file with signals to plot.")
                
        signals = opts.signals

    if opts.mark_regions is not None:

        mark_regions = pd.read_table(opts.mark_regions, names=["region_metaloci", "chr", "start", "end", "mark"], sep="\t")

    else:
        
        mark_regions = None

    if opts.debug:

        print(f"work_dir ->\n\t{opts.work_dir}")
        print(f"regions ->\n\t{opts.regions}")
        print(f"signals ->\n\t{signals}")
        print(f"metaloci_only ->\n\t{opts.metalocis}")
        print(f"quadrants ->\n\t{opts.quart}")
        print(f"signipval ->\n\t{opts.signipval}")
        print(f"rmtypes ->\n\t{opts.rm_types}")
        print(f"mark_regions ->\n\t{mark_regions}")
        print(f"influence ->\n\t{INFLUENCE}")
        print(f"bfact ->\n\t{BFACT}")
        print(f"restraints ->\n\t{opts.restraints}")

        sys.exit()


    parsed_args = pd.Series({"work_dir": opts.work_dir,
                             "signals": signals,
                             "metaloci_only": opts.metalocis,
                             "quadrants":  [int(x) for x in opts.quart],
                             "signipval": opts.signipval,
                             "rm_types": opts.rm_types,
                             "mark_regions": mark_regions,
                             "zscore_signal": opts.zscore_signal,
                             "debug": opts.debug,
                             "total_num": len(df_regions),
                             "INFLUENCE": INFLUENCE,
                             "BFACT": BFACT,
                             "clean_matrix": opts.clean_matrix,
                             "neighbourhood_circle": opts.neighbourhood_circle,
                             "restraints": opts.restraints
                             })

    start_timer = time()

    if opts.multiprocess:

        print(f"\n------> {len(df_regions)} regions will be computed.\n")

        try:

            progress = mp.Manager().dict(value=0, timer=start_timer, done=False)

            with mp.Pool(processes=opts.threads) as pool:

                pool.starmap(get_figures, [(row, parsed_args, progress) for _, row in df_regions.iterrows()])

                if progress["done"]:

                    print("\tSome regions had already been computed and have been skipped.", end="")

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

        print("\n")

    else:

        for counter, row in df_regions.iterrows():

            get_figures(row, parsed_args, counter=counter, silent=False)            

    print(f"\nInformation saved to: '{os.path.join(opts.work_dir, 'moran_info.txt')}'")
    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    misc.create_version_log("figure", opts.work_dir)
    print("\nall done.")
