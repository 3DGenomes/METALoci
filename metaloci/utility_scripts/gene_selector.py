"""
This script parses the LMI information files created by METALoci.

The output file will contain regions that pass the quadrant and p-value threshold for a given
signal. In case it doesn't pass this filters, the script will output NA.
"""
import multiprocessing as mp
import os
import pathlib
import sys
from argparse import SUPPRESS, HelpFormatter
from datetime import timedelta
from time import time

import pandas as pd
from metaloci.misc import misc
from tqdm import tqdm

HELP = "Extracts data about the point of interest of your regions."

DESCRIPTION = """This script parses the LMI information files created by METALoci.

The output file will contain regions that pass the quadrant and p-value threshold for a given
signal. In case it doesn't pass this filters, the script will output NA."""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step.

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller.
    """

    parser.formatter_class = lambda prog: HelpFormatter(prog, width=120,
                                                        max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")

    input_arg.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the working directory where LMI data is stored.",
    )

    input_arg.add_argument(
        "-o",
        "--output-dir",
        dest="out_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the directory where LMI data of the POI for the regions will be stored.",
    )

    input_arg.add_argument(
        "-g",
        "--gene-file",
        dest="gene_file",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the region file from where to search the POI.",
    )

    input_arg.add_argument(
        "-s",
        "--signals",
        dest="signals",
        metavar="STR",
        type=str,
        # nargs="*",
        # action="extend",
        required=True,
        help="Space separated list of signal names to use.",
    )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )

    optional_arg.add_argument(
        "-q",
        "--quadrants",
        dest="quadrant_list",
        metavar="INT",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        help="List of quadrant to select. Default: 1, 3. Choices: %(choices)s."
    )

    optional_arg.add_argument(
        "-p",
        "--pval",
        dest="pval",
        default=0.05,
        metavar="FLOAT",
        type=float,
        help="P-value significance threshold (default: %(default).2f).",
    )

    optional_arg.add_argument(
        "-r",
        "--region_file",
        action="store_true",
        help="Select wheter or not to store a metaloci region file with the significant regions.",
    )

    optional_arg.add_argument(
        "--name",
        dest="outname",
        metavar="STR",
        default="gene_selector_table",
        type=str,
        help="Name of the file with the selected regions/genes (default: %(default)s).",
    )

    optional_arg.add_argument(
        "-t",
        "--threads",
        dest="threads",
        metavar="INT",
        default=int(mp.cpu_count() - 2),
        type=int,
        help="Number of threads for the multiprocessing (default: %(default)s).",
    )

    optional_arg.add_argument(
        "-u",
        "--debug",
        dest="debug",
        action="store_true",
        help=SUPPRESS
    )


def run(opts: list):
    """
    Funtion to run this section of METALoci with the needed arguments

    Parameters
    ----------
    opts : list
        List of arguments
    """

    work_dir = opts.work_dir
    out_dir = opts.out_dir
    gene_file = opts.gene_file
    signals = opts.signals
    quadrant_list = opts.quadrant_list if opts.quadrant_list else [1, 3]
    pval = opts.pval
    region_file = opts.region_file
    outname = opts.outname
    threads = opts.threads
    debug = opts.debug

    if debug:

        debug_info = f"""
        Debug Information:
        ------------------
        work_dir: {work_dir}
        out_dir: {out_dir}
        gene_file: {gene_file}
        signals: {signals}
        quadrants: {quadrant_list}
        pval: {pval}
        region_file: {region_file}
        outname: {outname}
        ncpus: {threads}
        """
        print(debug_info)

        sys.exit()

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_file_name = os.path.join(out_dir, f"{outname}.txt")
    bad_file_name = os.path.join(out_dir, f"{outname}_bad_files.txt")
    region_file_name = os.path.join(out_dir, f"{outname}_sig_genes.txt")

    start_timer = time()

    # check if signals is a file. if it is read it and store it in a list

    if os.path.isfile(signals):

        with open(signals, mode="r", encoding="utf-8") as f:

            signal_list = [line.strip() for line in f]

    elif isinstance(signals, str):

        signal_list = signals.split()

    else:

        sys.exit("--signals must be a string or a file with the signals to use.")

    if len(signal_list) != 1 and region_file:

        print("Region file will not be computed because more than one signal was selected.")

        region_file = False

    try:
        
        genes = pd.read_table(gene_file)

    except FileNotFoundError:

        print(f"File {gene_file} not found.")
        sys.exit()

    except IsADirectoryError:

        print(f"File {gene_file} is a directory.")
        sys.exit()

    print(f"A total of {genes.shape[0]} genes will be parsed.")

    HEADER = "coords\tsymbol\tid"

    header_parts = [f"\t{sig}_LMIq\t{sig}_LMIscore\t{sig}_LMIpval\t{sig}_signal\t{sig}_lag" for sig in signal_list]

    HEADER += ''.join(header_parts)

    with open(out_file_name, mode="w", encoding="utf-8") as out_file_handler:

        out_file_handler.write(f"{HEADER}\n")

    with open(bad_file_name, mode="w", encoding="utf-8") as bad_file_handler:

        bad_file_handler.write("coords\tsymbol\tid\treason\n")

    if region_file:

        with open(region_file_name, mode="w", encoding="utf-8") as region_file_handler:

            region_file_handler.write(f"{HEADER}\n")

        args = [(line, signal_list, work_dir, bad_file_name, out_file_name, pval,
                    quadrant_list, region_file, region_file_name) for line in genes.itertuples(index=False)]

    else:

        args = [(line, signal_list, work_dir, bad_file_name, out_file_name, pval,
                    quadrant_list) for _, line in genes.iterrows()]

    try:
        
        # This piece of code works, but not progress bar
        # with mp.Pool(processes=ncpus) as pool:
        #     pool.starmap(misc.get_poi_data, args2do)
        # pool.close()
        # pool.join()

        # This piece of code works, has a progress bar, but the way it "counts" is weird (it counts the start of
        # the process, not the end of it)
        pbar = tqdm(total=len(args))

        def update(*a):

            pbar.update()

        with mp.Pool(processes=threads) as pool:

            for i in range(pbar.total):

                pool.apply_async(misc.get_poi_data, args=(args[i]), callback=update)

        pool.close()
        pool.join()

    except KeyboardInterrupt:

        pool.terminate()
        pool.join()
        print("\nKeyboard interrupt detected. Exiting...")

    if os.path.exists(region_file_name) and misc.has_exactly_one_line(region_file_name):

        print("There were no significative regions using the following parameters:")
        print(f"\tquadrants: {','.join([str(i) for i in quadrant_list])}")
        print(f"\tp-value threshold: {pval}")
        os.remove(region_file_name)

    if os.path.exists(bad_file_name) and misc.has_exactly_one_line(bad_file_name):

        os.remove(bad_file_name)

    elif os.path.exists(bad_file_name) and not misc.has_exactly_one_line(bad_file_name):

        print(f"Please, check {bad_file_name}. Some regions might be problematic.")

    print(f"Total time spent: {timedelta(seconds=round(time() - start_timer))}.")
