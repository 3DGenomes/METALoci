"""
This script parses the LMI information files created by METALoci.

The output file will contain regions that pass the quadrant and p-value threshold for a given 
signal. In case it doesn't pass this filters, the script will output NA.
"""
import sys
import os
import pathlib
from time import time
from datetime import timedelta
from argparse import SUPPRESS, HelpFormatter
import multiprocessing as mp
import subprocess as sp
import pandas as pd

from tqdm import tqdm

from metaloci.misc import misc

HELP = """
Extract data about the POI of your regions.
"""

DESCRIPTION = """This script parses the LMI information files created by METALoci.

The output file will contain regions that pass the quadrant and p-value threshold for a given 
signal. In case it doesn't pass this filters, the script will output NA."""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller
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
        nargs="*",
        action="extend",
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
        "--ncpus",
        dest="ncpus",
        metavar="INT",
        default=int(mp.cpu_count() - 2),
        type=int,
        help="Number of cpus for the multiprocessing (default: %(default)s).",
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

    if opts.quadrant_list is None:

        opts.quadrant_list = [1, 3]

    work_dir = opts.work_dir
    out_dir = opts.out_dir
    gene_file = opts.gene_file
    signals = opts.signals
    quadrant_list = opts.quadrant_list
    pval = opts.pval
    region_file = opts.region_file
    outname = opts.outname
    ncpus = opts.ncpus
    debug = opts.debug

    if debug:

        print(f"work_dir is:\n\t{work_dir}")
        print(f"out_dir is:\n\t{out_dir}")
        print(f"gene_file is:\n\t{gene_file}")
        print(f"signals is:\n\t{signals}")
        print(f"quadrants is:\n\t{quadrant_list}")
        print(f"pval is:\n\t{pval}")
        print(f"region_file is:\n\t{region_file}")
        print(f"outname is:\n\t{outname}s")
        print(f"ncpus is:\n\t{ncpus}")

        sys.exit()

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_file_name = os.path.join(out_dir, f"{outname}.txt")
    bad_file_name = os.path.join(out_dir, f"{outname}_bad_files.txt")
    region_file_name = os.path.join(out_dir, f"{outname}_sig_genes.txt")

    start_timer = time()

    if len(signals) != 1 and region_file:

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

    for sig in signals:

        HEADER += f"\t{sig}_LMIq\t{sig}_LMIscore\t{sig}_LMIpval\t{sig}_signal\t{sig}_lag"

    out_file_handler = open(out_file_name, mode="w", encoding="utf-8")
    out_file_handler.write(f"{HEADER}\n")
    out_file_handler.flush()

    bad_file_handler = open(bad_file_name, mode="w", encoding="utf-8")
    bad_file_handler.write("coords\tsymbol\tid\treason\n")
    bad_file_handler.flush()

    if region_file:

        region_file_handler = open(region_file_name, mode="w", encoding="utf-8")
        region_file_handler.write(f"{HEADER}\n")
        region_file_handler.flush()

        args2do = [(line, signals, work_dir, bad_file_name, out_file_name, pval,
                    quadrant_list, region_file, region_file_name) for _, line in genes.iterrows()]

    else:

        args2do = [(line, signals, work_dir, bad_file_name, out_file_name, pval,
                    quadrant_list, region_file) for _, line in genes.iterrows()]

    with mp.Pool(processes=ncpus) as pool:
        pool.starmap(misc.get_poi_data, args2do)

    out_file_handler.close()

    if region_file:
        region_file_handler.close()

    bad_file_handler.close()

    print(f"Total time spent: {timedelta(seconds=round(time() - start_timer))}.")

    if os.path.exists(region_file_name) and int(sp.getoutput(f"wc -l {region_file_name} | cut -f 1 -d' '")) == 1:

        print("There were no significative regions using the following parameters:")
        print(f"\tquadrants: {','.join([str(i) for i in quadrant_list])}")
        print(f"\tp-value threshold: {pval}")
        os.remove(region_file_name)

    if os.path.exists(bad_file_name) and int(sp.getoutput(f"wc -l {bad_file_name} | cut -f 1 -d' '")) == 1:

        os.remove(bad_file_name)

    elif os.path.exists(bad_file_name) and int(sp.getoutput(f"wc -l {bad_file_name} | cut -f 1 -d' '")) > 1:
        print(f"Please, check {bad_file_name}. Some regions might be problematic.")
