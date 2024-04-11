"""
Find the best combination of resolution, persistence length and cut-off for your HiC.\n\n

In order to test the results, please check the working directory supplied. In there you will find a folder for each
combination of parameters, and inside each folder you will find a plots for the layout and for the mixed-matrix.

NOTE: Best practice is to optimise resolution and persistent length first, then cutoff if needed.
"""
import sys
from argparse import SUPPRESS, HelpFormatter
from itertools import product
import multiprocessing as mp
from time import time
from datetime import timedelta
from tqdm import tqdm

from metaloci.misc import misc

HELP = """
Find the best combination of resolution, persistence length and cut-off for your HiC.
"""

DESCRIPTION = """
Find the best combination of resolution, persistence length and cut-off for your HiC.\n\n

In order to test the results, please check the working directory supplied. In there you will find a folder for each
combination of parameters, and inside each folder you will find a plots for the layout and for the mixed-matrix.\n\n

NOTE: Best practice is to optimise resolution and persistent length first, then cutoff if needed.
"""


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

    input_arg.add_argument('-w',
                           '--work-dir',
                           dest='work_dir',
                           metavar='PATH',
                           required=True,
                           type=str,
                           help='Path to working directory'
                           )

    input_arg.add_argument('-c',
                           '--hic',
                           dest='hic',
                           metavar='PATH',
                           required=True,
                           type=str,
                           help='Complete path to the cool/mcool/hic file',
                           )

    input_arg.add_argument('-r',
                           "--resolutions",
                           dest="resolutions",
                           metavar="INT",
                           required=True,
                           type=int,
                           nargs='+',
                           help="List of Hi-C resolutions to be tested (in bp)."
                           )

    input_arg.add_argument('-g',
                           '--region-files',
                           dest='region_files',
                           metavar='FILES',
                           required=True,
                           type=str,
                           nargs='+',
                           help="List of region files to be tested. Must be in the same order as --resolutions.",
                           )

    input_arg.add_argument('-l',
                           '--pls',
                           dest='pls',
                           metavar='INT',
                           required=True,
                           type=int,
                           nargs='+',
                           help='Persistence length; amount of possible rotation of points in layout (Normally 8 - 14)')

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument("-h",
                              "--help",
                              action="help",
                              help="Show this help message and exit.")

    optional_arg.add_argument('-s',
                              '--seed',
                              dest='seed',
                              metavar='INT',
                              type=int,
                              default=1,
                              help="Random seed for region sampling. (default: %(default)s)"
                              )

    optional_arg.add_argument('-n',
                              '--sample_num',
                              dest='sample_num',
                              metavar='int',
                              type=int,
                              default=5,
                              help="Numer of regions to sample from .txt file. (default: %(default)s)",
                              )

    optional_arg.add_argument('-o',
                              '--cutoffs',
                              dest='cutoffs',
                              metavar='FLOAT',
                              type=float,
                              nargs='+',
                              help="""Percent of top interactions to use from HiC.
                            METALoci Default = 0.2 (Normally 0.15 - 0.30)""")

    optional_arg.add_argument("--ncpus",
                              dest="ncpus",
                              metavar="INT",
                              default=int(mp.cpu_count() - 2),
                              type=int,
                              help="Number of CPUs to use in the multiprocessing step (default: %(default)s).",
                              )

    optional_arg.add_argument("-u",
                              "--debug",
                              dest="debug",
                              action="store_true",
                              help=SUPPRESS)


def run(opts: list):
    """
    Funtion to run this section of METALoci with the needed arguments

    Parameters
    ----------
    opts : list
        List of arguments
    """

    work_dir = opts.work_dir
    hic = opts.hic
    resolutions = opts.resolutions
    region_files = opts.region_files
    sample_num = opts.sample_num
    seed = opts.seed
    pls = opts.pls
    cutoffs = opts.cutoffs
    ncpus = opts.ncpus
    debug = opts.debug

    if cutoffs is None:
        cutoffs = [0.2]

    resolution_region_dict = {}

    for counter, reso in enumerate(resolutions):
        resolution_region_dict[reso] = region_files[counter]

    if debug:
        print(f"work_dir is:\n\t{work_dir}\n")
        print(f"hic is:\n\t{hic}\n")
        print(f"resolutions is:\n\t{resolutions}\n")
        print(f"region_files is:\n\t{region_files}\n")
        print(f"sample_num is:\n\t{sample_num}\n")
        print(f"seed is:\n\t{seed}\n")
        print(f"pls is:\n\t{pls}\n")
        print(f"cutoffs is:\n\t{cutoffs}\n")
        print(f"ncpus is:\n\t{ncpus}\n")
        print(f"resolution_region_dict is:\n\t{resolution_region_dict}")
        sys.exit(0)

    arg2do = [(work_dir, hic, f, resolution_region_dict[f], p, c, sample_num, seed)
              for f, p, c in product(resolution_region_dict, pls, cutoffs)]

    print("\nWelcome to MetaLoci parameter search. Let's start optimising!\n")

    start_timer = time()

    with mp.Pool(processes=ncpus) as pool:
        pool.starmap(misc.meta_param_search, arg2do)

    print(f"\n\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("All done! Check the plots in the working directory.")
