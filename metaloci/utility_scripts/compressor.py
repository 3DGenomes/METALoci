# Author: Leo Zuber
# Date: January 2025
"""
Utility for compressing and uncompressing METALoci working directories, as genome-wide runs can take a lot of space.\n
"""

HELP = """
Utility for compressing and uncompressing METALoci working directories.
"""

DESCRIPTION = """
Utility for compressing and uncompressing METALoci working directories, as a genome-wide run can take a lot of space.\n
"""

import os
import subprocess as sp
from argparse import ArgumentParser, HelpFormatter


def populate_args(parser):
    """
    Populate the ArgumentParser with the arguments needed for the METALoci caller.

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller.
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
    # Use a mutually exclusive group for -c and -d
    compression_group = input_arg.add_mutually_exclusive_group(required=True)

    compression_group.add_argument('-c',
                                   '--compress',
                                   dest="compress",
                                   action="store_true",
                                   help='Flag to compress the Hi-C file.'
                                   )

    compression_group.add_argument('-u',
                                   '--uncompress',
                                   dest="uncompress",
                                   action="store_true",
                                   help='Flag to uncompress the Hi-C file.'
                                   )
    
    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument("-h",
                              "--help",
                              action="help",
                              help="Show this help message and exit.")


def compress(work_dir: str):
    """
    Compress the working directory.

    Parameters
    ----------
    work_dir : str
        Path to the working directory.
    """
    base_name = os.path.basename(work_dir.rstrip('/'))

    # if there is a tar file in the work_dir, do not compress
    if os.path.exists(f"{work_dir}{base_name}.tar.gz"):

        print(f".tar.gz already found in working directory '{work_dir}'. Skipping compression.\n")
        return

    print(f"Compressing working directory {work_dir}")
    # Compress the working directory
    base_name = os.path.basename(work_dir.rstrip('/'))

    sp.run(f"tar -C {work_dir} -czf {os.path.join(work_dir, base_name)}.tar.gz .", shell=True)    
    # Delete all files int he work directory except the tar.gz file
    sp.run(f"find {work_dir} -mindepth 1 ! -name '*.tar.gz' -exec rm -rf {{}} +", shell=True)

    print(f"Compressed working directory to {work_dir}{base_name}.tar.gz")  


def uncompress(work_dir):
    """
    uncompress the working directory.

    Parameters
    ----------
    work_dir : str
        Path to the working directory.
    """
    base_name = os.path.basename(work_dir.rstrip('/'))

    # if there is no tar file in the work_dir, do not uncompress
    if not os.path.exists(f"{work_dir}{base_name}.tar.gz"):

        print(f".tar.gz not found in working directory '{work_dir}'. Skipping uncompression.\n")
        return

    print(f"uncompressing working directory {work_dir}")

    base_name = os.path.basename(work_dir.rstrip('/'))

    # uncompress the working directory
    sp.run(f"tar -C {work_dir} -xzf {work_dir}{base_name}.tar.gz", shell=True)
    # Delete the tar.gz file
    sp.run(f"rm {work_dir}{base_name}.tar.gz", shell=True)


def run(opts: list):
    """
    Main function for running compressor.

    Parameters
    ----------

    opts : list
        List of arguments.
    """
    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

    if opts.compress:

        compress(opts.work_dir)

    elif opts.uncompress:

        uncompress(opts.work_dir)

    print("all done.")
    
