"""
Takes a .gft file or a .bed file and parses it into a region list, with a
specific resolution and extension. The point of interest of the regions is the middle bin.
"""
import sys
import pathlib
import os
import subprocess as sp
from argparse import SUPPRESS, HelpFormatter
from datetime import timedelta
from time import time

import pandas as pd
from pybedtools import BedTool
from tqdm import tqdm

from metaloci.misc import misc

DESCRIPTION = """Takes a .gft file or a .bed file and parses it into a region list, with a
specific resolution and extension. The point of interest of the regions is the middle bin."""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller
    """

    parser.formatter_class=lambda prog: HelpFormatter(prog, width=120,
                                                      max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")

    input_arg.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the working directory where data will be stored.",
    )

    input_arg.add_argument(
        "-s",
        "--chrom-sizes",
        dest="chrom_sizes",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the chrom sizes file (where the script should stop).",
    )

    input_arg.add_argument(
        "-g",
        "--gene-file",
        dest="gene_file",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the gene annotation file. Uncompressed GTF or bed files.",
    )

    input_arg.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        metavar="INT",
        required=True,
        type=int,
        help="Resolution at which to split the genome.",
    )

    input_arg.add_argument(
        "-e",
        "--extend",
        dest="extension",
        metavar="INT",
        required=True,
        type=int,
        help="How many bp the script should extend the region (upstream and downstream).",
    )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.")

    optional_arg.add_argument(
        "-n",
        "--name",
        dest="name",
        metavar="STR",
        required=False,
        type=str,
        help="Name of the file.",
    )

    optional_arg.add_argument(
        "-u",
        "--debug",
        dest="debug",
        action="store_true",
        help=SUPPRESS)


def run(opts : list):
    """
    Funtion to run this section of METALoci with the needed arguments

    Parameters
    ----------
    opts : list
        List of arguments
    """

    work_dir = opts.work_dir
    chrom_sizes = opts.chrom_sizes
    gene_file = opts.gene_file
    resolution = opts.resolution
    extension = opts.extension
    name = opts.name
    debug = opts.debug

    if not work_dir.endswith("/"):

        work_dir += "/"

    if name is None:

        name = work_dir.split("/")[-2]

    if debug:

        print(f"work_dir ->\n\t{work_dir}")
        print(f"chrom_sizes ->\n\t{chrom_sizes}")
        print(f"gene_file ->\n\t{gene_file}")
        print(f"resolution ->\n\t{resolution}")
        print(f"extension ->\n\t{extension}")
        print(f"name ->\n\t{name}")

        sys.exit()

    start_timer = time()

    n_of_bins = int(extension / resolution)
    tmp_dir = os.path.join(work_dir, "tmp")

    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    temp_fn_unsorted = os.path.join(tmp_dir, f"{resolution}bp_bin_unsorted.bed")
    temp_fn_sorted = os.path.join(tmp_dir, f"{resolution}bp_bin.bed")

    BedTool().window_maker(g=chrom_sizes, w=resolution).saveas(temp_fn_unsorted)

    sort_com = f"sort {temp_fn_unsorted} -k1,1V -k2,2n -k3,3n | " + \
        f"grep -v random | grep -v Un | grep -v alt > {temp_fn_sorted}"

    sp.call(sort_com, shell=True)

    bin_genome = pd.read_table(f"{temp_fn_sorted}", names=["chrom", "start", "end"])

    bin_genome[["chrom"]] = bin_genome[["chrom"]].astype(str)
    bin_genome[["start"]] = bin_genome[["start"]].astype(int)
    bin_genome[["end"]] = bin_genome[["end"]].astype(int)

    if "gtf" in gene_file:

        ID_CHROM, ID_TSS, ID_NAME, FN = misc.gtfparser(gene_file, name, extension, resolution)

    elif "bed" in gene_file:

        ID_CHROM, ID_TSS, ID_NAME, FN = misc.bedparser(gene_file, name, extension, resolution)

    else:

        sys.exit("ERROR: The annotation file must be either a .gtf(.gz) or a .bed(.gz) file.")

    data = misc.binsearcher(ID_TSS, ID_CHROM, ID_NAME, bin_genome)

    print("Gathering information about bin index where the gene is located...")
    print(f"A total of {data.shape[0]} entries will be written to {os.path.join(work_dir, FN)}")

    with open(os.path.join(work_dir, FN), mode="w", encoding="utf-8") as handler:

        handler.write("coords\tsymbol\tid\n")

        for counter, row in tqdm(data.iterrows()):

            chrom_bin = bin_genome[(bin_genome["chrom"] == row.chrom)].reset_index()

            bin_start = max(0, (row.bin_index - n_of_bins))
            bin_end = min(chrom_bin.index.max(), (row.bin_index + n_of_bins))

            region_bin = chrom_bin.iloc[bin_start:bin_end + 1].reset_index()

            coords = f"{row.chrom}:{region_bin.iloc[0].start}-{region_bin.iloc[-1].end}"

            try:
                POI = region_bin[region_bin["level_0"] == row.bin_index].index.tolist()[0]
            except IndexError:
                print()
                print(row)
                print(region_bin.tail())

                print(f"bin_start: {bin_start}")
                print(f"bin_end: {bin_end}")
                print(f"bin_start pos from gene: {(row.bin_index - n_of_bins)}")
                print(f"bin_end pos from gene: {(row.bin_index + n_of_bins)}")
                print(f"{chrom_bin.index.max()}")

                continue

            handler.write(f"{coords}_{POI}\t{row.gene_name}\t{row.gene_id}\n")

            ## Print of the status of writting the file, so the user has a bit of feedback
            if counter % 1000 == 0 and counter != 0:
                print(f"\t{counter} entries written out of {data.shape[0]}", end = "\r")

    print("Cleaning temporary files...")
    sp.check_call(f"rm -rf {tmp_dir}/{resolution}bp_bin.bed", shell=True)
    sp.check_call(f"rm -rf {tmp_dir}/{resolution}bp_bin_unsorted.bed", shell=True)

    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("All done.")
