"""
Takes a .gft file or a .bed file and parses it into a region list, with a
specific resolution and extension. The point of interest of the regions is the middle bin.
"""
import os
import pathlib
import subprocess as sp
import sys
from argparse import SUPPRESS, HelpFormatter
from datetime import timedelta
from time import time

import pandas as pd
from metaloci.misc import misc
from pybedtools import BedTool
from tqdm import tqdm

HELP = "Converts a .gtf file to a list of regions for METALoci."

DESCRIPTION = """Takes a .gft file or a .bed file and parses it into a region list, with a
specific resolution and extension. The point of interest of the regions is the middle bin."""


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
        help="Path to the gene annotation file. GTF or bed files.",
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
        "--ucsc",
        dest="ucsc_bool",
        required=False,
        action="store_true",
        help="The gene file is in UCSC format.",
    )

    optional_arg.add_argument(
        "-u",
        "--debug",
        dest="debug",
        action="store_true",
        help=SUPPRESS)


def run(opts: list):
    """
    Funtion to run this section of METALoci with the needed arguments.

    Parameters
    ----------
    opts : list
        List of arguments.
    """

    work_dir = opts.work_dir
    chrom_sizes = opts.chrom_sizes
    gene_file = opts.gene_file
    resolution = opts.resolution
    extension = opts.extension
    name = opts.name
    ucsc_bool = opts.ucsc_bool
    debug = opts.debug

    if not work_dir.endswith("/"):

        work_dir += "/"

    if name is None:

        name = work_dir.split("/")[-2]

    if debug:

        debug_info = f"""
        Debug Information:
        ------------------
        work_dir: {work_dir}
        chrom_sizes: {chrom_sizes}
        gene_file: {gene_file}
        resolution: {resolution}
        extension: {extension}
        name: {name}
        """
        print(debug_info)

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

    bin_genome = pd.read_table(f"{temp_fn_sorted}", names=["chrom", "start", "end"],
                               dtype={"chrom": str, "start": int, "end": int})

    print("Parsing the gene annotation file...")

    if ucsc_bool:

        ID_CHROM, ID_TSS, ID_NAME, FN = misc.ucscparser(gene_file, name, extension, resolution)

    elif "gtf" in gene_file:

        ID_CHROM, ID_TSS, ID_NAME, FN = misc.gtfparser(gene_file, name, extension, resolution)

    elif "bed" in gene_file:

        ID_CHROM, ID_TSS, ID_NAME, FN = misc.bedparser(gene_file, name, extension, resolution)

    else:

        sys.exit("ERROR: The annotation file must be either a .gtf(.gz) or a .bed(.gz) file.")

    data = misc.binsearcher(ID_TSS, ID_CHROM, ID_NAME, bin_genome)

    print("Parsing the gene annotation file... Done.")
    print("Gathering information about bin index where the gene is located...")
    print(f"A total of {data.shape[0]} entries will be written to {os.path.join(work_dir, FN)}")

    bins_by_chrom = bin_genome.groupby("chrom")

    lines = ["coords\tsymbol\tid"]

    for row in tqdm(data.itertuples(), total=len(data)):

        chrom_bin = bins_by_chrom.get_group(row.chrom)

        bin_start = max(0, (row.bin_index - n_of_bins))
        bin_end = min(chrom_bin.index.max(), (row.bin_index + n_of_bins))

        region_bin = chrom_bin.loc[(chrom_bin.index >= bin_start) & (chrom_bin.index <= bin_end)].reset_index()

        coords = f"{row.chrom}:{region_bin.iloc[0].start}-{region_bin.iloc[-1].end}"

        try:

            POI = region_bin[region_bin["index"] == row.bin_index].index.tolist()[0]

        except IndexError:

            print(row, region_bin.tail(), f"bin_start: {bin_start}", f"bin_end: {bin_end}",
                  f"bin_start pos from gene: {(row.bin_index - n_of_bins)}",
                  f"bin_end pos from gene: {(row.bin_index + n_of_bins)}",
                  f"{chrom_bin.index.max()}", sep="\n")
            
            continue

        lines.append(f"{coords}_{POI}\t{row.gene_name}\t{row.gene_id}")

    with open(os.path.join(work_dir, FN), mode="w", encoding="utf-8") as handler:

        handler.write("\n".join(lines))

    print("Cleaning tmp files...")
    sp.check_call(f"rm -rf {tmp_dir}/{resolution}bp_bin.bed {tmp_dir}/{resolution}bp_bin_unsorted.bed", shell=True)
    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("All done.")
