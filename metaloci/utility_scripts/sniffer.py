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
from pybedtools import BedTool
from tqdm import tqdm

from metaloci.misc import misc

HELP = "Converts a .gtf/.bed file to a list of regions for METALoci."

DESCRIPTION = """Takes a .gft file or a .bed file and parses it into a region list, with a
specific resolution and extension. Each gene will be a point of interest and the region will be centered around it. 
Human/mouse gtf files can be downloaded from the GENCODE website. For other species, please
refer to the UCSC website. BED files can be used to create a custom region list, using the following format:
chromosome, start, end, gene_symbol, gene_id. Strandness can be added to the bed file by adding a 6th column; if not,
the script will consider the gene to be on the positive strand."""


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
        help="Path to working directory.",
    )

    input_arg.add_argument(
        "-s",
        "--chrom-sizes",
        dest="chrom_sizes",
        metavar="PATH",
        type=str,
        required=True,
        help="Full path to a file that contains the name of the chromosomes in the first column and the ending " \
        "coordinate of the chromosome in the second column. This can be found in UCSC Genome Browser website for your " \
        "species."
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
        help="Resolution at which to split the genome, in bp. This should be the resolution you have in your Hi-C data, but " \
        "it will not be checked.",
    )

    input_arg.add_argument(
        "-wi",
        "--window",
        dest="window",
        metavar="INT",
        required=True,
        type=int,
        help="Size of the regions in bp. The region will be centered around the point of interest. Regions close " \
        "to the telomeres will be smaller, but still centered around the point of interest." 
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
        help="Flag if the gene file you are using is in UCSC format.",
    )

    optional_arg.add_argument(
        "--strand",
        dest="strand",
        required=False,
        action="store_true",
        help="The file has strand information in the last column. ONLY FOR BED FILES.",
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
    extension = int(opts.window / 2)
    name = opts.name
    ucsc_bool = opts.ucsc_bool
    strand = opts.strand
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
        window: {extension}
        name: {name}
        ucsc_bool: {ucsc_bool}
        strand: {strand}
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

        ID_CHROM, ID_TSS, ID_NAME, file_name = misc.ucscparser(gene_file, name, extension, resolution)

    elif "gtf" in gene_file:

        ID_CHROM, ID_TSS, ID_NAME, file_name = misc.gtfparser(gene_file, name, extension, resolution)

    elif "bed" in gene_file:

        # get the first line of gene_file and check if it has 5 or 6 columns
        with open(gene_file, mode="r", encoding="utf-8") as handler:

            first_line = handler.readline().strip().split("\t")

        if strand and len(first_line) == 5:

            sys.exit("ERROR: The bed file must have 6 columns to include strand information.")

        elif not strand and len(first_line) == 6:

            sys.exit("ERROR: The bed file must have 5 columns to exclude strand information.")

        ID_CHROM, ID_TSS, ID_NAME, file_name = misc.bedparser(gene_file, name, extension, resolution, strand)

    else:

        sys.exit("ERROR: The annotation file must be either a .gtf(.gz) or a .bed(.gz) file.")

    data = misc.binsearcher(ID_TSS, ID_CHROM, ID_NAME, bin_genome)
    bins_by_chrom = bin_genome.groupby("chrom")
    lines = ["coords\tsymbol\tid"]

    print("Parsing the gene annotation file... Done.")
    print("Gathering information about bin index where the gene is located...")
    print(f"A total of {data.shape[0]} entries will be written to {os.path.join(work_dir, file_name)}")
    
    for row in tqdm(data.itertuples(), total=len(data)):

        chrom_bin = bins_by_chrom.get_group(row.chrom)
        # Filter out incomplete bins (bins smaller than resolution)
        chrom_bin = chrom_bin[chrom_bin['end'] - chrom_bin['start'] == resolution].copy()

        if len(chrom_bin) == 0:

            print(f"Warning: No complete bins found for chromosome {row.chrom}")

            continue
            
        chrom_max_index = chrom_bin.index.max()
        # Calculate desired window boundaries
        desired_start = row.bin_index - n_of_bins
        desired_end = row.bin_index + n_of_bins
        
        # Adjust boundaries to maintain full window size when near chromosome ends
        if desired_start < 0:

            # Near chromosome start: shift window right
            bin_start = chrom_bin.index.min()
            bin_end = min(chrom_max_index, bin_start + 2 * n_of_bins)

        elif desired_end > chrom_max_index:

            # Near chromosome end: shift window left
            bin_end = chrom_max_index
            bin_start = max(chrom_bin.index.min(), bin_end - 2 * n_of_bins)

        else:
            
            # Gene is in middle of chromosome: use centered window
            bin_start = desired_start
            bin_end = desired_end
        
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

    with open(os.path.join(work_dir, file_name), mode="w", encoding="utf-8") as handler:

        handler.write("\n".join(lines))

    print("Cleaning tmp files...")
    sp.check_call(f"rm -rf {tmp_dir}", shell=True)
    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    misc.create_version_log("sniffer", opts.work_dir)
    print("All done.")
