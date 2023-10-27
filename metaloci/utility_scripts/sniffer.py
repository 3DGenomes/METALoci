import subprocess as sp
import pathlib
from argparse import SUPPRESS, HelpFormatter
import sys
from pybedtools import BedTool
import pandas as pd
from metaloci.misc import misc
from tqdm import tqdm

DESCRIPTION = """Takes a .gft file or a .bed file and parses it into a region list, with a
specific resolution and extension. The point of interest of the regions is the middle bin."""

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
        dest="extend",
        metavar="INT",
        required=True,
        type=int,
        help="How many bp the script should extend the region (upstream and downstream).",
    )

    optional_arg.add_argument(
        "-n",
        "--name",
        dest="name",
        metavar="STR",
        required=False,
        type=str,
        help="Name of the file.",
    )

    optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    optional_arg.add_argument("-u", "--debug", dest="debug", action="store_true", help=SUPPRESS)


def run(opts):

    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

    if opts.name is None:

        opts.name = opts.work_dir.split("/")[-2]

    work_dir = opts.work_dir
    chrom_sizes = opts.chrom_sizes
    gene_file = opts.gene_file
    resolution = opts.resolution
    extend = opts.extend
    name = opts.name
    debug = opts.debug

    if debug:

        print(f"work_dir ->\t{work_dir}")
        print(f"chrom_sizes ->\t{chrom_sizes}")
        print(f"gene_file ->\t{gene_file}")
        print(f"resolution ->\t{resolution}")
        print(f"extend ->\t{extend}")
        print(f"name ->\t\t{name}")

        sys.exit()
    
    file_type = gene_file.split(".")[-1]
    n_of_bins = int(extend / resolution)

    tmp_dir = f"{work_dir}tmp"
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    temp_fn = f"{tmp_dir}/{resolution}bp_bin_unsorted.bed"
    BedTool().window_maker(g=chrom_sizes, w=resolution).saveas(temp_fn)

    sort_com = f"sort {tmp_dir}/{resolution}bp_bin_unsorted.bed -k1,1V -k2,2n -k3,3n | " + \
        f"grep -v random | grep -v Un | grep -v alt > {tmp_dir}/{resolution}bp_bin.bed"
    sp.call(sort_com, shell=True)

    bin_genome = pd.read_table(f"{tmp_dir}/{resolution}bp_bin.bed", names=["chrom", "start", "end"])
    bin_genome[["chrom"]] = bin_genome[["chrom"]].astype(str)
    bin_genome[["start"]] = bin_genome[["start"]].astype(int)
    bin_genome[["end"]] = bin_genome[["end"]].astype(int)

    if file_type == "gtf":

        ID_CHROM, ID_TSS, ID_NAME, FN = misc.gtfparser(gene_file, name, extend, resolution)

    elif file_type == "bed":

        ID_CHROM, ID_TSS, ID_NAME, FN = misc.bedparser(gene_file, name, extend, resolution)

    print("Gathering information about bin index where the gene is located...")

    data = misc.binsearcher(ID_TSS, ID_CHROM, ID_NAME, bin_genome)

    print(f"A total of {data.shape[0]} entries will be written to {work_dir + FN}")

    with open(f"{work_dir}{FN}", mode="w", encoding="utf-8") as handler:

        handler.write("coords\tsymbol\tid\n")

        for _, row in tqdm(data.iterrows()):

            chrom_bin = bin_genome[(bin_genome["chrom"] == row.chrom)].reset_index()

            bin_start = max(0, (row.bin_index - n_of_bins))
            bin_end = min(chrom_bin.index.max(), (row.bin_index + n_of_bins + 1))
            region_bin = chrom_bin.iloc[bin_start:bin_end].reset_index()
            coords = f"{row.chrom}:{region_bin.iloc[0].start}-{region_bin.iloc[-1].end}"
            POI = region_bin[region_bin["level_0"] == row.bin_index].index.tolist()[0]

            handler.write(f"{coords}_{POI}\t{row.gene_name}\t{row.gene_id}\n")

    sp.check_call(f"rm -rf {tmp_dir}/{resolution}bp_bin.bed", shell=True)
    sp.check_call(f"rm -rf {tmp_dir}/{resolution}bp_bin_unsorted.bed", shell=True)

    print("done.")
