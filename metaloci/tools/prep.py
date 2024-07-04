"""
Processes signal .bed files or .bedGraph files, binnarizing them at a given resolution, merging all signals in the 
same dataframe and subsetting by chromosomes.
"""

import os
import pathlib
import subprocess as sp
import sys
import warnings
from argparse import SUPPRESS, HelpFormatter
from datetime import timedelta
from time import time

import cooler
import h5py
import hicstraw
import pandas as pd
from metaloci.misc import misc
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from pybedtools import BedTool
from tqdm import tqdm

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HELP = "Processes signal .bed files to METALoci format."

DESCRIPTION = """
Processes signal .bed files or .bedGraph files, binnarizing them at a given resolution, merging all signals in the 
same dataframe and subsetting by chromosomes.
"""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the prep step.

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
        required=True,
        metavar="PATH",
        type=str,
        help="Path to a working directory."
    )

    input_arg.add_argument(
        "-c",
        "--hic",
        dest="hic_file",
        required=True,
        metavar="PATH",
        type=str,
        help="Complete path to the cool/mcool/hic file."
    )

    input_arg.add_argument(
        "-d",
        "--data",
        dest="data",
        required=True,
        metavar="PATH",
        type=str,
        nargs="*",
        action="extend",
        help="Path to file to process. The file must contain titles for the columns, being the first 3 columns coded "
        "as chrom, start, end. The following columns should contain the name of the signal. "
        "Names of the chromosomes must be the same as the coords file described below. "
    )

    input_arg.add_argument(
        "-r",
        "--resolution",
        dest="reso",
        required=True,
        metavar="INT",
        type=int,
        help="Resolution of the bins, to binnarize the signal (in bp)."
    )

    input_arg.add_argument(
        "-s",
        "--coords",
        dest="coords",
        required=True,
        metavar="PATH",
        type=str,
        help="Full path to a file that contains the name of the chromosomes in the "
        "first column and the ending coordinate of the chromosome in the "
        "second column. This can be found in UCSC Genome Browser website for your species of interest."
    )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.")

    optional_arg.add_argument(
        "-t",
        "--summarize_type",
        dest="sum_type",
        metavar="STR",
        type=str,
        default="median",
        choices=["median", "mean", "min", "max", "count"],
        help="Type of summarization to use. Options: %(choices)s. Default: %(default)s.")

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
    data = opts.data
    hic_path = opts.hic_file
    coords = opts.coords
    resolution = opts.reso
    sum_type = opts.sum_type

    if not work_dir.endswith("/"):

        work_dir += "/"

    # Debug 'menu' for testing purposes
    if opts.debug:

        print(f"work_dir ->\n\t{work_dir}")
        print(f"data ->\n\t{data}")
        print(f"hic_path ->\n\t{hic_path}")
        print(f"coords ->\n\t{coords}")
        print(f"resolution ->\n\t{resolution}")
        print(f"sum_type ->\n\t{sum_type}")

        sys.exit()

    start_timer = time()
    tmp_dir = os.path.join(work_dir, "tmp")

    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    print("Checking input files compatibility...", end="\r")

    # Checking if the resolution supplied by the user is within the resolutions of the cool/mcool/hic files
    if hic_path.endswith(".cool"):

        available_resolutions = cooler.Cooler(hic_path).binsize

        if resolution != available_resolutions:

            sys.exit("The given resolution is not the same as the provided cooler file. Exiting...")

        hic_chroms = misc.check_names(hic_path, data, coords)

    if hic_path.endswith(".mcool"):

        available_resolutions = [int(x) for x in list(h5py.File(hic_path)["resolutions"].keys())]

        if resolution not in available_resolutions:

            print(
                f"The given resolution is not in the provided mcooler file.\nThe available resolutions are: \
                {', '.join(misc.natural_sort([str(x) for x in available_resolutions]))}"
            )
            sys.exit("Exiting...")

        hic_chroms = misc.check_names(hic_path, data, coords, resolution)

    elif hic_path.endswith(".hic"):

        available_resolutions = hicstraw.HiCFile(hic_path).getResolutions()

        if resolution not in available_resolutions:

            print(
                f"The given resolution is not in the provided mcooler file.\nThe available resolutions are: \
                {', '.join(misc.natural_sort([str(x) for x in available_resolutions]))}"
            )
            sys.exit("Exiting...")

        hic_chroms = misc.check_names(hic_path, data, coords)

    else:

        print("Hi-C file format not supported. Supported formats are: cool, mcool, hic.")
        sys.exit("Exiting...")

    print("Checking input files compatibility... OK.")

    # Create a bed file that contains regions of a given resolution and sort it.
    bin_genome_fn = os.path.join(tmp_dir, f"{resolution}bp_bin" + "{}.bed")

    BedTool().window_maker(g=coords, w=resolution).saveas(bin_genome_fn.format('_unsorted'))

    sp.call(
        f"sort {bin_genome_fn.format('_unsorted')} -k1,1V -k2,2n -k3,3n > {bin_genome_fn.format('')}",
        shell=True
    )
    os.remove(bin_genome_fn.format("_unsorted"))

    last_header = None
    column_dict = {}

    # Check if the given bed file has a header and sort the file. If there is a header, save column
    # names to a dictionary to put it in the next step, then sort the file.
    for data_counter, data_fn in enumerate(data):

        print(f"\nProcessing signal file: {data_fn} [{data_counter + 1}/{len(data)}]\n")
        print("\tSorting signal...", end="\r")

        signal_file_name = os.path.basename(data_fn)
        tmp_signal_path = os.path.join(tmp_dir, "{}_" + f"{signal_file_name}")

        sp.call(f"cp {data_fn} {tmp_signal_path.format('tmp')}", shell=True)

        if sp.getoutput(f"head -n 1 {data_fn} | cut -f 1") != "chrom":

            sp.call(
                f"sort {tmp_signal_path.format('tmp')} -k1,1V -k2,2n -k3,3n > {tmp_signal_path.format('sorted')}",
                shell=True
            )

            header = False

        else:

            # Saving the corresponding column names in a dict, with only the file name as a key.
            column_dict[signal_file_name] = sp.getoutput(f"head -n 1 {data_fn}").split(sep="\t")

            sp.call(
                f"tail -n +1 {tmp_signal_path.format('tmp')} | sort -k1,1V -k2,2n -k3,3n > \
                {tmp_signal_path.format('sorted')}",
                shell=True
            )

            header = True

        if last_header is not None:

            if header != last_header:

                misc.remove_folder(pathlib.Path(tmp_dir))
                sys.exit("ERROR: Some of the files to be processed have a header but others do not. "
                         "Please, modify the files to keep them consistent.")

        last_header = header
        # Retrieve chromosomes from the data file.
        chroms = sp.getoutput(f"tail -n +2 {data_fn} | cut -f 1 | sort -u").split(sep="\n")
        # Keep only chromosomes that are in the Hi-C file.
        chroms = misc.natural_sort([chrom for chrom in chroms if chrom in hic_chroms])

        print("\tSorting signal... done.")
        print(f"\tIntersecting signal with {resolution}bp bins...")

        # Perhaps use a \t?
        pbar = tqdm(chroms, desc=f"        {chroms[0]}")
        chrom_bed_path = os.path.join(tmp_dir, "{}" + f'_{resolution}bp_bin.bed')

        for i, chrom in enumerate(pbar):

            # Do a subset of the signal
            awk_com = f"""awk '{{if($1=="{chrom}") {{print}}}}' {tmp_signal_path.format('sorted')} > """ + \
                f"{tmp_signal_path.format(f'{chrom}_sorted')}"

            sp.call(awk_com, shell=True)

            # Do a subset of the BedTools makewindows file
            awk_com = f"""awk '{{if($1=="{chrom}") {{print}}}}' {bin_genome_fn.format('')} > """ + \
                f"{chrom_bed_path.format(chrom)}"

            sp.call(awk_com, shell=True)

            # Do an intersection of the signal and the bed file, per each chromsome.
            BedTool(
                chrom_bed_path.format(chrom)).intersect(
                BedTool(tmp_signal_path.format(f'{chrom}_sorted')),
                wao=True, sorted=True).saveas(tmp_signal_path.format(f'{chrom}_intersected'))

            os.remove(chrom_bed_path.format(chrom))
            os.remove(tmp_signal_path.format(f'{chrom}_sorted'))

            concatenated_bed = os.path.join(tmp_dir, '{}concatenated.bed')

            # Concatenate all the intersected files.
            if chrom == chroms[0]:

                sp.call(
                    f"cp {tmp_signal_path.format(f'{chrom}_intersected')} {concatenated_bed.format('')}",
                    shell=True
                )

            else:

                sp.call(
                    f"cat {concatenated_bed.format('')} {tmp_signal_path.format(f'{chrom}_intersected')} > \
                    {concatenated_bed.format('tmp_')}",
                    shell=True
                )
                sp.call(f"mv {concatenated_bed.format('tmp_')} {concatenated_bed.format('')}", shell=True)

            os.remove(tmp_signal_path.format(f'{chrom}_intersected'))
            pbar.set_description(f"        {chroms[i+1] if i+1 < len(chroms) else 'all done! ('}")

        print("\tAssigning signal to bins...", end="\r")

        n_of_col = sp.getoutput(f"head -n 1 {concatenated_bed.format('')}" + " | awk '{print NF}'")
        awk_com = (
            f"awk -F \'\\t\' -v OFS=\'\\t\' '{{if (${n_of_col} == 0) {{$4 = $1; $5 = $2; $6 = $3}} \
            {{for (i = 7; i <= {n_of_col}; i++) {{$i = $i * (${n_of_col} / ($6 - $5))}}}} print}}' \
             {concatenated_bed.format('')} > {tmp_dir}/intersected_ok_{signal_file_name}"
        )

        sp.call(awk_com, shell=True)
        print("\tAssigning signal to bins... done.")

    print("\nConcatenating signals in one file...", end="\r")

    # Create a list of paths to the intersected files.
    intersected_files_paths = [os.path.join(tmp_dir, f"intersected_ok_{os.path.basename(f)}") for f in data]
    # Read the first intersected file,
    final_intersect = pd.read_csv(intersected_files_paths[0], sep="\t", header=None, low_memory=False)
    # Drop unnecesary columns,
    final_intersect = final_intersect.drop([3, 4, 5, final_intersect.columns[-1]], axis=1)

    if header:

        # Input the corresponding column names
        final_intersect.columns = column_dict[
            intersected_files_paths[0].rsplit("/", 1)[1].split("_", 2)[2]
        ]

    else:

        final_intersect.columns = [
            "chrom",
            "start",
            "end",
            f"{os.path.basename(intersected_files_paths[0].rsplit('.', 1)[0]).split('intersected_ok_', 1)[1]}"
        ]

    # Calculate the summary of all signals of the same bin.
    final_intersect = misc.signal_binnarize(final_intersect, sum_type)

    # Process the rest of the files the same way and merge with the previous one, until all files are merged.
    if len(intersected_files_paths) > 1:

        for i in range(1, len(intersected_files_paths)):

            tmp_intersect = pd.read_csv(intersected_files_paths[i], sep="\t", header=None, low_memory=False)
            tmp_intersect = tmp_intersect.drop([3, 4, 5, tmp_intersect.columns[-1]], axis=1)

            if header:

                # Input the corresponding column names
                tmp_intersect.columns = column_dict[
                    intersected_files_paths[i].rsplit("/", 1)[1].split("_", 2)[2]
                ]

            else:

                tmp_intersect.columns = [
                    "chrom",
                    "start",
                    "end",
                    f"{os.path.basename(intersected_files_paths[i].rsplit('.', 1)[0]).split('intersected_ok_', 1)[1]}"
                ]

            # Calculate the summary of all signals of the same bin.
            tmp_intersect = misc.signal_binnarize(tmp_intersect, sum_type)
            final_intersect = pd.merge(final_intersect, tmp_intersect,
                                       on=["chrom", "start", "end"], how="outer").fillna(0)

    # For each chromosome, create a directory and save the information for
    # that chromosome in .csv and .pkl.
    for chrom in sorted(final_intersect["chrom"].unique()):

        pathlib.Path(os.path.join(work_dir, "signal", chrom)).mkdir(parents=True, exist_ok=True)
        final_intersect[final_intersect.chrom == f"{chrom}"].to_csv(
            f"{work_dir}signal/{chrom}/{chrom}_signal.tsv",
            sep="\t", index=False
        )

    print("Concatenating signals in one file... done.")

    chroms_no_signal = [chrom for chrom in hic_chroms if chrom not in final_intersect["chrom"].unique()]

    if len(chroms_no_signal) == 1:

        print(f"Omited chromosome {chroms_no_signal[0]} as it did not have any signal.")

    elif len(chroms_no_signal) > 1:

        print(f"Omited chromosomes {', '.join(chroms_no_signal)} as they did not have any signal.")

    print(f"\nSignal bed files saved to {work_dir}signal/")
    misc.remove_folder(pathlib.Path(tmp_dir))
    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nAll done.")
