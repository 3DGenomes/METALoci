"""
This script processes signal .bed files or .bedGraph files, binnarizing them at a given resolution,
merging all signals in the same dataframe and subsetting by chromosomes.
"""

import pathlib
import subprocess as sp
import warnings
from argparse import HelpFormatter

import h5py
import hicstraw
import pandas as pd
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
from pybedtools import BedTool

from metaloci.misc import misc

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DESCRIPTION = """
Takes a bed file (UCSC format) of signals and parses it into the format needed for METALoci to work.
"""

def populate_args(parser):

    parser.formatter_class=lambda prog: HelpFormatter(prog, width=120,
                                                      max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")
    optional_arg = parser.add_argument_group(title="Optional arguments")


    input_arg.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        required=True,
        metavar="PATH",
        type=str,
        help="Path to a working directory.",
    )

    input_arg.add_argument(
        "-c",
        "--hic",
        dest="hic_file",
        metavar="PATH",
        type=str,
        required=True,
        help="Complete path to the cooler file.",
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
        help="Path to file to process. The file must contain titles for the columns,"
        " being the first 3 columns coded as chrom, start, end. The following "
        "columns should contain the name of the signal coded as: id1_id2.\n"
        "Names of the chromosomes must be the same as the coords_file "
        "described below. ",
    )

    input_arg.add_argument(
        "-r",
        "--resolution",
        dest="reso",
        required=True,
        metavar="INT",
        type=int,
        help="Resolution of the bins, to calculate the signal (in bp).",
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
        "second column. This can be found in UCSC for your species of interest.",
    )

    optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")

def run(opts):
    
    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

    work_dir = opts.work_dir
    data = opts.data
    hic_path = opts.hic_file
    coords = opts.coords
    resolution = opts.reso

    tmp_dir = f"{work_dir}tmp"
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    if hic_path.endswith(".cool") or hic_path.endswith(".mcool"):
        
        if resolution not in sorted([int(x) for x in list(h5py.File(hic_path)["resolutions"].keys())]):

            exit("The given resolution is not in the provided cooler file. Exiting...")

        hic_path = hic_path + "::/resolutions/" + str(resolution)

        misc.check_cooler_names(hic_path, data, coords)

    elif hic_path.endswith(".hic"):

        if resolution not in hicstraw.HiCFile(hic_path).getResolutions():
             
            exit("The given resolution is not in the provided Hi-C file. Exiting...")

        misc.check_hic_names(hic_path, data, coords)

    column_dict = {}

    # Create a bed file that contains regions of a given resolution and sort it.
    BedTool().window_maker(g=coords, w=resolution).saveas(f"{work_dir}tmp/{resolution}bp_bin_unsorted.bed")

    sp.call(
        f"sort {tmp_dir}/{resolution}bp_bin_unsorted.bed -k1,1V -k2,2n -k3,3n | grep -v random | grep -v Un | grep -v alt > {tmp_dir}/{resolution}bp_bin.bed",
        shell=True,
    )

    # Check if the given bed file has a header and sort the file. If there is a header, save column
    # names to a dictionary to put it in the next step, then sort the file.
    for f in data:

        signal_file_name = f.rsplit("/", 1)[1]
        tmp_signal_path = f"{tmp_dir}/tmp_{signal_file_name}"

        sp.call(f"cp {f} {tmp_signal_path}", shell=True)

        try:

            float(  # If first row fourth column is convertible to float, it is a signal, so the first row is not a header.
                sp.getoutput(f"head -n 1 {f} | cut -f 4")
            )

            sp.call(
                f"sort {tmp_signal_path} -k1,1V -k2,2n -k3,3n | grep -v random | grep -v Un | grep -v alt > {tmp_dir}/sorted_{f.rsplit('/', 1)[1]}",
                shell=True,
            )

            header = False

        except ValueError:

            # Saving the corresponding column names in a dict, with only the file name as a key.
            column_dict[signal_file_name] = sp.getoutput(f"head -n 1 {f}").split(sep="\t")

            sp.call(
                f"tail -n +1 {tmp_signal_path} | sort -k1,1V -k2,2n -k3,3n | grep -v random | grep -v Un | grep -v alt > {tmp_dir}/sorted_{f.rsplit('/', 1)[1]}",
                shell=True,
            )

            header = True

        # Intersect the sorted files and the binnarized file.
        BedTool(f"{tmp_dir}/{resolution}bp_bin.bed").intersect(BedTool(f"{tmp_dir}/sorted_{f.rsplit('/', 1)[1]}"), wo=True, sorted=True).saveas(f"{tmp_dir}/intersected_{f.rsplit('/', 1)[1]}")

        awk_com = (
            "awk '{print $0\"\t\"$7*($8/($6-$5))}' "
            + f"{tmp_dir}/intersected_{f.rsplit('/', 1)[1]} > {tmp_dir}/intersected_ok_{f.rsplit('/', 1)[1]}"
        )
        sp.call(awk_com, shell=True)

    # Create a list of paths to the intersected files.
    intersected_files_paths = [(f"{tmp_dir}/intersected_ok_" + i.rsplit("/", 1)[1]) for i in data]

    final_intersect = pd.read_csv(
        intersected_files_paths[0], sep="\t", header=None, low_memory=False
    )  # Read the first intersected file,
    final_intersect = final_intersect.drop([3, 4, 5, 7, 8], axis=1)  # Drop unnecesary columns,

    if header == True:

        final_intersect.columns = column_dict[
            intersected_files_paths[0].rsplit("/", 1)[1].split("_", 2)[2]
        ]  # Input the corresponding column names

    else:

        final_intersect.columns = [
            "chrom",
            "start",
            "end",
            f"{intersected_files_paths[0].rsplit('/', 1)[1].split('_', 2)[2]}",
        ]

    final_intersect = (  # Calculate the median of all signals of the same bin.
        final_intersect.groupby(["chrom", "start", "end"])[final_intersect.columns[3 : len(final_intersect.columns)]]
        .median()
        .reset_index()
    )

    # Process the rest of the files the same way and merge with next one, until all files are merged.
    if len(intersected_files_paths) > 1:

        for i in range(1, len(intersected_files_paths)):

            tmp_intersect = pd.read_csv(intersected_files_paths[i], sep="\t", header=None, low_memory=False)
            tmp_intersect = tmp_intersect.drop([3, 4, 5, 7, 8], axis=1)

            if header == True:

                tmp_intersect.columns = column_dict[
                    intersected_files_paths[i].rsplit("/", 1)[1].split("_", 2)[2]
                ]  # Input the corresponding column names

            else:

                tmp_intersect.columns = [
                    "chrom",
                    "start",
                    "end",
                    f"{intersected_files_paths[i].rsplit('/', 1)[1].split('_', 2)[2]}",
                ]

            tmp_intersect = (
                tmp_intersect.groupby(["chrom", "start", "end"])[tmp_intersect.columns[3 : len(tmp_intersect.columns)]]
                .median()
                .reset_index()
            )

            final_intersect = pd.merge(final_intersect, tmp_intersect, on=["chrom", "start", "end"], how="inner")

    # For each chromosome, create a directory and save the information for that chromosome in .csv and .pkl.
    for chrom in sorted(final_intersect["chrom"].unique()):

        pathlib.Path(f"{work_dir}signal/{chrom}").mkdir(parents=True, exist_ok=True)
        final_intersect[final_intersect.chrom == f"{chrom}"].to_pickle(
            f"{work_dir}signal/{chrom}/{chrom}_signal.pkl"
        )
        final_intersect[final_intersect.chrom == f"{chrom}"].to_csv(
            f"{work_dir}signal/{chrom}/{chrom}_signal.csv", sep="\t", index=False
        )

    print(f"\nSignal bed files saved to {work_dir}signal/")

    chroms_no_signal = [
        chrom
        for chrom in sp.getoutput(f"cut -f 1 {tmp_dir}/{resolution}bp_bin.bed | uniq").split("\n")
        if chrom not in final_intersect["chrom"].unique()
    ]

    if len(chroms_no_signal) == 1:

        print(f"Omited chromosome {chroms_no_signal[0]} as it did not have any signal.")

    elif len(chroms_no_signal) > 1:

        print(f"Omited chromosomes {', '.join(chroms_no_signal)} as they did not have any signal.")

    misc.remove_folder(pathlib.Path(tmp_dir))  

    print("\ndone.")
