"""
This script processes signal .bed files or .bedGraph files, binnarizing them at a given resolution,
merging all signals in the same dataframe and subsetting by chromosomes.
"""

import os
import pathlib
import subprocess as sp
import warnings
from argparse import HelpFormatter

import h5py
import hicstraw
import pandas as pd
from metaloci.misc import misc
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
from tqdm import tqdm
from pybedtools import BedTool

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

    print("Checking input files compatibility...", end = "\r")

    if hic_path.endswith(".cool") or hic_path.endswith(".mcool"):
        
        if resolution not in sorted([int(x) for x in list(h5py.File(hic_path)["resolutions"].keys())]):

            exit("The given resolution is not in the provided cooler file. Exiting...")

        hic_path = hic_path + "::/resolutions/" + str(resolution)

        hic_chroms = misc.check_cooler_names(hic_path, data, coords)

    elif hic_path.endswith(".hic"):

        aval_resolutions = hicstraw.HiCFile(hic_path).getResolutions()

        if resolution not in aval_resolutions:
             
            print("The given resolution is not in the provided Hi-C file.")
            print("The available resolutions are: " + ", ".join(misc.natural_sort([str(x) for x in aval_resolutions])))
            exit("Exiting...")

        hic_chroms = misc.check_hic_names(hic_path, data, coords)

    print("Checking input files compatibility... done.")

    column_dict = {}

    # Create a bed file that contains regions of a given resolution and sort it.
    BedTool().window_maker(g=coords, w=resolution).saveas(f"{work_dir}tmp/{resolution}bp_bin_unsorted.bed")

    sp.call(
        f"sort {tmp_dir}/{resolution}bp_bin_unsorted.bed -k1,1V -k2,2n -k3,3n > {tmp_dir}/{resolution}bp_bin.bed",
        shell=True,
    ) 

    os.remove(f"{work_dir}tmp/{resolution}bp_bin_unsorted.bed")

    last_header = None
    # Check if the given bed file has a header and sort the file. If there is a header, save column
    # names to a dictionary to put it in the next step, then sort the file.
    for f in data:

        print(f"\nProcessing signal file: {f}\n")
        print(f"\tSorting signal...", end = "\r")

        signal_file_name = os.path.basename(f)
        tmp_signal_path = f"{tmp_dir}/tmp_{signal_file_name}"

        sp.call(f"cp {f} {tmp_signal_path}", shell=True)

        if sp.getoutput(f"head -n 1 {f} | cut -f 1") != "chrom":
            
            sp.call(
                f"sort {tmp_signal_path} -k1,1V -k2,2n -k3,3n > {tmp_dir}/sorted_{signal_file_name}",
                shell=True,
            )
            
            header = False
            
        else:

            # Saving the corresponding column names in a dict, with only the file name as a key.
            column_dict[signal_file_name] = sp.getoutput(f"head -n 1 {f}").split(sep="\t")

            sp.call(
                f"tail -n +1 {tmp_signal_path} | sort -k1,1V -k2,2n -k3,3n > {tmp_dir}/sorted_{signal_file_name}",
                shell=True,
            )

            header = True
        
        if last_header is not None:
            
            if header != last_header:
                
                misc.remove_folder(pathlib.Path(tmp_dir)) 
                exit("ERROR: Some of the files to be processed have a header but others do not. Please, modify the files to keep them consistent.")
            
        last_header = header
        
        awk_com = f"tail -n +2 {f} | " + ("awk '{print $1}' | uniq")  # Retrieve chromosomes from the data file.
        chroms = sp.getoutput(awk_com).split(sep="\n")  
        chroms = misc.natural_sort([chrom for chrom in chroms if chrom in hic_chroms])  # Keep only chromosomes that are in the Hi-C file.
        input_file = f"{tmp_dir}/sorted_{signal_file_name}"

        print(f"\tSorting signal... done.")
        print(f"\tIntersecting signal with {resolution}bp bins...")

        pbar = tqdm(chroms, desc=f"        {chroms[0]}")

        for i, chrom in enumerate(pbar):
            
            # Do a subset of the signal
            awk_com = """awk '{{if($1=="%s") {{print}}}}' """ % f"{chrom}" + f"{input_file}" + f" > {tmp_dir}/{chrom}_sorted_{signal_file_name}"
            sp.call(awk_com, shell=True)

            # Do a subset of the BedTools makewindows file
            awk_com = """awk '{{if($1=="%s") {{print}}}}' """ % f"{chrom}" + f"{tmp_dir}/{resolution}bp_bin.bed" + f" > {tmp_dir}/{chrom}_{resolution}bp_bin.bed"
            sp.call(awk_com, shell=True)

            # Do an intersection of the signal and the bed file, per each chromsome.
            BedTool(f"{tmp_dir}/{chrom}_{resolution}bp_bin.bed").intersect(BedTool(f"{tmp_dir}/{chrom}_sorted_{signal_file_name}"), wao=True, sorted=True).saveas(f"{tmp_dir}/{chrom}_intersected_{signal_file_name}")

            os.remove(f"{tmp_dir}/{chrom}_{resolution}bp_bin.bed")
            os.remove(f"{tmp_dir}/{chrom}_sorted_{signal_file_name}")

            # Concatenate all the intersected files.
            if chrom == chroms[0]:

                sp.call(f"cp {tmp_dir}/{chrom}_intersected_{signal_file_name} {tmp_dir}/concatenated.bed", shell = True)

            else:
                
                sp.call(f"cat {tmp_dir}/concatenated.bed {tmp_dir}/{chrom}_intersected_{signal_file_name} > {tmp_dir}/tmp_concatenated.bed", shell=True)
                sp.call(f"mv {tmp_dir}/tmp_concatenated.bed {tmp_dir}/concatenated.bed", shell=True)

            os.remove(f"{tmp_dir}/{chrom}_intersected_{signal_file_name}")
            pbar.set_description(f"        {chroms[i+1] if i+1 < len(chroms) else 'all done! ('}")
        
        print("\tAssigning signal to bins...", end = "\r")
    
        n_of_col = sp.getoutput(f"head -n 1 {tmp_dir}/concatenated.bed" + " | awk '{print NF}'")
        awk_com = f"awk -F \'\\t\' -v OFS=\'\\t\' '{{if (${n_of_col} == 0) {{$4 = $1; $5 = $2; $6 = $3}} {{for (i = 7; i <= {n_of_col}; i++) {{$i = $i * (${n_of_col} / ($6 - $5))}}}} print}}' {tmp_dir}/concatenated.bed > {tmp_dir}/intersected_ok_{signal_file_name}"
        sp.call(awk_com, shell=True)

        # os.remove(f"{tmp_dir}/concatenated.bed")
        print("\tAssigning signal to bins... done.")

    print("\nConcatenating signals in one file...", end = "\r")

    # Create a list of paths to the intersected files.
    intersected_files_paths = [(f"{tmp_dir}/intersected_ok_" + os.path.basename(f)) for f in data]
    
    final_intersect = pd.read_csv(
        intersected_files_paths[0], sep="\t", header=None, low_memory=False
    )  # Read the first intersected file,

    final_intersect = final_intersect.drop([3, 4, 5, final_intersect.columns[-1]], axis=1)  # Drop unnecesary columns,

    if header == True:

        final_intersect.columns = column_dict[
            intersected_files_paths[0].rsplit("/", 1)[1].split("_", 2)[2]
        ]  # Input the corresponding column names

    else:

        final_intersect.columns = [
            "chrom",
            "start",
            "end",
            f"{os.path.basename(intersected_files_paths[0].rsplit('.', 1)[0]).split('intersected_ok_', 1)[1]}", 
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
            tmp_intersect = tmp_intersect.drop([3, 4, 5, tmp_intersect.columns[-1]], axis=1)

            if header == True:

                tmp_intersect.columns = column_dict[
                    intersected_files_paths[i].rsplit("/", 1)[1].split("_", 2)[2]
                ]  # Input the corresponding column names

            else:

                tmp_intersect.columns = [
                    "chrom",
                    "start",
                    "end",
            f"{os.path.basename(intersected_files_paths[i].rsplit('.', 1)[0]).split('intersected_ok_', 1)[1]}", 
                ]

            tmp_intersect = (
                tmp_intersect.groupby(["chrom", "start", "end"])[tmp_intersect.columns[3 : len(tmp_intersect.columns)]]
                .median()
                .reset_index()
            )

            final_intersect = pd.merge(final_intersect, tmp_intersect, on=["chrom", "start", "end"], how="outer").fillna(0)

    # For each chromosome, create a directory and save the information for that chromosome in .csv and .pkl.
    for chrom in sorted(final_intersect["chrom"].unique()):

        pathlib.Path(f"{work_dir}signal/{chrom}").mkdir(parents=True, exist_ok=True)
        final_intersect[final_intersect.chrom == f"{chrom}"].to_pickle(
            f"{work_dir}signal/{chrom}/{chrom}_signal.pkl"
        )
        final_intersect[final_intersect.chrom == f"{chrom}"].to_csv(
            f"{work_dir}signal/{chrom}/{chrom}_signal.tsv", sep="\t", index=False
        )

    print("Concatenating signals in one file... done.")

    chroms_no_signal = [
        chrom
        for chrom in hic_chroms
        if chrom not in final_intersect["chrom"].unique()
    ]

    if len(chroms_no_signal) == 1:

        print(f"Omited chromosome {chroms_no_signal[0]} as it did not have any signal.")

    elif len(chroms_no_signal) > 1:

        print(f"Omited chromosomes {', '.join(chroms_no_signal)} as they did not have any signal.")

    print(f"\nSignal bed files saved to {work_dir}signal/")
    misc.remove_folder(pathlib.Path(tmp_dir)) 

    print("\ndone.")
