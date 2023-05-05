"""
This script processes signal .bed files or .bedGraph files, binnarizing them at a given resolution,
merging all signals in the same dataframe and subsetting by chromosomes.
"""

import sys
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from metaloci.misc import misc
import subprocess as sp
import pandas as pd
import pathlib
import cooler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

description = """
This script picks a bed file (USCS format) of signals and transforms it into the format  
needed for METALoci to work.
Signal names must be in the format CLASS.ID_IND.ID.\n"
\tClass.ID can be the name of the signal mark.\n"
\tInd.ID can be the name of the patient/cell line/experiment.\n"""

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=description, add_help=False)

input_arg = parser.add_argument_group(title="Input arguments")

input_arg.add_argument(
    "-w",
    "--work-dir",
    dest="work_dir",
    required=True,
    metavar="PATH",
    type=str,
    help="Path to working directory.",
)

input_arg.add_argument(
    "-c",
    "--hic",
    dest="hic_file",
    metavar="PATH",
    type=str,
    required=True,
    help="complete path to the cooler file.",
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
    "described below. "
    "(More than one file can be specified, space separated)",
)
input_arg.add_argument("-o", "--name", dest="output", required=True, metavar="STR", type=str, help="Output file name.")
input_arg.add_argument(
    "-r",
    "--resolution",
    dest="reso",
    required=True,
    metavar="INT",
    type=int,
    help="Resolution of the bins to calculate the signal (in bp).",
)
region_arg = parser.add_argument_group(title="Region arguments")

region_arg.add_argument(
    "-s",
    "--coords",
    dest="coords",
    required=True,
    metavar="PATH",
    type=str,
    help="Full path to a file that contains the name of the chromosomes in the "
    "first column and the ending coordinate of the chromosome in the "
    "second column.",
)
optional_arg = parser.add_argument_group(title="Optional arguments")
optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
optional_arg.add_argument("-u", "--debug", dest="debug", action="store_true", help=SUPPRESS)

args = parser.parse_args(None if sys.argv[1:] else ["-h"])

if not args.work_dir.endswith("/"):

    args.work_dir += "/"

work_dir = args.work_dir
data = args.data
hic_path = args.hic_file
out_name = args.output
coords = args.coords
resolution = args.reso
debug = args.debug

if debug:

    table = [
        ["work_dir", work_dir],
        ["data", data],
        ["out_name", out_name],
        ["centro_and_telo_file", coords],
        ["binSize", resolution],
    ]

    print(table)
    sys.exit()

tmp_dir = f"{work_dir}tmp"
pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)

if hic_path.endswith(".cool") or hic_path.endswith(".mcool"):

    hic_path = hic_path + "::/resolutions/" + str(resolution)

misc.check_chromosome_names(hic_path, data, coords)

column_dict = {}

# Create a bed file that contains regions of a given resolution and sort it.
sp.call(
    f"bedtools makewindows -g {coords} -w {resolution} > {tmp_dir}/{resolution}bp_bin_unsorted.bed",
    shell=True,
)

sp.call(
    f"sort {tmp_dir}/{resolution}bp_bin_unsorted.bed -k1,1V -k2,2n -k3,3n | grep -v random | grep -v chrUn > {tmp_dir}/{resolution}bp_bin.bed",
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

        # sp.call(
        #     f"cat {tmp_signal_path} | sed -e 's/.*/chr&/' > {tmp_signal_path}_tmp && mv {tmp_signal_path}_tmp {tmp_signal_path}",
        #     shell=True,
        # )

        sp.call(
            f"sort {tmp_signal_path} -k1,1V -k2,2n -k3,3n | grep -v random | grep -v chrUn > {tmp_dir}/sorted_{f.rsplit('/', 1)[1]}",
            shell=True,
        )

        header = False

    except ValueError:

        # Saving the corresponding column names in a dict, with only the file name as a key.
        column_dict[signal_file_name] = sp.getoutput(f"head -n 1 {f}").split(sep="\t")

        # sp.call(
        #     f"tail -n +2 {tmp_signal_path} | sed -e 's/.*/chr&/' > {tmp_signal_path}_tmp && mv {tmp_signal_path}_tmp {tmp_signal_path}",
        #     shell=True,
        # )

        sp.call(
            f"tail -n +1 {tmp_signal_path} | sort -k1,1V -k2,2n -k3,3n | grep -v random | grep -v chrUn > {tmp_dir}/sorted_{f.rsplit('/', 1)[1]}",
            shell=True,
        )

        header = True

    # Intersect the sorted files and the binnarized file.
    sp.call(
        f"bedtools intersect -a {tmp_dir}/{resolution}bp_bin.bed -b {tmp_dir}/sorted_{f.rsplit('/', 1)[1]} -wo -sorted > {tmp_dir}/intersected_{f.rsplit('/', 1)[1]}",
        shell=True,
    )

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

        tmp_intersect = pd.read_csv(intersected_files_paths[i], sep="\t", header=None)
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

# For each chromosome, create a directory and save the information for that chromosome in .csv and
# .pkl.
for chrom in sorted(final_intersect["chrom"].unique()):

    pathlib.Path(f"{work_dir}signal/{chrom}").mkdir(parents=True, exist_ok=True)
    final_intersect[final_intersect.chrom == f"{chrom}"].to_pickle(
        f"{work_dir}signal/{chrom}/{out_name}_{chrom}_signal.pkl"
    )
    final_intersect[final_intersect.chrom == f"{chrom}"].to_csv(
        f"{work_dir}signal/{chrom}/{out_name}_{chrom}.csv", sep="\t", index=False
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
