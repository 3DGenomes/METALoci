"""
Miscellaneous functions for METALoci
"""
import gzip
import os
import pathlib
import re
import subprocess as sp
import sys
from collections import defaultdict
from pathlib import Path
from pickle import UnpicklingError

import cooler
import hicstraw
import numpy as np
import pandas as pd
from metaloci import mlo
from metaloci.spatial_stats import lmi


def signal_binnarize(data: pd.DataFrame, sum_type: str) -> pd.DataFrame:
    """
    Parsing the signal data frame with the appropiate summarising method.

    Parameters
    ----------
    data : pd.DataFrame
        Signal DataFrmae
    sum_type : str
        Method to binnarize the signal

    Returns
    -------
    data : pd.DataFrame
        Binnarized signal data frame
    """
    if sum_type == "median":

        data = data.groupby(["chrom", "start", "end"])[data.columns[3: len(data.columns)]].median().reset_index()

    elif sum_type == "mean":

        data = data.groupby(["chrom", "start", "end"])[data.columns[3: len(data.columns)]].mean().reset_index()

    elif sum_type == "min":

        data = data.groupby(["chrom", "start", "end"])[data.columns[3: len(data.columns)]].min().reset_index()

    elif sum_type == "max":

        data = data.groupby(["chrom", "start", "end"])[data.columns[3: len(data.columns)]].max().reset_index()

    elif sum_type == "count":

        data = data.groupby(["chrom", "start", "end"])[data.columns[3: len(data.columns)]].count().reset_index()

    return data


def remove_folder(path: Path):
    """
    Removes a folder that is not empty.

    Parameters
    ----------
    path : Path
        Path to the folder to be removed.
    """

    for subdirectory in path.iterdir():

        if subdirectory.is_dir():

            remove_folder(subdirectory)

        else:

            subdirectory.unlink()

    path.rmdir()


def check_diagonal(diagonal: np.ndarray) -> tuple[int, float, float, list]:
    """
    Checks the 0s/NaNs on the diagonal and saves the total number of zeroes, the the max stretch (number of zeros in 
    a row) as a percentage, the percentage of zeroes, and the location of zeroes.

    Parameters
    ----------
    diagonal : np.ndarray
        Diagonal of the HiC matrix.

    Returns
    -------
    total : int
        Total number of zeroes.
    percentage_stretch : float
        Percentage of max stretch.
    percentatge_zeroes : float
        Percentage of zeroes.
    zero_loc : list
        Location of zeroes in the diagonal.
    """

    total = 0
    stretch = 0
    max_stretch = 0
    zero_loc = []

    for i, element in enumerate(diagonal):

        if element == 0 or np.isnan(element):

            total += 1
            stretch += 1
            zero_loc.append(i)
            max_stretch = max(max_stretch, stretch)

        else:

            stretch = 0

    percentage_zeroes = np.round(total / len(diagonal) * 100, decimals=2)
    percentage_stretch = np.round(max_stretch / len(diagonal) * 100, decimals=2)

    return total, percentage_stretch, percentage_zeroes, zero_loc


def clean_matrix(mlobject: mlo.MetalociObject) -> mlo.MetalociObject:
    """
    Clean a given Hi-C matrix. It checks if the matrix has too many zeroes at
    he diagonal, removes values that are zero at the diagonal but are not in
    the rest of the matrix, adds pseudocounts to zeroes depending on the min
    value, scales all values depending on the min value and computes the log10
    of all values.

    Parameters
    ----------
    mlo : mlo.MetalociObject
        METALoci object with a matrix in it.

    Returns
    -------
    mlobject : mlo.MetalociObject
        mlo object with the assigned clean matrix.
    """

    diagonal = np.array(mlobject.matrix.diagonal())
    total_zeroes, max_stretch, percentage_zeroes, zero_loc = check_diagonal(diagonal)

    if total_zeroes == len(diagonal):

        mlobject.bad_region = "empty"

        return mlobject

    elif percentage_zeroes >= 20:

        mlobject.bad_region = "too many zeros"

    elif max_stretch >= 10:

        mlobject.bad_region = "stretch"

    mlobject.matrix[zero_loc] = 0
    mlobject.matrix[:, zero_loc] = 0

    # Pseudocounts if min is zero
    if np.nanmin(mlobject.matrix) == 0:

        pc = np.nanmin(mlobject.matrix[mlobject.matrix > 0])
        mlobject.matrix = mlobject.matrix + pc

    # Scale if all below 1
    if np.nanmax(mlobject.matrix) <= 1 or np.nanmin(mlobject.matrix) <= 1:

        sf = 1 / np.nanmin(mlobject.matrix)
        mlobject.matrix = mlobject.matrix * sf

    if np.nanmin(mlobject.matrix) < 1:

        mlobject.matrix[mlobject.matrix < 1] = 1

    mlobject.matrix = np.log10(mlobject.matrix)

    return mlobject


def signal_normalization(region_signal: pd.DataFrame, pseudocounts: float = None, norm: str = None) -> np.ndarray:
    """
    Normalize signal values.

    Parameters
    ----------
    ind_signal : pd.DataFrame
        Subset of the signal for a given region and for a signal type.
    pseudocounts : float, optional
        Pseudocounts to add if the signal is 0, by default corresponds to the median of the signal for the region.
    norm : str, optional
        Type of normalization to use. Values can be "max" (divide each value by the max value in the signal),
        "sum" (divide each value by the sum of all values), or "01" (value - min(signal) / max(signal) - min(signal)),
        by default "01"

    Returns
    -------
    signal : np.ndarray
        Array of normalized signal values for a region and a signal type.
    """

    if pseudocounts is None:

        median_default = np.nanmedian(region_signal)
        signal = [median_default if np.isnan(index) else index for index in region_signal]

    else:

        signal = [pseudocounts if np.isnan(index) else index + pseudocounts for index in region_signal]

    if isinstance(norm, (int, float)):

        signal = [float(value) / int(norm) for value in signal]

    elif norm == "max":

        signal = [float(value) / max(signal) for value in signal]

    elif norm == "sum":

        signal = [float(value) / sum(signal) for value in signal]

    elif norm == "01":

        signal = [(float(value) - min(signal)) / (max(signal) - min(signal) + 0.01) for value in signal]

    signal = np.array(signal)

    return signal


def check_names(hic_file: Path, data: Path, coords: Path, resolution: int = None) -> list:
    """
    Checks if the chromosome names in the signal, cool/mcool/hic and chromosome sizes
    files are the same.

    Parameters
    ----------
    hic_file : Path
        Path to the cooler file.
    data : Path
        Path to the signal file.
    coords : Path
        Path to the chromosome sizes file.
    resolution : int, optional
        Resolution to choose on the mcool file.

    Returns
    -------
    chrom_list : list
        List of chromosomes in the cooler file.

    """

    with open(data[0], "r", encoding="utf-8") as handler:

        if [line.strip() for line in handler][10].startswith("chr"):

            signal_chr_nom = "chrN"

        else:

            signal_chr_nom = "N"

    if hic_file.endswith(".cool"):

        chrom_list = cooler.Cooler(hic_file).chromnames

    elif hic_file.endswith(".mcool"):

        chrom_list = cooler.Cooler(f"{hic_file}::/resolutions/{resolution}").chromnames

    elif hic_file.endswith(".hic"):

        chrom_list = [chrom.name for chrom in hicstraw.HiCFile(hic_file).getChromosomes()][1:]

    if "chr" in chrom_list[0]:

        cooler_chr_nom = "chrN"

    else:

        cooler_chr_nom = "N"

    with open(coords, "r", encoding="utf-8") as handler:

        if [line.strip() for line in handler][0].startswith("chr"):

            coords_chr_nom = "chrN"

        else:

            coords_chr_nom = "N"

    if not signal_chr_nom == cooler_chr_nom == coords_chr_nom:

        sys.exit(
            "\nThe signal, cooler and chromosome sizes files do not have the same nomenclature for chromosomes:\n\
            \n\tSignal chromosomes nomenclature is '{signal_chr_nom}'.\
            \n\tHi-C chromosomes nomenclature is '{cooler_chr_nom}'.\
            \n\tChromosome sizes nomenclature is '{coords_chr_nom}'.\
            \n\nPlease, rename the chromosome names.\
            \n\nExiting..."
        )

    return chrom_list


def natural_sort(element_list: list) -> list:
    """
    Sort the list with natural sorting.

    Parameters
    ----------
    element_list : list
        List to be sorted.

    Returns
    -------
    sorted_element_list : list
        Sorted list.
    """

    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]

    sorted_element_list = sorted(element_list, key=alphanum_key)

    return sorted_element_list

def ucscparser(gene_file: Path, name: str, extend: int, resolution: int) -> tuple[dict, dict, dict, str]:
    """
    Parses a UCSC file and returns the information of the genes, excluding artifacts.

    Parameters
    ----------
    gene_file : str
        Path to the file that contains the genes.
    name : str
        Name of the project.
    extend : int
        Extent of the region to be analyzed.
    resolution : int
        Resolution at which to split the genome.

    Returns
    -------
    id_chrom : dict
        Dictionary linking the gene id to the chromosome.
    id_tss : dict
        Dictionary linking the gene id to the tss.
    id_name : dict
        Dictionary linking the gene id to the name.
    filename : str
        Name of the end file.
    """

    myopen = gzip.open if gene_file.endswith("gz") else open

    # Compile regular expressions
    re_comment = re.compile("##")
    re_chr = re.compile(r"chr\w+")
    re_gene_id = re.compile(r'gene_id "(\S+)";')
    re_gene_name = re.compile(r'gene_name "(\S+)";')

    id_tss = defaultdict(int)
    id_name = defaultdict(str)
    id_chrom = defaultdict(str)

    print("Using a UCSC annotation file. Considering anotated trasncripts as genes.")
    print("Gathering information from the annotation file...")

    with myopen(gene_file, mode="rt", encoding="utf-8") as ucsc_reader:

        for line in ucsc_reader:

            if re_comment.search(line) or not re_chr.search(line):

                continue

            line_s = line.split("\t")

            # if line_s[0] contains "_" skip the line
            if "_" in line_s[0]:
                continue

            if line_s[2] == "transcript" and line_s[0] != "chrM":

                try:
                    gene_id = re_gene_id.search(line).group(1)
                except AttributeError:
                    print(line)

                # if gene_id has a version number, remove it
                if "." in gene_id:
                    gene_id = gene_id.split(".")[0]

                if re_gene_name.search(line):
                    gene_name = re_gene_name.search(line).group(1)
                else:
                    gene_name = gene_id

                id_chrom[gene_id] = line_s[0]
                id_name[gene_id] = gene_name
                id_tss[gene_id] = int(line_s[3]) if line_s[6] == "+" else int(line_s[4])

    filename = f"{name}_all_{extend}_{resolution}_agg.txt"

    return id_chrom, id_tss, id_name, filename

def gtfparser(gene_file: Path, name: str, extend: int, resolution: int) -> tuple[dict, dict, dict, str]:
    """
    Parses a gtf file and returns the information of the genes, excluding artifacts

    Parameters
    ----------
    gene_file : str
        Path to the file that contains the genes.
    name : str
        Name of the project.
    extend : int
        Extent of the region to be analyzed.
    resolution : int
        Resolution at which to split the genome.

    Returns
    -------
    id_chrom : dict 
        Dictionary linking the gene id to the chromosome.
    id_tss : dict
        Dictionary linking the gene id to the tss.
    id_name : dict
        Dictionary linking the gene id to the name.
    filename : str
        Name of the end file
    """

    gene_type_count = defaultdict(int)

    myopen = gzip.open if gene_file.endswith("gz") else open

    # Compile regular expressions
    re_comment = re.compile("##")
    re_chr = re.compile(r"chr\w+")
    re_artifact = re.compile(r'gene_type "artifact";')
    re_gene_type = re.compile(r'gene_type "(\w+)";')
    re_gene_id = re.compile(r'gene_id "(ENS\w+)\.\d+"')
    re_gene_name = re.compile(r'gene_name "([\w\-\_\.]+)";')

    with myopen(gene_file, mode="rt", encoding="utf-8") as gtf_reader:

        for line in gtf_reader:

            if re_comment.search(line) or not re_chr.search(line):

                continue

            line_s = line.split("\t")

            if line_s[2] == "gene" and line_s[0] != "chrM":

                if re_artifact.search(line):

                    continue

                gene_type_count[re_gene_type.search(line).group(1)] += 1

    print("Index: 0; All genes")

    gene_type_keys = list(gene_type_count.keys())

    for type_index, type_name in enumerate(gene_type_keys):
        print(f"Index: {type_index+1}; {type_name}")

    chrom_index = None

    while not isinstance(chrom_index, int) or chrom_index not in range(len(gene_type_count.keys()) + 1):

        chrom_index = input("Choose the gene type to parse in the final file: ")

        try:

            chrom_index = int(chrom_index)

        except ValueError:

            print("Please, enter a number.")
            continue

        if chrom_index not in range(len(gene_type_count.keys()) + 1):

            print(f"Please, enter a number between 0 and {len(gene_type_count.keys())}")
            continue

    if chrom_index == 0:

        chrom_type_patttern = re.compile(r'gene_type "\w+";')
        filename = f"{name}_all_{extend}_{resolution}_agg.txt"
        print("Parsing all genes...")

    else:

        chrom_type_patttern = re.compile(f'gene_type "{gene_type_keys[chrom_index - 1]}";')
        filename = f"{name}_{gene_type_keys[chrom_index - 1]}_{extend}_{resolution}_gene_coords.txt"
        print(f"Gene type chosen: {gene_type_keys[chrom_index - 1]}")

    id_tss = defaultdict(int)
    id_name = defaultdict(str)
    id_chrom = defaultdict(str)

    print("Gathering information from the annotation file...")

    with myopen(gene_file, mode="rt", encoding="utf-8") as gtf_reader:

        for line in gtf_reader:

            if re_comment.search(line) or not re_chr.search(line):

                continue

            line_s = line.split("\t")

            if line_s[2] == "gene" and line_s[0] != "chrM" and chrom_type_patttern.search(line):

                gene_id = re_gene_id.search(line).group(1)
                gene_name = re_gene_name.search(line).group(1)

                id_chrom[gene_id] = line_s[0]
                id_name[gene_id] = gene_name
                id_tss[gene_id] = int(line_s[3]) if line_s[6] == "+" else int(line_s[4])

    return id_chrom, id_tss, id_name, filename


def bedparser(gene_file_f: Path, name: str, extend: int, resolution: int) -> tuple[dict, dict, dict, str]:
    """
    Parses a bed file and returns the information of the genes, excluding artifacts.

    Parameters
    ----------
    gene_file : path
        Path to the file that contains the genes.
    name : str
        Name of the project
    extend : int
        Extent of the region to be analyzed.
    resolution : int
        Resolution at which to split the genome.

    Returns
    -------
    id_chrom : dict
        Dictionary linking the gene id to the chromosome.
    id_tss : dict
        Dictionary linking the gene id to the tss.
    id_name : dict
        Dictionary linking the gene id to the name.
    filename : str
        Name of the end file.
    """

    id_tss = defaultdict(int)
    id_name = defaultdict(str)
    id_chrom = defaultdict(str)

    print("Gathering information from the annotation file...")

    filename = f"{name}_all_{extend}_{resolution}_agg.txt"
    bed_file = pd.read_table(gene_file_f, names=["coords", "symbol", "id"])

    for _, row in bed_file.iterrows():

        coords = row.coords.split(":")

        id_chrom[row.id] = coords[0]
        id_tss[row.id] = int(coords[1])
        id_name[row.id] = row.symbol

    return id_chrom, id_tss, id_name, filename


def binsearcher(id_tss: dict, id_chrom: dict, id_name: dict, bin_genome: pd.DataFrame) -> pd.DataFrame:
    """
    Searches the bin index where the gene is located.

    Parameters
    ----------
    id_tss : dict
        Dictionary linking the gene id to the tss.
    id_chrom : dict
        Dictionary linking the gene id to the chromosome.
    id_name : dict
        Dictionary linking the gene id to the name.
    bin_genome : pd.DataFrame
        DataFrame containing the bins of the genome.

    Returns
    -------
    data: pd.DataFrame
        DataFrame containing the information of the genes and the bin index
    """

    data = defaultdict(list)

    for g_id, tss_pos in id_tss.items():

        chrom_bin = bin_genome[(bin_genome["chrom"] == id_chrom[g_id])]
        sub_bin_index = chrom_bin.loc[(chrom_bin["start"] <= tss_pos) & (chrom_bin["end"] >= tss_pos)].index[0]

        data["chrom"].append(id_chrom[g_id])
        data["bin_index"].append(sub_bin_index)
        data["gene_name"].append(id_name[g_id])
        data["gene_id"].append(g_id)

    print("Aggregating genes that are located in the same bin...")

    data = pd.DataFrame(data)
    data = data.groupby(["chrom", "bin_index"])[data.columns].agg(",".join).reset_index()

    data[["chrom"]] = data[["chrom"]].astype(str)
    data[["bin_index"]] = data[["bin_index"]].astype(int)
    data[["gene_name"]] = data[["gene_name"]].astype(str)
    data[["gene_id"]] = data[["gene_id"]].astype(str)

    return data


def write_bed(mlobject: mlo.MetalociObject, signal_type: str, neighbourhood: int, BFACT: float, args=None,
              silent: bool = False) -> None:
    """
    Writes the bed file with the metalocis location.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.
    signal_type : str
        Signal type to be used.
    neighbourhood : int
        Neighbourhood to be used.
    BFACT : float
        BFACT value to be used.
    args : argparse.Namespace
        Arguments from the command line.
    silent : bool, optional
        Variable that controls the verbosity of the function (useful for multiprocessing), by default False.
    """

    merged_lmi_geometry = pd.merge(
        mlobject.lmi_info[signal_type],
        mlobject.lmi_geometry,
        on=["bin_index", "moran_index"],
        how="inner",
    )
    bed = lmi.get_bed(mlobject, merged_lmi_geometry, neighbourhood, args.quadrants, args.signipval, silent=silent)

    if bed is not None and len(bed) > 0:

        metaloci_bed_path = os.path.join(args.work_dir, mlobject.chrom, "metalocis_log", signal_type)
        bed_file_name = os.path.join(
            metaloci_bed_path,
            f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_{signal_type}_"
            f"q-{'_'.join([str(q) for q in args.quadrants])}_metalocis.bed")

        pathlib.Path(metaloci_bed_path).mkdir(parents=True, exist_ok=True)
        bed.to_csv(bed_file_name, sep="\t", index=False)

        if not silent:

            print(f"\t-> Bed file with metalocis location saved to: {bed_file_name}")


def write_moran_data(mlobject: mlo.MetalociObject, args, silent=False) -> None:
    """
    Writes the Moran data to a file.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with the Moran data.
    args : argparse.Namespace
        Arguments from the command line.
    silent : bool, optional
        Variable that controls the verbosity of the function (useful for multiprocessing), by default False.
    """

    for signal, df in mlobject.lmi_info.items():

        moran_data_path = os.path.join(args.work_dir, mlobject.chrom, "moran_data", signal)

        pathlib.Path(moran_data_path).mkdir(parents=True, exist_ok=True)
        df.to_csv(
            os.path.join(
                moran_data_path,
                f"{re.sub(':|-', '_', mlobject.region)}_{signal}.tsv"),
            sep="\t", index=False, float_format="%.12f")

        if not silent:

            print(F"\t-> Moran data saved to "
                  f"'{moran_data_path}/{re.sub(':|-', '_', mlobject.region)}_{signal}.tsv'")


def get_poi_data(info_tuple: tuple):
    """
    Function to extract data from the METALoci objects and parse it into a table.

    Parameters
    ----------
    info_tuple : tuple
        Tuple containing the parameters to extract the data.
        This tuple should contain the following elements:
        - line: Line object
        - signals: List of signals to extract the data from
        - work_dir: Path to the working directory
        - bad_file_filename: Path to the bad file
        - output_file: Path to the output file
        - pval: P-value to filter the data
        - quadrant_list: List of quadrants to filter the data
        - region_file: Boolean to indicate if a region file should be created
        - rf: Path to the region file
    """

    if len(info_tuple) == 9:

        line, signals, work_dir, bad_file_filename, output_file, pval, quadrant_list, region_file, rf = info_tuple

    else:
        
        line, signals, work_dir, bad_file_filename, output_file, pval, quadrant_list = info_tuple
        region_file = False
        rf = None

    mlo_filename = f"{line.coords.replace(':', '_').replace('-', '_')}.mlo"

    if not os.path.exists(os.path.join(work_dir, mlo_filename.split("_")[0], 'objects', mlo_filename)):

        with open(bad_file_filename, mode="a", encoding="utf-8") as bad_file_handler:

            bad_file_handler.write(f"{line.coords}\t{line.symbol}\t{line.id}\tno_file\n")

        return

    table_line = f"{line.coords}\t{line.symbol}\t{line.id}"
    region_line = f"{line.coords}\t{line.symbol}\t{line.id}"

    try:

        mlo_data = pd.read_pickle(os.path.join(work_dir, mlo_filename.split("_")[0],'objects', mlo_filename))

    except UnpicklingError:

        with open(bad_file_filename, mode="a", encoding="utf-8") as bad_file_handler:

            bad_file_handler.write(f"{line.coords}\t{line.symbol}\t{line.id}\tcorrupt_file\n")

        return

    if mlo_data.bad_region:

        with open(bad_file_filename, mode="a", encoding="utf-8") as bad_file_handler:

            bad_file_handler.write(f"{line.coords}\t{line.symbol}\t{line.id}\t{mlo_data.bad_region}\n")

        return

    bad_lines = []

    for signal in signals:

        try:

            poi_data = mlo_data.lmi_info[signal].loc[mlo_data.poi].to_dict()

            table_line += f"\t{poi_data['moran_quadrant']}\t{poi_data['LMI_score']}\t{poi_data['LMI_pvalue']}\t{poi_data['ZSig']}\t{poi_data['ZLag']}"

            if region_file:

                if poi_data['moran_quadrant'] in quadrant_list and poi_data['LMI_pvalue'] <= pval:

                    region_line += table_line

                else:

                    region_line += "\tNA\tNA\tNA\tNA\tNA"

        except KeyError:

            bad_lines.append(f"{line.coords}\t{line.symbol}\t{line.id}\tno_signal_{signal}")

    with open(bad_file_filename, mode="a", encoding="utf-8") as bad_file_handler:

        bad_file_handler.write('\n'.join(bad_lines))

    with open(output_file, mode="a", encoding="utf-8") as output_file_handler:

        output_file_handler.write(f"{table_line}\n")

    if region_file:

        with open(rf, mode="a", encoding="utf-8") as regionfile_h:

            regionfile_h.write(f"{region_line}\n")


def write_bad_region(mlobject, work_dir):
    """
    Writes the bad regions, after quality checking, to a file.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.
    work_dir : str
        Path to the working directory.
    """

    with open(f"{work_dir}bad_regions.txt", "a+", encoding="utf-8") as handler:

        log = f"{mlobject.region}\t{mlobject.bad_region}\n"

        handler.seek(0)

        if not any(mlobject.region in line.split('\t', 1)[0] for line in handler) and mlobject.bad_region is not None:

            handler.write(log)


def has_exactly_one_line(file_path: str) -> bool:
    """
    Function to check if a file has exactly one line.

    Parameters
    ----------
    file_path : str
        Path to the file to check.

    Returns
    -------
    bool
        True if the file has exactly one line, False otherwise.
    """
    with open(file_path, mode="r", encoding="utf-8") as f:

        line = f.readline()

        if not line:

            return False
        
        if f.readline():

            return False
        
    return True
