"""
Script that contains helper functions of the METALoci package
"""
import re
import sys
import gzip
import os
import subprocess as sp
from pickle import UnpicklingError
from pathlib import Path
from collections import defaultdict

import hicstraw
import cooler
import numpy as np
import pandas as pd

from metaloci import mlo


def signal_binnarize(data: pd.DataFrame, sum_type: str) -> pd.DataFrame:
    """
    Parsing the signal data frame with the appropiate binnarizing method

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
    Checks the 0s/NaNs on the diagonal and saves the total number of zeroes,
    the the max stretch (number of zeros in a row) as a percentage,
    the percentage of zeroes, and the location of zeroes.

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
    Clean a given HiC matrix. It checks if the matrix has too many zeroes at
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

    if percentage_zeroes >= 20:

        mlobject.bad_region = "too many zeros"

    if max_stretch >= 10:

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
    Normalize signal values

    Parameters
    ----------
    ind_signal : pd.DataFrame
        Subset of the signal for a given region and for a signal type.
    pseudocounts : float, optional
        Pseudocounts to add if the signal is 0, by default corresponds to the median of the signal for the region.
    norm : str, optional
        Type of normalization to use.
        Values can be "max" (divide each value by the max value in the signal),
        "sum" (divide each value by the sum of all values), or "01" ( value - min(signal) / max(signal) - min(signal) ),
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
            "\nThe signal, cooler and chromosome sizes files do not have the same nomenclature for chromosomes:\n"
            f"\n\tSignal chromosomes nomenclature is '{signal_chr_nom}'. "
            f"\n\tHi-C chromosomes nomenclature is '{cooler_chr_nom}'. "
            f"\n\tChromosome sizes nomenclature is '{coords_chr_nom}'. "
            "\n\nPlease, rename the chromosome names. "
            "\n\nExiting..."
        )

    return chrom_list


def natural_sort(element_list: list) -> list:
    """
    Sort the list with natural sorting.

    Parameters
    ----------
    element_list : list
        List to be sorted

    Returns
    -------
    sorted_element_list : list
        Sorted list
    """

    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]

    sorted_element_list = sorted(element_list, key=alphanum_key)

    return sorted_element_list


def gtfparser(gene_file_f: Path, name: str, extend: int, resolution: int) -> tuple[dict, dict, dict, str]:
    """
    Parses a gtf file and returns the information of the genes, excluding
    artifacts

    Parameters
    ----------
    gene_file : path
        Path to the file that contains the genes
    name : str
        Name of the project
    extend : int
        Extend of the region to be analyzed
    resolution : int
        Resolution at which to split the genome

    Returns
    -------
    id_tss_f : dict
        Dictionary linking the gene id to the tss
    id_chrom_f : dict 
        Dictionary linking the gene id to the chromosome
    id_name_f : dict
        Dictionary linking the gene id to the name
    fn_f : str
        Name of the end file
    """

    gene_type_count = defaultdict(int)

    if gene_file_f.endswith("gz"):

        myopen = gzip.open

    else:

        myopen = open

    with myopen(gene_file_f, mode="rt", encoding="utf-8") as gtf_reader:

        for line in gtf_reader:

            if re.compile("##").search(line) or not re.compile(r"chr\w+").search(line):

                continue

            line_s = line.split("\t")

            if line_s[2] == "gene" and line_s[0] != "chrM":

                if re.compile(r'gene_type "artifact";').search(line):

                    continue

                gene_type_count[re.compile(r'gene_type "(\w+)";').search(line).group(1)] += 1

    print("Index: 0; All genes")

    for type_index, type_name in enumerate(gene_type_count.keys()):

        print(f"Index: {type_index+1}; {type_name}")

    ch_index = None

    while not isinstance(ch_index, int) or ch_index not in range(len(gene_type_count.keys()) + 1):

        ch_index = input("Choose the gene type to parse in the final file: ")

        try:

            ch_index = int(ch_index)

        except ValueError:

            print("Please, enter a number.")
            continue

        if ch_index not in range(len(gene_type_count.keys()) + 1):

            print(f"Please, enter a number between 0 and {len(gene_type_count.keys())}")
            continue

    if ch_index == 0:

        ch_type_pat = re.compile(r'gene_type "\w+";')
        fn_f = f"{name}_all_{extend}_{resolution}_agg.txt"
        print("Parsing all genes...")

    else:

        ch_type_pat = re.compile(f'gene_type "{list(gene_type_count.keys())[ch_index - 1]}";')
        fn_f = f"{name}_{list(gene_type_count.keys())[ch_index - 1]}_{extend}_{resolution}_gene_coords.txt"
        print(f"Gene type chosen: {list(gene_type_count.keys())[ch_index - 1]}")

    id_tss_f = defaultdict(int)
    id_name_f = defaultdict(str)
    id_chrom_f = defaultdict(str)

    print("Gathering information from the annotation file...")

    with myopen(gene_file_f, mode="rt", encoding="utf-8") as gtf_reader:

        for line in gtf_reader:

            if re.compile("##").search(line) or not re.compile(r"chr\w+").search(line):

                continue

            line_s = line.split("\t")

            if line_s[2] == "gene" and line_s[0] != "chrM" and ch_type_pat.search(line):

                gene_id = re.compile(r'gene_id "(ENS\w+)\.\d+"').search(line).group(1)
                gene_name = re.compile(r'gene_name "([\w\-\_\.]+)";').search(line).group(1)
                id_chrom_f[gene_id] = line_s[0]
                id_name_f[gene_id] = gene_name

                if line_s[6] == "+":

                    id_tss_f[gene_id] = int(line_s[3])

                else:

                    id_tss_f[gene_id] = int(line_s[4])

    return id_chrom_f, id_tss_f, id_name_f, fn_f


def bedparser(gene_file_f: Path, name: str, extend: int, resolution: int) -> tuple[dict, dict, dict, str]:
    """
    Parses a bed file and returns the information of the genes, excluding
    artifacts.

    Parameters
    ----------
    gene_file : path
        Path to the file that contains the genes
    name : str
        Name of the project
    extend : int
        Extend of the region to be analyzed
    resolution : int
        Resolution at which to split the genome

    Returns
    -------
    id_tss_f : dict
        Dictionary linking the gene id to the tss
    id_chrom_f : dict
        Dictionary linking the gene id to the chromosome
    id_name_f : dict
        Dictionary linking the gene id to the name
    fn_f : str
        Name of the end file
    """

    id_tss_f = defaultdict(int)
    id_name_f = defaultdict(str)
    id_chrom_f = defaultdict(str)

    fn_f = f"{name}_all_{extend}_{resolution}_agg.txt"

    print("Gathering information from the annotation file...")

    bed_file = pd.read_table(gene_file_f, names=["coords", "symbol", "id"])

    for _, row_f in bed_file.iterrows():

        id_chrom_f[row_f.id] = row_f.coords.split(":")[0]
        id_tss_f[row_f.id] = int(row_f.coords.split(":")[1])
        id_name_f[row_f.id] = row_f.symbol

    return id_chrom_f, id_tss_f, id_name_f, fn_f


def binsearcher(id_tss_f: dict, id_chrom_f: dict, id_name_f: dict, bin_genome_f: pd.DataFrame) -> pd.DataFrame:
    """
    Searches the bin index where the gene is located    

    Parameters
    ----------
    id_tss_f : dict
        Dictionary linking the gene id to the tss
    id_chrom_f : dict
        Dictionary linking the gene id to the chromosome
    id_name_f : dict
        Dictionary linking the gene id to the name
    bin_genome_f : pd.DataFrame
        DataFrame containing the bins of the genome

    Returns
    -------
    data_f : pd.DataFrame
        DataFrame containing the information of the genes and the bin index
    """

    data_f = defaultdict(list)

    for g_id, tss_pos in id_tss_f.items():

        chrom_bin_f = bin_genome_f[(bin_genome_f["chrom"] == id_chrom_f[g_id])].reset_index()

        sub_bin_index = chrom_bin_f[(chrom_bin_f["start"] <= tss_pos) &
                                    (chrom_bin_f["end"] >= tss_pos)].index.tolist()[0]

        data_f["chrom"].append(id_chrom_f[g_id])
        data_f["bin_index"].append(sub_bin_index)
        data_f["gene_name"].append(id_name_f[g_id])
        data_f["gene_id"].append(g_id)

    print("Aggregating genes that are located in the same bin...")

    data_f = pd.DataFrame(data_f)
    data_f = data_f.groupby(["chrom", "bin_index"])[data_f.columns].agg(",".join).reset_index()

    data_f[["chrom"]] = data_f[["chrom"]].astype(str)
    data_f[["bin_index"]] = data_f[["bin_index"]].astype(int)
    data_f[["gene_name"]] = data_f[["gene_name"]].astype(str)
    data_f[["gene_id"]] = data_f[["gene_id"]].astype(str)

    return data_f


def meta_param_search(work_dir_f: str, hic_f: str, reso_f: int, reso_file: str, pl_f: list, cutoff_f: float,
                      sample_num_f: int, seed_f: int):
    """
    Test MetaLoci parameters to optimise Kamada-Kawai layout.

    Parameters
    ----------
    work_dir_f : str
        Path to working directory.
    hic_f : str
        Complete path to the cool/mcool/hic file.
    reso_f : int
        Hi-C resolution to be tested (in bp).
    reso_file : str
        Path to region file to be tested.
    pl_f : _type_
        Persistent length; amount of possible rotation of points in layout.
    cutoff_f : float
        Percent of top interactions to use from HiC.
    sample_num_f : int
        Number of regions to sample from the region file.
    seed_f : int
        Random seed for region sampling.
    """

    save_dir_f = os.path.join(work_dir_f, f"reso_{reso_f}_cutoff_{cutoff_f*100:.0f}_pl_{pl_f}")
    Path(save_dir_f).mkdir(parents=True, exist_ok=True)

    ml_comm = f"metaloci layout -w {save_dir_f} -c {hic_f}" + " -r {} -g {} --cutoff {} --pl {} --plot > /dev/null 2>&1"

    regions = pd.read_table(reso_file)
    sample = regions.sample(sample_num_f, random_state=seed_f)

    for region_coords in sample['coords']:
        com2run = ml_comm.format(reso_f, region_coords, cutoff_f, pl_f)
        sp.check_call(com2run, shell=True)


def get_poi_data(
        line_f: pd.Series, signals_f: list, work_dir_f: str,
        bf_f: str, of: str,
        pval: float, quadrant_list: str,
        region_file_f: bool = False, rf_h=None):
    """
    Function to extract data from the METALoci objects and parse it into a table.

    Parameters
    ----------
    line_f : pd.Series
        Row of the gene file to parse.
    signals_f : list
        List of signals to parse.
    work_dir_f : str
        Path to working directory.
    bf_f : str
        Bad file name.
    of : str
        Output file name.
    pval : float
        P-value threshold.
    quadrant_list : list
        List of quadrants to consider.
    region_file_f : bool, optional
        Select whether or not to store a metaloci region file with the significant regions.
    rf_h : TextIOWrapper, optional
        Region file name.

    Returns
    -------
    None
    """

    mlo_fn = f"{line_f.coords.replace(':', '_').replace('-', '_')}.mlo"

    bfh_f = open(bf_f, mode="a", encoding="utf-8")

    if not os.path.exists(os.path.join(work_dir_f, mlo_fn.split("_")[0], mlo_fn)):

        bfh_f.write(f"{line_f.coords}\t{line_f.symbol}\t{line_f.id}\tno_file\n")
        bfh_f.flush()
        return None

    table_line_f = f"{line_f.coords}\t{line_f.symbol}\t{line_f.id}"

    try:

        mlo_data_f = pd.read_pickle(os.path.join(work_dir_f, mlo_fn.split("_")[0], mlo_fn))

    except UnpicklingError:

        bfh_f.write(f"{line_f.coords}\t{line_f.symbol}\t{line_f.id}\tcorrupt_file\n")
        bfh_f.flush()
        return None

    if mlo_data_f.bad_region:

        bfh_f.write(f"{line_f.coords}\t{line_f.symbol}\t{line_f.id}\t{mlo_data_f.bad_region}\n")
        bfh_f.flush()
        return None

    of_h = open(of, mode="a", encoding="utf-8")

    for sig_f in signals_f:

        try:

            poi_data_f = mlo_data_f.lmi_info[sig_f][mlo_data_f.lmi_info[sig_f]["bin_index"] == mlo_data_f.poi]
            poi_lmi_f = poi_data_f.LMI_score.squeeze()
            poi_quadrant_f = poi_data_f.moran_quadrant.squeeze()
            poi_pval_f = poi_data_f.LMI_pvalue.squeeze()
            poi_signal_f = poi_data_f.ZSig.squeeze()
            poi_lag_f = poi_data_f.ZLag.squeeze()

            table_line_f += f"\t{poi_quadrant_f}\t{poi_lmi_f}\t{poi_pval_f}\t{poi_signal_f}\t{poi_lag_f}"

            if poi_quadrant_f in quadrant_list and poi_pval_f <= pval and region_file_f:

                regionfile_h = open(rf_h, mode="a", encoding="utf-8")
                regionfile_h.write(f"{table_line_f}\n")
                regionfile_h.flush()

        except KeyError:

            bfh_f.write(f"{line_f.coords}\t{line_f.symbol}\t{line_f.id}\tno_signal_{sig_f}\n")
            bfh_f.flush()

    of_h.write(f"{table_line_f}\n")
    of_h.flush()

    return None
