import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re
from collections import defaultdict
import sys
from metaloci import mlo
from pathlib import Path
import cooler
import subprocess as sp


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
    Checks the 0s on the diagonal and saves the total number os zeroes,
    the the max stretch (number of zeros in a row) as a percentage,
    the percentage of zeroes, and the location of zeroes.

    Parameters
    ----------
    diagonal : np.ndarray
        Diagonal of the HiC matrix.

    Returns
    -------
    int
        Total number of zeroes.
    float
        Percentage of max stretch.
    float
        Percentage of zeroes.
    list
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


def clean_matrix(mlobject: mlo.MetalociObject, bad_regions: pd.DataFrame) -> np.ndarray:
    """
    Clean a given HiC matrix. It checks if the matrix has too many zeroes at
    he diagonal, removes values that are zero at the diagonal but are not in
    the resto of the matrix, adds pseudocounts to zeroes depending on the min
    value, scales all values depending on the min value and computes the log10
    off all values.

    Parameters
    ----------
    mlo : np.ndarray
        METALoci object with a matrix in it.
    bad_regions : dict
        Dictionay {"region": [], "reason": []} in which to append bad regions.

    Returns
    -------
    np.ndarray
        Clean matrix.
    """

    diagonal = np.array(mlobject.matrix.diagonal())
    total_zeroes, max_stretch, percentage_zeroes, zero_loc = check_diagonal(diagonal)

    if total_zeroes == len(diagonal):

        # bad_regions[mlobject.region].append(mlobject.region)
        bad_regions[mlobject.region].append("empty")

        return None

    if percentage_zeroes >= 50:

        # bad_regions[mlobject.region].append(mlobject.region)
        bad_regions[mlobject.region].append("percentage_of_zeroes")

    if max_stretch >= 20:

        # bad_regions[mlobject.region].append(mlobject.region)
        bad_regions[mlobject.region].append("stretch")

    mlobject.matrix[zero_loc] = 0
    mlobject.matrix[:, zero_loc] = 0

    # Pseudocounts if min is zero
    if np.nanmin(mlobject.matrix) == 0:

        pc = np.nanmin(mlobject.matrix[mlobject.matrix > 0])
        # print(f"\t\tPseudocounts: {pc}")
        mlobject.matrix = mlobject.matrix + pc

    # Scale if all below 1
    if np.nanmax(mlobject.matrix) <= 1 or np.nanmin(mlobject.matrix) <= 1:

        sf = 1 / np.nanmin(mlobject.matrix)
        # print(f"\t\tScaling factor: {sf}")
        mlobject.matrix = mlobject.matrix * sf

    if np.nanmin(mlobject.matrix) < 1:

        mlobject.matrix[mlobject.matrix < 1] = 1

    mlobject.matrix = np.log10(mlobject.matrix)

    return mlobject.matrix


def signal_normalization(region_signal: pd.DataFrame, pseudocounts: float = None, norm=None) -> np.ndarray:
    """
    Normalize signal values

    Parameters
    ----------
    ind_signal : pd.DataFrame
        Subset of the signal for a given region and for a signal type.
    pseudocounts : float, optional
        Pseudocounts to and if the signal is 0, by default 0.01
    norm : str, optional
        Type of normalization to use.
        Values can be "max" (divide each value by the max value in the signal),
        "sum" (divide each value by the sum of all values),
        or "01" ( value - min(signal) / max(signal) - min(signal) ), by default "01"

    Returns
    -------
    np.ndarray
        Array of normalized signal values for a region and a signal type.
    """

    if pseudocounts is None:

        signal = [0.0 if np.isnan(index) else index for index in region_signal]

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

    return np.array(signal)


def check_chromosome_names(cooler_file: Path, data: Path, coords: bool):

    with open(data[0], "r") as handler:

        if [line.strip() for line in handler][1].startswith("chr"):

            signal_chr_nom = "chrN"

        else:

            signal_chr_nom = "N"

    if "chr" in [cooler_file.chromnames][0][0]:

        cooler_chr_nom = "chrN"

    else:

        cooler_chr_nom = "N"

    with open(coords, "r") as handler:

        if [line.strip() for line in handler][1].startswith("chr"):

            coords_chr_nom = "chrN"

        else:

            coords_chr_nom = "N"

    if not signal_chr_nom == cooler_chr_nom == coords_chr_nom:

        exit(
            "\nThe signal, cooler and chromosome sizes files do not have the same nomenclature for chromosomes:\n"
            f"\n\tSignal chromosomes nomenclature is '{signal_chr_nom}'. "
            f"\n\tCooler chromosomes nomenclature is '{cooler_chr_nom}'. "
            f"\n\tChromosome sizes nomenclature is '{coords_chr_nom}'. "
            "\n\nPlease, rename the chromosome names. "
            "\nYou may want to rename the chromosome names in a cooler file with cooler.rename_chroms() in python. "
            "\n\nExiting..."
        )

    del cooler_file


def bed_to_metaloci(data, coords, resolution):

    boundaries_dictionary = defaultdict(dict)

    # Open centromeres and telomeres coordinates file and assign the corresponding values to variables.
    with open(file=coords, mode="r", encoding="utf-8") as chrom:

        for l in chrom:

            line = l.rstrip().split("\t")
            boundaries_dictionary[line[0]]["end"] = int(line[1])  # tl_st: end of the initial telomere

    file_info = pd.read_table(data[0])

    for i in range(1, len(data)):

        temp = pd.read_table(data[i])
        file_info = pd.merge(file_info, temp, on=["chrom", "start", "end"], how="inner")

    col_names = file_info.columns.tolist()

    bad_col_names = []

    for i in range(3, len(col_names)):

        if len(re.findall("_", col_names[i])) != 1:

            bad_col_names.append(col_names[i])

    if len(bad_col_names) > 0:

        print("Problems with the following signal names:")
        print(", ".join(str(x) for x in bad_col_names))
        print("Names for signal must be in the following format: CLASS_NAME.")
        print("Class refers to data that can be potentially merged in downstream analysis.")

        sys.exit("Exiting due to improper signal names.")

    for chrm, chrm_value in boundaries_dictionary.items():

        print(f"chromosome {chrm.rsplit('r', 1)[1]} in progress.")

        pbar = tqdm(total=int(chrm_value["end"] / resolution) + 1)

        if chrm not in file_info["chrom"].unique():

            print(f"chromosome {chrm.rsplit('r', 1)[1]} not found in the signal file(s), skipping...")
            continue

        bin_start = 0
        bin_end = bin_start + resolution

        info = defaultdict(list)

        while bin_start <= chrm_value["end"]:

            pbar.update()

            info["Chr"].append(chrm)
            info["St"].append(bin_start)
            info["Sp"].append(bin_end)

            tmp_bin = file_info[
                (file_info["start"] >= int(bin_start))
                & (file_info["end"] <= int(bin_end))
                & (file_info["chrom"] == chrm)
            ]

            # Go over the columns to get all the signals.
            for j in range(3, tmp_bin.shape[1]):

                if tmp_bin.shape[0] == 0:

                    info[col_names[j]].append(np.nan)

                else:

                    info[col_names[j]].append(np.nanmedian(tmp_bin.iloc[:, j].tolist()))

            # If the end of the current bin is the start of the terminal telomere, stop
            if bin_end == boundaries_dictionary[f"{chrm}"]["end"]:

                break

            # Creating tmp variables for an easier check of overlap with centromeres and telomeres.
            bin_start = bin_end
            bin_end = bin_start + resolution

        pbar.close()

    return info
