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
import hicstraw


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


def clean_matrix(mlobject: mlo.MetalociObject) -> np.ndarray:
    """
    Clean a given HiC matrix. It checks if the matrix has too many zeroes at
    he diagonal, removes values that are zero at the diagonal but are not in
    the rest of the matrix, adds pseudocounts to zeroes depending on the min
    value, scales all values depending on the min value and computes the log10
    off all values.

    Parameters
    ----------
    mlo : np.ndarray
        METALoci object with a matrix in it.

    Returns
    -------
    np.ndarray
        Clean matrix.
    """

    diagonal = np.array(mlobject.matrix.diagonal())
    total_zeroes, max_stretch, percentage_zeroes, zero_loc = check_diagonal(diagonal)

    if total_zeroes == len(diagonal):

        mlobject.bad_region = "empty"
        return mlobject

    if percentage_zeroes >= 50:

        mlobject.bad_region = "too many zeros"

    if max_stretch >= 20:

        mlobject.bad_region = "stretch"

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

    return mlobject


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

        # signal = [0.0 if np.isnan(index) else index for index in region_signal]
        median_default = np.nanmedian(region_signal) # jfk fix
        signal = [median_default if np.isnan(index) else index for index in region_signal] # jfk fix

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


def check_cooler_names(hic_file: Path, data: Path, coords: bool):

    with open(data[0], "r") as handler:

        if [line.strip() for line in handler][10].startswith("chr"):

            signal_chr_nom = "chrN"

        else:

            signal_chr_nom = "N"

    hic_file = cooler.Cooler(hic_file)
    chrom_list = hic_file.chromnames

    if "chr" in chrom_list[0]:

        cooler_chr_nom = "chrN"

    else:

        cooler_chr_nom = "N"
    
    del hic_file

    with open(coords, "r") as handler:

        if [line.strip() for line in handler][0].startswith("chr"):

            coords_chr_nom = "chrN"

        else:

            coords_chr_nom = "N"

    if not signal_chr_nom == cooler_chr_nom == coords_chr_nom:

        exit(
            "\nThe signal, cooler and chromosome sizes files do not have the same nomenclature for chromosomes:\n"
            f"\n\tSignal chromosomes nomenclature is '{signal_chr_nom}'. "
            f"\n\tHi-C chromosomes nomenclature is '{cooler_chr_nom}'. "
            f"\n\tChromosome sizes nomenclature is '{coords_chr_nom}'. "
            "\n\nPlease, rename the chromosome names. "
            "\nYou may want to rename the chromosome names in a cooler file with cooler.rename_chroms() in python. "
            "\n\nExiting..."
        )
    
    return chrom_list


def check_hic_names(hic_file: Path, data: Path, coords: bool):

    with open(data[0], "r") as handler:

        if [line.strip() for line in handler][0].startswith("chr"):

            signal_chr_nom = "chrN"

        else:

            signal_chr_nom = "N"
    
    chrom_list = [i.name for i in hicstraw.HiCFile(hic_file).getChromosomes()][1:]

    if "chr" in chrom_list[0]:

        hic_chr_nom = "chrN"

    else:

        hic_chr_nom = "N"

    del hic_file

    with open(coords, "r") as handler:

        if [line.strip() for line in handler][0].startswith("chr"):

            coords_chr_nom = "chrN"

        else:

            coords_chr_nom = "N"

    if not signal_chr_nom == hic_chr_nom == coords_chr_nom:

        exit(
            "\nThe signal, cooler and chromosome sizes files do not have the same nomenclature for chromosomes:\n"
            f"\n\tSignal chromosomes nomenclature is '{signal_chr_nom}'. "
            f"\n\tHi-C chromosomes nomenclature is '{hic_chr_nom}'. "
            f"\n\tChromosome sizes nomenclature is '{coords_chr_nom}'. "
            "\n\nPlease, rename the chromosome names. "
            "\n\nExiting..."
        )

    return chrom_list

def natural_sort(list: list): 

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    
    return sorted(list, key=alphanum_key)