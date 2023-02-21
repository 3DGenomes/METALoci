__author__ = "Iago Maceda Porto and Leo Zuber Ponce"

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re
from collections import defaultdict
import sys


def remove_folder(path):

    """
    Removes a folder that is not empty.

    :param path: Folder to remove
    :type path: Path
    """

    for subdirectory in path.iterdir():

        if subdirectory.is_dir():

            remove_folder(subdirectory)

        else:

            subdirectory.unlink()

    path.rmdir()


def check_diagonal(diag_mat):

    """
    This function checks the 0s on the diagonal.

    :param diag_mat: The diagonal of the HiC matrix
    :type diag_mat: ndarray
    :return: total = total number of zeroes, max_stretch = max stretch of zeroes,
    percentage_zeroes = percentage of zeroes, zero_loc = localization of 0 in the
    diagonal.
    :rtype: _type_
    """

    total = 0
    stretch = 0
    max_stretch = 0
    zero_loc = []

    for i_diag, element in enumerate(diag_mat):

        if element == 0:

            total += 1
            stretch += 1
            zero_loc.append(i_diag)

            max_stretch = max(max_stretch, stretch)

        else:

            stretch = 0

    percentage_zeroes = np.round(total / len(diag_mat) * 100, decimals=2)

    return total, max_stretch, percentage_zeroes, zero_loc


def clean_matrix(mlo, bad_regions):
    """
    Clean a given HiC matrix. It checks if the matrix has too many zeroes at
    he diagonal, removes values that are zero at the diagonal but are not in
    the resto of the matrix, adds pseudocounts to zeroes depending on the min
    value, scales all values depending on the min value and computes the log10
    off all values.

    :param mlo: METALoci object with a matrix in it.
    :type mlo: np.array
    :param bad_regions: Dictionay {"region": [], "reason": []} in which to append bad regions.
    :type bad_regions: dict
    :return: Clean matrix.
    :rtype: np.array
    """

    diagonal = np.array(mlo.matrix.diagonal())
    total_zeroes, max_stretch, percentage_zeroes, zero_loc = check_diagonal(diagonal)

    if total_zeroes == len(diagonal):

        print("\tMatrix is empty; passing to the next region")
        bad_regions["region"].append(mlo.region)
        bad_regions["reason"].append("empty")

        return None

    if int(percentage_zeroes) >= 50:

        bad_regions["region"].append(mlo.region)
        bad_regions["reason"].append("perc")

    elif int(max_stretch) >= 50:

        bad_regions["region"].append(mlo.region)
        bad_regions["reason"].append("stretch")

    mlo.matrix[zero_loc] = 0
    mlo.matrix[:, zero_loc] = 0

    # Pseudocounts if min is zero
    if np.nanmin(mlo.matrix) == 0:

        pc = np.nanmin(mlo.matrix[mlo.matrix > 0])
        # print(f"\t\tPseudocounts: {pc}")
        mlo.matrix = mlo.matrix + pc

    # Scale if all below 1
    if np.nanmax(mlo.matrix) <= 1 or np.nanmin(mlo.matrix) <= 1:

        sf = 1 / np.nanmin(mlo.matrix)
        # print(f"\t\tScaling factor: {sf}")
        mlo.matrix = mlo.matrix * sf

    if np.nanmin(mlo.matrix) < 1:

        mlo.matrix[mlo.matrix < 1] = 1

    mlo.matrix = np.log10(mlo.matrix)

    return mlo.matrix


def bed_to_metaloci(data, coords, resolution):
    """
    _summary_

    :param data: _description_
    :type data: _type_
    :param coords: _description_
    :type coords: _type_
    :param resolution: _description_
    :type resolution: _type_
    :return: _description_
    :rtype: _type_
    """

    boundaries_dictionary = defaultdict(dict)

    # Open centromeres and telomeres coordinates file and assign the corresponding values to variables.
    with open(file=coords, mode="r", encoding="utf-8") as chrom:

        for l in chrom:

            line = l.rstrip().split("\t")
            boundaries_dictionary[line[0]]["end"] = int(
                line[1]
            )  # tl_st: end of the initial telomere

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

            print(
                f"chromosome {chrm.rsplit('r', 1)[1]} not found in the signal file(s), skipping..."
            )
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
