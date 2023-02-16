__author__ = "Iago Maceda Porto and Leo Zuber Ponce"

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re
from collections import defaultdict
from time import time



def delete_folder(path) :

    for subdirectory in path.iterdir() :

        if subdirectory.is_dir() :

            delete_folder(subdirectory)

        else :

            subdirectory.unlink()
    path.rmdir()


def bed_to_metaloci(data, coords, resolution):

    """_summary_

    Returns:
        _type_: _description_
    """

    boundaries_dictionary = defaultdict(dict)

    # Open centromeres and telomeres coordinates file and assign the corresponding values to variables.
    with open(file=coords, mode='r', encoding="utf-8") as chrom:

        for l in chrom:

            line = l.rstrip().split("\t")
            boundaries_dictionary[line[0]]['end'] = int(line[1])  # tl_st: end of the initial telomere

    file_info = pd.read_table(data[0])

    for i in range(1, len(data)):

        temp = pd.read_table(data[i])
        file_info = pd.merge(file_info, temp, on=["chrom", "start", "end"], how='inner')

    col_names = file_info.columns.tolist()

    bad_col_names = []

    for i in range(3, len(col_names)):

        if len(re.findall("_", col_names[i])) != 1:

            bad_col_names.append(col_names[i])

    if len(bad_col_names) > 0:

        print("Problems with the following signal names:")
        print(', '.join(str(x) for x in bad_col_names))
        print("Names for signal must be in the following format: CLASS_NAME.")
        print("Class refers to data that can be potentially merged in downstream analysis.")
        
        sys.exit("Exiting due to improper signal names.")

    for chrm, chrm_value in boundaries_dictionary.items():

        print(f"chromosome {chrm.rsplit('r', 1)[1]} in progress.")

        pbar = tqdm(total = int(chrm_value["end"] / resolution ) + 1)

        if chrm not in file_info["chrom"].unique():

            print(f"chromosome {chrm.rsplit('r', 1)[1]} not found in the signal file(s), skipping...")
            continue

        bin_start = 0
        bin_end = bin_start + resolution

        info = defaultdict(list)

        while bin_start <= chrm_value['end']:

            pbar.update()

            info["Chr"].append(chrm)
            info["St"].append(bin_start)
            info["Sp"].append(bin_end)

            tmp_bin = file_info[(file_info['start'] >= int(bin_start)) &
                            (file_info['end'] <= int(bin_end)) &
                            (file_info['chrom'] == chrm)]

            # Go over the columns to get all the signals.
            for j in range(3, tmp_bin.shape[1]):

                if tmp_bin.shape[0] == 0:

                    info[col_names[j]].append(np.nan)

                else:

                    info[col_names[j]].append(np.nanmedian(tmp_bin.iloc[:, j].tolist()))

            # If the end of the current bin is the start of the terminal telomere, stop
            if bin_end == boundaries_dictionary[f"{chrm}"]['end']:

                break

            # Creating tmp variables for an easier check of overlap with centromeres and telomeres.
            bin_start = bin_end
            bin_end = bin_start + resolution

        pbar.close()

    return info