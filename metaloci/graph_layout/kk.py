import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from metaloci.plot import plot


def get_restraints_matrix(mlobject, persistance_lenght, plot_bool):

    """
    Calculate top interaction, plot matrix and get restraints.

    :param file_name: File name
    :type file_name: str
    :param matrix: HiC matrix
    :type matrix: np.array
    :param persistance_lenght: persistance lenght
    :type persistance_lenght: np.array
    :param cutoff: Top interactions cutoff
    :type cutoff: float
    :param plot_bool: Flag to indicate if with only one cutoff the function still
    needs to plot.
    :type plot_bool: boolean
    :return: Matrix of restraints for KK layout
    :rtype: np.array
    """

    # Get subset matrix
    subset_matrix = get_subset_matrix(mlobject)

    # Mix matrices to plot:
    upper_triangle = np.triu(mlobject.matrix.copy() + 1, k=1)  # Original matrix
    lower_triangle = np.tril(subset_matrix, k=-1)  # Top interactions matrix
    mlobject.mixed_matrices = upper_triangle + lower_triangle

    # Modify the matrix and transform to restraints
    restraints_matrix = np.where(subset_matrix == 0, np.nan, subset_matrix)  # Remove zeroes
    restraints_matrix = (  # Convert to distance Matrix instead of similarity matrix
        1 / restraints_matrix
    )
    restraints_matrix = np.triu(restraints_matrix, k=0)  # Remove lower triangle

    restraints_matrix = np.nan_to_num(  # Clean nans and infs
        restraints_matrix, nan=0, posinf=0, neginf=0
    )

    # Giving some info about the matrix to the user (perhaps implement silent mode to skip all
    # this prints?)
    # print(f"\t\tMatrix size: {len(matrix_copy)}")
    # print(
    #     f"\t\tNumber of non-0 matrix elements: "
    #     f"{int(len(matrix_copy) - len(matrix_copy[matrix_copy <= np.nanmin(matrix_copy)]))}"
    # )
    # print(f"\t\tSize of the new matrix with {top} elements: {len(matrix_copy[top_indexes])}\n")

    # print(
    #     f"\t\tCutoff using the top {int(cutoff * 100)}% interactions: "
    #     f"{np.round(np.nanmin(matrix_copy[top_indexes]), decimals=2)}"
    # )
    # print(f"\t\tSize of the submatrix with the top interations: {len(top_indexes)}\n")

    return restraints_matrix, mlobject


def get_subset_matrix(mlobject):

    if mlobject.kk_cutoff is None:

        print(
            f"METALoci object {mlobject.region} does not have a cutoff. Set the cutoff with mlobject.kk_cutoff' first."
        )
        exit()

    mlobject.flat_matrix = mlobject.matrix.copy().flatten()

    # Calculating the top interactions of and subsetting the matrix to get those.
    # We do not use pseudocounts to calculate the top interactions.
    top = int(
        len(mlobject.flat_matrix[mlobject.flat_matrix > np.nanmin(mlobject.flat_matrix)])
        * mlobject.kk_cutoff
    )
    mlobject.kk_top_indexes = np.argpartition(mlobject.flat_matrix, -top)[-top:]

    # Subset to cutoff percentil
    subset_matrix = mlobject.matrix.copy()
    subset_matrix = np.where(subset_matrix == 1.0, 0, subset_matrix)
    subset_matrix[subset_matrix < np.nanmin(mlobject.flat_matrix[mlobject.kk_top_indexes])] = 0

    if mlobject.persistance_length is None:

        mlobject.persistance_length = np.nanquantile(subset_matrix[subset_matrix > 0], 0.99) ** 2

    rng = np.arange(  # rng = range of integers until size of matrix to locate the diagonal
        len(subset_matrix) - 1
    )
    subset_matrix[rng, rng + 1] = mlobject.persistance_length
    subset_matrix[rng, rng] = 0  # Remove diagonal

    return subset_matrix
