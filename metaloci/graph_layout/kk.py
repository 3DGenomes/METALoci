import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_restraints_matrix(mlo, persistance_lenght, cutoff, plot_bool):

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

    matrix_copy = mlo.matrix.copy().flatten()

    # Calculating the top interactions of and subsetting the matrix to get those.
    # We do not use pseudocounts to calculate the top interactions.
    top = int(len(matrix_copy[matrix_copy > np.nanmin(matrix_copy)]) * cutoff)
    top_indexes = np.argpartition(matrix_copy, -top)[-top:]

    # Subset to cutoff percentil SQUARE MATRIX
    restraints_matrix = mlo.matrix.copy()
    restraints_matrix = np.where(restraints_matrix == 1.0, 0, restraints_matrix)
    restraints_matrix[restraints_matrix < np.nanmin(matrix_copy[top_indexes])] = 0

    # TO-DO change perlen formula to use quantile 99?

    if persistance_lenght is None:

        persistance_lenght = np.nanmax(restraints_matrix[restraints_matrix > 0]) ** 2

    rng = np.arange(  # rng = range of integers until size of matrix to locate the diagonal
        len(restraints_matrix) - 1
    )
    restraints_matrix[rng, rng + 1] = persistance_lenght
    restraints_matrix[rng, rng] = 0  # Remove diagonal

    if plot_bool == True:

        # Mix matrices to plot:
        upper_triangle = np.triu(mlo.matrix.copy() + 1, k=1)  # Original matrix
        lower_triangle = np.tril(restraints_matrix, k=-1)  # Top interactions matrix
        mixed_matrices = upper_triangle + lower_triangle

        fig_matrix, (ax1, _) = plt.subplots(1, 2, figsize=(20, 5))

        # Plot of the mixed matrix (top triangle is the original matrix,
        # lower triange is the subsetted matrix)
        ax1.imshow(
            mixed_matrices, cmap="YlOrRd", vmax=np.nanquantile(matrix_copy[top_indexes], 0.99)
        )
        ax1.patch.set_facecolor("black")

        # Density plot of the subsetted matrix
        sns.histplot(
            data=matrix_copy[top_indexes].flatten(),
            stat="density",
            alpha=0.4,
            kde=True,
            legend=False,
            kde_kws={"cut": 3},
            **{"linewidth": 0},
        )

        fig_matrix.tight_layout()
        fig_matrix.suptitle(f"Matrix for {mlo.region} (cutoff: {cutoff})")

    # Modify the matrix and transform to restraints
    restraints_matrix = np.where(restraints_matrix == 0, np.nan, restraints_matrix)  # Remove zeroes
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

    return restraints_matrix, mixed_matrices, fig_matrix
