import numpy as np
from metaloci import mlo


def get_restraints_matrix(mlobject: mlo.MetalociObject) -> tuple[np.ndarray, mlo.MetalociObject]:
    """
    Calculate top interaction matrix subset, plot matrix and get restraints.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix and a cutoff (MetalociObject.matrix and MetalociObject.kk_cutoff) in it.
        MetalociObject.persistence_length is optional.
    plot_bool : bool, optional
        Boolean. If true, it will save a Kamada-Kawai of the region in the working directory.
        This is useful when exploring your data. By default True

    Returns
    -------
    restraints_matrix : np.ndarray
        Array with the calculated restraints for the Kamada-Kawai layout processing.
    mlobject : mlo.MetaloiObject
        METALoci object with a 'mixed matrix' (upper diagonal is the original HiC, lower diagonal
        is the subset matrix), the flattened matrix and the top indexes of the subset matrix
        added to it, for plotting purposes.
    """

    # Get subset matrix
    mlobject.subset_matrix = get_subset_matrix(mlobject)

    # Modify the matrix and transform to restraints
    restraints_matrix = np.where(mlobject.subset_matrix == 0, np.nan, mlobject.subset_matrix)  # Remove zeroes
    restraints_matrix = 1 / restraints_matrix  # Convert to distance Matrix instead of similarity matrix
    restraints_matrix = np.triu(restraints_matrix, k=0)  # Remove lower triangle
    restraints_matrix = np.nan_to_num(restraints_matrix, nan=0, posinf=0, neginf=0)  # Clean nans and infs

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


def get_subset_matrix(mlobject: mlo.MetalociObject) -> np.ndarray:
    """
    Get a subset of the Hi-C matrix with the top contact interactions in the matrix, defined by a cutoff.
    The diagonal is also removed.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix and a cutoff in it (MetalociObject.matrix and
        MetalociObject.kk_cutoff respectively).

    Returns
    -------
    subset_matrix : np.ndarray
        Subset matrix, containing only top interactions. The rest of the matrix is set to 0.

    Notes
    -----
    This function is called from::

    get_restraints_matrix()

    so it is not required to call it when computing
    the regular pipeline.
    """

    if mlobject.kk_cutoff is None:

        print(
            f"METALoci object {mlobject.region} does not have a cutoff. "
            "Set the cutoff in 'mlobject.kk_cutoff' first."
        )
        return None

    mlobject.flat_matrix = mlobject.matrix.copy().flatten()

    # Calculating the top interactions of and subsetting the matrix to get those.
    # We do not use pseudocounts to calculate the top interactions.
    top = int(len(mlobject.flat_matrix[mlobject.flat_matrix > np.nanmin(mlobject.flat_matrix)]) * mlobject.kk_cutoff)
    mlobject.kk_top_indexes = np.argpartition(mlobject.flat_matrix, -top)[-top:]

    # Subset to cutoff percentil
    subset_matrix = mlobject.matrix.copy()
    subset_matrix = np.where(subset_matrix == 1.0, 0, subset_matrix)
    subset_matrix[subset_matrix < np.nanmin(mlobject.flat_matrix[mlobject.kk_top_indexes])] = 0

    # rng = range of integers until size of matrix to locate the diagonal
    rng = np.arange(len(subset_matrix) - 1)

    if mlobject.persistence_length is None:

        mlobject.persistence_length = np.nanquantile(subset_matrix[subset_matrix > 0], 0.99) ** 2

    subset_matrix[rng, rng + 1] = mlobject.persistence_length
    subset_matrix[0, 0] = 0
    subset_matrix[rng + 1, rng + 1] = 0  # Remove diagonal

    return subset_matrix
