"""
Functions to calculate the top interactions matrix subset, plot matrix and get restraints for the Kamada-Kawai layout
processing.
"""
import numpy as np
from metaloci import mlo


def get_restraints_matrix(mlobject: mlo.MetalociObject, optimise: bool = False, silent: bool = False) -> mlo.MetalociObject:
    """
    Calculate top interaction matrix subset, plot matrix and get restraints.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix and a cutoff (MetalociObject.matrix and MetalociObject.kk_cutoff) in it.
        MetalociObject.persistence_length is optional.
    silent : bool, optional
        Variable passed to get_subset_matrix function to control the verbosity of it (useful for multiprocessing).

    Returns
    -------
    mlobject : mlo.MetaloiObject
        METALoci object with the calculated restraints for the Kamada-Kawai layout processing, the 'mixed matrix'
        (upper diagonal is the original HiC, lower diagonal is the subset matrix), the flattened matrix and the 
        top indexes of the subset matrix added to it, for plotting purposes.
    """

    # Get subset matrix
    mlobject = get_subset_matrix(mlobject, optimise, silent)

    if mlobject.subset_matrix is None:

        mlobject.kk_restraints_matrix = None

        return mlobject

    # Modify the matrix and transform to restraints
    restraints_matrix = np.where(mlobject.subset_matrix == 0, np.nan, mlobject.subset_matrix)  # Remove zeroes
    restraints_matrix = 1 / restraints_matrix  # Convert to distance matrix instead of similarity matrix
    restraints_matrix = np.triu(restraints_matrix, k=0)  # Remove lower triangle
    restraints_matrix = np.nan_to_num(restraints_matrix, nan=0, posinf=0, neginf=0)  # Clean nans and infs

    mlobject.kk_restraints_matrix = restraints_matrix

    return mlobject


def get_subset_matrix(mlobject: mlo.MetalociObject, optimise: bool = False, silent=False) -> np.ndarray:
    """
    Get a subset of the Hi-C matrix with the top contact interactions in the matrix, defined by a cutoff.
    The diagonal is also removed.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix and a cutoff in it (MetalociObject.matrix and
        MetalociObject.kk_cutoff respectively).
    silent : boolean
        Variable that controls the verbosity of the function (useful for multiprocessing).

    Returns
    -------
    subset_matrix : np.ndarray
        Subset matrix, containing only top interactions. The rest of the matrix is set to 0.

    Notes
    -----
    This function is called from:

    get_restraints_matrix()

    so it is not required to call it when computing the regular pipeline.
    """

    if mlobject.kk_cutoff is None:

        if not silent:

            print(
                f"\tMETALoci object {mlobject.region} does not have a cutoff. \
                    Set the cutoff in 'mlobject.kk_cutoff' first."
            )

        return None

    mlobject.flat_matrix = mlobject.matrix.copy().flatten()

    if mlobject.kk_cutoff["cutoff_type"] == "percentage":

        # Calculating the top interactions of and subsetting the matrix to get those.
        top = int(len(mlobject.flat_matrix[mlobject.flat_matrix > np.nanmin(
            mlobject.flat_matrix)]) * mlobject.kk_cutoff["values"])

        if not silent:

            print(
                f"\tCut-off = {sorted(mlobject.flat_matrix, reverse = True)[top]:.4f} | Using top: {round(mlobject.kk_cutoff['values'] * 100, ndigits=2)}% highest interactions")

    elif mlobject.kk_cutoff["cutoff_type"] == "absolute":

        # Get the interaction values that are higher than the cutoff
        top = int(len(mlobject.flat_matrix[mlobject.flat_matrix > mlobject.kk_cutoff["values"]]))

        if not silent:

            perc_temp = top/len(mlobject.flat_matrix[mlobject.flat_matrix > np.nanmin(mlobject.flat_matrix)])

            print(
                f"\tCut-off = {mlobject.kk_cutoff['values']:.4f} | Using top: {round(perc_temp * 100, ndigits=2)}% highest interactions")

    if top < len(np.diag(mlobject.matrix)):

        if not silent:

            print(f"\tCut-off is too high for {mlobject.region}. Try lowering it.")

        mlobject.bad_region = "cut-off"

        if not optimise:
                
            mlobject.subset_matrix = None

            return mlobject

    mlobject.kk_top_indexes = np.argpartition(mlobject.flat_matrix, -top)[-top:]

    # Subset to cutoff percentile
    subset_matrix = mlobject.matrix.copy()
    subset_matrix = np.where(subset_matrix == 1.0, 0, subset_matrix)
    subset_matrix[subset_matrix < np.nanmin(mlobject.flat_matrix[mlobject.kk_top_indexes])] = 0

    # rng = range of integers until size of matrix to locate the diagonal
    rng = np.arange(len(subset_matrix) - 1)

    if mlobject.persistence_length is None:

        mlobject.persistence_length = np.nanquantile(subset_matrix[subset_matrix > 0], 0.99) ** 2

    subset_matrix[rng, rng + 1] = mlobject.persistence_length  # Add persistence length to bins next to diagonal
    subset_matrix[0, 0] = 0
    subset_matrix[rng + 1, rng + 1] = 0  # Remove diagonal

    mlobject.subset_matrix = subset_matrix

    return mlobject
