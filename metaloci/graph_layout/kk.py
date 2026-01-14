"""
Functions to calculate the top interactions matrix subset, plot matrix and get restraints for the Kamada-Kawai layout
processing.
"""
import random

import networkx as nx
import numpy as np
from metaloci import mlo
from scipy.sparse import csr_matrix


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

            perc_temp = top / len(mlobject.flat_matrix[mlobject.flat_matrix > np.nanmin(mlobject.flat_matrix)])

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

    if mlobject.persistence_length is None: # Neither -l nor -i arguments set, calculate persistence length the old way

        mlobject.persistence_length = np.nanquantile(subset_matrix[subset_matrix > 0], 0.99) ** 2
        subset_matrix[rng, rng + 1] = mlobject.persistence_length  # Add persistence length to bins next to diagonal
    
    # -i flag set, use a tmp persistence length to calculate the subset matrix
    # This persistence length is for the cut-off optimisation
    # The proper persistence length will be calculated once the cut-off has been estimated
    elif mlobject.persistence_length == "optimise": 
    
        tmp_persistence_length = np.nanquantile(subset_matrix[subset_matrix > 0], 0.99) ** 2
        subset_matrix[rng, rng + 1] = tmp_persistence_length 

    else: # -l argument used, use the given persistence length to calculate the subset matrix

        subset_matrix[rng, rng + 1] = mlobject.persistence_length
    
    if not silent and mlobject.persistence_length != "optimise":
        
        print(f"\tPersistence length = {mlobject.persistence_length:.4f}")

    subset_matrix[0, 0] = 0
    subset_matrix[rng + 1, rng + 1] = 0  # Remove diagonal
    mlobject.subset_matrix = subset_matrix

    return mlobject


def estimate_cutoff(mlobject: mlo.MetalociObject, optimise : bool = False) -> float:
    """
    Estimate the cutoff for the Kamada-Kawai layout processing. It will iteratively try several cutoffs until it finds
    the one that satisfies that the number of conections the node with the least amount of connections is 25% of
    the node with higher amount of connections in the graph.

    Parameters
    ----------

    mlobject : mlo.MetalociObject
        METALoci object with a matrix in it (MetalociObject.matrix).

    Returns
    -------
    float
        Estimated cutoff for the Kamada-Kawai layout processing.
    """

    # Set the initial interval for the cutoff, and counter for early stopping
    min_cut = 0.1
    max_cut = 0.4
    counter = 0

    # We first compute a hypothetical cutoff to get the a maximum amount of connections.
    mlobject.kk_cutoff["values"] = 0.25
    mlobject = get_restraints_matrix(mlobject, optimise, silent=True)
    
    try: # As it can fail if restraints matrix is empty

        mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject.kk_restraints_matrix))

    except:

        return [0.2] # Default
    
    # Get the degrees of the nodes (number of connections each nodes has)
    degrees = []

    for i in range(1, len(mlobject.kk_graph)):
        
        degrees.append(mlobject.kk_graph.degree(i))

    degrees = [x for x in degrees if x != 2]
    degrees = [x for x in degrees if x != 1]
    degrees = [x for x in degrees if x != 0]

    # remove top 1% values to avoid outliers
    degrees = np.sort(degrees)[:-int(len(degrees) * 0.01)]

    try:

        max_degrees = int(np.max(degrees))

    except ValueError:
        
        return [0.2] # Default
    
    min_degrees = 0
    target = int(max_degrees * 0.25) 

    # The number of connections the node with the least amount of connections has to be 25% of the node with highest 
    # amount of connections. We find the cut-off that satisfies this condition.
    while int(min_degrees) != target:

        mlobject.kk_cutoff["values"] = round(random.uniform(min_cut, max_cut), 3) # Random cut-off
        mlobject = get_restraints_matrix(mlobject, silent=True)

        try: # As it can fail if restraints if the cut-off is too extreme

            mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject.kk_restraints_matrix))

        except:

            counter += 1
            continue

        degrees = []

        for i in range(1, len(mlobject.kk_graph)):
            
            degrees.append(mlobject.kk_graph.degree(i))

        degrees = [x for x in degrees if x != 2] # As this is always the minimum, out
        degrees = [x for x in degrees if x != 1] # First or last without connections, out
        degrees = [x for x in degrees if x != 0] # Empty bins, out
        min_degrees = int(np.min(degrees))

        # binary search
        if min_degrees > target:

            max_cut = mlobject.kk_cutoff["values"] + 0.02

        elif min_degrees < target:

            min_cut = mlobject.kk_cutoff["values"] - 0.02

        counter += 1

        if counter > 40 and min_degrees != target: # If it does not find it in 40 tries, it will return the default.
            
            return [0.2] # Default
    
    mlobject.kk_cutoff["values"] = [mlobject.kk_cutoff["values"]]

    return mlobject.kk_cutoff["values"]


def estimate_persistence_length(mlobject: mlo.MetalociObject, optimise : bool = False) -> float:
    """
    Estimate the persistence length for the Kamada-Kawai layout processing.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix in it (MetalociObject.matrix).

    Returns
    -------
    float
        Estimated persistence length for the Kamada-Kawai layout processing.
    """
    original_cutoff = mlobject.kk_cutoff["values"] # We save this to restore it as get_restraints_matrix modifies it.
    mlobject.kk_cutoff["values"] = mlobject.kk_cutoff["values"][0]
    mlobject = get_restraints_matrix(mlobject, optimise, silent=True)
    mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject.kk_restraints_matrix))
    
    # Get the degrees of the nodes (number of connections each nodes has)
    degrees = [] 

    for i in mlobject.kk_graph.nodes():
            
        degrees.append(mlobject.kk_graph.degree(i))

        degrees = [x for x in degrees if x != 2] # As this is always the minimum, out
        degrees = [x for x in degrees if x != 1] # First or last without connections, out
        degrees = [x for x in degrees if x != 0] # Empty bins, out

    try:

        mean_degrees = int(np.mean(degrees))

    except ValueError:

        mlobject.kk_cutoff["values"] = original_cutoff

        return None
    
    scaling_factor = 0.8 + (200 / len(mlobject.kk_graph)) # This depends on the size of the graph, scales decently.
    persistence_length = np.sqrt(mean_degrees) * scaling_factor
    mlobject.kk_cutoff["values"] = original_cutoff
    
    return persistence_length