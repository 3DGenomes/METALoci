"""
Class for the METALoci object
"""

import re

import dill as pickle


class MetalociObject:
    """
    This object is base class of METALoci. It contains all the information for a given region.

    Parameters
    ----------
    region : str
        Region of interest
    resolution : int
        Resolution of the HiC experiment
    persistence_length : float
        Persistence length (parameter needed for the Kamada-Kawai algorithm)
    save_path : str
        Path where to save the mlo
    chrom : str, optional
        Chromosome of the region
    start : int, optional
        Start of the region
    end : int, optional
        End of the region
    poi : int, optional
        Bin position of the point of interest (usually the TSS of the gene)
    resolution : int
        Resolution of the Hi-C to use.
    persistence_length : float
        Persistence length to use.
    save_path : str
        Path to save the object.
    chrom : str, optional
        Chromosome of the region.
    start : int, optional
        Start of the region.
    end : int, optional
        End of the region.
    poi : int, optional
        Point of interest of the region.
    matrix : numpy.ndarray, optional
        Hi-C matrix for the region.
    subset_matrix : numpy.ndarray, optional
        Subset of the matrix.
    bad_region : bool, optional
        Bool to indicate if the region does not meet quelity criteria to be considered significant.
    mixed_matrices : dict, optional
        Dictionary with the mixed matrices (top half if the full matrix, bottom part is the subset matrix).
    signals_dict : dict, optional
        Dictionary with the signals for the region.
    flat_matrix : numpy.ndarray, optional
        Flattened matrix.
    kk_cutoff : dict, optional
        Dictionary with the cutoffs for the Kamada-Kawai algorithm.
    kk_restraints_matrix : numpy.ndarray, optional
        Matrix with the restraints to be used by the Kamada-Kawai algorithm.
    kk_top_indexes : numpy.ndarray, optional
        Top indexes of the Kamada-Kawai algorithm, used to build the subset matrix.
    kk_graph : networkx.Graph, optional
        Graph of the Kamada-Kawai algorithm.
    kk_nodes : numpy.ndarray, optional
        Nodes of the Kamada-Kawai algorithm.
    kk_coords : numpy.ndarray, optional
        Coordinates of the Kamada-Kawai algorithm.
    kk_distances : numpy.ndarray, optional
        Distances of the Kamada-Kawai algorithm.
    agg : numpy.ndarray, optional
        Aggregated signals.
    lmi_geometry : numpy.ndarray, optional
        Geometry of the Gaudí plot.
    lmi_info : dict, optional
        Information of the the Local Moran's I output for this region.
    """

    def __init__(self, region: str, resolution: int, persistence_length: float, save_path: str,
                 chrom: str = None, start: int = None, end: int = None, poi: int = None):
        """
        
        """

        self.region = region

        if chrom is None:

            self.chrom, _, _, _ = re.split(r":|-|_", region)

        else:

            self.chrom = chrom

        if start is None:

            _, temp, _, _ = re.split(r":|-|_", region)
            self.start = int(temp)

        else:

            self.start = start

        if end is None:

            _, _, temp, _ = re.split(r":|-|_", region)
            self.end = int(temp)

        else:

            self.end = end

        if poi is None:

            _, _, _, temp = re.split(r":|-|_", region)
            self.poi = int(temp)

        else:

            self.poi = poi

        self.resolution = resolution
        self.persistence_length = persistence_length
        self.save_path = save_path
        self.subset_matrix = None
        self.matrix = None
        self.bad_region = None
        self.mixed_matrices = None
        self.signals_dict = None
        self.flat_matrix = None
        self.kk_cutoff = {}
        self.kk_restraints_matrix = None
        self.kk_top_indexes = None
        self.kk_graph = None
        self.kk_nodes = None
        self.kk_coords = None
        self.kk_distances = None
        self.agg = None
        self.lmi_geometry = None
        self.lmi_info = {}

    def save(self, file_handler: str):
        """
        Function to save the mlobject.

        Parameters
        ----------
        file_handler : str
            Path to the file name where to save the mlo
        """

        pickle.dump(self, file_handler)
