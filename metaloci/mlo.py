"""
Class for the METALoci object
"""

import re
import dill as pickle


class MetalociObject:
    """
    This object is base class of METALoci. It contains all the information for a given region.
    """

    def __init__(self, region: str, resolution: int, persistence_length: float, save_path: str,
                 chrom: str = None, start: int = None, end: int = None, poi: int = None):
        """
        Creation of the METALoci object

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
