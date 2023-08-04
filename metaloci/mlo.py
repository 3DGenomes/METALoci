import dill as pickle


class MetalociObject:
    def __init__(self, region, chrom, start, end, resolution, poi, persistence_length, save_path):

        self.region = region
        self.chrom = chrom
        self.start = start
        self.end = end
        self.resolution = resolution
        self.poi = poi
        self.persistence_length = persistence_length
        self.save_path = save_path
        self.subset_matrix = None
        self.matrix = None
        self.bad_region = None
        self.mixed_matrices = None
        self.signals_dict = None
        self.flat_matrix = None
        self.kk_top_indexes = None
        self.kk_cutoff = None
        self.kk_nodes = None
        self.kk_coords = None
        self.kk_distances = None
        self.agg = None
        self.lmi_geometry = None
        self.lmi_info = {}

    def save(self, file_handler):

        pickle.dump(self, file_handler)
