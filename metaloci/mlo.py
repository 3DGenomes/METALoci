import dill as pickle


class mlo:
    def __init__(self, region, chrom, start, end, resolution):

        self.region = region
        self.chrom = chrom
        self.start = start
        self.end = end
        self.resolution = resolution
        self.midpoint = None
        self.matrix = None
        self.mixed_matrices = None
        self.persistance_length = None
        self.flat_matrix = None
        self.kk_top_indexes = None
        self.kk_cutoff = None
        self.kk_nodes = None
        self.kk_coords = None
        self.kk_distances = None
        self.lmi_geometry = None
        self.lmi_info = None
        self.save_path = None

    def save(self, file_handler):

        pickle.dump(self, file_handler)
