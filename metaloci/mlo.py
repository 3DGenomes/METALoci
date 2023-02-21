class mlo:
    def __init__(
        self,
        region,
        midpoint,
        resolution,
        matrix,
        mixed_matrices,
        persistance_length,
        kk_nodes,
    ):

        self.region = region
        self.midpoint = midpoint
        self.resolution = resolution
        self.matrix = matrix
        self.mixed_matrices = mixed_matrices
        self.persistance_length = persistance_length
        self.kk_nodes = kk_nodes
