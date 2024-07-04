"""
    Subset of modules that contain methods to compute spatial statistics on the Hi-C data. 
    For now, only Moran's I is implemented.      
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
