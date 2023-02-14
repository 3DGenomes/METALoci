"""
This script parses a bed file into the proper pickle file needed for METALoci
"""
__author__ = "Iago Maceda Porto and Leo Zuber Ponce"

import os.path
import pathlib
import re
import sys
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from datetime import timedelta
from time import time
from metaloci.misc import misc


description = """
This script picks a bed file (USCS format) of signals and transforms it into the format  
needed for METALoci to work.
Signal names must be in the format CLASS.ID_IND.ID.\n"
\tClass.ID can be the name of the signal mark.\n"
\tInd.ID can be the name of the patient/cell line/experiment.\n"""

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                        description=description,
                        add_help=False)

input_arg = parser.add_argument_group(title="Input arguments")

input_arg.add_argument('-w', '--work-dir', dest='work_dir', required=True,
                       metavar='PATH', type=str,
                       help='Path to working directory.')

input_arg.add_argument('-d', '--data', dest='data', required=True,
                       metavar='PATH', type=str,
                       nargs="*", action="extend",
                       help='Path to file to process. The file must contain titles for the columns,'
                            ' being the first 3 columns coded as chrom, start, end. The following '
                            'columns should contain the name of the signal coded as: id1_id2.\n'
                            'Names of the chromosomes must be the same as the coords_file '
                            'described below. '
                            '(More than one file can be specified, space separated)')

input_arg.add_argument('-o', '--name', dest='output', required=True,
                       metavar='STR', type=str,
                       help='Output file name.')

input_arg.add_argument('-r', '--resolution', dest='reso', required=True,
                       metavar='INT', type=int,
                       help="Resolution of the bins to calculate the signal")

region_arg = parser.add_argument_group(title="Region arguments")

region_arg.add_argument('-c', '--coords', dest='coords', required=True,
                        metavar='PATH', type=str,
                        help='Full path to a file that contains the name of the chromosomes in the '
                             'first column and the ending coordinate of the chromosome in the '
                              'second column.')

optional_arg = parser.add_argument_group(title="Optional arguments")

optional_arg.add_argument('-h', '--help', action="help",
                          help="Show this help message and exit.")

optional_arg.add_argument('-u', '--debug', dest='debug',
                          action='store_true', help=SUPPRESS)

args = parser.parse_args(None if sys.argv[1:] else ['-h'])

if not args.work_dir.endswith('/'):

    args.work_dir += '/'

work_dir = args.work_dir
data = args.data
out_name = args.output
coords = args.coords
resolution = args.reso * 1000
debug = args.debug

if debug:

    table = [["work_dir", work_dir],
             ["data", data],
             ["out_name", out_name],
             ["centro_and_telo_file", coords],
             ["binSize", resolution]
             ]
    
    print(table)
    sys.exit()

# Computing

start_timer = time()

info = misc.bed_to_metaloci(data, coords, resolution)

# Create a folder to store the pickle file.
out_path = os.path.join(work_dir, "signal", f"chr{chrm.rsplit('r', 1)[1]}")  
pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)  

# Creating a pandas DataFrame based on the dictionary created before and save it as pickle.
# df_info = pd.DataFrame(info)  
pickle_path = os.path.join(work_dir, "signal", f"chr{chrm.rsplit('r', 1)[1]}", f"{out_name}_chr{chrm.rsplit('r', 1)[1]}")
info.to_pickle(f"{pickle_path}_signal.pkl")

print(f"execution time: {timedelta(seconds=round(time() - start_timer))}")
print("all done.")
