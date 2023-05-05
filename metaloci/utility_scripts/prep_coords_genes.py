import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore') # Warnings filtered

from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter

from tqdm import tqdm

# Arguments
description = """

This script helps generating the coordinates file nedded for MNETALoci.\n

"""

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=description, add_help=False)

input_arg = parser.add_argument_group(title="Input arguments")
input_arg.add_argument(
	"-s",
	"--size",
	dest="size",
	required=True,
	type=int,
	help="Extension in bp of the region of interest both up- and down-stream.",
)
input_arg.add_argument(
	"-r",
	"--resolution",
	dest="reso",
	required=True,
	type=int,
	help="Resolutoin in bp of the region of interest.",
)
input_arg.add_argument(
	"-c",
	"--coords",
	dest="coords",
	required=True,
	type=str,
	help="Path to the chromosome size file.",
)
input_arg.add_argument(
	"-g",
	"--genefile",
	dest="gfile",
	required=True,
	type=str,
	help="Path to the gene list file in bed format.",
)
input_arg.add_argument(
	"-o",
	"--outfile",
	dest="ofile",
	required=True,
	type=str,
	help="Path to the output file with coordinates.",
)
args = parser.parse_args(None if sys.argv[1:] else ["-h"])


size = args.size
resolution = args.reso
coords = args.coords
genes_file = args.gfile
output_file = args.ofile

# Chromosome sizes
chrom_sizes = pd.read_csv(coords, sep='\t',header=None)
chrom_sizes.columns = "chrom_size".split()

print(f"Chromosome sizes (top lines): \n{chrom_sizes.head(10)} \n-------------")

# Read genes
genes = pd.read_csv(genes_file, sep='\t')
print(f"Genes (top lines): \n{genes.head(10)} \n-------------")


# Get code region for each gene
print("Generating coordinates...")

rids = pd.DataFrame(columns=['coords','id'])

for i, row in tqdm(genes.iterrows(), total=len(genes)):
	#print(row)
	rid  = '{}_{}'.format(i, row.symbol)
	size = row.end - row.start
	bins = int(size / resolution)
	
	if (row.strand == '+'):
		
		midp = row.start
		
	else:
		
		midp = row.end
		
	rsta = midp - size 
	
	if rsta < 1:
		
		rsta = 1
		
	rend = midp + size
	cend = chrom_sizes["size"][chrom_sizes.chr == row.chrom].values[0]
	
	if (rend >= cend):
		
		rend = cend
		
	midp = int((midp - rsta) / resolution)
	region = f"{row.chrom}:{rsta}-{rend}_{midp}"
	nrow = {"coords":region, "id":rid}
	#print(region,'\t',rid)
	rids = rids.append(nrow, ignore_index=True)

# Save coordinates
print(f"Saving data to file: {output_file}")
rids.to_csv(output_file, sep='\t', index=None)