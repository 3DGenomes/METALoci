#!/usr/bin/env python
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # Warnings filtered

from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from tqdm import tqdm

# Arguments
description = """This script helps generating the coordinates file nedded for MNETALoci.\n"""
parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=description, add_help=False)
input_arg = parser.add_argument_group(title="Input arguments")
input_arg.add_argument(
	"-e",
	"--extension",
	dest="exte",
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
	"--chromfile",
	dest="cfile",
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
exte = args.exte
reso = args.reso
cfile= args.cfile
gfile= args.gfile
ofile= args.ofile

# Chromosome sizes
csizes = pd.read_csv(cfile, sep='\t',header=None)
csizes.columns='chr size'.split()
print('Chromosome sizes (top lines):')
print(csizes.head(10))
print('-------------\n\n')

# Read genes
genes = pd.read_csv(gfile, sep='\t')
print('Genes (top lines):')
print(genes.head(10))
print('-------------\n\n')

# Get code region for each gene
print('Generating coordinates...')
rids = pd.DataFrame(columns=['coords','id'])
for i, row in tqdm(genes.iterrows(), total=len(genes)):
	#print(row)
	rid  = '{}_{}'.format(i,row.symbol)
	size = row.end-row.start
	bins = int(size/reso)
	if (row.strand == '+'):
		midp = row.start
	else:
		midp = row.end
	rsta = midp-exte 
	if rsta < 1:
		rsta = 1
	rend = midp+exte
	cend = csizes['size'][csizes.chr==row.chrom].values[0]
	if (rend>=cend):
		rend = cend
	midp = int((midp-rsta)/reso)
	region = '{}:{}-{}_{}'.format(row.chrom,rsta,rend,midp)
	nrow = {'coords':region, 'id':rid}
	#print(region,'\t',rid)
	rids = rids.append(nrow, ignore_index=True)

# Save coordinates
print('Saving data to file:',ofile)
rids.to_csv(ofile, sep='\t', index=None)