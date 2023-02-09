import sys
import warnings
warnings.filterwarnings('ignore') # Warnings filtered
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate a list of regions from a list of genes.')

# Required arguments
parser.add_argument('-c','--cfile', required=True, dest='cfile',
                    help='Path to the chromosome size file (two column).')
parser.add_argument('-i','--ifile', required=True, dest='ifile',
                    help='Path to the annotation file (three column).')
parser.add_argument('-n','--numpf', required=True, dest='numpf', type=int,
                    help='Split total in n lines per file.')
parser.add_argument('-r','--resolution', required=True, dest='reso', type=int,
                    help='Resolution in bp to work at. ')
parser.add_argument('-e','--extension', required=True, dest='exte', type=int,
                    help='Extension from TSS up- and down-stream (in bp). ')

# Parse arguments
args = parser.parse_args()

print("Input argument values:")
for k in args.__dict__:
    if args.__dict__[k] is not None:
        print("\t{} --> {}".format(k,args.__dict__[k]))
#sys.exit()

# Variables
cfile   = args.cfile
ifile   = args.ifile
numpf   = args.numpf
reso    = args.reso
exte    = args.exte

# LIBRARIES
import os
import re
import pandas as pd
from tqdm import tqdm

# Read chrom sizes
csizes = pd.read_csv(cfile, sep='\t', header=None)
csizes.columns = ['chrom','size']
print('Read size for {} chromosomes'.format(len(csizes)))
print(csizes)

# Read coords
coords = pd.read_csv(ifile, sep='\t')
print('Read a total of {} entries'.format(len(coords)))
print(coords)

# Parse data
data = pd.DataFrame(columns='coords id'.split())
for i,row in tqdm(coords.iterrows(), total=coords.shape[0]):
    chrm, ini, end = row.chr, row.start, row.end
    if (chrm in list(csizes.chrom)):        
        if (row.strand == "+"):
            tss = ini
        else:
            tss = end
        tss = int(tss)
        ini = tss-exte
        end = tss+exte
        csize = csizes['size'][csizes.chrom==chrm].values[0]
        #print(chrm,tss,ini,end,csize)
        # Check start and end beyond chromosome
        if (ini<1):
            ini = 1
        if (end>csize):
            end = csize
        # Determine bin of TSS
        mid = int((tss-ini)/reso)
        tot = int((end-ini)/reso)
        # Write results
        coo = '{}:{}-{}_{}'.format(chrm,ini,end,mid)
        #print(coo)
        nrow= {'coords':coo,
               'id':row.gene}
        data = data.append(nrow, ignore_index=True)
        #print("{}:{}-{}\t{}\t{}".format(chrm,ini,end,row.symbol,row.id))

# Write files...
print('Writting files...')
n = 0
for id, i in enumerate(range(0,len(data),numpf)):
    n = n + 1
    start = i
    end = i + numpf-1 #neglect last row ...
    data.iloc[start:end].to_csv('{}_{}_{}.{}.lst'.format(ifile,reso,exte,n), sep='\t', index=False)

print('Done!')
