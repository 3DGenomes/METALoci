#!/usr/bin/env python
import sys
import warnings
warnings.filterwarnings('ignore') # Warnings filtered
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate a Kamada-Kawai layout for a list of regions.')

# Required arguments
parser.add_argument('-w','--wfolder', required=True, dest='work_dir',
                    help='Path to the folder where results will be stored. THe script will create subfolders inside.')
parser.add_argument('-c','--cfolder', required=True, dest='cooler_dir',
                    help='Path to the folder where the cooler files are stored. You only need to specify the folder, not each cooler file.')
parser.add_argument('-g','--gfile', required=True, dest='gene_file',
                    help='Path to the file that contains the regions to analyze.')
parser.add_argument('-r','--resolution', required=True, dest='reso', type=int,
                    help='Resolution in bp to work at. ')
# Optional arguments
parser.add_argument('-p','--pcutoff', required=False, dest='cutoff', type=float, default=20,
                    help='Select the top X percentage of HiC interaction to generate Kamada-Kawai graph [20].')
parser.add_argument('-x','--silent', required=False, dest='silent', action='store_false',
                    help='Silent mode [True]')
parser.add_argument('-f','--force', required=False, dest='force', action='store_true',
                    help='Force rewriting existing data [False]')

# Parse arguments
args = parser.parse_args()

print("Input argument values:")
for k in args.__dict__:
    if args.__dict__[k] is not None:
        print("\t{} --> {}".format(k,args.__dict__[k]))
#sys.exit()

# Variables
work_dir   = args.work_dir
cooler_dir = args.cooler_dir
genes_2_do = args.gene_file
reso       = args.reso
cutoff     = args.cutoff
silent     = args.silent
force      = args.force

# Massage arguments
# Divide the cutoff by 100, so we can just multiply the length of the array of contacts
cutoff  = int(cutoff)/100

# Checking if the user has ended the path with the slash, if not add it
if not args.cooler_dir.endswith("/"):
    cooler_dir = args.cooler_dir+"/"
else:
    cooler_dir = args.cooler_dir
if not args.work_dir.endswith("/"):
    work_dir = args.work_dir+"/"
else:
    work_dir = args.work_dir

# LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
import os
import h5py
import cooler
import pickle
import subprocess
import pathlib
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.signal import argrelextrema

# FUNCTIONS
def CreateStructure(work_dir):

    ## This function creates the folder structure in the working directory on where to store the 
    ## Kamada-Kawai layaouts extracted from the Hi-C matrices
        
    subfolders = ["mls/coords", "mls/datas", "mls/figures", "mls/pdfs/png", "mls/pdfs/pdf", "mls/pdfs/matdata"]
    
    for folder in subfolders:
        dir_com = work_dir+folder

	## parents is to create the structure "above", exist_ok is to skip the folder in case it exists
        pathlib.Path(dir_com).mkdir(parents=True, exist_ok=True) 

def CreateSubStructure(work_dir, dirnum):

    ## This function creates the a subfolder structure in order to separate the outputs in a way 
    ## that the OS does not explode (separates the output by chromosome)
    
    subfolders = ["mls/coords/", "mls/datas/", "mls/pdfs/png/", "mls/pdfs/pdf/", "mls/pdfs/matdata/"]
    
    for folder in subfolders:
        dir_com = work_dir+folder+str(dirnum)

	## parents is to create the structure "above", exist_ok is to skip the folder in case it exists
        pathlib.Path(dir_com).mkdir(parents=True, exist_ok=True) 

def cool2mat(cfile, region, pdff, silent): 
        region,nu = region.split('_')
        c = cooler.Cooler(cfile)
        
        print("COOL2MAT> ",region)
        print("COOL2MAT> ",c.info)
        
        mat = c.matrix(sparse=True, balance=True).fetch(region)
        arr = mat.toarray()
        #print(arr)
        arr[np.isnan(arr)] = 0 # Replace NaNs by 0
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        ## This line was added in order to avoid the "divided by 0" error from the log10 function below.
        ## It deactivates the error, and the script reactivates it after the calculation.
        np.seterr(divide='ignore')
        im = ax.matshow(np.log10(arr), cmap='YlOrRd')
        
        np.seterr(divide='warn')
        
        fig.colorbar(im)
        
        if (pdff):
            plt.savefig(pdff)
        if (silent==False):
            plt.show()
            
        plt.close()
        
        return(arr)

def MatTransform(mat):

    ## We transform the matrix (normalized contacts?) in order to work more easily with it 

    print('Matrix data...')
    print('\tMin: {}, Max: {}'.format(np.min(mat), np.max(mat)))
    
    ## First of all we check if the diagonal has 0s. If this is the case, we set the row and 
    ## column for that point on the diagonal to also 0s (sometimes the normalization algorithm
    ## has artifacts)

    diag = np.array(mat.diagonal())
    
    for i in range(0, len(diag)):
        if (diag[i] == 0):
            
            mat[i] = 0
            mat[:, i] = 0

    ## Pseudocounts if min is zero
    if (np.min(mat) == 0):
        pc = np.min(mat[mat>0])
        print('Pseudocounts: {}'.format(pc))
        mat = mat+pc

    ## Scale if all below 1
    if (np.max(mat)<=1 or np.min(mat)<=1): ## Why np.min?
        sf = 1/np.min(mat)
        print('Scaling factor: {}'.format(sf))
        mat = mat*sf
    
    ## Recheck that the mnimum of the matrix is, at least, 1    
    if (np.min(mat)<1):
        mat[mat<1] = 1
        
    print('\tMin: {}, Max: {}'.format(np.min(mat), np.max(mat)))

    ## Log10 the data
    print('Log10 matrix...')
    mat = np.log10(mat)
    print('\tMin: {}, Max: {}'.format(np.min(mat), np.max(mat)))
   
    ## Mean of the data
    print('Mean of non zero...')
    me = mat[mat>1].mean()
    print('\tMean: {}'.format(me))
    print()
    
    return(mat)

def CheckDiagonal(diag):
    
    ## This function checks the number of 0s of the diagonal and return
    ## the total number of 0s from the diagonal and the maximum stretch of
    ## consecutive 0s at the diagonal.

    total = 0
    stretch = 0
    max_stretch = 0
    
    for i in range(0, len(diag)):
        
        if (diag[i] == 0):
            total += 1
            stretch +=1
            
            if (stretch > max_stretch):
                max_stretch = stretch
            
        else:
            stretch = 0
            
    return(total, max_stretch)

def AnaMat(runid, mat, reso, pl, silent, work_dir, dirnum, cutoff):

    ## Function to change the matrix in order to get the cutoff% of max hits

    ## runid is used to print the coordinates of the gene, but most OSs do not 
    ## like to have ":" or "-" in the filename. It is changed to "_"    
    runid_4_linux = runid.replace(":", "_").replace("-", "_")
    
    plotsize = 5
    
    ## Always remember to copy the numpy array using copy. If you do the usual
    ## (temp = mat), it acts as a link between them and changes in temp will affect
    ## the original matrix!
    temp = mat.copy().flatten()
    print("Matrix size: {}\n".format(len(temp)))
    
    ## Put the minimum values as NaNs, (remember, the minimums of this matrix correspond
    ## to the pseudocounts, which are the 0 of the Hi-C)
    temp[temp<=np.min(temp)] = np.nan
    
    ## Give the user info about how empty (or filled) is the matrix.
    non0 = int(len(temp)-len(temp[np.isnan(temp)]))
    perc_non0 = np.round(non0 / len(temp) * 100, decimals = 2)
    
    print("Non 0 matrix size: {}".format(non0))
    print("% of non 0 in the original matrix: {}%\n".format(perc_non0))   
    
    ## Cutoff percentil
    print('Cutoff...')
    per = cutoff
    subf = mat.flatten()
    subf_nonz = subf[subf>min(subf)] # Percentil of NON-ZEROES in original matrix
    #top = int(len(subf)*per)
    top = int(len(subf_nonz)*per)
    ind = np.argpartition(subf, -top)[-top:]
    temp = subf[ind]
    print("\tSize of the whole matrix: {}".format(len(subf)))
    print("\tNumber of top interactions to pick and get the cutoff: {}".format(top))
    print("\tSize of the submatrix with the top interations: {}".format(len(ind)))
    print("\tSize of the new matrix with {} elements: {}".format(top, len(temp)))
    print()
    ct = np.min(temp)
    ctmax = np.max(temp)
    print('\tCutoff {}'.format(ct))
    print('\tCutoff (%) {}'.format(len(temp)/len(subf)*100))
    print()
    
    # Data distribution    
    print('Data distr...')
    #sns.distplot(mat)
    if (silent==False):
        plt.show()
    plt.close()

    # Subset to cutoff percentil
    logcutoff = ct
    subm = mat.copy()
    subm = np.where(subm==1., 0, subm) 
    subm[subm < logcutoff] = 0
    
    # Connectivity (~persistance length)
    print('Connectivity between consequtive particles...')
    if (pl):
        perlen = pl
    else:
        perlen = np.nanmax(subm[subm>0])**2
    print('\t"Persistance lenght": {}'.format(perlen))
    print()
    
    rng = np.arange(len(subm)-1)
    subm[rng, rng+1] = perlen

    # Remove exact diagonal
    subm[rng, rng] = 0
    
    # Plot
    #fig, ax = plt.subplots(figsize=(plotsize, plotsize))
    #ax.imshow(mat, cmap='YlOrRd', vmax=ctmax)
    #temp = work_dir+'mls/pdfs/matdata/subfolder_'+str(dirnum)+'/{}_hic.pdf'
    #plt.savefig(temp.format(runid_4_linux))
    #plt.close()
    
    # Mix matrices to plot
    u = np.triu(mat+1,k=1)
    l = np.tril(subm,k=-1)
    mixmat = u+l
    
    # Plot data
    #fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle('Matrix for {}'.format(runid))
    ax1.imshow(mixmat, cmap='YlOrRd', vmax=ctmax+1)
    #sns.histplot(data=temp.flatten(), stat="density", alpha=0.4, kde=True, legend=False, 
    #             kde_kws={"cut": 3}, **{"linewidth" : 0})
    fig.tight_layout()
    
    temp = work_dir+'mls/pdfs/matdata/'+str(dirnum)+'/{}_matdata.pdf'
    if (silent==False):
        plt.savefig(temp.format(runid_4_linux))
    plt.close()

    # Modify the matrix and transform to restraints
    subm = np.where(subm==0, np.nan, subm) # Remove zeroes
    subm = 1/subm # Convert to distance Matrix instead of similarity matrix
    subm = np.triu(subm, k=0) # Remove lower diagonal

    # This function (nan_to_num) allows the user to change nans, posinf and neginf 
    # to a number the user specifies
    subm = np.nan_to_num(subm, nan=0, posinf=0, neginf=0) # Clean nans and infs
    
    # FINAL RESTRAINTS
    #print("FINAL RESTRAINTS CALCULATED...")
    #fig, ax = plt.subplots(figsize=(plotsize, plotsize))
    #ax.imshow(subm)
    #plt.savefig('mls/pdfs/{}_restraints.pdf'.format(runid_4_linux))
    #if (silent=False)
    #    plt.show()
    #plt.close()    
 
    return(subm)

def Mat2KK(runid, mat, reso, midp, silent, work_dir, dirnum):

    runid_4_linux = runid.replace(":", "_").replace("-", "_")

    ## Create sparse matrix
    matrix = csr_matrix(mat)
    print("Sparse matrix contains {:,} restraints".format(np.count_nonzero(matrix.toarray())))

    ## Get the KK layout
    G = nx.from_scipy_sparse_array(matrix)
    #G = nx.from_scipy_sparse_matrix(matrix)
    print("Layouting KK...")
    pos = nx.kamada_kawai_layout(G)

    ## Get distance matrix
    coords = list(pos.values())
    dists = distance.cdist(coords, coords, 'euclidean')
    ## Plot KK
    ## The following code is duplicated in order to produce visible lines in both pdf
    ## and png. Silver lines in png where almost invisible.
    print("Plotting KK...")
    
    plotsize = 10
    color_map = []
    plt.figure(figsize=(plotsize,plotsize)) 
    options = {
        'node_size': 50,
        'edge_color': 'silver',
        'linewidths': 0.1,
        'width': 0.05
    }
    
    ## With nx.draw we draw the network (the Kamada-Kawai layout), to add the numbers of the
    ## nodes we use plt.text.
    nx.draw(G, pos, node_color=range(len(pos)), cmap=plt.cm.coolwarm, **options)
    if (midp):
        plt.scatter(pos[midp][0],pos[midp][1], s=80, facecolors='none', edgecolors='r')
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    sns.lineplot(x=xs, y=ys, sort=False, lw=2, color='black', legend = False, zorder=1)
    for p in range(1,len(coords)+1):#range(Dpar-Dlen,Dpar+Dlen+1):
        x = coords[p-1][0]
        y = coords[p-1][1]
        plt.text(x, y, s=p, color='black', fontsize=10)
        
    temp = work_dir+'mls/pdfs/pdf/'+str(dirnum)+'/{}_kk.pdf'
    if (silent==False):
        plt.savefig(temp.format(runid_4_linux))
    plt.close()
    
    if (silent==False):
        plt.show()
    
    options['edge_color'] = 'black'
    plotsize = 15
    plt.figure(figsize=(plotsize,plotsize))
    
    nx.draw(G, pos, node_color=range(len(pos)), cmap=plt.cm.coolwarm, **options)
    if (midp):
        plt.scatter(pos[midp][0],pos[midp][1], s=80, facecolors='none', edgecolors='r')
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    sns.lineplot(x=xs, y=ys, sort=False, lw=2, color='black', legend = False, zorder=1)
    for p in range(1,len(coords)+1):#range(Dpar-Dlen,Dpar+Dlen+1):
        x = coords[p-1][0]
        y = coords[p-1][1]
        plt.text(x, y, s=p, color='black', fontsize=15)
    
    temp = work_dir+'mls/pdfs/png/'+str(dirnum)+'/{}_kk.png'
    if (silent==False):
        plt.savefig(temp.format(runid_4_linux))
    
    plt.close()
    
    ## Save KK data in pkl data, so it can be read at later time/scripts
    temp = work_dir+'mls/coords/'+str(dirnum)+'/{}.pkl'
    cfile = temp.format(runid_4_linux)
    with open(cfile, 'wb') as output:
        pickle.dump(matrix, output)    
        pickle.dump(pos, output)
        pickle.dump(dists, output)
        pickle.dump(coords, output)    
        pickle.dump(G, output)    
    
    return(coords,pos)

CreateStructure(work_dir)

## In this file the script stores bad regions (> 50% of the diagonal are 0s or
## the stretch of 0s is bigger than 50)
perc_file = work_dir+"bad_regions.txt"
perc_f = open(perc_file, 'a')

# Input file with all regions to KK
regions = pd.read_csv(genes_2_do, sep='\t')

for i,row in regions.iterrows():
    
    region = row.coords
   
    runid = '{}_{}'.format(region,reso)
    #runid = '{}'.format(region)
    
    runid_4_linux = runid.replace(":", "_").replace("-", "_")
    
    chrom = runid_4_linux.split("_")[0]
   
    CreateSubStructure(work_dir, chrom)
    
    grep_com = "ls -1v "+cooler_dir+" | grep mcool"
    grep_res = subprocess.getoutput(grep_com)
    #print("GC>>>>>>>>>>>", grep_res)
    if (len(grep_res.split()) > 1): # Multiple chromosomes files in folder
        grep_com = "ls -1v "+cooler_dir+" | grep mcool | grep _"+chrom+"_"
        grep_res = subprocess.getoutput(grep_com)
    #print("GP>>>>>>>>>>>", grep_res)

    cfile = cooler_dir+grep_res+"::/resolutions/"+str(reso)
       
    print('\n---> Working on region {}: {} [{}/{}]'.format(row.id,runid,i+1,len(regions)))
    print("Will use {} file for chromosome {}... ".format(cfile, chrom))
    
    # Check if file already exists
    temp = work_dir+'mls/coords/'+str(chrom)+'/{}.pkl'
    
    coordsfile = temp.format(runid_4_linux)
    
    print("Checking if region {} is already done...".format(region))

    grep_com = "grep {} {}".format(region, perc_file)    
    grep_check = subprocess.getoutput(grep_com)


    ## This part is modified. In some of the datasets (NeuS) the matrix was completely empty, 
    ## so the pkl file was not created (gave an error in the MatTransform function: pc = np.min(mat[mat>0]))
    ## To bypass this part, the script checks the bad_regions.txt; if the regions is also found in this 
    ## file, the script skips the calculations (as the script already has checked the region in question).
    if os.path.isfile(coordsfile) and force==False:
        print('WARNING! already done! If you want to rewrite it, use -f or --force.')    
        continue
    elif (grep_check != ""):
        print('ERROR! no data file... {}!'.format(perc_file))
        continue
  
    try:
        ## Run the code...
        ## Get the submatrix for the region from the general Hi-C matrix
        print(">>>>>>>>",cfile, region, None, silent)
        mat = cool2mat(cfile, region, None, silent)

	## Get the diagonal to check if it is correct
        diag = np.array(mat.diagonal())
    
        total_0, max_stretch = CheckDiagonal(diag)
        perc_0 = np.round(total_0 / len(diag)*100, decimals = 2)

	## If the whole diagonal is 0s, the script cannot do anything, so it goes to the next 
	## region after saying so to the user.
	## flush is needed in order to write to the file and continue (memory can get full and 
	## not write properly to file)
        if (total_0 == len(diag)):
            print("Void matrix; passing to the next one")
            perc_f.write("{}\tvoid\n".format(region))
            perc_f.flush()
            continue

	## If half (or more) of the diagonal corresponds to 0s, save that region in
	## a file to check
        if (int(perc_0) >= 50):
            perc_f.write("{}\tperc\n".format(region))
            perc_f.flush()
        
	## If half (or more) of the diagonal corresponds to 0s, save that region in
	## a file to check
        elif (int(max_stretch) >= 50):
            perc_f.write("{}\tstretch\n".format(region))
            perc_f.flush()
        
        ## Change the matrix in order to properly calculate the Kamada-Kawai layout
        mat = MatTransform(mat)

        ## Get submatrix of restraints
        subm = AnaMat(runid, mat, reso, None, silent, work_dir, chrom, cutoff)

        ## Kamada Kawai Layout
        coords,pos = Mat2KK(runid, subm, reso, None, silent, work_dir, chrom)
    
        print('<---done!')
    
    except Exception:
        print('ERROR!>Failed: {}'.format(runid))

