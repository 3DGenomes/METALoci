#!/usr/bin/env python
import sys
import warnings
warnings.filterwarnings('ignore') # Warnings filtered
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate LMI for any genomic signal for a list of regions.')

# Required arguments
parser.add_argument('-w','--wfolder', required=True, dest='work_dir',
                    help='Path to the folder where results will be stored. THe script will create subfolders inside.')
parser.add_argument('-c','--cfolder', required=True, dest='cooler_dir',
                    help='Path to the folder where the cooler files are stored. You only need to specify the folder, not each cooler file.')
parser.add_argument('-g','--gfile', required=True, dest='gene_file',
                    help='Path to the file that contains the regions to analyze.')
parser.add_argument('-s','--signal_dir', required=True, dest='signal_dir',
                    help='Path to the folder where the signal bigwig files are stored. Need to have the same name as signals and end with \".bw\". ')
parser.add_argument('-t','--typesignals', required=True, dest='signals',
                    help='Signals to run seperated by \",\". ')
parser.add_argument('-r','--resolution', required=True, dest='reso', type=int,
                    help='Resolution in bp to work at. ')
# Optional arguments
parser.add_argument('-p','--pval', required=False, dest='signipval', default=0.05,
                    help='P-value to select significant LMIs [0.05].')
parser.add_argument('-i','--influence', required=False, dest='influence', default=1.5,
                    help='Bins influence radius [1.5 bins average distance].')
parser.add_argument('-b','--buffer', required=False, dest='bfact', default=2.0,
                    help='Buffer to multiply influence [2.0].')
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
gene_file = args.gene_file
reso      = args.reso
signipval = args.signipval
silent    = args.silent
force     = args.force
signals   = args.signals.split(',')
influence = args.influence
bfact     = args.bfact

# Massage arguments
# Checking if the user has ended the path with the slash, if not add it
if not args.cooler_dir.endswith("/"):
    cooler_dir = args.cooler_dir+"/"
else:
    cooler_dir = args.cooler_dir
if not args.work_dir.endswith("/"):
    work_dir = args.work_dir+"/"
else:
    work_dir = args.work_dir
if not args.signal_dir.endswith("/"):
     signal_dir = args.signal_dir+"/"
else:
     signal_dir = args.signal_dir

# LIBRARIES
print('Importing...')
import time
import os
import math
import re
import pickle
import pyBigWig
import pathlib
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import geopandas as gpd
import networkx as nx
import libpysal as lp

import shapely.geometry
import shapely.ops

from esda.moran import Moran_Local
from libpysal.weights.spatial_lag import lag_spatial
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, LineString, Point
print("Imports done!")

# FUNCTIONS
def signalProfile(bw,reg,res,norm,log10):    
    sig = []
    c,i,e = re.split(':|-',reg)
    i = int(i)
    e = int(e)+1
    for coor in range(i,e,res):
        amean = np.mean(bw.values(c, coor, coor+res))
        sig.append(amean)
    print("signal min and max: {} {}".format(np.nanmin(sig),np.nanmax(sig)))
    if (isinstance(norm, (int, float))):
        sig = [float(i)/norm for i in sig]
    if (norm == 'max'):
        sig = [float(i)/max(sig) for i in sig]
    if (norm == 'sum'):
        sig = [float(i)/sum(sig) for i in sig]
    if (norm == '01'):
        sig = [(float(i)-min(sig))/(max(sig)-min(sig)+0.01) for i in sig]
    
    if (log10):
        sig = np.log10(sig)
    
    return(sig)

def signalProfile20230123(bw,reg,res,pc,norm,log10):    
    sig = []
    c,i,e = re.split(':|-',reg)
    i = int(i)
    e = int(e)+1
    for coor in range(i,e,res):
        try:
            #asum = np.nansum(bw.values(c, coor, coor+res)) # SUMS ALL OVER 10K TIMES!
            asum = np.nanmean(bw.values(c, coor, coor+res))
            if (asum < pc):
                #asum = 0
                #asum = np.NaN
                asum = pc
        except RuntimeError:
            asum = pc
            #asum = np.NaN
        sig.append(asum)
    if (isinstance(norm, (int, float))):
        sig = [float(i)/norm for i in sig]
    if (norm == 'max'):
        sig = [float(i)/max(sig) for i in sig]
    if (norm == 'sum'):
        sig = [float(i)/sum(sig) for i in sig]
    if (norm == '01'):
        sig = [(float(i)-min(sig))/(max(sig)-min(sig)+0.01) for i in sig]

    if (log10):
        sig = np.log10(sig)
    
    return(sig)

def signalProfileBED(bd,reg,res,pc,norm):    
    sig = []
    c,i,e = re.split(':|-',reg)
    i = int(i)
    e = int(e)+1
    sig = bd[(bd.chromo ==c) & (bd.start>=i) & (bd.end<=e)].iloc[:, 3]
    if (isinstance(norm, (int, float))):
        sig = [float(i)/norm for i in sig]
    if (norm == 'max'):
        sig = [float(i)/max(sig) for i in sig]
    if (norm == 'sum'):
        sig = [float(i)/sum(sig) for i in sig]
    if (norm == '01'):
        sig = [(float(i)-min(sig))/(max(sig)-min(sig)+0.01) for i in sig]
    return(sig)

def signalProfileOK(bw,reg,res,pc,norm,log10):    
    sig = []
    c,i,e = re.split(':|-',reg) ## Split the region name intro chromosome, start and end (chr11:128908798-132928998). Get the name from the pkl file?
    i = int(i)
    e = int(e)+1
    
    for coor in range(i,e,res): ## Coords from start to end by resolution step
        try:
            ## Get the signal data from the bigwig file (be careful with the namings of the chr)
            if (np.all(np.isnan(bw.values(c, coor, coor+res)))):
                asum = pc                
            else:
                asum = np.nanmedian(bw.values(c, coor, coor+res))
                
        except RuntimeError:
            asum = pc
        
        if (asum < pc): ## If the signal is below a minimum, asume it's the minimum
            asum = pc

        sig.append(asum)
    
    if (isinstance(norm, (int, float))):
        sig = [float(i)/int(norm) for i in sig]
    elif (norm == 'max'):
        sig = [float(i)/max(sig) for i in sig]
    elif (norm == 'sum'):
        sig = [float(i)/sum(sig) for i in sig]
    elif (norm == '01'):
        sig = [(float(i)-min(sig))/(max(sig)-min(sig)+0.01) for i in sig]
    
    if (log10):
        sig = np.log10(sig)
    
    return(sig)

def GetPointId(cs,p):
    i = 0
    for c in cs:
        if (p.contains(Point(c))):
            return(i)
            continue
        else:
            i=i+1

def GetRegionFromFile(cfile): ## Function to get the region from the filename of the pkl (assuming files where generated with the pipeline)
    
    region_temp = cfile.split("/")[-1]
    
    region_split = region_temp.split("_")
    
    region = region_split[0] + ":" + region_split[1] + "-" + region_split[2]
        
    return(region)

def GetSignalPlot(cfile, sig_data, reso, G, pos): ## This function "paints" each node of the Kamada-Kawai with the signal from the signal file
    
    region = GetRegionFromFile(cfile)
    
    #minsig = -1
    #minsig = min(-1,sig_data.header()['minVal'])#*reso
    #print('Signal profiling over {} max value and a pseudo-count of {}'.format(sig_data.header()['maxVal'], minsig))
    #signal = signalProfile(sig_data,region,reso,minsig,'01',False) # Decided to test non normalized data
    #signal = signalProfile(sig_data,region,reso,minsig,None,False)
    signal = signalProfile(sig_data,region,reso,None,False)
    signal = np.array(signal)
    
    plotsize = 10

    options = {
        'node_size': 50,
        'edge_color': 'silver',
        'linewidths': 0.1,
        'width': 0.05, 
    }

    print("Mapping into KK signal data...")
    color_map = []
    plt.figure(figsize=(plotsize,plotsize)) 
    nx.draw(G, pos, node_color=signal[0:len(pos)], cmap=plt.cm.coolwarm, **options)
    plt.scatter(pos[midp][0],pos[midp][1], s=80, facecolors='none', edgecolors='r')
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    sns.lineplot(x = xs, y = ys, sort=False, lw=2, color='black', legend = False, zorder=1)
    for p in range(1,len(coords)+1):#range(Dpar-Dlen,Dpar+Dlen+1):
        x = coords[p-1][0]
        y = coords[p-1][1]
        plt.text(x, y, s=p, color='black', fontsize=10)

    plt.show()
    plt.close()

def LMI_data(cfile, sig_data, reso, poly_from_lines, coords, buffer):

    region = GetRegionFromFile(cfile)
    
    #minsig = -1
    #minsig = min(-1,sig_data.header()['minVal'])#*reso
    #print('Signal profiling over {} max value and a pseudo-count of {}'.format(sig_data.header()['maxVal'], minsig))
    #signal = signalProfile(sig_data,region,reso,minsig,'01',False) # Decided not to use normalized data
    #signal = signalProfile(sig_data,region,reso,minsig,None,False)
    signal = signalProfile(sig_data,region,reso,None,False)
    signal = np.array(signal)
          
    mydata = gpd.GeoDataFrame(columns=['v','geometry']) ## Stored in Geopandas DataFrame to do LMI
        
    for poly in poly_from_lines:
        coordid =  GetPointId(coords,poly)
        try:
            v = signal[coordid]
        except IndexError:
            v = 0.    
        
        mydata = mydata.append({'v':v,'geometry':poly}, ignore_index=True)

    # Voronoi shaped & closed
    shape = LineString(coords).buffer(buffer)
    close = mydata.convex_hull.union(mydata.buffer(0.1, resolution=1)).geometry.unary_union
    i = 0
    for q in mydata.geometry:
        mydata.geometry[i] = q.intersection(close)
        mydata.geometry[i] = shape.intersection(q)
        i = i+1
        
    return(mydata)

# START PROCESS
startTime = time.perf_counter()

if (signals == ""):
    signal_switch = 0
    print("All signals in the signal folder will be used")
else:
    signal_switch = 1
    print("Signals choosen are: " + " ".join(signals))

c_dir = os.path.join(work_dir, "mls", "coords")

info = {}

signal_files = [f for f in glob.glob(signal_dir + "*.bw")]

signal_names = []

for sf in signal_files:
    # print(row.IID)
    
    name_f = os.path.splitext(os.path.split(sf)[1])[0]
    
    if (signal_switch):
        
        if (name_f in signals):
            
            signal_names.append(name_f)
            
        else:
            
            continue
        
    else:
        
        signal_names.append(name_f)  
    
    cosa = pyBigWig.open(sf)
        
    info[name_f] = cosa
print(">>>>>>>>",signal_names)
    
#How to read the genes
genes = pd.read_table(gene_file)    
    
for i_gene, line in genes.iterrows():
       
    gene_c = line.coords
    gene_n = line.id
    gene_i = line.id
    print(gene_c,gene_n,gene_i)
    
    region = gene_c.split("_")[0]
    midp = gene_c.split("_")[1]
    
    # region_lin = region.
    
    gene_i_t = gene_i.split(".")[0]
      
    chr_temp = region.split(":")[0]
    
    cfile_temp = region.replace(":", "_").replace("-", "_")
    
    # cfile_temp = cfile_temp + "_" + str(reso) + ".pkl"
    
    cfile = os.path.join(c_dir, chr_temp, cfile_temp + "_" + str(midp) + "_" + str(reso) + ".pkl")
        
    if os.path.isfile(cfile):
        print ("File exists. Loading data...")
        with open(cfile, 'rb') as input:
            matrix = pickle.load(input)
            pos = pickle.load(input)
            dists = pickle.load(input)
            coords = pickle.load(input)
            G = pickle.load(input)
    else:
        print("File does not exist {}. Exiting script...".format(cfile))
        continue
        #sys.exit(1)
    
#    scatter_2_save = os.path.join(work_dir, "datasets", dataset_name, chr_temp, cfile_temp, "scatter")
#    lisa_2_save = os.path.join(work_dir, "datasets", dataset_name, chr_temp, cfile_temp, "lisa_cluster")
#    choro_2_save = os.path.join(work_dir, "datasets", dataset_name, chr_temp, cfile_temp, "choropleth")
#    moran_2_save = os.path.join(work_dir, "datasets", dataset_name, chr_temp, cfile_temp, "moran_geom")
#    moran_2_save = os.path.join(work_dir, dataset_name, "mls/datas", chr_temp)
    moran_2_save = os.path.join(work_dir, "mls/datas", chr_temp)

    
#    pathlib.Path(scatter_2_save).mkdir(parents=True, exist_ok=True)
#    pathlib.Path(lisa_2_save).mkdir(parents=True, exist_ok=True)
#    pathlib.Path(choro_2_save).mkdir(parents=True, exist_ok=True)
#    pathlib.Path(moran_2_save).mkdir(parents=True, exist_ok=True)
    
    # info_file = open(os.path.join(work_dir, "datasets", dataset_name, chr_temp, gene_i_t, "info_file.txt"), 'w')
    # info_file.write("IID\tMoran_I\tMoran_pvalue\n")
    
    
    # From points to Voronoi polygons
    print("Voronoing KK...")
    # set limits
    lims = 2
    points = coords.copy()
    points.append(np.array([-lims,lims]))
    points.append(np.array([lims,-lims]))
    points.append(np.array([lims,lims]))
    points.append(np.array([-lims,-lims]))
    points.append(np.array([-lims,0]))
    points.append(np.array([0,-lims]))
    points.append(np.array([lims,0]))
    points.append(np.array([0,lims]))
    # get voronoi
    vor = Voronoi(points)
    #voronoi_plot_2d(vor, show_vertices=False)

    print("Voronoi Polygons and mapped data...")
    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]

    poly_from_lines = list(shapely.ops.polygonize(lines))
    
    print("Voronois polygons for {} points...".format(sum(1 for _ in poly_from_lines)))

    # Get average distance between consequitive points to define influence, which should be 2 particles
    mdist = dists.diagonal(1).mean()
    buffer = mdist*influence
    print("Average distance between consecutive particles {:6.4f} [{:6.4f}]".format(mdist,buffer))
    
    j = 0
    print(signal_names)
    for signal_type in signal_names:
        print(signal_type)
        
        dfile =  os.path.join(moran_2_save, "{}_{}_{}.tsv").format(cfile_temp, reso, signal_type)
        pfile =  os.path.join(moran_2_save, "{}_{}_{}_{}.midp").format(cfile_temp, reso, midp, signal_type)
        
        print('\n---> Working on gene {} [{}/{}] and signal {} [{}/{}]'.format(gene_n,i_gene+1,len(genes), signal_type, j+1, len(signal_names)))
        
        j += 1
        
        if os.path.isfile(dfile and force==False):
            print(dfile)
            print("LMI for gene {} and signal {} already done, skipping".format(gene_n, signal_type)) 
            continue
        
        mydata = LMI_data(cfile, info[signal_type], reso, poly_from_lines, coords, buffer)
        
        # Get weights for geometric distance
        print("Getting weights and geometric distance for LM")
        y = mydata['v'].values
        #w = Queen.from_dataframe(mydata)
        w = lp.weights.DistanceBand.from_dataframe(mydata, buffer*bfact)
        w.transform = 'r'

        # # Get Global Moran Index
        # moran = Moran(y, w, permutations=10000)
        # if (math.isnan(moran.I)):
        #     print(">>> ERROR... MATH NAN")
        #     #return()
        # print("Global Moran Index: {:4.2f} with p-val: {:8.6f} ".format(moran.I,moran.p_sim))
        # info_file.write(row.IID + "\t" + str(moran.I) + "\t" + str(moran.p_sim) + "\n")
        
        # Show Global Moran Plot
        # #plot_moran(moran, zstandard=False, figsize=(10,4))

        # calculate Moran_Local and plot
        moran_loc = Moran_Local(y, w, permutations=10000)
        lags = lag_spatial(moran_loc.w, moran_loc.z)
        print("There are a total of {} significant points in Local Moran Index".format(len(moran_loc.p_sim[(moran_loc.p_sim<signipval) & (moran_loc.q == 1)])))

        # Plot results
#        print("Final LMI plots...")
        # plot_local_autocorrelation(moran_loc, mydata, 'v', p=signipval, cmap="coolwarm", scheme='EqualInterval', )
        
        ## The Moran scatterplot of the function
#        plot_2_save = os.path.join(scatter_2_save, "{}_{}.png")
#        fig, axs = plt.subplots(1, 1, figsize=(4.5,4.5), subplot_kw={'aspect': 'equal', 'adjustable':'datalim'})
#        moran_scatterplot(moran_loc, p=signipval, ax=axs)
#        plt.savefig(plot_2_save.format(cfile_temp, signal_type))
        # plt.show()
#        plt.close()
        
        ## The LISA cluster plot (HH, HL, LH, LL)
#        plot_2_save = os.path.join(lisa_2_save, "{}_{}.png")
#        lisa_cluster(moran_loc, mydata, legend=False)
#        plt.savefig(plot_2_save.format(cfile_temp, signal_type))
        # plt.show()
#        plt.close()
        
        ## The Choropleth plot with the data
#        plot_2_save = os.path.join(choro_2_save, "{}_{}.png")
#        mydata.plot(column='v', scheme='EqualInterval', cmap="coolwarm", legend=False, alpha=1)
#        plt.axis('off')
#        plt.savefig(plot_2_save.format(cfile_temp, signal_type))
        # plt.show()
#        plt.close()

        # Get ids for all sites in the LMI map
        print("Getting ids for all sites in the LMI map: {:,}".format(len(mydata)))
        pids = []
        for p in mydata.geometry:
            pids.append(GetPointId(coords,p))

        # midp_pv = moran_loc.p_sim[int(midp)]
        # midp_ty = moran_loc.q[int(midp)]
        # midp_Is = moran_loc.Is[int(midp)]

        # # Get ids of significant High-High
        # # HH=1, LH=2, LL=3, HL=4 on moran_loc.q
        # print("Subsetting significant HH and LH geometries...")
        # mask = (moran_loc.p_sim<=signipval) & (moran_loc.q<=2)
        # HHdata = mydata[mask]

        # # Which points are HH?
        # hhpoints = []
        # #hhpvals = moran_loc.p_sim[(moran_loc.p_sim<=signipval) & (moran_loc.q==1)]
        # for index, ld in HHdata.iterrows():
        #     hhpoints.append(GetPointId(coords,ld.geometry))
        # hhpoints = list(set(hhpoints))
        
        # Save data in file
        print("Saving info into file: {}".format(dfile))
#        f = open(dfile, "w")
#        f.write("Index\tBinIndex\tMoran_quadrant\tLMI_score\tLMI_pvalue\n")
#        #####################################
#        for i,x in enumerate(pids):
#            f.write('{}\t{}\t{}\t{}\t{}\n'.format(i,x,moran_loc.q[i],moran_loc.Is[i],moran_loc.p_sim[i]))
#        #####################################
#        f.close()

        ################################################################################################
        # Collect all info per LD block
        ldbd = pd.DataFrame(columns=['Chr','Start','End','Region','Index','pType','X','Y',
                                        'Signal','ZSig','ZLag','LMIpval','LMIq','LMIs','geometry'])
        print("Saving data {:,} entries for region {}...".format(len(mydata),region))
        chrm,cini,cend = re.split(':|-', region)
        for index, row in mydata.iterrows():
            pid  = GetPointId(coords,row.geometry)
            if (pid+1 == int(midp)):
                ptype = 'midp'
            else:
                ptype = 'other'
            pini = int(cini)+(reso*pid)
            pend = pini+reso-1
            new_row = {'Chr':chrm,
                       'Start':pini,
                       'End':pend,
                       'Region':region,
                       'Index':pid+1,
                       'pType':ptype,
                       'X':coords[pid][0],
                       'Y':coords[pid][1],
                       'Signal':row.v,
                       'ZSig':y[pid],
                       'ZLag':lags[pid],
                       'LMIpval':moran_loc.p_sim[index],
                       'LMIq':moran_loc.q[index],
                       'LMIs':moran_loc.Is[index],
                       'geometry':row.geometry}
            ldbd = ldbd.append(new_row, ignore_index=True)

        # Reindex, resort and save
        ldbd.index = pids
        ldbd = ldbd.sort_index()
        ldbd.to_csv(dfile, sep='\t', index=False)
        ################################################################################################
        print()
        
    # info_file.close()
    
    
endTime = time.perf_counter()

print ('The script took {0} second !'.format(round((endTime - startTime), 2)))
