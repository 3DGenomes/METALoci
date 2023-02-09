#!/usr/bin/env python
import sys
import warnings
warnings.filterwarnings('ignore') # Warnings filtered
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate LMI figure for a list of regions.')

# Required arguments
parser.add_argument('-w','--wfolder', required=True, dest='work_dir',
                    help='Path to the folder where results will be stored. THe script will create subfolders inside.')
parser.add_argument('-c','--cfolder', required=True, dest='cooler_dir',
                    help='Path to the folder where the cooler files are stored. You only need to specify the folder, not each cooler file.')
parser.add_argument('-g','--gfile', required=True, dest='gene_file',
                    help='Path to the file that contains the regions to analyze.')
parser.add_argument('-s','--signal_dir', required=True, dest='signal_dir',
                    help='Path to the folder where the signal bigwig files are stored. Need to have the same name as signals and end with \".bw\". ')
parser.add_argument('-t','--typesignals', required=True, dest='signals2choose',
                    help='Signals to run seperated by \",\". ')
parser.add_argument('-r','--resolution', required=True, dest='reso', type=int,
                    help='Resolution in bp to work at. ')
# Optional arguments
parser.add_argument('-q','--quandrants', required=False, dest='qs', nargs='+', type=int,
                    help='Quandrants to highlight [1,2].')
parser.add_argument('-p','--pval', required=False, dest='signipval', default=0.05,
                    help='P-value to select significant LMIs [0.05].')
parser.add_argument('-d','--del', required=False, dest='rm', default="png",
                    help='Files types to delete to save space [png].')
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
run     = args.work_dir
minpv   = args.signipval
qs      = args.qs
rm      = args.rm.split(',')
reso    = args.reso
silent  = args.silent
force   = args.force
rfile   = args.gene_file
signals = args.signals2choose.split(',')

# Massage arguments
# default quartiles
if not qs:
    qs = [1,2] 
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
print('Importing...')
import matplotlib.pyplot as plt

import os
import re
import time
import datetime
import subprocess
import numpy as np
import fanc as fanc
import cooler
import seaborn as sns
import pandas as pd
import geopandas as gpd
import libpysal as lp
import scipy.stats as stats
import matplotlib.ticker as tick

from scipy.spatial import distance
from scipy.ndimage import rotate
from descartes import PolygonPatch
from matplotlib.lines import Line2D
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from shapely.geometry import Point
from shapely.geometry.multipolygon import MultiPolygon
print("Imports done!")

# FUNCTIONS
def Region2Parts(r):
    c,nu = r.split(':')
    i,e  = nu.split('-')
    return(c,int(i),int(e))
                    
def GetMLData(mlfile):
    # Read data
    #Chr	Start	End	Region	Index	X	Y	Signal	ZSig	Zlag	LMIpval	LMIq	LMIs	geometry
    mlgdata = gpd.read_file(mlfile, GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")
    mlgdata.Signal = mlgdata.Signal.astype('float') # Otherwise geopandas thinks is not a float!
    mlgdata.LMIpval = mlgdata.LMIpval.astype('float') # Otherwise geopandas thinks is not a float!
    mlgdata.Index = mlgdata.Index.astype('int') # Otherwise geopandas thinks is not a float!
    mlgdata.X = mlgdata.X.astype('float') # Otherwise geopandas thinks is not a float!
    mlgdata.Y = mlgdata.Y.astype('float') # Otherwise geopandas thinks is not a float!
    mlgdata.ZSig = mlgdata.ZSig.astype('float') # Otherwise geopandas thinks is not a float!
    mlgdata.ZLag = mlgdata.ZLag.astype('float') # Otherwise geopandas thinks is not a float!
    mlgdata.LMIq = mlgdata.LMIq.astype('int') # Otherwise geopandas thinks is not a float!
    mlgdata.LMIs = mlgdata.LMIs.astype('float') # Otherwise geopandas thinks is not a float!
    return(mlgdata)

def GetHiCPlot(hicfile,region,midp,signal):
    # HiC
    c = cooler.Cooler(hicfile)
    mat = c.matrix(sparse=True, balance=False).fetch(region)
    arr = mat.toarray()
    arr = MatTransform(arr)
    minv= np.min(arr)
    maxv= np.max(arr)
    lmat= arr.shape[0]

    # Massage the matrix
    arr = np.triu(arr, -1)
    arr = rotate(arr, angle=45)
    fac = midp/lmat
    midm= int(arr.shape[0]/2)
    midp= int(arr.shape[0]*fac)
    arr = arr[:midm,:]
    arr[arr == 0] = np.nan

    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = ax.matshow(arr, cmap='YlOrRd', vmin=minv, vmax=maxv)
    sns.scatterplot(x=[midp],y=[midm+4], color='lime', marker="^")
    plt.axis('off')
    #ax.legend().set_visible(False)
    nbar = 6
    sm = plt.cm.ScalarMappable(cmap='YlOrRd')
    cbar = plt.colorbar(sm, ticks=np.linspace(minv,maxv,nbar), 
                 values=np.linspace(minv,maxv,nbar), 
                 shrink=0.3)
    cbar.set_label('log10(Hi-C interactions)', rotation=270, size=14, labelpad=20)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    plt.title('{} [{}] {}'.format(gene,region, signal))
    return(plt)

def GetKKPlot(mlgdata,region,midp):
    # Kamada Kawai Plot
    chrm,cini,cend = Region2Parts(region)
    plotsize = 10
    poinsize = plotsize*5
    fig = plt.figure(figsize=(plotsize, plotsize))
    x = mlgdata.X
    y = mlgdata.Y
    plt.axis('off')
    g = sns.lineplot(x=x, y=y, sort=False, lw=1, color='grey', legend = False)
    g.set_aspect('equal', adjustable='box')
    sns.scatterplot(x=x, y=y, hue=range(0,len(x)), palette='coolwarm', legend = False, s=poinsize)
    g.set(ylim=(-1.1, 1.1))
    g.set(xlim=(-1.1, 1.1))
    g.tick_params(bottom=False, left=False)
    g.annotate('{}:{:,}'.format(chrm,cini), (x[0], y[0]))
    g.annotate('{}:{:,}'.format(chrm,cend), (x[len(x)-1], y[len(y)-1]))
    sns.scatterplot(x=[x[midp-1]], y=[y[midp-1]], s=poinsize*2, ec="lime", fc="none")
    return(plt)

def GaudiSignalPlot(mlgdata,region,midp,vmin,vmax):
    # Gaudi plot Signal
    cmap = 'copper_r'
    cmap = 'PuOr_r'
    x    = mlgdata.X
    y    = mlgdata.Y
    q1 = mlgdata.Signal.quantile(0.95)
    q2 = mlgdata.Signal.quantile(0.05)
    lims = max(abs(q1),abs(q2))
    if (np.sign(q1) == np.sign(q2)):
        vmax = q1
        vmin = q2
    if (vmax):
        maxv = vmax
    else:
        #maxv = max(mlgdata.Signal)
        maxv = lims
    if (vmin):
        minv = vmin
    else:
        #minv = min(mlgdata.Signal)
        minv = -lims
    polh = mlgdata[mlgdata.Index == midp]
    fig, ax = plt.subplots(figsize=(12,10), subplot_kw={'aspect':'equal'})
    mlgdata.plot(column='Signal', cmap=cmap, linewidth=2, edgecolor='white', vmax=maxv, vmin=minv, ax=ax)
    sns.scatterplot(x=[x[midp-1]], y=[y[midp-1]], s=50, ec="none", fc="lime")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    nbar = 11
    cbar = plt.colorbar(sm, ticks=np.linspace(minv,maxv,nbar), 
                 values=np.linspace(minv,maxv,nbar), 
                 shrink=0.5)
    #cbar.ax.label_params(labelsize=20)
    cbar.set_label('Signal', rotation=270, size=20, labelpad=35)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    plt.axis('off')
    return(plt)

def GaudiTypePlot(mlgdata,quartiles,minpv,region,midp):
    # Gaudi plot LMI type
    x     = mlgdata.X
    y     = mlgdata.Y
    colors= ['firebrick', 'lightskyblue', 'steelblue', 'orange']
    fig, ax = plt.subplots(figsize=(12,10), subplot_kw={'aspect':'equal'})
    ax = fig.gca() 
    alphas = []
    polc   = []
    for i,row in mlgdata.iterrows():
        pol = row.geometry
        pty = row.LMIq
        ppv = row.LMIpval
        polc.append(colors[pty-1])
        if (ppv<=minpv):
            alphas.append(1.0)
        else:
            alphas.append(0.3)
        #ax.add_patch(PolygonPatch(pol, ec='white', fc=colors[pty-1], alpha=alpha, linewidth=2))
    mlgdata.plot(column='LMIq', linewidth=2, edgecolor='white', color=polc, alpha=alphas, ax=ax)
    sns.scatterplot(x=[x[midp-1]], y=[y[midp-1]], s=50, ec="none", fc="lime", zorder=len(mlgdata))
    plt.axis('off')
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], label='HH', markersize=15),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], label='LH', markersize=15),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], label='LL', markersize=15),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], label='HL', markersize=15)]
    plt.legend(handles=legend_elements, frameon=False, fontsize=15, loc='center right', bbox_to_anchor=(1.2, 0.5))
    return(plt)

# THIS NEEDS TO BE FIXED AT THE LEVEL OF SCRIPT 01.
# The problem is that ZSig or ZLag are not really Z-scored and NOT correctly ordered!!!
# Once fixed in SCRIPT 01, we will go back here.
def MLIScatterPlotFastNotWorking(mlgdata,quartiles,minpv,region,midp):
    # LM I's scatter
    colors= ['firebrick', 'lightskyblue', 'steelblue', 'orange']
    #x,y = mlgdata.ZSig,mlgdata.ZLag
    mlgdata.ZSig = stats.zscore(mlgdata.ZSig)
    mlgdata.ZLag = stats.zscore(mlgdata.ZLag)
    print('\tGet spatial slope...')
    slope, intercept, r_value, p_value, std_err = stats.linregress(mlgdata.ZSig,mlgdata.ZLag)
    print('\tGet spatial plot...')
    fig, ax = plt.subplots(figsize=(5,5))#, subplot_kw={'aspect':'equal'})
    for i,row in mlgdata.iterrows():
        idx = row.Index
        pty = row.LMIq
        ppv = row.LMIpval
        x   = row.ZSig
        y   = row.ZLag
        if (ppv<=minpv):
            alpha = 1.0
        else:
            alpha = 0.1
        print(idx,pty,ppv,x,y,colors[pty-1],alpha)
#        plt.scatter(x=x,y=y, s=100, ec='white', fc=colors[pty-1], alpha=alpha)
    #sns.scatterplot(x=[x[midp-1]], y=[y[midp-1]], s=150, ec="lime", fc="none", zorder=len(mlgdata))
#    sns.regplot(x=x, y=y, scatter=False, color='k')
#    plt.title('Moran Local Scatterplot\nr: {:4.2f}   p-value: {:.1e}'.format(r_value, p_value))
#    plt.axvline(x=0, color='k', linestyle=':')
#    plt.axhline(y=0, color='k', linestyle=':')
#    ax.set_xlabel('Z-score(Signal)')
#    ax.set_ylabel('Z-score(Signal Spatial Lag)')
#    sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=False)
#    plt.xlim(min(x)-0.5,max(x)+0.5)
#    plt.ylim(min(y)-0.5,max(y)+0.5)
    return(plt)

def MLIScatterPlot(mlgdata,quartiles,minpv,region,midp):
    # LM I's scatter
    # Get distances to get spatial lag
    #print('\tGet spatial Lag...')
    coords = mlgdata[['X', 'Y']].values.tolist()
    dists = distance.cdist(coords, coords, 'euclidean')
    influence = 1.5
    bfact = 2
    colors= ['firebrick', 'lightskyblue', 'steelblue', 'orange']
    mdist = dists.diagonal(1).mean()
    buffer = mdist*influence
    wq =  lp.weights.DistanceBand.from_dataframe(mlgdata,buffer*bfact)
    wq.transform = 'r'
    y = mlgdata['Signal']
    ylag = lp.weights.lag_spatial(wq, y)
    x,y = stats.zscore(mlgdata.Signal),stats.zscore(ylag)
    #print('\tGet spatial slope...')
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #print('\tGet spatial plot...')
    fig, ax = plt.subplots(figsize=(5,5))#, subplot_kw={'aspect':'equal'})
    for i,row in mlgdata.iterrows():
#        print(i,'...')
        pty = row.LMIq
        ppv = row.LMIpval
        if (ppv<=minpv):
            alpha = 1.0
        else:
            alpha = 0.1
        plt.scatter(x=x[i],y=y[i], s=100, ec='white', fc=colors[pty-1], alpha=alpha)
    #print('\tGet spatial point...')
    sns.scatterplot(x=[x[midp-1]], y=[y[midp-1]], s=150, ec="lime", fc="none", zorder=len(mlgdata))
    #print('\tGet regression...')
    sns.regplot(x=x, y=y, scatter=False, color='k')
    #print('\tGet decoration...')
    plt.title('Moran Local Scatterplot\nr: {:4.2f}   p-value: {:.1e}'.format(r_value, p_value))
    plt.axvline(x=0, color='k', linestyle=':')
    plt.axhline(y=0, color='k', linestyle=':')
    ax.set_xlabel('Z-score(Signal)')
    ax.set_ylabel('Z-score(Signal Spatial Lag)')
    sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=False)
    plt.xlim(min(x)-0.5,max(x)+0.5)
    plt.ylim(min(y)-0.5,max(y)+0.5)
    #print('\tReturning...')
    return(plt)

def SignalPlotBed(mlgdata,quartiles,minpv,region,midp):
    # Get Bed file
    chrm,cini,cend = Region2Parts(region)
    reso = (cend-cini)/len(mlgdata)
    coords= mlgdata[['X', 'Y']].values.tolist()
    dists = distance.cdist(coords, coords, 'euclidean')
    midds = dists[midp-1]
    mdist = dists.diagonal(1).mean()
    influence = 1.5
    bfact = 2
    buffer = mdist*influence
    close = midds[midds<=buffer*bfact]
    selsebins = []
    selsebins = mlgdata[(mlgdata.LMIq.isin(quartiles)) & (mlgdata.LMIpval <= minpv)].Index.values.tolist()
    mls = mlgdata[mlgdata.Index.isin(selsebins)].unary_union
    if (mls):
        if mls.geom_type == 'Polygon':
            mls = MultiPolygon([mls])
    x,y = mlgdata[mlgdata.Index==midp].X,mlgdata[mlgdata.Index==midp].Y
    s = Point((x,y))
    selmetaloci = []
    beddata = pd.DataFrame(columns=['chr','start','end','bin'])
#    print(mls)
#    for i, ml in enumerate(mls):
#        ml = gpd.GeoSeries(ml)
#        if (s.within(ml[0])):
#            if (silent==False):
#                ax = ml.plot()
#                sns.scatterplot(x=x,y=y, color='red', ax=ax)
#                plt.show()
#                plt.close()
#            for j,row in mlgdata.iterrows():  
#                p,x,y = row.Index,row.X,row.Y
#                s2 = Point((x,y))
#                if (s2.within(ml[0])):
#                    selmetaloci.append(p)
#            # Add close particles
#            selmetaloci.sort()
#            closebins = [i for i, val in enumerate(midds) if val<=buffer*bfact]
#            selmetaloci = np.sort(list(set(closebins + selmetaloci)))
#            print('\tGetting BED...')
#            for p in selmetaloci:
#                pini = int(cini+(p*reso))
#                pend = int(pini+reso-1)
#                bed = '{}\t{}\t{}\t{}'.format(chrm,pini,pend,p)
#                row = {'chr': chrm,
#                       'start': pini,
#                       'end': pend,
#                       'bin': p}
#                beddata = beddata.append(row, ignore_index=True)
    # Signal plot
    print('\tSignal profile...')
    fig = plt.figure(figsize=(10, 1))
    x = mlgdata.Index
    y = mlgdata.Signal
    g = sns.lineplot(x=x,y=y)
    for p in selmetaloci:
        plt.axvline(x=p, color='red', linestyle=':', lw=0.5)
    plt.axvline(x=midp, color='lime', linestyle='--', lw=0.5)
    bins = list(range(50,len(mlgdata),int((len(mlgdata)+52)/6)))
    coords = ['{:,}'.format(int(cini+x*reso)) for x in bins]
    plt.xticks(bins,coords)    
    #g.yaxis.set_major_locator(MaxNLocator(integer=True))
    g.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xlabel('chromosome {}'.format(chrm))
    g.margins(x=0)
    sns.despine(top=True, right=True, left=False, bottom=True, offset=None, trim=False)
    return(plt,beddata)

def PlaceImage(new,ifile,ifactor,ixloc,iyloc):
    img = Image.open(ifile)
    niz = tuple([int(i*ifactor) for i in img.size])
    img = img.resize(niz, Image.ANTIALIAS)
    new.paste(img, (ixloc,iyloc))
    return(new)

def MatTransform(mat):
    if (silent == False):
        print('Matrix data...')
        print('\tMin: {}, Max: {}'.format(np.min(mat), np.max(mat)))
    
    diag = np.array(mat.diagonal())
    
    for i in range(0, len(diag)):
        if (diag[i] == 0):
            
            mat[i] = 0
            mat[:, i] = 0

    # Pseudocounts if min is zero
    if (np.min(mat) == 0):
        pc = np.min(mat[mat>0])
        if (silent == False):
            print('Pseudocounts: {}'.format(pc))
        mat = mat+pc

    # Scale if all below 1
    if (np.max(mat)<=1 or np.min(mat)<=1): ## Why np.min?
        sf = 1/np.min(mat)
        if (silent == False):
            print('Scaling factor: {}'.format(sf))
        mat = mat*sf
        
    if (np.min(mat)<1):
        mat[mat<1] = 1
        
    if (silent == False):
        print('\tMin: {}, Max: {}'.format(np.min(mat), np.max(mat)))

    # Log10 the data
    if (silent == False):
        print('Log10 matrix...')
    mat = np.log10(mat)
    if (silent == False):
        print('\tMin: {}, Max: {}'.format(np.min(mat), np.max(mat)))
   
    # Mean of the data
    if (silent == False):
        print('Mean of non zero...')
    me = mat[mat>1].mean()
    if (silent == False):
        print('\tMean: {}'.format(me))
        print()
    
    return(mat)

def AnaML(work_dir,gene,midp,signal,mlname,mlfile,hicfile,minpv,quartiles,silent,rmfiles):
    
    # Create directory if does not exist
    if not os.path.exists(work_dir+"/mls/figures/"+mlname):
        os.makedirs(work_dir+"/mls/figures/"+mlname)
    else:
        print('\tFolder already exists... overwriting...')
    
    # Get data
    print('\tReading data file: {}'.format(mlfile))
    mlgdata = GetMLData(mlfile)
    if (silent == False):
        print(mlgdata.head())   
    
    # Define region coordinates
    chrm   = mlgdata.Chr[0]
    cini   = int(mlgdata.Start[0])
    cend   = int(mlgdata.End[0])
    region = mlgdata.Region[0]
    print('\tWorking on region: {}'.format(region))
    
    # Hi-C data and plot
    print('\tHi-C matrix...')
    hic_plt = GetHiCPlot(hicfile,region,midp,signal)
    hic_pdf = '{0}/mls/figures/{1}/{1}_hic.pdf'.format(work_dir,mlname)
    hic_png = '{0}/mls/figures/{1}/{1}_hic.png'.format(work_dir,mlname)
    hic_plt.savefig(hic_pdf, bbox_inches='tight', transparent=True)    
    hic_plt.savefig(hic_png, bbox_inches='tight', dpi=300, transparent=True)
    if (silent == False):
        hic_plt.show()
    hic_plt.close() 
    
    # Kamada-Kawai plot
    print('\tKamada-Kaway layout...')
    kk_plt = GetKKPlot(mlgdata,region,midp)
    kk_pdf = '{0}/mls/figures/{1}/{1}_kk.pdf'.format(work_dir,mlname)
    kk_png = '{0}/mls/figures/{1}/{1}_kk.png'.format(work_dir,mlname)
    kk_plt.savefig(kk_pdf, bbox_inches='tight', transparent=True)    
    kk_plt.savefig(kk_png, bbox_inches='tight', dpi=300, transparent=True)
    if (silent == False):
        kk_plt.show()
    kk_plt.close() 
    
    # Gaudi plot Signal
    print('\tGaudi signal...')
    gs_plt = GaudiSignalPlot(mlgdata,region,midp,None,None)
    gs_pdf = '{0}/mls/figures/{1}/{1}_ml_signal.pdf'.format(work_dir,mlname)
    gs_png = '{0}/mls/figures/{1}/{1}_ml_signal.png'.format(work_dir,mlname)
    gs_plt.savefig(gs_pdf, bbox_inches='tight', transparent=True)    
    gs_plt.savefig(gs_png, bbox_inches='tight', dpi=300, transparent=True)
    if (silent == False):
        gs_plt.show()
    gs_plt.close() 
    
    # Gaudi plot Type
    print('\tGaudi type...')
    gt_plt = GaudiTypePlot(mlgdata,quartiles,minpv,region,midp)
    gt_pdf = '{0}/mls/figures/{1}/{1}_type.pdf'.format(work_dir,mlname)
    gt_png = '{0}/mls/figures/{1}/{1}_type.png'.format(work_dir,mlname)
    gt_plt.savefig(gt_pdf, bbox_inches='tight', transparent=True)    
    gt_plt.savefig(gt_png, bbox_inches='tight', dpi=300, transparent=True)
    if (silent == False):
        gt_plt.show()
    gt_plt.close() 
    
    # Local Moran I scatter plot
    print('\tLocal Moran I scatter...')
    mli_plt = MLIScatterPlot(mlgdata,quartiles,minpv,region,midp)
    #print('\tReturned...')
    mli_pdf = '{0}/mls/figures/{1}/{1}_mli.pdf'.format(work_dir,mlname)
    mli_png = '{0}/mls/figures/{1}/{1}_mli.png'.format(work_dir,mlname)
    mli_plt.savefig(mli_pdf, bbox_inches='tight', transparent=True)    
    mli_plt.savefig(mli_png, bbox_inches='tight', dpi=300, transparent=True)
    #print('\tSaved...')
    if (silent == False):
        mli_plt.show()
    mli_plt.close() 
    #print('\tClosed..')
    
    # Signal Plot and Bed file
    print('\tSignal plot and bed file (if relevant)...')
    s_plt,bed_data = SignalPlotBed(mlgdata,quartiles,minpv,region,midp)
    s_pdf = '{0}/mls/figures/{1}/{1}_signal.pdf'.format(work_dir,mlname)
    s_png = '{0}/mls/figures/{1}/{1}_signal.png'.format(work_dir,mlname)
    if (len(bed_data)):
        print('\t\tSaving bed data with {} bins...'.format(len(bed_data)))
        b_file= '{0}/mls/figures/{1}/{1}.bed'.format(work_dir,mlname)
        bed_data.to_csv(b_file, index=False, sep='\t')
    s_plt.savefig(s_pdf, bbox_inches='tight', transparent=True)    
    s_plt.savefig(s_png, bbox_inches='tight', dpi=300, transparent=True)
    if (silent == False):
        s_plt.show()
    s_plt.close() 
    
    # ALL TOGETHER
    print('\tFinal composite figure:')
    img1 = Image.open(mli_png)
    img2 = Image.open(gs_png)
    img3 = Image.open(gt_png)
    maxx = int((img1.size[1]*0.4+img2.size[1]*0.25+img3.size[1]*0.25)*1.3)
    new  = Image.new("RGBA", (maxx,1550))
    # HiC image 
    new = PlaceImage(new,hic_png,0.5,100,50)
    # Singal image
    new = PlaceImage(new,s_png,0.4,42,660)
    # KK image
    new = PlaceImage(new,kk_png,0.3,1300,50)
    # MLI scatter image
    new = PlaceImage(new,mli_png,0.4,75,900)
    # Gaudi signal image
    new = PlaceImage(new,gs_png,0.25,900,900)
    # Gaudi signi image
    new = PlaceImage(new,gt_png,0.25,1600,900)
    
    # Remove files?
    if (rmfiles):
        for rf in rmfiles:
            folderfiles = os.listdir(work_dir+"/mls/figures/"+mlname)
            for item in folderfiles:
                if item.endswith(".{}".format(rf)):
                    os.remove(os.path.join(work_dir+"/mls/figures/"+mlname, item))
    
    # Save image
#    if (silent == False):
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(new)
    plt.axis('off')
    plt.show()
    plt.close()
    print('\tFinal figure saved here: {0}/mls/figures/{1}/{1}.png'.format(work_dir,mlname))
    new.save('{0}/mls/figures/{1}/{1}.png'.format(work_dir,mlname))
    
    # Return composite image
    return(new)
    
#####################
### START THE ACTION!
#####################

# Regions to work on
reglst = pd.read_csv(rfile, sep='\t')

# Create plots≤
start_time = time.time()
for i, region in reglst.iterrows():
    for signal in signals:
        print ('>>>> {}\t{}\t{} [{}/{}]'.format(region.id, run, signal, i+1, reglst.shape[0]))
        (chrm,start,end,midp) = re.split(':|_|-',region.coords)
        start = int(start)
        end   = int(end)
        midp  = int(midp)#-1 # TO MAKE IT PYTHON BASED (0->N instead 1->N)
        gene  = region.id
        linux_region = '{}_{}_{}'.format(chrm,start,end)
        mlname  = '{}_{}_{}'.format(linux_region,reso,signal)
        mlfile  = '{}/mls/datas/{}/{}.tsv'.format(run,chrm,mlname)
        runname = '{}_{}_{}'.format(mlname,gene,run)
        print(gene,runname,mlfile,minpv,qs,silent,rm)
        if os.path.isfile(mlfile):
            mldata  = pd.read_csv(mlfile,sep='\t')
            #grep_com = "ls -1v "+cooler_dir+" | grep \".mcool\""
            #hicfile = cooler_dir+subprocess.getoutput(grep_com)+"::/resolutions/"+str(reso)
            grep_com = "ls -1v "+cooler_dir+" | grep mcool"
            grep_res = subprocess.getoutput(grep_com)
            #print("GC>>>>>>>>>>>", grep_res)
            if (len(grep_res.split()) > 1): # Multiple chromosomes files in folder
                grep_com = "ls -1v "+cooler_dir+" | grep mcool | grep _"+chrm+"_"
                grep_res = subprocess.getoutput(grep_com)
            #print("GP>>>>>>>>>>>", grep_res)
            hicfile = cooler_dir+grep_res+"::/resolutions/"+str(reso)
            print("Will use {} as cool file...".format(hicfile))
            if os.path.isdir(work_dir+"/mls/figures/"+runname) and force==False:
                print('WARNING! {} ALREADY DONE! If you want to recreate it, use -f or --force ...'.format(work_dir+"/mls/figures/"+runname))
            else:
                print('Generating {}'.format(work_dir+"/mls/figures/"+runname))
                figure  = AnaML(work_dir,gene,midp,signal,runname,mlfile,hicfile,minpv,qs,silent,rm)
        else:
            print('ERROR! File {} does not exists. Cannot produce the figure...'.format(mlfile))
            continue
    print ('>>>>\n\n')
print("--- {} ---\n".format(str(datetime.timedelta(seconds=(time.time() - start_time)))))
