# Author: Leo Zuber
# Date: July 2024

"""
Find the best combination persistence length and cut-off for your Hi-C, given a Hi-C resolution.

Check first the maximum resolution yout Hi-C allows, and supply it to this script. This script will then determine, by
computing layouts on a sample of regions, the best combination of persistence length and cut-off for your Hi-C. You can 
then use these parameters to run 'metaloci layout' on your regions of interest.

This can take from minutes to a few hours, depending on the resolution, the size of the regions and the number of 
cpu's available in your machine. Once you run it for a specific Hi-C and a specific resolution, the parameters will
remain the same for other runs with other signals.
"""
import glob
import multiprocessing as mp
import os
import pathlib
import subprocess as sp
import sys
from argparse import SUPPRESS, HelpFormatter
from datetime import timedelta
from itertools import product
from time import time

import cooler
import hicstraw
import networkx as nx
import numpy as np
import pandas as pd
from metaloci import mlo
from metaloci.graph_layout import kk
from metaloci.misc import misc
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

HELP = """
Finds the best combination of parameters for your Hi-C.
"""

DESCRIPTION = """
Find the best combination persistence length and cut-off for your Hi-C, given a Hi-C resolution.\n

Check first the maximum resolution yout Hi-C allows, and supply it to this script. This script will then determine, by
computing layouts on a sample of regions, the best combination of persistence length and cut-off for your Hi-C. You can 
then use these parameters to run 'metaloci layout' on your regions of interest.\n

This can take from minutes to a few hours, depending on the resolution, the size of the regions and the number of 
cpu's available in your machine. Once you run it for a specific Hi-C and a specific resolution, the parameters will
remain the same for other runs with other signals.\n
"""


def populate_args(parser):
    """
    Populate the ArgumentParser with the arguments needed for the METALoci caller.

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller.
    """

    parser.formatter_class = lambda prog: HelpFormatter(prog, width=120,
                                                        max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")

    input_arg.add_argument('-w',
                           '--work-dir',
                           dest='work_dir',
                           metavar='PATH',
                           required=True,
                           type=str,
                           help='Path to working directory'
                           )

    input_arg.add_argument('-c',
                           '--hic',
                           dest='hic',
                           metavar='PATH',
                           required=True,
                           type=str,
                           help='Complete path to the cool/mcool/hic file',
                           )

    input_arg.add_argument('-r',
                           "--resolution",
                           dest="resolution",
                           metavar="INT",
                           required=True,
                           type=int,
                           nargs='+',
                           help="List of Hi-C resolutions to be tested (in bp)."
                           )

    input_arg.add_argument(
                            "-g",
                            "--region",
                            dest="regions",
                            metavar="PATH",
                            type=str,
                            help="Region to apply LMI in format chrN:start-end_midpoint or file with the regions of interest. If a file \
                            is provided, it must contain as a header 'coords', 'symbol' and 'id', and one region per line, tab separated.",
                        )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument("-h",
                              "--help",
                              action="help",
                              help="Show this help message and exit.")

    optional_arg.add_argument('-s',
                              '--seed',
                              dest='seed',
                              metavar='INT',
                              type=int,
                              default=1,
                              help="Random seed for region sampling. (default: %(default)s)"
                              )

    optional_arg.add_argument('-n',
                              '--sample_num',
                              dest='sample_num',
                              metavar='int',
                              type=int,
                              default=100,
                              help="Number of regions to sample from .txt file. (default: %(default)s)",
                              )

    optional_arg.add_argument('-o',
                              '--cutoffs',
                              dest='cutoffs',
                              metavar='FLOAT',
                              type=float,
                              nargs='+',
                              default=[0.15, 0.175, 0.2, 0.225, 0.25],
                              help="""Percent of top interactions to use from HiC.
                              METALoci Default %(default)s""")

    optional_arg.add_argument('-l',
                              '--pls',
                              dest='pls',
                              metavar='FLOAT',
                              type=float,
                              nargs='+',
                              default=None,
                              help='Persistence length; usual values are between 9 and 12, although this can vary a '
                                'lot depending on the absolute values of your Hi-C matrix.')

    optional_arg.add_argument("-t"
                              "--threads",
                              dest="threads",
                              metavar="INT",
                              default=int(mp.cpu_count() - 2),
                              type=int,
                              action="store",
                              help="Number of threads to use in multiprocessing. (default: %(default)s)"
                              )

    optional_arg.add_argument("-u",
                              "--debug",
                              dest="debug",
                              action="store_true",
                              help=SUPPRESS)


def bts_ratio(hic_path: str, resolution: int, region: str, cutoff: float, pl: float) -> float:

    """
    Calculate the ratio of the correlation between the linear and spherical layouts against the layout we actually
    want to use. This is a good mesure to estimate the best set of parameters for the Kamada-Kawai layout.

    Returns
    -------
    linear_correlation / spherical_correlation : float
        Ratio of the correlation between the linear and spherical layouts.
    """

    mlobject = mlo.MetalociObject(
            region=region,
            resolution=resolution,
            persistence_length=pl,
            save_path=None,
    )

    mlobject.kk_cutoff["cutoff_type"] = "percentage"
    mlobject.kk_cutoff["values"] = cutoff

    if hic_path.endswith(".cool"):

        mlobject.matrix = cooler.Cooler(hic_path).matrix(sparse=True).fetch(mlobject.region).toarray()

    if hic_path.endswith(".mcool"):

        mlobject.matrix = cooler.Cooler(
            f"{hic_path}::/resolutions/{mlobject.resolution}").matrix(sparse=True).fetch(mlobject.region).toarray()

    elif hic_path.endswith(".hic"):

        mlobject.matrix = hicstraw.HiCFile(hic_path).getMatrixZoomData(
            mlobject.chrom, mlobject.chrom, 'observed', 'VC_SQRT', 'BP', mlobject.resolution).getRecordsAsMatrix(
            mlobject.start, mlobject.end, mlobject.start, mlobject.end)

    mlobject = misc.clean_matrix(mlobject)
    
    if mlobject.matrix is None and mlobject.bad_regions is not None:

        return np.nan

    else:

        mlobject_test = kk.get_restraints_matrix(mlobject, optimise = True, silent=True)
        mlobject_test.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject_test.kk_restraints_matrix))
        mlobject_test.kk_nodes = nx.kamada_kawai_layout(mlobject_test.kk_graph)

        xy = np.array([mlobject_test.kk_nodes[n] for n in mlobject_test.kk_nodes])

        mlobject_linear = mlobject
        mlobject_linear.kk_cutoff["values"] = 0.01

        mlobject_linear = kk.get_restraints_matrix(mlobject_linear, optimise = True, silent=True)
        mlobject_linear.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject_linear.kk_restraints_matrix))
        mlobject_linear.kk_nodes = nx.kamada_kawai_layout(mlobject_linear.kk_graph)

        xy_linear = np.array([mlobject_linear.kk_nodes[n] for n in mlobject_linear.kk_nodes])

        mlobject_spherical = mlobject
        mlobject_spherical.kk_cutoff["values"] = 0.6

        mlobject_spherical = kk.get_restraints_matrix(mlobject_spherical, optimise = True, silent=True)
        mlobject_spherical.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject_spherical.kk_restraints_matrix))
        mlobject_spherical.kk_nodes = nx.kamada_kawai_layout(mlobject_spherical.kk_graph)

        xy_spherical = np.array([mlobject_spherical.kk_nodes[n] for n in mlobject_spherical.kk_nodes])

        linear_correlation = pearsonr(xy.flatten(), xy_linear.flatten())[0]
        spherical_correlation = pearsonr(xy.flatten(), xy_spherical.flatten())[0]

    return linear_correlation / spherical_correlation

    
def param_search(row: pd.Series, args: pd.Series, progress = None):
    """
    Test METALoci parameters to optimise Kamada-Kawai layout.

    Parameters
    ----------
    row : pd.Series
        Row of the region file.
    args : pd.Series
        Arguments to test.
    progress : mp.Manager().dict
        Progress bar.
    """

    for arg_set in args:

        work_dir, hic, resolution, pl, cutoff, sample_num = arg_set
        
        pathlib.Path(os.path.join(work_dir, "bts")).mkdir(parents=True, exist_ok=True)

        arg_set_string = f"{int(int(resolution) / 1000)}_kb_cutoff_{cutoff:.3f}_pl_{pl}"
        ratio = bts_ratio(hic, resolution, row.coords, cutoff, pl)

        with open(f"{work_dir}bts/{arg_set_string}.txt", "a") as handler:

            handler.write(f"{row.coords}_{arg_set_string}\t{ratio}\n")
            handler.flush()

    if progress is not None:

        progress['value'] += 1
        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t{progress['value']}/{sample_num} done.", end='\r')


def sum_hic_columns(hic_path: str, resolution: int, region: str, cutoff: float) -> float:
    """
    Function to count the number of interacctions in a subseted Hi-C matrix.

    Parameters
    ----------
    hic_path : str
        Path to the Hi-C file.
    resolution : int
        Resolution of the Hi-C file.
    region : tuple
        Region of interest.
    cutoff : float
        Cutoff to use for the subset matrix.
    
    Returns
    -------
    median_sum : float
        Median sum of the interactions in the subset matrix.
    """

    mlobject = mlo.MetalociObject(
            region=region,
            resolution=resolution,
            persistence_length=None,
            save_path=None,
    )

    mlobject.kk_cutoff["cutoff_type"] = "percentage"
    mlobject.kk_cutoff["values"] = cutoff

    if hic_path.endswith(".cool"):

        mlobject.matrix = cooler.Cooler(hic_path).matrix(sparse=True).fetch(mlobject.region).toarray()

    if hic_path.endswith(".mcool"):

        mlobject.matrix = cooler.Cooler(
            f"{hic_path}::/resolutions/{mlobject.resolution}").matrix(sparse=True).fetch(mlobject.region).toarray()

    elif hic_path.endswith(".hic"):

        mlobject.matrix = hicstraw.HiCFile(hic_path).getMatrixZoomData(
            mlobject.chrom, mlobject.chrom, 'observed', 'VC_SQRT', 'BP', mlobject.resolution).getRecordsAsMatrix(
            mlobject.start, mlobject.end, mlobject.start, mlobject.end)

    mlobject = kk.get_subset_matrix(misc.clean_matrix(mlobject), silent=True)

    # for every value in the diagonal, sum all the values in that column that are below the diagonal
    if not mlobject.subset_matrix is None:
        
        median_sum = np.nanmean([np.nansum(mlobject.subset_matrix[i, i+1:]) for i in range(len(mlobject.subset_matrix))])

        return median_sum


def pl_estimation(row: pd.Series, args: pd.Series, progress = None) -> float:
    """
    Wrapper function to count the number of interacctions in a subseted Hi-C matrix.

    Parameters
    ----------
    row : pd.Series
        Row of the region file.
    args : pd.Series
        Arguments to test.
    progress : mp.Manager().dict
        Progress bar.

    Returns
    -------
    sum_dict : dict
        Dictionary with the sum of interactions for each set of parameters
    """

    sum_dict = {}

    for arg_set in args:

        _, hic, resolution, cutoff, sample_num = arg_set

        sum_dict[f"{arg_set}"] = sum_hic_columns(hic, resolution, row.coords, cutoff)

    if progress is not None:

        progress['value'] += 1
        print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
        print(f"\t{progress['value']}/{sample_num} done.", end='\r')

    return sum_dict


def run(opts: list):
    """
    Main function to run bts.

    Parameters
    ----------
    opts : list
        List of arguments
    """

    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

    if opts.regions is None:

        if len(glob.glob(f"{opts.work_dir}*coords.txt")) > 1:

            sys.exit("More than one region file found. Please, provide a region or one file with regions of interest.")

        try:

            df_regions = pd.read_table(glob.glob(f"{opts.work_dir}*coords.txt")[0])

        except IndexError:

            sys.exit("No regions provided. Please provide a region or a file with regions of interest or run "
                     "'metaloci sniffer'.")

    elif os.path.isfile(opts.regions):

        df_regions = pd.read_table(opts.regions)

    else:

        df_regions = pd.DataFrame({"coords": [opts.regions], "symbol": ["symbol"], "id": ["id"]})

    if opts.cutoffs is None:

        opts.cutoffs = [0.2]

    if opts.debug:

        print(f"work_dir is:\n\t{opts.work_dir}\n")
        print(f"hic is:\n\t{opts.hic}\n")
        print(f"resolution is:\n\t{opts.resolution}\n")
        print(f"region_files is:\n\t{opts.regions}\n")
        print(f"sample_num is:\n\t{opts.sample_num}\n")
        print(f"seed is:\n\t{opts.seed}\n")
        print(f"pls is:\n\t{opts.pls}\n")
        print(f"cutoffs is:\n\t{opts.cutoffs}\n")
        print(f"threads is:\n\t{opts.threads}\n")
        # print(f"resolution_region_dict is:\n\t{resolution_region_dict}")
        sys.exit(0)
    
    if os.path.exists(f"{opts.work_dir}bts"):

        sp.check_call(f"rm -r {opts.work_dir}bts", shell=True)
        
    if not opts.pls is None:
        
        
        parsed_args = pd.Series([(opts.work_dir, opts.hic, opts.resolution[0], pl, cutoff, opts.sample_num) 
                                 for pl, cutoff in product(opts.pls, opts.cutoffs)]) 
        
    else:

        if len(df_regions) >= 4000:

            number_of_regions_pl = 4000

        else:

            number_of_regions_pl = len(df_regions)

        subset_pl = df_regions.sample(n=number_of_regions_pl, random_state=opts.seed)
        parsed_args = pd.Series([(opts.work_dir, opts.hic, opts.resolution[0], cutoff, len(subset_pl))
                                    for cutoff in opts.cutoffs])

    try:

        with mp.Pool(processes=opts.threads) as pool:
            
            if opts.pls is None:
                
                start_timer = time()
                progress = mp.Manager().dict(value=0, timer=start_timer)

                print("Estimating persistence length according to the Hi-C matrix.\n")
                print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
                print(f"\t{progress['value']}/{number_of_regions_pl} done.", end='\r')

                pls_dict = list(pool.starmap(pl_estimation, [(row, parsed_args, progress) for _, row in subset_pl.iterrows()]))
                pls_dict = {key: value for dic in pls_dict for key, value in dic.items()}
                predicted_pl = np.mean([v for v in pls_dict.values() if not np.isnan(v)]) / 3.5
                pls = [predicted_pl - predicted_pl * 0.1, predicted_pl, predicted_pl + predicted_pl * 0.1]
                pls = [round(pl, 3) for pl in pls]


                print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
                print(f"\t{number_of_regions_pl}/{number_of_regions_pl} done.", end='\r')

                print(f"\n\nThese persistence lengths will be tested: {pls}\n")

                parsed_args = pd.Series([(opts.work_dir, opts.hic, opts.resolution[0], pl, cutoff, opts.sample_num) 
                                         for pl, cutoff in product(pls, opts.cutoffs)])
                
            print(f"------> {opts.sample_num} regions will be tested for optimisation.")
            print("This may take a while.")

            start_timer = time()
            progress = mp.Manager().dict(value=0, timer=start_timer)
            
            print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
            print(f"\t{progress['value']}/{opts.sample_num} done.", end='\r')

            df_regions = df_regions.sample(n=opts.sample_num, random_state=opts.seed)

            pool.starmap(param_search, [(row, parsed_args, progress) for _, row in df_regions.iterrows()])

    except KeyboardInterrupt:

            pool.terminate()
            sp.check_call(f"rm -r {opts.work_dir}bts", shell=True)
            exit()

    print("\n")

    ratios_dict = {}

    for file in os.listdir(f"{opts.work_dir}bts"):

        if file.endswith(".txt"):

            with open(f"{opts.work_dir}bts/{file}", "r") as handler:

                ratios = [float(line.split("\t")[1]) for line in handler.readlines()]
                mean_ratio_diff = abs(1 - np.nanmean(ratios))
                ratios_dict[file] = mean_ratio_diff

    best_params = min(ratios_dict, key=ratios_dict.get)

    print("Mean ratio difference for every set of parameters:")

    for key in sorted(ratios_dict, key=ratios_dict.get):

        print(f"Cut-off (-o): {float(key.split('_')[3]):.3f}; Persistence length (-l): "
              f"{float(key.split('_')[5].rsplit('.', 1)[0]):.3f}; "
                f"BTS score: {float(ratios_dict[key]):.5f}")

    best_params = min(ratios_dict, key=ratios_dict.get)
    best_params_score = ratios_dict[best_params]
    best_params_list = [best_params]

    for key in sorted(ratios_dict, key=ratios_dict.get):

        if ratios_dict[key] < best_params_score * 1.1 and key != best_params:

            best_params_list.append(key)

    if len(best_params_list) > 1:

        print("You can use any of the below set of parameters to run 'metaloci layout' as they are similarly good.\n")
    
    else:

        print(f"\nBest parameters are:")

    for best in best_params_list:
            
        print(f"Cut-off (-o): {float(best.split('_')[3]):.3f}; Persistence length (-l): "
              f"{float(best.split('_')[5].rsplit('.', 1)[0]):.3f}; "
              f"BTS score: {float(ratios_dict[best]):.5f}")

    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("all done.")
