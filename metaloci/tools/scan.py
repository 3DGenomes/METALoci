# Author: Leo Zuber
# Date: May 2025
"""
Utility to make metaloci models iteratively deleting stretches of bins.
"""
import copy
import glob
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import subprocess as sp
import sys
from argparse import SUPPRESS, HelpFormatter
from datetime import timedelta
from time import time

import cooler
import geopandas as gpd
import h5py
import hicstraw
import imageio.v2 as imageio  # use v2 to avoid deprecation warnings
import matplotlib

matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from metaloci import mlo
from metaloci.graph_layout import kk
from metaloci.misc import misc
from metaloci.plot import plot
from metaloci.spatial_stats import lmi
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.spatial import distance

HELP = "METALoci modeling with iterative bin deletion.\n"

DESCRIPTION = """
Creates several METALoci models by iteratively deleting stetches of bins in the region. If asked to, it generates a 
video with all possible deletions.
"""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step.

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller.
    """

    parser.formatter_class = lambda prog: HelpFormatter(prog, width=120, max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")

    input_arg.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to working directory.",
    )

    input_arg.add_argument(
        "-c",
        "--hic",
        dest="hic_file",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the cool/mcool/hic file.",
    )

    input_arg.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        metavar="INT",
        type=int,
        required=True,
        help="Resolution of the Hi-C files to be used (in bp).",
    )

    input_arg.add_argument(
        "-g",
        "--region",
        dest="regions",
        metavar="PATH",
        type=str,
        help="Region to apply LMI in format chrN:start-end_poi. "
        "'poi' is the point of interest in the region (its bin number)."
    )

    input_arg.add_argument(
        "-s",
        "--signal",
        dest="signals",
        metavar="FILE",
        type=str,
        required=True,
        nargs="*",
        action="extend",
        help="Name of the signal to process."    )


    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.")
    
    optional_arg.add_argument(
        "-n",
        "--num-bins-to-delete",
        dest="num_bins_to_delete",
        metavar="INT",
        type=int,
        default=5,
        help="Number of bins to delete in each iteration. Default: %(default)i",
    )

    optional_arg.add_argument(
        "-p",
        "--signipval",
        dest="signipval",
        metavar="FLOAT",
        type=float,
        default=0.05,
        help="Significance p-value threshold for the LMI signal. Default: %(default)s.",
    )

    optional_arg.add_argument(
        "-m",
        "--mp",
        dest="multiprocess",
        action="store_true",
        help="Flag to set use of multiprocessing.",
    )

    optional_arg.add_argument(
        "-t",
        "--threads",
        dest="threads",
        default=int(mp.cpu_count() - 2),
        metavar="INT",
        type=int,
        action="store",
        help="Number of threads to use in multiprocessing. (default: %(default)s)"
    )

    optional_arg.add_argument(
        "-gif",
        "--gif",
        dest="create_gif",
        default=None,
        metavar="INT",
        type=int,
        action="store",
        help="Flag to create a gif of the plots. "
        "If set, the value is the point of interest to highlight (bin index). "
    )

    optional_arg.add_argument(
        "-fd",
        "--frame-duration",
        dest="frame_duration",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help="Duration of each frame in the gif in seconds. Default: %(default)s.",
    )

    optional_arg.add_argument(
        "-l",
        "--persistence-length",
        dest="persistence_length",
        metavar="FLOAT",
        type=float,
        default=None,
        help="Persistence length to use. If not set, it will be optimised.",
    )

    optional_arg.add_argument(
        "-o",
        "--cutoffs",
        dest="cutoffs",
        metavar="FLOAT",
        type=float,
        nargs="+",
        default=None,
        help="Cutoffs to use for the Kamada-Kawai algorithm. If not set, it will be optimised.",
    )

    optional_arg.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        help="Flag to force overwrite of existing files.",
    )


def scan(row: pd.Series, args: pd.Series, silent):

    INFLUENCE = 1.5 
    BFACT = 2

    region_chrom, _, _, _ = re.split(r":|-|_", row.coords)
    save_path = os.path.join(args.work_dir, 'scan', region_chrom, 'objects', f"{re.sub(':|-', '_', row.coords)}", f"{re.sub(':|-', '_', row.coords)}.mlo")

    pathlib.Path(os.path.join(args.work_dir, 'scan', region_chrom, 'objects', f"{re.sub(':|-', '_', row.coords)}")).mkdir(parents=True, exist_ok=True)

    mlobject = mlo.MetalociObject(
            region=row.coords,
            resolution=args.resolution,
            save_path=save_path,
        )
    
    mlobject.persistence_length = args.persistence_length
    mlobject.kk_cutoff["cutoff_type"] = args.cutoffs["cutoff_type"]
    mlobject.kk_cutoff["values"] = args.cutoffs["values"]
    
    if args.hic_path.endswith(".cool"):

        mlobject.matrix = cooler.Cooler(args.hic_path).matrix(sparse=True).fetch(mlobject.region).toarray()

    if args.hic_path.endswith(".mcool"):

        mlobject.matrix = cooler.Cooler(
            f"{args.hic_path}::/resolutions/{mlobject.resolution}").matrix(sparse=True).fetch(mlobject.region).toarray()

    elif args.hic_path.endswith(".hic"):

        mlobject.matrix = hicstraw.HiCFile(args.hic_path).getMatrixZoomData(
            mlobject.chrom, mlobject.chrom, 'observed', 'VC_SQRT', 'BP', mlobject.resolution).getRecordsAsMatrix(
            mlobject.start, mlobject.end, mlobject.start, mlobject.end)

    mlobject = misc.clean_matrix(mlobject)

    if mlobject.matrix is None:

        return

    if args.persistence_length == "optimise":

        mlobject.persistence_length = kk.estimate_persistence_length(mlobject, args.optimise)

    if mlobject.kk_cutoff["values"] == "optimise":

        mlobject.kk_cutoff["values"] = kk.estimate_cutoff(mlobject, args.optimise)


    print("Computing wild-type...", end="\r")

    cutoffs = mlobject.kk_cutoff["values"]
    mlobject.kk_cutoff["values"] = cutoffs[0]  # Select cut-off for this iteration
    mlobject = kk.get_restraints_matrix(mlobject, False, True)  # Get submatrix of restraints
    mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject.kk_restraints_matrix))
    mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)
    mlobject.kk_coords = list(mlobject.kk_nodes.values())
    mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

    try:

        signal_data = pd.read_csv(
            glob.glob(f"{os.path.join(args.work_dir, 'signal', mlobject.chrom)}/*_signal.tsv")[0], sep="\t", header=0)

    except IndexError:

        exit(f"\n\nNo signal data found for region {mlobject.region}.\n")

    mlobject.signals_dict = lmi.load_region_signals(mlobject, signal_data, args.signals)
    neighbourhood = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT

    if mlobject.lmi_geometry is None: 

        mlobject.lmi_geometry = lmi.construct_voronoi(mlobject, mlobject.kk_distances.diagonal(1).mean() * INFLUENCE) 

    for signal_type in list(mlobject.signals_dict.keys()):

        mlobject.lmi_info[signal_type] = lmi.compute_lmi(mlobject, signal_type, neighbourhood,
                                                            9999, 0.05, True, False)
        
        args.delete_indices = None
        
        plot.create_composite_figure(mlobject, signal_type, del_args = args, silent = True)    
        misc.write_moran_data(mlobject, args, scan = True, silent = True)

    with open(mlobject.save_path, "wb") as hamlo_namendle:
        
        mlobject.save(hamlo_namendle)

    print("Computing wild-type -> done.")
    print("Computing iterative deletions...\n")

    intact_matrix = mlobject.matrix.copy()
    intact_signals = mlobject.signals_dict.copy()
    intact_poi = mlobject.poi
    num_bins_to_delete = args.num_bins_to_delete
    num_bins = intact_matrix.shape[0]

    # Check if the number of bins to delete is larger than the number of bins in the matrix
    if num_bins_to_delete > num_bins:
        
        print(f"\tNumber of bins to delete ({num_bins_to_delete}) is larger than the number of bins in the matrix "
                f" ({num_bins}). Exiting...")
        
        return
    
    del_args = pd.Series({
        "work_dir": args.work_dir,
        "num_bins_to_delete": num_bins_to_delete,
        "intact_matrix": intact_matrix,
        "intact_poi": intact_poi,
        "intact_signals": intact_signals,
        "signipval": args.signipval,
        "create_gif": args.create_gif,
        "force": args.force,
        "wt": False
    })
    
    start_timer = time()

    if args.multiprocess:

        try:

            progress = mp.Manager().dict(value=0, timer=start_timer)

            with mp.Pool(processes=args.threads) as pool:

                pool.starmap(compute_deletion, [(del_args, copy.deepcopy(mlobject), i, True, progress) 
                                                for i in range(num_bins - num_bins_to_delete)]
                                                )

                pool.close()
                pool.join()

        except KeyboardInterrupt:

            pool.terminate()

    else:

        for i in range(num_bins - num_bins_to_delete):

            compute_deletion(del_args, mlobject, i, silent=False)


    if args.create_gif:

        for signal_type in mlobject.signals_dict:
            
            # Generate and mp4 video with all the deletions concatenated.
            base_name = f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{intact_poi}_{args.create_gif}"
            folder = os.path.join(args.work_dir, "scan", mlobject.chrom, "plots", signal_type, base_name)
            results_folder_path = os.path.join(args.work_dir, "scan", mlobject.chrom, "results", signal_type)
            pathlib.Path(results_folder_path).mkdir(parents=True, exist_ok=True)
            output_mp4 = os.path.join(results_folder_path, f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{intact_poi}_{args.create_gif}.mp4")
            fps = 1 / args.frame_duration

            # Load and sort image paths
            png_files = misc.natural_sort([f for f in os.listdir(folder) if f.endswith('.png')])

            if not png_files:

                print(f"No PNG files found in {folder}.")

                continue

            image_paths = [os.path.join(folder, f) for f in png_files]
            images_rgba = [Image.open(p).convert("RGBA") for p in image_paths]

            # Determine uniform frame size (rounded to nearest multiple of 16)
            def pad_to_16(x): return ((x + 15) // 16) * 16
            max_width = pad_to_16(max(img.width for img in images_rgba))
            max_height = pad_to_16(max(img.height for img in images_rgba))
            background_color = (255, 255, 255)

            # Composite each image onto a white background
            frames = []
            for img in images_rgba:

                bg = Image.new("RGBA", (max_width, max_height), background_color + (255,))
                offset = ((max_width - img.width) // 2, (max_height - img.height) // 2)

                bg.paste(img, offset, mask=img)
                frames.append(np.array(bg.convert("RGB")))

            # Save video
            imageio.mimsave(
                output_mp4, frames, fps=fps, codec='libx264',
                quality=10, ffmpeg_log_level="error"
            )

            print(f"MP4 video saved as {output_mp4}")

            # Plot LMI change scan
            moran_data_folder = os.path.join(args.work_dir, "scan", mlobject.chrom, "moran_data", signal_type,
                                    f"{re.sub(':|-', '_', mlobject.region)}_{args.create_gif}")

            plot.get_lmi_change_scan_plot(moran_data_folder, poi=args.create_gif, results_folder = results_folder_path)

def compute_deletion(del_args: pd.Series, mlobject: mlo.MetalociObject, i: int, silent: bool, progress=None):
    """
    Compute the deletion of a specified region in a matrix and update the associated metadata.
    This function removes a stretch of bins from a matrix, generates visualizations and saves the updated object.
    It also handles cases where the computation has already been performed and skips redundant operations.
    Parameters:
        del_args (pd.Series): A pandas Series containing arguments for the deletion process, 
            including the number of bins to delete, intact matrix, intact signals, and other 
            parameters.
        mlobject (mlo.MetalociObject): An instance of the MetalociObject class representing the 
            current state of the matrix and associated metadata.
        i (int): The starting index of the region to delete.
        silent (bool): If True, suppresses output messages.
        progress (dict, optional): A dictionary to track progress, containing keys like 'value' 
            (current progress) and 'timer' (start time).
    Returns:
        None
    Notes:
        - If the file corresponding to the current deletion already exists and the point of 
          interest (POI) matches, the computation is skipped.
        - If the POI differs, only the figure is regenerated.
        - The function updates the matrix, signals, and metadata, and recalculates the KK layout 
          and LMI geometry.
    """

    INFLUENCE = 1.5
    BFACT = 2
    del_args.i = i
    mlobject = copy.deepcopy(mlobject)
    len_matrix = len(mlobject.matrix)
    gif_poi = del_args.create_gif
    save_path_i = mlobject.save_path.replace('.mlo', f'_{i}.mlo')
    signipval = del_args.signipval
    to_do = True

    # If file exists, check if the poi set in the arg is the same that has already been computed. 
    # If it is, skip the computation.
    # if it is not, redo only the figure with the new poi.
    # if the file does not exist, compute the whole thing.
    if os.path.isfile(save_path_i) and not del_args.force:

        # Load existing object
        with open(save_path_i, "rb") as f:

            state = pickle.load(f)
            mlobject = mlo.reconstruct(state)

        delete_indices = list(range(i, i + del_args.num_bins_to_delete))
        
        # Adjust gif_poi if needed
        if min(delete_indices) < gif_poi:
            gif_poi -= del_args.num_bins_to_delete

        # Determine which point of interest to compare
        compare = gif_poi if mlobject.poi is None else mlobject.poi

        if compare == gif_poi: # check if the previously computed poi is the same as the one in the args

            if not silent:
                print(f"\tFile \"{save_path_i}\" already exists. Skipping to next region...")

            if progress is not None:
                progress['value'] += 1
                time_spent = time() - progress['timer']
                total = len_matrix - del_args.num_bins_to_delete
                time_remaining = int(time_spent / progress['value'] * (total - progress['value']))

                # Clear previous line and print progress
                terminal_width = int(sp.Popen(['tput', 'cols'], stdout=sp.PIPE).communicate()[0].strip())
                print(f"\033[A{'  ' * terminal_width}\033[A")
                print(f"\t[{progress['value']}/{total}] | Time spent: {timedelta(seconds=round(time_spent))} | "
                      f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')
                
            return
        
        else:
        
            to_do = False
    
    # Set what chunk of bins to remove according to the i in the outside loop
    num_bins = del_args.intact_matrix.shape[0]
    delete_indices = list(range(i, i + del_args.num_bins_to_delete))
    keep_indices = [j for j in range(num_bins) if j not in delete_indices]

    del_args.delete_indices = delete_indices
    
    if to_do:

        mlobject.save_path = mlobject.save_path.replace('.mlo', f'_{i}.mlo')
        mlobject.kk_distances = None

        # Remove a stretch of bins from the matrix
        mlobject.matrix = del_args.intact_matrix[np.ix_(keep_indices, keep_indices)]

        # Properly set the poi, because after the deletion bins are shifted
        if gif_poi in delete_indices:

            mlobject.poi = None

        elif min(delete_indices) < gif_poi:

            mlobject.poi = gif_poi - del_args.num_bins_to_delete

        elif min(delete_indices) > gif_poi:

            mlobject.poi = gif_poi

        # Remove those same bins from the signal data, which is a numpy array
        for signal_type in del_args.intact_signals.keys():

            mlobject.signals_dict[signal_type] = del_args.intact_signals[signal_type][keep_indices]

        # Calculate KK layout
        mlobject = kk.get_restraints_matrix(mlobject, False, silent)  # Get submatrix of restraints
        mlobject.kk_graph = nx.from_scipy_sparse_array(csr_matrix(mlobject.kk_restraints_matrix))
        mlobject.kk_nodes = nx.kamada_kawai_layout(mlobject.kk_graph)
        mlobject.kk_coords = list(mlobject.kk_nodes.values())
        mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")
        neighbourhood = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT
        mlobject.lmi_geometry = lmi.construct_voronoi(mlobject, mlobject.kk_distances.diagonal(1).mean() * INFLUENCE, 
                                                      del_args = del_args)

    for signal_type in mlobject.signals_dict.keys():

        if to_do:

            mlobject.lmi_info[signal_type] = lmi.compute_lmi(mlobject, signal_type, neighbourhood,
                                                            9999, 0.05, silent, False, del_args = del_args)

        plot.create_composite_figure(mlobject, signal_type, del_args = del_args, signipval = signipval,  silent = silent)
        misc.write_moran_data(mlobject, del_args, scan = True, silent = silent)
        
        with open(mlobject.save_path, "wb") as hamlo_namendle:

            mlobject.save(hamlo_namendle)

        if progress is not None:

            progress['value'] += 1
            time_spent = time() - progress['timer']
            time_remaining = int(time_spent / progress['value'] * (len(keep_indices) - progress['value']))

            print(f"\033[A{'  '*int(sp.Popen(['tput','cols'], stdout=sp.PIPE).communicate()[0].strip())}\033[A")
            print(f"\t[{progress['value']}/{len(keep_indices)}] | Time spent: {timedelta(seconds=round(time_spent))} | "
                f"ETR: {timedelta(seconds=round(time_remaining))}", end='\r')

        

def run(opts: list):

    if not opts.work_dir.endswith("/"):

        opts.work_dir += "/"

    cutoffs = {}
    cutoffs["cutoff_type"] = "percentage"

    if opts.cutoffs is None:

        cutoffs["values"] = "optimise"

    else:

        cutoffs["values"] = opts.cutoffs

    if opts.persistence_length is None:

        opts.persistence_length = "optimise"

    else:

        persistence_length = opts.persistence_length

    df_regions = pd.DataFrame({"coords": [opts.regions], "symbol": ["symbol"], "id": ["id"]})
    signals = [opts.signals[0]]

    if opts.hic_file.endswith(".cool"):

        available_resolutions = cooler.Cooler(opts.hic_file).binsize

        if opts.resolution != available_resolutions:

            sys.exit("The given resolution is not the same as the provided cooler file. Exiting...")

    elif opts.hic_file.endswith(".mcool"):

        available_resolutions = [int(x) for x in list(h5py.File(opts.hic_file)["resolutions"].keys())]

        if opts.resolution not in available_resolutions:

            print(
                f"The given resolution is not in the provided mcooler file.\nThe available resolutions are: "
                f"{', '.join(misc.natural_sort([str(x) for x in available_resolutions]))}"
            )
            sys.exit("Exiting...")

    elif opts.hic_file.endswith(".hic"):

        available_resolutions = hicstraw.HiCFile(opts.hic_file).getResolutions()

        if opts.resolution not in available_resolutions:

            print(
                f"The given resolution is not in the provided mcooler file.\nThe available resolutions are: "
                f"{', '.join(misc.natural_sort([str(x) for x in available_resolutions]))}"
            )
            sys.exit("Exiting...")

    else:

        print("HiC file format not supported. Supported formats are: cool, mcool, hic.")
        sys.exit("Exiting...")


    parsed_args = pd.Series({"work_dir": opts.work_dir,
                             "hic_path": opts.hic_file,
                             "resolution": opts.resolution,
                             "cutoffs": cutoffs,
                             "persistence_length": persistence_length,
                             "optimise": False, # perhaps not needed
                             "signals": signals,
                             "num_bins_to_delete": opts.num_bins_to_delete,
                             "multiprocess": opts.multiprocess,
                             "threads": opts.threads,
                             "create_gif": opts.create_gif,
                             "frame_duration": opts.frame_duration,
                             "signipval": opts.signipval,
                             "wt": True,
                             "force": opts.force,
    })

    start_timer = time()

    for counter, row in df_regions.iterrows():

        scan(row, parsed_args, silent=False)

    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    misc.create_version_log("layout", opts.work_dir)
    print("\nAll done.")
