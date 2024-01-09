"""
This script generates METALoci plots.
"""
from collections import defaultdict

import os
import pathlib
import pickle
import re
from argparse import HelpFormatter
from datetime import timedelta
from time import time

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from metaloci.plot import plot

DESCRIPTION = "Outputs the different plots to show METALoci."
DESCRIPTION += " It creates the following plots:\n"
DESCRIPTION += "HiC matrix"
DESCRIPTION += ", Signal plot"
DESCRIPTION += ", Kamada-Kawai layout"
DESCRIPTION += ", Local Moran's I scatterplot"
DESCRIPTION += ", Gaudí plot for signal"
DESCRIPTION += ", Gaudí plot for LMI quadrant"
DESCRIPTION += " and a composite image with all the above."


def populate_args(parser):

    parser.formatter_class=lambda prog: HelpFormatter(prog, width=120,
                                                      max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")
    optional_arg = parser.add_argument_group(title="Optional arguments")

    input_arg.add_argument(
        "-w", "--work-dir", dest="work_dir", required=True, metavar="PATH", type=str, help="Path to working directory."
    )

    input_arg.add_argument(
        "-s",
        "--types",
        dest="signals",
        metavar="STR",
        type=str,
        nargs="*",
        action="extend",
        help="Space-separated list of signals to plot or path to the file with the list of signals to plot, one per line.",
    )

    input_arg.add_argument(
        "-g",
        "--region",
        dest="regions",
        metavar="PATH",
        type=str,
        required=True,
        help="Region to apply LMI in format chrN:start-end_midpoint or file with the regions of interest. If a file is provided, "
        "it must contain as a header 'coords', 'symbol' and 'id', and one region per line, tab separated.",
    
    )

    optional_arg.add_argument(
        "-e",
        "--delete",
        dest="rm_types",
        required=False,
        metavar="STR",
        type=str,
        nargs="*",
        default=["png"],
        help="Delete temporal image files, determined by extension " "(default: %(default)s)",
    )

    optional_arg.add_argument(
        "-m",
        "--metalocis",
        dest="metalocis",
        action="store_true",
        help="Flag to select highlightning of the signal plots. If True, only the neighbouring bins from the point of interest will be "
        "highlighted (independently of the quadrant and significance of those bins, but only if the point of interest is significant). "
        "If False, all significant regions that correspond to the quadrant selected with -q will be highlighted (default: False).",
    )

    optional_arg.add_argument(
        "-a",
        "--aggregated",
        dest="agg",
        required=False,
        action="store_true",
        help="Use the file with aggregated signals (*_LMI_byType.pkl)",
    )

    optional_arg.add_argument(
        "-q",
        "--quarts",
        dest="quart",
        default=[1, 3],
        metavar="INT",
        nargs="*",
        help="Space-separated list with the LMI quadrants to highlight "
        "(default: %(default)s) "
        "1: High-high (signal in bin is high, signal on neighbours is "
        "high) "
        "2: Low-High (signal in bin is low, signal on neighbours is high) "
        "3: Low-Low (signal in bin is low, signal on neighbours is low) "
        "4: High-Low (signal in bin is high, signal on neighbours is low).",
    )

    optional_arg.add_argument(
        "-v",
        "--pval",
        dest="signipval",
        default=0.05,
        metavar="FLOAT",
        type=float,
        help="P-value significance threshold (default: %(default)s).",
    )

    optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")


def run(opts):
        
    work_dir = opts.work_dir
    regions = opts.regions
    signals = opts.signals
    metaloci_only = opts.metalocis
    quadrants = opts.quart
    signipval = opts.signipval
    rmtypes = opts.rm_types
    agg = opts.agg

    quadrants = [int(x) for x in quadrants]

    if not work_dir.endswith("/"):

        work_dir += "/"

    INFLUENCE = 1.5
    BFACT = 2

    colors = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}

    start_timer = time()

    if os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    # Parse list of signals to plot. If it is a file, strip it, if there are 
    if os.path.isfile(signals[0]) and os.access(signals[0], os.R_OK):

        with open(signals[0], "r", encoding="utf-8") as handler:

            signals = [line.strip() for line in handler]

    plot_opt = {"bbox_inches": "tight", "dpi": 300, "transparent": True}

    for i, region_iter in df_regions.iterrows():

        region = region_iter.coords

        print(f"\n------> Working on region {region} [{i + 1}/{len(df_regions)}]")

        try:

            with open(
                f"{work_dir}{region.split(':', 1)[0]}/{re.sub(':|-', '_', region)}.mlo",
                "rb",
            ) as mlobject_handler:

                mlobject = pickle.load(mlobject_handler)
                mlobject.save_path = f"{work_dir}{region.split(':', 1)[0]}/{re.sub(':|-', '_', region)}.mlo"

        except FileNotFoundError:

            print("\n\t.mlo file not found for this region. \n\tSkipping to the next one.")

            continue

        if mlobject.lmi_info == None:

            print("\n\tLMI not calculated for this region. \n\tSkipping to the next one...")
            continue        

        buffer = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE

        for signal in signals:

            if len(signal) == 0:

                continue

            print(f"\n\tPlotting signal: {signal}")

            plot_filename = os.path.join(work_dir, mlobject.chrom, "plots", signal, f"{mlobject.start}_{mlobject.end}_{mlobject.poi}")
            pathlib.Path(plot_filename).mkdir(parents=True, exist_ok=True)

            plot_filename = os.path.join(
                plot_filename,
                f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_{mlobject.resolution}_{signal}",
            )

            merged_lmi_geometry = pd.merge(
                mlobject.lmi_info[signal],
                mlobject.lmi_geometry,
                on=["bin_index", "moran_index"],
                how="inner",
            )

            merged_lmi_geometry = gpd.GeoDataFrame(merged_lmi_geometry, geometry=merged_lmi_geometry.geometry)

            # TODO The creation of the merged dataframe should be a function in misc. Put there the coditions of aggregation.

            print("\t\tHi-C plot", end="\r")
            hic_plt = plot.get_hic_plot(mlobject)
            hic_plt.savefig(f"{plot_filename}_hic.pdf", **plot_opt)
            hic_plt.savefig(f"{plot_filename}_hic.png", **plot_opt)
            plt.close()
            print("\t\tHi-C plot -> done.")

            print("\t\tKamada-Kawai plot", end="\r")
            
            """If metaloci_bed is True and the LMI p-value of the mlobject.poi is significant and the quadrant is in quadrants,"""
            if metaloci_only and merged_lmi_geometry.loc[merged_lmi_geometry["bin_index"] == mlobject.poi, "LMI_pvalue"].values[0] <= signipval and merged_lmi_geometry.loc[merged_lmi_geometry["bin_index"] == mlobject.poi, "moran_quadrant"].values[0] in quadrants:

                kk_plt = plot.get_kk_plot(mlobject, neighbourhood=buffer * BFACT)

            else:

                kk_plt = plot.get_kk_plot(mlobject)

            kk_plt.savefig(f"{plot_filename}_kk.pdf", **plot_opt)
            kk_plt.savefig(f"{plot_filename}_kk.png", **plot_opt)
            plt.close()
            print("\t\tKamada-Kawai plot -> done.")  #

            print("\t\tGaudi Signal plot", end="\r")
            gs_plt = plot.get_gaudi_signal_plot(mlobject, merged_lmi_geometry)
            gs_plt.savefig(f"{plot_filename}_gsp.pdf", **plot_opt)
            gs_plt.savefig(f"{plot_filename}_gsp.png", **plot_opt)
            plt.close()
            print("\t\tGaudi Signal plot -> done.")

            print("\t\tGaudi Type plot", end="\r")
            gt_plt = plot.get_gaudi_type_plot(mlobject, merged_lmi_geometry, signipval, colors)
            gt_plt.savefig(f"{plot_filename}_gtp.pdf", **plot_opt)
            gt_plt.savefig(f"{plot_filename}_gtp.png", **plot_opt)
            plt.close()
            print("\t\tGaudi Type plot -> done.")

            print("\t\tSignal plot", end="\r")
            sig_plt, ax = plot.signal_plot(mlobject, merged_lmi_geometry, INFLUENCE, BFACT, quadrants, signipval, metaloci_only)
            sig_plt.savefig(f"{plot_filename}_signal.pdf", **plot_opt)
            sig_plt.savefig(f"{plot_filename}_signal.png", **plot_opt)
            plt.close()
            print("\t\tSignal plot -> done.")
            
            print("\t\tLMI Scatter plot", end="\r")
            lmi_plt, r_value, p_value = plot.get_lmi_scatterplot(
                mlobject, merged_lmi_geometry, buffer * BFACT, signipval, colors
            )

            if lmi_plt is not None:

                lmi_plt.savefig(f"{plot_filename}_lmi.pdf", **plot_opt)
                lmi_plt.savefig(f"{plot_filename}_lmi.png", **plot_opt)
                plt.close()
                print("\t\tLMI Scatter plot -> done.")

            else: 

                if signal == signals[-1]:

                    print("\t\tSkipping to next region...")
                    continue
                
                else:

                    print("\t\tSkipping to next signal...")
                    continue

            print(f"\t\tFinal composite figure for region '{region}' and signal '{signal}'", end="\r")
            
            img1 = Image.open(f"{plot_filename}_lmi.png")
            img2 = Image.open(f"{plot_filename}_gsp.png")
            img3 = Image.open(f"{plot_filename}_gtp.png")

            maxx = int((img1.size[1] * 0.4 + img2.size[1] * 0.25 + img3.size[1] * 0.25) * 1.3)
            yticks_signal = [f"{round(i, 3):.2f}" for i in ax.get_yticks()[1:-1]]                
            signal_left = {3 : 39, 4 : 32, 5 : 21, 6 : 10, 7: -1, 8: -11}
            max_chr_yax = max(len(str(i)) for i in yticks_signal)
            
            if float(min(yticks_signal)) < 0:
                
                negative_axis_correction = 5
            
            else:
                
                negative_axis_correction = 0
            
            if max_chr_yax not in signal_left.keys():

                signal_left[max_chr_yax] = -21

            composite_image = Image.new(mode="RGBA", size=(maxx, 1550))
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_hic.png", 0.5, 100, 50)  # HiC image            
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_signal.png", 0.4, signal_left[max_chr_yax] + negative_axis_correction, 640)  # Signal image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_kk.png", 0.3, 1300, 50)  # KK image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_lmi.png", 0.4, 75, 900)  # LMI scatter image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_gsp.png", 0.25, 900, 900)  # Gaudi signal image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_gtp.png", 0.25, 1600, 900)  # Gaudi signal image

            composite_image.save(f"{plot_filename}.png")

            plt.figure(figsize=(15, 15))
            plt.imshow(composite_image)
            plt.axis("off")
            plt.savefig(f"{plot_filename}.pdf", **plot_opt)
            plt.close()
            print(f"\t\tFinal composite figure for region '{region}' and signal '{signal}' -> done.")
            
            bed = plot.get_bed(mlobject, merged_lmi_geometry, INFLUENCE, BFACT, signipval, quadrants)

            if bed is not None and len(bed) > 0:

                metaloci_bed_path = f"{work_dir}{mlobject.chrom}/metalocis_log/{signal}"
                
                pathlib.Path(metaloci_bed_path).mkdir(parents=True, exist_ok=True) 
                bed.to_csv(f"{metaloci_bed_path}/{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_{signal}_q-{'_'.join([str(q) for q in quadrants])}_metalocis.bed", sep="\t", index=False)
                
                print(f"\t\tBed file with metalocis location saved to: "
                      f"{metaloci_bed_path}/{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_{signal}_q-{'_'.join([str(q) for q in quadrants])}_metalocis.bed")
                
            for signal_key, df in mlobject.lmi_info.items():
                                
                if signal_key != signal:
                    
                    continue

                sq = [0] * 4
                q = [0] * 4

                for i, row in df.iterrows():

                    quadrant = row["moran_quadrant"]
                    lmi_pvalue = row["LMI_pvalue"]

                    q[quadrant - 1] += 1

                    if lmi_pvalue <= signipval:

                        sq[quadrant - 1] += 1

                q_string = "\t".join([f"{sq[i]}\t{q[i]}" for i in range(4)])

                with open(f"{work_dir}moran_info.txt", "a+") as handler:

                    log = f"{region}\t{region_iter.name}\t{region_iter.id}\t{signal_key}\t{r_value}\t{p_value}\t{q_string}\n"

                    handler.seek(0)

                    if os.stat(f"{work_dir}moran_info.txt").st_size == 0:

                        handler.write("region\tsymbol\tgene_id\tsignal\tr_value\tp_value\tsq1\tq1\tsq2\tq2\tsq3\tq3\tsq4\tq4\n")

                    if not any(log in line for line in handler):

                        handler.write(log)

            # Remove image used for composite figure.
            if rmtypes:

                for ext in rmtypes:

                    os.remove(f"{plot_filename}_hic.{ext}")
                    os.remove(f"{plot_filename}_signal.{ext}")
                    os.remove(f"{plot_filename}_kk.{ext}")
                    os.remove(f"{plot_filename}_lmi.{ext}")
                    os.remove(f"{plot_filename}_gsp.{ext}")
                    os.remove(f"{plot_filename}_gtp.{ext}")

    print(f"\nInformation saved to: '{os.path.join(work_dir, 'moran_info.txt')}'")
    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}")
    print("\nall done.")
