"""
This script generates METALoci plots.
"""
import os
import sys
import pathlib
import pickle
import re
from argparse import HelpFormatter, SUPPRESS, RawTextHelpFormatter

from datetime import timedelta
from time import time

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from metaloci.plot import plot

HELP = "Plots METALoci output."

DESCRIPTION = """Outputs the different plots to show METALoci.
It creates the following plots:\n
\tHiC matrix
\tSignal plot
\tKamada-Kawai layout
\tLocal Moran's I scatterplot
\tGaudí plot for signal
\tGaudí plot for LMI quadrant\n
and a composite image with all the above."""


def populate_args(parser):
    """
    Function to give the main METALoci script the arguments needed to run the layout step

    Parameters
    ----------
    parser : ArgumentParser
        ArgumentParser to populate the arguments through the normal METALoci caller
    """

    parser.formatter_class = lambda prog: HelpFormatter(prog, width=120, max_help_position=60)

    input_arg = parser.add_argument_group(title="Input arguments")

    input_arg.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        required=True,
        metavar="PATH",
        type=str,
        help="Path to working directory."
    )

    input_arg.add_argument(
        "-s",
        "--types",
        dest="signals",
        metavar="STR",
        type=str,
        nargs="*",
        action="extend",
        help="Space-separated list of signals to plot or path to the file with the list of signals to plot, "
        "one per line."
    )

    input_arg.add_argument(
        "-g",
        "--region",
        dest="regions",
        metavar="PATH",
        type=str,
        required=True,
        help="Region to apply LMI in format chrN:start-end_midpoint or file with the regions of interest. \
        If a file is provided, it must contain as a header 'coords', 'symbol' and 'id', and one region per line, \
        tab separated."
    )

    optional_arg = parser.add_argument_group(title="Optional arguments")

    optional_arg.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.")

    optional_arg.add_argument(
        "-e",
        "--preserve",
        dest="rm_types",
        action="store_false",
        default=True,
        help="Preserve temporal .png image files that are used for making the composed figure (default: %(default)s)."
    )

    optional_arg.add_argument(
        "-m",
        "--metalocis",
        dest="metalocis",
        action="store_true",
        help="Flag to select highlightning of the signal plots. If True, only the neighbouring bins from the point of \
        interest will be highlighted (independently of the quadrant and significance of those bins, but only if the \
        point of interest is significant). If False, all significant regions that correspond to the quadrant selected \
        with -q will be highlighted (default: False)."
    )

    optional_arg.add_argument(
        "-q",
        "--quarts",
        dest="quart",
        default=[1, 3],
        metavar="INT",
        nargs="*",
        help="Space-separated list with the LMI quadrants to highlight (default: %(default)s). \
        1: High-high (signal in bin is high, signal on neighbours is high). \
        2: High-Low (signal in bin is high, signal on neighbours is low). \
        3: Low-Low (signal in bin is low, signal on neighbours is low). \
        4: Low-High (signal in bin is low, signal on neighbours is high).",
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

    optional_arg.add_argument(
        "-k",
        "--mark_regions",
        dest="mark_regions",
        metavar="PATH",
        type=str,
        help="Path to a file to mark certain regions on the gaudí plots. The file must have the following columns "
        "(tab-separated): region_metaloci chr start end label. The label will be used to mark the region on the plot."
    )

    optional_arg.add_argument(
        "-u",
        "--debug",
        dest="debug",
        action="store_true",
        help=SUPPRESS)


def run(opts: list):
    """
    Function to run this section of METALoci with the needed arguments

    Parameters
    ----------
    opts : list
        List of arguments
    """

    work_dir = opts.work_dir
    regions = opts.regions
    signals = opts.signals
    metaloci_only = opts.metalocis
    quadrants = opts.quart
    signipval = opts.signipval
    rmtypes = opts.rm_types
    mark_regions = opts.mark_regions
    debug = opts.debug
    quadrants = [int(x) for x in quadrants]

    if not work_dir.endswith("/"):

        work_dir += "/"

    INFLUENCE = 1.5
    BFACT = 2

    # Functions assume this schema of colors if the user says nothing.
    # colors = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}
    if os.path.isfile(regions):

        df_regions = pd.read_table(regions)

    else:

        df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

    # Parse list of signals to plot. If it is a file, strip it, if there are
    if os.path.isfile(signals[0]) and os.access(signals[0], os.R_OK):

        with open(signals[0], "r", encoding="utf-8") as handler:

            signals = [line.strip() for line in handler]

    if mark_regions is not None:

        regions2mark = pd.read_table(mark_regions, names=["region_metaloci", "chr", "start", "end", "mark"], sep="\t")

    else:
        
        regions2mark = None

    plot_opt = {"bbox_inches": "tight", "dpi": 300, "transparent": True}

    if debug:

        print(f"work_dir ->\n\t{work_dir}")
        print(f"regions ->\n\t{regions}")
        print(f"signals ->\n\t{signals}")
        print(f"metaloci_only ->\n\t{metaloci_only}")
        print(f"quadrants ->\n\t{quadrants}")
        print(f"signipval ->\n\t{signipval}")
        print(f"rmtypes ->\n\t{rmtypes}")
        print(f"influence ->\n\t{INFLUENCE}")
        print(f"bfact ->\n\t{BFACT}")

        sys.exit()

    start_timer = time()

    for i, region_row in df_regions.iterrows():

        print(f"\n------> Working on region {region_row.coords} [{i + 1}/{len(df_regions)}]")

        save_path = f"{work_dir}{region_row.coords.split(':', 1)[0]}/{re.sub(':|-', '_', region_row.coords)}.mlo"

        try:

            with open(save_path, "rb") as mlobject_handler:

                mlobject = pickle.load(mlobject_handler)
                mlobject.save_path = save_path

        except FileNotFoundError:

            print("\n\t.mlo file not found for this region.\n\tSkipping to the next one.")

            continue

        if mlobject.lmi_info is None:

            print("\n\tLMI not calculated for this region.\n\tSkipping to the next one...")

            continue

        neighbourhood = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT

        for signal in signals:

            if len(signal) == 0:

                continue

            print(f"\n\tPlotting signal: {signal}")

            plot_filename = os.path.join(work_dir, mlobject.chrom, "plots",
                                         signal, f"{mlobject.start}_{mlobject.end}_{mlobject.poi}")
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

            print("\t\tHi-C plot", end="\r")

            hic_plt = plot.get_hic_plot(mlobject)

            hic_plt.savefig(f"{plot_filename}_hic.pdf", **plot_opt)
            hic_plt.savefig(f"{plot_filename}_hic.png", **plot_opt)
            plt.close()

            print("\t\tHi-C plot -> done.")
            print("\t\tKamada-Kawai plot", end="\r")

            mlg_poi = merged_lmi_geometry.loc[merged_lmi_geometry["bin_index"] == mlobject.poi].squeeze()

            if (metaloci_only and mlg_poi.LMI_pvalue <= signipval and mlg_poi.moran_quadrant in quadrants):

                kk_plt = plot.get_kk_plot(mlobject, neighbourhood=neighbourhood)

            else:

                kk_plt = plot.get_kk_plot(mlobject)

            kk_plt.savefig(f"{plot_filename}_kk.pdf", **plot_opt)
            kk_plt.savefig(f"{plot_filename}_kk.png", **plot_opt)
            plt.close()
            print("\t\tKamada-Kawai plot -> done.")

            print("\t\tGaudi Signal plot", end="\r")
            gs_plt = plot.get_gaudi_signal_plot(mlobject, merged_lmi_geometry, regions2mark=regions2mark)
            gs_plt.savefig(f"{plot_filename}_gsp.pdf", **plot_opt)
            gs_plt.savefig(f"{plot_filename}_gsp.png", **plot_opt)
            plt.close()
            print("\t\tGaudi Signal plot -> done.")

            print("\t\tGaudi Type plot", end="\r")
            gt_plt = plot.get_gaudi_type_plot(mlobject, merged_lmi_geometry, signipval, regions2mark=regions2mark)
            gt_plt.savefig(f"{plot_filename}_gtp.pdf", **plot_opt)
            gt_plt.savefig(f"{plot_filename}_gtp.png", **plot_opt)
            plt.close()
            print("\t\tGaudi Type plot -> done.")
            print("\t\tSignal plot", end="\r")

            sig_plt, ax = plot.signal_plot(mlobject, merged_lmi_geometry, neighbourhood,
                                           quadrants, signipval, metaloci_only)

            sig_plt.savefig(f"{plot_filename}_signal.pdf", **plot_opt)
            sig_plt.savefig(f"{plot_filename}_signal.png", **plot_opt)
            plt.close()
            print("\t\tSignal plot -> done.")
            print("\t\tLMI Scatter plot", end="\r")

            lmi_plt, r_value, p_value = plot.get_lmi_scatterplot(mlobject, merged_lmi_geometry,
                                                                 neighbourhood, signipval)

            if lmi_plt is not None:

                lmi_plt.savefig(f"{plot_filename}_lmi.pdf", **plot_opt)
                lmi_plt.savefig(f"{plot_filename}_lmi.png", **plot_opt)
                plt.close()
                print("\t\tLMI Scatter plot -> done.")

            else:

                if signal == signals[-1]:

                    print("\t\tSkipping to next region...")
                    continue

                print("\t\tSkipping to next signal...")

            print(f"\t\tFinal composite figure for region '{region_row.coords}' and signal '{signal}'", end="\r")

            img1 = Image.open(f"{plot_filename}_lmi.png")
            img2 = Image.open(f"{plot_filename}_gsp.png")
            img3 = Image.open(f"{plot_filename}_gtp.png")
            maxx = int((img1.size[1] * 0.4 + img2.size[1] * 0.25 + img3.size[1] * 0.25) * 1.3)
            yticks_signal = [f"{round(i, 3):.2f}" for i in ax.get_yticks()[1:-1]]
            signal_left = {3: 39, 4: 32, 5: 21, 6: 10, 7: -1, 8: -11}
            max_chr_yax = max(len(str(i)) for i in yticks_signal)

            if float(min(yticks_signal)) < 0:

                negative_axis_correction = 5

            else:

                negative_axis_correction = 0

            if max_chr_yax not in list(signal_left.keys()):

                signal_left[max_chr_yax] = -21

            composite_image = Image.new(mode="RGBA", size=(maxx, 1550))
            # HiC image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_hic.png", 0.5, 100, 50)
            # Signal image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_signal.png", 0.4,
                                                   signal_left[max_chr_yax] + negative_axis_correction, 640)
            # KK image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_kk.png", 0.3, 1300, 50)
            # LMI scatter image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_lmi.png", 0.4, 75, 900)
            # Gaudi signal image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_gsp.png", 0.25, 900, 900)
            # Gaudi signal image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_gtp.png", 0.25, 1600, 900)

            composite_image.save(f"{plot_filename}.png")
            plt.figure(figsize=(15, 15))
            plt.imshow(composite_image)
            plt.axis("off")
            plt.savefig(f"{plot_filename}.pdf", **plot_opt)
            plt.close()
            print(f"\t\tFinal composite figure for region '{region_row.coords}' and signal '{signal}' -> done.")

            # bed = plot.get_bed(mlobject, merged_lmi_geometry, neighbourhood, BFACT, quadrants, signipval, plotit=True)

            # if bed is not None and len(bed) > 0:

            #     metaloci_bed_path = os.path.join(work_dir, mlobject.chrom, "metalocis_log", signal)

            #     bed_file_name = os.path.join(
            #         metaloci_bed_path,
            #         f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_{signal}_\
            #         q-{'_'.join([str(q) for q in quadrants])}_metalocis.bed")

            #     pathlib.Path(metaloci_bed_path).mkdir(parents=True, exist_ok=True)
            #     bed.to_csv(bed_file_name, sep="\t", index=False)

            #     print(f"\t\tBed file with metalocis location saved to: {bed_file_name}")

            for signal_key, df in mlobject.lmi_info.items():

                if signal_key != signal:

                    continue

                sq = [0] * 4
                q = [0] * 4

                for i, row in df.iterrows():

                    q[row.moran_quadrant - 1] += 1

                    if row.LMI_pvalue <= signipval:

                        sq[row.moran_quadrant - 1] += 1

                q_string = "\t".join([f"{sq[i]}\t{q[i]}" for i in range(4)])

                with open(f"{work_dir}moran_info.txt", "a+", encoding="utf-8") as handler:

                    log = f"{region_row.coords}\t{region_row.name}\t{region_row.id}\t{signal_key}\t{r_value}\
                        \t{p_value}\t{q_string}\n"

                    handler.seek(0)

                    if os.stat(f"{work_dir}moran_info.txt").st_size == 0:

                        handler.write("region\tsymbol\tgene_id\tsignal\tr_value\tp_value\tsq1\tq1\tsq2\tq2\tsq3\tq3\
                                    \tsq4\tq4\n")

                    if not any(log in line for line in handler):

                        handler.write(log)

            if rmtypes:

                os.remove(f"{plot_filename}_hic.png")
                os.remove(f"{plot_filename}_signal.png")
                os.remove(f"{plot_filename}_kk.png")
                os.remove(f"{plot_filename}_lmi.png")
                os.remove(f"{plot_filename}_gsp.png")
                os.remove(f"{plot_filename}_gtp.png")

    print(f"\nInformation saved to: '{os.path.join(work_dir, 'moran_info.txt')}'")
    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}.")
    print("\nall done.")
