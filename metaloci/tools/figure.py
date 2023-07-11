"""
This script generates METALoci plots.
"""
import os
import pathlib
import pickle
import re
from argparse import (ArgumentParser, HelpFormatter, RawTextHelpFormatter)
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

def run(opts):
        
    work_dir = opts.work_dir
    regions = opts.regions
    signals = opts.signals
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
    data_moran = {"Coords": [], "Symbol": [], "Gene_id": [], "Signal": [], "R_value": [], "P_value": []}

    for i, region_iter in df_regions.iterrows():

        region = region_iter.coords

        print(f"\n------> Working on region {region} [{i + 1}/{len(df_regions)}]")

        try:

            with open(
                f"{work_dir}{region.split(':', 1)[0]}/{re.sub(':|-', '_', region)}.mlo",
                "rb",
            ) as mlobject_handler:

                mlobject = pickle.load(mlobject_handler)

        except FileNotFoundError:

            print("\n\t.mlo file not found for this region. \n\tSkipping to next region.")

            continue

        buffer = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE

        bins = []
        coords_b = []

        for i in range(1, 4):

            if i == 1:

                bins.append(int(0))

            elif i == 4:

                bins.append(len(mlobject.lmi_geometry) - 1)

            else:

                bins.append(int(((i - 1) / 2) * len(mlobject.lmi_geometry)) - 1)

            coords_b.append(f"{mlobject.start + bins[i - 1] * mlobject.resolution:,}")

        for signal in signals:

            if len(signal) == 0:

                continue

            print(f"\n\tPlotting signal: {signal}")

            plot_filename = os.path.join(work_dir, mlobject.chrom, "plots", signal, mlobject.region)
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

            print("\t\tHiC plot", end="\r")
            hic_plt = plot.get_hic_plot(mlobject)
            hic_plt.savefig(f"{plot_filename}_hic.pdf", **plot_opt)
            hic_plt.savefig(f"{plot_filename}_hic.png", **plot_opt)
            plt.close()
            print("\t\tHiC plot -> done.")

            print("\t\tKamada-Kawai plot", end="\r")
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

            print("\t\tLMI Scatter plot", end="\r")
            lmi_plt, r_value, p_value = plot.get_lmi_scatterplot(
                mlobject, merged_lmi_geometry, buffer * BFACT, signipval, colors
            )
            lmi_plt.savefig(f"{plot_filename}_lmi.pdf", **plot_opt)
            lmi_plt.savefig(f"{plot_filename}_lmi.png", **plot_opt)
            plt.close()
            print("\t\tLMI Scatter plot -> done.")

            print("\t\tSignal plot", end="\r")
            bed_data, selmetaloci = plot.signal_bed(
                mlobject,
                merged_lmi_geometry,
                buffer * BFACT,
                quadrants,
                signipval,
            )

            selmetaloci = []

            sig_plt = plot.signal_plot(mlobject, merged_lmi_geometry, selmetaloci, bins, coords_b)
            sig_plt.savefig(f"{plot_filename}_signal.pdf", **plot_opt)
            sig_plt.savefig(f"{plot_filename}_signal.png", **plot_opt)
            plt.close()
            print("\t\tSignal plot -> done.")

            print(f"\t\tFinal composite figure for region '{region}' and signal '{signal}'", end="\r")
            img1 = Image.open(f"{plot_filename}_lmi.png")
            img2 = Image.open(f"{plot_filename}_gsp.png")
            img3 = Image.open(f"{plot_filename}_gtp.png")

            maxx = int((img1.size[1] * 0.4 + img2.size[1] * 0.25 + img3.size[1] * 0.25) * 1.3)

            composite_image = Image.new(mode="RGBA", size=(maxx, 1550))

            composite_image = plot.place_composite(composite_image, f"{plot_filename}_hic.png", 0.5, 100, 50)  # HiC image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_signal.png", 0.4, 42, 660)  # Signal image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_kk.png", 0.3, 1300, 50)  # KK image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_lmi.png", 0.4, 75, 900)  # LMI scatter image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_gsp.png", 0.25, 900, 900)  # Gaudi signal image
            composite_image = plot.place_composite(composite_image, f"{plot_filename}_gtp.png", 0.25, 1600, 900)  # Gaudi signi image

            composite_image.save(f"{plot_filename}.png")

            fig = plt.figure(figsize=(15, 15))
            plt.imshow(composite_image)
            plt.axis("off")
            plt.savefig(f"{plot_filename}.pdf", **plot_opt)
            plt.close()
            print(f"\t\tFinal composite figure for region '{region}' and signal '{signal}' -> done.")

            data_moran["Coords"].append(region)
            data_moran["Symbol"].append(region_iter.name)
            data_moran["Gene_id"].append(region_iter.id)
            data_moran["Signal"].append(signal)
            data_moran["R_value"].append(r_value)
            data_moran["P_value"].append(p_value)

            if rmtypes:

                for ext in rmtypes:

                    os.remove(f"{plot_filename}_hic.{ext}")
                    os.remove(f"{plot_filename}_signal.{ext}")
                    os.remove(f"{plot_filename}_kk.{ext}")
                    os.remove(f"{plot_filename}_lmi.{ext}")
                    os.remove(f"{plot_filename}_gsp.{ext}")
                    os.remove(f"{plot_filename}_gtp.{ext}")

    data_moran = pd.DataFrame(data_moran)
    data_moran.to_csv(f"{os.path.join(work_dir, 'moran_info.txt')}", sep="\t", index=False, mode="a")

    print(f"\nInformation saved to {os.path.join(work_dir, 'moran_info.txt')}")
    print(f"\nTotal time spent: {timedelta(seconds=round(time() - start_timer))}")
    print("\nall done.")
