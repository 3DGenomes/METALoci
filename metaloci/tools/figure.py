"""
Script to prduce the different plots of METAloci
"""
import argparse
import os
import pathlib
import pickle
import re
import sys
from datetime import timedelta
from time import time

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

from metaloci.plot import plot

description = "This script outputs the different plots to show METALoci.\n"
description += "It creates the following plots:\n"
description += "    - HiC matrix\n"
description += "    - Signal plot\n"
description += "    - Kamada-Kawai layout\n"
description += "    - Local Moran's I scatterplot\n"
description += "    - Gaudí plot for signal\n"
description += "    - Gaudí plot for LMI quadrant\n"
description += "\n"
description += "The script will create a structure of folders and subfolders as follows:\n"
description += "\n"
description += "WORK_DIR\n"
description += "   |-DATASET_NAME\n"
description += "     |-CHROMOSOME\n"
description += "         |-SIGNAL(n)\n"
description += "             |-region_plot\n"

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description, add_help=False)

# The script will create a folder called 'datasets' and a subfolder for the signal,
# with subfolders for every chromosome and region of choice.
input_arg = parser.add_argument_group(title="Input arguments")

input_arg.add_argument(
    "-w", "--work-dir", dest="work_dir", required=True, metavar="PATH", type=str, help="Path to working directory."
)

signal_arg = parser.add_argument_group(title="Signal arguments", description="Choose one of the following options:")

signal_arg.add_argument(
    "-s",
    "--types",
    dest="signals",
    metavar="STR",
    type=str,
    nargs="*",
    action="extend",
    help="Space-separated list of signals to plot.",
)

signal_arg.add_argument(
    "-S",
    "--types-file",
    dest="signals",
    metavar="STR",
    type=str,
    help="Path to the file with the list of signals to plot.",
)

region_input = parser.add_argument_group(title="Region arguments", description="Choose one of the following options:")

region_input.add_argument(
    "-g",
    "--region",
    dest="region_file",
    metavar="FILE",
    type=str,
    nargs="*",
    action="extend",
    help="Region to apply LMI in the format chrN:start-end_midpoint " "gene_symbol gene_id.\n" "Space separated ",
)

region_input.add_argument(
    "-G",
    "--region-file",
    dest="region_file",
    metavar="FILE",
    type=str,
    help="Path to the file with the regions of interest.",
)

optional_arg = parser.add_argument_group(title="Optional arguments")

optional_arg.add_argument("-h", "--help", action="help", help="Show this help message and exit.")

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
    "(default: %(default)s)\n"
    "1: High-high (signal in bin is high, signal on neighbours is "
    "high)\n"
    "2: Low-High (signal in bin is low, signal on neighbours is high)\n"
    "3: Low-Low (signal in bin is low, signal on neighbours is low)\n"
    "4: High-Low (signal in bin is high, signal on neighbours is low)",
)

optional_arg.add_argument(
    "-v",
    "--pval",
    dest="signipval",
    default=0.05,
    metavar="FLOAT",
    type=float,
    help="P-value significance threshold (default: %(default)d).",
)

optional_arg.add_argument("-u", "--debug", dest="debug", action="store_true", help=argparse.SUPPRESS)

args = parser.parse_args(None if sys.argv[1:] else ["-h"])

work_dir = args.work_dir
regions = args.region_file
signals = args.signals
quadrants = args.quart
signipval = args.signipval
rmtypes = args.rm_types
debug = args.debug
agg = args.agg

quadrants = [int(x) for x in quadrants]

if not work_dir.endswith("/"):
    work_dir += "/"

if debug:

    table = [
        ["work_dir", work_dir],
        ["rfile", regions],
        ["signals", signals],
        ["quart", quadrants],
        ["rmtypes", rmtypes],
        ["debug", debug],
    ]
    print(table)
    sys.exit()

INFLUENCE = 1.5
BFACT = 2

colors = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], label="HH", markersize=20),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[2], label="LH", markersize=20),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[3], label="LL", markersize=20),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[4], label="HL", markersize=20),
]

start_timer = time()

if os.path.isfile(regions):

    df_regions = pd.read_table(regions)

else:

    df_regions = pd.DataFrame({"coords": [regions], "symbol": ["symbol"], "id": ["id"]})

# hacer que un archivo de señales sera una lista. cambia esta lógica que es super enrevesada.
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

        print(".mlo file not found for this region. \nSkipping to next region.")

        continue

    buffer = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE
    bbfact = buffer * BFACT

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

        # if re.compile("_").search(signal):

        #     s_class, s_id = signal.split("_")
        #     lmi_data_temp = mlobject.lmi_info[(mlobject.lmi_info.Class == s_class) & (mlobject.lmi_info.ID == s_id)]

        # else:

        merged_lmi_geometry = pd.merge(
            mlobject.lmi_info[signal],
            mlobject.lmi_geometry,
            on=["bin_index", "moran_index"],
            how="inner",
        )

        merged_lmi_geometry = gpd.GeoDataFrame(merged_lmi_geometry, geometry=merged_lmi_geometry.geometry)

        # The creation of the merged dataframe should be a function in misc. Put there the coditions of aggregation.

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

        print(f"\t\tFinal composite figure for region of interest: {region} and signal: {signal}", end="\r")
        img1 = Image.open(f"{plot_filename}_lmi.png")
        img2 = Image.open(f"{plot_filename}_gsp.png")
        img3 = Image.open(f"{plot_filename}_gtp.png")

        maxx = int((img1.size[1] * 0.4 + img2.size[1] * 0.25 + img3.size[1] * 0.25) * 1.3)

        composite_image = Image.new(mode="RGBA", size=(maxx, 1550))

        # HiC image
        composite_image = plot.place_composite(composite_image, f"{plot_filename}_hic.png", 0.5, 100, 50)
        # Singal image
        composite_image = plot.place_composite(composite_image, f"{plot_filename}_signal.png", 0.4, 42, 660)
        # KK image
        composite_image = plot.place_composite(composite_image, f"{plot_filename}_kk.png", 0.3, 1300, 50)
        # MLI scatter image
        composite_image = plot.place_composite(composite_image, f"{plot_filename}_lmi.png", 0.4, 75, 900)
        # Gaudi signal image
        composite_image = plot.place_composite(composite_image, f"{plot_filename}_gsp.png", 0.25, 900, 900)
        # Gaudi signi image
        composite_image = plot.place_composite(composite_image, f"{plot_filename}_gtp.png", 0.25, 1600, 900)

        composite_image.save(f"{plot_filename}.png")

        fig = plt.figure(figsize=(15, 15))
        plt.imshow(composite_image)
        plt.axis("off")
        plt.savefig(f"{plot_filename}.pdf", **plot_opt)
        plt.close()
        print(f"\t\tFinal composite figure for region of interest: {region} and signal: {signal} -> done.")

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
