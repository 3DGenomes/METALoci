from collections import defaultdict

import geopandas as gpd
import libpysal as lp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image
from scipy.ndimage import rotate
from scipy.stats import linregress, zscore
from shapely.geometry import Point
from shapely.geometry.multipolygon import MultiPolygon

from metaloci import mlo


def get_kk_plot(mlobject):

    plt.figure(figsize=(10, 10))
    options = {"node_size": 50, "edge_color": "black", "linewidths": 0.1, "width": 0.05}

    nx.draw(
        mlobject.kk_graph,
        mlobject.kk_nodes,
        node_color=range(len(mlobject.kk_nodes)),
        cmap=plt.cm.coolwarm,
        **options,
    )

    if mlobject.poi is not None:

        plt.scatter(
            mlobject.kk_nodes[mlobject.poi - 1][0],
            mlobject.kk_nodes[mlobject.poi - 1][1],
            s=80,
            facecolors="none",
            edgecolors="r",
        )

    xs = [mlobject.kk_nodes[n][0] for n in mlobject.kk_nodes]
    ys = [mlobject.kk_nodes[n][1] for n in mlobject.kk_nodes]

    sns.lineplot(x=xs, y=ys, sort=False, lw=2, color="black", legend=False, zorder=1)

    return plt


def get_kk_plot2(mlobject):

    PLOTSIZE = 10
    POINTSIZE = PLOTSIZE * 5

    xs = [mlobject.kk_nodes[n][0] for n in mlobject.kk_nodes]
    ys = [mlobject.kk_nodes[n][1] for n in mlobject.kk_nodes]

    plt.figure(figsize=(PLOTSIZE, PLOTSIZE))
    plt.axis("off")

    g = sns.lineplot(x=xs, y=ys, sort=False, lw=1, color="grey", legend=False, zorder=1)
    g.set_aspect("equal", adjustable="box")
    g.set(ylim=(-1.1, 1.1))
    g.set(xlim=(-1.1, 1.1))
    g.tick_params(bottom=False, left=False)
    g.annotate(f"    {mlobject.chrom}:{mlobject.start}", (xs[0], ys[0]), size=8)
    g.annotate(f"    {mlobject.chrom}:{mlobject.start}", (xs[len(xs) - 1], ys[len(ys) - 8]), size=8)

    sns.scatterplot(x=xs, y=ys, hue=range(len(xs)), palette="coolwarm", legend=False, s=POINTSIZE, zorder=2)
    sns.scatterplot(
        x=[xs[mlobject.poi - 1]], y=[ys[mlobject.poi - 1]], s=POINTSIZE * 1.5, ec="lime", fc="none", zorder=3
    )

    return plt


def mixed_matrices_plot(mlobject):

    if mlobject.flat_matrix is None:

        print("Flat matrix not found in metaloci object. Run get_subset_matrix() first.")
        exit()

    fig_matrix, (ax1, _) = plt.subplots(1, 2, figsize=(20, 5))

    # Plot of the mixed matrix (top triangle is the original matrix,
    # lower triange is the subsetted matrix)
    ax1.imshow(
        mlobject.mixed_matrices,
        cmap="YlOrRd",
        vmax=np.nanquantile(mlobject.flat_matrix[mlobject.kk_top_indexes], 0.99),
    )
    ax1.patch.set_facecolor("black")

    # Density plot of the subsetted matrix
    sns.histplot(
        data=mlobject.flat_matrix[mlobject.kk_top_indexes].flatten(),
        stat="density",
        alpha=0.4,
        kde=True,
        legend=False,
        kde_kws={"cut": 3},
        **{"linewidth": 0},
    )

    fig_matrix.tight_layout()
    fig_matrix.suptitle(f"Matrix for {mlobject.region} (cutoff: {mlobject.kk_cutoff})")

    return mlobject, fig_matrix


def get_hic_plot(mlobject: mlo.MetalociObject, poi_factor: int):
    """
    Create a plot of the HiC matrix for the region. It presents the HiC triangle along the diagonal
    as a horizontal line
    :param arr: HiC matrix
    :param gene: gene name
    :param region_hic: gene region
    :param midp_factor_hic: factor calculated to correctly position the TSS of the gene
    :return: plot object
    """

    array = mlobject.matrix
    matrix_min_value = np.nanmin(array)
    matrix_max_value = np.nanmax(array)

    # Massage the matrix
    array = np.nan_to_num(array, nan=0)  # Rotate get crazy when rotating with nans.
    array = np.triu(array, -1)
    array = rotate(array, angle=45)
    mid = int(array.shape[0] / 2)
    poi_X = int(poi_factor * int(array.shape[0]))

    array = array[:mid, :]
    array[array == 0] = np.nan

    hic_fig = plt.figure(figsize=(10, 10))
    ax = hic_fig.add_subplot(111)

    ax.matshow(array, cmap="YlOrRd", vmin=matrix_min_value, vmax=matrix_max_value, label="")

    sns.scatterplot(x=[poi_X], y=[mid + 4], color="lime", marker="^", label="", ax=ax)

    plt.axis("off")

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="YlOrRd"),
        ticks=np.linspace(matrix_min_value, matrix_max_value, 6),
        values=np.linspace(matrix_min_value, matrix_max_value, 6),
        shrink=0.3,
        label="log10(Hi-C interactions)",
        format=FormatStrFormatter("%.2f"),
        ax=ax,
    )
    cbar.set_label(
        "log10(Hi-C interactions)", rotation=270, size=14, labelpad=20
    )  ## try to squeeze this in previous line
    plt.title(f"[{mlobject.region}]")

    return plt


def get_gaudi_signal_plot(mlobject, lmi_geometry):
    """
    'Gaudi' plot of the signal painted over the Kamada-Kawai layout
    :param mlgdata:
    :param midp_gs:
    :return:
    """
    # Gaudi plot Signal

    poi = lmi_geometry.loc[lmi_geometry["moran_index"] == mlobject.poi - 1, "bin_index"].iloc[0]

    cmap = "PuOr_r"
    min_value = lmi_geometry.signal.min()
    max_value = lmi_geometry.signal.max()

    _, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="signal", cmap=cmap, linewidth=2, edgecolor="white", ax=ax)

    sns.scatterplot(x=[lmi_geometry.X[poi]], y=[lmi_geometry.Y[poi]], s=50, ec="none", fc="lime")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_value, vmax=max_value))
    sm.set_array([1, 2, 3, 4])

    nbar = 11
    cbar = plt.colorbar(
        sm,
        ticks=np.linspace(min_value, max_value, nbar),
        values=np.linspace(min_value, max_value, nbar),
        shrink=0.5,
        format=FormatStrFormatter("%.2f"),
        ax=ax,
    )
    cbar.set_label("Signal", rotation=270, size=20, labelpad=35)
    plt.axis("off")

    return plt


def get_gaudi_type_plot(
    mlobject, lmi_geometry, signipval=0.05, colors={1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}
):
    """
    'Gaudi' plot of the LMI type over the Kamada-Kawai layout
    :param mlgdata:
    :param minpv_gt:
    :param midp_gt:
    :param colors_gt:
    :param legend_info:
    :return:
    """
    # Gaudi plot LMI type

    poi_gaudi = lmi_geometry.loc[lmi_geometry["moran_index"] == mlobject.poi - 1, "bin_index"].iloc[0]

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], label="HH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[2], label="LH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[3], label="LL", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[4], label="HL", markersize=20),
    ]

    cmap = LinearSegmentedColormap.from_list("Custom cmap", [colors[nu] for nu in colors], len(colors))
    alpha = [1.0 if quad <= signipval else 0.3 for quad in lmi_geometry.LMI_pvalue]

    _, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="moran_quadrant", cmap=cmap, alpha=alpha, linewidth=2, edgecolor="white", ax=ax)
    plt.axis("off")

    sns.scatterplot(
        x=[lmi_geometry.X[poi_gaudi]],
        y=[lmi_geometry.Y[poi_gaudi]],
        s=50,
        ec="none",
        fc="lime",
        zorder=len(lmi_geometry),
    )

    ax.legend(handles=legend_elements, frameon=False, fontsize=20, loc="center left", bbox_to_anchor=(1, 0.5))

    return plt


def get_lmi_scatterplot(
    mlobject,
    lmi_geometry,
    neighbourhood,
    signipval=0.05,
    colors_lmi={1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"},
):
    """
    Scatterplot of the LMI signal and the spatial lag of the data.
    :param mlgdata:
    :param bbfact_lmi:
    :param midp_lmi:
    :param minpv_lmi:
    :param colors_lmi:
    :return:
    """

    weights = lp.weights.DistanceBand.from_dataframe(lmi_geometry, neighbourhood)
    weights.transform = "r"

    y_lag = lp.weights.lag_spatial(weights, lmi_geometry["signal"])

    x = zscore(lmi_geometry.signal)
    y = zscore(y_lag)

    _, _, r_value_scat, p_value_scat, _ = linregress(x, y)
    _, ax = plt.subplots(figsize=(5, 5))  # , subplot_kw={'aspect':'equal'})

    alpha_sp = [1.0 if val < signipval else 0.1 for val in lmi_geometry.LMI_pvalue]
    colors_sp = [colors_lmi[val] for val in lmi_geometry.moran_quadrant]

    plt.scatter(x=x, y=y, s=100, ec="white", fc=colors_sp, alpha=alpha_sp)

    sns.scatterplot(
        x=[x[mlobject.poi - 1]], y=[y[mlobject.poi - 1]], s=150, ec="lime", fc="none", zorder=len(lmi_geometry)
    )
    sns.regplot(x=x, y=y, scatter=False, color="k")
    sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=False)

    plt.title(f"Moran Local Scatterplot\nr: {r_value_scat:4.2f}   p-value: {p_value_scat:.1e}")
    plt.axvline(x=0, color="k", linestyle=":")
    plt.axhline(y=0, color="k", linestyle=":")

    ax.set_xlabel("Z-score(Signal)")
    ax.set_ylabel("Z-score(Signal Spatial Lag)")

    return plt, r_value_scat, p_value_scat


def signal_bed(mlobject, lmi_geometry, quartiles, signipval, midds, bbfact):
    """
    Check for the METALoci bins and creation of a BED-type data-frame
    :param mlgdata:
    :param quartiles:
    :param minpv_bed:
    :param midp_bed:
    :param midds_bed:
    :param bbfact_bed:
    :return:
    """
    selsebins = lmi_geometry[
        (lmi_geometry.moran_quadrant.isin(quartiles)) & (lmi_geometry.LMI_pvalue <= signipval)
    ].bin_index.values.tolist()

    mls = lmi_geometry[lmi_geometry.bin_index.isin(selsebins)].unary_union

    if mls:
        if mls.geom_type == "Polygon":
            mls = MultiPolygon([mls])

    s = Point(
        (lmi_geometry[lmi_geometry.bin_index == mlobject.poi].X, lmi_geometry[lmi_geometry.bin_index == mlobject.poi].Y)
    )

    selmetaloci_bed = []

    beddata = defaultdict(list)

    try:

        for _, ml in enumerate(mls.geoms):

            ml = gpd.GeoSeries(ml)

            if s.within(ml[0]):

                for _, row_ml in lmi_geometry.iterrows():

                    s2 = Point((row_ml.X, row_ml.Y))
                    if s2.within(ml[0]):
                        selmetaloci_bed.append(row_ml.bin_index)

                # Add close particles
                selmetaloci_bed.sort()
                closebins = [nu for nu, val in enumerate(midds) if val <= bbfact]
                selmetaloci_bed = np.sort(list(set(closebins + selmetaloci_bed)))

                for p in selmetaloci_bed:
                    beddata["chr"].append(lmi_geometry.bin_chr[p])
                    beddata["start"].append(lmi_geometry.bin_start[p])
                    beddata["end"].append(lmi_geometry.bin_end[p])
                    beddata["bin"].append(p)
    except:

        pass  # Ask Marc about this.
    beddata = pd.DataFrame(beddata)

    return beddata, selmetaloci_bed


def signal_plot(mlobject, lmi_geometry, selmetaloci_sig, bins_sig, coords_sig):
    """
    Lineplot of the signal
    :param mlgdata:
    :param selmetaloci_sig:
    :param midp_sig:
    :param bins_sig:
    :param coords_sig:
    :param chrm_sig:
    :return:
    """
    # Signal plot
    # print('Signal profile for the region of interest: {}'.format(region))

    plt.figure(figsize=(10, 1.5))
    plt.tick_params(axis="both", which="minor", labelsize=24)

    g = sns.lineplot(x=lmi_geometry.bin_index, y=lmi_geometry.signal)

    for p in selmetaloci_sig:

        plt.axvline(x=p, color="red", linestyle=":", lw=1.5)

    plt.axvline(x=mlobject.poi, color="lime", linestyle="--", lw=1.5)

    g.yaxis.set_major_locator(MaxNLocator(integer=True))
    g.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.xlabel(f"chromosome {mlobject.chrom}")
    plt.xticks(bins_sig, coords_sig)
    g.margins(x=0)
    g.axhline(y=0, color="k", zorder=0)

    sns.despine(top=True, right=True, left=False, bottom=True, offset=None, trim=False)

    return plt


def place_composite(new_PI, ifile, ifactor, ixloc, iyloc):
    """
    Image stitcher
    :param new_PI:
    :param ifile:
    :param ifactor:
    :param ixloc:
    :param iyloc:
    :return:
    """
    img = Image.open(ifile)
    niz = tuple([int(nu * ifactor) for nu in img.size])
    img = img.resize(niz, Image.Resampling.LANCZOS)
    new_PI.paste(img, (ixloc, iyloc))
    return new_PI
