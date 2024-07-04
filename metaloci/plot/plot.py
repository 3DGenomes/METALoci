"""
Functions to generate plots from METALoci objects.
"""
import os
import pathlib
import re
from collections import defaultdict

import libpysal as lp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from metaloci import mlo
from metaloci.graph_layout import kk
from metaloci.misc import misc
from PIL import Image
from pybedtools import BedTool
from scipy.ndimage import rotate
from scipy.stats import linregress
from shapely.geometry import Point


def mixed_matrices_plot(mlobject: mlo.MetalociObject):
    """
    Get a plot of a subset of the Hi-C matrix with the top contact interactions in the matrix (defined by a cutoff) in
    the lower diagonal and the original Hi-C matrix in the upper diagonal, including a Kamada-Kawai plot.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with the flat original matrix (MetalociObject.flat_matrix), the MetalociObject.kk_top_indexes,
        and MetalociObject.kk_cutoff in it.

    Returns
    -------
    fig_matrix : matplotlib.pyplot.figure.Figure
        matplotlib object containing the mixed matrices figure.

    Notes
    -----
    The mixed matrices plot is a plot of the Hi-C matrix for the region. Only the upper triangle of the array is
    represented, rotated 45º counter-clockwise. The top triangle is the original matrix, and the lower triangle is the
    subsetted matrix.
    """

    if mlobject.subset_matrix is None:

        mlobject.subset_matrix = kk.get_subset_matrix(mlobject)

    upper_triangle = np.triu(mlobject.matrix.copy(), k=1)  # Original matrix
    lower_triangle = np.tril(mlobject.subset_matrix, k=-1)  # Top interactions matrix
    mlobject.mixed_matrices = upper_triangle + lower_triangle

    if mlobject.flat_matrix is None:
        
        print("Flat matrix not found in metaloci object. Run get_subset_matrix() first.")
        return None

    fig_matrix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Plot mixed Hi-C matrices
    ax1.imshow(
        mlobject.mixed_matrices,
        cmap="YlOrRd",
        vmax=np.nanquantile(mlobject.flat_matrix[mlobject.kk_top_indexes], 0.99),
    )
    ax1.patch.set_facecolor("black")
    ax1.set_title("Mixed Hi-C Matrices")

    # Kamada-Kawai plot
    kk_plot_to_subplot(ax2, mlobject)
    ax2.set_title("Kamada-Kawai Plot")

    # Density plot of the subsetted matrix
    sns.histplot(
        data=mlobject.flat_matrix[mlobject.kk_top_indexes].flatten(),
        stat="density",
        alpha=0.4,
        kde=True,
        legend=False,
        kde_kws={"cut": 3},
        **{"linewidth": 0},
        ax=ax3
    )
    ax3.set_title("Density Plot of Subsetted Matrix")

    fig_matrix.tight_layout()
    fig_matrix.suptitle(f"{mlobject.chrom}:{mlobject.start}-{mlobject.end}_{mlobject.poi} | "
                        f"cut-off type: {mlobject.kk_cutoff['cutoff_type']}, "
                        f"cut-off: {mlobject.kk_cutoff['values']:.4f} " 
                        f"| persistence length: {mlobject.persistence_length:.2f}", y = 0.025)

    return fig_matrix

def kk_plot_to_subplot(ax, mlobject: mlo.MetalociObject, restraints: bool = True, neighbourhood: float = None):
    """
    Generate Kamada-Kawai plot from pre-calculated restraints and plot it on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes on which to plot the Kamada-Kawai layout.
    mlobject : mlo.MetalociObject
        METALoci object with Kamada-Kawai graphs and nodes (MetalociObject.kk_nodes and MetalociObject.kk_graph)
    restraints : bool, optional
        Boolean to set whether or not to plot restraints, by default True
    neighbourhood: float, optional
        Draw a circle showing the neighbourhood around the point, by default False

    Returns
    -------
    None
    """
    xs = [mlobject.kk_nodes[n][0] for n in mlobject.kk_nodes]
    ys = [mlobject.kk_nodes[n][1] for n in mlobject.kk_nodes]

    options = {"node_size": 50, "edge_color": "black", "linewidths": 0.1, "width": 0.05}

    ax.axis("off")

    if restraints:
        nx.draw(
            mlobject.kk_graph,
            mlobject.kk_nodes,
            node_color=range(len(mlobject.kk_nodes)),
            cmap=plt.cm.coolwarm,
            ax=ax,
            **options,
        )
    else:
        sns.scatterplot(x=xs, y=ys, hue=range(len(xs)), palette="coolwarm", legend=False, s=50, zorder=2, ax=ax)

    g = sns.lineplot(x=xs, y=ys, sort=False, lw=2, color="black", legend=False, zorder=1, ax=ax)
    g.set_aspect("equal", adjustable="box")
    g.set(ylim=(-1.1, 1.1))
    g.set(xlim=(-1.1, 1.1))
    g.tick_params(bottom=False, left=False)
    g.annotate(f"       {mlobject.chrom}:{mlobject.start}", (xs[0], ys[0]), size=9)
    g.annotate(f"       {mlobject.chrom}:{mlobject.end}", (xs[len(xs) - 1], ys[len(ys) - 1]), size=9)

    poi_x, poi_y = xs[mlobject.poi], ys[mlobject.poi]

    if neighbourhood:
        circle = plt.Circle((poi_x, poi_y), neighbourhood, color="red",
                            fill=False, linestyle=":", alpha=0.5, lw=1, zorder=3)
        ax.add_patch(circle)

    sns.scatterplot(x=[poi_x], y=[poi_y], s=50 * 1.5, ec="lime", fc="none", zorder=4, ax=ax)


def get_kk_plot(mlobject: mlo.MetalociObject, restraints: bool = True, neighbourhood: float = None):
    """
    Generate Kamada-Kawai plot from pre-calculated restraints.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with Kamada-Kawai graphs and nodes (MetalociObject.kk_nodes and MetalociObject.kk_graph).
    restraints : bool, optional
        Boolean to set whether or not to plot restraints, by default True.
    neighbourhood: float, optional
        Draw a circle showing the neighbourhood around the point, by default False.

    Returns
    -------
    kk_plt : matplotlib.pyplot.figure.Figure
        Kamada-Kawai layout plot object.
    """
    xs = [mlobject.kk_nodes[n][0] for n in mlobject.kk_nodes]
    ys = [mlobject.kk_nodes[n][1] for n in mlobject.kk_nodes]

    kk_plt = plt.figure(figsize=(10, 10))
    options = {"node_size": 50, "edge_color": "black", "linewidths": 0.1, "width": 0.05}

    plt.axis("off")

    if restraints:
        nx.draw(
            mlobject.kk_graph,
            mlobject.kk_nodes,
            node_color=range(len(mlobject.kk_nodes)),
            cmap=plt.cm.coolwarm,
            **options,
        )
    else:
        sns.scatterplot(x=xs, y=ys, hue=range(len(xs)), palette="coolwarm", legend=False, s=50, zorder=2)

    g = sns.lineplot(x=xs, y=ys, sort=False, lw=2, color="black", legend=False, zorder=1)
    g.set_aspect("equal", adjustable="box")
    g.set(ylim=(-1.1, 1.1))
    g.set(xlim=(-1.1, 1.1))
    g.tick_params(bottom=False, left=False)
    g.annotate(f"       {mlobject.chrom}:{mlobject.start}", (xs[0], ys[0]), size=9)
    g.annotate(f"       {mlobject.chrom}:{mlobject.end}", (xs[len(xs) - 1], ys[len(ys) - 1]), size=9)

    poi_x, poi_y = xs[mlobject.poi], ys[mlobject.poi]

    if neighbourhood:
        circle = plt.Circle((poi_x, poi_y), neighbourhood, color="red",
                            fill=False, linestyle=":", alpha=0.5, lw=1, zorder=3)
        plt.gca().add_patch(circle)

    sns.scatterplot(x=[poi_x], y=[poi_y], s=50 * 1.5, ec="lime", fc="none", zorder=4)

    return kk_plt


def get_hic_plot(mlobject: mlo.MetalociObject, cmap_user: str = "YlOrRd"):
    """
    Create a plot of the HiC matrix for the region. Only the upper triangle of the array is represented, rotated
    45ª counter-clock wise.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix (MetalociObject.matrix) in it.
    cmap_user : str
        Color map used on the plotting of the HiC data, by default YlOrRd.

    Returns
    -------
    matplotlib.pyplot.figure.Figure
        matplotlib object containing the Hi-C matrix figure.
    """

    poi_factor = mlobject.poi / mlobject.lmi_geometry["bin_index"].shape[0]
    array = mlobject.matrix
    matrix_min_value = np.nanmin(array)
    matrix_max_value = np.nanmax(array)
    array = rotate(np.triu(np.nan_to_num(array, nan=0), -1), angle=45)  # Rotate get crazy when rotating with nans.
    mid = int(array.shape[0] / 2)
    poi_x = int(poi_factor * int(array.shape[0]))
    array = array[:mid, :]
    array[array == 0] = np.nan
    hic_fig = plt.figure(figsize=(10, 10))
    ax = hic_fig.add_subplot(111)

    ax.matshow(array, cmap="YlOrRd", vmin=matrix_min_value, vmax=matrix_max_value, label="")
    sns.scatterplot(x=[poi_x], y=[mid + 4], color="lime", marker="^", label="", ax=ax)
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_user),
        ticks=np.linspace(matrix_min_value, matrix_max_value, 6),
        values=np.linspace(matrix_min_value, matrix_max_value, 6),
        shrink=0.3,
        format=FormatStrFormatter("%.2f"),
        ax=ax,
    ).set_label("$log_{10}$ (Hi-C interactions)", rotation=270, size=12, labelpad=20)
    plt.title(f"{mlobject.region}")
    plt.axis("off")

    return hic_fig


def get_gaudi_signal_plot(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame,
                          cmap_user: str = "PuOr_r", mark_regions=None):
    """
    Get a Gaudí signal plot.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.
    lmi_geometry : pd.DataFrame
        Merging from MetalociObject.lmi_info for a specific signal and MetalociObject.geometry for a specific region.
    cmap_user : str
        Color map used on the plotting of the HiC data, by default PuOr_r.

    Returns
    -------
    gsp : matplotlib.pyplot.figure.Figure
        matplotlib figure containing the Gaudi signal plot.

    Notes
    -----
    The Gaudi signal plot is a plot of the LMI values for each bin in the region, where the color of the bin represents
    the signal value. The point of interest is represented by a green point.
    """

    min_value = lmi_geometry.signal.min()
    max_value = lmi_geometry.signal.max()

    gaudi_signal_fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="signal", cmap=cmap_user, linewidth=2, edgecolor="white", ax=ax)

    if mark_regions is not None:

        mini_mark_regions = mark_regions[mark_regions.region_metaloci == mlobject.region].copy()
        mini_mark_regions.drop(columns=["region_metaloci"], inplace=True)

        mini_geometry = lmi_geometry[["bin_chr", "bin_start", "bin_end", "X", "Y"]].copy()

        if not mini_geometry["bin_chr"].str.contains("chr").any():
            mini_geometry["bin_chr"] = "chr" + mini_geometry["bin_chr"]

        intersected_bed = BedTool.from_dataframe(mini_geometry).intersect(
            BedTool.from_dataframe(mini_mark_regions), wa=True, wb=True)

        intersected_bed = intersected_bed.to_dataframe(header=None,
                                                       names=[*mini_geometry.columns, *mini_mark_regions.columns])

        intersected_bed = intersected_bed[["X", "Y", "mark"]]
        intersected_bed = intersected_bed.groupby(["X", "Y"])["mark"].value_counts().unstack(fill_value=0)

        texts = []

        for mini_geo_row in intersected_bed.itertuples():

            sns.scatterplot(x=[mini_geo_row.Index[0]], y=[mini_geo_row.Index[1]], s=10, ec="none", fc="green")

            text = [f" {name}({count})" for name, count in zip(mini_geo_row._fields, mini_geo_row) if name != "Index"]

            texts.append(plt.text(mini_geo_row.Index[0], mini_geo_row.Index[1],
                                  ",".join(text[1:]), ha='center', va="center", fontsize=6, color="black"))

        adjust_text(texts,  force_static=(0.5, 0.5), force_explode=(0.5, 0.5),
                    arrowprops={"arrowstyle": "->, head_length=1, head_width=1", "color": 'black', "lw": 1},
                    ensure_inside_axes=False)

    sns.scatterplot(
        x=[lmi_geometry.X[mlobject.poi]],
        y=[lmi_geometry.Y[mlobject.poi]],
        s=50, ec="none", fc="lime", zorder=len(lmi_geometry))

    sm = plt.cm.ScalarMappable(cmap=cmap_user, norm=plt.Normalize(vmin=min_value, vmax=max_value))
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
    cbar.set_label(f"{lmi_geometry.ID[0]}", rotation=270, size=20, labelpad=35)
    plt.axis("off")

    return gaudi_signal_fig


def get_gaudi_type_plot(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame,
                        signipval: float = 0.05, colors_lmi: dict = None, mark_regions=None
                        ):
    """
    Get a Gaudí type plot.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.
    lmi_geometry : pd.DataFrame
        Merging from MetalociObject.lmi_info for a specific signal and MetalociObject.geometry for a specific region.
    signipval : float, optional
        Significance threshold for p-value, by default 0.05.
    colors_lmi: dict, optional
        Dictionary containing the colors of the quadrants to use in the plot, in matplotlib format;
        by default {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}.

    Returns
    -------
    gtp : matplotlib.pyplot.figure.Figure
        matplotlib figure containing the Gaudi type plot.

    Notes
    -----
    The Gaudi type plot is a plot of the LMI values for each bin in the region, where the color of the bin represents
    the quadrant to which the bin belongs. The significance of the LMI value is represented by the transparency of the
    color, where the more significant the LMI value, the more opaque the color. The point of interest is represented
    by a green point.
    """

    if colors_lmi is None:
        colors_lmi = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_lmi[1], label="HH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_lmi[2], label="LH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_lmi[3], label="LL", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_lmi[4], label="HL", markersize=20),
    ]

    cmap = LinearSegmentedColormap.from_list("Custom cmap", [colors_lmi[nu] for nu in colors_lmi], len(colors_lmi))
    alpha = [1.0 if pval <= signipval else 0.3 for pval in lmi_geometry.LMI_pvalue]

    gaudi_type_fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="moran_quadrant", cmap=cmap, alpha=alpha, linewidth=2, edgecolor="white", ax=ax)
    plt.axis("off")

    if mark_regions is not None:

        miniregions2mark = mark_regions[mark_regions.region_metaloci == mlobject.region].copy()
        miniregions2mark.drop(columns=["region_metaloci"], inplace=True)

        mini_geometry = lmi_geometry[["bin_chr", "bin_start", "bin_end", "X", "Y"]].copy()

        if not mini_geometry["bin_chr"].str.contains("chr").any():
            mini_geometry["bin_chr"] = "chr" + mini_geometry["bin_chr"]

        intersected_bed = BedTool.from_dataframe(mini_geometry).intersect(
            BedTool.from_dataframe(miniregions2mark), wa=True, wb=True)

        intersected_bed = intersected_bed.to_dataframe(header=None,
                                                       names=[*mini_geometry.columns, *miniregions2mark.columns])

        intersected_bed = intersected_bed[["X", "Y", "mark"]]
        intersected_bed = intersected_bed.groupby(["X", "Y"])["mark"].value_counts().unstack(fill_value=0)

        texts = []

        for mini_geo_row in intersected_bed.itertuples():

            sns.scatterplot(x=[mini_geo_row.Index[0]], y=[mini_geo_row.Index[1]], s=10, ec="none", fc="green")

            text = [f" {name}({count})" for name, count in zip(mini_geo_row._fields, mini_geo_row) if name != "Index"]

            texts.append(plt.text(mini_geo_row.Index[0], mini_geo_row.Index[1],
                                  ",".join(text[1:]), ha='center', va="center", fontsize=6, color="black"))

        adjust_text(texts,  force_static=(0.5, 0.5), force_explode=(0.5, 0.5),
                    arrowprops={"arrowstyle": "->, head_length=1, head_width=1", "color": 'black', "lw": 1},
                    ensure_inside_axes=False)

    sns.scatterplot(
        x=[lmi_geometry.X[mlobject.poi]],
        y=[lmi_geometry.Y[mlobject.poi]],
        s=50,
        ec="none",
        fc="lime",
        zorder=len(lmi_geometry),
    )

    ax.legend(handles=legend_elements, frameon=False, fontsize=20, loc="center left", bbox_to_anchor=(1, 0.5))

    return gaudi_type_fig


def signal_plot(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame, neighbourhood: float,
                quadrants: list = None, signipval: float = 0.05, metaloci_only: bool = False):
    """
    Generate a signal plot to visualize the signal intensity and the positions of significant bins.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        A METALoci object.
    lmi_geometry : pd.DataFrame
        A DataFrame containing LMI information and geometry of all bins in the region.
    neighbourhood : float
        Radius of the circle that determines the neighbourhood of each of the points in the Kamada-Kawai layout.
    quadrants : list, optional
        The list of quadrants to consider for selecting bins (default is [1, 3]).
    signipval : float, optional
        The significance p-value threshold for selecting bins (default is 0.05).
    metaloci_only : bool, optional
        If True, the function will only consider the region within a neighborhood radius around 'mlobject.poi',
        otherwise, it will consider all significant bins in 'lmi_geometry' (default is False).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the signal plot.
    matplotlib.axes._subplots.AxesSubplot
        The matplotlib AxesSubplot object representing the plot.

    Notes
    -----
    The function generates a line plot of the signal intensity in 'lmi_geometry' with respect to the bin indices.
    It highlights the bins in the 'metalocis' list with red vertical lines if 'metaloci_only' is True,
    otherwise, it uses different colors for different quadrants.

    Returns matplotlib.axes._subplots.AxesSubplot as it is used to create the composite plot.
    """

    if quadrants is None:

        quadrants = [1, 3]

    bins, coords_b = get_x_axis_label_signal_plot(mlobject)
    metalocis = get_highlight(mlobject, lmi_geometry, neighbourhood, quadrants, signipval, metaloci_only)
    sig_plt = plt.figure(figsize=(10, 1.5))

    for q, m in metalocis.items():

        for p in m:

            if metaloci_only:

                plt.axvline(x=p, color="green", linestyle=":", lw=1, zorder=0, alpha=0.5)

            else:

                color, alpha = get_color_alpha(q)

                plt.axvline(x=p, color=color, linestyle=":", lw=1, zorder=0, alpha=alpha)

    plt.tick_params(axis="both", which="minor", labelsize=24)
    plt.axvline(x=mlobject.poi, color="lime", linestyle="--", lw=1, zorder=1, alpha=0.6)
    plt.xlabel(f"chromosome {mlobject.chrom}")
    plt.xticks(bins, coords_b)
    plt.ylabel(f"{lmi_geometry.ID[0]}")

    ax = sns.lineplot(x=lmi_geometry.bin_index, y=lmi_geometry.signal, color="black", lw=0.7)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.margins(x=0)
    ax.axhline(y=0, color="k", zorder=0)
    sns.despine(top=True, right=True, left=False, bottom=True, offset=None, trim=False)

    return sig_plt, ax


def get_highlight(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame, neighbourhood: float,
                  quadrants: list = None, signipval: float = 0.05, metaloci_only: bool = False) -> dict:
    """
    Generate a dictionary containing metaloci information and their bin indices based on given parameters.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        A METALoci object.
    lmi_geometry : pd.DataFrame
        A DataFrame containing LMI information and geometry of all bins in the region.
    neighbourhood : float
        Radius of the circle that determines the neighbourhood of each of the points in the Kamada-Kawai layout.
    quadrants : list, optional
        The list of quadrants to consider for selecting bins (default is [1, 3]).
    signipval : float, optional
        The significance p-value threshold for selecting bins (default is 0.05).
    metaloci_only : bool, optional
        If True, the function will only consider the region within a neighborhood radius around 'mlobject.poi',
        otherwise, it will consider all significant bins in 'lmi_geometry' (default is False).

    Returns
    -------
    dict
        A dictionary containing bin indices of bins that meet the selection criteria, grouped by quadrants.

    Notes
    -----
    The function filters bins based on their significance p-value and quadrant, as specified by the parameters.
    If 'metaloci_only' is True, it will select bins that are adjacent to the specified 'mlobject.poi' within
    a distance of 'neighbourhood'.

    The resulting bed file data will contain the following columns:
        - 'chr': The chromosome name of the metaloci.
        - 'start': The start position of the metaloci on the chromosome.
        - 'end': The end position of the metaloci on the chromosome.
        - 'bin': The bin index representing the metaloci.
        - 'quadrant': The quadrant to which the metaloci belongs.

    """

    if quadrants is None:

        quadrants = [1, 3]

    bins_to_highlight = defaultdict(list)

    if not metaloci_only:

        for _, row_ml in lmi_geometry.iterrows():

            if row_ml.LMI_pvalue <= signipval and row_ml.moran_quadrant in quadrants:

                bins_to_highlight[row_ml.moran_quadrant].append(row_ml.bin_index)

    else:

        poi_row = lmi_geometry[lmi_geometry.bin_index == mlobject.poi].squeeze()

        if poi_row.LMI_pvalue <= signipval and poi_row.moran_quadrant in quadrants:

            poi_point = Point(lmi_geometry.loc[lmi_geometry["bin_index"] == mlobject.poi, ["X", "Y"]].iloc[0])

            for _, row_ml in lmi_geometry.iterrows():

                adjacent_point = Point((row_ml.X, row_ml.Y))

                if adjacent_point.distance(poi_point) <= neighbourhood:

                    bins_to_highlight[row_ml.moran_quadrant].append(row_ml.bin_index)

                bins_to_highlight[row_ml.moran_quadrant].sort()

    return bins_to_highlight


def get_lmi_scatterplot(mlobject: mlo.MetalociObject, merged_lmi_geometry: pd.DataFrame, neighbourhood: float,
                        signipval: float = 0.05, colors_lmi: dict = None) -> tuple:
    """
    Get a scatterplot of Z-scores of signal vs Z-score of signal spacial lag, given LMI values.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.
    lmi_geometry : pd.DataFrame
        Merging from MetalociObject.lmi_info for a specific signal and MetalociObject.geometry for a specific region.
    neighbourhood : float
        Radius of the circle that determines the neighbourhood of each of the points in the Kamada-Kawai layout.
    signipval : float, optional
        Significance threshold for p-value, by default 0.05.
    colors_lmi : _type_, optional
        Dictionary containing the colors of the quadrants to use in the plot, in matplotlib format;
        by default {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}.

    Returns
    -------
    scatter : matplotlib.pyplot.figure.Figure
        matplotlib figure containing the LMI scatterplot.
    r_value_scat : float
        r-value of the linear regression.
    p-value
        p-value of the linear regression.
    """

    if colors_lmi is None:

        colors_lmi = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}

    weights = lp.weights.DistanceBand.from_dataframe(merged_lmi_geometry, neighbourhood)
    weights.transform = "r"

    # The weird bit is to get the name of the signal it's currently being calculated.
    x = mlobject.lmi_info[merged_lmi_geometry["ID"][0]]["ZSig"]
    y = mlobject.lmi_info[merged_lmi_geometry["ID"][0]]["ZLag"]

    _, _, r_value_scat, p_value_scat, _ = linregress(x, y)
    scatter_fig, ax = plt.subplots(figsize=(5, 5)) 
    alpha_sp = [1.0 if val < signipval else 0.1 for val in merged_lmi_geometry.LMI_pvalue]
    colors_sp = [colors_lmi[val] for val in merged_lmi_geometry.moran_quadrant]

    plt.scatter(x=x, y=y, s=100, ec="white", fc=colors_sp, alpha=alpha_sp)

    sns.scatterplot(
        x=[x[mlobject.poi]], y=[y[mlobject.poi]], s=150, ec="lime", fc="none", zorder=len(merged_lmi_geometry)
    )
    sns.regplot(x=x, y=y, scatter=False, color="k")
    sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=False)

    plt.title(f"Moran Local Scatterplot\n[r: {r_value_scat:4.2f} | p-value: {p_value_scat:.1e}]", fontsize=11)
    plt.axvline(x=0, color="k", linestyle=":")
    plt.axhline(y=0, color="k", linestyle=":")

    ax.set_xlabel(f"Z-score ({merged_lmi_geometry.ID[0]})")
    ax.set_ylabel(f"Z-score ({merged_lmi_geometry.ID[0]} Spatial Lag)")

    r_value_scat = float(r_value_scat)
    p_value_scat = float(r_value_scat)

    return scatter_fig, r_value_scat, p_value_scat


def place_composite(image: Image.Image, image_to_add: str, ifactor: float, ixloc: int, iyloc: int):
    """
    Place an image on a composite image (PIL Image) at a specified location.

    Parameters
    ----------
    new_PI : PIL.Image.Image
        The composite image (PIL Image) to which the new image will be added.
    ifile : str
        The filename or path of the image to be placed on the composite image.
    ifactor : float
        The scaling factor for resizing the input image before placement.
    ixloc : int
        The x-coordinate (horizontal position) at which the new image will be placed on the composite image.
    iyloc : int
        The y-coordinate (vertical position) at which the new image will be placed on the composite image.

    Returns
    -------
    PIL.Image.Image
        The updated composite image (PIL Image) after adding the new image.
    """
    img = Image.open(image_to_add)
    niz = tuple(int(nu * ifactor) for nu in img.size)
    img = img.resize(niz, Image.Resampling.LANCZOS)

    image.paste(img, (ixloc, iyloc))

    return image


def get_x_axis_label_signal_plot(mlobject: mlo.MetalociObject):
    """
    Get bin indexes and corresponding coordinates for a given METALoci object.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        The METALoci object containing LMI geometry information.

    Returns
    -------
    bins : list
        A list of bin indices, representing the start position of each bin.
    coords_b : list
        A list of coordinate values, representing the start position of each bin with comma formatting.

    Notes
    -----
    This function calculates the bin indices and corresponding coordinates based on the 'mlobject' parameters.
    It extracts the LMI geometry information from the 'mlobject' and calculates the bin indices for each quadrant.

    The function returns two lists:
    - 'bins': A list of bin indices, with one element for each quadrant (total of 3 elements).
    - 'coords_b': A list of coordinate values, formatted with commas, representing the start position of each bin.
    """

    bins = []
    coords_b = []

    for i in range(1, 4):

        if i == 1:

            bins.append(int(0))

        else:

            bins.append(int(((i - 1) / 2) * len(mlobject.lmi_geometry)) - 1)

        coords_b.append(f"{mlobject.start + bins[i - 1] * mlobject.resolution:,}")

    return bins, coords_b


def get_color_alpha(quadrant: int):
    """
    Get the color and alpha value for a given quadrant.

    Parameters
    ----------
    quadrant : int
        The quadrant number for which the color and alpha value are to be obtained.

    Returns
    -------
    str
        The color name corresponding to the given quadrant (e.g., "firebrick", "steelblue", or "lightskyblue").
    float
        The alpha value (opacity) for the color, ranging from 0 to 1.
   """
    colors = {1: "firebrick", 3: "steelblue", 2: "lightskyblue"}
    return colors.get(quadrant, "orange"), 0.5


def save_mm_kk(mlobject: mlo.MetalociObject, work_dir: str):
    """
    Save the mixed matrices and Kamada-Kawai plot for a given METALoci object.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        The METALoci object containing the mixed matrices and Kamada-Kawai plot.
    work_dir : str
        The directory path where the plots will be saved.
    """

    pathlib.Path(os.path.join(work_dir, mlobject.chrom, "plots", "KK")).mkdir(
        parents=True, exist_ok=True
    )
    pathlib.Path(os.path.join(work_dir, mlobject.chrom, "plots", "mixed_matrices")).mkdir(
        parents=True, exist_ok=True
    )

    plot_name = f"{re.sub(':|-', '_', mlobject.region)}_"\
            f"{mlobject.kk_cutoff['cutoff_type']}_"\
            f"{mlobject.kk_cutoff['values']:.2f}_"\
            f"pl-{mlobject.persistence_length:.2f}_" + "{}.pdf"

    get_kk_plot(mlobject).savefig(
        os.path.join(
            work_dir,
            mlobject.chrom,
            "plots",
            "KK",
            plot_name.format("KK")
        ), dpi=300
    )

    plt.close()

    mixed_matrices_plot(mlobject).savefig(
        os.path.join(
            work_dir,
            mlobject.chrom,
            "plots",
            "mixed_matrices",
            plot_name.format("mixed_matrices")
        ), dpi=300,
    )

    plt.close()
