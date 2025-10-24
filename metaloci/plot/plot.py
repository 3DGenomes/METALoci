"""
Functions to generate plots from METALoci objects.
"""
import glob
import os
import pathlib
import re
from collections import defaultdict

import geopandas as gpd
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
from PIL import Image, ImageDraw, ImageFont
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

    if mlobject.poi is not None: 

        poi_x, poi_y = xs[mlobject.poi], ys[mlobject.poi]
        sns.scatterplot(x=[poi_x], y=[poi_y], s=50 * 1.5, ec="lime", fc="none", zorder=4, ax=ax)

    if neighbourhood:

        circle = plt.Circle((poi_x, poi_y), neighbourhood, color="red",
                            fill=False, linestyle=":", alpha=0.5, lw=1, zorder=3)
        ax.add_patch(circle)


def get_kk_plot(mlobject: mlo.MetalociObject, restraints: bool = True, 
                neighbourhood: bool =  False, remove_poi: bool = False):
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

    if remove_poi and mlobject.poi is not None:
        
        mlobject.poi = None

    if mlobject.poi is not None: # in case of 'metaloci scan' where we are removing the poi

        poi_x, poi_y = xs[mlobject.poi], ys[mlobject.poi]

        if neighbourhood:

            INFLUENCE = 1.5
            BFACT = 2

            neighbourhood_value = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT

            circle = plt.Circle((poi_x, poi_y), neighbourhood_value, color="black",
                                fill=False, linestyle="--", alpha=0.6, lw=1.8, zorder=3)
            plt.gca().add_patch(circle)

        sns.scatterplot(x=[poi_x], y=[poi_y], s=50 * 1.5, ec="lime", fc="none", zorder=4)

    return kk_plt


def get_hic_plot(mlobject: mlo.MetalociObject, del_args = None, cmap_user: str = "YlOrRd", clean_mat: bool = False):
    """
    Create a plot of the HiC matrix for the region. Only the upper triangle of the array is represented, rotated
    45ª counter-clock wise.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix (MetalociObject.matrix) in it.
    cmap_user : str
        Color map used on the plotting of the HiC data, by default YlOrRd.
    clean_mat : bool
        If True, the function will use the subset matrix, otherwise it will use the original matrix, by default False.

    Returns
    -------
    matplotlib.pyplot.figure.Figure
        matplotlib object containing the Hi-C matrix figure.
    """

    if clean_mat: # This takes the top interactions matrix used in KK instead of the original Hi-C.

        array = mlobject.subset_matrix
        array[np.diag_indices_from(array)] = 0
        np.fill_diagonal(array[:-1, 1:], 0)
        np.fill_diagonal(array[1:, :-1], 0)

    else:

        array = mlobject.matrix

    # If doing metaloci scan deletions, insert a stretch of zeroes
    if del_args is not None and del_args.delete_indices is not None:  

        delete_indices = sorted(set(del_args.delete_indices))
        delete_indices = [i for i in delete_indices if 0 <= i <= array.shape[0]]
        original_size = array.shape[0]
        final_size = original_size + len(delete_indices)

        # Initialize new larger matrix filled with NaNs
        new_array = np.full((final_size, final_size), np.nan)

        # Insert rows and columns from original array into the new array,
        # skipping positions that are in delete_indices
        original_i = 0  # index in the original matrix

        for new_i in range(final_size):

            if new_i in delete_indices:

                continue  # leave row/column as NaN

            original_j = 0

            for new_j in range(final_size):

                if new_j in delete_indices:

                    continue

                new_array[new_i, new_j] = array[original_i, original_j]
                original_j += 1

            original_i += 1

        array = new_array

    len_array = array.shape[0]
    matrix_min_value = np.nanmin(array)
    matrix_max_value = np.nanmax(array)
    array = rotate(np.triu(np.nan_to_num(array, nan=0), -1), angle=45)  # Rotate get crazy when rotating with nans.
    mid = int(array.shape[0] / 2)

    if mlobject.poi is not None: # In case of 'metaloci scan' where we are removing the poi
    
        poi_factor = mlobject.poi / mlobject.lmi_geometry["bin_index"].shape[0]
        poi_x = int(poi_factor * int(array.shape[0]))
        
        if del_args is not None and del_args.delete_indices is not None:  

            poi_factor = mlobject.poi / (len_array) # lmi_geometry now has different length due to deletions
            poi_x = int(poi_factor * int(array.shape[0]))

    array = array[:mid, :]
    array[array == 0] = np.nan
    hic_fig = plt.figure(figsize=(10, 10))
    ax = hic_fig.add_subplot(111)

    ax.matshow(array, cmap="YlOrRd", vmin=matrix_min_value, vmax=matrix_max_value, label="")
    
    if mlobject.poi is not None: # In case of 'metaloci scan' where we are removing the poi
        
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
                          cmap_user: str = "PuOr_r", mark_regions: pd.DataFrame = None, neighbourhood: bool = False):
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
    mark_regions : pd.DataFrame, optional
        DataFrame containing the regions to be marked on the plot, by default None.
    neighbourhood : bool, optional
        Draw a circle showing the neighbourhood around the point, by default False.
    
    Returns
    -------
    gsp : matplotlib.pyplot.figure.Figure
        matplotlib figure containing the Gaudi signal plot.

    Notes
    -----
    The Gaudi signal plot is a plot of the LMI values for each bin in the region, where the color of the bin represents
    the signal value. The point of interest is represented by a green point.
    """

    min_value = lmi_geometry.Sig.min()
    max_value = lmi_geometry.Sig.max()

    gaudi_signal_fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="Sig", cmap=cmap_user, linewidth=2, edgecolor="white", ax=ax)

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

    if mlobject.poi is not None: # in case of 'metaloci scan' where we are removing the poi

        sns.scatterplot(
            x=[lmi_geometry.X[mlobject.poi]],
            y=[lmi_geometry.Y[mlobject.poi]],
            s=120, ec="none", fc="lime", zorder=len(lmi_geometry))
        
        if neighbourhood:

            xs = [lmi_geometry.X[n] for n in lmi_geometry.index]
            ys = [lmi_geometry.Y[n] for n in lmi_geometry.index]

            poi_x, poi_y = xs[mlobject.poi], ys[mlobject.poi]

            INFLUENCE = 1.5
            BFACT = 2

            neighbourhood_value = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT

            circle = plt.Circle((poi_x, poi_y), neighbourhood_value, color="black",
                                fill=False, linestyle="--", alpha=0.6, lw=1.8, zorder=3)
            ax.add_patch(circle)
        
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
                        signipval: float = 0.05, colors_lmi: dict = None, mark_regions=None, neighbourhood=False):
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

    if mlobject.poi is not None: # in case of 'metaloci scan' where we are removing the poi

        sns.scatterplot(
            x=[lmi_geometry.X[mlobject.poi]],
            y=[lmi_geometry.Y[mlobject.poi]],
            s=120,
            ec="none",
            fc="lime",
            zorder=len(lmi_geometry),
        )

        if neighbourhood:
            
            xs = [lmi_geometry.X[n] for n in lmi_geometry.index]
            ys = [lmi_geometry.Y[n] for n in lmi_geometry.index]

            poi_x, poi_y = xs[mlobject.poi], ys[mlobject.poi]

            INFLUENCE = 1.5
            BFACT = 2

            neighbourhood_value = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT
            circle = plt.Circle((poi_x, poi_y), neighbourhood_value, color="black",
                                fill=False, linestyle="--", alpha=0.6, lw=1.8, zorder=3)
            ax.add_patch(circle)

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

    if mlobject.poi is not None: # in case of 'metaloci scan' where we are removing the poi

        plt.axvline(x=mlobject.poi, color="lime", linestyle="--", lw=1, zorder=1, alpha=0.6)

    plt.xlabel(f"chromosome {mlobject.chrom}")
    plt.xticks(bins, coords_b)
    plt.ylabel(f"{lmi_geometry.ID[0]}")

    # Cast lmi_geometry.Sig to float32 to avoid new matplotlib version error
    lmi_geometry.Sig = lmi_geometry.Sig.astype(np.float32)
    
    ax = sns.lineplot(x=lmi_geometry.bin_index, y=lmi_geometry.Sig, color="black", lw=0.7)

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

        if mlobject.poi is not None:

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
                        signipval: float = 0.05, zscore_signal: bool = False, colors_lmi: dict = None) -> tuple:
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

    if zscore_signal:

        # The weird bit is to get the name of the signal it's currently being calculated.
        x = mlobject.lmi_info[merged_lmi_geometry["ID"][0]]["ZSig"]
        y = mlobject.lmi_info[merged_lmi_geometry["ID"][0]]["ZLag"]

    else:

        x = mlobject.lmi_info[merged_lmi_geometry["ID"][0]]["Sig"]
        y = mlobject.lmi_info[merged_lmi_geometry["ID"][0]]["Lag"]

    _, _, r_value_scat, p_value_scat, _ = linregress(x, y)
    scatter_fig, ax = plt.subplots(figsize=(5, 5))
    alpha_sp = [1.0 if val < signipval else 0.1 for val in merged_lmi_geometry.LMI_pvalue]
    colors_sp = [colors_lmi[val] for val in merged_lmi_geometry.moran_quadrant]

    plt.scatter(x=x, y=y, s=100, ec="white", c=colors_sp, alpha=alpha_sp)

    if mlobject.poi is not None: # in case of 'metaloci scan' where we are removing the poi

        sns.scatterplot(
            x=[x[mlobject.poi]], y=[y[mlobject.poi]], s=150, ec="lime", fc="none",
            zorder=len(merged_lmi_geometry), marker = "o"
            )
        
    sns.regplot(x=x, y=y, scatter=False, color="k", truncate = True, n_boot=10000, ci=None)
    sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=False)

    plt.title(f"Moran Local Scatterplot\n[r: {r_value_scat:4.2f} | p-value: {p_value_scat:.1e}]", fontsize=11)
    plt.axvline(x=np.mean(x), color="k", linestyle=":")
    plt.axhline(y=np.mean(y), color="k", linestyle=":")

    if zscore_signal:
        
        ax.set_xlabel(f"Z-score ({merged_lmi_geometry.ID[0]})")
        ax.set_ylabel(f"Z-score ({merged_lmi_geometry.ID[0]} Spatial Lag)")

    else:

        ax.set_xlabel(f"{merged_lmi_geometry.ID[0]}")
        ax.set_ylabel(f"{merged_lmi_geometry.ID[0]} Spatial Lag")

    r_value_scat = float(r_value_scat)
    p_value_scat = float(p_value_scat)

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
    Get the x-axis labels for the signal plot.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.

    Returns
    -------
    bins : list
        List of bin indices for the x-axis labels.
    coords_b : list
        List of coordinate labels for the x-axis.
    """

    bins = [0, len(mlobject.lmi_geometry) // 2 - 1, len(mlobject.lmi_geometry) - 1]
    coords_b = [f"{mlobject.start + b * mlobject.resolution:,}" for b in bins]

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


def save_mm_kk(mlobject: mlo.MetalociObject, work_dir: str, remove_poi: bool = False):
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

    get_kk_plot(mlobject, neighbourhood = True, remove_poi = remove_poi).savefig(
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


def create_composite_figure(mlobject: mlo.MetalociObject, signal_type: str, del_args: pd.Series = False, 
                            neighourhood_circle: bool = False, args: pd.Series = False, mark_regions: pd.DataFrame = None, 
                            signipval : float = 0.05, silent : bool = False):
    """
    Generates a composite figure consisting of multiple plots related to Hi-C data analysis and saves the resulting 
    images in various formats. The function handles different configurations based on the provided arguments.
    Parameters:
        mlobject (mlo.MetalociObject): The MetalociObject containing Hi-C data and associated metadata.
        signal_type (str): The type of signal to be plotted (e.g., "wt", "intact").
        del_args (pd.Series, optional): A pandas Series containing deletion-related arguments. Defaults to False.
        neighourhood_circle (bool, optional): Whether to include neighborhood circles in the plots. Defaults to False.
        args (pd.Series, optional): A pandas Series containing additional arguments. Defaults to False.
        mark_regions (pd.DataFrame, optional): A DataFrame specifying regions to be marked on the plots. Defaults to None.
        signipval (float, optional): The significance p-value threshold for certain plots. Defaults to 0.05.
        silent (bool, optional): If True, suppresses console output. Defaults to False.

    Returns:
        None

    Side Effects:
        - Saves multiple plot images (e.g., Hi-C plot, Kamada-Kawai plot, Gaudi Signal plot, etc.) in PNG and PDF formats.
        - Creates a composite image combining all individual plots.
        - Deletes intermediate plot files after the composite image is created.

    Notes:
        - The function dynamically adjusts plot filenames and configurations based on the provided arguments.
        - If `del_args` is provided, additional processing is performed to handle deletion-specific configurations.
        - The composite image is saved as a PNG and optionally as a PDF, with intermediate files being removed afterward.
    
    """

    plot_opt = {"bbox_inches": "tight", "dpi": 300, "transparent": True}
    INFLUENCE = 1.5
    BFACT = 2
    
    if del_args is not None:

        if del_args.wt:

            plot_filename = os.path.join(del_args.work_dir, "scan", mlobject.chrom, "plots",
                            signal_type, 
                            f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_{del_args.create_gif}")
            pathlib.Path(plot_filename).mkdir(parents=True, exist_ok=True)

            plot_filename = os.path.join(
                plot_filename,
                f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_"
                f"{mlobject.resolution}_{signal_type}_wt",
            )

            number = "wt"

        else:
            
            plot_filename = os.path.join(del_args.work_dir, "scan", mlobject.chrom, "plots",
                            signal_type, 
                            f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{del_args.intact_poi}_{del_args.create_gif}")

            if del_args.create_gif is not None:

                delete_indices = list(range(del_args.i, del_args.i + del_args.num_bins_to_delete))

                if del_args.create_gif in delete_indices:

                    mlobject.poi = None

                else:

                    mlobject.poi = del_args.create_gif

            else:
            
                mlobject.poi = del_args.intact_poi

            plot_filename = os.path.join(
                plot_filename,
                f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{del_args.intact_poi}_"
                f"{mlobject.resolution}_{signal_type}_{mlobject.save_path.rsplit('.mlo', 1)[0].rsplit('_', 1)[1]}",
            )

            number = mlobject.save_path.rsplit('.mlo', 1)[0].rsplit('_', 1)[1]

        neighourhood_circle = True
        signipval = 0.05
        mark_regions = None

    else: 

        plot_filename = os.path.join(args.work_dir, mlobject.chrom, "plots",
                                    signal_type, f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}")
        pathlib.Path(plot_filename).mkdir(parents=True, exist_ok=True)

        plot_filename = os.path.join(
            plot_filename,
            f"{mlobject.chrom}_{mlobject.start}_{mlobject.end}_{mlobject.poi}_{mlobject.resolution}_{signal_type}",
        )

    merged_lmi_geometry = pd.merge(
        mlobject.lmi_info[signal_type],
        mlobject.lmi_geometry,
        on=["bin_index", "moran_index"],
        how="left",
    )

    merged_lmi_geometry = gpd.GeoDataFrame(merged_lmi_geometry, geometry=merged_lmi_geometry.geometry)
    neighbourhood = mlobject.kk_distances.diagonal(1).mean() * INFLUENCE * BFACT

    if not silent:
        
        print("\t\tHi-C plot", end="\r")

    hic_plt = get_hic_plot(mlobject, del_args, clean_mat=False)

    hic_plt.savefig(f"{plot_filename}_hic.pdf", **plot_opt)
    hic_plt.savefig(f"{plot_filename}_hic.png", **plot_opt)
    plt.close()

    if not silent:

        print("\t\tHi-C plot -> done.")
        print("\t\tKamada-Kawai plot", end="\r")

    kk_plt = get_kk_plot(mlobject, neighbourhood=neighourhood_circle)
    kk_plt.savefig(f"{plot_filename}_kk.pdf", **plot_opt)
    kk_plt.savefig(f"{plot_filename}_kk.png", **plot_opt)
    plt.close()

    if not silent:
        
        print("\t\tKamada-Kawai plot -> done.")
        print("\t\tGaudi Signal plot", end="\r")

    gs_plt = get_gaudi_signal_plot(mlobject, merged_lmi_geometry, mark_regions=mark_regions,
                                        neighbourhood=neighourhood_circle)
    gs_plt.savefig(f"{plot_filename}_gsp.pdf", **plot_opt)
    gs_plt.savefig(f"{plot_filename}_gsp.png", **plot_opt)
    plt.close()

    if not silent:
        
        print("\t\tGaudi Signal plot -> done.")
        print("\t\tGaudi Type plot", end="\r")

    gt_plt = get_gaudi_type_plot(mlobject, merged_lmi_geometry, signipval, mark_regions=mark_regions,
                                    neighbourhood=neighourhood_circle)
    gt_plt.savefig(f"{plot_filename}_gtp.pdf", **plot_opt)
    gt_plt.savefig(f"{plot_filename}_gtp.png", **plot_opt)
    plt.close()
    
    if not silent:
        
        print("\t\tGaudi Type plot -> done.")
        print("\t\tSignal plot", end="\r")

    sig_plt, ax = signal_plot(mlobject, merged_lmi_geometry, neighbourhood,
                                    [1, 3], signipval, False)

    sig_plt.savefig(f"{plot_filename}_signal.pdf", **plot_opt)
    sig_plt.savefig(f"{plot_filename}_signal.png", **plot_opt)
    plt.close()

    if not silent:

        print("\t\tSignal plot -> done.")
        print("\t\tLMI Scatter plot", end="\r")

    lmi_plt, r_value, p_value = get_lmi_scatterplot(mlobject, merged_lmi_geometry,
                                                            neighbourhood, signipval)

    if lmi_plt is not None:

        lmi_plt.savefig(f"{plot_filename}_lmi.pdf", **plot_opt)
        lmi_plt.savefig(f"{plot_filename}_lmi.png", **plot_opt)
        plt.close()
        
        if not silent:
            
            print("\t\tLMI Scatter plot -> done.")

    if del_args is not None:

        create_number_image(
            number=number,
            output_path=f"{plot_filename}_number.png",
            font_size=192,
            image_size=(512, 512)
        )

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

    # create a new png image of size 2x2 with the number 1

    composite_image = Image.new(mode="RGBA", size=(maxx, 1550))
    # HiC image
    composite_image = place_composite(composite_image, f"{plot_filename}_hic.png", 0.5, 100, 50)
    # Signal image
    composite_image = place_composite(composite_image, f"{plot_filename}_signal.png", 0.4,
                                            signal_left[max_chr_yax] + negative_axis_correction, 640)
    # KK image
    composite_image = place_composite(composite_image, f"{plot_filename}_kk.png", 0.3, 1300, 50)
    # LMI scatter image
    composite_image = place_composite(composite_image, f"{plot_filename}_lmi.png", 0.4, 75, 900)
    # Gaudi signal image
    composite_image = place_composite(composite_image, f"{plot_filename}_gsp.png", 0.25, 900, 900)
    # Gaudi type image
    composite_image = place_composite(composite_image, f"{plot_filename}_gtp.png", 0.25, 1600, 900)
    # Number image
    if del_args is not None:

        composite_image = place_composite(composite_image, f"{plot_filename}_number.png", 0.5, 15, 10)

    composite_image.save(f"{plot_filename}.png")
    plt.figure(figsize=(15, 15))
    plt.imshow(composite_image)
    plt.axis("off")
    plt.savefig(f"{plot_filename}.pdf", **plot_opt)
    plt.close()

    os.remove(f"{plot_filename}_hic.png")
    os.remove(f"{plot_filename}_signal.png")
    os.remove(f"{plot_filename}_kk.png")
    os.remove(f"{plot_filename}_lmi.png")
    os.remove(f"{plot_filename}_gsp.png")
    os.remove(f"{plot_filename}_gtp.png")

    if del_args is not None:

        os.remove(f"{plot_filename}_hic.pdf")
        os.remove(f"{plot_filename}_signal.pdf")
        os.remove(f"{plot_filename}_kk.pdf")
        os.remove(f"{plot_filename}_lmi.pdf")
        os.remove(f"{plot_filename}_gsp.pdf")
        os.remove(f"{plot_filename}_gtp.pdf")
        os.remove(f"{plot_filename}.pdf")
        os.remove(f"{plot_filename}_number.png")


def create_number_image(output_path: os.path, number: int = None, font_size: float = 192, 
                        image_size: tuple = (512, 512), font_path: os.path = None):
    """
    Creates an image with a specified number rendered at the center and saves it as a PNG file.
    Parameters:
        number (int or str): The number to render on the image.
        output_path (str): The file path where the generated image will be saved.
        font_size (int): The size of the font to use for rendering the number.
        image_size (tuple): The size of the image in pixels as a tuple (width, height).
        font_path (str, optional): The file path to a TTF font file. If not provided, the function
        will attempt to use the "DejaVuSans-Bold.ttf" font. If this font is not available,
        a RuntimeError will be raised.
    Notes:

        - The image is created with a transparent background.
        - The number is rendered in black color.
        - The output directory will be created if it does not exist.
    """
    # Create a blank RGBA image
    img = Image.new("RGBA", image_size, (255, 255, 255, 0))  # Transparent background
    draw = ImageDraw.Draw(img)

    # Use a real TTF font
    if font_path is None:

        try:

            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)

        except OSError:

            raise RuntimeError("Please provide a TTF font path via `font_path`, or install DejaVuSans.")
        
    else:

        font = ImageFont.truetype(font_path, font_size)

    # Get text bounding box and center it
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    # Draw the number
    draw.text(position, text, fill=(0, 0, 0, 255), font=font)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")


def get_lmi_change_scan_plot(moran_data_folder: os.path, results_folder: os.path = None, poi: int = None):
    """
    Generates and saves a scatter plot visualizing the change in LMI (Local Moran's I) scores 
    over a series of deletions for a given point of interest (POI).
    The function reads multiple TSV files from the specified folder, extracts the LMI scores 
    for the given POI, and calculates the mean and standard deviation of the scores. It then 
    plots the scores along with the mean, ±2 standard deviation range, and highlights points 
    that fall outside this range.

    Parameters:
        moran_data_folder (str): Path to the folder containing the TSV files with Moran's I data.
        poi (int): The point of interest (POI) for which the LMI scores are analyzed.

    Saves:
        A PNG file named `lmi_change_scan_plot_<poi>.png` in the specified folder, containing 
        the generated plot.
        
    Notes:
        - Points with LMI scores below (mean - 2 * std_dev) are highlighted in blue, and points 
          above (mean + 2 * std_dev) are highlighted in red.
    """

    moran_data_paths = glob.glob(f"{moran_data_folder}/*.tsv")[:-1]
    moran_data_wt_path = glob.glob(f"{moran_data_folder}/*.tsv")[-1]
    moran_data_paths = misc.natural_sort(moran_data_paths)

    moran_data = {}

    for i, path in enumerate(moran_data_paths):

        df = pd.read_csv(path, sep="\t")
        
        match = df[df.iloc[:, 1] == poi]  

        if not match.empty:

            moran_data[i] = match.iloc[0, 8] 

    moran_data = pd.DataFrame.from_dict(moran_data, orient="index", columns=["LMI_score"])
    # moran_data_wt = pd.read_csv(moran_data_wt_path, sep="\t")
    # moran_data_wt = moran_data_wt.iloc[poi, 8]

    # Calculate mean and standard deviation
    mean_lmi = moran_data["LMI_score"].mean()
    std_dev = moran_data["LMI_score"].std()

    plt.figure(figsize=(20, 3))
    sns.scatterplot(x=moran_data.index, y=moran_data["LMI_score"],
                    color="gray", edgecolor="black", label="LMI score")
    plt.axhline(y=mean_lmi, color="black", linestyle="--", label="mean LMI score")
    plt.axhspan(mean_lmi - 2 * std_dev, mean_lmi + 2 * std_dev, color="gray", alpha=0.3, label="±2 SD")

    # Tight limits with slight padding
    ymin = moran_data["LMI_score"].min()
    ymax = moran_data["LMI_score"].max()
    padding = 0.05 * (ymax - ymin)
    ymin -= padding
    ymax += padding
    plt.ylim(ymin, ymax)

    for x, row in moran_data.iterrows():

        score = row["LMI_score"]

        if score < mean_lmi - 2 * std_dev:

            plt.vlines(x=x, ymin=ymin, ymax=score, color="blue", linestyle=":", lw=1, alpha=0.5)

        elif score > mean_lmi + 2 * std_dev:

            plt.vlines(x=x, ymin=ymin, ymax=score, color="red", linestyle=":", lw=1, alpha=0.5)
    
    xticks = [x for x in moran_data.index if x % 50 == 0]
    plt.xticks(xticks, xticks)    
    plt.ylabel("LMI Score")
    plt.title(f"LMI Score change over deletions for poi {poi}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"lmi_change_scan_plot_{poi}.png"), dpi=300)
    plt.close()