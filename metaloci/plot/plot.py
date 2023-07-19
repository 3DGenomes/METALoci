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
from metaloci.misc import misc


def get_kk_plot(mlobject: mlo.MetalociObject, restraints: bool = True):
    """
    Generate Kamada-Kawai plot from pre-calculated restraints.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with Kamada-Kawai graphs and nodes (MetalociObject.kk_nodes and MetalociObject.kk_graph)
    restraints : bool, optional
        Boolean to set whether or not to plot restraints, by default True

    Returns
    -------
    matplotlib.pyplot.figure.Figure
        Kamada-Kawai layout plot object.
    """

    PLOTSIZE = 10
    POINTSIZE = PLOTSIZE * 5

    xs = [mlobject.kk_nodes[n][0] for n in mlobject.kk_nodes]
    ys = [mlobject.kk_nodes[n][1] for n in mlobject.kk_nodes]

    kk_plt = plt.figure(figsize=(PLOTSIZE, PLOTSIZE))
    plt.axis("off")
    options = {"node_size": 50, "edge_color": "black", "linewidths": 0.1, "width": 0.05}

    if restraints == True:

        nx.draw(
            mlobject.kk_graph,
            mlobject.kk_nodes,
            node_color=range(len(mlobject.kk_nodes)),
            cmap=plt.cm.coolwarm,
            **options,
        )

    else:

        sns.scatterplot(x=xs, y=ys, hue=range(len(xs)), palette="coolwarm", legend=False, s=POINTSIZE, zorder=2)

    g = sns.lineplot(x=xs, y=ys, sort=False, lw=2, color="black", legend=False, zorder=1)
    g.set_aspect("equal", adjustable="box")
    g.set(ylim=(-1.1, 1.1))
    g.set(xlim=(-1.1, 1.1))
    g.tick_params(bottom=False, left=False)
    g.annotate(f"      {mlobject.chrom}:{mlobject.start}", (xs[0], ys[0]), size=8)
    g.annotate(f"      {mlobject.chrom}:{mlobject.end}", (xs[len(xs) - 1], ys[len(ys) - 1]), size=8)

    sns.scatterplot(
        x=[xs[mlobject.poi]], y=[ys[mlobject.poi]], s=POINTSIZE * 1.5, ec="lime", fc="none", zorder=3
    )

    return kk_plt


def mixed_matrices_plot(mlobject: mlo.MetalociObject):
    """
    Get a plot of a subset of the Hi-C matrix with the top contact interactions in the matrix (defined by a cutoff) in
    the lower diagonal and the original Hi-C matrix in the upper diagonal.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with the flat original matrix (MetalociObject.flat_matrix), the MetalociObject.kk_top_indexes,
        and MetalociObject.kk_cutoff in it.

    Returns
    -------
    fig_matrix : matplotlib.pyplot.figure.Figure
        matplotlib object containing the mixed matrices figure.

    """

    # Mix matrices to plot:
    if mlobject.subset_matrix is None:

        mlobject.subset_matrix = misc.get_subset_matrix(mlobject)

    upper_triangle = np.triu(mlobject.matrix.copy(), k=1)  # Original matrix
    lower_triangle = np.tril(mlobject.subset_matrix, k=-1)  # Top interactions matrix
    mlobject.mixed_matrices = upper_triangle + lower_triangle

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

    return fig_matrix


def get_hic_plot(mlobject: mlo.MetalociObject):
    """
    Create a plot of the HiC matrix for the region. Only the upper triangle of the array is represented, rotated
    45ª counter-clock wise.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with a matrix (MetalociObject.matrix) in it.

    Returns
    -------
    matplotlib.pyplot.figure.Figure
        matplotlib object containing the Hi-C matrix figure.
    """

    poi_factor = mlobject.poi / mlobject.lmi_geometry["bin_index"].shape[0]

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

    plt.colorbar(
        plt.cm.ScalarMappable(cmap="YlOrRd"),
        ticks=np.linspace(matrix_min_value, matrix_max_value, 6),
        values=np.linspace(matrix_min_value, matrix_max_value, 6),
        shrink=0.3,
        label="log10(Hi-C interactions)",
        format=FormatStrFormatter("%.2f"),
        ax=ax,
    ).set_label("log10(Hi-C interactions)", rotation=270, size=14, labelpad=20)

    plt.title(f"[{mlobject.region}]")
    plt.axis("off")

    return hic_fig


def get_gaudi_signal_plot(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame):
    """
    Get a Gaudí signal plot.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.
    lmi_geometry : pd.DataFrame
        Merging from MetalociObject.lmi_info for a specific signal and MetalociObject.geometry for a specific region.

    Returns
    -------
    gsp : matplotlib.pyplot.figure.Figure
        matplotlib figure containing the Gaudi signal plot.
    """

    poi = lmi_geometry.loc[lmi_geometry["moran_index"] == mlobject.poi, "bin_index"].iloc[0]

    cmap = "PuOr_r"
    min_value = lmi_geometry.signal.min()
    max_value = lmi_geometry.signal.max()

    gsp, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
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

    return gsp


def get_gaudi_type_plot(
    mlobject: mlo.MetalociObject,
    lmi_geometry: pd.DataFrame,
    signipval: float = 0.05,
    colors: dict = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"},
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
        Significance threshold for p-value, by default 0.05
    colors : dict, optional
        Dictionary containing the colors to use in the plot, in matplotlib format;
        by default {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}

    Returns
    -------
    gtp : matplotlib.pyplot.figure.Figure
        matplotlib figure containing the Gaudi type plot.
    """
    poi = lmi_geometry.loc[lmi_geometry["moran_index"] == mlobject.poi, "bin_index"].iloc[0]

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], label="HH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[2], label="LH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[3], label="LL", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[4], label="HL", markersize=20),
    ]

    cmap = LinearSegmentedColormap.from_list("Custom cmap", [colors[nu] for nu in colors], len(colors))
    alpha = [1.0 if quad <= signipval else 0.3 for quad in lmi_geometry.LMI_pvalue]

    gtp, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="moran_quadrant", cmap=cmap, alpha=alpha, linewidth=2, edgecolor="white", ax=ax)
    plt.axis("off")

    sns.scatterplot(
        x=[lmi_geometry.X[poi]],
        y=[lmi_geometry.Y[poi]],
        s=50,
        ec="none",
        fc="lime",
        zorder=len(lmi_geometry),
    )

    ax.legend(handles=legend_elements, frameon=False, fontsize=20, loc="center left", bbox_to_anchor=(1, 0.5))

    return gtp


def get_lmi_scatterplot(
    mlobject: mlo.MetalociObject,
    lmi_geometry: pd.DataFrame,
    neighbourhood: float,
    signipval: float = 0.05,
    colors_lmi: dict = {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"},
):
    """
    Get a scatterplot of Z-scores of signal vs Z-score of signal spacial lag, given LMI values.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object.
    lmi_geometry : pd.DataFrame
        Merging from MetalociObject.lmi_info for a specific signal and MetalociObject.geometry for a specific region.
    neighbourhood : float
        buffer * BFACT. It determines the size of the neihbourhood of each point.
    signipval : float, optional
        Significance threshold for p-value, by default 0.05
    colors_lmi : _type_, optional
        Dictionary containing the colors to use in the plot, in matplotlib format;
        by default {1: "firebrick", 2: "lightskyblue", 3: "steelblue", 4: "orange"}

    Returns
    -------
    scatter : matplotlib.pyplot.figure.Figure
        matplotlib figure containing the LMI scatterplot.
    r_value_scat : float
        r-value of the linear regression.
    p-value
        p-value of the linear regression.
    """

    weights = lp.weights.DistanceBand.from_dataframe(lmi_geometry, neighbourhood)
    weights.transform = "r"

    y_lag = lp.weights.lag_spatial(weights, lmi_geometry["signal"])

    x = zscore(lmi_geometry.signal)
    y = zscore(y_lag)

    try:

        _, _, r_value_scat, p_value_scat, _ = linregress(x, y)
        scatter, ax = plt.subplots(figsize=(5, 5))  # , subplot_kw={'aspect':'equal'})

        alpha_sp = [1.0 if val < signipval else 0.1 for val in lmi_geometry.LMI_pvalue]
        colors_sp = [colors_lmi[val] for val in lmi_geometry.moran_quadrant]

        plt.scatter(x=x, y=y, s=100, ec="white", fc=colors_sp, alpha=alpha_sp)

        sns.scatterplot(
            x=[x[mlobject.poi]], y=[y[mlobject.poi]], s=150, ec="lime", fc="none", zorder=len(lmi_geometry)
        )
        sns.regplot(x=x, y=y, scatter=False, color="k")
        sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=False)

        plt.title(f"Moran Local Scatterplot\nr: {r_value_scat:4.2f}   p-value: {p_value_scat:.1e}")
        plt.axvline(x=0, color="k", linestyle=":")
        plt.axhline(y=0, color="k", linestyle=":")

        ax.set_xlabel("Z-score(Signal)")
        ax.set_ylabel("Z-score(Signal Spatial Lag)")

        r_value_scat = float(r_value_scat)
        p_value_scat = float(r_value_scat)
    
    except ValueError:

        print("\t\tCannot compute lineal regression as all values are identical.")

        return None, None, None

    return scatter, r_value_scat, p_value_scat


def signal_bed(
    mlobject: mlo.MetalociObject,
    lmi_geometry: pd.DataFrame,
    neighbourhood: float,
    quadrants: list = [1, 3],
    signipval: float = 0.05,
):
    """
    signal_bed _summary_

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        _description_
    lmi_geometry : pd.DataFrame
        _description_
    neighbourhood : float
        _description_
    quadrants : list, optional
        _description_, by default [1, 3]
    signipval : float, optional
        _description_, by default 0.05

    Returns
    -------
    _type_
        _description_
    """

    poi_distance = mlobject.kk_distances[mlobject.poi]

    # Select the polygons that are in the quadrant of interest and are significative.
    ml_indexes = lmi_geometry[
        (lmi_geometry.moran_quadrant.isin(quadrants)) & (lmi_geometry.LMI_pvalue <= signipval)
    ].bin_index.values.tolist()

    # Make a big polygon from the small poligons that are significant.
    metalocis = lmi_geometry[lmi_geometry.bin_index.isin(ml_indexes)].unary_union

    if metalocis and metalocis.geom_type == "Polygon":

        metalocis = MultiPolygon([metalocis])  # Need a multipolygon in order for the code to work.

    poi_point = Point(
        (lmi_geometry[lmi_geometry.bin_index == mlobject.poi].X, lmi_geometry[lmi_geometry.bin_index == mlobject.poi].Y)
    )

    metalocis_bed = []

    bed_data = defaultdict(list)

    try:

        for metaloci in metalocis.geoms:

            metaloci = gpd.GeoSeries(metaloci)

            if poi_point.within(metaloci[0]):

                for _, row_ml in lmi_geometry.iterrows():

                    adjacent_point = Point((row_ml.X, row_ml.Y))

                    if adjacent_point.within(metaloci[0]):

                        metalocis_bed.append(row_ml.bin_index)

                # Add close particles
                metalocis_bed.sort()
                close_bins = [i for i, distance in enumerate(poi_distance) if distance <= neighbourhood / 2]
                metalocis_bed = np.sort(list(set(close_bins + metalocis_bed)))

                for point in metalocis_bed:

                    bed_data["chr"].append(lmi_geometry.bin_chr[point])
                    bed_data["start"].append(lmi_geometry.bin_start[point])
                    bed_data["end"].append(lmi_geometry.bin_end[point])
                    bed_data["bin"].append(point)

    except:

        pass

    bed_data = pd.DataFrame(bed_data)

    return bed_data, metalocis_bed


def signal_plot(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame, metalocis, bins_sig, coords_sig):

    sig_plt = plt.figure(figsize=(10, 1.5))
    plt.tick_params(axis="both", which="minor", labelsize=24)

    g = sns.lineplot(x=lmi_geometry.bin_index, y=lmi_geometry.signal)

    for p in metalocis:

        plt.axvline(x=p, color="red", linestyle=":", lw=1, zorder=0, alpha=0.3)

    plt.axvline(x=mlobject.poi, color="lime", linestyle="--", lw=1.5)

    g.yaxis.set_major_locator(MaxNLocator(integer=True))
    g.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.xlabel(f"chromosome {mlobject.chrom}")
    plt.xticks(bins_sig, coords_sig)
    g.margins(x=0)
    g.axhline(y=0, color="k", zorder=0)

    sns.despine(top=True, right=True, left=False, bottom=True, offset=None, trim=False)

    return sig_plt, g


def place_composite(new_PI, ifile, ifactor, ixloc, iyloc):

    img = Image.open(ifile)
    niz = tuple([int(nu * ifactor) for nu in img.size])
    img = img.resize(niz, Image.Resampling.LANCZOS)
    new_PI.paste(img, (ixloc, iyloc))

    return new_PI
