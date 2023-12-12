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
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.ndimage import rotate
from scipy.stats import linregress, zscore
from shapely.geometry import Point
import bioframe

from metaloci import mlo
from metaloci.misc import misc
from metaloci.plot import plot
# print first 500 rows pandas
pd.set_option('display.max_rows', 500)


def get_kk_plot(mlobject: mlo.MetalociObject, restraints: bool = True, neighbourhood: float = None):
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
        Kamada-Kawai layout plot object
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
    g.annotate(f"       {mlobject.chrom}:{mlobject.start}", (xs[0], ys[0]), size=9)
    g.annotate(f"       {mlobject.chrom}:{mlobject.end}", (xs[len(xs) - 1], ys[len(ys) - 1]), size=9)

    poi_x, poi_y = xs[mlobject.poi], ys[mlobject.poi]

    if neighbourhood:

        circle = plt.Circle((poi_x, poi_y), neighbourhood, color="red", fill=False, linestyle=":", alpha=0.5, lw=1, zorder=3)
        plt.gca().add_patch(circle)

    sns.scatterplot(
        x=[poi_x], y=[poi_y], s=POINTSIZE * 1.5, ec="lime", fc="none", zorder=4
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

    Notes
    -----
    The mixed matrices plot is a plot of the Hi-C matrix for the region. Only the upper triangle of the array is
    represented, rotated 45ª counter-clock wise. The top triangle is the original matrix, and the lower triange is the
    subsetted matrix.

    """

    if mlobject.subset_matrix is None:

        mlobject.subset_matrix = misc.get_subset_matrix(mlobject)

    upper_triangle = np.triu(mlobject.matrix.copy(), k=1)  # Original matrix
    lower_triangle = np.tril(mlobject.subset_matrix, k=-1)  # Top interactions matrix
    mlobject.mixed_matrices = upper_triangle + lower_triangle

    if mlobject.flat_matrix is None:

        print("Flat matrix not found in metaloci object. Run get_subset_matrix() first.")
        return None

    fig_matrix, (ax1, _) = plt.subplots(1, 2, figsize=(20, 5))

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
        label="$log_{10}$ (Hi-C interactions)",
        format=FormatStrFormatter("%.2f"),
        ax=ax,
    ).set_label("$log_{10}$ (Hi-C interactions)", rotation=270, size=12, labelpad=20)

    plt.title(f"{mlobject.region}")
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

    Notes
    -----
    The Gaudi signal plot is a plot of the LMI values for each bin in the region, where the color of the bin represents
    the signal value. The point of interest is represented by a green point.
    """

    cmap = "PuOr_r"
    min_value = lmi_geometry.signal.min()
    max_value = lmi_geometry.signal.max()

    gaudi_signal_fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="signal", cmap=cmap, linewidth=2, edgecolor="white", ax=ax)

    sns.scatterplot(x=[lmi_geometry.X[mlobject.poi]], y=[lmi_geometry.Y[mlobject.poi]], s=50, ec="none", fc="lime")
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
    cbar.set_label(f"{lmi_geometry.ID[0]}", rotation=270, size=20, labelpad=35)
    plt.axis("off")

    return gaudi_signal_fig


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
        matplotlib figure containing the Gaudi type plot.ç
    
    Notes
    -----
    The Gaudi type plot is a plot of the LMI values for each bin in the region, where the color of the bin represents
    the quadrant to which the bin belongs. The significance of the LMI value is represented by the transparency of the
    color, where the more significant the LMI value, the more opaque the color. The point of interest is represented
    by a green point.
    """

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], label="HH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[2], label="LH", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[3], label="LL", markersize=20),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[4], label="HL", markersize=20),
    ]

    cmap = LinearSegmentedColormap.from_list("Custom cmap", [colors[nu] for nu in colors], len(colors))
    alpha = [1.0 if pval <= signipval else 0.3 for pval in lmi_geometry.LMI_pvalue]

    gaudi_type_fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    lmi_geometry.plot(column="moran_quadrant", cmap=cmap, alpha=alpha, linewidth=2, edgecolor="white", ax=ax)
    plt.axis("off")

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

    _, _, r_value_scat, p_value_scat, _ = linregress(x, y)
    scatter_fig, ax = plt.subplots(figsize=(5, 5))  # , subplot_kw={'aspect':'equal'})

    alpha_sp = [1.0 if val < signipval else 0.1 for val in lmi_geometry.LMI_pvalue]
    colors_sp = [colors_lmi[val] for val in lmi_geometry.moran_quadrant]

    plt.scatter(x=x, y=y, s=100, ec="white", fc=colors_sp, alpha=alpha_sp)

    sns.scatterplot(
        x=[x[mlobject.poi]], y=[y[mlobject.poi]], s=150, ec="lime", fc="none", zorder=len(lmi_geometry)
    )
    sns.regplot(x=x, y=y, scatter=False, color="k")
    sns.despine(top=True, right=True, left=False, bottom=False, offset=10, trim=False)

    plt.title(f"Moran Local Scatterplot\n[r: {r_value_scat:4.2f} | p-value: {p_value_scat:.1e}]", fontsize=11)
    plt.axvline(x=0, color="k", linestyle=":")
    plt.axhline(y=0, color="k", linestyle=":")

    ax.set_xlabel(f"Z-score ({lmi_geometry.ID[0]})")
    ax.set_ylabel(f"Z-score ({lmi_geometry.ID[0]} Spatial Lag)")

    r_value_scat = float(r_value_scat)
    p_value_scat = float(r_value_scat)
    
    return scatter_fig, r_value_scat, p_value_scat


def get_highlight(
    mlobject: mlo.MetalociObject,
    lmi_geometry: pd.DataFrame,
    influence: float,
    bfact: float,
    quadrants: list = [1, 3],
    signipval: float = 0.05,
    metaloci_only: bool = False,
):
    
    """
    Generate a dictionary containing metaloci information and their bin indices based on given parameters.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        A METALoci object.
    lmi_geometry : pd.DataFrame
        A DataFrame containing LMI information and geometry of all bins in the region.
    neighbourhood : float
        The distance threshold to identify adjacent bins within a neighborhood.
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

    neighbourhood = mlobject.kk_distances.diagonal(1).mean() * influence * bfact
    bins_to_highlight = defaultdict(list)

    if metaloci_only == False:

        for _, row_ml in lmi_geometry.iterrows():

            if row_ml.LMI_pvalue <= signipval and row_ml.moran_quadrant in quadrants:

                bins_to_highlight[row_ml.moran_quadrant].append(row_ml.bin_index)

    else:

        poi_row = lmi_geometry[lmi_geometry.bin_index == mlobject.poi].squeeze()

        if poi_row.LMI_pvalue <= signipval and poi_row.moran_quadrant in quadrants:

            poi_point = Point(
            (lmi_geometry[lmi_geometry.bin_index == mlobject.poi].X, lmi_geometry[lmi_geometry.bin_index == mlobject.poi].Y)
            )
            poi_point = Point(lmi_geometry.loc[lmi_geometry["bin_index"] == mlobject.poi, ["X", "Y"]].iloc[0])

            for _, row_ml in lmi_geometry.iterrows():

                adjacent_point = Point((row_ml.X, row_ml.Y))

                if adjacent_point.distance(poi_point) <= neighbourhood:

                    bins_to_highlight[row_ml.moran_quadrant].append(row_ml.bin_index)

                bins_to_highlight[row_ml.moran_quadrant].sort()

    return bins_to_highlight


def get_bed(mlo, lmi_geometry, influence, bfact, signipval = 0.05, quadrants = [1, 3], poi=None, plotit=False):
    
    if poi == None:
        
        poi = mlo.poi
    
    signal = lmi_geometry.ID[0]  # Extract signal name from lmi_geometry
    data = mlo.lmi_info[signal]
    
    # Check tha the point of interest is significant in the given quadrant
    significants = len(data[(data.bin_index == poi) & (data.moran_quadrant.isin(quadrants)) & (data.LMI_pvalue <= signipval)])

    if significants == 0:
        
        print(f"\t\tPoint of interest {poi} is not significant for quadrant(s) {quadrants} (p-value > {signipval})")
        return None
           
    # Get bins within a distance to point and plot them in a gaudi plot          
    buffer = mlo.kk_distances.diagonal(1).mean() * influence
    bbfact = buffer * bfact
    indices = np.nonzero(mlo.kk_distances[mlo.poi] < bbfact)[0]
    
    if plotit:
        
        print(f"\tGetting data for point of interest: {poi}...")    

        g = plot.get_gaudi_type_plot(mlo,lmi_geometry)
        x = lmi_geometry.X[lmi_geometry.bin_index==poi]
        y = lmi_geometry.Y[lmi_geometry.bin_index==poi]
        
        sns.scatterplot(x=lmi_geometry.X, y=lmi_geometry.Y, color='lime', s=10)
        sns.scatterplot(x=x, y=y, color='yellow', s=100)
        circle = Circle((x, y), bbfact, color='yellow', fill=False)
        
        plt.gcf().gca().add_artist(circle)
        
        xs = lmi_geometry.X.iloc[indices]
        ys = lmi_geometry.Y.iloc[indices]
        
        sns.scatterplot(x=xs, y=ys, color='yellow')
        
        plt.show()
    
    neighbouring_bins = lmi_geometry.iloc[indices]
    neighbouring_bins = neighbouring_bins[['bin_chr','bin_start','bin_end']]
    neighbouring_bins.columns = 'chrom start end'.split()
    bed = pd.DataFrame(bioframe.merge(neighbouring_bins, min_dist=2))  # Merge overlapping intervals and create a continuous BED file

    return bed


def signal_plot(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame, influence: float, bfact : float, quadrants: list = [1, 3], signipval: float = 0.05, metaloci_only: bool = False):

    """
    Generate a signal plot to visualize the signal intensity and the positions of significant bins.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        A METALoci object.
    lmi_geometry : pd.DataFrame
        A DataFrame containing LMI information and geometry of all bins in the region.
    neighbourhood : float
        The distance threshold to identify adjacent bins within a neighborhood.
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
    dict
        A dictionary containing bin indices of bins that meet the selection criteria, grouped by quadrants.


    Notes
    -----
    The function generates a line plot of the signal intensity in 'lmi_geometry' with respect to the bin indices.
    It highlights the bins in the 'metalocis' list with red vertical lines if 'metaloci_only' is True,
    otherwise, it uses different colors for different quadrants.

    Returns matplotlib.axes._subplots.AxesSubplot as it is used to create the composite plot.
    """
    bins, coords_b = get_bins_coords(mlobject)    
    metalocis = get_highlight(mlobject, lmi_geometry, influence, bfact, quadrants, signipval, metaloci_only)
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
    ax.yaxis.set_major_locator(MaxNLocator(nbins = 5, integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.margins(x=0)
    ax.axhline(y=0, color="k", zorder=0)

    sns.despine(top=True, right=True, left=False, bottom=True, offset=None, trim=False)

    return sig_plt, ax


def place_composite(new_PI, ifile, ifactor, ixloc, iyloc):
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
    img = Image.open(ifile)
    niz = tuple([int(nu * ifactor) for nu in img.size])
    img = img.resize(niz, Image.Resampling.LANCZOS)
    new_PI.paste(img, (ixloc, iyloc))

    return new_PI


def get_bins_coords(mlobject):

    """
    Get bin indices and corresponding coordinates for a given METALoci object.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        The METALoci object containing LMI geometry information.

    Returns
    -------
    list
        A list of bin indices, representing the start position of each bin.
    list
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

        elif i == 4:

            bins.append(len(mlobject.lmi_geometry) - 1)

        else:

            bins.append(int(((i - 1) / 2) * len(mlobject.lmi_geometry)) - 1)

        coords_b.append(f"{mlobject.start + bins[i - 1] * mlobject.resolution:,}")

    return bins, coords_b


def get_color_alpha(quadrant):
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
