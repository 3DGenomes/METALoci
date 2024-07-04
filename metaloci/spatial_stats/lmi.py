"""
Functions to compute Local Moran's I for a given signal in a METALoci object.
"""
import os
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path

import bioframe
import geopandas as gpd
import libpysal as lp
import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from libpysal.weights.spatial_lag import lag_spatial
from metaloci import mlo
from metaloci.misc import misc
from scipy.spatial import Voronoi
from scipy.stats import zscore
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import LineString, Point
from shapely.ops import polygonize

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def construct_voronoi(mlobject: mlo.MetalociObject, buffer: float) -> pd.DataFrame:
    """
    Takes a Kamada-Kawai layout in a METALoci object and calculates the geometry of each voronoi
    around each point of the Kamada-Kawai, in order the make a gaudi plot.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with Kamada-Kawai layout coordinates in it (MetalociObject.kk_coords).
    buffer : float
        Distance to buffer around the point to be painted in the gaudi plot.

    Returns
    -------
    df_geometry : pd.DataFrame
        Dataframe containing the information about the geometry of each point of the gaudi plot,
        for each bin. Needed for plotting the gaudi plot.
    """

    # The number two is a constant to define borders in the construction of the voronoi. Without those limits,
    # the voronoi could extend to infinity in the areas than do not adjoin with another point of the KK layout
    # (and we do not want that for our gaudi plot).
    points = mlobject.kk_coords.copy()
    points.extend(list(product([0, 2, -2], repeat=2))[1:])

    # Construct the voronoi polygon around the Kamada-Kawai points.
    vor = Voronoi(points)

    # Lines that construct the voronoi polygon.
    voronoid_lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

    # Iteratable of the sub-polygons composing the voronoi figure. This is done here in
    # order to construct a dictionary that relates the bin_order and the polygon_order,
    # as they tend to be different.
    poly_from_lines = list(polygonize(voronoid_lines))
    coord_id = coord_to_id(mlobject, poly_from_lines)

    # Constructing a GeoPandas DataFrame to work properly with the LMI function
    geometry_data = gpd.GeoDataFrame({"geometry": [poly for poly in poly_from_lines]})

    # Voronoi shaped & closed
    shape = LineString(mlobject.kk_coords).buffer(buffer)
    close = geometry_data.convex_hull.union(geometry_data.buffer(0.1, resolution=1)).geometry.unary_union

    for i, q in enumerate(geometry_data.geometry):

        geometry_data.geometry[i] = q.intersection(close)
        geometry_data.geometry[i] = shape.intersection(q)

    df_geometry = defaultdict(list)

    for i, x in sorted(coord_id.items(), key=lambda item: item[1]):

        df_geometry["bin_index"].append(x)
        df_geometry["moran_index"].append(i)
        df_geometry["X"].append(mlobject.kk_coords[x][0])
        df_geometry["Y"].append(mlobject.kk_coords[x][1])
        df_geometry["geometry"].append(geometry_data.loc[i, "geometry"])

    df_geometry = pd.DataFrame(df_geometry)

    return df_geometry


def coord_to_id(mlobject: mlo.MetalociObject, poly_from_lines: list) -> dict:
    """
    Correlates the bin index, determined by genomic positions, and the LMI index, which is
    determined by the LMI function. This is called in:

    construct_voronoi()

    and saved into columns of a dataframe, mostly for plotting purposes.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with Kamada-Kawai layout coordinates in it (MetalociObject.kk_coords).
    poly_from_lines : list
        List of polygon information for gaudi plots, calculated in:

        construct_voronoi().

    Returns
    -------
    coord_id_dict : dict
        Dictionary containing the correspondance between bin index and moran index, as they tend to be different.
    """

    coord_id_dict = {}

    for i, poly in enumerate(poly_from_lines):

        for j, coords in enumerate(mlobject.kk_coords):

            if poly.contains(Point(coords)):

                coord_id_dict[i] = j

                break

    return coord_id_dict


def load_region_signals(mlobject: mlo.MetalociObject, signal_data: dict, signal_file: Path) -> dict:
    """
    Does a subset of the signal file to contain only the signal corresponding to the region being processed.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with kk_coords in it.
    signal_data : dict
        Dictionary with signal for each chromosome. Each key is chrN, each value is a dataframe
        with its corresponding signal.
    signal_file : Path
        Path to text file with all the signal types to compute, one per line.

    Returns
    -------
    signals_dict : dict
        Dictionary containing signal information, but only for the region being processed.
    signal_types : list(str)
        List containing the signal types to compute for this region.
    """

    # Read signal file. Will only process the signals present in this list.
    if os.path.isfile(signal_file):

        with open(signal_file, "r", encoding="utf-8") as signals_handler:

            signal_types = [line.rstrip() for line in signals_handler]

    else:

        signal_types = [signal_file]

    if not all(signal in signal_data.columns for signal in signal_types):

        print("Trying to compute a signal that has not been previously processed in prep. Exiting.")

        return None

    region_signal = signal_data[
        (signal_data["start"] >= int(np.floor(mlobject.start / mlobject.resolution)) * mlobject.resolution) &
        (signal_data["end"] <= int(np.ceil(mlobject.end / mlobject.resolution)) * mlobject.resolution)
    ]

    if len(region_signal) != len(mlobject.kk_coords):

        tmp = len(mlobject.kk_coords) - len(region_signal)
        tmp = np.empty((tmp, len(region_signal.columns)))
        tmp[:] = np.nan  # jfm: leave NAs in for now. (in misc.signal_normalization() those NaNS will be substituted for
        # the median of the signal of the region)

        region_signal = pd.concat([region_signal, pd.DataFrame(tmp, columns=list(region_signal))], ignore_index=True)

    signals_dict = defaultdict(list)

    for signal_type in signal_types:

        try:

            signals_dict[signal_type] = misc.signal_normalization(region_signal[signal_type])

        except KeyError:

            return None

    return signals_dict


def compute_lmi(mlobject: mlo.MetalociObject, signal_type: str, neighbourhood: float, n_permutations: int = 9999,
                signipval: float = 0.05, silent: bool = False, poi_only=False) -> pd.DataFrame:
    """
    Computes Local Moran's Index for a signal type and outputs information of the LMI value and its p-value
    for each bin for a given signal, as well as some other information.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci with signals for that region (MetalociObject.signals_dict) and MetalociObject.lmi_geometry
        calculated with:

        lmi.construct_voronoi()

    signal_type : str
        Name of the type of the signal to use for the computation.
    neighbourhood : float
        Radius of the circle that determines the neighbourhood of each of the points in the Kamada-Kawai layout.
    n_permutations : int, optional
        Number of permutations to do in the randomization, by default 9999.
    signipval : float, optional
        Significancy threshold for p-value, by default 0.05.
    silent : bool, optional
        Controls the verbosity of the function (useful for multiprocessing), by default False.

    Returns
    -------
    pd.DataFrame
        Dataframe with ID, bin index, chromosome, start, end, value of signal, moran index, moran quadrant,
        LMI score, LMI p-value and LMI inverse of p-value for this signal.
    """

    signal_values = []
    filtered_signal_type = dict(filter(lambda item: signal_type == item[0], mlobject.signals_dict.items()))

    for _, row in mlobject.lmi_geometry.iterrows():

        signal_values.append(np.nanmedian([filtered_signal_type[key][row.bin_index] for key in filtered_signal_type]))

    signal_geometry = {"v": [], "geometry": []}

    for _, poly in mlobject.lmi_geometry.sort_values(by="moran_index").reset_index().iterrows():

        signal_geometry["v"].append(signal_values[poly.bin_index])
        signal_geometry["geometry"].append(poly.geometry)

    gpd_signal = gpd.GeoDataFrame(signal_geometry)  # Stored in Geopandas DataFrame to do LMI

    # Get weights for geometric distance
    y = gpd_signal["v"].values
    weights = lp.weights.DistanceBand.from_dataframe(gpd_signal, neighbourhood)
    weights.transform = "r"

    # Calculate Local Moran's I
    moran_local_object = Moran_Local(
        y, weights, permutations=n_permutations, n_jobs=1
    )  # geoda_quadsbool (default=False) If False use PySAL Scheme: HH=1, LH=2, LL=3, HL=4

    ZSig = zscore(y)
    ZLag = zscore(lag_spatial(moran_local_object.w, moran_local_object.z))

    if not silent:

        print(
            f"\tThere is a total of {len(moran_local_object.p_sim[(moran_local_object.p_sim < signipval)])} "
            f"significant points in Local Moran's I for signal {signal_type}"
        )

    df_lmi = defaultdict(list)

    for i, row in mlobject.lmi_geometry.iterrows():

        # if poi_only is True, only the points of interest will be saved
        if poi_only and mlobject.poi != row.bin_index:

            continue

        bin_start = int(mlobject.start) + (mlobject.resolution * row.bin_index) + row.bin_index
        bin_end = bin_start + mlobject.resolution

        df_lmi["ID"].append(signal_type)
        df_lmi["bin_index"].append(row.bin_index)
        df_lmi["bin_chr"].append(mlobject.chrom[3:])
        df_lmi["bin_start"].append(bin_start)
        df_lmi["bin_end"].append(bin_end)
        df_lmi["signal"].append(signal_values[row.bin_index])
        df_lmi["moran_index"].append(row.moran_index)
        df_lmi["moran_quadrant"].append(moran_local_object.q[row.moran_index])
        df_lmi["LMI_score"].append(round(moran_local_object.Is[row.moran_index], 9))
        df_lmi["LMI_pvalue"].append(round(moran_local_object.p_sim[row.moran_index], 9))
        # df_lmi["ZSig"].append(y[row.moran_index])
        # df_lmi["ZLag"].append(lags[row.moran_index])
        df_lmi["ZSig"].append(ZSig[row.moran_index])
        df_lmi["ZLag"].append(ZLag[row.moran_index])

    df_lmi = pd.DataFrame(df_lmi)
    # Changing the data types to the proper ones so the pickle file has a smaller size.
    df_lmi["ID"] = df_lmi["ID"].astype(str)
    df_lmi["bin_index"] = df_lmi["bin_index"].astype(np.ushort)
    df_lmi["bin_chr"] = df_lmi["bin_chr"].astype(str)
    df_lmi["bin_start"] = df_lmi["bin_start"].astype(np.uintc)
    df_lmi["bin_end"] = df_lmi["bin_end"].astype(np.uintc)
    df_lmi["signal"] = df_lmi["signal"].astype(np.single)
    df_lmi["moran_index"] = df_lmi["moran_index"].astype(np.ushort)
    df_lmi["moran_quadrant"] = df_lmi["moran_quadrant"].astype(np.ubyte)
    df_lmi["LMI_score"] = df_lmi["LMI_score"].astype(np.half)
    df_lmi["LMI_pvalue"] = df_lmi["LMI_pvalue"].astype(np.half)
    df_lmi["ZSig"] = df_lmi["ZSig"].astype(np.half)
    df_lmi["ZLag"] = df_lmi["ZLag"].astype(np.half)

    return df_lmi


def aggregate_signals(mlobject: mlo.MetalociObject):
    """
    Aggregates the signals in the signals_dict of the METALoci object.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object that contains the needed information
    """

    for condition, signal_type in mlobject.agg.items():

        if not isinstance(signal_type, list):

            signal_type = [signal_type]

        mlobject.signals_dict[condition] = np.nanmedian(
            np.array([mlobject.signals_dict[signal] for signal in signal_type]), axis=0
        ) 


def get_bed(mlobject: mlo.MetalociObject, lmi_geometry: pd.DataFrame, neighbourhood: float,
            quadrants: list = None, signipval: float = 0.05, poi: int = None, silent: bool = True) -> pd.DataFrame:
    """
    Get the bins that are significant in the Local Moran's I for a given point of interest.

    Parameters
    ----------
    mlobject : mlo.MetalociObject
        METALoci object with the needed information.
    lmi_geometry : pd.DataFrame
        DataFrame with the geometry of the bins.
    neighbourhood : float
        Radius of the circle that determines the neighbourhood of each of the points in the Kamada-Kawai layout.
    bfact : float
        Factor to multiply the neighbourhood by.
    quadrants : list, optional
        List of quadrants to consider, by default None.
    signipval : float, optional
        Significancy threshold for p-value, by default 0.05.
    poi : int, optional
        Point of interest, by default None.
    silent : bool, optional
        Controls the verbosity of the function (useful for multiprocessing), by default True.
    Returns
    -------

    bed : pd.DataFrame
        BED file with the bins that are significant in the Local Moran's I.
    """

    if poi is None:

        poi = mlobject.poi

    if quadrants is None:

        quadrants = [1, 3]

    signal = lmi_geometry.ID[0]  # Extract signal name from lmi_geometry
    data = mlobject.lmi_info[signal]
    # Check tha the point of interest is significant in the given quadrant
    significants = len(data[(data.bin_index == poi) &
                            (data.moran_quadrant.isin(quadrants)) &
                            (data.LMI_pvalue <= signipval)])

    if significants == 0:

        if not silent:

            print(
                f"\t-> Point of interest {poi} is not significant for quadrant(s) {quadrants} (p-value > {signipval})")

        return None

    # Get bins within a distance to point and plot them in a gaudi plot
    indices = np.nonzero(mlobject.kk_distances[mlobject.poi] < neighbourhood)[0]

    neighbouring_bins = lmi_geometry.iloc[indices]
    neighbouring_bins = neighbouring_bins[['bin_chr', 'bin_start', 'bin_end']]
    neighbouring_bins.columns = ['chrom', 'start', 'end']

    # Merge overlapping intervals and create a continuous BED file
    return pd.DataFrame(bioframe.merge(neighbouring_bins, min_dist=2))
