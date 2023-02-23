from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, distance
from shapely.geometry import LineString, Point
from shapely.ops import polygonize
from metaloci.misc import misc
import re
import glob
import os
import libpysal as lp
from esda.moran import Moran_Local
import time


def coord_to_id(mlobject, poly_from_lines):

    coord_to_id = {}

    for i, poly in enumerate(poly_from_lines):

        for j, coords in enumerate(mlobject.kk_coords):

            if poly.contains(Point(coords)):

                coord_to_id[i] = j

                break

    return coord_to_id


def construct_voronoi(mlobject, LIMITS, buffer):

    mlobject.kk_coords = list(mlobject.kk_nodes.values())
    mlobject.kk_distances = distance.cdist(mlobject.kk_coords, mlobject.kk_coords, "euclidean")

    points = mlobject.kk_coords.copy()

    points.append(np.array([-LIMITS, LIMITS]))
    points.append(np.array([LIMITS, LIMITS]))
    points.append(np.array([LIMITS, -LIMITS]))
    points.append(np.array([-LIMITS, -LIMITS]))
    points.append(np.array([0, -LIMITS]))
    points.append(np.array([LIMITS, 0]))
    points.append(np.array([0, LIMITS]))
    points.append(np.array([-LIMITS, 0]))

    # Construct the voronoi polygon around the Kamada-Kawai points.
    vor = Voronoi(points)

    # Lines that construct the voronoi polygon.
    voronoid_lines = [
        LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line
    ]

    # Iteratable of the sub-polygons composing the voronoi figure. This is done here in
    # order to construct a dictionary that relates the bin_order and the polygon_order,
    # as they tend to be different.
    poly_from_lines = list(polygonize(voronoid_lines))

    coord_id = coord_to_id(mlobject, poly_from_lines)

    # Constructing a GeoPandas DataFrame to work properly with the LMI function
    geometry_data = gpd.GeoDataFrame({"geometry": [poly for poly in poly_from_lines]})

    # Voronoi shaped & closed
    shape = LineString(mlobject.kk_coords).buffer(buffer)
    close = geometry_data.convex_hull.union(
        geometry_data.buffer(0.1, resolution=1)
    ).geometry.unary_union

    for i, q in enumerate(geometry_data.geometry):

        geometry_data.geometry[i] = q.intersection(close)
        geometry_data.geometry[i] = shape.intersection(q)

    df_geometry = defaultdict(list)

    for i, x in sorted(coord_id.items(), key=lambda item: item[1]):

        df_geometry["bin_index"].append(x)
        df_geometry["moran_index"].append(i)
        df_geometry["X"].append(mlobject.kk_coords[i][0])
        df_geometry["Y"].append(mlobject.kk_coords[i][1])
        df_geometry["geometry"].append(geometry_data.loc[i, "geometry"])

    df_geometry = pd.DataFrame(df_geometry)

    return df_geometry


def load_signals(df_regions, work_dir):

    chrom_to_do = list(
        dict.fromkeys(
            re.compile("chr[0-9]*").findall(
                "\n".join([x for y in df_regions["coords"] for x in y.split(":")])
            )
        )
    )

    signal_data = {}

    for chrom in chrom_to_do:

        signal_data[chrom] = pd.read_pickle(
            glob.glob(f"{os.path.join(work_dir, 'signal', chrom)}/*_signal.pkl")[0]
        )

    return signal_data


def load_region_signals(mlobject, signal_data, signal_file):

    # Read signal file. Will only process the signals present in this list.
    with open(signal_file) as signals_handler:

        signal_types = [line.rstrip() for line in signals_handler]

    region_signal = signal_data[mlobject.chrom][
        (signal_data[mlobject.chrom]["start"] >= int(mlobject.start))
        & (signal_data[mlobject.chrom]["end"] <= int(mlobject.end))
    ]

    if len(region_signal) != len(mlobject.kk_coords):

        tmp = len(mlobject.kk_coords) - len(region_signal)
        tmp = np.empty((tmp, len(region_signal.columns)))
        tmp[:] = np.nan

        region_signal = region_signal.append(
            pd.DataFrame(tmp, columns=list(region_signal)), ignore_index=True
        )

    signals_dict = defaultdict(list)

    for signal_type in signal_types:

        signal_values = region_signal[signal_type]
        signals_dict[signal_type] = misc.signal_normalization(signal_values, 0.01, "01")

    return signals_dict, signal_types


def compute_lmi(
    mlobject,
    signal_type,
    neighbourhood,
    n_permutations=9999,
    signipval=0.05,
):

    signal = []

    res = dict(filter(lambda item: signal_type in item[0], mlobject.signals_dict.items()))

    for _, row in mlobject.lmi_geometry.iterrows():

        signal.append(np.nanmedian([res[key][row.bin_index] for key in res]))

    signal_geometry = {"v": [], "geometry": []}

    df_tmp = mlobject.lmi_geometry.sort_values(by="moran_index").reset_index()

    for _, poly in df_tmp.iterrows():

        signal_geometry["v"].append(signal[poly.bin_index])
        signal_geometry["geometry"].append(poly.geometry)

    gpd_signal = gpd.GeoDataFrame(signal_geometry)  # Stored in Geopandas DataFrame to do LMI

    # Get weights for geometric distance
    # print("\tGetting weights and geometric distance for LM")
    y = gpd_signal["v"].values
    weights = lp.weights.DistanceBand.from_dataframe(gpd_signal, neighbourhood)
    weights.transform = "r"

    # Calculate Local Moran's I
    moran_local = Moran_Local(y, weights, permutations=n_permutations)
    print(
        f"\tThere are a total of {len(moran_local.p_sim[(moran_local.p_sim < signipval)])} "
        f"significant points in Local Moran's I for signal {signal_type}"
    )
    chrom_number = mlobject.chrom[3:]

    df_lmi = defaultdict(list)

    for _, row in mlobject.lmi_geometry.iterrows():

        bin_start = int(mlobject.start) + (mlobject.resolution * row.bin_index) + row.bin_index
        bin_end = bin_start + mlobject.resolution

        df_lmi["Class"].append(signal_type)
        df_lmi["ID"].append(signal_type)

        df_lmi["bin_index"].append(row.bin_index)
        df_lmi["bin_chr"].append(chrom_number)
        df_lmi["bin_start"].append(bin_start)
        df_lmi["bin_end"].append(bin_end)

        df_lmi["signal"].append(signal[row.bin_index])

        df_lmi["moran_index"].append(row.moran_index)
        df_lmi["moran_quadrant"].append(moran_local.q[row.moran_index])
        df_lmi["LMI_score"].append(round(moran_local.Is[row.moran_index], 9))
        df_lmi["LMI_pvalue"].append(round(moran_local.p_sim[row.moran_index], 9))
        df_lmi["LMI_inv_pval"].append(round((1 - moran_local.p_sim[row.moran_index]), 9))

    df_lmi = pd.DataFrame(df_lmi)

    # Changing the data types to the proper ones so the pickle file has a smaller size.
    df_lmi["Class"] = df_lmi["Class"].astype(str)
    df_lmi["ID"] = df_lmi["ID"].astype(str)

    df_lmi["bin_index"] = df_lmi["bin_index"].astype(np.uintc)
    df_lmi["bin_chr"] = df_lmi["bin_chr"].astype(str)
    df_lmi["bin_start"] = df_lmi["bin_start"].astype(np.uintc)
    df_lmi["bin_end"] = df_lmi["bin_end"].astype(np.uintc)

    df_lmi["signal"] = df_lmi["signal"].astype(np.half)

    df_lmi["moran_index"] = df_lmi["moran_index"].astype(np.uintc)
    df_lmi["moran_quadrant"] = df_lmi["moran_quadrant"].astype(np.uintc)
    df_lmi["LMI_score"] = df_lmi["LMI_score"].astype(np.half)
    df_lmi["LMI_pvalue"] = df_lmi["LMI_pvalue"].astype(np.half)
    df_lmi["LMI_inv_pval"] = df_lmi["LMI_inv_pval"].astype(np.half)
    # zSig y Zlag

    return df_lmi
