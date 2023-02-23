from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, distance
from shapely.geometry import LineString, Point
from shapely.ops import polygonize


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

    return df_geometry, coord_id, geometry_data
