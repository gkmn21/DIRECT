import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, MultiPolygon, LineString, Polygon
from shapely.ops import split
import numpy as np
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd
import pygeohash as pgh

from enum import Enum
from haversine import haversine, Unit

from tqdm import tqdm

import folium
from shapely import wkt
from math import isnan
import time
import os
import ast

from collections import Counter

SEED = 100
np.random.seed(SEED)
GEOHASH_PRECISION = 7

def read_data(IMPL_DIRECTORY):
    '''
    Read data files
    '''
    print('read_data()')

    final_pois = pd.read_csv(os.path.join(IMPL_DIRECTORY,'saved_data/final_pois.csv'))
    print(f'final_pois.shape {final_pois.shape}')
    final_pois = preprocess_final_pois(final_pois)

    Bonn_walking_network = ox.io.load_graphml(os.path.join(IMPL_DIRECTORY,'saved_data/walking_network.graphml')) #'content/saved_data/Bonn_walking_network.graphml'
    print(f'Bonn_walking_network {Bonn_walking_network}')

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/start_node_pois_within_radius.pkl'), 'rb') as f:
        start_node_pois_within_radius = pickle.load(f)
    print(f'len(start_node_pois_within_radius) {len(start_node_pois_within_radius)}')

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/indexing_dicts.pkl'), 'rb') as f:
        indexing_dicts = pickle.load(f)
    print(f'indexing_dicts {len(indexing_dicts)}')
    print(f"idx2poiid {len(indexing_dicts['idx2poiid'])}")
    print(f"poiid2idx {len(indexing_dicts['poiid2idx'])}")
    idx2poiid = indexing_dicts['idx2poiid']
    poiid2idx = indexing_dicts['poiid2idx']

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/distance_matrix.npy'), 'rb') as f:
        unfiltered_distance_matrix = np.load(f)
    print(f'unfiltered_distance_matrix {unfiltered_distance_matrix.shape}')

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/path_related_dicts.pkl'), 'rb') as f:
        path_related_dicts = pickle.load(f)
    print(f'path_related_dicts {len(path_related_dicts)}')
    print(f"paths {len(path_related_dicts['paths'])}")
    paths = path_related_dicts['paths']
    if path_related_dicts.get('paths_with_intermediate_pois', None):
        print(f"paths_with_intermediate_pois {len(path_related_dicts['paths_with_intermediate_pois'])}")
        paths_with_intermediate_pois = path_related_dicts['paths_with_intermediate_pois']
    

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/node_category_attr_dict.pkl'), 'rb') as f:
        node_category_attr_dict = pickle.load(f)
    print(f'len(node_category_attr_dict) {len(node_category_attr_dict)}')

    POI_graph = nx.read_graphml(os.path.join(IMPL_DIRECTORY,'saved_data/POI_graph_updated.graphml'), node_type = int)
    print(f'POI_graph {POI_graph}')

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/distance_matrix_updated.npy'), 'rb') as f:
        distance_matrix = np.load(f)
    print(f'distance_matrix {distance_matrix.shape}')

    with open(os.path.join(IMPL_DIRECTORY, 'saved_data/bearing_matrix.npy'), 'rb') as f:
        bearing_matrix = np.load(f)
    print(f'bearing_matrix {bearing_matrix.shape}')



    return (
        final_pois, Bonn_walking_network, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    )
    


def preprocess_final_pois(final_pois):

    print('preprocess_final_pois()')

    final_pois['tourism_category'] = final_pois['tourism_category'].apply(
        lambda x: [int(i) for i in x.strip('[].').split('. ')]
    )

    final_pois["geometry"] = gpd.GeoSeries.from_wkt(final_pois["geometry"])
    final_pois_gdf = gpd.GeoDataFrame(final_pois, geometry = "geometry")

    # (lat, lon)
    final_pois_gdf['plotting_coords'] = final_pois_gdf['geometry'].apply(lambda x: (x.y, x.x) if isinstance(x, Point) else (x.centroid.y, x.centroid.x))
    
    # compute geohashes
    final_pois_gdf['geohash'] = final_pois_gdf['plotting_coords'].apply(
        lambda x: pgh.encode(x[0], x[1], precision = GEOHASH_PRECISION)
    )

    print(f'final_pois_gdf.shape {final_pois_gdf.shape}')

    return final_pois_gdf



def rescale_inputs(
        POI_graph,
        final_pois,
        poiid2idx,
        distance_matrix,
        unfiltered_distance_matrix,
        start_node_pois_within_radius,
        train_set_start_node_osmids = []
    ):

    print('rescale_inputs()')

    ##
    # Create new POI graph with ids from 1
    ##
    _POI_graph = nx.relabel_nodes(POI_graph, poiid2idx)
    print(f'{_POI_graph}')
    print(f'_POI_graph.nodes[0] {_POI_graph.nodes[0]}')

    # rescale min_visit_time and max_visit_time to hours
    for node in _POI_graph.nodes:
        _POI_graph.nodes[node]['min_visit_time'] /= 3600
        _POI_graph.nodes[node]['max_visit_time'] /= 3600
    print(f'_POI_graph.nodes[0], _POI_graph.nodes[50] {_POI_graph.nodes[0], _POI_graph.nodes[50]}')

    # inserting _osm_id column in final_pois in prepare_data.py script
    # final_pois['_osm_id'] = final_pois['osm_id'].apply(lambda x: poiid2idx[x])

    # rescale distance matrix to km
    distance_matrix = distance_matrix/1000
    unfiltered_distance_matrix = unfiltered_distance_matrix/1000

    _start_node_pois_within_radius = [poiid2idx[s] for s in start_node_pois_within_radius]
    print(f'len(_start_node_pois_within_radius) {len(_start_node_pois_within_radius)}')

    _train_set_start_node_osmids = None
    if len(train_set_start_node_osmids) > 0:
        _train_set_start_node_osmids = [poiid2idx[s] for s in train_set_start_node_osmids]
        print(f'len(_train_set_start_node_osmids) {len(_train_set_start_node_osmids)}')

    return (
        _POI_graph,
        final_pois,
        distance_matrix,
        unfiltered_distance_matrix,
        _start_node_pois_within_radius,
        _train_set_start_node_osmids
    )

def normalise_inputs(distance_matrix, unfiltered_distance_matrix = None):

    # ignoring np.inf for normalization
    if distance_matrix is not None:
        matrix_with_nan = np.where(np.isinf(distance_matrix), np.nan, distance_matrix)
        _distance_matrix = (matrix_with_nan - np.nanmin(matrix_with_nan)) / (np.nanmax(matrix_with_nan) - np.nanmin(matrix_with_nan))

    # matrix_with_nan = np.where(np.isinf(unfiltered_distance_matrix), np.nan, unfiltered_distance_matrix)
    # _unfiltered_distance_matrix = (matrix_with_nan - np.nanmin(matrix_with_nan)) / (np.nanmax(matrix_with_nan) - np.nanmin(matrix_with_nan))

    # return _distance_matrix, _unfiltered_distance_matrix

    return _distance_matrix

##
# Evaluation related utilities
##
def check_positions(vectors, second_vector):
    # Combine the vectors by performing a logical OR on each position
    combined_vector = np.max(vectors, axis=0)
    
    for i in range(len(second_vector)):
        if second_vector[i] == 1 and combined_vector[i] != 1:
            return False
    return True






    

    





