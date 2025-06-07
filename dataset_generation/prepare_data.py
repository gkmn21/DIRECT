#%%
import pandas as pd
import numpy as np
import random

import osmnx as ox
import networkx as nx
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.model_selection import train_test_split
from shapely.geometry import Point

from enum import Enum
from haversine import haversine, Unit

from tqdm import tqdm

import folium
import requests
from shapely import wkt
from math import isnan
import time
import os
import copy

from collections import Counter

from constants import (
    TOURIST_ATTR_TAGS, ACCOMMODATION_TAGS, FOOD_ESTB_TAGS,
    NODE_CATEGORIES, VISIT_DURATION_BASED_ON_CATEGORIES,
    TIME_CONSTRAINTS
)


#%%
def prepare_poi_data(
        data_path = None,
        location_dict = None,
        osm_id = None,
        city_lat_lon = None,
        max_pois = 250,
        max_start_nodes = 200
    ):
    '''
    Prepare POI data for the given location
    data_path: path to the POI data file (if poi data is available in a file)
    location_dict: dictionary with city, state, country of location
    osm_id: OSM ID of the location
    city_lat_lon: latitude and longitude of the city center (i.e location)
    '''

    ##
    # Read POI data from file (if provided)
    ##
    if data_path:
        
        # read poi data from data_path
        poi_data = pd.read_csv(data_path)
        print(f'poi_data.shape {poi_data.shape}')
        pois_gdf = gpd.GeoDataFrame(
            poi_data, geometry = gpd.points_from_xy(poi_data.Longitude, poi_data.Latitude), crs = 'EPSG:4326'
        )

        pois_gdf['tourism_category'] = pois_gdf.apply(assign_categories_to_poi, axis=1)

        # insert dummy osm_id
        pois_gdf['osm_id'] = pois_gdf.index

        # extract accommodation pois from OSM
        accommodation_pois_gdf = extract_pois_using_tags(
            ACCOMMODATION_TAGS,
            location_dict,
            osm_id
        )
    
    ##
    #  Extract POI data from OSM
    ##
    else:
        pois_gdf = extract_pois_using_tags(
            TOURIST_ATTR_TAGS,
            location_dict
        )

        accommodation_pois_gdf = extract_pois_using_tags(
            ACCOMMODATION_TAGS,
            location_dict
        )

    # remove point pois in polygon pois
    pois_gdf = remove_point_pois_in_polygon(
        pois_gdf
    )
    accommodation_pois_gdf = remove_point_pois_in_polygon(
        accommodation_pois_gdf
    )
    print(f'A/f removing points in polygon, pois_gdf.shape {pois_gdf.shape}')
    print(f'A/f removing points in polygon, accommodation_pois_gdf.shape {accommodation_pois_gdf.shape}')

    # reduce accommodation pois to only start node POIs within 10km radius of city center and within max_start_nodes 
    start_node_poi_ids, accommodation_pois_gdf = get_start_node_poi_ids(accommodation_pois_gdf, city_lat_lon, max_start_nodes)


    # get importance scores of pois and accommodations
    importance_scores = get_importance_scores_of_pois(pois_gdf)
    if len(importance_scores) == pois_gdf.shape[0]:
        pois_gdf['importance_score'] = importance_scores
        print(f'len(importance_scores) {len(importance_scores)}')
    importance_scores = get_importance_scores_of_pois(accommodation_pois_gdf)
    if len(importance_scores) == accommodation_pois_gdf.shape[0]:
        accommodation_pois_gdf['importance_score'] = importance_scores
        print(f'len(importance_scores) {len(importance_scores)}')

    ##
    # FILTER pois (after sorting based on importance scores)
    ##
    if len(pois_gdf) > max_pois:
        print('Filtering POIs')
        pois_gdf = pois_gdf.sort_values(by='importance_score', ascending=False).head(max_pois).reset_index(drop=True)
        print(f'Af filtering pois_gdf.shape {pois_gdf.shape}')

    # merge accommodation and tourist attraction pois
    pois_gdf = pd.concat([pois_gdf, accommodation_pois_gdf], axis = 0)
    print(f'Merged pois_gdf.shape {pois_gdf.shape}')
    

    return pois_gdf, start_node_poi_ids


def extract_pois_using_tags(tags = None, location_dict = None, osm_id = None):
    '''
    Extract pois from OSM with 'tags'
    '''

    # extract poi data from OSM based on k-v tags
    # query separately for each tag and then remove duplicate pois
    dfs = []
    for k, v in tags:
        try:
            if osm_id:
                location = ox.geocoder.geocode_to_gdf(osm_id, by_osmid = True)
                polygon = location['geometry'].iloc[0]
                pois_gdf = ox.features_from_polygon(
                    polygon,
                    {k: v}
                )

            else:
                pois_gdf = ox.features_from_place(
                    location_dict,
                    # {
                    #     "city": "Bonn",
                    #     "state": "North Rhine-Westphalia",
                    #     "country": "Germany",
                    # },
                    {k: v}
                )
            dfs.append(pois_gdf)
            print(k, v, pois_gdf.shape)
        except ox._errors.InsufficientResponseError as e:
            print(f"{e} for {k, v}")
            continue

    pois_gdf = pd.concat(dfs)

    # remove duplicates
    pois_gdf = pois_gdf[~pois_gdf.duplicated()]

    # remove pois without name
    pois_gdf = pois_gdf[pois_gdf['name'].notnull()]
    print(f'A/f removing duplicates and pois without names {pois_gdf.shape}')

    # assign category to poi based on tags
    pois_gdf['tourism_category'] = pois_gdf.apply(assign_categories_to_poi, axis=1)

    # create osm_id column
    pois_gdf['osm_id'] = [_id for _type, _id in pois_gdf.index]

    return pois_gdf


def assign_categories_to_poi(row):
    '''
    Insert key 'category' in each poi object to store a binary vector of categories of the poi
    '''
    category_list = np.zeros((len(NODE_CATEGORIES),))

    # ACCOMMODATION
    if any(x in row.values for x in [
        'hotel', 'hostel', 'guest_house', 'apartment', 'camp_pitch', 'camp_site',
        'caravan_site', 'chalet', 'motel', 'spa_resort', 'Serviced Apartments'
    ]):
        category_list[NODE_CATEGORIES.index('accommodation')] = 1

    # FOOD ESTABLISHMENT
    elif any(x in row.values for x in ['restaurant', 'fast_food', 'canteen', 'food court', 'cafeteria']):
        category_list[NODE_CATEGORIES.index('food establishment')] = 1

    # CAFE
    elif 'cafe' in row.values:
        category_list[NODE_CATEGORIES.index('cafe')] = 1

    # ARTWORK
    if not pd.isna(row.get('artwork', np.nan)) or not pd.isna(row.get('artwork_type', np.nan)) or ('artwork' in row.values):
        category_list[NODE_CATEGORIES.index('artwork')] = 1

    # SCULPTURES
    if any(not pd.isna(row.get(x, np.nan)) for x in ['sculpture', 'sculpture_brick', 'sculpture_group']) or \
       any(x in row.values for x in ['sculpture', 'sculpture_brick', 'sculpture_group']):
        category_list[NODE_CATEGORIES.index('sculpture')] = 1

    # STATUES
    if 'statue' in row.values or not pd.isna(row.get('statue', np.nan)):
        category_list[NODE_CATEGORIES.index('statue')] = 1

    # 'graffiti and mural'
    if any(x in row.values for x in ['graffiti', 'mural', 'mural_painting', 'bust']):
        category_list[NODE_CATEGORIES.index('graffiti and mural')] = 1

    # MUSEUMS
    if 'museum' in row.values or not pd.isna(row.get('museum', np.nan)):
        category_list[NODE_CATEGORIES.index('museum')] = 1

    # GALLERIES
    if not pd.isna(row.get('gallery', np.nan)) or any(x in row.values for x in ['gallery', 'contemporary_art_gallery', 'art_gallery']):
        category_list[NODE_CATEGORIES.index('gallery')] = 1

    # HISTORIC LANDMARKS
    if not pd.isna(row.get('historic', np.nan)) or any(x in row.values for x in ['historic', 'history', 'art,history,nature']):
        category_list[NODE_CATEGORIES.index('historic landmark')] = 1

    # NATURAL WONDERS
    if not pd.isna(row.get('natural', np.nan)) or 'natural' in row.values:
        category_list[NODE_CATEGORIES.index('natural wonder')] = 1

    # VIEW POINTS
    if not pd.isna(row.get('viewpoint', np.nan)) or 'viewpoint' in row.values:
        category_list[NODE_CATEGORIES.index('viewpoint')] = 1

    # OTHER
    if not any(category_list):
        category_list[NODE_CATEGORIES.index('other')] = 1

    return category_list


def remove_point_pois_in_polygon(pois_gdf):

    ##
    # Removing POINT geometry POIs present within POLYGON geometry
    # To overcome issues such as animals in zoo marked as individual POIs
    ##
    point_pois = pois_gdf[pois_gdf.geometry.type == 'Point']
    polygon_pois = pois_gdf[pois_gdf.geometry.type == 'Polygon']
    print(f'point_pois.shape {point_pois.shape}, polygon_pois.shape {polygon_pois.shape}')

    # unify all polygon geometries
    unified_polygon = polygon_pois.union_all()
    filtered_point_pois = point_pois[
        ~point_pois.geometry.apply(lambda x: x.within(unified_polygon))
    ]
    print(f'filtered_point_pois.shape {filtered_point_pois.shape}')

    pois_gdf = gpd.GeoDataFrame(
        pd.concat([filtered_point_pois, polygon_pois])
    )
    print(f'pois_gdf.shape {pois_gdf.shape}')

    return pois_gdf


def get_start_node_poi_ids(pois, city_lat_lon, max_start_nodes = None):
    '''
    Create a list of start nodes from poi ids
    start nodes: Accommodation nodes 10km from city center
    '''
    start_node_pois_within_radius = []
    start_nodes_mask = []
    # start_node_pois = pois[
    #     pois['tourism_category'].apply(
    #         lambda x: np.array_equal(x, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    #     )
    # ]
    start_node_pois = pois.copy()

    # city_lat_lon = (50.735851, 7.10066)
    # compute haversine distance
    for idx, poi in start_node_pois.iterrows():
        is_start_node = False

        # extract centroid of the POI
        if isinstance(poi.geometry, Point):
            poi_lat_lon = (poi.geometry.y, poi.geometry.x)
        else:
            centroid = poi.geometry.centroid
            poi_lat_lon = (centroid.y, centroid.x)

        hotel_citycenter_distance = haversine(
            city_lat_lon,
            poi_lat_lon
        ) # in km

        if hotel_citycenter_distance <= 10:
            start_node_pois_within_radius.append(idx[1]) # since idx is (node|way|relation, id)
            is_start_node = True
        
        start_nodes_mask.append(is_start_node)

    # filter start nodes within max start nodes limit
    if max_start_nodes is not None and len(start_node_pois_within_radius) > max_start_nodes:
        print('Filtering start nodes')
        start_node_pois_within_radius = start_node_pois_within_radius[:MAX_START_NODES]
        count = 0
        start_nodes_mask_filtered = []
        for val in start_nodes_mask:
            if val is True and count < max_start_nodes:
                start_nodes_mask_filtered.append(True)
                count += 1
            else:
                start_nodes_mask_filtered.append(False)
        start_nodes_mask = start_nodes_mask_filtered

    print(f'len(start_node_pois_within_radius) {len(start_node_pois_within_radius)}')
    print(f'start_node_pois[start_nodes_mask].shape {start_node_pois[start_nodes_mask].shape}')
    
    return start_node_pois_within_radius, start_node_pois[start_nodes_mask].copy(deep = True)


def get_walkable_street_network(osm_id = None, location_dict = None):

    if osm_id:

        location = ox.geocoder.geocode_to_gdf(osm_id, by_osmid = True)
        polygon = location['geometry'].iloc[0]
        walking_network = ox.graph_from_polygon(
            polygon,
            network_type = 'walk'
        )   
    
    else:
        walking_network = ox.graph_from_place(
            location_dict,
            network_type = 'walk'
        )

    return walking_network
    

def get_distance_matrix(poi_data, walking_network, save_path):
    
    # setting tourism_category of existing nodes in walking network to junction_tourism_category
    junction_tourism_category = np.zeros((len(NODE_CATEGORIES),))
    junction_tourism_category[NODE_CATEGORIES.index('junction')] = 1
    nx.set_node_attributes(walking_network, junction_tourism_category, 'tourism_category')

    # add POI nodes(including accommodation nodes) in walking path graph
    # set 'label' = 1 for POIs
    original_network_unmutated = walking_network.copy()
    for idx, row in poi_data.iterrows():

        if isinstance(row.geometry, Point):
            x, y = row.geometry.x, row.geometry.y
        else:
            centroid = row.geometry.centroid
            x, y = centroid.x, centroid.y

        # using OSM IDS for node_ids
        # node_id = idx[1] # since idx is (node|way|relation, id)
        node_id = row['osm_id']
        _node_attrs = row.to_dict()  # Convert all tags to a dictionary
        node_attrs = {k: v for k, v in _node_attrs.items() if not (pd.isna(v).any() if isinstance(v, (np.ndarray, list, tuple)) else pd.isna(v))}  # Remove nan values
        node_attrs.update({"x": x, "y": y, "label": 1})  # Add coordinates to the attributes
        walking_network.add_node(node_id, **node_attrs)
    
        # connect the POI node to the ''nearest node in the walking network''
        nearest_node = ox.distance.nearest_nodes(original_network_unmutated, x, y)
        distance = ox.distance.euclidean(
            y, x,
            original_network_unmutated.nodes[nearest_node]['y'],
            original_network_unmutated.nodes[nearest_node]['x']
        )

        # Add an edge between the POI node and the nearest node with attribute length set to distance
        walking_network.add_edge(node_id, nearest_node, length=distance)
        walking_network.add_edge(nearest_node, node_id, length=distance)


    fig, ax = ox.plot_graph(walking_network, show = False, close = False)
    poi_data.plot(ax = ax, color = 'red', markersize = 50, label = 'POIs')
    plt.legend()
    plt.title(f"Walking paths and POIs")
    plt.savefig(os.path.join(save_path, f"walking_network.png"))

    # label remaining nodes (junction nodes) as 0
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(walking_network)
    gdf_nodes['label'] = gdf_nodes['label'].apply(lambda x: 0 if pd.isnull(x) else x)
    print(f'{gdf_nodes['label'].value_counts()}')
    node_label_attr_dict = gdf_nodes['label'].to_dict()
    nx.set_node_attributes(walking_network, node_label_attr_dict, 'label')
    # sanity check of labels
    label_values = list(nx.get_node_attributes(walking_network, name = 'label').values())
    print(f'{label_values.count(0), label_values.count(1)}')

    n_pois = label_values.count(1)

    # index pois
    poi_ids = []
    for node_id, data in walking_network.nodes(data=True):
        if data['label'] == 1:
            poi_ids.append(node_id)

    print(f'len(poi_ids) {len(poi_ids)}')

    # indexing POIs (including accommodation nodes)
    idx2poiid = {}
    poiid2idx = {}

    for idx, poi_id in enumerate(poi_ids):
        idx2poiid[idx] = poi_id
        poiid2idx[poi_id] = idx

    print(f'len(idx2poiid), len(poiid2idx) {len(idx2poiid), len(poiid2idx)}')

    # distance matrix
    # Intialise distance matrix with np.inf
    distance_matrix = np.full((n_pois, n_pois), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    # Matrix to store paths
    paths = [[[] for _ in range(n_pois)] for _ in range(n_pois)]

    # Precompute paths and distances for each node
    for source in tqdm(poiid2idx.keys()):
        # Get shortest paths and distances from the source to all other nodes
        path_lengths, path_dicts = nx.single_source_dijkstra(walking_network, source=source, weight='length')

        for target in poiid2idx.keys():
            distance_matrix[poiid2idx[source], poiid2idx[target]] = path_lengths[target]
            paths[poiid2idx[source]][poiid2idx[target]] = path_dicts[target]
    
    print(f'distance_matrix[0] {distance_matrix[0]}')
    return walking_network, idx2poiid, poiid2idx, distance_matrix, paths


def distance_matrix_to_poi_graph(poi_data, distance_matrix, threshold = 7500):
    
    updated_distance_matrix = copy.deepcopy(distance_matrix)
    updated_distance_matrix[updated_distance_matrix > threshold] = np.inf
    
    # construct POI graph based on new distance matrix
    POI_graph = nx.Graph()
    # insert edges in POI_graph if distance is non-zero and not np.inf in distance_matrix
    for i, source in idx2poiid.items():
        for j, target in idx2poiid.items():
            if i != j and updated_distance_matrix[i, j] != np.inf:
                POI_graph.add_edge(source, target, weight = updated_distance_matrix[i, j])
    
    print(f'POI_graph.number_of_nodes(), POI_graph.number_of_edges() {POI_graph.number_of_nodes(), POI_graph.number_of_edges()}')
    print(f'Min and max degree {min([x[1] for x in list(POI_graph.degree())]), max([x[1] for x in list(POI_graph.degree())])}')
    print(f'Min Max edge weight {max(data['weight'] for u, v, data in POI_graph.edges(data=True)), min(data['weight'] for u, v, data in POI_graph.edges(data=True))}')

    node_min_visit_time_attr_dict, node_max_visit_time_attr_dict, node_category_attr_dict = get_visit_time_dicts(POI_graph, poi_data)

    nx.set_node_attributes(POI_graph, node_min_visit_time_attr_dict, 'min_visit_time')
    nx.set_node_attributes(POI_graph, node_max_visit_time_attr_dict, 'max_visit_time')

    return POI_graph, updated_distance_matrix, node_category_attr_dict


    

def get_visit_time_dicts(POI_graph, poi_data):

    node_category_attr_dict = dict.fromkeys(list(dict(POI_graph.nodes(data=True)).keys()))
    
    for node_id in node_category_attr_dict.keys():
        node_category_attr_dict[node_id] = list(poi_data[poi_data['osm_id'] == node_id]['tourism_category'].iloc[0])
    
    node_min_visit_time_attr_dict = dict.fromkeys(list(dict(POI_graph.nodes()).keys()))
    node_max_visit_time_attr_dict = dict.fromkeys(list(dict(POI_graph.nodes()).keys()))
    for node_id in node_min_visit_time_attr_dict.keys():

        category_vector = node_category_attr_dict[node_id]
        category_indices_from_vector = list((np.array(category_vector) == 1).nonzero()[0])

        min_visit_time = min([
            VISIT_DURATION_BASED_ON_CATEGORIES[NODE_CATEGORIES[idx]][0] for idx in category_indices_from_vector
        ])

        max_visit_time = max([
            VISIT_DURATION_BASED_ON_CATEGORIES[NODE_CATEGORIES[idx]][1] for idx in category_indices_from_vector
        ])
        node_min_visit_time_attr_dict[node_id] = min_visit_time
        node_max_visit_time_attr_dict[node_id] = max_visit_time

    return node_min_visit_time_attr_dict, node_max_visit_time_attr_dict, node_category_attr_dict
    

def prepare_bearing_matrix(poiid2idx, poi_data):

    # initialise bearing_matrix with zeros
    bearing_matrix = np.zeros((len(poiid2idx), len(poiid2idx)))

    for source in tqdm(poiid2idx.keys()):

        lat1, lon1 = poi_data[poi_data['osm_id'] == source]['plotting_coords'].iloc[0]

        for target in poiid2idx.keys():

            if source == target:
                continue

            lat2, lon2 = poi_data[poi_data['osm_id'] == target]['plotting_coords'].iloc[0]

            # Get bearing (in decimal degrees) from the source to all other nodes; (lat1, lon1, lat2, lon2)
            bearing = ox.bearing.calculate_bearing(
                lat1, lon1, lat2, lon2
            )

            bearing_matrix[poiid2idx[source], poiid2idx[target]] = bearing
    
    print(f'bearing_matrix.shape {bearing_matrix.shape}')
    print(f'np.count_nonzero(bearing_matrix) {np.count_nonzero(bearing_matrix)}')
    
    return bearing_matrix


def get_importance_scores_of_pois(poi_data, data_not_from_osm = False):
    '''
    Get importance scores of POIs from OSM
    '''

    fetching_failed_pois = []
    importance_scores = []

    if data_not_from_osm:
        importance_scores = [1]* len(poi_data)
        return importance_scores
    
    for idx, row in tqdm(poi_data.iterrows(), total = poi_data.shape[0]):

        osm_id = idx[1] # since idx is (node|way|relation, id)
        prefix = 'N' if idx[0] == 'node' else 'W' if idx[0] == 'way' else 'R'
        osm_id = prefix + str(osm_id)

        params = {
            'osm_ids': osm_id,
            'format': 'json'
        }
        url = 'https://nominatim.openstreetmap.org/lookup'
        headers = {
            'User-Agent': 'YourAppName/1.0 (your@email.com)'
        }

        response = requests.get(url, params = params, headers = headers, timeout = 240)
        importance = 0
        if 200 == response.status_code:
            if len(response.json()) > 0:
                importance = response.json()[0].get('importance', 0)         
        else:
            print(url)
            fetching_failed_pois.append(osm_id)
        
        importance_scores.append(importance)

    return importance_scores


#%%
if __name__ == '__main__':
    #%%
    SEED = 0
    np.random.seed(SEED)
    
    #%%
    duplicate_copies = 3 # Number of duplicate requests for each request in test set
    MAX_POIS = 250
    MAX_START_NODES = 200
    CITY = 'berlin' #'hamburg' # 'bonn'

    location_dict = {
        "city": "Berlin",
        "country": "Germany",
    }
    # location_dict =     {
    #     "city": "Hamburg",
    #     "country": "Germany",
    # }
    # location_dict = {
    #     "city": "Bonn",
    #     "state": "North Rhine-Westphalia",
    #     "country": "Germany",
    # }

    OSM_ID = 'R62422' # Berlin #'R62508' Bonn #'R62782'  Hamburg#'R62422'
    CITY_LAT_LON = (52.5108850, 13.3989367) # Berlin #(50.7352621, 7.1024635) Bonn #(53.550341,10.000654) Hamburg #(52.5108850, 13.3989367) 
    SAVE_PATH = f'./data/{CITY}/saved_data'
    os.makedirs(SAVE_PATH, exist_ok = True)

    # ratio for splitting start node pois into train, val and test sets
    TEST_SET_RATIO = 0.2
    VAL_SET_RATIO = 0.1
    #%%
    poi_data, start_node_poi_ids = prepare_poi_data(
        # data_path = POI_DATA_PATH,
        location_dict = location_dict,
        osm_id = OSM_ID,
        city_lat_lon = CITY_LAT_LON,
        max_pois = MAX_POIS,
        max_start_nodes = MAX_START_NODES
    )


    #%%
    # get walkable street network from OSMNx
    walking_network = get_walkable_street_network(location_dict = location_dict) # get_walkable_street_network(osm_id = OSM_ID) # get_walkable_street_network(location_dict = location_dict)
    print(f'Walking network: {walking_network}')


    # prepare indexes and distance matrix
    walking_network, idx2poiid, poiid2idx, distance_matrix, paths = get_distance_matrix(
        poi_data = poi_data,
        walking_network = walking_network,
        save_path = SAVE_PATH
    )
    poi_data['_osm_id'] = poi_data['osm_id'].apply(lambda x: poiid2idx[x])
    #%%
    # prepare POI graph
    POI_graph, updated_distance_matrix, node_category_attr_dict = distance_matrix_to_poi_graph(
        poi_data,
        distance_matrix
    )
    #%%

    # bearing_matrix
    # (lat, lon)
    poi_data['plotting_coords'] = poi_data['geometry'].apply(lambda x: (x.y, x.x) if isinstance(x, Point) else (x.centroid.y, x.centroid.x))
    bearing_matrix = prepare_bearing_matrix(poiid2idx, poi_data)

    #%%

    # split start node pois into train, val and test sets
    test_val_start_node_poi_ids, train_start_node_poi_ids = train_test_split(
        start_node_poi_ids,
        test_size = 1 - (TEST_SET_RATIO + VAL_SET_RATIO),
        random_state = SEED,
        shuffle = True
    )
    val_ratio_adjusted = VAL_SET_RATIO / (VAL_SET_RATIO + TEST_SET_RATIO)
    val_start_node_poi_ids, test_start_node_poi_ids = train_test_split(
        test_val_start_node_poi_ids,
        test_size = 1 - val_ratio_adjusted,
        random_state = SEED,
        shuffle = True
    )

    print(f'len(train_start_node_poi_ids) {len(train_start_node_poi_ids)}')
    print(f'len(val_start_node_poi_ids) {len(val_start_node_poi_ids)}')
    print(f'len(test_start_node_poi_ids) {len(test_start_node_poi_ids)}')

    df_poi_train = pd.DataFrame(columns=[
        'req_id', 'start_node', 'time_constraint'
        ]
    )
    df_poi_val = pd.DataFrame(columns=[
        'req_id', 'start_node', 'time_constraint'
        ]
    )
    df_poi_test = pd.DataFrame(columns=[
        'req_id', 'start_node', 'time_constraint'
        ]
    )
    # create time constraints for train and test set
    req_idx = 0
    for start_node in train_start_node_poi_ids:
        for tc in TIME_CONSTRAINTS:
            df_poi_train.loc[req_idx] = [req_idx, start_node,  tc]
            req_idx += 1
    print(f'df_poi_train.shape {df_poi_train.shape}')
    req_idx = 0
    for start_node in val_start_node_poi_ids:
        for tc in TIME_CONSTRAINTS:
            df_poi_val.loc[req_idx] = [req_idx, start_node,  tc]
            req_idx += 1
    print(f'df_poi_val.shape {df_poi_val.shape}')
    req_idx = 0
    for start_node in test_start_node_poi_ids:
        for tc in TIME_CONSTRAINTS:
            df_poi_test.loc[req_idx] = [req_idx, start_node,  tc]
            req_idx += 1
    print(f'df_poi_test.shape {df_poi_test.shape}')

    #%%

    # insert category prefs and duplicate test set
    SELECTED_CATEGORIES = NODE_CATEGORIES[4:13]
    test_set_w_duplicate_requests_and_catprefs = pd.DataFrame(
        columns = ['req_id', 'start_node', 'time_constraint', 'cat_prefs']
    )

    req_idx = 0
    for start_node in test_start_node_poi_ids:
        for tc in TIME_CONSTRAINTS:
            cat = random.sample(SELECTED_CATEGORIES, 3)
            for i in range(duplicate_copies):
                test_set_w_duplicate_requests_and_catprefs.loc[req_idx] = [
                    req_idx, start_node,  tc, cat
                ]
                req_idx += 1
    print(f'test_set_w_duplicate_requests_and_catprefs.shape {test_set_w_duplicate_requests_and_catprefs.shape}')




    #%%
    # save all files
    poi_data.to_csv(os.path.join(SAVE_PATH, 'final_pois.csv'))
    ox.io.save_graphml(walking_network, os.path.join(SAVE_PATH, 'walking_network.graphml'))
    with open(os.path.join(SAVE_PATH,'start_node_pois_within_radius.pkl'), 'wb') as f:
        pickle.dump(start_node_poi_ids, f)
    with open(os.path.join(SAVE_PATH,'train_start_node_pois.pkl'), 'wb') as f:
        pickle.dump(train_start_node_poi_ids, f)
    with open(os.path.join(SAVE_PATH,'test_start_node_pois.pkl'), 'wb') as f:
        pickle.dump(test_start_node_poi_ids, f)
    indexing_dicts = {'idx2poiid': idx2poiid, 'poiid2idx': poiid2idx}
    with open(os.path.join(SAVE_PATH,'indexing_dicts.pkl'), 'wb') as f:
        pickle.dump(indexing_dicts, f)
    with open(os.path.join(SAVE_PATH,'distance_matrix.npy'), 'wb') as f:
        np.save(f, distance_matrix)
    path_related_dicts = {'paths': paths}
    with open(os.path.join(SAVE_PATH, 'path_related_dicts.pkl'), 'wb') as f:
        pickle.dump(path_related_dicts, f)
    with open(os.path.join(SAVE_PATH, 'node_category_attr_dict.pkl'), 'wb') as f:
        pickle.dump(node_category_attr_dict, f)
    nx.write_graphml(POI_graph, os.path.join(SAVE_PATH,'POI_graph_updated.graphml'))
    with open(os.path.join(SAVE_PATH, 'distance_matrix_updated.npy'), 'wb') as f:
        np.save(f, updated_distance_matrix)
    with open(os.path.join(SAVE_PATH,'bearing_matrix.npy'), 'wb') as f:
        np.save(f, bearing_matrix)

    #%%
    df_poi_train.to_csv(os.path.join(SAVE_PATH, 'train_set.csv'))
    df_poi_val.to_csv(os.path.join(SAVE_PATH, 'val_set.csv'))
    test_set_w_duplicate_requests_and_catprefs.to_csv(os.path.join(SAVE_PATH, 'test_set.csv'))
    

# %%
