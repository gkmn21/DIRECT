#%%
import os
import copy

from tqdm import tqdm
import pandas as pd
import networkx as nx
import numpy as np

os.chdir('/media/data/mann/RL/RL/')
from utils import read_data, rescale_inputs

#%%
def greedy_path_builder(final_pois_df, distance_matrix, idx2poiid, iteration_POI_graph, request_params):
    """
    Constructs paths starting from start node in request_params, inserting the highest scoring neighbor
    while respecting a distance budget. Returns a list of nodes forming route.
    
    Args:
        distance_matrix : Distance matrix
        final_pois_df : DataFrame with rows for each node and columns for node attributes
        iteration_POI_graph: NetworkX graph with nodes matching distance_matrix and final_pois_df
        request_params: Dictionary with keys ['start_node', 'time_constraint']
    
    Returns:
        route: list of node_ids
    """

    walking_speed = request_params['walking_speed']
    start_node = request_params['start_node']
    time_constraint = request_params['time_constraint']
    # walkable distance
    distance_constraint = 10 if time_constraint <= 2 else 15
    score_col = request_params['score_col']
    
    path = [start_node] # path without osm IDS
    visited = set(path)
    current_node = start_node
    total_distance = 0
    temporal_total_distance = 0

    while True:

        neighbors = [n for n in iteration_POI_graph.neighbors(current_node) if n not in visited]
        if not neighbors:
            break

        # Sort neighbors by score column
        neighbors = sorted(
            neighbors, key = lambda n: final_pois_df[final_pois_df['_osm_id'] == n][score_col].values[0], reverse = True
        )

        nearest_neighbor = neighbors[0]
        dist_to_nearest = distance_matrix[current_node][nearest_neighbor]
        dist_back_to_start = distance_matrix[nearest_neighbor][start_node]
        projected_total = total_distance + dist_to_nearest + dist_back_to_start

        temporal_dist_to_nearest = (dist_to_nearest/walking_speed) + iteration_POI_graph.nodes[nearest_neighbor].get('min_visit_time', 0)
        temporal_dist_back_to_start = distance_matrix[nearest_neighbor][start_node]/walking_speed
        temporal_projected_total = temporal_total_distance + temporal_dist_to_nearest + temporal_dist_back_to_start

        if (projected_total <= distance_constraint) and (temporal_projected_total <= time_constraint):

            path.append(nearest_neighbor)
            visited.add(nearest_neighbor)
            total_distance += dist_to_nearest
            temporal_total_distance += temporal_dist_to_nearest
            current_node = nearest_neighbor

        else:
            break  # No neighbor can be added within budget

    # Add return to start
    path.append(start_node)

    # convert all path nodes ids to OSM IDS
    _path = [idx2poiid[n] for n in path]

    return _path


#%%

if __name__ == '__main__':
    #%%
    CITY = 'berlin' #'bonn'
    DATA_DIR = f'/media/data/mann/RL/RL/data/{CITY}'
    OUTPUT_DIR = f'/media/data/mann/RL/RL/baseline_results/f_greedy_popularity/{CITY}' #'/media/data/mann/RL/Baselines/greedy_popularity'
    TEST_SET_PATH = f'/media/data/mann/RL/RL/data/{CITY}/saved_data/test_set_w_duplicate_requests_and_catprefs.csv' #'/media/data/mann/RL/RL/baseline_results/test_set.csv'
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    
    # dataframe with route requests of test set
    test_set = pd.read_csv(
        TEST_SET_PATH
    )
    print(f'test_set.columns {test_set.columns}, test_set.shape {test_set.shape}')

    # dataframe to store model results on test set
    results_df = pd.DataFrame(
        columns = ['req_id', 'start_node', 'time_constraint', 'route']
    )

    (
        final_pois, Bonn_walking_network, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    ) = read_data(DATA_DIR)

    POI_graph, final_pois, distance_matrix, unfiltered_distance_matrix, start_node_pois_within_radius, _ = rescale_inputs(
        POI_graph,
        final_pois,
        poiid2idx,
        distance_matrix,
        unfiltered_distance_matrix,
        start_node_pois_within_radius
    )
    #%%
    iteration_POI_graph = copy.deepcopy(POI_graph)
    # remove all start nodes from iteration_POI_graph graph, and then iterate over test set and insert each start node of a request and remove the start node on termination
    iteration_POI_graph.remove_nodes_from(start_node_pois_within_radius)
    print(f'iteration_POI_graph {iteration_POI_graph}')
    
    for idx, row in tqdm(test_set.iterrows(), total = len(test_set)):

        req_id = row['req_id']
        start_node = poiid2idx[row['start_node']]
        time_constraint = row['time_constraint']

        # insert start node and its attributes and edges in iteration_POI_graph
        iteration_POI_graph.add_node(start_node, **POI_graph.nodes[start_node])
        iteration_POI_graph.add_edges_from((start_node, nbr, POI_graph[start_node][nbr]) for nbr in POI_graph.neighbors(start_node) if nbr not in start_node_pois_within_radius)

        route = greedy_path_builder(
            final_pois,
            unfiltered_distance_matrix,
            idx2poiid,
            iteration_POI_graph,
            {
                'walking_speed': 5,
                'start_node': start_node,
                'time_constraint': time_constraint,
                'score_col': 'importance_score'
            }
        )
        
        results_df.loc[idx] = [
            req_id,
            start_node,
            time_constraint,
            route
        ]

        # remove start node and its attributes and edges in iteration_POI_graph
        iteration_POI_graph.remove_node(start_node)

    #%%
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'episode_results.csv'))
# %%
