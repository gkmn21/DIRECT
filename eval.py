#%%
import os
import pickle
import json
import ast

import pandas as pd 
import numpy as np
from itertools import combinations
from scipy.spatial import distance as scp_dist
from functools import reduce
import operator

import pygeohash as pgh

from constants import FINAL_CATEGORIES
from dataset_generation.constants import VISIT_DURATION_BASED_ON_CATEGORIES


import pdb
#%%

def arp_measure(poi_list, itinerary_list):
    arp = 0
    for poi in poi_list:
        cont = 0
        for itinerary in itinerary_list:
            if poi in itinerary:
                cont += 1
        arp += cont / len(itinerary_list)
    return arp / len(poi_list)

#%%
def total_intra_list_distance(vectors):
    packed = [int(''.join('1' if x else '0' for x in v), 2) for v in vectors]
    total = 0
    n = len(packed)
    n_cat = len(vectors[0])
    for i in range(n):
        for j in range(i+1, n):
            total += ((packed[i] ^ packed[j]).bit_count()/n_cat) # normalised hamming distance
    return total

def compute_ild(vectors):
    n = len(vectors)
    tot = total_intra_list_distance(vectors)
    # number of distinct pairs = n*(n-1)/2
    return tot / (n*(n-1)/2)

##### Average Dice Coefficient #####

def average_duplicate_dice_sets(
    df,
    set_col,
    duplicates_per_sample
):

    # assign artificial group IDs
    df['orig_group'] = df.index // duplicates_per_sample

    def dice(a, b):
        inter = len(a & b)
        total = len(a) + len(b)
        return 1.0 if total == 0 else 2 * inter / total

    scores = []
    for _, grp in df.groupby('orig_group'):
        sets = grp[set_col].tolist()
        pair_scores = [dice(set(a), set(b)) for a, b in combinations(sets, 2)]
        scores.append(np.mean(pair_scores))

    return float(np.mean(scores))

#############
#%%

def compute_metrics(
        MODEL_NAME,
        VARIANT,
        results_df,
        test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        METRICS_RESULTS_CSV_PATH,
        TEST_SET_W_CATEGORIES
    ):
    
    # merge test set with results dataframe
    merged_results_df = pd.merge(results_df, test_set, left_on = REQ_ID_COL, right_on = 'req_id', how = 'inner')
    print(f'merged_results_df.columns {merged_results_df.columns}, merged_results_df.shape {merged_results_df.shape}')

    # change type of route column values to list
    if type(merged_results_df[ROUTE_COL].loc[0]) != list:
        try:
            merged_results_df[ROUTE_COL] = merged_results_df[ROUTE_COL].apply(lambda x: ast.literal_eval(x.replace('np.int64(', '').replace(')', '')))
        except Exception as e:
            print(e)
    
    # if route_col does not contain osm ids then replace id with osm id
    if not ROUTE_COL_CONTAINS_OSM_IDS:
        merged_results_df[ROUTE_COL] = merged_results_df[ROUTE_COL].apply(
            lambda x: [final_pois_df[final_pois_df['_osm_id'] == _id]['osm_id'].values[0] for _id in x]
        )
    
    # remove start node from routes
    if REMOVE_START_NODE:
        merged_results_df[ROUTE_COL] = merged_results_df[ROUTE_COL].apply(
            lambda x: x[1:] if len(x) > 1 else []
        )

    ##
    # Remove end node (if end node == start node in request)
    ##
    routes_wo_end_node = []
    for i, row in merged_results_df.iterrows():

        if row[ROUTE_COL] == []:
            routes_wo_end_node.append(row[ROUTE_COL])

        elif row[ROUTE_COL][-1] == int(row['start_node']):
           routes_wo_end_node.append(row[ROUTE_COL][:-1])

        else:
            routes_wo_end_node.append(row[ROUTE_COL])
    
    merged_results_df['routes_wo_end_node'] = routes_wo_end_node


    metrics_dict = {
        'mean_route_length': None,
        'n_unique_pois': None,
        'rbl': None,
        'rbld': None, # mean reachability distance
        'n_routes_within_walkable_distance': None,
        'n_routes_within_time_constraint': None,
        'fbl': None, # routes within walkable distance and time constraint
        'ilcd': None,
        'agc': None,
        'arf': None, # average recommendation popularity
        'ards': None,
        'pc': None,
        'arp': None
    }
    metrics_std_dict = {
        'route_length': None,
        'rbl': None,
        'rbld': None, # mean reachability distance
        'fbl': None, # routes within walkable distance and time constraint
        'ilcd': None,
        'agc': None, # median of coverage, coverage is the no. of geohash cells covered by POIs in route
        'ards': None
    }

    if TEST_SET_W_CATEGORIES:
        metrics_dict['acps'] = None
        metrics_std_dict['acps'] = None


    merged_results_df['route_pois_count'] = merged_results_df['routes_wo_end_node'].apply(
        lambda x: len(x)
    ) 
    merged_results_df['n_unique_pois'] = merged_results_df['routes_wo_end_node'].apply(
        lambda x: len(set(x))
    )

    metrics_dict['mean_route_length'] = merged_results_df['route_pois_count'].mean()
    metrics_std_dict['route_length'] = merged_results_df['route_pois_count'].std(ddof = 0)
    
    # set reachability to 1 for classical model
    if MODEL_NAME == 'classical_model':

        # sum(merged_results_df['Cycle_w_osm_ids'].apply(lambda cycle: cycle[0] == cycle[-1]))
        metrics_dict['rbl'] = 1
        metrics_dict['rbld'] = 0
    else:

        # in how many cases did the route end at start node
        metrics_dict['rbl'] = sum(
            merged_results_df.apply(
                lambda row: int(row['start_node']) == row[ROUTE_COL][-1] if len(row[ROUTE_COL]) > 0 else False,
                axis = 1
            )
        )/ len(merged_results_df)
        metrics_std_dict['rbl'] = merged_results_df.apply(
                lambda row: int(row['start_node']) == row[ROUTE_COL][-1] if len(row[ROUTE_COL]) > 0 else False,
                axis = 1
            ).std(ddof = 0)

        reachability_distance = []
        for i, row in merged_results_df.iterrows():

            if row[ROUTE_COL] == []:
                reachability_distance.append(None)

            elif row[ROUTE_COL][-1] == int(row['start_node']):
                reachability_distance.append(0)
            else:
                # get distance from start node to end node
                start_node_idx = poiid2idx[int(row['start_node'])]
                end_node_idx = poiid2idx[row[ROUTE_COL][-1]]
                distance = distance_matrix[start_node_idx][end_node_idx] / 1000 # converting distance to km
                reachability_distance.append(distance)
        merged_results_df['reachability_distance'] = reachability_distance

        metrics_dict['rbld'] = merged_results_df['reachability_distance'].mean()
        metrics_std_dict['rbld'] = merged_results_df['reachability_distance'].std(ddof = 0)

        
    n_routes_within_walkable_distance = []
    n_routes_within_time_constraint = []
    
    if MODEL_NAME == 'classical_model':
        # Note: In classical baseline, the cycle length is the time taken to travel the route	
        for i, row in merged_results_df.iterrows():

            if row['Cycle length']:
                route_time = row['Cycle length']/3600
                time_constraint = row['time_constraint']

                walkable_threshold = 2 if time_constraint == 2 else 3

                n_routes_within_walkable_distance.append(
                    False if route_time > walkable_threshold else True
                )
                n_routes_within_time_constraint.append(
                    False if route_time > time_constraint else True
                )
            else:
                n_routes_within_walkable_distance.append(False)
                n_routes_within_time_constraint.append(False)
    else:       

        for i, row in merged_results_df.iterrows():
            if row[ROUTE_COL] == []:
                n_routes_within_walkable_distance.append(False)
                n_routes_within_time_constraint.append(False)
            else:
                # get distance from start node to end node.... 
                start_node_idx = poiid2idx[int(row['start_node'])]
                first_poi_idx = poiid2idx[row[ROUTE_COL][0]]

                route_distance = distance_matrix[start_node_idx][first_poi_idx]
                visit_time = 0
                for idx in range(len(row[ROUTE_COL]) - 1):
                    route_distance += distance_matrix[poiid2idx[row[ROUTE_COL][idx]]][poiid2idx[row[ROUTE_COL][idx + 1]]]
                    visit_time += final_pois_df[final_pois_df['osm_id'] == row[ROUTE_COL][idx]]['min_visit_duration'].values[0]
                
                visit_time += final_pois_df[final_pois_df['osm_id'] == row[ROUTE_COL][-1]]['min_visit_duration'].values[0]
                route_distance += distance_matrix[poiid2idx[row[ROUTE_COL][-1]]][start_node_idx] 
                route_time = visit_time + ((route_distance/1000)/5) # 5km/h walking speed
                route_distance = route_distance/1000 # convert to km

                time_constraint = row['time_constraint']
                walkable_threshold = 10 if time_constraint == 2 else 15 # in km

                n_routes_within_walkable_distance.append(
                    False if route_distance > walkable_threshold else True
                )
                n_routes_within_time_constraint.append(
                    False if route_time > time_constraint else True
                )
            
    merged_results_df['within_walkable_distance'] = n_routes_within_walkable_distance
    merged_results_df['within_time_constraint'] = n_routes_within_time_constraint
    merged_results_df['feasible'] = merged_results_df['within_walkable_distance'] & merged_results_df['within_time_constraint']
    metrics_dict['fbl'] = merged_results_df['feasible'].sum()/len(merged_results_df['feasible'])
    metrics_std_dict['fbl'] = merged_results_df['feasible'].std(ddof = 0)
    metrics_dict['n_routes_within_walkable_distance'] = sum(n_routes_within_walkable_distance) 
    metrics_dict['n_routes_within_time_constraint'] = sum(n_routes_within_time_constraint)

    
    # compute average recommendation popularity
    poi_list = list(final_pois_df[~final_pois_df['_osm_id'].isin(start_node_pois_within_radius)]['osm_id'].values)
    arp = arp_measure(poi_list, list(merged_results_df['routes_wo_end_node'].values))
    metrics_dict['arf'] = arp

    # POI Coverage
    route_pois = set(poi for route in list(merged_results_df['routes_wo_end_node'].values) for poi in route)
    total_pois = len(poi_list)
    metrics_dict['pc'] = len(route_pois)/total_pois

    
    # compute diversity

    # remove category 1 to 4 since they are not related to tourist attractions
    # Also remove start nodes from route
    category_list = list(range(len(FINAL_CATEGORIES[4:])))
    merged_results_df['route_w_categories'] =  merged_results_df['routes_wo_end_node'].apply(
        lambda x: [
            final_pois_df[final_pois_df['osm_id'] == osm_id]['tourism_category'].values[0][4:] for osm_id in x
        ]
    )
        
    ## Intra-list diversity
    ild_result_list = []
    for route in merged_results_df['route_w_categories']:
        if len(route) < 2:
            ild_result_list.append(0)
            continue
        ild_result = compute_ild(route)
        # normalise and save ild
        ild_result_list.append(ild_result/len(FINAL_CATEGORIES[4:]))
    
    metrics_dict['ilcd'] = np.mean(ild_result_list)
    metrics_std_dict['ilcd'] = np.std(ild_result_list, ddof = 0)
    print(f'metrics_dict[ilcd] {metrics_dict["ilcd"]}')

    # compute coverage with geohashes
    # route with geohashes of all pois except start node
    merged_results_df['route_w_geohashes'] = merged_results_df['routes_wo_end_node'].apply(
        lambda x: [
            final_pois_df[final_pois_df['osm_id'] == osm_id]['geohash'].values[0] for osm_id in x
        ]
    )

    coverage_list = []
    for route in merged_results_df['route_w_geohashes']:
        coverage = len(set(route))
        coverage_list.append(coverage)
    merged_results_df['coverage'] = coverage_list
    metrics_dict['agc'] = merged_results_df['coverage'].mean()
    metrics_std_dict['agc'] = merged_results_df['coverage'].std(ddof = 0)

    
    print(f'metrics_dict[agc] {metrics_dict['agc']}')

    ## Average Dice Coefficent
    avg_dice_coeff = average_duplicate_dice_sets(merged_results_df, set_col = 'routes_wo_end_node', duplicates_per_sample = 3)
    metrics_dict['ards'] = avg_dice_coeff

    ## Category preference dice similarity
    if TEST_SET_W_CATEGORIES:
        cat_pref_sim_results = []
        cat_pref_sim_results2 = []
        merged_results_df['cat_prefs_binary'] = merged_results_df['cat_prefs'].apply(lambda x: np.array([1 if i in x else 0 for i in FINAL_CATEGORIES[4:]], dtype = np.float32))
        for idx, row in merged_results_df.iterrows():

            similarity = 0
            similarity2 = 0
            if len(row['route_w_categories']) > 0:

                route_cats = reduce(operator.or_, map(np.array, row['route_w_categories']))
                # route_cats = np.logical_or(row['route_w_categories'])
                request_cats = row['cat_prefs_binary']
                similarity = 1 - scp_dist.dice(route_cats, request_cats)
                #similarity2 = sum([1 - scp_dist.dice(rc, request_cats) for rc in np.array(row['route_w_categories'])])/len(row['route_w_categories'])

            cat_pref_sim_results.append(similarity)
            #cat_pref_sim_results2.append(similarity2)
        metrics_dict['acps'] = float(np.mean(cat_pref_sim_results))
        #metrics_dict['avg_category_pref_dice_similarity2'] = float(np.mean(cat_pref_sim_results2))

    # ARP
    routes_w_popularity = []
    for r in list(merged_results_df['routes_wo_end_node']):
        route_popularity = 0
        if len(r) != 0:
            route_popularity = np.mean(
                [
                    final_pois_df[final_pois_df['osm_id'] == osm_id]['importance_score'].values[0] for osm_id in r
                ]
            )
        routes_w_popularity.append(route_popularity)
    # merged_results_df['route_popularity'] = merged_results_df['routes_wo_end_node'].apply(
    #     lambda x: np.mean([
    #         final_pois_df[final_pois_df['osm_id'] == osm_id]['importance_score'].values[0] for osm_id in x
    #     ])
    # )
    merged_results_df['route_popularity'] = routes_w_popularity
    metrics_dict['arp'] = merged_results_df['route_popularity'].mean()


    print(f'metrics_dict {metrics_dict}')
    print(f'metrics std dict {metrics_std_dict}')
    with open(os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_metrics.txt'), 'a') as f:
        f.write(VARIANT)
        json.dump(metrics_dict, f, indent = 4)
        json.dump(metrics_std_dict, f, indent = 4)

    # save metrics dict to pickle file
    with open(os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_{VARIANT}_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_dict, f)
    with open(os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_{VARIANT}_std_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_std_dict, f)
    # save merged results dataframe to csv
    merged_results_df.to_csv(
        os.path.join(METRICS_RESULTS_CSV_PATH, f'{MODEL_NAME}_{VARIANT}_metrics.csv'),
        index = False
    )

#############
#%%

if __name__ == '__main__':
    #%%
    CITY = 'berlin'
    DATA_PATH = f'./data/{CITY}/saved_data'
    
    # path to model results dataframe and metrics save path
    MODEL_NAME = 'exp1'
    RESULTS_PATH = f'./results/{MODEL_NAME}/{CITY}/episode_results.csv' 
    
    TEST_SET_PATH = f'./data/{CITY}/saved_data/test_set.csv'
    METRICS_SAVE_PATH = f'./results/{MODEL_NAME}/{CITY}/metrics'
    METRICS_RESULTS_CSV_PATH = METRICS_SAVE_PATH 
    os.makedirs(METRICS_SAVE_PATH, exist_ok = True)
    os.makedirs(METRICS_RESULTS_CSV_PATH, exist_ok = True)
    
    # model specific column names
    REQ_ID_COL = 'req_id'
    ROUTE_COL = 'route'
    ROUTE_COL_CONTAINS_OSM_IDS = False # Change to True for Naive baselines
    REMOVE_START_NODE = True 
    GEOHASH_PRECISION = 7 # 153m × 153m	
    TEST_SET_W_CATEGORIES = True # whether test set contains user category prefs


    # load poi data, distance matrices, start nodes
    with open(f'{DATA_PATH}/indexing_dicts.pkl', 'rb') as f:
        indexing_dicts = pickle.load(f)
    print(f'indexing_dicts {len(indexing_dicts)}')
    idx2poiid = indexing_dicts['idx2poiid']
    poiid2idx = indexing_dicts['poiid2idx']
    print(f'len(idx2poiid) {len(idx2poiid)}, len(poiid2idx) {len(poiid2idx)}')

    final_pois_df = pd.read_csv(f'{DATA_PATH}/final_pois.csv')
    print(final_pois_df.shape)

    final_pois_df['tourism_category'] = final_pois_df['tourism_category'].apply(
        lambda x:
        [
            int(i) for i in x.strip('[.]').split('. ')
        ]
    )
    final_pois_df['plotting_coords'] = final_pois_df['plotting_coords'].apply(
        lambda x: ast.literal_eval(x)
    )

    final_pois_df['geohash'] = final_pois_df['plotting_coords'].apply(
        lambda x: pgh.encode(x[0], x[1], precision = GEOHASH_PRECISION) # precision 6: 1.22km×0.61km
    )


    with open(f'{DATA_PATH}/start_node_pois_within_radius.pkl', 'rb') as f:
        start_node_pois_within_radius = pickle.load(f)
    start_node_pois_within_radius = [poiid2idx[_id] for _id in start_node_pois_within_radius]
    print(f'len(start_node_pois_within_radius) {len(start_node_pois_within_radius)}')

    # Distance matrix in meters
    with open(f'{DATA_PATH}/distance_matrix.npy', 'rb') as f:
        distance_matrix = np.load(f)
    print(f'distance_matrix {distance_matrix.shape}')
    


    if MODEL_NAME != 'classical_model':
        
        # convert visit duration from seconds to hours and insert into final_pois_df
        final_pois_df['min_visit_duration'] = final_pois_df['tourism_category'].apply(
            lambda x: min([
                VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][0]/3600 for idx, val in enumerate(x) if val == 1
            ])
        )
        final_pois_df['max_visit_duration'] = final_pois_df['tourism_category'].apply(
            lambda x: max([
                VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][1]/3600 for idx, val in enumerate(x) if val == 1
            ])
        ) 

    # dataframe with route requests of test set
    test_set = pd.read_csv(
        TEST_SET_PATH
    )
    small_timebudget_test_set = test_set[test_set['time_constraint'].isin([2, 3, 4])]
    medium_timebudget_test_set = test_set[test_set['time_constraint'].isin([5, 6, 7])]
    large_timebudget_test_set = test_set[test_set['time_constraint'].isin([8, 9, 10])]
    print(f'test_set.columns {test_set.columns}, test_set.shape {test_set.shape}, small_timebudget_test_set.shape {small_timebudget_test_set.shape}, medium_timebudget_test_set.shape {medium_timebudget_test_set.shape}, large_timebudget_test_set.shape {large_timebudget_test_set.shape} ')

    results_df = pd.read_csv(
        RESULTS_PATH
    )
    # insert time_constraint in classical model results
    if MODEL_NAME == 'classical_model' or MODEL_NAME == 'rl_baseline':
        results_df['time_constraint'] = results_df[REQ_ID_COL].apply(
            lambda rid: test_set[test_set['req_id'] == rid]['time_constraint'].iloc[0]
        )
        results_df['start_node'] = results_df[REQ_ID_COL].apply(
            lambda rid: test_set[test_set['req_id'] == rid]['start_node'].iloc[0]
        )
    small_timebudget_results_df = results_df[results_df['time_constraint'].isin([2, 3, 4])]
    medium_timebudget_results_df = results_df[results_df['time_constraint'].isin([5, 6, 7])]
    large_timebudget_results_df = results_df[results_df['time_constraint'].isin([8, 9, 10])]
    print(f'results_df.columns {results_df.columns}, results_df.shape {results_df.shape}, small_timebudget_results_df.shape {small_timebudget_results_df.shape}, medium_timebudget_results_df.shape {medium_timebudget_results_df.shape}, large_timebudget_results_df.shape {large_timebudget_results_df.shape}')
   
    # remove 'start_node' and 'time_constraint' column for results_df of my approach
    results_df = results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)
    small_timebudget_results_df = small_timebudget_results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)
    medium_timebudget_results_df = medium_timebudget_results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)
    large_timebudget_results_df = large_timebudget_results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)

    
    ##
    # Global metrics
    ##
    print('########### GLOBAL METRICS ################')
    variant = 'global'
    compute_metrics(
        MODEL_NAME,
        variant,
        results_df,
        test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        METRICS_RESULTS_CSV_PATH,
        TEST_SET_W_CATEGORIES
    )
    print('##########################################')
    
    print('########### 2h-4h METRICS ################')
    variant = '2h_to_4h'
    compute_metrics(
        MODEL_NAME,
        variant,
        small_timebudget_results_df,
        small_timebudget_test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        METRICS_RESULTS_CSV_PATH,
        TEST_SET_W_CATEGORIES
    )
    print('##########################################')

    print('########### 5h-7h METRICS ################')
    variant = '5h_to_7h'
    compute_metrics(
        MODEL_NAME,
        variant,
        medium_timebudget_results_df,
        medium_timebudget_test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        METRICS_RESULTS_CSV_PATH,
        TEST_SET_W_CATEGORIES
    )
    print('##########################################')

    print('########### 8h-10h METRICS ################')
    variant = '8h_to_10h'
    compute_metrics(
        MODEL_NAME,
        variant,
        large_timebudget_results_df,
        large_timebudget_test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        METRICS_RESULTS_CSV_PATH,
        TEST_SET_W_CATEGORIES
    )
    print('##########################################')


# %%
