'''
This script is used to run inference on a test set using a trained reinforcement learning model.
It loads the model, sets up the environment, and runs the model on the test set, saving the results to a CSV file.
'''
#%%
import pandas as pd
import numpy as np
import gymnasium as gym
# from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, MultiPolygon, LineString, Polygon
from shapely.ops import split
import numpy as np
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd

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
from utils import read_data, rescale_inputs, normalise_inputs
from direct_env import CityEnv

from constants import FINAL_CATEGORIES

SEED = 100
#%%
if __name__ == '__main__':

    #%%
    np.random.seed(SEED)

    EXP_NAME = 'abl_wo_dist_time_penalty_berlin'#'exp0605_2'#'exp3004_1'#'exp2604_6' 
    MODEL_ITERATION = '1390000.zip'#'1360000.zip'#'930000.zip'
    CITY = 'berlin' #'wangerland'#'wangerland' # 'bonn'
    DATA_DIR = f'/media/data/mann/RL/RL/data/{CITY}'
    IMPL_DIR = '/media/data/mann/RL/RL/content/' #'/media/data/mann/RL/motores_prez/hanover'#'/media/data/mann/RL/RL'
    OUTPUT_DIR = f'/media/data/mann/RL/Outputs/{EXP_NAME}/{CITY}'#'/media/data/mann/RL/Outputs/motores_prez/hanover/exp1'
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    #%%
        #---------------------------------------------------------------------------------------#
    TEST_SET_PATH = f'/media/data/mann/RL/RL/data/{CITY}/saved_data/test_set_w_duplicate_requests_and_catprefs.csv' # '/media/data/mann/RL/RL/baseline_results/test_set.csv'
    # dataframe with route requests of test set
    test_set = pd.read_csv(
        TEST_SET_PATH
    )
    print(f'test_set.columns {test_set.columns}, test_set.shape {test_set.shape}')
    # convert test set category prefs to binary vectors
    test_set['cat_prefs_binary'] = test_set['cat_prefs'].apply(lambda x: np.array([1 if i in x else 0 for i in FINAL_CATEGORIES[4:]], dtype = np.float32))

    # dataframe to store model results on test set
    results_df = pd.DataFrame(
        columns = ['req_id', 'start_node', 'time_constraint', 'route', 'cum_reward', 'rewards_list']
    )

    (
        final_pois, Bonn_walking_network, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    ) = read_data(DATA_DIR)
    #%%

    POI_graph, final_pois, distance_matrix, unfiltered_distance_matrix, start_node_pois_within_radius, _ = rescale_inputs(
        POI_graph,
        final_pois,
        poiid2idx,
        distance_matrix,
        unfiltered_distance_matrix,
        start_node_pois_within_radius
    )
    # distance_matrix, unfiltered_distance_matrix = normalise_inputs(distance_matrix, unfiltered_distance_matrix) # now nans are present instead of np.inf
    distance_matrix = normalise_inputs(distance_matrix) # now nans are present instead of np.inf

    # change poiid2idx after rescaling inputs
    # new poiid2idx where both key and value are ids and not poi ids
    poiid2idx2 = dict.fromkeys(list(poiid2idx.values()))
    for key in poiid2idx2.keys():
        poiid2idx2[key] = key
    print(f'len(poiid2idx2) {len(poiid2idx2)}')

    #---------------------------------------------------------------------------------------#

    # register custom environment
    gym.register(
        id = 'gymnasium_env/CityEnv-v0',
        entry_point = CityEnv
    )

    # make gym environment
    MAX_CITY_GRAPH_NODES = 500
    env = gym.make(
        'gymnasium_env/CityEnv-v0',
        city_graph = POI_graph,
        poi_limit = 20, # Arbitrary value for now
        # start_node_osmids = start_node_pois_within_radius,
        # distance_matrix = distance_matrix,
        all_start_nodes = start_node_pois_within_radius,
        bearing_matrix = bearing_matrix,
        poiid2idx = poiid2idx2,
        final_pois_gdf = final_pois,
        unfiltered_distance_matrix = unfiltered_distance_matrix,
        output_dir = OUTPUT_DIR,
        current_mode = 'test',
        max_city_graph_nodes = MAX_CITY_GRAPH_NODES,
        candidate_poi_generator_k = 5,
        alpha_params_dict = {
            'temporal_distance': 1,# 0.33,
            'diversity': 0.5, # 0.33,
            'coverage': 0.5, # 0.33
            'cat_prefs': 0
        },

        #max_episode_steps = 200
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 100)

    print('checking env with gym and stable baselines............')
    gym.utils.env_checker.check_env(env.unwrapped)
    check_env(env)

    ##
    # testing environment with random actions
    ##
    obs = env.reset(
        options = {'test_sample_parameters': {
            'start_node': 1,
            'time_constraint': 3,
            'request_id': '',
            'cat_prefs': None
        }}
    )[0]
    #%%
    # Take some random actions
    for i in range(10):

        rand_action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(rand_action)
        # print(f"Obs: {obs}")
        # print(f"Reward: {reward}")

        if terminated or truncated:
            if (i%2 == 0):
                env.render()
            obs = env.reset()[0]

    #---------------------------------------------------------------------------------------#

    # Load saved model and run inference
    print('Run inference')
    model_name = EXP_NAME#'exp0703_2'#'exp1'#"DQN_one_reward"
    models_dir = os.path.join(IMPL_DIR, 'models', model_name)
    model_path = os.path.join(models_dir, MODEL_ITERATION)

    # load model
    model = DQN.load(model_path, env = env)
    # model.policy.set_training_mode(False)
    # env = model.get_env()
    

    for idx, ep in tqdm(test_set.iterrows(), total = len(test_set)):

        obs, info = env.reset(
            options = {
                'test_sample_parameters': {
                    'start_node': poiid2idx[ep['start_node']],
                    'time_constraint': ep['time_constraint'],
                    'cat_prefs': ep['cat_prefs_binary'],
                    'request_id': ep['req_id']
                }
            }
        )
        done = False
        truncated = False
        cum_reward = None
        rewards_list = []
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            rewards_list.append(reward)
            if cum_reward is None:
                cum_reward = reward
            else:
                cum_reward += reward

            if done or truncated:
                env.render()
                # env.render()
                time.sleep(1)

                # store metrics after removing padding from route
                # PADDING_TOKEN = MAX_CITY_GRAPH_NODES # max(POI_graph.nodes) + 1
                # filtered_route_nodes = [i for i in obs['route_nodes'] if i != PADDING_TOKEN]
                # route = info.get('route', [poiid2idx[ep['start_node']]])
                route = info.get('route', [poiid2idx[ep['start_node']]]) # info['route']
                results_df.loc[idx] = [
                    ep['req_id'],
                    poiid2idx[ep['start_node']],
                    ep['time_constraint'],
                    route,
                    cum_reward,
                    rewards_list
                    # filtered_route_nodes
                ]
        
                
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'episode_results.csv'))
             

