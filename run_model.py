'''
This script is used to train a reinforcement learning model on a city environment using start nodes in train set.
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
from constants import FINAL_CATEGORIES
#from model_with_one_reward import Route, CityEnv
# from model_return_to_start import Route, CityEnv
from direct_env import CityEnv

SEED = 100
#%%
if __name__ == '__main__':
    #%%
    np.random.seed(SEED)

    EXP_NAME = 'exp1'
    CITY = 'bonn'
    DATA_DIR = f'./data/{CITY}'
    IMPL_DIR = './content/'
    OUTPUT_DIR = f'./results/{EXP_NAME}'
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    
    #---------------------------------------------------------------------------------------#
    # read data (start nodes) of training dataset
    TRAIN_SET_PATH = f'./data/{CITY}/saved_data/train_set.csv'
    # dataframe with route requests of test set
    train_set = pd.read_csv(
        TRAIN_SET_PATH
    )
    print(f'test_set.columns {train_set.columns}, test_set.shape {train_set.shape}')
    train_set_start_node_osmids = np.unique(train_set['start_node'].values)
    print(f'len(train_set_start_node_osmids) {len(train_set_start_node_osmids)}')

    #%%
    # read city data i.e walking network, poi graph, distance and bearing matrix, etc.
    (
        final_pois, Bonn_walking_network, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    ) = read_data(DATA_DIR)
    #%%
    # replacing start node pois with the pois in the train set
    # start_node_pois_within_radius = train_set_start_node_osmids
    #%%
    (
        POI_graph, final_pois, distance_matrix,
        unfiltered_distance_matrix, start_node_pois_within_radius, train_set_start_node_osmids
    ) = rescale_inputs(
        POI_graph,
        final_pois,
        poiid2idx,
        distance_matrix,
        unfiltered_distance_matrix,
        start_node_pois_within_radius,
        train_set_start_node_osmids
    )

    # now nans are present instead of np.inf
    # distance_matrix, unfiltered_distance_matrix = normalise_inputs(distance_matrix, unfiltered_distance_matrix) # now nans are present instewad of np.inf
    # Note: Only normalising distance_matrix here.
    # 'unfiltered_distance_matrix' is not normalised, it has actual distance values which will be subtracted from dist. budget in the model 
    distance_matrix = normalise_inputs(distance_matrix)
    

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
        entry_point= CityEnv
    )

    # make gym environment
    env = gym.make(
        'gymnasium_env/CityEnv-v0',
        city_graph = POI_graph,
        poi_limit = 20,
        all_start_nodes = start_node_pois_within_radius,
        bearing_matrix = bearing_matrix,
        poiid2idx = poiid2idx2,
        final_pois_gdf = final_pois,
        unfiltered_distance_matrix = unfiltered_distance_matrix,
        output_dir = OUTPUT_DIR,
        train_samples = {
            'start_node_ids': train_set_start_node_osmids,
            'time_constraint': [2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        current_mode = 'train',
        max_city_graph_nodes = 500,
        alpha_params_dict = {
            'temporal_distance': 1,# 0.33,
            'diversity': 0.33, # 0.33,
            'coverage': 0.33, # 0.33
            'cat_prefs': 0.33
        }
    )

    env = gym.wrappers.TimeLimit(env, max_episode_steps = 100)

    print('checking env with gym and stable baselines............')
    gym.utils.env_checker.check_env(env.unwrapped)
    check_env(env)

    ##
    # Testing environment with random actions
    ##
    obs = env.reset()[0]

    # Take some random actions
    for i in range(10):

        rand_action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(rand_action)
        # print(f"Obs: {obs}")
        # print(f"Reward: {reward}")

        if terminated or truncated:
            if (i%2 == 0):
                env.render()
            env.reset()

    #---------------------------------------------------------------------------------------#

    # Train model
    print('Train model')
    model_name = EXP_NAME #"exp1"
    LOG_TIMESTEPS = 10000
    TIMESTEPS = 1000000
    models_dir = os.path.join(IMPL_DIR, 'models', model_name)
    log_dir = os.path.join(IMPL_DIR, 'logs')
    print(f'{model_name}, {models_dir}, {log_dir}')
    print(f'TIMESTEPS {TIMESTEPS}, LOG_TIMESTEPS {LOG_TIMESTEPS}')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env.reset()
    model = DQN(
        'MultiInputPolicy',
        env,
        verbose = 1,
        gamma = 0.99,  
        learning_rate = 0.0005,
        buffer_size = 100000,
        batch_size = 64,
        train_freq = 4, 
        exploration_initial_eps = 1.0,
        exploration_final_eps = 0.01, 
        exploration_fraction = 0.1,
        tensorboard_log = log_dir
    )

    for i in tqdm(range(1, int(TIMESTEPS/LOG_TIMESTEPS))):
        model.learn(total_timesteps = LOG_TIMESTEPS, reset_num_timesteps = False, tb_log_name = f"{model_name}")
        model.save(f"{models_dir}/{LOG_TIMESTEPS*i}")
    print('Training complete.......')



    




    







    


# %%
