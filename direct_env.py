import numpy as np
from numba import njit
from scipy.special import softmax
from scipy.spatial import distance as scp_dist
import pandas as pd
import gymnasium as gym
import folium
import copy

import os
import time
import json

from constants import FINAL_CATEGORIES


SEED = 100
np.random.seed(SEED)
INCLUDE_KEYS = [
    'element',
    'id',
    'addr:city',
    'addr:country',
    'amenity',
    'name',
    'tourism'
]

@njit
def score_candidates(distances, diversity_deltas, coverage_deltas, cat_pref_scores, alpha_diversity, alpha_distance, alpha_coverage, alpha_cat_pref):
    
    out = np.empty_like(distances)

    for i in range(len(distances)):
        out[i] = (
            (alpha_diversity * diversity_deltas[i]) + (alpha_coverage * coverage_deltas[i]) + (alpha_cat_pref * cat_pref_scores[i])
        )
    return out

@njit
def calculate_turn_angle_based_penalty(bearing1, bearing2):
    turn_angle = abs(bearing2 - bearing1)
    penalty = turn_angle/360
        
    return -1 * penalty

@njit
def normalise_value(value, max_value):
    return value / max_value
###


def total_intra_list_distance(vectors):
    packed = [int(''.join('1' if x else '0' for x in v), 2) for v in vectors]
    total = 0
    n = len(packed)
    for i in range(n):
        for j in range(i+1, n):
            total += (packed[i] ^ packed[j]).bit_count()
    return total

def compute_ild(vectors):
    '''
    Normalised ILD
    '''

    n = len(vectors)
    tot = total_intra_list_distance(vectors)
    # number of distinct pairs = n*(n-1)/2
    return tot / (n*(n-1)*len(FINAL_CATEGORIES[4:])/2)

###

class Route:

    def __init__(
        self,
        start_node,
        end_node,
        walking_speed = 5 #5km/h
    ):
        
        self.walking_speed = walking_speed
        self.start_node = start_node
        self.end_node = end_node
        self.route = [self.start_node] # Route is a list of nodes
        self.time_elapsed = 0
        self.distance_elapsed = 0


    def __repr__(self):
        print(f"Start Node: {self.start_node}")
        print(f"End Node: {self.end_node}")
        print(f"Route: {self.route}")
        print(f"Time Elapsed: {self.time_elapsed}")
        print(f"Distance Elapsed: {self.distance_elapsed}")

        return str(self.route)


    def insert_node(self, node, distance_of_new_node, visit_time_of_new_node):
        '''
        Insert a node in route and update time_elapsed, distance_elapsed
        '''

        self.distance_elapsed += distance_of_new_node
        self.time_elapsed += (distance_of_new_node/self.walking_speed) + visit_time_of_new_node

        self.route.append(node)



    def remove_node(self, distance_of_removed_node, visit_time_of_removed_node):
        removed_node = self.route.pop(-1)
        self.distance_elapsed -= distance_of_removed_node
        self.time_elapsed -= (distance_of_removed_node/self.walking_speed) + visit_time_of_removed_node
        return removed_node


class CityEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        city_graph = None,
        poi_limit = None,
        all_start_nodes = None,
        bearing_matrix = None,
        poiid2idx = None,
        final_pois_gdf = None,
        unfiltered_distance_matrix = None,
        output_dir = None,
        train_samples = None,
        current_mode = None,
        max_city_graph_nodes = None,
        candidate_poi_generator_k = 3,
        alpha_params_dict = {
            'temporal_distance': 1,# 0.33,
            'diversity': 0.33,
            'coverage': 0.33, 
            'cat_prefs': 0.33
        },
        render_mode = None
    ):
        self.walking_speed = 5 #5km/h
        self.distance_threshold = 7.5 # 7.5 km
        self.all_start_nodes = all_start_nodes # list of all start nodes for masking
        self.current_mode = current_mode
        self.train_samples = train_samples
        self.original_graph = city_graph
        self.city_graph = copy.deepcopy(city_graph) # original graph with start nodes
        # remove all start nodes from city graph, and then insert each start node in reset() and remove the start node on termination
        self.city_graph.remove_nodes_from(self.all_start_nodes)
        self.max_city_graph_nodes = max_city_graph_nodes
        self.nodes_count = self.city_graph.number_of_nodes()
        self.edges_count = self.city_graph.number_of_edges()
        self.request_id = None
        
        
        # self.start_node_osmids = start_node_osmids
        self.poi_limit	= poi_limit
        # self.distance_matrix = distance_matrix
        self.bearing_matrix = bearing_matrix
        self.unfiltered_distance_matrix = unfiltered_distance_matrix
        self.poiid2idx = poiid2idx
        self.final_pois_gdf = final_pois_gdf
        self.idx_to_tourism_category = {
            row['_osm_id'] : np.array(row['tourism_category'][4:], dtype = np.float32)
            for _, row in self.final_pois_gdf.iterrows()
        }
        self.tourism_category_arr = np.zeros((len(self.poiid2idx), len(FINAL_CATEGORIES[4:])), dtype=np.float32)
        for poiid, idx in self.poiid2idx.items():
            self.tourism_category_arr[idx] = self.idx_to_tourism_category[idx]

        self.sorted_neighbors_dict = {}
        for poiid, idx in self.poiid2idx.items():
            row = self.unfiltered_distance_matrix[idx]
            nbrs = list(self.original_graph.neighbors(poiid))
            self.sorted_neighbors_dict[idx] = sorted(
                ((nbr, row[self.poiid2idx[nbr]]) for nbr in nbrs),
                key=lambda x: x[1]
            )
        self.current_graph_nodes = None

        self.route_instance = None
        self.distance_from_end_node = None
        self.constraints_dict = None

        self.terminated = False
        self.reward = 0
        self.output_dir = output_dir

        self.map = None

        self.candidate_poi_generator_k = candidate_poi_generator_k
        self.alpha_params_dict = alpha_params_dict
        self.rejected_nodes = None
        self.rejected_nodes_from_last_insertion = None
        self.previously_inserted_pois_tracker = set()
        self.selected_neighbors = None
        self.neighbor_nodes_scores = None
        self.all_neighbors_sorted_by_distances = None
        self.route_diversity = None
        self.route_poi_geohashes_list = None
        self.idx_to_geohash = {
            row['_osm_id'] : row['geohash']
            for _, row in self.final_pois_gdf.iterrows()
        }
        self.selected_neighbors_dist_benefit = None
        self.selected_neighbors_time_benefit = None
        self.selected_neighbors_turn_angle = None

        self.invalid_action_flag = False

        self.action_space = gym.spaces.Discrete(5 + 2) # insert, delete, return to start node

        self.observation_space = gym.spaces.Dict(
            {
                'route_nodes': gym.spaces.Box(low=min(self.city_graph.nodes), high=self.max_city_graph_nodes+1, shape=(self.poi_limit,), dtype= np.int16), #high=max(self.city_graph.nodes)+1
                'distance_elapsed': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # max distance of route is 15 km (10km for 2hrs)
                'time_elapsed': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # max duration of route is 10 hrs
                'distance_from_end_node': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'temporal_distance_from_end_node': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'poi_count': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                'selected_neighbors_dist_benefit': gym.spaces.Box(low=-1, high=1, shape=(self.candidate_poi_generator_k,), dtype= np.float32),
                'selected_neighbors_time_benefit': gym.spaces.Box(low=-1, high=1, shape=(self.candidate_poi_generator_k,), dtype= np.float32),
                'selected_neighbors_turn_angle': gym.spaces.Box(low =-1, high=0, shape=(self.candidate_poi_generator_k,), dtype=np.float32)
            }
        )

    
    def _get_obs(self):
        '''
        Convert environment's state into observation and
        return the current observation
        '''
        
        # pad route for route_nodes
        route = self.route_instance.route
        if len(route) < self.poi_limit:
            PAD_NODE_ID = self.max_city_graph_nodes # max(self.city_graph.nodes) + 1
            route = route + [PAD_NODE_ID for _ in range(self.poi_limit - len(route))]

        # pad neighbor node scores
        neighbor_nodes_scores = self.neighbor_nodes_scores
        if len(neighbor_nodes_scores) < self.candidate_poi_generator_k:
            neighbor_nodes_scores = np.append(neighbor_nodes_scores, [0 for _ in range(self.candidate_poi_generator_k - len(self.neighbor_nodes_scores))]).astype(np.float32)
        
        # pad neighbors dist and time benefit
        selected_neighbors_dist_benefit = self.selected_neighbors_dist_benefit
        if len(selected_neighbors_dist_benefit) < self.candidate_poi_generator_k:
            selected_neighbors_dist_benefit = np.append(selected_neighbors_dist_benefit, [-1 for _ in range(self.candidate_poi_generator_k - len(self.selected_neighbors_dist_benefit))]).astype(np.float32)
        selected_neighbors_time_benefit = self.selected_neighbors_time_benefit
        if len(selected_neighbors_time_benefit) < self.candidate_poi_generator_k:
            selected_neighbors_time_benefit = np.append(selected_neighbors_time_benefit, [-1 for _ in range(self.candidate_poi_generator_k - len(self.selected_neighbors_time_benefit))]).astype(np.float32)
        selected_neighbors_turn_angle = self.selected_neighbors_turn_angle
        if len(selected_neighbors_turn_angle) < self.candidate_poi_generator_k:
            selected_neighbors_turn_angle = np.append(selected_neighbors_turn_angle, [-1 for _ in range(self.candidate_poi_generator_k - len(self.selected_neighbors_turn_angle))]).astype(np.float32)
        
        # normalise distance_elapsed and distance_from_end_node
        _distance_elapsed = np.clip(normalise_value(self.route_instance.distance_elapsed, self.constraints_dict['distance_constraint']),
        0, 1
        )

        _time_elapsed = np.clip(normalise_value(self.route_instance.time_elapsed, self.constraints_dict['time_constraint']),
        0, 1
        )
        remaining_distance_budget = self.constraints_dict['distance_constraint'] - self.route_instance.distance_elapsed
        remaining_time_budget = self.constraints_dict['time_constraint'] - self.route_instance.time_elapsed
        _distance_from_end_node = np.clip(
            normalise_value(remaining_distance_budget - self.distance_from_end_node, remaining_distance_budget),
            -1, 1
        )
        _temporal_distance_from_end_node = np.clip(
            normalise_value(remaining_time_budget - (self.distance_from_end_node / self.walking_speed), remaining_time_budget),
            -1, 1
        )


        # normalise poi count
        _poi_count = normalise_value(len(self.route_instance.route), self.poi_limit)

        obs = {
            'route_nodes': np.array([np.array(x) for x in route], dtype = np.int16),
            'distance_elapsed': np.array([_distance_elapsed], dtype=np.float32),
            'time_elapsed': np.array([_time_elapsed], dtype=np.float32),
            'distance_from_end_node': np.array([_distance_from_end_node], dtype=np.float32),
            'temporal_distance_from_end_node': np.array([_temporal_distance_from_end_node], dtype=np.float32),
            'poi_count': np.array([_poi_count], dtype=np.int8),
            'selected_neighbors_dist_benefit': selected_neighbors_dist_benefit,
            'selected_neighbors_time_benefit': selected_neighbors_time_benefit,
            'selected_neighbors_turn_angle': selected_neighbors_turn_angle
        }
        

        return obs



    def reset(self, seed = None, options = {'test_sample_parameters': None}):
        '''
        Resets the environment
        '''

        super().reset(seed = seed)
        if seed:
            np.random.seed(seed)

        self.previously_inserted_pois_tracker = set()

        # initialise route instance
        start_node = 1 # None
        time_constraint = 2 # None
        cat_prefs = None
        cat_prefs_binary = None
        if options:
            if options.get('test_sample_parameters', None):
                start_node = options['test_sample_parameters']['start_node']
                time_constraint = options['test_sample_parameters']['time_constraint']
                self.request_id = options['test_sample_parameters']['request_id']
                cat_prefs = options['test_sample_parameters']['cat_prefs']

        elif self.current_mode == 'train':
            # start_node = np.random.choice(self.start_node_osmids)
            # time_constraint = np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
            cat_prefs = np.random.choice(FINAL_CATEGORIES[4:] + [None], 3)
            cat_prefs_binary = None if cat_prefs is None else np.array([1 if i in cat_prefs else 0 for i in FINAL_CATEGORIES[4:]], dtype = np.float32)
            start_node = np.random.choice(self.train_samples['start_node_ids'])
            time_constraint = np.random.choice(self.train_samples['time_constraint'])
            
        end_node = start_node 

        
        # insert start node and its attributes and edges in city graph
        self.city_graph.add_node(start_node, **self.original_graph.nodes[start_node])
        self.city_graph.add_edges_from((start_node, nbr, self.original_graph[start_node][nbr]) for nbr in self.original_graph.neighbors(start_node) if nbr not in self.all_start_nodes)
        self.current_graph_nodes = set(self.city_graph.nodes())

        # walkable distance
        distance_constraint = 10 if time_constraint <= 2 else 15

        self.constraints_dict = {
            'time_constraint': time_constraint,
            'distance_constraint': distance_constraint,
            'poi_count_limit': self.poi_limit,
            'cat_prefs': cat_prefs_binary
        }
        
        self.route_instance = Route(
            start_node,
            end_node,
            self.walking_speed
        )
        self.distance_from_end_node = 0
        self.rejected_nodes = set([])
        self.route_poi_geohashes_list = [self.idx_to_geohash[start_node]]

        self.route_diversity = self.compute_route_ild(self.route_instance.route)
        (
            self.selected_neighbors,
            self.neighbor_nodes_scores,
            self.all_neighbors_sorted_by_distances,
            self.selected_neighbors_dist_benefit,
            self.selected_neighbors_time_benefit,
            self.selected_neighbors_turn_angle
        )= self.candidate_poi_generator(self.candidate_poi_generator_k)

        observation = self._get_obs()


        # initialise map with all pois for visualisation
        #(lat, lon)
        start_node_coordinates = self.final_pois_gdf[self.final_pois_gdf['_osm_id'] == self.route_instance.start_node]['plotting_coords'].values[0]
        self.map = folium.Map(
            location=[start_node_coordinates[0], start_node_coordinates[1]], zoom_start=20
        )

        self.terminated = False
        self.reward = 0

        info = {
            'start_node': start_node,
            'time_constraint': time_constraint,
            'distance_constraint': distance_constraint,
            'cat_prefs': cat_prefs_binary,
            'route': self.route_instance.route
        }
        return observation, info
        
    
    def candidate_poi_generator(self, k):
        '''
        Generates POI candidates for insert operation
        '''

        request_cat_prefs = self.constraints_dict.get('cat_prefs', None)

        current_node = self.route_instance.route[-1]
        start_node = self.route_instance.start_node
        # current_node_neighbors = list(self.city_graph.neighbors(current_node))
        current_node_neighbors = self.sorted_neighbors_dict[current_node]
        remaining_distance_budget = self.constraints_dict['distance_constraint'] - self.route_instance.distance_elapsed
        remaining_time_budget = self.constraints_dict['time_constraint'] - self.route_instance.time_elapsed
        

        # Filter: remove previously traversed nodes and self.rejected_nodes from neighbors
        route_set = set(self.route_instance.route)
        route_geohash_set = set(self.route_poi_geohashes_list)
        all_neighbors_sorted_by_distances = [
           (nbr, dist)
           for nbr, dist in current_node_neighbors
           if (nbr in self.current_graph_nodes and nbr not in route_set and nbr not in self.rejected_nodes)
        ]
        current_node_neighbors = [
            nbr for nbr, _ in all_neighbors_sorted_by_distances
        ]

        # if len(self.route_instance.route) > 1:
        #     current_node_neighbors = [
        #         neigh for neigh in list(self.city_graph.neighbors(current_node)) if (neigh not in self.route_instance.route)
        #     ]
        #     if len(self.rejected_nodes) > 0:
        #         current_node_neighbors = [
        #             neigh for neigh in current_node_neighbors if all(neigh != tup for tup in self.rejected_nodes)
        #         ]
        # if len(self.route_instance.route) > 1:
        #     # Precompute sets for O(1) membership tests
        #     route_set = set(self.route_instance.route)
        #     rejected_set = self.rejected_nodes

        #     # Grab neighbors once, filter out both route & rejected in one comprehension
        #     current_node_neighbors = [
        #         neigh
        #         for neigh in self.city_graph.neighbors(current_node)
        #         if neigh not in route_set and neigh not in rejected_set
        #     ]


        # also passing all neighbor nodes sorted by distance
        # all_neighbors_sorted_by_distances = sorted(
        #     [
        #     (s, self.unfiltered_distance_matrix[self.poiid2idx[current_node]][self.poiid2idx[s]]) for s in current_node_neighbors
        #     ],
        #     key = lambda x:x[1]
        # )

        if len(current_node_neighbors) == 0:
            return [], np.array([], dtype=np.float32), all_neighbors_sorted_by_distances, np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)



        # Candidate Scoring
        current_node_neighbors_scores = []
        current_node_neighbors_idx_dict = dict.fromkeys(current_node_neighbors, None)
        distances = []
        diversity_deltas = []
        coverage_deltas = []
        cat_pref_scores = []

        # benefit of visiting neighbor and returning to start
        neighbors_distance_benefit = []
        neighbors_time_benefit = []
        neighbors_turn_angle = []
        route_with_two_or_more_nodes = len(self.route_instance.route) > 1
        if route_with_two_or_more_nodes:
            poi1 = self.poiid2idx[self.route_instance.route[-2]]
            poi2 = self.poiid2idx[self.route_instance.route[-1]]

        for idx, neighbor in enumerate(current_node_neighbors):

            ##
            # Category diversity score
            ##
            # diversity_after_neighbor_insertion = self.compute_alpha_ndcg(self.route_instance.route + [neighbor])
            # diversity_deltas.append(0.5 * ((diversity_after_neighbor_insertion - self.route_diversity) + 1))
            diversity_after_neighbor_insertion = self.compute_route_ild(self.route_instance.route + [neighbor])
            diversity_deltas.append(
                0.5 * (1 + (diversity_after_neighbor_insertion - self.route_diversity))
            )


            # current_node_neighbors_scores[neighbor] = diversity_delta

            ##
            # Coverage score
            ##
            coverage_deltas.append(1 - int(self.idx_to_geohash[neighbor] in route_geohash_set))

            ##
            # Temporal Distance score (walking distance + visiting time)
            ##
            neighbor_visit_time = self.city_graph.nodes[neighbor].get('min_visit_time', 0)
            neighbor_distance = self.unfiltered_distance_matrix[self.poiid2idx[current_node]][self.poiid2idx[neighbor]]
            distances.append((neighbor_distance/self.walking_speed) + neighbor_visit_time)
            
            ##
            # Benefit of visiting neighbor (for returning in observation space)
            ##
            neighbor_distance_cost = (
                self.unfiltered_distance_matrix[self.poiid2idx[current_node]][self.poiid2idx[neighbor]]
                +
                self.unfiltered_distance_matrix[self.poiid2idx[neighbor]][self.poiid2idx[start_node]] 
            )
            neighbor_distance_benefit = np.clip(
                normalise_value(remaining_distance_budget - neighbor_distance_cost, remaining_distance_budget),
                -1, 1
            )
            neighbors_distance_benefit.append(neighbor_distance_benefit)
            neighbor_time_cost = (neighbor_distance_cost/self.walking_speed) + neighbor_visit_time
            neighbor_time_benefit = np.clip(
                normalise_value(remaining_time_budget - neighbor_time_cost, remaining_time_budget),
                -1, 1
            )
            neighbors_time_benefit.append(neighbor_time_benefit)

            # turn angle benefit
            if route_with_two_or_more_nodes:
                poi3 = self.poiid2idx[neighbor]
                neighbors_turn_angle.append(calculate_turn_angle_based_penalty(
                    self.bearing_matrix[poi1][poi2],
                    self.bearing_matrix[poi2][poi3]
                ))
            else:
                neighbors_turn_angle.append(0)

            ##
            # Category Prefs
            ##
            cat_pref_score = 0
            if not (request_cat_prefs is None) and sum(request_cat_prefs) > 0 and sum(self.tourism_category_arr[neighbor]) > 0:
                #cat_pref_score = 1 if np.any(np.logical_and(self.tourism_category_arr[neighbor], request_cat_prefs)) else 0
                cat_pref_score = 1 - scp_dist.dice(self.tourism_category_arr[neighbor], request_cat_prefs)
            cat_pref_scores.append(cat_pref_score)


            current_node_neighbors_idx_dict[neighbor] = idx 

        current_node_neighbors_scores = score_candidates(
            np.array(distances, dtype = np.float64),
            np.array(diversity_deltas, dtype = np.float64),
            np.array(coverage_deltas),
            np.array(cat_pref_scores, dtype = np.float64),
            np.float32(self.alpha_params_dict['diversity']),
            np.float32(self.alpha_params_dict['temporal_distance']),
            np.float32(self.alpha_params_dict['coverage']),
            np.float32(self.alpha_params_dict['cat_prefs'])
        )

        # Neighbor scores to probabilities
        if np.isnan(cat_pref_scores).any():
            print(cat_pref_scores)
            print(request_cat_prefs)
        current_node_neighbors_probs = softmax(current_node_neighbors_scores)
        
        # Sample k neighbors based on probabilities
        non_zero_probs = len(current_node_neighbors_probs[current_node_neighbors_probs > 0])
        selected_neighbors = np.random.choice(
            current_node_neighbors,
            size = k if non_zero_probs >= k else non_zero_probs,
            replace = False,
            p = current_node_neighbors_probs
        )
        selected_neighbor_probs = np.array(
            [current_node_neighbors_probs[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )

        selected_neighbors_dist_benefit = np.array(
            [neighbors_distance_benefit[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )
        selected_neighbors_time_benefit = np.array(
            [neighbors_time_benefit[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )
        selected_neighbors_turn_angle = np.array(
            [neighbors_turn_angle[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )


        return selected_neighbors, selected_neighbor_probs, all_neighbors_sorted_by_distances, selected_neighbors_dist_benefit, selected_neighbors_time_benefit, selected_neighbors_turn_angle


    
    def compute_route_ild(self, route):
        '''
        Remove start node and compute ild of route
        '''
        ild = 0
        if len(route) > 2:
            _route = route[1:]
            cats = self.tourism_category_arr[_route] 
            ild = compute_ild(cats)
        return ild




    def perform_action(self, action):

        action_return_tuple = None
        current_node = self.route_instance.route[-1]


        # Action value in [0, 4] and node has neighbors: Insert node in route
        # Note: Any one of the top 5 neighbors can be inserted
        if action in range(5) and action < len(self.selected_neighbors):
            self.invalid_action_flag = False
            
            inserted_node = self.selected_neighbors[action]
            distance_of_inserted_node = None
            new_rejected_nodes = []
            for neigh, dist in self.all_neighbors_sorted_by_distances:
                if neigh == inserted_node:
                    distance_of_inserted_node = dist
                    break
                else:
                    new_rejected_nodes.append(neigh)
            self.route_instance.insert_node(
                inserted_node, # inserted node
                distance_of_inserted_node, # distance of inserted node from current node
                self.city_graph.nodes[inserted_node].get('min_visit_time', 0) # visit duration
            )
            self.distance_from_end_node = self.unfiltered_distance_matrix[self.poiid2idx[self.route_instance.route[-1]]][self.poiid2idx[self.route_instance.end_node]]

            # insert nodes with distance less than selected node as rejected nodes
            self.rejected_nodes.update(new_rejected_nodes)
            self.rejected_nodes_from_last_insertion = new_rejected_nodes

            # update route_poi_geohashes_list, selected neighbors, scores, route diversity
            self.route_poi_geohashes_list.append(self.idx_to_geohash[inserted_node])
            inserted_node_score = self.neighbor_nodes_scores[action]
            # self.route_diversity = self.compute_alpha_ndcg(self.route_instance.route)
            self.route_diversity = self.compute_route_ild(self.route_instance.route)

            # generate POI candidates
            (
                self.selected_neighbors,
                self.neighbor_nodes_scores,
                self.all_neighbors_sorted_by_distances,
                self.selected_neighbors_dist_benefit,
                self.selected_neighbors_time_benefit,
                self.selected_neighbors_turn_angle 
            )= self.candidate_poi_generator(self.candidate_poi_generator_k)

            action_return_tuple = ('insert', inserted_node, inserted_node_score)


        # Remove last node from route if route length > 1
        elif action == 5 and (len(self.route_instance.route) > 1):
            self.invalid_action_flag = False

            # print('ACTION: REMOVE LAST NODE')
            removed_node = self.route_instance.remove_node(
                self.unfiltered_distance_matrix[self.poiid2idx[self.route_instance.route[-2]]][self.poiid2idx[self.route_instance.route[-1]]],
                self.city_graph.nodes[self.route_instance.route[-1]].get('min_visit_time', 0)
            )
            self.rejected_nodes -= set(self.rejected_nodes_from_last_insertion)
            self.rejected_nodes_from_last_insertion = []
            self.distance_from_end_node = self.unfiltered_distance_matrix[self.poiid2idx[self.route_instance.route[-1]]][self.poiid2idx[self.route_instance.end_node]]
            self.route_poi_geohashes_list.pop(-1)

            # update neighbor node scores
            # self.route_diversity = self.compute_alpha_ndcg(self.route_instance.route)
            self.route_diversity = self.compute_route_ild(self.route_instance.route)
            (
                self.selected_neighbors,
                self.neighbor_nodes_scores,
                self.all_neighbors_sorted_by_distances,
                self.selected_neighbors_dist_benefit,
                self.selected_neighbors_time_benefit,
                self.selected_neighbors_turn_angle
            )= self.candidate_poi_generator(self.candidate_poi_generator_k)
            action_return_tuple = ('remove', removed_node)

        # Action value 2: Go to end node
        elif action == 6 and (len(self.route_instance.route) > 1):
            self.invalid_action_flag = False
            # print('ACTION: GO TO END NODE')
            action_return_tuple = ('end_node')
            distance_from_current_node_to_end_node = self.unfiltered_distance_matrix[self.poiid2idx[current_node]][self.poiid2idx[self.route_instance.end_node]]
            self.route_instance.insert_node(
                self.route_instance.end_node,
                distance_from_current_node_to_end_node,
                0
            )
            self.distance_from_end_node = 0
        else:
            self.invalid_action_flag = True
        # else:
        #     print('!!!Invalid Action', type(action), action)

        return action_return_tuple
    
    

    def get_reward(self, action_return_tuple):

        if self.invalid_action_flag:
            return -100

        reward = 0
        reward_components_counter = 0
        # after using 1/4 distance budget, add penalty for distance from end node and time constraint
        distance_constraint = self.constraints_dict['distance_constraint']
        time_constraint = self.constraints_dict['time_constraint']
        distance_based_go_to_end_node_weight = 1 if self.route_instance.distance_elapsed >= (distance_constraint/4) else 0
        time_based_go_to_end_node_weight = 1 if self.route_instance.time_elapsed >= (time_constraint/4) else 0


        if not self.terminated:


            ##
            # Component A1: Reward for distance elapsed
            ##
            if distance_based_go_to_end_node_weight:

                reward -= self.distance_from_end_node
                reward_components_counter += 1

            ##
            # Component A2: Reward for time elapsed
            ##
            if time_based_go_to_end_node_weight:
                reward -= (self.distance_from_end_node/self.walking_speed)
                reward_components_counter += 1
                

            ##
            # Component B: Reward for insertion
            # Rewards/penalties on insertion
            ##
            if action_return_tuple and action_return_tuple[0] == 'insert':
                # +1 point for each inserted node
                inserted_node_score = 1
                if action_return_tuple[1] not in self.previously_inserted_pois_tracker:
                    self.previously_inserted_pois_tracker.add(action_return_tuple[1])
                    reward += inserted_node_score
                
                # based on bearing
                if len(self.route_instance.route) > 2:
                    poi1 = self.poiid2idx[self.route_instance.route[-3]]
                    poi2 = self.poiid2idx[self.route_instance.route[-2]]
                    poi3 = self.poiid2idx[self.route_instance.route[-1]]
                    reward += calculate_turn_angle_based_penalty(
                        self.bearing_matrix[poi1][poi2],
                        self.bearing_matrix[poi2][poi3]
                    )
                    reward_components_counter += 1

        ##
        # On Termination
        ##
        else:
            # Component C: Reward for reaching end node
            reward_components_counter += 1
            if self.route_instance.end_node == self.route_instance.route[-1] and len(self.route_instance.route) > 1 and self.route_instance.distance_elapsed <= distance_constraint and self.route_instance.time_elapsed <= time_constraint:
                reward += (100 * (len(self.route_instance.route) - 2))
            else:
                reward -= 100

        self.reward = reward
        return self.reward
    

    def step(self, action):
        '''
        Perform action and compute state of environment
        '''

        action_return_tuple = self.perform_action(action)

        observation = self._get_obs()

        # Episode terminates when any of the below conditions are True:
        # 1. agent violates time or distance constraints
        # 2. no. of pois exceed poi limit
        # 3. agent reaches end_node
        terminated = False
        distance_constraint = self.constraints_dict['distance_constraint']
        time_constraint = self.constraints_dict['time_constraint']
        if (
            self.route_instance.distance_elapsed >= distance_constraint
        ) or (
            self.route_instance.time_elapsed >= time_constraint
        ) or (
            len(self.route_instance.route) >= self.poi_limit
        ) or (
            self.route_instance.end_node == self.route_instance.route[-1] and len(self.route_instance.route) > 1
        ) :
            terminated = True
            self.terminated = True
            self.city_graph.remove_node(self.route_instance.start_node)
            self.current_graph_nodes = []

        reward = self.get_reward(action_return_tuple)
        truncated = False

        info = {}
        if self.terminated:
            info = {
                'start_node': self.route_instance.start_node,
                'time_constraint': self.constraints_dict['time_constraint'],
                'distance_constraint': self.constraints_dict['distance_constraint'],
                'route': self.route_instance.route
            }

        return observation, reward, terminated, truncated, info
    


    def render(self):
        '''
        Render the environment
        '''

        identifier = self.request_id

        # render pois
        for idx, row in self.final_pois_gdf.iterrows():
            osm_id = row['_osm_id']

            if osm_id in self.route_instance.route:
                continue

            reduced_keys = []
            for key in row.keys():
                if key not in INCLUDE_KEYS:
                    continue

                if isinstance(row[key], list) and not pd.isna(row[key]).all():
                    reduced_keys.append(key)
                elif not pd.isna(row[key]):
                    reduced_keys.append(key)

            tooltip_string = ''.join([
                '<p><strong>'+str(key) + '</strong>'+ ':' + str(row[key]).translate(str.maketrans({'`': '', '´': ''})) + '</p>' for key in reduced_keys
            ])

            folium.Marker(
                location = [row['plotting_coords'][0],row['plotting_coords'][1]],
                icon = folium.Icon(color = 'red', icon = 'circle-dot', angle = 0, prefix = 'fa'),
                tooltip = tooltip_string
            ).add_to(self.map)

        # render route
        polyline_coords = []
        for route_poi_idx, osm_id in enumerate(self.route_instance.route):
            row = self.final_pois_gdf[self.final_pois_gdf['_osm_id'] == osm_id].iloc[0]

            reduced_keys = []
            for key in row.keys():
                if key not in INCLUDE_KEYS:
                    continue

                if isinstance(row[key], list) and not pd.isna(row[key]).all():
                    reduced_keys.append(key)
                elif not pd.isna(row[key]):
                    reduced_keys.append(key)

            # tooltip_string = f'<b>Route POI {route_poi_idx} </b> <br> Visit Time {0}'+'<br><br><br>'+''.join([
            #     '<b>'+str(key) + '</b>'+ ':' + str(row[key]) + '<br>' for key in reduced_keys
            # ])

            # remove characters which cause issues with html rendering
            tooltip_string = f'<p><strong>Route POI {route_poi_idx}</strong></p><p>Visit Time {0}'+'</p>'+''.join([
                '<p><strong>'+str(key) + '</strong>'+ ':' + str(row[key]).translate(str.maketrans({'`': '', '´': ''})) + '</p>' for key in reduced_keys
            ])

            folium.Marker(
                location = [row['plotting_coords'][0],row['plotting_coords'][1]],
                icon = folium.Icon(color = 'blue', icon = 'circle-dot', angle = 0, prefix = 'fa'),
                tooltip = tooltip_string
            ).add_to(self.map)

            polyline_coords.append([row['plotting_coords'][0], row['plotting_coords'][1]])
                
        folium.PolyLine(polyline_coords, color="blue", weight=2.5, opacity=1, popup='Connection').add_to(self.map)

        div_string = f'<div style="position: absolute; top: 10px; left: 10px; background-color: rgba(255, 255, 255, 0.7); padding: 10px; font-size: 16px; font-weight: bold; border-radius: 5px; z-index: 9999;">ID: {identifier}<br>Route Distance: {self.route_instance.distance_elapsed}<br>Time Elapsed: {self.route_instance.time_elapsed} <br> Time Constraint: {self.constraints_dict["time_constraint"]} <br> Distance Constraint: {self.constraints_dict["distance_constraint"]} <br> Reward: {self.reward} </div>'

        label = folium.Element(div_string)
        popup = folium.Popup(label, max_width=300)

        # Add the label to the map (use it as a custom overlay)
        self.map.get_root().html.add_child(folium.Element(div_string))

        output_file = f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time()))}.html'
        if self.output_dir:
            output_file = os.path.join(self.output_dir, output_file)
        self.map.save(output_file)

        
        return str(self.route_instance.route)
    





                
