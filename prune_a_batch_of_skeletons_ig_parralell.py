import json
import os
import numpy as np
from itertools import repeat
import common_functions as cf
import pickle
from multiprocessing import Pool
from google.cloud import bigquery             
from google.oauth2 import service_account
import igraph as ig

credentials_file = '/home/alexshapsoncoe/drive/alexshapsoncoe.json'
input_file = '/home/alexshapsoncoe/drive/axons_targeting_multi_bipolar_basal_d_noais_upper_bound_45_lower_bound_-45_phase2_bound_0.json'
skel_dir = '/home/alexshapsoncoe/drive/20200916c3_skeletons_6class_plus_myelin/'
save_dir = '/home/alexshapsoncoe/drive/axons_targeting_multi_bipolar_basal_d_noais_upper_bound_45_lower_bound_-45_phase2_bound_0'
chosen_parameters_file = '/home/alexshapsoncoe/drive/axon_skeleton_pruning_parameters.json'
synapse_classification_model = '/home/alexshapsoncoe/drive/axon_synapse_classification_model.pkl'
shard_divisor = 42356404
skel_voxel_size = [32,32,33]
synapse_voxel_size = [8,8,33]
syn_db_name = 'goog14r0s5c3.synaptic_connections_with_skeleton_classes'
neurite_type = 'axon'
nodes_or_lengths = 'nodes'
cpu_num = 10

def do_one_dir(i, segs_to_load, skel_dir, skel_voxel_size, credentials_file, neurite_type, syn_db_name, synapse_voxel_size, save_dir, chosen_params, nodes_or_lengths, clf):
    
    segs_to_load = [x for x in segs_to_load if f'{x}.gml' not in os.listdir(save_dir)]
    
    skel_graphs = cf.make_skel_graphs_batch(i, segs_to_load, skel_dir, skel_voxel_size, credentials_file, neurite_type, syn_db_name, synapse_voxel_size)

    for seg_id in skel_graphs.keys():
        
        g = cf.assign_root_and_bb_nodes(
            skel_graphs[seg_id], 
            chosen_params['max_stalk_len'], 
            chosen_params['lower_min'], 
            chosen_params['higher_min'], 
            chosen_params['min_branch_len'], 
            nodes_or_lengths
            )
        
        # Use trained model to turn stalk nodes of en-passant synapses back to 'c' and re-assign ep synapses to bb:
        g = cf.classify_syn_remove_stalks(g, clf)
        
        # Save dists only:
        dists = []

        for n in g.vs:
            
            if 's' in n['ntype']:

                dists.append(tuple([float(n['eucdistshaft']), float(n['eucdistroot']), float(n['stalkdist']), n['stype']]))

        this_seg_dists = tuple([seg_id, tuple(dists)])
        
        with open(f'{save_dir}/{seg_id}_dists.json', 'w') as fp:
            json.dump(this_seg_dists, fp)
        
        # Then save as gml file:
        g.write_gml(f'{save_dir}/{seg_id}.gml')
        
   



if __name__ == '__main__':

    with open(input_file, 'r') as fp:
        neurites_todo = json.load(fp)

    # Make skeleton graphs:
    pool = Pool(cpu_num)

    organized_neurites = [[] for i in range(10000)]

    for neurite in neurites_todo:
        organized_neurites[int(neurite)//shard_divisor].append(neurite)
        
    with open(chosen_parameters_file, 'r') as fp:
        chosen_params = json.load(fp)
        
    with open(synapse_classification_model, 'rb') as fp:
        clf = pickle.load(fp)

    args = zip(
        range(10000), 
        organized_neurites, 
        repeat(skel_dir), 
        repeat(skel_voxel_size), 
        repeat(credentials_file), 
        repeat(neurite_type), 
        repeat(syn_db_name),
        repeat(synapse_voxel_size),
        repeat(save_dir), 
        repeat(chosen_params), 
        repeat(nodes_or_lengths), 
        repeat(clf),
        )

    pool.starmap(do_one_dir, args)


    # Save and stalk and euclidean distances separately:

    all_distances = []

    for seg_id in neurites_todo:
        
        with open(f'{save_dir}/{seg_id}_dists.json', 'r') as fp:
            this_seg_dists = json.load(fp)

        all_distances.append(this_seg_dists)

    with open(f'{save_dir}/distance_measurements.json', 'w') as fp:
        json.dump(all_distances, fp)





