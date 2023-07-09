import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

import json
from itertools import repeat
from common_functions_h01 import make_skel_graphs_batch, assign_root_and_bb_nodes, classify_syn_remove_stalks
import pickle
from multiprocessing import Pool



credentials_file = 'alexshapsoncoe.json' # or your credentials file
input_file = 'random_sample_of_10000e_or_i_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_0ax_list.json' # List of segment IDs to be processed
skel_dir = '20200916c3_skeletons_6class_plus_myelin' # available at  gs://h01_paper_public_files/20200916c3_skeletons_6class_plus_myelin       -  Skeleton data for all segments in the segmentation data
save_dir = 'random_sample_of_10000e_or_i_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_pruned' # Name for folder where you would like to save the results - results avaialble at gs://h01_paper_public_files/random_sample_of_10000e_or_i_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_pruned
chosen_parameters_file = 'axon_skeleton_pruning_parameters.json'
synapse_classification_model = 'axon_synapse_classification_model.pkl'
shard_divisor = 42356404
skel_voxel_size = [32,32,33]
synapse_voxel_size = [8,8,33]
syn_db_name = 'goog14r0s5c3.synapse_c3_eirepredict_clean_dedup'
neurite_type = 'axon'
nodes_or_lengths = 'nodes'
cpu_num = 10

save_dir = f'{working_dir}/{save_dir}'

def do_one_dir(i, segs_to_load, skel_dir, skel_voxel_size, credentials_file, neurite_type, syn_db_name, synapse_voxel_size, save_dir, chosen_params, nodes_or_lengths, clf):
    
    if not os.path.exists(f'{save_dir}/{i}'):
        os.mkdir(f'{save_dir}/{i}')

    segs_to_load = [x for x in segs_to_load if f'{x}.gml' not in os.listdir(f'{save_dir}/{i}')]
    
    skel_graphs = make_skel_graphs_batch(i, segs_to_load, skel_dir, skel_voxel_size, credentials_file, neurite_type, syn_db_name, synapse_voxel_size)

    for seg_id in skel_graphs.keys():
        
        g = assign_root_and_bb_nodes(
            skel_graphs[seg_id], 
            chosen_params['max_stalk_len'], 
            chosen_params['lower_min'], 
            chosen_params['higher_min'], 
            chosen_params['min_branch_len'], 
            nodes_or_lengths
            )
        
        # Use trained model to turn stalk nodes of en-passant synapses back to 'c' and re-assign ep synapses to bb:
        g = classify_syn_remove_stalks(g, clf)
        
        # Save dists only:
        dists = []

        for n in g.vs:
            
            if 's' in n['ntype']:

                dists.append(tuple([float(n['eucdistshaft']), float(n['eucdistroot']), float(n['stalkdist']), n['stype']]))

        this_seg_dists = tuple([seg_id, tuple(dists)])
        
        with open(f'{save_dir}/{i}/{seg_id}_dists.json', 'w') as fp:
            json.dump(this_seg_dists, fp)
        
        # Then save as gml file:
        g.write_gml(f'{save_dir}/{i}/{seg_id}.gml')
        
   



if __name__ == '__main__':


    if not os.path.exists(f'{save_dir}'):
        os.mkdir(f'{save_dir}')

    credentials_file = f'{working_dir}/{credentials_file}'
    skel_dir = f'{working_dir}/{skel_dir}'

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





