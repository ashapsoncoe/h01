from google.oauth2 import service_account
import json
import urllib
import zlib
from google.auth.transport import requests as auth_request
import os
import numpy as np
import math
import time
from random import sample, choices, uniform, choice, shuffle
import igraph as ig
import pickle
import common_functions as cf
from scipy import stats
from zipfile import ZipFile

neurites_todo_path = '/home/alexshapsoncoe/drive/random_sample_of_500000_axons_from_each_type_agg20200916c3_list_multisyn_only.json'
skel_graph_dir = '/home/alexshapsoncoe/drive/random_sample_of_500000_axons_from_each_type_agg20200916c3_multisyn_only'
results_dir = '/home/alexshapsoncoe/drive/sampled_synapse_points_random_sample_of_500000_axons_from_each_type_agg20200916c3_multisyn_only_unconstrained'
credentials_file = '/home/alexshapsoncoe/drive/alexshapsoncoe.json'
acceptable_partners_dir = '/home/alexshapsoncoe/drive/agglo_20200916c3_neurons_and_pure_and_majority_dendrites.json'
max_parallel_requests = 200
batch_size = 75
invalid_point_factor = 20
shard_divisor = 42356404
calculate_shaft_dist_of_stalk_syn = False # This will slow things down if true
interface_seg = '964355253395:h01:goog14r0seg1_agg20200916c3_flat_mio8_interfaces'
min_sim_per_synapse = 1
sim_type = 'unconstrained' # or 'constrained'
use_real_syn_dists = False
plot_sampling_convergence = False
plot_sampled_point_dists = False
add_real_partners_to_acceptable_partners = True


# Convergence rate thresholds:
c_min = 0 # I.e. not using this as threshold atm
gc_avg_max_val = 10 # I.e. not using this as threshold atm
gc_max_max_val = 10 # I.e. not using atm, usually set to 0.005 # More stringent option: 0.001


models_dirs = {
    'shaft_to_shaft': '/home/alexshapsoncoe/drive/syn_dist_distributions/shaft_euclidean_distance_to_shaft_distribution_model.pkl',
    'stalk_to_shaft': '/home/alexshapsoncoe/drive/syn_dist_distributions/stalk_euclidean_distance_to_shaft_distribution_model.pkl',
    'stalk_to_root': '/home/alexshapsoncoe/drive/syn_dist_distributions/stalk_euclidean_distance_to_root_distribution_model.pkl',
}

# Settings for checking neuron post synaptic structures:
check_neuron_post_structure = False
acceptable_partner_types = ['soma', 'dendrite']
cell_ids = '/home/alexshapsoncoe/drive/agglo_20200916c3_cell_data.json'
skel_dir = '/home/alexshapsoncoe/drive/20200916c3_skeletons_6class_plus_myelin'
skel_voxel_size = [32,32,33]
skel_divisor = 42356404

class_lookup = {
    '0': 'axon', 
    '1': 'dendrite', 
    '2': 'astrocyte',
    '3': 'soma',
    '4': 'cilium',
    '5': 'axon initial segment',
    '1000': 'myelinated axon',
    '1001': 'myelinated axon',
    '1002': 'myelinated axon internal fragment',
    '1003': 'myelinated axon internal fragment',
    '1004': 'myelinated axon',
    '1005': 'myelinated axon',
    '-1': 'unclassified',
}



def get_root_neighbour_edge(skel_graph, root_node):

    bb_types = set(['r', 'sr', 'sb', 'b'])

    root_neighbours = skel_graph.neighbors(root_node)
    root_neighbours = [x for x in root_neighbours if skel_graph.vs[x]['ntype'] in bb_types]

    if len(root_neighbours) == 0:
        root_neighbours = skel_graph.neighbors(root_node)

    if len(root_neighbours) == 1:
        root_neighbours.append(root_node)

    first_coord = [skel_graph.vs[root_neighbours[0]][a] for a in ['x', 'y', 'z']]
    second_coord = [skel_graph.vs[root_neighbours[-1]][a] for a in ['x', 'y', 'z']]

    # Don't return an edge with the same x and y coord:
    while (first_coord[0]-second_coord[0]==0 and first_coord[1]-second_coord[1]==0):
        all_bb_nodes = [n.index for n in skel_graph.vs if n['ntype'] in bb_types]
        random_bb_node1 = choice(all_bb_nodes)
        random_bb_node2 = choice(all_bb_nodes)
        first_coord = [skel_graph.vs[random_bb_node1][a] for a in ['x', 'y', 'z']]
        second_coord = [skel_graph.vs[random_bb_node2][a] for a in ['x', 'y', 'z']]

    return [first_coord, second_coord]

def get_random_points_on_shaft(skel_graph, num_random_points):

    all_edges = []

    bb_types = set(['r', 'sr', 'sb', 'b'])

    for edge in skel_graph.es:

        if edge.source_vertex['ntype'] in bb_types and edge.target_vertex['ntype'] in bb_types:
            source_node = [edge.source_vertex[a] for a in ['x', 'y', 'z']]
            target_node = [edge.target_vertex[a] for a in ['x', 'y', 'z']]

            # Don't allow edges with same x and y coordinate:
            if not (source_node[0]==target_node[0] and source_node[1]==target_node[1]):
                all_edges.append([source_node, target_node])

    selected_edges = choices(all_edges, k=num_random_points)
    rand_props = np.random.uniform(0,1, num_random_points)
    vector_roots = np.array([x[0] for x in selected_edges])
    vector_ends = np.array([x[1] for x in selected_edges])
    sel_edges = vector_ends - vector_roots
    dists_from_roots = np.multiply(sel_edges, np.vstack((rand_props,rand_props,rand_props)).T)
    sel_origins = vector_roots+dists_from_roots

    return sel_edges, sel_origins

def get_2df_random_points(sel_edges, sel_origins, model, r_dists=None):

    num_rp = len(sel_origins)

    random_angles = np.random.uniform(0,2*math.pi, num_rp)
    random_orthog = cf.get_orthogonal_vec_stack(sel_edges, random_angles)
    mags = np.linalg.norm(random_orthog, axis=1)
    random_orthog_unit = np.divide(random_orthog, np.vstack((mags,mags,mags)).T)

    if r_dists == None:
        r_dists = model['distr'].rvs(*model['arg'], loc=model['loc'], scale=model['scale'], size=num_rp)

    random_orthog_actual_mag = np.multiply(random_orthog_unit, np.vstack((r_dists,r_dists,r_dists)).T)

    r_points = random_orthog_actual_mag + sel_origins

    return r_points, r_dists

def get_3df_random_points(sel_origins, model, r_dists=None):

    num_rp = len(sel_origins)
    random_unit_v = np.array([cf.random_three_vector() for x in range(num_rp)])

    if r_dists == None:
        r_dists = model['distr'].rvs(*model['arg'], loc=model['loc'], scale=model['scale'], size=num_rp)

    random_v_actual_mag = np.multiply(random_unit_v, np.vstack((r_dists,r_dists,r_dists)).T)
    r_points = random_v_actual_mag + sel_origins

    return r_points, r_dists

def discard_oob_points(r_points, sel_origins, sel_edges, r_dists, agglo_vx_s):

    final_rp_agglo_vx = r_points/agglo_vx_s
    final_rp_agglo_vx = final_rp_agglo_vx.astype(int)

    accepted_real_points = []
    accepted_vx_points = []
    accepted_origins = []
    accepted_edges = []
    accepted_dists = []

    combo = zip(r_points, final_rp_agglo_vx, sel_origins, sel_edges, r_dists)

    for real_point, agglo_v, origin_p, edge, dist in combo:

        bounds_status = set([0 <= agglo_v[a] < agglo_ub[a] for a in range(3)])

        if bounds_status == {True}:
            accepted_real_points.append(real_point)
            accepted_vx_points.append(agglo_v)
            accepted_origins.append(origin_p)
            accepted_edges.append(edge)
            accepted_dists.append(dist)

    accepted_real_points = [tuple([int(a) for a in x]) for x in accepted_real_points]
    accepted_dists = [int(x) for x in accepted_dists]
    accepted_origins =  [tuple([int(a) for a in x]) for x in accepted_origins] 

    return accepted_real_points, accepted_vx_points, accepted_origins, accepted_edges, accepted_dists

def get_agglo_seg_point_ids(requester, points_agglo_vx):


    consecutive_failure_count = 0
    retrieved_ids = None
    
    url = f'https://brainmaps.googleapis.com/v1/volumes/{interface_seg}/values'

    while retrieved_ids == None:
        
        try:
            start = time.time()
            retrieved_ids = requester.retrieve_locations(batch_size, url, points_agglo_vx)
            print(f'... queried {len(retrieved_ids)} locations in {time.time()-start} seconds')

        except:
            consecutive_failure_count += 1

            if consecutive_failure_count == 20:
                print('20 CONSECUTIVE FAILURES, PAUSING FOR 30 MINS')
                time.sleep(1800)
                consecutive_failure_count = 0
            requester._scoped_credentials.refresh(auth_request.Request())
            #requester = cf.ParallelLocationRequester(max_parallel_requests, credentials_file)

    assert len(retrieved_ids) == len(points_agglo_vx)

    return retrieved_ids

def get_neuron_graphs(check_neuron_post_structure, cell_ids, acceptable_partners, skel_divisor, skel_dir, skel_voxel_size):
    
    neuron_graphs = {}
    
    if check_neuron_post_structure == True:
        
        with open(cell_ids, 'r') as fp:
            all_cell_data = json.load(fp)

        neurons = set([x['agglo_seg'] for x in all_cell_data if 'neuron' in x['type']])

        neurons_todo = neurons & acceptable_partners
        
        for i in range(10000):
            
            this_dir_neurons = [x for x in neurons_todo if int(x)//skel_divisor == i]

            if this_dir_neurons == []: continue

            print(i)

            shard_dir = ZipFile(f'{skel_dir}/{i}.zip', 'r')

            raw_skel_dict = cf.get_skel_data_from_shard_dir(this_dir_neurons, shard_dir)

            for seg_id in this_dir_neurons:

                neuron_graphs[seg_id] = cf.make_one_skel_graph_ig(raw_skel_dict[seg_id], skel_voxel_size, join_separate_components=False)
                
    return neuron_graphs
    
        


if __name__ == '__main__':

    # Create parrallel location requester:
    requester = cf.ParallelLocationRequester(max_parallel_requests, credentials_file)

    # Get voxel sizes:
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/brainmaps'])
    scoped_credentials.refresh(auth_request.Request())
    agglo_vx_s, agglo_ub = cf.get_vx_size_and_upper_bounds(scoped_credentials, interface_seg)

    # Load neurites todo:
    with open(neurites_todo_path, 'r') as fp:
        neurites_todo = json.load(fp)

    shuffle(neurites_todo)

    # Load models:
    models = {}
    for k in models_dirs:
        with open(models_dirs[k], 'rb') as fp:
            models[k] = pickle.load(fp)


    # Load acceptable partners:
    with open(acceptable_partners_dir, 'r') as fp:
        acceptable_partners = set(json.load(fp))
    
    # Get any neurons into graph form:
    neuron_graphs = get_neuron_graphs(check_neuron_post_structure, cell_ids, acceptable_partners, skel_divisor, skel_dir, skel_voxel_size)
    
    # for i in range(10000):
    #     if not os.path.exists(f'{results_dir}/{i}'):
    #         os.mkdir(f'{results_dir}/{i}')

    # Sample points around each neurite:

    for seg_id in neurites_todo:

        i = int(int(seg_id)//shard_divisor)

        if os.path.exists(f'{results_dir}/neurite_{seg_id}_simulations.json'):
            continue

        print(f'Starting neurite {seg_id}, {neurites_todo.index(seg_id)+1} of {len(neurites_todo)}')

        neurite_start_time = time.time()

        start = time.time()

        # Load graph
        skel_graph = ig.read(f'{skel_graph_dir}/{i}/{seg_id}.gml')

        # Get real and simulated data per synapse:

        this_seg_data = {dtype: {'real_partners': [], 'simulated_partners': {}} for dtype in ['shaft', 'stalk']}

        syn_nodes = [n for n in skel_graph.vs if n['stype'] in ('shaft', 'stalk')]

        # Record real data:
        for n in syn_nodes:

            real_final_datum = {
                'partner_id': str(n['partner']), 
                'point': tuple([int(n[f'true{a}']) for a in ['x','y','z']]), 
                'origin': tuple([int(skel_graph.vs[int(n['rootnode'])][a]) for a in ['x','y','z']]), 
                'eucdist_shaft': int(n['eucdistshaft']),
                'eucdist_root': int(n['eucdistroot']),
                'syn_id': str(n['synid']), 
                }

            dtype = n['stype']
            this_seg_data[dtype]['real_partners'].append(real_final_datum)

            if add_real_partners_to_acceptable_partners == True:
                acceptable_partners.add(real_final_datum['partner_id'])
            
        min_points_to_sample_this_neurite =  min_sim_per_synapse*len(syn_nodes)
        
        selected_pos = None
        all_sim_partners = []

        while selected_pos == None: 

            points_needed = min_points_to_sample_this_neurite - len(all_sim_partners)
            
            if points_needed < 1:
                points_needed = min_points_to_sample_this_neurite
            
            num_to_attempt_to_sample_per_syn = (int(points_needed/len(syn_nodes))+1)*invalid_point_factor

            # Sample a batch of points:
            start = time.time()
            
            sel_origins_all_syn = []
            r_points_all_syn = []
            r_voxels_all_syn = []
            r_dists_all_syn = []
            shaft_dists_all_syn = []
            dtype_all_syn = []
            syn_id_all_syn = []

            for n in syn_nodes:

                dtype = n['stype']
                syn_id = str(n['synid'])

                # Get location(s) on shaft to simulate synapses from:

                if sim_type == 'constrained':

                    origin = tuple([int(skel_graph.vs[int(n['rootnode'])][a]) for a in ['x','y','z']])
                    root_node = [int(n['rootnode']) for n in skel_graph.vs if n['synid']==syn_id][0]
                    rn_edge = get_root_neighbour_edge(skel_graph, root_node)
                    sel_edges = [rn_edge for x in range(num_to_attempt_to_sample_per_syn)]
                    sel_edges = np.array([x[1] for x in sel_edges]) - np.array([x[0] for x in sel_edges])
                    sel_origins = np.array([origin for x in range(num_to_attempt_to_sample_per_syn)])

                if sim_type == 'unconstrained':

                    sel_edges, sel_origins = get_random_points_on_shaft(skel_graph, num_to_attempt_to_sample_per_syn)


                # Determine whether to use the real distance of each synapse
                
                if use_real_syn_dists == True:

                    if dtype == 'stalk':
                        eucdist_root = int(n['eucdistroot'])
                        real_syn_dist = [eucdist_root for x in range(num_to_attempt_to_sample_per_syn)]

                    if dtype == 'shaft':
                        eucdist_shaft = int(n['eucdistshaft'])
                        real_syn_dist = [eucdist_shaft for x in range(num_to_attempt_to_sample_per_syn)]
                else:
                    real_syn_dist = None
                
                # Get a new set of random points:
        
                if dtype == 'shaft':
                    r_points, r_dists = get_2df_random_points(sel_edges, sel_origins, models[f'{dtype}_to_shaft'], r_dists=real_syn_dist)
                
                if dtype == 'stalk':
                    r_points, r_dists = get_3df_random_points(sel_origins, models[f'{dtype}_to_root'], r_dists=real_syn_dist)
                    
                r_points, r_voxels, sel_origins, sel_edges, r_dists = discard_oob_points(r_points, sel_origins, sel_edges, r_dists, agglo_vx_s)


                if dtype == 'shaft':

                    shaft_dists = r_dists

                if dtype == 'stalk':

                    if calculate_shaft_dist_of_stalk_syn == False:

                        shaft_dists = ['not_done' for x in r_points]
                    
                    if calculate_shaft_dist_of_stalk_syn == True:

                        bb_nodes = [n.index for n in skel_graph.vs if n['ntype'] in set(['r', 'sr', 'sb', 'b'])]
                        root_nodes = cf.get_skel_nodes_closest_to_synapses(sel_origins, skel_graph, bb_nodes)
                        shaft_dists = [cf.get_euc_dist_to_edges_ig(skel_graph, x, root_node, cutoff=5) for x, root_node in zip(r_points, root_nodes)]
                    
                sel_origins_all_syn.extend(sel_origins)
                shaft_dists_all_syn.extend(shaft_dists)
                r_points_all_syn.extend(r_points)
                r_voxels_all_syn.extend(r_voxels)
                r_dists_all_syn.extend(r_dists)
                dtype_all_syn.extend([dtype for x in r_points])
                syn_id_all_syn.extend([syn_id for x in r_points])

            print(f'... getting points to request took {time.time()-start} seconds')
                    
                
            # Make request:
            retrieved_ids = get_agglo_seg_point_ids(requester, r_voxels_all_syn)
            
            assert len(retrieved_ids) == len(r_points_all_syn) == len(r_dists_all_syn) == len(sel_origins_all_syn) == len(shaft_dists_all_syn)
            
            # Get acceptable points:
            start = time.time()
            accepted_points = 0
            
            zipped_data = zip(retrieved_ids, r_points_all_syn, r_dists_all_syn, shaft_dists_all_syn, sel_origins_all_syn, syn_id_all_syn, dtype_all_syn)

            for r_id, point, root_dist, shaft_dist, origin, syn_id, dtype in zipped_data:
                
                if syn_id not in this_seg_data[dtype]['simulated_partners']:
                    this_seg_data[dtype]['simulated_partners'][syn_id] = []

                final_datum = {
                    'partner_id': r_id, 
                    'point': point, 
                    'origin': origin, 
                    'eucdist_shaft': shaft_dist,
                    'eucdist_root': root_dist,
                    'syn_id': syn_id,
                    }

                is_acceptable_partner = False
                
                if r_id in acceptable_partners:

                    if r_id not in neuron_graphs.keys():
                        is_acceptable_partner = True

                    else:

                        skel_graph = neuron_graphs[r_id]

                        nodes_with_classes = [n.index for n in skel_graph.vs if n['nodeclass'] != '-1']

                        original_node = cf.get_skel_nodes_closest_to_synapses([point], skel_graph, nodes_with_classes)[0]

                        final_type = cf.get_neighbourhood_node_type(skel_graph, original_node, class_lookup)

                        if final_type in acceptable_partner_types:
                            is_acceptable_partner = True
                            
                if is_acceptable_partner:
                    this_seg_data[dtype]['simulated_partners'][syn_id].append(final_datum)
                    all_sim_partners.append(r_id)
                    accepted_points += 1

            print(f'... accepted {accepted_points} out of {len(retrieved_ids)} voxels in {time.time()-start} seconds')

            

            # Check whether converged in proportions and exceeeded specified min number to sample, if not, do another batch:
            
            shuffle(all_sim_partners)

            ic_max_list, ic_avg_list, num_samples = cf.get_cg_max_and_cg_avg_vals(all_sim_partners, 1, c_min=c_min)

            for ic_max, ic_avg, pos in zip(ic_max_list, ic_avg_list, num_samples):

                if (ic_max <= gc_max_max_val and ic_avg <= gc_avg_max_val and pos >= min_points_to_sample_this_neurite):
                    selected_pos = pos
                    print(f'... ic_max value of {ic_max} and ic_avg value of {ic_avg}, terminating sampling for this neurite')
                    break
            
            if selected_pos == None:
                print(f'... ic_max value of {ic_max} and ic_avg value of {ic_avg}, sampling another batch for this neurite')
            
            print(f'... {len(all_sim_partners)} points sampled in total from neurite with {len(syn_nodes)} synapses')
        

        this_seg_data['simulation_stopping_position'] = pos
        this_seg_data['simulation_stopping_ic_max'] = ic_max
        this_seg_data['simulation_stopping_ic_avg'] = ic_avg
        this_seg_data['simulation_all_sampled_partners'] = all_sim_partners

        # Plot convergence of sampling:
        if plot_sampling_convergence == True:
            
            start = time.time()
            save_path = f'{results_dir}/neurite_{seg_id}_sampling_convergence.png'
            cf.plot_seg_sampling_convergence(all_sim_partners, save_path, v_line_vals = [gc_max_max_val])
            print(f'... plotting sampling convergence took {time.time()-start} seconds')

        # Plot and save difference with true distance distribution:
        if plot_sampled_point_dists == True:
            
            start = time.time()
            
            for dist_type in ['shaft', 'root']:

                if dist_type == 'root' and dtype == 'shaft': continue

                if (dist_type == 'shaft' and dtype == 'stalk' and calculate_shaft_dist_of_stalk_syn == False): continue

                empirical_dists = []
                
                for syn_id in this_seg_data[dtype]['simulated_partners'].keys():
                    empirical_dists.extend([x[f'eucdist_{dist_type}'] for x in this_seg_data[dtype]['simulated_partners'][syn_id]])
                        
                if len(empirical_dists) == 0: continue

                m = models[f'{dtype}_to_{dist_type}']
                theoretical_dists = m['distr'].rvs(*m['arg'], loc=m['loc'], scale=m['scale'], size=len(empirical_dists))

                ks, p_val = stats.ks_2samp(np.array(empirical_dists), theoretical_dists)

                filename = f'neurite_{seg_id}_{sim_type}_sampled_points_vs_true_dist_{dtype}_to_{dist_type}_KS_{p_val}.png'
                x_axis = f'Distance from {dist_type} (nm)'
                y_axis = 'Probability density'
                title = f'Distribution of distance from {dtype}-type synapses to {dist_type} for {sim_type} sampled points and true distribution'

                cf.plot_empirical_vs_theoretical_dist(m, empirical_dists, f'{results_dir}/', filename, x_axis, y_axis, title) 

            print(f'... plotting sampled synapse distance distributions took {time.time()-start} seconds')
            
        with open(f'{results_dir}/neurite_{seg_id}_simulations.json', 'w') as fp:
            json.dump(this_seg_data, fp)

        print('overall time', time.time()-neurite_start_time)

    # Once done all axons, save chosen parameters:
        
    cs = {
        'neurites_todo_path': neurites_todo_path,
        'skel_graph_dir': skel_graph_dir,
        'results_dir': results_dir,
        'credentials_file': credentials_file,
        'shard_divisor': shard_divisor,
        'neuron_list_dir': cell_ids,
        'acceptable_partners_dir': acceptable_partners_dir,
        'models': models_dirs,
        'max_parallel_requests': max_parallel_requests,
        'batch_size': batch_size,
        'calculate_shaft_dist_of_stalk_syn': calculate_shaft_dist_of_stalk_syn, # This will slow things down if true
        'interface_seg': interface_seg,
        'min_sim_per_neurite': min_points_to_sample_this_neurite,
        'acceptable_partner_types': acceptable_partner_types,
        'sim_type': sim_type,
    }

    with open(f'{results_dir}/config.json', 'w') as fp:
        json.dump(cs, fp)



