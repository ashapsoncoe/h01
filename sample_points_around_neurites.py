import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

from google.oauth2 import service_account
from google.cloud import bigquery        
from google.cloud import bigquery_storage    
from scipy.spatial.distance import cdist     
import json
from google.auth.transport import requests as auth_request
import numpy as np
import math
import time
from random import choices, choice, shuffle
from itertools import product
import igraph as ig
import pickle
import common_functions_h01 as cf
from scipy import stats
from zipfile import ZipFile
from copy import deepcopy


models_pkl_files = { # syn_dist_distributions needs to be unzipped before running
    'shaft_to_shaft': 'syn_dist_distributions/shaft_euclidean_distance_to_shaft_distribution_model.pkl',
    'stalk_to_shaft': 'syn_dist_distributions/stalk_euclidean_distance_to_shaft_distribution_model.pkl',
    'stalk_to_root': 'syn_dist_distributions/stalk_euclidean_distance_to_root_distribution_model.pkl',
}

neurites_todo_list = 'random_sample_of_10000e_or_i_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_0ax_list.json'
skel_graph_dir = 'random_sample_of_10000e_or_i_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_pruned' # available from gs://h01_paper_public_files/random_sample_of_10000e_or_i_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_pruned
results_dir = 'sim_synaptic_points_from_random_sample_of_10000e_or_i_axons_from_each_gp_strength_agg20200916c3_eirepredict_0ax_15um_displacement_xy_rp_not_added_20um_soma_exc' # available from gs://h01_paper_public_files/h01_paper_public_files/sim_synaptic_points_from_random_sample_of_10000e_or_i_axons_from_each_gp_strength_agg20200916c3_eirepredict_0ax_15um_displacement_xy_rp_not_added_20um_soma_exc
credentials_file = 'alexshapsoncoe.json' # or your credentials_file
skel_sql_table = 'goog14r0seg1.agg20200916c3_subcompartment_skeleton_counts_v1'
cell_ids = 'agglo_20200916c3_cell_data.json'
soma_exclusion_dist = 20000 # In nm
max_parallel_requests = 50
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
add_real_partners_to_acceptable_partners = False
save_neuroglancer_state = False
em = 'brainmaps://964355253395:h01:goog14r0_8nm'
agglo_seg = 'brainmaps://964355253395:h01:goog14r0seg1_agg20200916c3_flat'
nm_to_translate_neurite = {'x': 15000, 'y': 15000, 'z': 0} # Set to (0,0,0) to not use



# Convergence rate thresholds:
c_min = 0 # I.e. not using this as threshold atm
gc_avg_max_val = 10 # I.e. not using this as threshold atm
gc_max_max_val = 10 # I.e. not using atm, usually set to 0.005 # More stringent option: 0.001




# Settings for checking neuron post synaptic structures:
check_neuron_post_structure = False
acceptable_partner_types = ['soma', 'dendrite']
skel_dir = '20200916c3_skeletons_6class_plus_myelin'
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

    requester._scoped_credentials.refresh(auth_request.Request())

    consecutive_failure_count = 0
    retrieved_ids = None
    
    url = f'https://brainmaps.googleapis.com/v1/volumes/{interface_seg}/values'

    while retrieved_ids == None:

        try:
            start = time.time()
            retrieved_ids = requester.retrieve_locations(batch_size, url, points_agglo_vx)
            #print(f'... queried {len(retrieved_ids)} locations in {time.time()-start} seconds')

        except:
            consecutive_failure_count += 1

            if consecutive_failure_count == 20:
                print('20 CONSECUTIVE FAILURES, PAUSING FOR 30 MINS')
                time.sleep(1800)
                consecutive_failure_count = 0
            requester._scoped_credentials.refresh(auth_request.Request())


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

            shard_dir = ZipFile(f'{skel_dir}/{i}.zip', 'r')

            raw_skel_dict = cf.get_skel_data_from_shard_dir(this_dir_neurons, shard_dir)

            for seg_id in this_dir_neurons:

                neuron_graphs[seg_id] = cf.make_one_skel_graph_ig(raw_skel_dict[seg_id], skel_voxel_size, join_separate_components=False)
                
    return neuron_graphs
    
def get_batch_sample_locations(syn_ids_without_min, current_min_sim_per_synapse, invalid_point_factor, syn_nodes, skel_graph, sim_type, calculate_shaft_dist_of_stalk_syn):

    # Sample a batch of points:
    start = time.time()

    #print(f'... currently {len(syn_ids_without_min)} synapses without the minimum number of {current_min_sim_per_synapse} sampled points')

    num_to_attempt_to_sample_per_syn = current_min_sim_per_synapse*invalid_point_factor

    target_point_info = {
        'sel_origins_all_syn': [],
        'r_points_all_syn': [],
        'r_voxels_all_syn': [],
        'r_dists_all_syn': [],
        'shaft_dists_all_syn': [],
        'dtype_all_syn': [],
        'syn_id_all_syn': [],
    }

    for n in syn_nodes:

        syn_id = str(n['synid'])

        if syn_id in syn_ids_without_min:

            dtype = n['stype']

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
            
            target_point_info['sel_origins_all_syn'].extend(sel_origins)
            target_point_info['shaft_dists_all_syn'].extend(shaft_dists)
            target_point_info['r_points_all_syn'].extend(r_points)
            target_point_info['r_voxels_all_syn'].extend(r_voxels)
            target_point_info['r_dists_all_syn'].extend(r_dists)
            target_point_info['dtype_all_syn'].extend([str(n['stype']) for x in r_points])
            target_point_info['syn_id_all_syn'].extend([syn_id for x in r_points])

    #print(f'... getting points to request took {time.time()-start} seconds')

    return  target_point_info
             
def get_translated_neurite(skel_graph_original, rand_translate):

    skel_graph = deepcopy(skel_graph_original)

    for node in skel_graph.vs:

        for dim in ('x', 'y', 'z'):

            node[dim] += rand_translate[dim] 

            if node['synid'] != 'None':

                node[f'true{dim}'] += rand_translate[dim] 
    
    return skel_graph

def get_acceptable_points_from_sampled(this_seg_data, all_sim_partners, retrieved_ids, target_point_info):

    start = time.time()

    accepted_points_count = 0

    zipped_data = zip(
        retrieved_ids, 
        target_point_info['r_points_all_syn'], 
        target_point_info['r_dists_all_syn'], 
        target_point_info['shaft_dists_all_syn'], 
        target_point_info['sel_origins_all_syn'], 
        target_point_info['syn_id_all_syn'], 
        target_point_info['dtype_all_syn'],
        )

    for r_id, point, root_dist, shaft_dist, origin, syn_id, dtype in zipped_data:

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

            if sim_type == 'unconstrained':
                for any_syn_id in this_seg_data[dtype]['simulated_partners']:
                    if len(this_seg_data[dtype]['simulated_partners'][any_syn_id]) < current_min_sim_per_synapse:
                        this_seg_data[dtype]['simulated_partners'][any_syn_id].append(final_datum)
                        all_sim_partners.append(r_id)
                        accepted_points_count += 1
                        break
            
            if sim_type == 'constrained':

                if len(this_seg_data[dtype]['simulated_partners'][syn_id]) < current_min_sim_per_synapse:
                    this_seg_data[dtype]['simulated_partners'][syn_id].append(final_datum)
                    all_sim_partners.append(r_id)
                    accepted_points_count += 1

    #print(f'... accepted {accepted_points_count} out of {len(retrieved_ids)} voxels in {time.time()-start} seconds')

    return accepted_points_count

def get_syn_ids_todo(this_seg_data, current_min_sim_per_synapse):

    syn_ids_without_min = []

    for dtype in ('shaft', 'stalk'):
        for syn_id in this_seg_data[dtype]['simulated_partners']:
            num_sampled_from_this_syn = len(this_seg_data[dtype]['simulated_partners'][syn_id])
            #print(f'... {num_sampled_from_this_syn} points sampled around synapse {syn_id}')
            if num_sampled_from_this_syn < current_min_sim_per_synapse:
                syn_ids_without_min.append(syn_id)

    return syn_ids_without_min

if __name__ == '__main__':

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Get cell location data:

    with open(cell_ids, 'r') as fp:
        all_cell_data = json.load(fp)

    all_cb_locations_nm = np.array([[a[f'true_{c}'] for c in ('x', 'y', 'z')] for a in all_cell_data])

    query = f"""
                SELECT CAST(skeleton_id AS STRING) AS agglo_id FROM {skel_sql_table} 
                WHERE num_spine+num_dendrite+num_soma > num_axon+num_astrocyte+num_cilium+num_ais+num_myelinated_axon+num_myelin_internal
                """
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)
    df = client.query(query).result().to_dataframe(bqstorage_client=bqstorageclient)

    acceptable_partners = set(df['agglo_id'])

    neurons = [x['agglo_seg'] for x in all_cell_data if 'neuron' in x['type']]
    non_neurons = [x['agglo_seg'] for x in all_cell_data if 'neuron' not in x['type']]
    acceptable_partners.update(neurons)
    acceptable_partners -= set(non_neurons)
    if '0' in acceptable_partners:
        acceptable_partners.remove('0')


    # Create parrallel location requester:
    requester = cf.ParallelLocationRequester(max_parallel_requests, credentials_file)

    # Get voxel sizes:
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/brainmaps'])
    scoped_credentials.refresh(auth_request.Request())
    agglo_vx_s, agglo_ub = cf.get_vx_size_and_upper_bounds(scoped_credentials, interface_seg)

    # Load neurites todo:
    with open(neurites_todo_list, 'r') as fp:
        neurites_todo = json.load(fp)

    shuffle(neurites_todo)

    # Load models:
    models = {}
    for k in models_pkl_files:
        with open(f'{working_dir}/{models_pkl_files[k]}', 'rb') as fp:
            models[k] = pickle.load(fp)


    skel_dir = f'{working_dir}/{skel_dir}'
    
    # Get any neurons into graph form:
    neuron_graphs = get_neuron_graphs(check_neuron_post_structure, cell_ids, acceptable_partners, skel_divisor, skel_dir, skel_voxel_size)
    
    # Sample points around each neurite:
    
    for seg_id in neurites_todo:

        if os.path.exists(f'{results_dir}/neurite_{seg_id}_simulations.json'): continue

        print(f'Starting neurite {seg_id}, {neurites_todo.index(seg_id)+1} of {len(neurites_todo)}')

        neurite_start_time = time.time()

        i = int(int(seg_id)//shard_divisor)

        skel_graph_original = ig.read(f'{skel_graph_dir}/{i}/{seg_id}.gml')

        this_seg_data = {dtype: {'real_partners': [], 'simulated_partners': {}} for dtype in ['shaft', 'stalk']}

        syn_nodes = [n for n in skel_graph_original.vs if n['stype'] in ('shaft', 'stalk')]

        # Record real data:
        for n in syn_nodes:

            try:
                origin = tuple([str(skel_graph_original.vs[int(n['rootnode'])][a]) for a in ['x','y','z']])
            except OverflowError:
                origin = 'not_recorded'

            try:
                eucdist_shaft = int(n['eucdistshaft'])
            except OverflowError:
                eucdist_shaft = 'not_recorded'  

            try:
                eucdist_root = int(n['eucdistroot'])
            except OverflowError:
                eucdist_root = 'not_recorded' 

            real_final_datum = {
                'partner_id': str(n['partner']), 
                'point': tuple([n[f'true{a}'] for a in ['x','y','z']]), # Set to str as precision error in np causes some values to be inf
                'origin': origin, 
                'eucdist_shaft': eucdist_shaft,
                'eucdist_root': eucdist_root,
                'syn_id': str(n['synid']), 
                }

            dtype = str(n['stype'])
            syn_id = str(n['synid'])

            this_seg_data[dtype]['real_partners'].append(real_final_datum)

            if add_real_partners_to_acceptable_partners == True:
                acceptable_partners.add(real_final_datum['partner_id'])

        tmp = [set([nm_to_translate_neurite[a]*b for b in (1,-1)]) for a in ('x', 'y', 'z')]
        random_translations = [{'x':x, 'y':y, 'z':z} for x,y,z in product(tmp[0], tmp[1], tmp[2])]
        shuffle(random_translations)
        
        for rand_translate in random_translations:

            for n in syn_nodes:
                this_seg_data[n['stype']]['simulated_partners'][n['synid']] = []

            # Translate neurite by specified amount:
            skel_graph = get_translated_neurite(skel_graph_original, rand_translate)

            # If too close to a cell body, skip:
            ave_xyz_nm = [np.mean([n[a] for n in skel_graph.vs], axis=0) for a in ('x', 'y', 'z')]

            shortest_dist_nm = cdist(all_cb_locations_nm, [ave_xyz_nm], 'euclidean').min()

            print(f'... translated neurite to {ave_xyz_nm}')

            if shortest_dist_nm < soma_exclusion_dist:
                print(f'... translated neurite at {ave_xyz_nm} excluded due to soma centre within {shortest_dist_nm}nm')
                continue


            # Then sample simulated synapses:
            sampled_enough_from_all_syn = False
            current_min_sim_per_synapse = min_sim_per_synapse
            all_sim_partners = []
            consecutive_sampling_failures = 0

            while not sampled_enough_from_all_syn: 

                syn_ids_without_min = get_syn_ids_todo(this_seg_data, current_min_sim_per_synapse)

                # Get locations to sample this batch:
                target_point_info = get_batch_sample_locations(syn_ids_without_min, current_min_sim_per_synapse, invalid_point_factor, syn_nodes, skel_graph, sim_type, calculate_shaft_dist_of_stalk_syn)

                # Make request:
                retrieved_ids = get_agglo_seg_point_ids(requester, target_point_info['r_voxels_all_syn'])

                # Get acceptable points:
                accepted_points_count = get_acceptable_points_from_sampled(this_seg_data, all_sim_partners, retrieved_ids, target_point_info)

                # Check whether any syn_ids have not met quota:
                syn_ids_without_min = get_syn_ids_todo(this_seg_data, current_min_sim_per_synapse)

                # If all have met quota, check whether converged in proportions and exceeeded specified min number to sample, if not, do another batch:
                if syn_ids_without_min == []:

                    print(f'... have sampled {current_min_sim_per_synapse} points around each synapse')

                    shuffle(all_sim_partners)

                    ic_max_list, ic_avg_list, num_samples = cf.get_cg_max_and_cg_avg_vals(all_sim_partners, 1, c_min=c_min)

                    for ic_max, ic_avg, pos in zip(ic_max_list, ic_avg_list, num_samples):

                        if (ic_max <= gc_max_max_val and ic_avg <= gc_avg_max_val):
                            sampled_enough_from_all_syn = True
                            selected_pos = pos
                            print(f'... ic_max value of {ic_max} and ic_avg value of {ic_avg}, terminating sampling for this neurite')
                            break


                    if sampled_enough_from_all_syn == False:
                        current_min_sim_per_synapse += min_sim_per_synapse
                        print(f'... ic_max value of {ic_max} and ic_avg value of {ic_avg}, increasing sampling target to {current_min_sim_per_synapse} points per synapse')

                # Avoid endlessly sampling from positions without acceptable partners:
                if accepted_points_count == 0:
                    consecutive_sampling_failures += 1
                else:
                    consecutive_sampling_failures = 0

                if consecutive_sampling_failures == 20:
                    print('... sampling from current position failed')
                    break

            if sampled_enough_from_all_syn:

                print(f'... successfully sampled {len(all_sim_partners)} points from neurite with {len(syn_nodes)} synapses')
                
                this_seg_data['simulation_stopping_position'] = pos
                this_seg_data['simulation_stopping_ic_max'] = ic_max
                this_seg_data['simulation_stopping_ic_avg'] = ic_avg
                this_seg_data['simulation_all_sampled_partners'] = all_sim_partners
                this_seg_data['random_translation'] = rand_translate

                # Save edges too:
                edge_coords = []

                for edge in skel_graph.es:
                    s, t = edge.source, edge.target
                    source_xyz = tuple([int(skel_graph.vs[s][a]) for a in ('x', 'y', 'z')])
                    target_xyz = tuple([int(skel_graph.vs[t][a]) for a in ('x', 'y', 'z')])
                    to_save = (source_xyz, target_xyz)
                    edge_coords.append(to_save)

                this_seg_data['edge_coords'] = edge_coords

                # Save neuroglancer state with sampled synapse points
                if save_neuroglancer_state == True:
                    cf.save_ng_state_of_sampled_points(em, agglo_seg, seg_id, this_seg_data, results_dir, skel_graph=skel_graph)

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

                #print('... overall time', time.time()-neurite_start_time)
                break # stop trying different translations

            else:
                if rand_translate == random_translations[-1]:
                    print('... sampling failed from all positions')

    # Once done all axons, save chosen parameters:
        
    cs = {
        'neurites_todo_list': neurites_todo_list,
        'skel_graph_dir': skel_graph_dir,
        'results_dir': results_dir,
        'credentials_file': credentials_file,
        'shard_divisor': shard_divisor,
        'neuron_list_dir': cell_ids,
        'models': models_pkl_files,
        'max_parallel_requests': max_parallel_requests,
        'batch_size': batch_size,
        'calculate_shaft_dist_of_stalk_syn': calculate_shaft_dist_of_stalk_syn, # This will slow things down if true
        'interface_seg': interface_seg,
        'min_sim_per_synapse': min_sim_per_synapse,
        'acceptable_partner_types': acceptable_partner_types,
        'sim_type': sim_type,
        'nm_to_translate_neurite': nm_to_translate_neurite, 
    }

    with open(f'{results_dir}/config.json', 'w') as fp:
        json.dump(cs, fp)



