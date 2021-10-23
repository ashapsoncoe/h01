import common_functions as cf
import json
from zipfile import ZipFile
import networkx as nx
from copy import deepcopy
import os
import sys
from random import choice
import numpy as np
from scipy.spatial.distance import cdist
import time
from multiprocessing import Pool


# Later expand to include all proofread files in a directory



cell_ids_todo_dir = '/home/alexshapsoncoe/drive/basal_dendrite_cell_list_20200916c3_864_agglo_ids_oct_2021.json'
save_dir = '/home/alexshapsoncoe/drive/separate_neuron_components_Layer_6_basal_cell_list_agglo_20200916c3'
skel_dir = '/home/alexshapsoncoe/drive/20200916c3_skeletons_6class_plus_myelin'
skel_voxel_size = [32,32,33]
max_cb_radius_nm = 15000
skel_shard_divisor = 42356404
max_n_cc_to_rejoin = 5000
cpu_num = 15
remove_astrocyte_nodes = False


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

def identify_cell_body(g, class_lookup):

    temp_g = deepcopy(g)

    types_to_remove = (
        'axon', 
        'dendrite', 
        'astrocyte', 
        'cilium', 
        'axon initial segment', 
        'myelinated axon', 
        'myelinated axon internal fragment',
    )

    non_cb_nodes = [x for x in temp_g.nodes() if class_lookup[temp_g.nodes[x]['nodeclass']] in types_to_remove]
    temp_g.remove_nodes_from(non_cb_nodes)
    con_coms = list(nx.connected_components(temp_g))

    for cc in con_coms:
        if set([temp_g.nodes[x]['nodeclass'] for x in cc]) == {'-1'}:
            temp_g.remove_nodes_from(cc)  

    # Identify soma cc by greatest average edge weight:
    con_coms = list(nx.connected_components(temp_g))
    soma_cc = None
    greatest_mean_edge_rad = 0

    for cc in con_coms:

        for node in cc: # All marked as unclassified until proven otherwise
            g.nodes[node]['nodeclass'] = '-1'

        mean_edge_rad = np.mean([g.edges[x]['radius'] for x in g.edges() if set(x).issubset(cc) and 'radius' in g.edges[x]])

        if mean_edge_rad > greatest_mean_edge_rad:
            greatest_mean_edge_rad = mean_edge_rad
            soma_cc = cc

    return soma_cc


def remove_astro_nodes(g):

    astro_nodes = set([x for x in g.nodes() if 'astrocyte' in class_lookup[g.nodes[x]['nodeclass']]])

    astro_neighbours = set([x for x in g.nodes() if set(g[x]) & astro_nodes != set()])

    g.remove_nodes_from(astro_nodes)
    
    if nx.number_connected_components(g)> max_n_cc_to_rejoin: return
    
    cf.add_cc_bridging_edges_pairwise(g, joining_nodes=astro_neighbours)


def do_one_cell(seg_id):

    start = time.time()

    i = int(int(seg_id)/skel_shard_divisor)
    
    shard_dir = ZipFile(skel_dir + '/' + str(i) + '.zip', 'r')

    raw_skel_dict = cf.get_skel_data_from_shard_dir([seg_id], shard_dir)
    
    g = cf.make_one_skel_graph_nx(raw_skel_dict[seg_id], skel_voxel_size, join_components=False)


    # Join any end soma nodes within the cutoff distance of each other:
    all_soma_nodes = [x for x in g.nodes() if g.nodes[x]['nodeclass'] == '3']

    all_soma_locs = [[int(g.nodes[node][a]) for a in ['x','y','z']] for node in all_soma_nodes]

    f = cdist(all_soma_locs, all_soma_locs, 'euclidean')

    c = np.argwhere(f<max_cb_radius_nm)

    new_edges = [(all_soma_nodes[pos1], all_soma_nodes[pos2]) for pos1, pos2 in c if int(pos1) < int(pos2)]

    new_edges = [x for x in new_edges if x not in g.edges()]

    g.add_edges_from(new_edges)

    # Then join up any outstanding ccs:
    if nx.number_connected_components(g)> max_n_cc_to_rejoin: return
    cf.add_cc_bridging_edges_pairwise(g)

    # First identify the cell bodies:
    soma_cc = identify_cell_body(g, class_lookup)

    # Having identified the soma, change all of its nodes to '3':

    for node in soma_cc:
        g.nodes[node]['nodeclass'] = '3'

    if remove_astrocyte_nodes == True:
        remove_astro_nodes(g)
    

    # Then inspect each remaining connected component, to see whether axon or dendrite

    temp_g = deepcopy(g)
    temp_g.remove_nodes_from(soma_cc)
    branch_ccs = list(nx.connected_components(temp_g))

    # Change unclassified, axon, dendrite or soma to axon / dendrite:
    axons = []
    dendrites = []
    unclear = []

    for cc in branch_ccs:

        ax_nodes = [x for x in cc if 'axon' in class_lookup[temp_g.nodes[x]['nodeclass']]]
        d_nodes = [x for x in cc if 'dendrite' in class_lookup[temp_g.nodes[x]['nodeclass']]]

        total_classified = len(ax_nodes)+len(d_nodes)

        if total_classified == 0: 
            unclear.append(cc)
            continue

        ax_prop = round((len(ax_nodes)/total_classified)*100)
        d_prop = round((len(d_nodes)/total_classified)*100)

        if total_classified < 100 and d_prop > ax_prop:
            unclear.append(cc)
            continue

        if total_classified < 5 or (total_classified < 10 and max(ax_prop, d_prop) < 0.8):
            unclear.append(cc)
            continue

        if len(ax_nodes) > len(d_nodes):
            axons.append([cc, ax_prop, d_prop])

        if len(ax_nodes) < len(d_nodes):
            dendrites.append([cc, ax_prop, d_prop])

        if len(ax_nodes) == len(d_nodes):
            unclear.append(cc)
            continue
    

    for cc in unclear:
        g.remove_nodes_from(cc)

    print('axons', len(axons), 'dendrites', len(dendrites))


    for dataset, label in [[axons, 'axon'], [dendrites, 'dendrite'], [[[soma_cc,0,0]], 'soma']]:

        for cc, ax_prop, d_prop in dataset:

            num = dataset.index([cc, ax_prop, d_prop])

            for node in cc:

                g.nodes[node]['nodeclasstype'] = label
                g.nodes[node]['typecomponentnumber'] = num 

            
    nx.write_gml(g, f'{save_dir}/cell_{seg_id}_wholecell_final_labels_{len(axons)}_axons_{len(dendrites)}_dendrites.gml')
    
    print(f'... cell {seg_id} took {time.time()-start} seconds')



if __name__ == '__main__':
    
    completed_cells = [x.split('_')[1] for x in os.listdir(save_dir) if 'wholecell' in x]
    
    with open(cell_ids_todo_dir, 'r') as fp:
        cell_ids = json.load(fp)

    cell_ids = [x for x in cell_ids if x not in completed_cells]
        
    pool = Pool(cpu_num)
    pool.map(do_one_cell, cell_ids)
    pool.close()
    pool.join()    


        
