from google.cloud import bigquery             
from google.oauth2 import service_account
from google.cloud import bigquery_storage  
from scipy.spatial.distance import cdist, euclidean
import os
import common_functions as cf
import json
import pandas as pd
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import repeat
import time


credentials_file = '/home/alexshapsoncoe/drive/alexshapsoncoe.json'
save_dir = '/home/alexshapsoncoe/drive/Layer_6_basal_cell_partners_agglo_20200916c3_v3_pure_axons_only'
processed_neurons_dir = '/home/alexshapsoncoe/drive/separate_neuron_components_Layer_6_basal_cell_list_agglo_20200916c3'
basal_dendrite_df_dir = '/home/alexshapsoncoe/drive/goog14_L6basal_matrix_c3.csv'
syn_db_name = 'goog14r0s5c3.synaptic_connections_with_skeleton_classes'
segment_types_lists = '/home/alexshapsoncoe/drive/axon_dendrite_astrocyte_cilia_pure_and_majority_agglo_20200916c3/all_classifications.json'
seg_info_db = 'goog14r0seg1.agg20200916c3_regions_types'
syn_vx_size = [8,8,33]
basal_dendrite_df_vx_size = [8,8,33]
max_nm_from_syn = 3000
cpu_num = 14

def do_one_basal_dendrite(i, cell_df, available_cells):

    start = time.time()

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    agglo_id = str(cell_df.at[i, 'google_agglo_id'])

    if os.path.exists(f'{save_dir}/{agglo_id}_data.json'): return

    print(i, agglo_id)
    
    final_result = {}
    final_result['base_seg'] = str(cell_df.at[i, 'google_base_id'])
    final_result['dbcellid'] = str(cell_df.at[i, 'dbcellid'])
    final_result['elevation_angle'] = int(cell_df.at[i, ' basal dendrite elevation angle (degrees)'])
    final_result['azimuth_angle'] = int(cell_df.at[i, ' basal dendrite azimuth angle (degrees)'])

    cb_loc = (
        int(cell_df.at[i, 'cell body x']*basal_dendrite_df_vx_size[0]),
        int(cell_df.at[i, 'y']*basal_dendrite_df_vx_size[1]),
        int(cell_df.at[i, 'z(in_full-res_pixels)']*basal_dendrite_df_vx_size[2]),
    )

    final_result['cb_loc'] = cb_loc

    if agglo_id not in available_cells: return

    whole_g = nx.read_gml(f'{processed_neurons_dir}/{available_cells[agglo_id]}')


    basal_d_com = [
        cell_df.at[i, ' basal dendrite center of mass shifted to within dendrite (x']*basal_dendrite_df_vx_size[0],
        cell_df.at[i, 'y.1']*basal_dendrite_df_vx_size[1],
        cell_df.at[i, 'z)']*basal_dendrite_df_vx_size[2],
    ]

    final_result['basal_d_com'] = [int(x) for x in basal_d_com]

    dendrite_nodes = [n for n in whole_g.nodes() if whole_g.nodes[n]['nodeclasstype'] == 'dendrite']
    
    if dendrite_nodes == []:
        print(f'Skipping cell {agglo_id} as no dendrite nodes')
        return

    basal_node = cf.get_skel_nodes_closest_to_synapses([basal_d_com], whole_g, dendrite_nodes)[0]

    dendrite_component = whole_g.nodes[basal_node]['typecomponentnumber']

    selected_nodes = [n for n in whole_g.nodes if whole_g.nodes[n]['typecomponentnumber']==dendrite_component]

    sel_node_locs = [(int(whole_g.nodes[n]['x']), int(whole_g.nodes[n]['y']), int(whole_g.nodes[n]['z'])) for n in selected_nodes]

    final_result['basal_node_locations'] = sel_node_locs


    query = f"""
                    with pure_axons as (
                    select CAST(agglo_id AS STRING) as agglo_id
                    from {seg_info_db}
                    where type = 'pure axon fragment'
                    ),

                    rel_pres as (
                    SELECT CAST(pre_synaptic_site.neuron_id AS STRING) AS pre_id,
                    location.x*{syn_vx_size[0]} AS x, 
                    location.y*{syn_vx_size[1]} AS y, 
                    location.z*{syn_vx_size[2]} AS z, 
                    pre_synaptic_site.id AS pre_syn_id,
                    post_synaptic_partner.id AS post_syn_id,
                    type, 
                    LOWER(post_synaptic_partner.class_label) AS post_type,
                    LOWER(pre_synaptic_site.class_label) AS pre_type
                    from {syn_db_name}
                    WHERE post_synaptic_partner.neuron_id = {agglo_id}
                    )

                SELECT pre_id,x,y,z,pre_syn_id,post_syn_id,type,post_type,pre_type
                    from rel_pres A
                    inner join pure_axons B
                    on A.pre_id = B.agglo_id
            """
    #print(query)
    
    raw_data = [dict(x) for x in client.query(query).result()]
    print(len(raw_data))

    all_syn_locations = [(int(r['x']), int(r['y']), int(r['z'])) for r in raw_data]
    all_syn_types = [int(r['type']) for r in raw_data]
    all_syn_partners = [str(r['pre_id']) for r in raw_data]
    all_pre_syn_ids = [str(r['pre_syn_id']) for r in raw_data]
    all_post_syn_ids = [str(r['post_syn_id']) for r in raw_data]

    temp_dists = cdist(all_syn_locations, sel_node_locs, 'euclidean')

    close_enough = [min(x)<max_nm_from_syn for x in temp_dists]

    final_result['basal_synapses'] = []

    zipped_data = zip(all_syn_locations, all_syn_types, all_syn_partners, all_pre_syn_ids, all_post_syn_ids, close_enough)

    for loc, ptype, pseg, pre_synid, post_synid, basal in zipped_data:

        if basal == True:

            final_result['basal_synapses'].append({
                'pre_seg_id': pseg,
                'syn_location': loc,
                'syn_type': ptype,
                'pre_syn_id': pre_synid,
                'post_syn_id': post_synid,
            })

    print(time.time()-start)
    print('a')
    with open(f'{save_dir}/{agglo_id}_data.json', 'w') as fp:
        json.dump(final_result, fp)
    print('b')
    
    



if __name__ == '__main__':


    cell_df = pd.read_csv(basal_dendrite_df_dir)

    available_cells = {x.split('_')[1]: x for x in os.listdir(processed_neurons_dir)}

    pool = Pool(cpu_num)

    args = zip(cell_df.index, repeat(cell_df), repeat(available_cells))

    pool.starmap(do_one_basal_dendrite, args)
    pool.join()
    pool.close()

    



