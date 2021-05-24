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


credentials_file = '/home/alexshapsoncoe/drive/alexshapsoncoe.json'
save_path = '/home/alexshapsoncoe/drive/Layer_6_basal_cell_partners_agglo_20200916c3.json'
all_axons_flat_list_save_path = '/home/alexshapsoncoe/drive/Layer_6_basal_cell_partners_agglo_20200916c3_flat_list.json'
all_axons_multi_basal_d_save_path = '/home/alexshapsoncoe/drive/Layer_6_basal_cell_partners_agglo_20200916c3_multi_basal_d_targets.json'
processed_neurons_dir = '/home/alexshapsoncoe/drive/separate_neuron_components_Layer_6_basal_cell_list_agglo_20200916c3'
basal_dendrite_df_dir = '/home/alexshapsoncoe/drive/goog14_L6basal_matrix_c3.csv'
syn_db_name = 'goog14r0s5c3.synaptic_connections_ei_merge_correction1'
segment_types_lists = '/home/alexshapsoncoe/drive/axon_dendrite_astrocyte_cilia_pure_and_majority_agglo_20200916c3/all_classifications.json'
syn_vx_size = [8,8,33]
basal_dendrite_df_vx_size = [8,8,33]
max_nm_from_syn = 1000


if __name__ == '__main__':

    with open(segment_types_lists, 'r') as fp:
        segment_types = json.load(fp)

    pure_axons = set(segment_types['axon']['pure'])

    del segment_types

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    cell_df = pd.read_csv(basal_dendrite_df_dir)

    available_cells = {x.split('_')[1]: x for x in os.listdir(processed_neurons_dir)}

    agglo_ids = cell_df['google_agglo_id']

    final_data = {}




    for i in cell_df.index:
        print(i)

        agglo_id = str(cell_df.at[i, 'google_agglo_id'])
        

        final_data[agglo_id] = {}
        final_data[agglo_id]['base_seg'] = str(cell_df.at[i, 'google_base_id'])
        final_data[agglo_id]['dbcellid'] = str(cell_df.at[i, 'dbcellid'])
        final_data[agglo_id]['elevation_angle'] = int(cell_df.at[i, ' basal dendrite elevation angle (degrees)'])
        final_data[agglo_id]['azimuth_angle'] = int(cell_df.at[i, ' basal dendrite azimuth angle (degrees)'])

        cb_loc = (
            int(cell_df.at[i, 'cell body x']*basal_dendrite_df_vx_size[0]),
            int(cell_df.at[i, 'y']*basal_dendrite_df_vx_size[1]),
            int(cell_df.at[i, 'z(in_full-res_pixels)']*basal_dendrite_df_vx_size[2]),
        )

        final_data[agglo_id]['cb_loc'] = cb_loc

        if agglo_id not in available_cells: continue

        whole_g = nx.read_gml(f'{processed_neurons_dir}/{available_cells[agglo_id]}')


        basal_d_com = [
            cell_df.at[i, ' basal dendrite center of mass shifted to within dendrite (x']*basal_dendrite_df_vx_size[0],
            cell_df.at[i, 'y.1']*basal_dendrite_df_vx_size[1],
            cell_df.at[i, 'z)']*basal_dendrite_df_vx_size[2],
        ]

        dendrite_nodes = [n for n in whole_g.nodes() if whole_g.nodes[n]['nodeclasstype'] == 'dendrite']
        
        if dendrite_nodes == []:
            print(f'Skipping cell {agglo_id} as no dendrite nodes')
            continue

        basal_node = cf.get_skel_nodes_closest_to_synapses([basal_d_com], whole_g, dendrite_nodes)[0]

        # if whole_g.nodes[basal_node]['nodeclasstype'] != 'dendrite':
        #     print(f'Skipping cell {agglo_id} as basal node is not dendrite')
        #     continue

        dendrite_component = whole_g.nodes[basal_node]['typecomponentnumber']

        selected_nodes = [n for n in whole_g.nodes if whole_g.nodes[n]['nodeclasstype']=='dendrite' and whole_g.nodes[n]['typecomponentnumber']==dendrite_component]

        sel_node_locs = [(int(whole_g.nodes[n]['x']), int(whole_g.nodes[n]['y']), int(whole_g.nodes[n]['z'])) for n in selected_nodes]

        final_data[agglo_id]['basal_node_locations'] = sel_node_locs

        info_to_get = [
            f'location.x*{syn_vx_size[0]} AS x', 
            f'location.y*{syn_vx_size[1]} AS y', 
            f'location.z*{syn_vx_size[2]} AS z', 
            'pre_synaptic_site.id AS pre_syn_id',
            'post_synaptic_partner.id AS post_syn_id',
            'type', 
            'pre_synaptic_site.neuron_id AS pre_id'
            ]


        raw_data = cf.get_info_from_bigquery(info_to_get, 'post_synaptic_partner.neuron_id', [agglo_id], syn_db_name, client)

        all_syn_locations = [(int(r['x']), int(r['y']), int(r['z'])) for r in raw_data]
        all_syn_types = [int(r['type']) for r in raw_data]
        all_syn_partners = [str(r['pre_id']) for r in raw_data]
        all_pre_syn_ids = [str(r['pre_syn_id']) for r in raw_data]
        all_post_syn_ids = [str(r['post_syn_id']) for r in raw_data]

        temp_dists = cdist(all_syn_locations, sel_node_locs, 'euclidean')

        close_enough = [min(x)<max_nm_from_syn for x in temp_dists]

        final_data[agglo_id]['basal_synapses'] = []

        zipped_data = zip(all_syn_locations, all_syn_types, all_syn_partners, all_pre_syn_ids, all_post_syn_ids, close_enough)

        for loc, ptype, pseg, pre_synid, post_synid, basal in zipped_data:

            if basal == True and pseg in pure_axons:

                final_data[agglo_id]['basal_synapses'].append({
                    'pre_seg_id': pseg,
                    'syn_location': loc,
                    'syn_type': ptype,
                    'pre_syn_id': pre_synid,
                    'post_syn_id': post_synid,
                })

    with open(save_path, 'w') as fp:
        json.dump(final_data,fp)

    # Save all input axons as separate list:
    all_axons = {}

    for agglo_id in final_data:
        if 'basal_synapses' in final_data[agglo_id]:
            all_axons_this_b_dendrite = set([x['pre_seg_id'] for x in final_data[agglo_id]['basal_synapses']])

            for axon in all_axons_this_b_dendrite:
                if axon not in all_axons:
                    all_axons[axon] = 0
                
                all_axons[axon] += 1

            
    multi_basal_d_axons = [x for x in all_axons.keys() if all_axons[x]>1]
    all_axons_flat_list = list(all_axons.keys())

    with open(all_axons_flat_list_save_path, 'w') as fp:
        json.dump(all_axons_flat_list, fp)

    with open(all_axons_multi_basal_d_save_path, 'w') as fp:
        json.dump(multi_basal_d_axons, fp)



