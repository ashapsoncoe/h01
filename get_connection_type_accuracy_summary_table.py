import json
import pandas as pd
import common_functions as cf
import numpy as np
from google.cloud import bigquery             
from google.oauth2 import service_account


save_path_connection_summaries = 'c:/work/final/pairwise_connection_checks_circle_layer_bounds_20210715.csv'
save_path_example_true_connections = 'c:/work/final/example_connections_of_each_type_circle_layer_bounds_20210715.csv'
verified_false_edges_save_path = 'c:/work/final/verified_false_edges_agg20200916c3_20210716.json'
verified_true_edges_save_path = 'c:/work/final/verified_true_edges_agg20200916c3_20210716.json'
all_connections_dir = 'C:/work/FINAL/104_pr_neurons_maps/all_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2.json'
layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']
neuron_types = ['excitatory', 'inhibitory']
syn_voxel_size = [8,8,33]
layer_bounds_path = 'c:/work/final/conical_bounds_final.json'
syn_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2'
cred_path = 'c:/work/alexshapsoncoe.json'
seg_class_update_table = 'goog14r0seg1.agg20200916c3_regions_types_circ_bounds'

input_data_dirs = [
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part1.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part3.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part3.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part4.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_Layer_5_excitatory_to_Layer_2_excitatory.json',
    'c:/work/final/extra_39_connections_to_check_after_boundary_change.json',

]

celltype_to_ei = {
'pyramidal neuron': 'excitatory', 
'excitatory/spiny neuron with atypical tree': 'excitatory', 
'interneuron': 'inhibitory', 
'spiny stellate neuron': 'excitatory',
'blood vessel cell': None, 
'unclassified neuron': None, 
'microglia/opc': None, 
'astrocyte': None, 
'c-shaped cell': None, 
'unknown cell': None, 
'oligodendrocyte': None,
}

if __name__ == '__main__':

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    input_data = []

    for data_dir in input_data_dirs:

        with open(data_dir, 'r') as fp:
            input_datum = json.load(fp)

        input_data.extend(input_datum)

    with open(all_connections_dir, 'r') as fp:
        all_connections = json.load(fp)

    with open(layer_bounds_path, 'rb') as fp:
        layer_bounds = json.load(fp)

    # Update layers and types:
    if seg_class_update_table != None:

        agglo_ids = list(set([x['pre_seg_id'] for x in all_connections]+[x['post_seg_id'] for x in all_connections]))

        results = cf.get_info_from_bigquery(['agglo_id', 'region', 'type'], 'agglo_id', agglo_ids, seg_class_update_table, client)

        segid2type = {str(x['agglo_id']): celltype_to_ei[x['type']] for x in results}
        segid2region = {str(x['agglo_id']): x['region'] for x in results}

        for conn in all_connections:

            for dtype in ('pre', 'post'):

                seg_id = conn[f'{dtype}_seg_id']

                if seg_id in segid2region:

                    conn[f'{dtype}_region'] = segid2region[seg_id]

        all_connections = [x for x in all_connections if x['pre_region'] in layers and x['post_region'] in layers]

        for datum in input_data:

            new_overall_types = []

            for pos, dtype in enumerate(('pre', 'post')):

                seg_id = datum[f'{dtype}_seg']

                if seg_id in segid2region:

                    new_region = segid2region[seg_id]

                    new_type = datum['connection_type'].split(' to ')[pos].split(' ')[-1]

                    new_overall_type = f'{new_region} {new_type}'

                else:
                    new_overall_type = datum['connection_type'].split(' to ')[pos]
                
                new_overall_types.append(new_overall_type)

            datum['connection_type'] = ' to '.join(new_overall_types)

        input_data = [x for x in input_data if ' '.join(x['connection_type'].split(' to ')[0].split(' ')[:2]) in layers]
        input_data = [x for x in input_data if ' '.join(x['connection_type'].split(' to ')[1].split(' ')[:2]) in layers]
        
        
    df_cols = [f'{layer} {n_type}' for layer in layers for n_type in neuron_types]

    starting_data = []

    for pre_type in df_cols:

        this_pre_list = []

        all_connections_this_pre = [x for x in all_connections if f"{x['pre_region']} {celltype_to_ei[x['pre_type']]}" == pre_type]

        for post_type in df_cols:

            n_conn_this_type = len([x for x in all_connections_this_pre if f"{x['post_region']} {celltype_to_ei[x['post_type']]}" == post_type])
            
            this_pre_list.append(f'{n_conn_this_type}t')

        starting_data.append(this_pre_list)

    total_true_connections = 0
    most_common_conn_type = None
    most_common_conn_type_count = 0

    accuracy = pd.DataFrame(starting_data, index=df_cols, columns=df_cols)

    checked_types = set([x['connection_type'] for x in input_data if 'connection_decision' in x])

    verified_false_edges = []
    verified_true_edges = []

    for checked_type in checked_types:

        data_this_type = [x for x in input_data if 'connection_decision' in x and x['connection_type']==checked_type]

        true_syn_locs = [(x['syn_loc'], int(x['pre_seg'])) for x in data_this_type if x['connection_decision']=='true']

        true_syn_locs_nm = np.array([(pre_seg, loc[0]*syn_voxel_size[0],  loc[1]*syn_voxel_size[1]) for loc, pre_seg in true_syn_locs])

        if len(true_syn_locs_nm) > 0:

            cortical_layers, cortical_layers_coords = cf.fix_layer_mem(layer_bounds, true_syn_locs_nm)

            layer_num = max([(k, len(cortical_layers[k])) for k in cortical_layers], key = lambda x: x[1])[0][-1]

        else:
            layer_num = ' '

        checked_conns_this_type = set([f"{x['pre_seg']}_{x['post_seg']}" for x in data_this_type])

        n_tp = 0

        for checked_conn in checked_conns_this_type:

            decisions = [x['connection_decision'] for x in data_this_type if f"{x['pre_seg']}_{x['post_seg']}" == checked_conn]

            pre_seg_id, post_seg_id = checked_conn.split('_')

            if 'true' in decisions:
                n_tp += 1
                verified_true_edges.append((pre_seg_id, post_seg_id))
            else:
                verified_false_edges.append((pre_seg_id, post_seg_id))


        n_checked = len(checked_conns_this_type)

        tp_rate = int((n_tp/n_checked)*100)

        pre_type, post_type = checked_type.split(' to ')

        result = f'{accuracy.at[pre_type, post_type]}, {n_checked}c, {tp_rate}%, L{layer_num}'

        accuracy.at[pre_type, post_type] = result

        ml_found_conn_this_type = len([x for x in all_connections if checked_type == f"{x['pre_region']} {celltype_to_ei[x['pre_type']]} to {x['post_region']} {celltype_to_ei[x['post_type']]}"])

        predicted_total_count_this_type = (n_tp/n_checked)*ml_found_conn_this_type

        total_true_connections += predicted_total_count_this_type

        if predicted_total_count_this_type > most_common_conn_type_count:
            most_common_conn_type = checked_type
            most_common_conn_type_count = predicted_total_count_this_type

    accuracy.to_csv(save_path_connection_summaries)

    with open(verified_true_edges_save_path, 'w') as fp:
        json.dump(verified_true_edges, fp)

    with open(verified_false_edges_save_path, 'w') as fp:
        json.dump(verified_false_edges, fp)

    # Then get a unique exmaple of each type:

    syn_id_to_class = {}

    for pos, p in enumerate(['pre_synaptic_site', 'post_synaptic_partner']):

        all_syn_ids = [x['syn_id'].split('_')[pos] for x in input_data]

        results = cf.get_info_from_bigquery([f'{p}.id', f'{p}.class_label'], f'{p}.id', all_syn_ids, syn_db, client)

        for r in results:
            syn_id_to_class[str(r['id'])] = r['class_label']

    individual_examples = []
    indv_ex_types_already_obs = set()


    true_cases = [x for x in input_data if 'connection_decision' in x and x['connection_decision']=='true']

    for true_case in true_cases:

        syn_loc = np.array([(1, true_case['syn_loc'][0]*syn_voxel_size[0],  true_case['syn_loc'][1]*syn_voxel_size[1])])

        cortical_layers = cf.fix_layer_mem(layer_bounds, syn_loc)[0]

        layer_num = max([(k, len(cortical_layers[k])) for k in cortical_layers], key = lambda x: x[1])[0][-1]

        pre_syn_id = true_case['syn_id'].split('_')[0]
        post_syn_id = true_case['syn_id'].split('_')[1]

        pre_struc = syn_id_to_class[pre_syn_id]
        post_struc = syn_id_to_class[post_syn_id]

        pre_layer, pre_type = true_case['connection_type'].split(' to ')[0].split(' ')[1:]
        post_layer, post_type = true_case['connection_type'].split(' to ')[1].split(' ')[1:]

        if pre_type == 'excitatory' and post_struc == 'AIS': continue

        loc_string = '  '.join([str(x) for x in true_case['syn_loc']])

        this_datum = [
            pre_layer, pre_type, post_layer, post_type, 
            pre_struc, post_struc, layer_num, 
            true_case['pre_seg'], true_case['post_seg'],
            pre_syn_id, post_syn_id, loc_string,
            ]
        
        conn_type = pre_layer + pre_type + post_layer + post_type + pre_struc + post_struc

        if conn_type in indv_ex_types_already_obs:
            continue
        else:
            indv_ex_types_already_obs.add(conn_type)
            individual_examples.append(this_datum)


    col_names = ['pre_layer', 'pre_type', 'post_layer', 'post_type', 
            'pre_structure', 'post_structure', 'synapse_layer', 
            'pre_segment_id', 'post_segment_id',
            'pre_syn_id', 'post_syn_id', 'synapse_location',
            ]

    df = pd.DataFrame(individual_examples, columns = col_names)

    df.to_csv(save_path_example_true_connections)






'''
new_v = pd.read_csv(save_path_example_true_connections)
old_v = pd.read_csv('c:/work/final/example_connections_of_each_type_20210714.csv')

old_v_types = set([
    f"  {old_v.at[x, 'pre_layer']}, {old_v.at[x, 'pre_type']}, {old_v.at[x, 'post_layer']}, {old_v.at[x, 'post_type']}, {old_v.at[x, 'pre_structure']}, {old_v.at[x, 'post_structure']}" 
        for x in old_v.index])

new_v_types = set([
    f"  {new_v.at[x, 'pre_layer']}, {new_v.at[x, 'pre_type']}, {new_v.at[x, 'post_layer']}, {new_v.at[x, 'post_type']}, {new_v.at[x, 'pre_structure']}, {new_v.at[x, 'post_structure']}" 
        for x in new_v.index])
'''
