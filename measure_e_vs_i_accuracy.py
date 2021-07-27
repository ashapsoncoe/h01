import json
import pandas as pd
import common_functions as cf
import numpy as np
from google.cloud import bigquery             
from google.oauth2 import service_account
from sklearn.metrics import roc_auc_score


all_connections_dir = 'C:/work/FINAL/104_pr_neurons_maps/all_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2.json'
syn_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_spinecorrected'
cred_path = 'c:/work/alexshapsoncoe.json'
save_path_for_by_cell_df = 'c:/work/final/auc_e_vs_i_by_pre_cell_layer_conserv_reorient_fix_ei_spinecorrected_merge_correction2_07272021.csv'
save_path_for_syn_layer_df = 'c:/work/final/auc_e_vs_i_by_syn_layer_conserv_reorient_fix_ei_spinecorrected_merge_correction2_07272021.csv'
layer_bounds_path = 'c:/work/final/conical_bounds_final.json'
syn_voxel_size = [8,8,33]

layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']

input_data_dirs = [
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part1.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part3.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part3.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_part4.json',
    'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_Layer_5_excitatory_to_Layer_2_excitatory.json',
    'c:/work/final/excitatory_connections_in_layer_1.json',
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


    all_site_ids = [x['syn_id'].split('_')[0] for x in input_data]

    res = cf.get_info_from_bigquery(['type', 'pre_synaptic_site.id'], 'pre_synaptic_site.id', all_site_ids, syn_db, client)

    site_id2type = {str(r['id']): r['type'] for r in res}

    # Get accuracy by layer of pre-synaptic cell:
    site_id_to_ei_gt = {}

    auc_results_by_pre_cell_layer = {x: {} for x in layers}

    for layer in layers:

        verified_c_this_type = [x for x in input_data if layer in x['connection_type'].split(' to ')[0] and 'connection_decision' in x and x['connection_decision']=='true']

        ei_truth = []
        ei_predictions = []

        for verified_c in verified_c_this_type:

            pre_cell_type = verified_c['connection_type'].split(' to ')[0]

            pre_site_id = verified_c['syn_id'].split('_')[0]

            if 'excitatory' in pre_cell_type:
                site_id_to_ei_gt[pre_site_id] = 2
                ei_truth.append(2)
            else:
                assert 'inhibitory' in pre_cell_type
                site_id_to_ei_gt[pre_site_id] = 1
                ei_truth.append(1)

            pre_syn_site_id = verified_c['syn_id'].split('_')[0]
            ei_pred = site_id2type[pre_syn_site_id]
            ei_predictions.append(ei_pred)

        if len(set(ei_truth)) == 1:
            auc_results_by_pre_cell_layer[layer]['auc'] = 'only one value in GT'

        else:
            this_layer_roc_auc = roc_auc_score(ei_truth, ei_predictions)
            auc_results_by_pre_cell_layer[layer]['auc'] = this_layer_roc_auc

        true_e = [x for x in zip(ei_truth, ei_predictions) if x[0]==2]
        n_actual_e = len(true_e)

        if n_actual_e > 0:
            percent_e_correct = int((len([x for x in true_e if x[0]==x[1]])/n_actual_e)*100)
        else:
            percent_e_correct = 'NA'

        true_i = [x for x in zip(ei_truth, ei_predictions) if x[0]==1]
        n_actual_i = len(true_i)

        if n_actual_i > 0:
            percent_i_correct = int((len([x for x in true_i if x[0]==x[1]])/n_actual_i)*100)
        else:
            percent_i_correct = 'NA'

        auc_results_by_pre_cell_layer[layer]['Number of E synapses'] = n_actual_e
        auc_results_by_pre_cell_layer[layer]['Percent E synapses correct'] = percent_e_correct
        auc_results_by_pre_cell_layer[layer]['Number of I synapses'] = n_actual_i
        auc_results_by_pre_cell_layer[layer]['Percent I synapses correct'] = percent_i_correct
        auc_results_by_pre_cell_layer[layer]['Number of synpases total'] = len(ei_truth)

    auc_pre_cell_df = pd.DataFrame(auc_results_by_pre_cell_layer)

    auc_pre_cell_df.to_csv(save_path_for_by_cell_df)


    # Get by accuracy by layer in which synapse occurs:

    with open(layer_bounds_path, 'rb') as fp:
        layer_bounds = json.load(fp)

    verified_conns = [x for x in input_data if 'connection_decision' in x and x['connection_decision']=='true']

    true_syn_locs = [(x['syn_loc'], int(x['syn_id'].split('_')[0])) for x in verified_conns]

    true_syn_locs_nm = np.array([(pre_syn_id, loc[0]*syn_voxel_size[0],  loc[1]*syn_voxel_size[1]) for loc, pre_syn_id in true_syn_locs])

    site_ids_by_layer = cf.fix_layer_mem(layer_bounds, true_syn_locs_nm)[0]



    auc_results_by_syn_layer = {x: {} for x in layers}

    for layer in layers:

        site_ids_this_layer = site_ids_by_layer[layer]

        ei_predictions = [site_id2type[str(x)] for x in site_ids_this_layer]
        ei_truth = [site_id_to_ei_gt[str(x)] for x in site_ids_this_layer]

            
        if len(set(ei_truth)) == 1:
            auc_results_by_syn_layer[layer]['auc'] = 'only one value in GT'

        else:
            this_layer_roc_auc = roc_auc_score(ei_truth, ei_predictions)
            auc_results_by_syn_layer[layer]['auc'] = this_layer_roc_auc

        true_e = [x for x in zip(ei_truth, ei_predictions) if x[0]==2]
        n_actual_e = len(true_e)

        if n_actual_e > 0:
            percent_e_correct = int((len([x for x in true_e if x[0]==x[1]])/n_actual_e)*100)
        else:
            percent_e_correct = 'NA'

        true_i = [x for x in zip(ei_truth, ei_predictions) if x[0]==1]
        n_actual_i = len(true_i)

        if n_actual_i > 0:
            percent_i_correct = int((len([x for x in true_i if x[0]==x[1]])/n_actual_i)*100)
        else:
            percent_i_correct = 'NA'

        auc_results_by_syn_layer[layer]['Number of E synapses'] = n_actual_e
        auc_results_by_syn_layer[layer]['Percent E synapses correct'] = percent_e_correct
        auc_results_by_syn_layer[layer]['Number of I synapses'] = n_actual_i
        auc_results_by_syn_layer[layer]['Percent I synapses correct'] = percent_i_correct
        auc_results_by_syn_layer[layer]['Number of synpases total'] = len(ei_truth)

    auc_syn_layer_df = pd.DataFrame(auc_results_by_syn_layer)

    auc_syn_layer_df.to_csv(save_path_for_syn_layer_df)




'''
true_syn_locs_nm = np.array([(int(x['syn_id'].split('_')[0]), x['syn_loc'][0]*syn_voxel_size[0],  x['syn_loc'][1]*syn_voxel_size[1]) for x in all_syn_data])

site_ids_by_layer = cf.fix_layer_mem(layer_bounds, true_syn_locs_nm)[0]

in_layer1 = [x for x in all_syn_data if int(x['syn_id'].split('_')[0]) in site_ids_by_layer['Layer 1']]

inhibitory_in_layer_1 = [x for x in in_layer1 if 'inhibitory' in x['connection_type'].split(' to ')[0]]

with open(f'c:/work/final/inhibitory_connections_in_layer_1.json', 'w') as fp:
    json.dump(inhibitory_in_layer_1, fp)

'''