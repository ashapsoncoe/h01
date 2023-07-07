import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

from google.cloud import bigquery             
from google.oauth2 import service_account
from common_functions_h01 import get_info_from_bigquery, fix_layer_mem
import json
import pandas as pd
import numpy as np


credentials_file = 'alexshapsoncoe.json'
results_file = 'goog14r0s5c3_e_i_ratios_feb_2023.csv'
syn_db_name = 'lcht-goog-connectomics.goog14r0s5c3.synapse_c3_eirepredict_clean_dedup'
cell_ids = 'agglo_20200916c3_cell_data.json'
layer_bounds = 'cortical_bounds_circles.json'


if __name__ == '__main__':

    with open(cell_ids, 'r') as fp:
        all_cell_data = json.load(fp)

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    all_neurons = list(set([str(int(float(x['agglo_seg']))) for x in all_cell_data])) #if 'neuron' in x['type']]))

    raw_data = get_info_from_bigquery(['type', 'post_synaptic_partner.neuron_id AS agglo_id'], 'post_synaptic_partner.neuron_id', all_neurons, syn_db_name, client, batch_size=1000)

    aggloid2type = {str(int(float(x['agglo_seg']))): x['type'] for x in all_cell_data} # if 'neuron' in x['type']}


    with open(layer_bounds, "r") as f:
        bounds = json.load(f)

    id_and_xy = [[int(float(x['agglo_seg'])), int(x['true_x']), int(x['true_y'])] for x in all_cell_data]# if 'neuron' in x['type']]

    id_and_xy = np.array(id_and_xy)

    shard_layers = fix_layer_mem(bounds, id_and_xy)[0]

    aggloid2layer = {}
    for layer in shard_layers.keys():
        for seg_id in shard_layers[layer]:
            aggloid2layer[str(seg_id)] = layer

    counts_and_type = {x: {'e': 0, 'i': 0, 'type': aggloid2type[x], 'layer': aggloid2layer[x]} for x in all_neurons}

    for r in raw_data:

        if r['type'] == 1:
            counts_and_type[str(r['agglo_id'])]['i'] += 1

        if r['type'] == 2:
            counts_and_type[str(r['agglo_id'])]['e'] += 1

    data = [[k, counts_and_type[k]['e'], counts_and_type[k]['i'], counts_and_type[k]['layer'], counts_and_type[k]['type']] for k in counts_and_type.keys()]
    df = pd.DataFrame(data, columns=['segment id', 'excitatory synapses', 'inhibitory synapses', 'cortical layer', 'cell type'])

        
    df.to_csv(results_file, index=0)

