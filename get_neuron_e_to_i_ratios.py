from google.cloud import bigquery             
from google.oauth2 import service_account
import common_functions as cf
import json
import pandas as pd
import numpy as np


credentials_file = 'C:/work/alexshapsoncoe.json'
results_file = 'C:/work/FINAL/goog14r0s5c3_e_i_ratios_v6.csv'
syn_db_name = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_spinecorrected'
cell_ids = 'C:/work/FINAL/agglo_20200916c3_cell_data.json'
layer_bounds = "C:/work/FINAL/conical_bounds_final.json"

'''
df = pd.read_csv(results_file)

import math

ex_indices = [x for x in df.index if df.at[x, 'cell type'] in ('pyramidal neuron', 'excitatory/spiny neuron with atypical tree', 'spiny stellate neuron')]


ex_percent_ex = [df.at[x, 'excitatory synapses']/(df.at[x, 'excitatory synapses']+df.at[x, 'inhibitory synapses']) for x in ex_indices]
np.mean([a for a in ex_percent_ex if math.isnan(a)==False])



in_indices = [x for x in df.index if df.at[x, 'cell type'] in ('interneuron')]


ex_percent_in = [df.at[x, 'excitatory synapses']/(df.at[x, 'excitatory synapses']+df.at[x, 'inhibitory synapses']) for x in in_indices]
np.mean([a for a in ex_percent_in if math.isnan(a)==False])
'''


if __name__ == '__main__':

    with open(cell_ids, 'r') as fp:
        all_cell_data = json.load(fp)

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    all_neurons = list(set([str(int(float(x['agglo_seg']))) for x in all_cell_data])) #if 'neuron' in x['type']]))

    raw_data = cf.get_info_from_bigquery(['type', 'post_synaptic_partner.neuron_id AS agglo_id'], 'post_synaptic_partner.neuron_id', all_neurons, syn_db_name, client, batch_size=1000)

    aggloid2type = {str(int(float(x['agglo_seg']))): x['type'] for x in all_cell_data} # if 'neuron' in x['type']}


    with open(layer_bounds, "r") as f:
        bounds = json.load(f)

    id_and_xy = [[int(float(x['agglo_seg'])), int(x['true_x']), int(x['true_y'])] for x in all_cell_data]# if 'neuron' in x['type']]

    id_and_xy = np.array(id_and_xy)

    shard_layers = cf.fix_layer_mem(bounds, id_and_xy)[0]

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

