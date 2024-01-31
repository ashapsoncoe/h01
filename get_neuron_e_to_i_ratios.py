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
import matplotlib.pyplot as plt
from scipy import stats
import math
from scipy import stats


credentials_file = 'alexshapsoncoe.json'
results_file = 'goog14r0s5c3_e_i_ratios_jan_2024.csv'
syn_db_name = 'lcht-goog-connectomics.goog14r0s5c3.synapse_c3_eirepredict_clean_dedup'
cell_ids = 'agglo_20200916c3_cell_data.json'
layer_bounds = 'cortical_bounds_circles.json'


if __name__ == '__main__':

    with open(cell_ids, 'r') as fp:
        all_cell_data = json.load(fp)

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    all_neurons = list(set([str(int(float(x['agglo_seg']))) for x in all_cell_data]))

    raw_data = get_info_from_bigquery(['type', 'post_synaptic_partner.neuron_id AS agglo_id'], 'post_synaptic_partner.neuron_id', all_neurons, syn_db_name, client, batch_size=1000)

    aggloid2type = {str(int(float(x['agglo_seg']))): x['type'] for x in all_cell_data}


    with open(layer_bounds, "r") as f:
        bounds = json.load(f)

    id_and_xy = [[int(float(x['agglo_seg'])), int(x['true_x']), int(x['true_y'])] for x in all_cell_data]

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
    df['e_prop'] = df['excitatory synapses'] / (df['excitatory synapses']+df['inhibitory synapses'])
    df.to_csv(results_file, index=0)


    all_data = []

    cort_layers = ('Layer 1', 'Layer 2', 'Layer 3', 'Layer 4','Layer 5', 'Layer 6')

    for layer in cort_layers:

        print(f'For {layer}')

        if layer == 'Layer 1':
            all_data.append([df.loc[x, 'e_prop'] for x in df.index if df.loc[x, 'cortical layer']==layer and df.loc[x, 'cell type']=='interneuron' and math.isnan(df.loc[x, 'e_prop'])==False])
            print(f'Interneuron mean e proportion {np.mean(all_int)}')
            print(f'Interneuron std e proportion {np.std(all_int)}')
            print(f'Interneuron N {len(all_int)}')
        else:
            all_pyr = [df.loc[x, 'e_prop'] for x in df.index if df.loc[x, 'cortical layer']==layer and df.loc[x, 'cell type']=='pyramidal neuron' and math.isnan(df.loc[x, 'e_prop'])==False]
            all_int = [df.loc[x, 'e_prop'] for x in df.index if df.loc[x, 'cortical layer']==layer and df.loc[x, 'cell type']=='interneuron' and math.isnan(df.loc[x, 'e_prop'])==False]
            t_statistic, p_value = stats.ttest_ind(all_pyr, all_int, equal_var=False)
            print(f'Pyramidal cell mean e proportion {np.mean(all_pyr)}')
            print(f'Pyramidal cell std e proportion {np.std(all_pyr)}')
            print(f'Pyramidal cell N {len(all_int)}')
            print(f'Interneuron mean e proportion {np.mean(all_int)}')
            print(f'Interneuron std e proportion {np.std(all_int)}')
            print(f'Interneuron N {len(all_pyr)}')
            print(f'Independent t test p value {p_value}, t value {t_statistic}')
            all_data.append(all_pyr)
            all_data.append(all_int)


    x_pos = range(1,12)

    c = plt.violinplot(dataset=all_data, positions=x_pos, showmeans=True, showmedians=False, showextrema=False)

    for pc in c['bodies']:
        _ = pc.set_edgecolor('black')
        _ = pc.set_linewidth(1)

    # Customize colors
    colours = ['blue', 'blue', 'orange','blue', 'orange','blue', 'orange','blue', 'orange','blue', 'orange']
    for i, pc in enumerate(plt.gca().collections):
        if i < 11:
            _ = pc.set_facecolor(colours[i])


    _ = plt.errorbar(x=x_pos, y=[np.mean(x) for x in all_data], yerr=[np.std(x) for x in all_data], fmt='none', capsize=3, color='grey', alpha=0.5)
    _ = plt.ylabel('E / (E+I)')
    tick_positions = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]
    _ = plt.xticks(tick_positions, [f'Layer 1                       ']+[f'{x}                                          ' for x in cort_layers[1:]])
    plt.show()


