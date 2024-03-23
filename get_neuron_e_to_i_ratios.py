import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

from google.cloud import bigquery             
from google.oauth2 import service_account
from common_functions_h01 import get_info_from_bigquery, fix_layer_mem, get_base2agglo
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.transforms as mt
from matplotlib import collections  as mc
import matplotlib.patches as patches


credentials_file = 'alexshapsoncoe.json'
results_file = 'goog14r0s5c3_e_i_ratios_march_2024'
syn_db_name = 'lcht-goog-connectomics.goog14r0s5c3.synapse_c3_eirepredict_clean_dedup'
syn_db_vx_size = [8,8,33]
cell_ids = 'agglo_20200916c3_cell_data.json'
layer_bounds = 'cortical_bounds_circles.json'
pr_spines_ng_state_file = 'proofread_spines_neuron_30535700448_c3_agglomeration.json'
pr_cells = False # If false, will use all pyramidal and interneurons from the c3 agglomeration
pr_cells_input_dir = 'proofread104_neurons_20210511' # Only needed if pr_cells == True
agglo_db = 'goog14r0seg1.agg20200916c3_resolved_fixed' # Only needed if pr_cells == True
adjust_syn_fp_an_fn = True


def project_e_and_i_counts(raw_e_count, raw_i_count):

    fnr_e = 0.11
    fnr_i = 0.35
    fdr_e = 0.032
    fdr_i = 0.027
    false_classification_rate_e_pred = 0.1151832461
    false_classification_rate_i_pred = 0.17
    correct_classification_rate_e = 0.8689
    correct_classification_rate_i = 0.8498

    predicted_e_nofp = raw_e_count*(1-fdr_e)
    e_predictions_actually_e = predicted_e_nofp*(1-false_classification_rate_e_pred)
    e_predictions_actually_i = predicted_e_nofp*false_classification_rate_e_pred


    predicted_i_nofp = raw_i_count*(1-fdr_i)
    i_predictions_actually_i = predicted_i_nofp*(1-false_classification_rate_i_pred)
    i_predictions_actually_e = predicted_i_nofp*false_classification_rate_i_pred

    projected_total_e = e_predictions_actually_e+i_predictions_actually_e / (1-fnr_e)
    projected_total_i = i_predictions_actually_i+e_predictions_actually_i / (1-fnr_i)

    return projected_total_e, projected_total_i













def get_organised_inputs_to_pr_neurons(input_dir, agglo_db, all_cells, all_neurons, syn_db_name, client):

    agglo_id_to_syn = {}

    for f in os.listdir(input_dir):
 
        with open(f'{input_dir}/{f}', 'r') as fp:
            temp = json.load(fp)

        pr_base_segs_dict = temp['base_segments']

        all_base_ids = [a for b in pr_base_segs_dict.values() for a in b]
        base2agglo = get_base2agglo(all_base_ids, agglo_db, client)

        base_id = f.split('_')[2]
        
        if base2agglo[base_id] in all_cells:
            agglo_id = str(base2agglo[base_id])
        else:
            all_agglo_ids = set(base2agglo.values())

            involved_neurons = all_agglo_ids & all_neurons

            if len(involved_neurons) > 0:
        
                agglo_id = str(list(involved_neurons)[0])
            else:
                agglo_id = str(base2agglo[base_id])
        
        pr_base_ids = ','.join([str(x) for x in all_base_ids])

        query = f"""SELECT DISTINCT
            type,
            CAST(location.x*{syn_db_vx_size[0]} AS INT) AS x,
            CAST(location.y*{syn_db_vx_size[0]} AS INT) AS y,
            from {syn_db_name}
            WHERE
                CAST(post_synaptic_partner.base_neuron_ids[0] AS INT64) IN ({pr_base_ids})
            """

        query_job = client.query(query)  

        agglo_id_to_syn[agglo_id] = [dict(row) for row in query_job.result()]

    return agglo_id_to_syn



if __name__ == '__main__':

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    # Estimate effect of missing spines from proofread cell on E:I ratio
    with open(pr_spines_ng_state_file, 'r') as fp:
        pr_spines_ng = json.load(fp)

    all_pr_segs = pr_spines_ng['layers'][1]['segments']
    spine_segs_only = pr_spines_ng['layers'][2]['segments']
    pre_spine_pr_segs = [x for x in all_pr_segs if x not in spine_segs_only]

    syn_without_sep_spines = get_info_from_bigquery(['type', 'location.x', 'location.y', 'location.z'], 'post_synaptic_partner.neuron_id', pre_spine_pr_segs, syn_db_name, client, batch_size=1000)

    syn_sep_spines_only = get_info_from_bigquery(['type', 'location.x', 'location.y', 'location.z'], 'post_synaptic_partner.neuron_id', spine_segs_only, syn_db_name, client, batch_size=1000)


    n_e_syn_without_sep_spines = len([x for x in syn_without_sep_spines if x['type']==2])
    n_i_syn_without_sep_spines = len([x for x in syn_without_sep_spines if x['type']==1])
    n_e_syn_sep_spines_only = len([x for x in syn_sep_spines_only if x['type']==2])
    n_i_syn_sep_spines_only = len([x for x in syn_sep_spines_only if x['type']==1])

    n_e_syn_all = n_e_syn_without_sep_spines+n_e_syn_sep_spines_only
    n_i_syn_all = n_i_syn_without_sep_spines+n_i_syn_sep_spines_only

    print(f'Proportion of E synapses without separate spines: {n_e_syn_without_sep_spines/len(syn_without_sep_spines)}, n = {len(syn_without_sep_spines)}')
    print(f'Proportion of E synapses separate spines only: {n_e_syn_sep_spines_only/len(syn_sep_spines_only)}, n = {len(syn_sep_spines_only)}')
    print(f'Proportion of E synapses all segments: {n_e_syn_all/(n_e_syn_all+n_i_syn_all)}, n = {n_e_syn_all+n_i_syn_all}')


    old_to_new_format_naming = {
        'Layer 1': 'layer_1', 
        'Layer 2': 'layer_2', 
        'Layer 3': 'layer_3', 
        'Layer 4': 'layer_4', 
        'Layer 5': 'layer_5', 
        'Layer 6': 'layer_6', 
        'White matter': 'white_matter'
        }
    
    new_to_old_format_naming = {v: k for k, v in old_to_new_format_naming.items()}

    with open(cell_ids, 'r') as fp:
        all_cell_data = json.load(fp)

    aggloid2type = {str(int(float(x['agglo_seg']))): x['type'] for x in all_cell_data}

    all_neurons = set([x['agglo_seg'] for x in all_cell_data if 'neuron' in x['type']])
    all_cells = set([x['agglo_seg'] for x in all_cell_data])

    with open(layer_bounds, "r") as f:
        bounds = json.load(f)

    id_and_xy = [[int(float(x['agglo_seg'])), int(x['true_x']), int(x['true_y'])] for x in all_cell_data]

    id_and_xy = np.array(id_and_xy)

    shard_layers = fix_layer_mem(bounds, id_and_xy)[0]

    aggloid2layer = {}
    for layer in shard_layers.keys():
        for seg_id in shard_layers[layer]:
            aggloid2layer[str(seg_id)] = old_to_new_format_naming[layer]


    if pr_cells == False:

        info_to_get = [
            'type', 
            f'CAST(location.x*{syn_db_vx_size[0]} AS INT) as x',
            f'CAST(location.y*{syn_db_vx_size[0]} AS INT) as y',
            'post_synaptic_partner.neuron_id AS agglo_id'
        ]

        raw_data = get_info_from_bigquery(info_to_get, 'post_synaptic_partner.neuron_id', list(all_neurons), syn_db_name, client, batch_size=1000)

        agglo_id_to_syn = {}

        for syn in raw_data:

            agglo_id = str(syn['agglo_id'])

            if agglo_id not in agglo_id_to_syn:
                agglo_id_to_syn[agglo_id] = []

            agglo_id_to_syn[agglo_id].append({k: syn[k] for k in syn if k!='agglo_id'})

    if pr_cells == True:
        
        agglo_id_to_syn = get_organised_inputs_to_pr_neurons(pr_cells_input_dir, agglo_db, all_cells, all_neurons, syn_db_name, client)


    counts_and_type = {}

    all_layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5', 'layer_6', 'white_matter']

    for agglo_id in agglo_id_to_syn:

        counts_and_type[agglo_id] = {}
        counts_and_type[agglo_id]['c3_agglomeration_id'] = agglo_id
        counts_and_type[agglo_id]['cell_type'] = aggloid2type[agglo_id]
        counts_and_type[agglo_id]['cell_body_layer'] = aggloid2layer[agglo_id]

        syn_array = np.array([(syn['type'], syn['x'], syn['y']) for syn in agglo_id_to_syn[agglo_id]])

        ei_count_by_layer = fix_layer_mem(bounds, syn_array)[0]

        for cort_layer in ['all_inputs'] + all_layers:

            if cort_layer == 'all_inputs':
                syn_this_layer = [a for b in ei_count_by_layer.values() for a in b]
            else:
                syn_this_layer = ei_count_by_layer[new_to_old_format_naming[cort_layer]]

            e_count_this_layer = syn_this_layer.count(2)
            i_count_this_layer = syn_this_layer.count(1)

            if adjust_syn_fp_an_fn == True:

                e_count_this_layer, i_count_this_layer = project_e_and_i_counts(e_count_this_layer, i_count_this_layer)

            counts_and_type[agglo_id][f'e_count_{cort_layer}'] = e_count_this_layer
            counts_and_type[agglo_id][f'i_count_{cort_layer}'] = i_count_this_layer

            assert syn_this_layer.count(1)+syn_this_layer.count(2) == len(syn_this_layer)

            if len(syn_this_layer) > 0:
                counts_and_type[agglo_id][f'e_prop_{cort_layer}'] = e_count_this_layer / (e_count_this_layer+i_count_this_layer)
            else:
                counts_and_type[agglo_id][f'e_prop_{cort_layer}'] = 'NA'



    ind_cell_df = pd.DataFrame(counts_and_type).transpose()

    if pr_cells == True:
        addendum = 'pr_neurons_only'
    else:
        addendum = 'all_neurons'

    if adjust_syn_fp_an_fn == True:
        addendum2 = 'fpfn_adjusted'
    else:
        addendum2 = 'unadjusted'

    ind_cell_df.to_csv(f'{results_file}_{addendum}_{addendum2}.csv')

    ind_cell_df_previous = pd.read_csv(f'{results_file}_{addendum}.csv', index_col=0)


    # Statistical analysis:
    columns = [
        'volumetric_e_prop',
        'layer_1_pyramidal_neurons_n',
        'layer_1_pyramidal_neurons_inputs_e_prop_mean',
        'layer_1_pyramidal_neurons_inputs_e_prop_sd',
        'layer_1_interneurons_n',
        'layer_1_interneurons_inputs_e_prop_mean',
        'layer_1_interneurons_inputs_e_prop_sd',
        'layer_1_pyr_vs_int_tvalue',
        'layer_1_pyr_vs_int_pvalue',
        'layer_2_pyramidal_neurons_n',
        'layer_2_pyramidal_neurons_inputs_e_prop_mean',
        'layer_2_pyramidal_neurons_inputs_e_prop_sd',
        'layer_2_interneurons_n',
        'layer_2_interneurons_inputs_e_prop_mean',
        'layer_2_interneurons_inputs_e_prop_sd',
        'layer_3_pyramidal_neurons_n',
        'layer_3_pyramidal_neurons_inputs_e_prop_mean',
        'layer_3_pyramidal_neurons_inputs_e_prop_sd',
        'layer_3_interneurons_n',
        'layer_3_interneurons_inputs_e_prop_mean',
        'layer_3_interneurons_inputs_e_prop_sd',
        'layer_4_pyramidal_neurons_n',
        'layer_4_pyramidal_neurons_inputs_e_prop_mean',
        'layer_4_pyramidal_neurons_inputs_e_prop_sd',
        'layer_4_interneurons_n',
        'layer_4_interneurons_inputs_e_prop_mean',
        'layer_4_interneurons_inputs_e_prop_sd',
        'layer_5_pyramidal_neurons_n',
        'layer_5_pyramidal_neurons_inputs_e_prop_mean',
        'layer_5_pyramidal_neurons_inputs_e_prop_sd',
        'layer_5_interneurons_n',
        'layer_5_interneurons_inputs_e_prop_mean',
        'layer_5_interneurons_inputs_e_prop_sd',
        'layer_6_pyramidal_neurons_n',
        'layer_6_pyramidal_neurons_inputs_e_prop_mean',
        'layer_6_pyramidal_neurons_inputs_e_prop_sd',
        'layer_6_interneurons_n',
        'layer_6_interneurons_inputs_e_prop_mean',
        'layer_6_interneurons_inputs_e_prop_sd',
        'white_matter_pyramidal_neurons_n',
        'white_matter_pyramidal_neurons_inputs_e_prop_mean',
        'white_matter_pyramidal_neurons_inputs_e_prop_sd',
        'white_matter_interneurons_n',
        'white_matter_interneurons_inputs_e_prop_mean',
        'white_matter_interneurons_inputs_e_prop_sd',
    ]

    all_data = []

    percentage_e_per_layer = {  # Excitatory percentage by layer - see 'plot_synapse_density_and_ei_ratio.py' for derivation
        'white_matter': 70.86277572905995, 
        'layer_6': 74.65357381549842, 
        'layer_5': 76.93247559133833, 
        'layer_4': 76.23462577888455, 
        'layer_3': 77.05790755129284, 
        'layer_2': 70.88836685180763, 
        'layer_1': 64.03911757963151, 
    }

    for syn_cort_layer in ['all_inputs'] + all_layers:

        data_this_syn_cort_layer = []

        if syn_cort_layer == 'all_inputs':
            volumetric_prop = ''
        else:
            volumetric_prop = percentage_e_per_layer[syn_cort_layer]/100

        data_this_syn_cort_layer.append(volumetric_prop)

        for cb_cort_layer in all_layers:

            pyr_eprop = [x for x in ind_cell_df[(ind_cell_df.cell_type == 'pyramidal neuron') & (ind_cell_df.cell_body_layer == cb_cort_layer)][f'e_prop_{syn_cort_layer}'] if x != 'NA']
            int_eprop = [x for x in ind_cell_df[(ind_cell_df.cell_type == 'interneuron') & (ind_cell_df.cell_body_layer == cb_cort_layer)][f'e_prop_{syn_cort_layer}'] if x != 'NA']
            
            data_this_syn_cort_layer.append(len(pyr_eprop))

            if len(pyr_eprop) > 0:
                data_this_syn_cort_layer.append(np.mean(pyr_eprop))
                data_this_syn_cort_layer.append(np.std(pyr_eprop))
            else:
                data_this_syn_cort_layer.append('NA')
                data_this_syn_cort_layer.append('NA')

            data_this_syn_cort_layer.append(len(int_eprop))

            if len(int_eprop) > 0:
                data_this_syn_cort_layer.append(np.mean(int_eprop))
                data_this_syn_cort_layer.append(np.std(int_eprop))
            else:
                data_this_syn_cort_layer.append('NA')
                data_this_syn_cort_layer.append('NA')

            if syn_cort_layer == 'all_inputs' and cb_cort_layer != 'layer_1':
                t_statistic, p_value = stats.ttest_ind(list(pyr_eprop), list(int_eprop), equal_var=False)
            else:
                t_statistic, p_value = '', ''

            data_this_syn_cort_layer.append(t_statistic)
            data_this_syn_cort_layer.append(p_value)
            
        
        assert len(data_this_syn_cort_layer) == 45

        all_data.append(data_this_syn_cort_layer)

    meta_df = pd.DataFrame(all_data, columns=columns).transpose()
    meta_df.columns = ['all_inputs'] + all_layers
    meta_df.to_csv(f'{results_file}_meta_analysis_{addendum}_{addendum2}.csv')

    # Plot
    x_start = 0
    x_fin = 22
    y_start = 6.5
    y_fin = 0
    e_colour = 'orange'
    i_colour = 'navy'#'dodgerblue'
    even_layer_col = 'aqua'
    odd_layer_col = 'mediumspringgreen'
    percent_col = 'white'
    ex_block_col = 'firebrick'
    inh_block_col = 'royalblue'
    top_row_label_size = 7
    numbers_label_size = 7
    min_neurons_for_layer  = 100
    add_numbers = True
    ais_thickness = 0.5
    ic_spacing = 1.5
    apical_thickness = 6
    triangle_size = 0.4
    circle_rad = 0.4
    dendrite_offset = 0.4
    move_cell_lat = 0.15

    fig, ax = plt.subplots() 

    y_tick_labels = ['', '', 'I', '', 'II', '', 'III', '', 'IV', '', 'V', '', 'VI' ,'']
    y_tick_locs = [0, 0.5, 1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5]

    ax.set_xlim(x_start,x_fin*ic_spacing)
    ax.set_ylim(y_start,y_fin)
    ax.set_yticks(y_tick_locs)
    ax.set_yticklabels(y_tick_labels)

    ax.get_xaxis().set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    for pos, ytick in enumerate(yticks):
        if pos%2 == 0:
            ytick.tick1line.set_markersize(0)




    # Make layers different colours:
    for layer in range(0,6):

        ax.hlines(layer+0.5, xmin=x_start, xmax=x_fin*2, linewidth=ais_thickness, colors='black')


    # Plot cells:
    for layer in range(1,7):
    
        if layer >1:
            pyr_eprop_means_this_layer = {syn_layer: meta_df[f'layer_{syn_layer}'][f'layer_{layer}_pyramidal_neurons_inputs_e_prop_mean'] for syn_layer in range(1, 7) if meta_df[f'layer_{syn_layer}'][f'layer_{layer}_pyramidal_neurons_n'] >=min_neurons_for_layer}
        else:
            pyr_eprop_means_this_layer = {0:0}

        int_eprop_means_this_layer = {syn_layer: meta_df[f'layer_{syn_layer}'][f'layer_{layer}_interneurons_inputs_e_prop_mean'] for syn_layer in range(1, 7) if meta_df[f'layer_{syn_layer}'][f'layer_{layer}_interneurons_n'] >=min_neurons_for_layer}

        highest_input = {'inhibitory': min(int_eprop_means_this_layer.keys()), 'excitatory': min(pyr_eprop_means_this_layer.keys())}
        lowest_input = {'inhibitory': max(int_eprop_means_this_layer.keys()), 'excitatory': max(pyr_eprop_means_this_layer.keys())}

        centre_point = {
            'excitatory': ((layer-1.25)*ic_spacing*2, layer+0.00),
            'inhibitory': (((layer-1.25)*ic_spacing*2)+(6*ic_spacing*2), layer+0.00),
        }

        pts = np.array([
            [centre_point['excitatory'][0]+move_cell_lat-triangle_size, centre_point['excitatory'][1]+triangle_size], 
            [centre_point['excitatory'][0]+move_cell_lat+triangle_size,centre_point['excitatory'][1]+triangle_size], 
            [centre_point['excitatory'][0]+move_cell_lat,centre_point['excitatory'][1]-triangle_size]])

        if layer != 1:
            
            ax.vlines(x=centre_point['excitatory'][0]+move_cell_lat, ymin=min(highest_input['excitatory'],layer)-dendrite_offset, ymax=max(lowest_input['excitatory'],layer)+dendrite_offset, color=e_colour, linewidth = apical_thickness)    
            ax.vlines(x=centre_point['excitatory'][0]+(ic_spacing/2), ymin=y_fin, ymax=y_start, color='black', linewidth = ais_thickness)
            triangle = Polygon(pts, closed=True, color=e_colour, ec=None)
            ax.add_patch(triangle)
            ax.text(centre_point['excitatory'][0]-1.5, 0.3, ' E     %     I', ha="center", va='center', fontsize=top_row_label_size)
            ax.text(centre_point['inhibitory'][0]-1.5, 0.3, ' E     %     I', ha="center", va='center', fontsize=top_row_label_size)

            for l in pyr_eprop_means_this_layer:


                rectangle_width = 1.75
                e_start = centre_point['excitatory'][0]-2.25
                e_width = pyr_eprop_means_this_layer[l]*rectangle_width
                i_start = e_start+e_width
                i_width = (1-pyr_eprop_means_this_layer[l])*rectangle_width

                e_rectangle = patches.Rectangle((e_start, l-0.5), e_width, 1, facecolor=ex_block_col)
                e_rectangle.set_linewidth(0)
                ax.add_patch(e_rectangle)
                i_rectangle = patches.Rectangle((i_start, l-0.5), i_width, 1, facecolor=inh_block_col)
                i_rectangle.set_linewidth(0)
                ax.add_patch(i_rectangle)

                if add_numbers == True:

                    e_percent = int(round(pyr_eprop_means_this_layer[l]*100, 0))
                    i_percent = 100-e_percent
            
                    ax.text(e_start+0.15, l, f'{e_percent}         {i_percent}', ha="left", va='center', fontsize=numbers_label_size, color=percent_col)
            

                
                    
        else:

            ax.hlines(y=0.5, xmin=x_fin*ic_spacing, xmax=x_start, color='black', linewidth = 0.5)
            ax.text(centre_point['inhibitory'][0]-1.4, 0.3, 'E     %     I', ha="center", va='center', fontsize=top_row_label_size)
        
        for l in int_eprop_means_this_layer:

            rectangle_width = 1.75
            e_start = centre_point['inhibitory'][0]-2.25
            e_width = int_eprop_means_this_layer[l]*rectangle_width
            i_start = e_start+e_width
            i_width = (1-int_eprop_means_this_layer[l])*rectangle_width

            e_rectangle = patches.Rectangle((e_start, l-0.5), e_width, 1, facecolor=ex_block_col)
            e_rectangle.set_linewidth(0)
            ax.add_patch(e_rectangle)
            i_rectangle = patches.Rectangle((i_start, l-0.5), i_width, 1, facecolor=inh_block_col)
            i_rectangle.set_linewidth(0)
            ax.add_patch(i_rectangle)

            if add_numbers == True:

                e_percent = int(round(int_eprop_means_this_layer[l]*100, 0))
                i_percent = 100-e_percent
        
                ax.text(e_start+0.15, l, f'{e_percent}         {i_percent}', ha="left", va='center', fontsize=numbers_label_size, color=percent_col)
        

                
        ax.vlines(x=centre_point['inhibitory'][0], ymin=min(highest_input['inhibitory'],layer)-dendrite_offset, ymax=max(lowest_input['inhibitory'],layer)+dendrite_offset, color=i_colour, linewidth = apical_thickness) 
        ax.vlines(x=centre_point['inhibitory'][0]+(ic_spacing/2), ymin=y_fin, ymax=y_start, color='black', linewidth = ais_thickness)
        circle = plt.Circle(centre_point['inhibitory'], circle_rad, color=i_colour, ec=None)
        ax.add_patch(circle)
        

    plt.show()


