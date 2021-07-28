import json
import igraph as ig
import pandas as pd
import common_functions as cf
from google.cloud import bigquery             
from google.oauth2 import service_account
from collections import Counter
import numpy as np
from pandas.core.frame import DataFrame
from copy import deepcopy

edgelist_path = 'C:/work/FINAL/104_proofread_cells_edge_list_including_unproofread_targets_20200916c3.json'
cell_data_path = 'C:/work/FINAL/agglo_20200916c3_cell_data.json'
save_dir = 'c:/work/FINAL/104_pr_neurons_maps_circular_bounds_v'
seg_info_db = 'goog14r0seg1.agg20200916c3_regions_types_circ_bounds'
syn_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2'
cred_path = 'c:/work/alexshapsoncoe.json'
use_true_positions = True
layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']
neuron_types = ['excitatory', 'inhibitory']
plot_layer_shapes_for_proofread_network = False

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


chosen_col_shape = {
'excitatory': ['green', 'circle'], 
'inhibitory': ['red', 'circle'], 
}



def get_connectivity_dfs(layers, neuron_types, pre_neurons_to_include, all_cell_data, edge_list, type_lookup, layer_lookup, save_type):

    all_known_cells = set([x['agglo_seg'] for x in all_cell_data if 'neuron' in x['type']])

    edge_list = [e for e in edge_list if set(e[:2]).issubset(all_known_cells)]

    df_cols = [f'{layer} {n_type}' for layer in layers for n_type in neuron_types]

    all_dfs = {x: {} for x in ('syn_weighted', 'unweighted')}

    for count_type in ('syn_weighted', 'unweighted'):

        for layer in layers:

            for n_type in neuron_types:

                df_data = []

                n_pre_type = [x for x in layer_lookup.keys() if layer_lookup[x]==layer and celltype_to_ei[type_lookup[x]]==n_type and x in pre_neurons_to_include]

                for neuron_id in n_pre_type:

                    this_neuron_output_data = [neuron_id]

                    outgoing_partners = [x[1:] for x in edge_list if x[0]==neuron_id]

                    if count_type == 'unweighted':

                        partner_types = [f'{layer_lookup[x[0]]} {celltype_to_ei[type_lookup[x[0]]]}' for x in outgoing_partners]
                    
                    if count_type == 'syn_weighted':

                        partner_types = [[f'{layer_lookup[x[0]]} {celltype_to_ei[type_lookup[x[0]]]}' for a in range(x[1])] for x in outgoing_partners]

                        partner_types = [a for b in partner_types for a in b]

                    counts = Counter(partner_types)

                    for post_type in df_cols:

                        if post_type in counts:
                            this_neuron_output_data.append(counts[post_type])
                        else:
                            this_neuron_output_data.append(0)
                    
                    df_data.append(this_neuron_output_data)

                this_pre_df = pd.DataFrame(df_data, columns=['seg_id']+df_cols)

                all_dfs[count_type][f'{layer} {n_type}'] = this_pre_df

                this_pre_df.to_csv(f'{save_dir}/{save_type}_{count_type}_{layer}_{n_type}_output_summary.csv')
                    
    return all_dfs     

def plot_canonical_circuit(all_dfs, use_true_positions, file_name, weighted=True):

    df_cols = list(list(all_dfs.values())[0].columns)[1:]

    edges = []

    for pre_type in df_cols:

        for post_type in df_cols:

            edge_weight = np.mean(all_dfs[pre_type][post_type])

            if edge_weight > 0:

                edge = [pre_type, post_type, edge_weight]
                edges.append(edge)


    g = ig.Graph(directed=True)
    g.add_vertices(df_cols)
    g.add_edges([x[:2] for x in edges])
    g.es['weight'] = [x[2] for x in edges]

    if weighted == True:
        edge_weights = [x['weight'] for x in g.es]
    else:
        edge_weights = [1 for x in g.es]

    vertex_shapes = []
    node_colours = []
    node_labels = []
    xy_pos = []

    for n in g.vs:

        node_name = n['name']

        for n_type in chosen_col_shape:

            if n_type in node_name:

                vertex_shapes.append(chosen_col_shape[n_type][1])
                node_colours.append(chosen_col_shape[n_type][0])

        
        layer_num = [int(s) for s in node_name.split() if s.isdigit()][0]

        # displacement_dict = {
        #     1: 1,
        #     2: 10,
        #     3: 20,
        #     4: 30,
        #     5: 40,
        #     6: 50,
        # }

        if 'excitatory' in node_name:
            x_pos = layer_num**3 #displacement_dict[layer_num]

        if 'inhibitory' in node_name:
            x_pos = layer_num**3 + 6**3 #displacement_dict[layer_num] + max(displacement_dict.values())

        xy_pos.append([x_pos, layer_num])

        node_labels.append(layer_num)

        #node_labels.append(len(all_dfs[node_name]))

    if use_true_positions == False:
        layout = g.layout_auto()
    else:
        layout = xy_pos

    ig.plot(
                g, 
                target=f'{save_dir}/{file_name}.png',
                margin = (200,200,200,200), 
                bbox = (2000,2000), 
                edge_width = edge_weights,
                #edge_color =  edge_colours,
                vertex_label = node_labels,
                #vertex_label_color = node_lab_colours,
                #vertex_label_dist = [0 for v in sg.vs],
                vertex_label_font = [100 for v in g.vs],
                vertex_shape = vertex_shapes, 
                vertex_color = node_colours, 
                vertex_size = 50, 
                edge_arrow_size=1.0, 
                layout=layout
                )

def get_syn_and_conn_counts(all_dfs):

    all_connection_types_found = set()

    df_cols = [f'{layer} {n_type}' for layer in layers for n_type in neuron_types]

    synapse_counts = pd.DataFrame([[0 for x in df_cols] for y in df_cols], index=df_cols, columns=df_cols)
    connection_counts = pd.DataFrame([[0 for x in df_cols] for y in df_cols], index=df_cols, columns=df_cols)
    synapse_counts_norm = pd.DataFrame([[0 for x in df_cols] for y in df_cols], index=df_cols, columns=df_cols, dtype=float)
    connection_counts_norm = pd.DataFrame([[0 for x in df_cols] for y in df_cols], index=df_cols, columns=df_cols, dtype=float)

    for pre_type in all_dfs['syn_weighted']:

        total_syn_this_pre = sum([sum(all_dfs['syn_weighted'][pre_type][post_type]) for post_type in all_dfs['syn_weighted'][pre_type].columns if post_type != 'seg_id'])
        total_conn_this_pre = sum([sum(all_dfs['unweighted'][pre_type][post_type]) for post_type in all_dfs['unweighted'][pre_type].columns if post_type != 'seg_id'])

        for post_type in all_dfs['syn_weighted'][pre_type].columns:

            if post_type == 'seg_id': continue

            n_syn = sum(all_dfs['syn_weighted'][pre_type][post_type])
            synapse_counts.at[pre_type, post_type] = n_syn

            if total_syn_this_pre > 0:
                synapse_counts_norm.at[pre_type, post_type] = n_syn/total_syn_this_pre
            else:
                synapse_counts_norm.at[pre_type, post_type] = 0

            n_conn = sum(all_dfs['unweighted'][pre_type][post_type])

            connection_counts.at[pre_type, post_type] = n_conn

            if total_conn_this_pre > 0:
                connection_counts_norm.at[pre_type, post_type] = n_conn/total_conn_this_pre
            else:
                connection_counts_norm.at[pre_type, post_type] = 0

            if n_conn > 0:
                all_connection_types_found.add(f'{pre_type} {post_type}')

    return synapse_counts, connection_counts, all_connection_types_found, synapse_counts_norm, connection_counts_norm




if __name__ == '__main__':

    with open(cell_data_path, 'r') as fp:
        all_cell_data = json.load(fp)

    type_lookup = {x['agglo_seg']: x['type'] for x in all_cell_data}
    loc_lookup = {x['agglo_seg']: (x['true_x'], x['true_y'])  for x in all_cell_data}

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    res = cf.get_info_from_bigquery(['agglo_id', 'region'], 'agglo_id', list(loc_lookup.keys()), seg_info_db, client)
    layer_lookup = {x['agglo_id']: x['region'] for x in res}

    with open(edgelist_path, 'r') as fp:
        manual_edge_list = json.load(fp)

    # Make manual true circuit plot:

    accepted_neuron_types = set(['pyramidal neuron', 'excitatory/spiny neuron with atypical tree', 'interneuron','spiny stellate neuron'])

    g = ig.Graph(directed=True)
    all_vertices = list(set([str(a) for b in [e[:2] for e in manual_edge_list if type_lookup[e[0]] in accepted_neuron_types and type_lookup[e[1]] in accepted_neuron_types] for a in b]))
    g.add_vertices(all_vertices)
    g.add_edges([x[:2] for x in manual_edge_list if set(x[:2]).issubset(set(all_vertices))])
    g.es['weight'] = [x[2] for x in manual_edge_list if set(x[:2]).issubset(set(all_vertices))]

    edge_weights = [x['weight'] for x in g.es]
            
    vertex_shapes = []
    node_colours = []
    xy_cb_pos = []

    for n in g.vs:

        seg_id = n['name']

        cell_type = type_lookup[seg_id]

        assert cell_type in accepted_neuron_types

        if plot_layer_shapes_for_proofread_network == False:

            save_path = f'{save_dir}/manual_plot_without_layer_shapes.png'

            if cell_type in ('pyramidal neuron', 'excitatory/spiny neuron with atypical tree'):

                vertex_shapes.append('triangle')
                node_colours.append('orange')

            if cell_type == 'interneuron':

                vertex_shapes.append('circle')
                node_colours.append('blue')

            if cell_type == 'spiny stellate neuron':

                vertex_shapes.append('circle')
                node_colours.append('orange')

        else:

            save_path = f'{save_dir}/manual_plot_with_layer_shapes.png'

            if cell_type in ('pyramidal neuron', 'excitatory/spiny neuron with atypical tree', 'spiny stellate neuron'):
                node_colours.append('orange')

            if cell_type == 'interneuron':
                node_colours.append('blue')

            layer2shapes = {
                'Layer 1': 'circle', 
                'Layer 2': 'square', 
                'Layer 3': 'circle', 
                'Layer 4': 'square', 
                'Layer 5': 'circle',
                'Layer 6': 'square', 
                'White matter': 'circle', 
                'unclassified': 'triangle',
            }

            layer = layer_lookup[seg_id]
            vertex_shapes.append(layer2shapes[layer])


        xy_cb_pos.append(loc_lookup[seg_id])


    if use_true_positions:
        layout = xy_cb_pos
    else:
        layout = g.layout_auto()

#max([g.degree(n.index) for n in g.vs])

    ig.plot(
                g, 
                target=save_path,
                margin = (100,100,100,100), 
                bbox = (2000,1000), 
                edge_width = edge_weights,
                #edge_color =  edge_colours,
                #vertex_label = node_labels,
                #vertex_label_color = node_lab_colours,
                #vertex_label_dist = [0 for v in sg.vs],
                #vertex_label_font = [0.1 for v in sg.vs],
                vertex_shape = vertex_shapes, 
                vertex_color = node_colours, 
                vertex_size = 20, 
                edge_arrow_size=1.0, 
                layout=layout
                )


    # Make manual canonical circuit plot:
    pre_neurons_to_include = set([x[0] for x in manual_edge_list])

    all_manual_dfs = get_connectivity_dfs(layers, neuron_types, pre_neurons_to_include, all_cell_data, manual_edge_list, type_lookup, layer_lookup, 'manual')
    
    plot_canonical_circuit(all_manual_dfs['unweighted'], True, 'manual_graph')
    plot_canonical_circuit(all_manual_dfs['unweighted'], True, 'manual_graph_unweighted', weighted=False)

    manual_synapse_counts, manual_connection_counts, manual_connection_types_found, manual_synapse_counts_norm, manual_connection_counts_norm = get_syn_and_conn_counts(all_manual_dfs)

    manual_synapse_counts.to_csv(f'{save_dir}/manual_connectivity_summary_counts_synapse_numbers.csv')
    manual_connection_counts.to_csv(f'{save_dir}/manual_connectivity_summary_counts_connection_numbers.csv')
    manual_synapse_counts_norm.to_csv(f'{save_dir}/manual_connectivity_summary_counts_synapse_numbers_norm.csv')
    manual_connection_counts_norm.to_csv(f'{save_dir}/manual_connectivity_summary_counts_connection_numbers_norm.csv')

    # Get ML-generated network


    query = f"""
    with 

        acceptable_syn AS (
            SELECT *
            FROM {syn_db}
            WHERE pre_synaptic_site.class_label = 'AXON' 
            AND post_synaptic_partner.class_label IN ('DENDRITE', 'SOMA', "AIS")
        ),


        all_edges AS (
        SELECT 
            CAST(pre_synaptic_site.neuron_id AS STRING) AS pre_seg_id, 
            CAST(post_synaptic_partner.neuron_id AS STRING) AS post_seg_id, 
            COUNT(*) AS pair_count
        FROM acceptable_syn
        GROUP BY pre_synaptic_site.neuron_id, post_synaptic_partner.neuron_id
        ),

    all_edges_with_pre AS (
        SELECT pre_seg_id, post_seg_id, pair_count, B.type AS pre_type, B.region AS pre_region
        from all_edges A
        inner join {seg_info_db} B
        on A.pre_seg_id = B.agglo_id 
    ),

    all_edges_with_both AS (
        SELECT pre_seg_id, post_seg_id, pair_count, pre_type, pre_region, B.type AS post_type, B.region AS post_region
        from all_edges_with_pre A
        inner join {seg_info_db} B
        on A.post_seg_id = B.agglo_id 
    )

    select * from all_edges_with_both
    WHERE post_type LIKE '%neuron%' AND pre_type LIKE '%neuron%'
    """

    result = [dict(x) for x in client.query(query).result()]

    with open(f'{save_dir}/all_neuron_neuron_axon_to_dendritesomaais_connections_{syn_db}_{seg_info_db}.json', 'w') as fp:
        json.dump(result, fp)

    automatic_edge_list = [[x['pre_seg_id'], x['post_seg_id'], x['pair_count']] for x in result]

    pre_neurons_to_include = set([a for b in [x[:2] for x in automatic_edge_list] for a in b])

    all_auto_dfs = get_connectivity_dfs(layers, neuron_types, pre_neurons_to_include, all_cell_data, automatic_edge_list, type_lookup, layer_lookup, 'automatic')
    plot_canonical_circuit(all_auto_dfs['unweighted'], True, 'ml_graph')
    plot_canonical_circuit(all_auto_dfs['unweighted'], True, 'ml_graph_unweighted', weighted=False)


    auto_synapse_counts, auto_connection_counts, auto_connection_types_found, auto_synapse_counts_norm, auto_connection_counts_norm = get_syn_and_conn_counts(all_auto_dfs)

    auto_synapse_counts.to_csv(f'{save_dir}/ml_connectivity_summary_counts_synapse_numbers.csv')
    auto_connection_counts.to_csv(f'{save_dir}/ml_connectivity_summary_counts_connection_numbers.csv')
    auto_synapse_counts_norm.to_csv(f'{save_dir}/ml_connectivity_summary_counts_synapse_numbers_norm.csv')
    auto_connection_counts_norm.to_csv(f'{save_dir}/ml_connectivity_summary_counts_connection_numbers_norm.csv')







'''
# Some counts:
total_pre_seg = 0
pre_seg_with_1plus_syn = 0

for pre_type in all_auto_dfs['syn_weighted']:

    for row in all_auto_dfs['syn_weighted'][pre_type].index:

        total_pre_seg += 1

        n_syn = sum(list(all_auto_dfs['syn_weighted'][pre_type].loc[row])[1:])

        if n_syn > 0:
            pre_seg_with_1plus_syn += 1
'''





