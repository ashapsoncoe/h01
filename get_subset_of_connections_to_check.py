import json
from random import sample, shuffle
from google.cloud import bigquery             
from google.oauth2 import service_account



input_list_dir = 'C:/work/FINAL/104_pr_neurons_maps/all_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2_goog14r0seg1.agg20200916c3_regions_types_circ_bounds.json'
num_each_type_to_sample = 50
syn_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2'
cred_path = 'c:/work/alexshapsoncoe.json'
save_path = 'c:/work/final/50_each_category_neuron_neuron_axon_to_dendritesomaais_connections_goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2.json'

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

    with open(input_list_dir, 'r') as fp:
        input_list = json.load(fp)

    all_types = set()

    for x in input_list:

        x['pre_ei_type'] = celltype_to_ei[x['pre_type']]
        x['post_ei_type'] = celltype_to_ei[x['post_type']]

        combo_type = f"{x['pre_region']} {x['pre_ei_type']} to {x['post_region']} {x['post_ei_type']}"

        x['combo_type'] = combo_type

        if 'White matter' not in combo_type and 'None' not in combo_type and 'unclassified' not in combo_type:
            all_types.add(combo_type)


    selected_connections = []

    for combo_type in all_types:

        print(combo_type)

        connections_this_type = [x for x in input_list if x['combo_type'] == combo_type]

        shuffle(connections_this_type)

        sample_this_type = sample(connections_this_type, min(len(connections_this_type), num_each_type_to_sample))

        pre_neurons = list(set([x['pre_seg_id'] for x in sample_this_type]))
        pre_neurons_joined = ','.join(pre_neurons)

        query = f"""
            SELECT
            location,
            pre_synaptic_site.neuron_id AS pre_neuron_id,
            pre_synaptic_site.id AS pre_syn_id,
            post_synaptic_partner.neuron_id AS post_neuron_id,
            post_synaptic_partner.id AS post_syn_id,
            LOWER(post_synaptic_partner.class_label) AS post_class_label,
            LOWER(pre_synaptic_site.class_label) AS pre_class_label
            FROM {syn_db}
            WHERE pre_synaptic_site.neuron_id IN ({pre_neurons_joined})
            AND
            pre_synaptic_site.class_label = 'AXON' 
            AND post_synaptic_partner.class_label IN ('DENDRITE', 'SOMA', 'AIS')

            """


        results = [dict(x) for x in client.query(query).result()]

        for conn in sample_this_type:

            pre_seg = conn['pre_seg_id']
            post_seg = conn['post_seg_id']

            rel_syn = [x for x in results if str(x['pre_neuron_id'])==pre_seg and str(x['post_neuron_id'])==post_seg]

            for syn in rel_syn:

                syn_loc = [syn['location'][a] for a in ('x', 'y', 'z')]
                syn_id = f"{syn['pre_syn_id']}_{syn['post_syn_id']}"

                final_syn_datum = {
                    'syn_loc': syn_loc, 
                    'syn_id': syn_id,
                    'pre_seg': pre_seg,
                    'post_seg': post_seg,
                    'connection_type': combo_type,
                    'post_class_label': syn['post_class_label'],
                    'pre_class_label': syn['pre_class_label'],
                    }

                selected_connections.append(final_syn_datum)


    with open(save_path, 'w') as fp:
        json.dump(selected_connections, fp)



'''
len([x for x in selected_connections if x['post_class_label']=='dendrite'])
len([x for x in selected_connections if x['post_class_label']=='ais'])
len([x for x in selected_connections if x['post_class_label']=='soma'])
'''
