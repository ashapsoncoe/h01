import json
import common_functions as cf
from google.cloud import bigquery             
from google.oauth2 import service_account
import os

input_dir = 'C:/work/FINAL/proofread104_neurons_20210511'
syn_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2'
credentials_path = 'c:/work/alexshapsoncoe.json'
output_path = 'C:/work/FINAL/104_proofread_cells_edge_list_including_unproofread_targets_20200916c3_no_ais.json'
file2aggloid_output_path = 'C:/work/FINAL/file2aggloid_for_104_proofread_cells_20200916c3_no_ais.json'
agglo_db = 'goog14r0seg1.agg20200916c3_resolved_fixed'
cell_ids = 'C:/work/FINAL/agglo_20200916c3_cell_data.json'
include_unproofread_targets = True
exclude_ais_syn = False


if __name__ == "__main__":

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    with open(cell_ids, 'r') as fp:
        all_cell_data = json.load(fp)

    neurons = set([x['agglo_seg'] for x in all_cell_data if 'neuron' in x['type']])
    all_cells = set([x['agglo_seg'] for x in all_cell_data])

    pr_data = {}


    all_base_ids = [f.split('_')[2] for f in os.listdir(input_dir)]
    base2agglo = cf.get_base2agglo(all_base_ids, agglo_db, client)
    all_agglo_ids = set([base2agglo[x] for x in all_base_ids])

    file2aggloid = {f: base2agglo[f.split('_')[2]] for f in os.listdir(input_dir)}

    for f in os.listdir(input_dir):

        with open(f'{input_dir}/{f}', 'r') as fp:
            temp = json.load(fp)

        all_base_ids = [a for b in temp['base_segments'].values() for a in b]
        base2agglo = cf.get_base2agglo(all_base_ids, agglo_db, client)

        base_id = f.split('_')[2]
        
        if base2agglo[base_id] in all_cells:
            agglo_id = base2agglo[base_id]
        else:
            all_agglo_ids = set(base2agglo.values())

            involved_neurons = all_agglo_ids & neurons

            if len(involved_neurons) > 0:
                agglo_id = list(involved_neurons)[0]
            else:
                agglo_id = base2agglo[base_id]
                print(agglo_id)
        
        pr_data[agglo_id] = temp['base_segments']






    final_outputs = {x: {} for x in pr_data.keys()}

    for pre_agglo in pr_data.keys():

        print(pre_agglo)

        if len(pr_data[pre_agglo]['axon']) == 0:
            print('no axonal segs')
            continue

        rel_ids = ','.join([str(x) for x in pr_data[pre_agglo]['axon']])

        query = f"""SELECT
            pre_synaptic_site.base_neuron_id AS pre_base,
            post_synaptic_partner.base_neuron_id AS post_base,
            pre_synaptic_site.neuron_id AS pre_agglo,
            post_synaptic_partner.neuron_id AS post_agglo,
            pre_synaptic_site.id AS pre_syn_id,
            post_synaptic_partner.id AS post_syn_id
            from {syn_db}
            WHERE
                CAST(pre_synaptic_site.base_neuron_id AS INT64) IN ({rel_ids})
            """
        
        if exclude_ais_syn == True:
            extra_bit = " AND NOT post_synaptic_partner.class_info = 'AIS'"
            query = query + extra_bit

        query_job = client.query(query)  
        results = [dict(row) for row in query_job.result()]

        for syn in results:

            post_agglo =str(syn['post_agglo'])

            if post_agglo not in pr_data.keys() and include_unproofread_targets==False:
                continue

            if (post_agglo in neurons) or (post_agglo in pr_data.keys()):
                
                if post_agglo not in final_outputs[pre_agglo]:
                    final_outputs[pre_agglo][post_agglo] = []

                syn_info = {x: syn[x] for x in ('pre_base', 'post_base', 'pre_syn_id', 'post_syn_id')}

                final_outputs[pre_agglo][post_agglo].append(syn_info)
        
#[len(final_outputs[x].keys()) for x in final_outputs]

    simple_edge_list = []

    for source in final_outputs.keys():
        for target in final_outputs[source].keys():
            n_syn = len(final_outputs[source][target])
            simple_edge_list.append([source, target, n_syn])

    with open(output_path, 'w') as fp:
        json.dump(simple_edge_list, fp)

    with open(file2aggloid_output_path, 'w') as fp:
        json.dump(file2aggloid, fp)

