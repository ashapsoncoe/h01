import json
import math
import os
from collections import Counter
from random import sample, shuffle

import common_functions as cf
import matplotlib.pyplot as plt
import neuroglancer
import numpy as np
import scipy
from google.cloud import bigquery, bigquery_storage
from google.oauth2 import service_account
from scipy import stats
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity
from statsmodels.stats.proportion import multinomial_proportions_confint

raw_data_dir = 'D:/sampled_synapse_points_random_sample_of_50000_axons_from_each_type_agg20200916c3_multisyn_only_unconstrained'
multi_synapse_axons_path = 'D:/random_sample_of_50000_axons_from_each_type_agg20200916c3_list_multisyn_only.json'
organised_axon_data_path = 'D:/random_sample_of_50000_axons_from_each_type_agg20200916c3_dict.json'
results_dir = 'D:/connection_strength_analysis_unconstrained'
synapse_db = 'goog14r0s5c3.synaptic_connections_with_skeleton_classes'
region_type_db = 'goog14r0seg1.agg20200916c3_regions_types'
syn_vx_size = {'x': 8, 'y': 8, 'z': 33}
cred_path = 'D:/alexshapsoncoe.json'


if __name__ == '__main__':

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)




    # Check there is a file available for each of the multisubset neurites:
    with open(multi_synapse_axons_path, 'r') as fp:
        multisyn_neurite_subset = json.load(fp)

    available_neurites = [x.split('_')[1] for x in os.listdir(raw_data_dir) if 'simulations.json' in x]

    assert set(multisyn_neurite_subset).issubset(set(available_neurites))

    with open(organised_axon_data_path, 'r') as fp:
        all_sampled_axons_data = json.load(fp)

    dtypes = ('simulated', 'real')
    axon_types = ('excitatory', 'inhibitory')

    all_counts = {layer: {axon_type: {dtype: {x: 0 for x in range(1, 1001)} for dtype in dtypes} for axon_type in axon_types} for layer in (all_sampled_axons_data)}
    all_counts_gp = {layer: {axon_type: {dtype: {x: 0 for x in range(1, 1001)} for dtype in dtypes} for axon_type in axon_types} for layer in (all_sampled_axons_data)}

    for layer in all_sampled_axons_data.keys():
        print(layer)
        for axon_type in all_sampled_axons_data[layer].keys():

            this_layer_type_axons = all_sampled_axons_data[layer][axon_type]
            this_layer_type_multis = set(this_layer_type_axons) & set(multisyn_neurite_subset)
            this_layer_type_singles = set(this_layer_type_axons) - this_layer_type_multis

            for axon_id in this_layer_type_singles:
                all_counts[layer][axon_type]['real'][1] += 1
                all_counts[layer][axon_type]['simulated'][1] += 1
                all_counts_gp[layer][axon_type]['real'][1] += 1
                all_counts_gp[layer][axon_type]['simulated'][1] += 1

            for axon_id in this_layer_type_multis:

                with open(f'{raw_data_dir}/neurite_{axon_id}_simulations.json', 'r') as fp:
                    raw_data = json.load(fp)

                simc_stalk = raw_data['stalk']['simulated_partners']
                simc_shaft = raw_data['shaft']['simulated_partners']
                least_samples_from_a_syn_id = min([len(simc_stalk[x]) for x in simc_stalk]+[len(simc_shaft[x]) for x in simc_shaft])

                all_sampled_ids = []
                n_syn_this_axon = 0

                for stype in ('stalk', 'shaft'):
                    
                    real_partners = [x['partner_id'] for x in raw_data[stype]['real_partners']]

                    partner_counts = list(Counter(real_partners).values())

                    for c in partner_counts:
                        all_counts[layer][axon_type]['real'][c] += 1

                    n_syn_this_axon += sum(partner_counts)

                    for syn_id in raw_data[stype]['simulated_partners']:
                        all_partners_this_syn_id = [x['partner_id'] for x in raw_data[stype]['simulated_partners'][syn_id]]

                        shuffle(all_partners_this_syn_id)
                        all_sampled_ids.extend(all_partners_this_syn_id[:max(10, least_samples_from_a_syn_id)])

                shuffle(all_sampled_ids)

                assert len(all_sampled_ids) >= n_syn_this_axon

                batched_simulations = [all_sampled_ids[x:x+n_syn_this_axon] for x in range(int(len(all_sampled_ids)/n_syn_this_axon))]

                batched_simulations = [x for x in batched_simulations if len(x)==n_syn_this_axon]

                sampled_patterns = []

                for sim in batched_simulations:

                    sampled_pattern = list(Counter(sim).values())

                    # for c in sampled_pattern:
                    #     all_counts[layer][axon_type]['simulated'][c] += 1

                    sampled_pattern.sort()
                    sampled_pattern = ','.join([str(y) for y in sampled_pattern])
                    sampled_patterns.append(sampled_pattern)

                most_common_pattern = [int(x) for x in Counter(sampled_patterns).most_common()[0][0].split(',')]

                for c in most_common_pattern:
                    all_counts[layer][axon_type]['simulated'][c] += 1

                strongest_connection = max(most_common_pattern)
                all_counts_gp[layer][axon_type]['simulated'][strongest_connection] += 1








    range_to_plot = range(1, 1001)

    all_counts_sim = {dtype: {x: 0 for x in range_to_plot} for dtype in list(axon_types)+['all']}
    all_counts_real = {dtype: {x: 0 for x in range_to_plot} for dtype in list(axon_types)+['all']}
    all_counts_sim_gp = {dtype: {x: 0 for x in range_to_plot} for dtype in list(axon_types)+['all']}
    all_counts_real_gp = {dtype: {x: 0 for x in range_to_plot} for dtype in list(axon_types)+['all']}

    for layer in all_counts:
        for axon_type in all_counts[layer]:
            for r in range_to_plot:
                all_counts_real[axon_type][r] += all_counts[layer][axon_type]['real'][r]
                all_counts_sim[axon_type][r] += all_counts[layer][axon_type]['simulated'][r]
                all_counts_real['all'][r] += all_counts[layer][axon_type]['real'][r]
                all_counts_sim['all'][r] += all_counts[layer][axon_type]['simulated'][r]
                all_counts_real_gp[axon_type][r] += all_counts_gp[layer][axon_type]['real'][r]
                all_counts_sim_gp[axon_type][r] += all_counts_gp[layer][axon_type]['simulated'][r]
                all_counts_real_gp['all'][r] += all_counts_gp[layer][axon_type]['real'][r]
                all_counts_sim_gp['all'][r] += all_counts_gp[layer][axon_type]['simulated'][r]

    # Get all counts:
    
    if f'{synapse_db}_connection_counts.json' not in os.listdir(results_dir):
        
        all_counts_total = {dtype: {x: 0 for x in range(1, 1001)} for dtype in list(axon_types)+['all']}
        all_counts_total_gp = {dtype: {x: 0 for x in range(1, 1001)} for dtype in list(axon_types)+['all']}

        for syn_type, syn_type_key in zip(axon_types, (2,1)):

            query = f"""WITH
                        all_edges AS (
                            SELECT 
                                CAST(pre_synaptic_site.neuron_id AS STRING) AS pre_seg_id, 
                                CAST(post_synaptic_partner.neuron_id AS STRING) AS post_seg_id, 
                                COUNT(*) AS pair_count
                            FROM {synapse_db}
                            where type = {syn_type_key}
                            GROUP BY pre_synaptic_site.neuron_id, post_synaptic_partner.neuron_id
                            ),


                        pure_axons AS (
                            select distinct CAST(agglo_id AS STRING) AS agglo_id 
                            from  {region_type_db}
                            where type = 'pure axon fragment'
                        )
                        select pair_count, pre_seg_id from all_edges A
                        inner join pure_axons B
                        on A.pre_seg_id = B.agglo_id
                        """

            df = client.query(query).result().to_dataframe(bqstorage_client=bqstorageclient) #, progress_bar_type='tqdm_gui')

            max_partner_lookup = {x: 0 for x in df['pre_seg_id']}

            for i in df.index:

                count = int(df.at[i, 'pair_count'])
                axon_id = str(df.at[i, 'pre_seg_id'])

                if count > max_partner_lookup[axon_id]:
                    max_partner_lookup[axon_id] = count

                if count in range_to_plot:
                    all_counts_total[syn_type][count] += 1
                    all_counts_total['all'][count] += 1
            
            for max_partner_count in max_partner_lookup.values():
                if max_partner_count in range_to_plot:
                    all_counts_total_gp[syn_type][max_partner_count] += 1
                    all_counts_total_gp['all'][max_partner_count] += 1
                    
        with open(f'{results_dir}/{synapse_db}_connection_counts.json', 'w') as fp:
            json.dump(all_counts_total, fp)

        with open(f'{results_dir}/{synapse_db}_greatest_partner_counts.json', 'w') as fp:
            json.dump(all_counts_total_gp, fp)
            
    else:
        with open(f'{results_dir}/{synapse_db}_connection_counts.json', 'r') as fp:
            all_counts_total = json.load(fp)
            
        all_counts_total = {stype: {int(k): all_counts_total[stype][k] for k in all_counts_total[stype]} for stype in all_counts_total}

        with open(f'{results_dir}/{synapse_db}_greatest_partner_counts.json', 'r') as fp:
            all_counts_total_gp = json.load(fp)
            
        all_counts_total_gp = {stype: {int(k): all_counts_total_gp[stype][k] for k in all_counts_total_gp[stype]} for stype in all_counts_total_gp}


for real_data_to_plot in ('total', 'sample'):

    for axon_type in ('excitatory', 'inhibitory', 'all'):

        if real_data_to_plot == 'sample':
            real_counts = deepcopy(all_counts_real_gp[axon_type])

        if real_data_to_plot == 'total':
            real_counts = deepcopy(all_counts_total_gp[axon_type])
            
        # Stop once get to first value with < 5 real counts:
        real_counts_to_test = []
        sim_counts_to_test = []

        for i in range(1, 1000):
            if real_counts[i] < 5: 
                print(i, real_counts[i])
                break
            real_counts_to_test.append(real_counts[i])
            sim_counts_to_test.append(all_counts_sim[axon_type][i])

        sim_props = [x/sum(sim_counts_to_test) for x in sim_counts_to_test]
        sim_counts_to_test = [round(x*sum(real_counts_to_test)) for x in sim_props]

        plt.figure(figsize=(20,10))
        p_val = scipy.stats.chisquare(real_counts_to_test, f_exp=sim_counts_to_test)[1]

        plt.plot(range(1, len(real_counts_to_test)+1), real_counts_to_test, '-o', color='red', label = 'Observed')
        plt.plot(range(1, sim_counts_to_test.index(0)+1), sim_counts_to_test[:sim_counts_to_test.index(0)], '-o', color='slateblue', label = 'Expected under null model')
        plt.yscale('log')
        plt.legend(loc='lower center', bbox_to_anchor=(0, 1))
        plt.xticks(range(1, len(real_counts_to_test)+1))
        plt.xlabel('Strongest connection of axon')
        plt.ylabel('Number of axons')
        pv_str = str(p_val)
        plt.title(f'Expected and Observed (in {real_data_to_plot}) Counts of Strongest Connection Stregnths for {axon_type} axons. P = {pv_str}')

        plt.savefig(f'{results_dir}/Expected and Observed (in {real_data_to_plot}) Counts of Strongest Connection Stregnths for {axon_type} axons.png')
        plt.clf()


