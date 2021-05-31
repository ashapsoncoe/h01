import json
import numpy as np
import common_functions as cf
from collections import Counter
import matplotlib.pyplot as plt
import os
from random import shuffle
from scipy import stats
from google.cloud import bigquery        
from google.cloud import bigquery_storage         
from google.oauth2 import service_account


balanced_strength_data = 'c:/work/final/random_sample_of_500_axons_from_each_gp_stregnth_and_type_agg20200916c3_dict.json'
balanced_strength_sample_dir = 'D:/sampled_synapse_points_random_sample_of_500_axons_from_each_gp_stregnth_and_type_agg20200916c3_unconstrained'
results_dir = 'c:/work/final/connection_strength_analysis_unconstrained_v2'
synapse_db = 'goog14r0s5c3.synaptic_connections_with_skeleton_classes'
seg_info_db = 'goog14r0seg1.agg20200916c3_regions_types'
cred_path = 'c:/work/alexshapsoncoe.json'


def get_patterns_from_simulated_neurite(raw_data):

    simc_stalk = raw_data['stalk']['simulated_partners']
    simc_shaft = raw_data['shaft']['simulated_partners']
    least_samples_from_a_syn_id = min([len(simc_stalk[x]) for x in simc_stalk]+[len(simc_shaft[x]) for x in simc_shaft])

    all_sampled_ids = []
    all_real_ids = []

    for stype in ('stalk', 'shaft'):
        
        real_partners_this_type = [x['partner_id'] for x in raw_data[stype]['real_partners']]

        all_real_ids.extend(real_partners_this_type)

        for syn_id in raw_data[stype]['simulated_partners']:
            all_partners_this_syn_id = [x['partner_id'] for x in raw_data[stype]['simulated_partners'][syn_id]]

            shuffle(all_partners_this_syn_id)
            all_sampled_ids.extend(all_partners_this_syn_id[:max(10, least_samples_from_a_syn_id)])


    n_syn_this_axon = len(all_real_ids)

    shuffle(all_sampled_ids)

    assert len(all_sampled_ids) >= n_syn_this_axon

    batched_simulations = [all_sampled_ids[x:x+n_syn_this_axon] for x in range(int(len(all_sampled_ids)/n_syn_this_axon))]

    batched_simulations = [x for x in batched_simulations if len(x)==n_syn_this_axon]

    sampled_patterns = []

    for sim in batched_simulations:

        sampled_pattern = list(Counter(sim).values())
        sampled_pattern.sort()
        sampled_pattern = ','.join([str(y) for y in sampled_pattern])
        sampled_patterns.append(sampled_pattern)


    most_common_sim_pattern = [int(x) for x in Counter(sampled_patterns).most_common()[0][0].split(',')]

    real_pattern = list(Counter(all_real_ids).values())

    real_pattern.sort()

    assert sum(real_pattern) == sum(most_common_sim_pattern)

    return most_common_sim_pattern, real_pattern


if __name__ == '__main__':

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)


    # FIRST GET TOTAL COUNTS FOR NON-AIS CONTACTING PURE AXONS:
    axon_types = ('excitatory', 'inhibitory')


    if not os.path.exists(f'{results_dir}/all_counts_total_{synapse_db}.json'):

        all_counts_total = {dtype: {x: 0 for x in range(1000)} for dtype in axon_types}
        all_counts_total_gp = {dtype: {x: 0 for x in range(1000)} for dtype in axon_types}

        for syn_type in axon_types:

            if syn_type == 'inhibitory':
                where_clause = 'InhibCount > ExciteCount'

            if syn_type == 'excitatory':
                where_clause = 'InhibCount < ExciteCount'


            query = f"""with pure_axons as (
                        select CAST(agglo_id AS STRING) as agglo_id
                        from {seg_info_db}
                        where type = 'pure axon fragment'
                        ),

                        all_edges AS (
                        SELECT 
                            CAST(pre_synaptic_site.neuron_id AS STRING) AS pre_seg_id, 
                            CAST(post_synaptic_partner.neuron_id AS STRING) AS post_seg_id, 
                            COUNT(*) AS pair_count
                        FROM {synapse_db}
                        GROUP BY pre_synaptic_site.neuron_id, post_synaptic_partner.neuron_id
                        ),

                        e_and_i_counts as (
                        SELECT CAST(pre_synaptic_site.neuron_id AS STRING) AS agglo_id,
                            count(*) AS total,
                            sum(case when type = 1 then 1 else 0 end) AS InhibCount,
                            sum(case when type = 2 then 1 else 0 end) AS ExciteCount
                        FROM {synapse_db}
                        GROUP BY agglo_id
                        ),

                        this_type_pre_segs as (
                        SELECT agglo_id,
                        FROM e_and_i_counts
                        WHERE {where_clause} 
                        ),

                        pure_axons_making_synapses_this_type as (
                            select agglo_id from this_type_pre_segs
                            intersect distinct
                            select agglo_id from pure_axons
                        ),

                        segments_synapsing_onto_ais as (
                            select distinct CAST(pre_synaptic_site.neuron_id AS STRING) as seg_id
                            from {synapse_db}
                            where LOWER(post_synaptic_partner.class_label) = 'ais'
                        ),

                        non_ais_axons_this_type as (
                        select distinct agglo_id from pure_axons_making_synapses_this_type A
                        left join segments_synapsing_onto_ais B
                        on A.agglo_id = B.seg_id 
                        where B.seg_id IS NULL
                        )

                        select pair_count, pre_seg_id from all_edges A
                        inner join non_ais_axons_this_type B
                        on A.pre_seg_id = B.agglo_id
                        """

            df = client.query(query).result().to_dataframe(bqstorage_client=bqstorageclient) #, progress_bar_type='tqdm_gui')

            max_partner_lookup = {x: 0 for x in df['pre_seg_id']}

            for i in df.index:

                count = int(df.at[i, 'pair_count'])
                axon_id = str(df.at[i, 'pre_seg_id'])

                if count > max_partner_lookup[axon_id]:
                    max_partner_lookup[axon_id] = count

                all_counts_total[syn_type][count] += 1

            for max_partner_count in max_partner_lookup.values():
                all_counts_total_gp[syn_type][max_partner_count] += 1


        with open(f'{results_dir}/all_counts_total_gp_{synapse_db}.json', 'w') as fp:
            json.dump(all_counts_total_gp, fp)

        with open(f'{results_dir}/all_counts_total_{synapse_db}.json', 'w') as fp:
            json.dump(all_counts_total, fp)


    with open(f'{results_dir}/all_counts_total_gp_{synapse_db}.json', 'r') as fp:
        all_counts_total_gp = json.load(fp)

    with open(f'{results_dir}/all_counts_total_{synapse_db}.json', 'r') as fp:
        all_counts_total = json.load(fp)





    # ANALYSIS: Combining E and I proportionally:


    with open(balanced_strength_data, 'r') as fp:
        balanced_axons_data = json.load(fp)

    range_to_plot = range(1,17)

    real_strength_counts_to_plot = [all_counts_total_gp['excitatory'][x]+all_counts_total_gp['inhibitory'][x] for x in range_to_plot]
    total_num_real_axons_in_range = sum(real_strength_counts_to_plot)
    expected_strength_counts = {x:0 for x in range_to_plot}

    for dtype in ['inhibitory', 'excitatory']:

        for strength in range_to_plot:

            n_real_axons_this_strength = all_counts_total_gp[dtype][strength]

            sim_strength_counts = {x:0 for x in range_to_plot}

            for axon_id in balanced_axons_data[str(strength)][dtype]:

                with open(f'{balanced_strength_sample_dir}/neurite_{axon_id}_simulations.json', 'r') as fp:
                    raw_data = json.load(fp)

                most_common_sim_pattern = get_patterns_from_simulated_neurite(raw_data)[0]
                greatest_partner_in_sim = max(most_common_sim_pattern)

                if greatest_partner_in_sim in sim_strength_counts:
                    sim_strength_counts[greatest_partner_in_sim] +=1
                else:
                    print(strength, most_common_sim_pattern)

            num_sim_axons_this_strength = sum(sim_strength_counts.values())

            if num_sim_axons_this_strength == 0: continue

            for sim_strength in sim_strength_counts:
                
                prop_axons_with_this_strength = sim_strength_counts[sim_strength]/num_sim_axons_this_strength

                expected_n_axons_this_strength = prop_axons_with_this_strength*n_real_axons_this_strength

                expected_strength_counts[sim_strength] += expected_n_axons_this_strength



    expected_strength_counts_to_plot = [expected_strength_counts[x] for x in range_to_plot]

    plt.plot(range_to_plot, real_strength_counts_to_plot, '-o', color='red', label = 'Observed', markersize=3)
    plt.plot(range_to_plot, expected_strength_counts_to_plot, '-o', color='slateblue', label = 'Expected under null model', markersize=3)
    plt.xticks(range_to_plot)
    plt.xlabel('Strongest connection of axon')
    plt.ylabel('Number of axons')
    plt.yscale('log')
    plt.show()


