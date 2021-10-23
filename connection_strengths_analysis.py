import json
from collections import Counter
import matplotlib.pyplot as plt
import os
from random import shuffle
from scipy import stats
from google.cloud import bigquery        
from google.cloud import bigquery_storage         
from google.oauth2 import service_account
import scipy
import common_functions as cf
import pandas as pd

balanced_strength_sample_dir = 'D:/ss_points_random_sample_of_10000_axons_from_each_gp_strength_agg20200916c3_merge_correction2_0ax_15um_displacement_rp_not_added_25um_xy_soma_exc'
results_dir = 'D:/csa_10000_axons_from_each_gp_strength_agg20200916c3_mc2_0ax_15um_xy_displacement_rp_not_added_25um_soma_exc'
synapse_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_spinecorrected_merge_correction2'
seg_info_db = 'goog14r0seg1.agg20200916c3_regions_types_circ_bounds'
cred_path = 'D:/alexshapsoncoe.json'
skel_sql_table = 'goog14r0seg1.agg20200916c3_subcompartment_skeleton_counts_v1'
min_num_axon_nodes = 0
axon_simulations_to_exclude = ['6716812084'] # based on manual inspection

# If want to make NG plots:
gp_sim_partner_stregnths_to_make_plots_for = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
em = 'brainmaps://964355253395:h01:goog14r0_8nm'
agglo_seg = 'brainmaps://964355253395:h01:goog14r0seg1_agg20200916c3_flat'


def get_patterns_from_simulated_neurite(raw_data):

    simc_stalk = raw_data['stalk']['simulated_partners']
    simc_shaft = raw_data['shaft']['simulated_partners']
    least_samples_from_a_syn_id = min([len(simc_stalk[x]) for x in simc_stalk]+[len(simc_shaft[x]) for x in simc_shaft])

    all_sampled_ids = []
    all_real_ids = []

    for stype in ('stalk', 'shaft'):
        
        real_partner_data_this_type = [x for x in raw_data[stype]['real_partners'] if x['partner_id'] != 'unidentifiedpartner' ]

        real_partners_this_type = [x['partner_id'] for x in real_partner_data_this_type]
        
        all_real_ids.extend(real_partners_this_type)

        real_syn_ids_this_type = [x['syn_id'] for x in real_partner_data_this_type]

        for syn_id in real_syn_ids_this_type:
            all_partners_this_syn_id = [x['partner_id'] for x in raw_data[stype]['simulated_partners'][syn_id]]

            shuffle(all_partners_this_syn_id)
            all_sampled_ids.extend(all_partners_this_syn_id[:max(10, least_samples_from_a_syn_id)])

    if all_sampled_ids == []:
        return [], []


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

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    
    # FIRST GET TOTAL COUNTS FOR NON-AIS CONTACTING PURE AXONS:
    print('Getting all counts')

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)
    
    query = f"""with pure_axons as (
                select CAST(agglo_id AS STRING) as agglo_id
                from {seg_info_db}
                where type = 'pure axon fragment'
                ),

                acceptable_partners as (
                select CAST(agglo_id AS STRING) as agglo_id
                from {seg_info_db}
                where type LIKE '%neuron%' or type LIKE '%dendrite%'
                ),

                    
                large_enough_axons as (
                select CAST(skeleton_id AS STRING) as agglo_id
                from {skel_sql_table}
                where num_axon >= {min_num_axon_nodes}
                ),

                pure_axons_large_enough as (
                    select agglo_id from pure_axons
                    intersect distinct
                    select agglo_id from large_enough_axons
                ),


                all_edges_precursor AS (
                SELECT 
                    CAST(pre_synaptic_site.neuron_id AS STRING) AS pre_seg_id, 
                    CAST(post_synaptic_partner.neuron_id AS STRING) AS post_seg_id, 
                    COUNT(*) AS pair_count,
                    COUNTIF(type=1) AS i_count,
                    COUNTIF(type=2) AS e_count
                FROM {synapse_db}
                WHERE pre_synaptic_site.neuron_id IS NOT NULL AND post_synaptic_partner.neuron_id IS NOT NULL 
                GROUP BY pre_synaptic_site.neuron_id, post_synaptic_partner.neuron_id
                ),

                all_edges as (

                select A.pair_count, A.pre_seg_id, A.post_seg_id, A.i_count, A.e_count, B.region
                from all_edges_precursor A
                inner join {seg_info_db} B
                on A.pre_seg_id = B.agglo_id
                ),

                pure_axons_making_synapses as (
                    select agglo_id from pure_axons_large_enough
                    intersect distinct
                    select pre_seg_id from all_edges AS agglo_id
                ),

                segments_synapsing_onto_ais as (
                    select distinct CAST(pre_synaptic_site.neuron_id AS STRING) as seg_id
                    from {synapse_db}
                    where LOWER(post_synaptic_partner.class_label) = 'ais'
                ),

                non_ais_axons as (
                select distinct agglo_id from pure_axons_making_synapses A
                left join segments_synapsing_onto_ais B
                on A.agglo_id = B.seg_id 
                where B.seg_id IS NULL
                ),

                edges_from_acceptable_axons as (

                    select pair_count, pre_seg_id, post_seg_id, i_count, e_count, region from all_edges A
                    inner join non_ais_axons B
                    on A.pre_seg_id = B.agglo_id
                )
                select pair_count, pre_seg_id, i_count, e_count, region from edges_from_acceptable_axons A
                inner join acceptable_partners B
                on A.post_seg_id = B.agglo_id
                """

    df = client.query(query).result().to_dataframe(bqstorage_client=bqstorageclient, progress_bar_type='tqdm_gui')

    all_regions = set(df['region'])

    all_counts_total = {t: {l: {x: 0 for x in range(1000)} for l in all_regions} for t in ('excitatory', 'inhibitory')}
    all_counts_total_gp = {t: {l: {x: 0 for x in range(1000)} for l in all_regions} for t in ('excitatory', 'inhibitory')}

    max_partner_lookup = {x: (0, None, None) for x in df['pre_seg_id']}

    for i in df.index:

        if i%100000 == 0:
            print(i)

        count = int(df.at[i, 'pair_count'])
        axon_id = str(df.at[i, 'pre_seg_id'])
        layer = str(df.at[i, 'region'])

        if df.at[i, 'i_count'] == df.at[i, 'e_count']:
            axon_type = 'unknown'

        if df.at[i, 'i_count'] < df.at[i, 'e_count']:
            axon_type = 'excitatory'

        if df.at[i, 'i_count'] > df.at[i, 'e_count']:
            axon_type = 'inhibitory'

        if axon_type != 'unknown':
            all_counts_total[axon_type][layer][count] += 1

        if count > max_partner_lookup[axon_id][0]:
            max_partner_lookup[axon_id] = (count, layer, axon_type)


    for max_partner_count, layer, axon_type in max_partner_lookup.values():

        if axon_type  != 'unknown':
            all_counts_total_gp[axon_type][layer][max_partner_count] += 1

    with open(f'{results_dir}/all_counts_gp_{synapse_db}.json', 'w') as fp:
        json.dump(all_counts_total_gp, fp)

    with open(f'{results_dir}/all_counts_total_{synapse_db}.json', 'w') as fp:
        json.dump(all_counts_total, fp)

    df = df.astype(str)
    df.to_csv(f'{results_dir}/all_edges.csv')


    # Get essential data from simulation files:
    available_axons = [f.split('_')[1] for f in os.listdir(balanced_strength_sample_dir) if 'simulations.json' in f]
    
    available_axons = [x for x in available_axons if x not in axon_simulations_to_exclude]

    available_axons = list(set(available_axons) & set(df['pre_seg_id']))

    avail_axons_df = df[df['pre_seg_id'].isin(set(available_axons))]

    layer_lookup = {}
    type_lookup = {}

    for i in avail_axons_df.index:

        count = int(avail_axons_df.at[i, 'pair_count'])
        axon_id = str(avail_axons_df.at[i, 'pre_seg_id'])
        layer = str(avail_axons_df.at[i, 'region'])

        if avail_axons_df.at[i, 'i_count'] == avail_axons_df.at[i, 'e_count']:
            axon_type = 'unknown'

        if avail_axons_df.at[i, 'i_count'] < avail_axons_df.at[i, 'e_count']:
            axon_type = 'excitatory'

        if avail_axons_df.at[i, 'i_count'] > avail_axons_df.at[i, 'e_count']:
            axon_type = 'inhibitory'

        layer_lookup[axon_id] = layer
        type_lookup[axon_id] = axon_type



    sim_essential_data = {t: {l: {} for l in all_regions} for t in ('excitatory', 'inhibitory')}

    for axon_id in available_axons:

        layer = layer_lookup[axon_id]
        axon_type = type_lookup[axon_id]

        if axon_type == 'unknown': continue

        with open(f'{balanced_strength_sample_dir}/neurite_{axon_id}_simulations.json', 'r') as fp:
            
            raw_data = json.load(fp)

        most_common_sim_pattern, real_pattern = get_patterns_from_simulated_neurite(raw_data)

        if real_pattern == []:
            continue # If not even one synapse with an identified post-synaptic partner, don't include axon in analysis
        
        total_syn = sum(real_pattern)
        
        greatest_partner_in_sim = max(most_common_sim_pattern)
        greatest_partner_in_real = max(real_pattern)

        sim_essential_data[axon_type][layer][axon_id] = {
            'total_syn': total_syn,
            'greatest_partner_in_sim': greatest_partner_in_sim,
            'greatest_partner_in_real': greatest_partner_in_real,
        }

        if greatest_partner_in_sim in gp_sim_partner_stregnths_to_make_plots_for:
            print('Plotting', axon_id)

            plot_dir = f'{results_dir}/neuroglancer_plots'

            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)

            cf.save_ng_state_of_sampled_points(em, agglo_seg, axon_id, raw_data, plot_dir)




















# Whole dataset-extrapolation red and blue curves plot:

for analysis_type in ['all', 'excitatory', 'inhibitory']:
    print(analysis_type)
    if analysis_type == 'all':
        axon_types_to_include = ('excitatory', 'inhibitory')
    else:
        axon_types_to_include = tuple([analysis_type])

    range_to_plot = range(1,21)

    real_strength_counts_to_plot = [sum([sum([all_counts_total_gp[axon_type][layer][x] for layer in all_regions]) for axon_type in axon_types_to_include]) for x in range_to_plot]
    total_num_real_axons_in_range = sum(real_strength_counts_to_plot)
    expected_strength_counts = {x:0 for x in range_to_plot}
    temp = 0
    
    for axon_type in axon_types_to_include:
        
        for layer in sim_essential_data[axon_type]:
            
            for strength in range_to_plot:
                
                n_real_axons_this_type_layer_strength = all_counts_total_gp[axon_type][layer][strength]
                
                sim_axons_of_this_type_layer_and_gp_strength = [axon_id for axon_id in sim_essential_data[axon_type][layer] if sim_essential_data[axon_type][layer][axon_id]['greatest_partner_in_real'] == strength and axon_id not in axon_simulations_to_exclude]
            
                if len(sim_axons_of_this_type_layer_and_gp_strength) > 0:
            
                    n_axons_to_add_to_sim_counts_per_sim_axon = n_real_axons_this_type_layer_strength/len(sim_axons_of_this_type_layer_and_gp_strength)
                
                    for axon_id in sim_axons_of_this_type_layer_and_gp_strength:

                        greatest_partner_in_sim = sim_essential_data[axon_type][layer][axon_id]['greatest_partner_in_sim']
                        
                        expected_strength_counts[greatest_partner_in_sim] += n_axons_to_add_to_sim_counts_per_sim_axon

    expected_strength_counts_to_plot = [expected_strength_counts[x] for x in range_to_plot]
    expected_strength_counts_to_plot = [x*(total_num_real_axons_in_range/sum(expected_strength_counts_to_plot)) for x in expected_strength_counts_to_plot]

    p_val = scipy.stats.chisquare(real_strength_counts_to_plot, f_exp=expected_strength_counts_to_plot)[1]

    print(f'p = {p_val}')

    real_percentages_to_plot = [x/sum(real_strength_counts_to_plot)*100 for x in real_strength_counts_to_plot]
    expected_percentages_to_plot = [x/sum(expected_strength_counts_to_plot)*100 for x in expected_strength_counts_to_plot]

    plt.plot(range_to_plot, real_percentages_to_plot, '-o', color='red', label = 'Observed', markersize=3)
    plt.plot(range_to_plot, expected_percentages_to_plot, '-o', color='slateblue', label = 'Expected under null model', markersize=3)
    plt.xticks(range_to_plot)
    plt.xlabel('Number of synapses in strongest connection')
    plt.ylabel('Percentage of axons')
    plt.yscale('log')
    plt.show()


