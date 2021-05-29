import json
import numpy as np
import common_functions as cf
from collections import Counter
import matplotlib.pyplot as plt
import neuroglancer
import os
from random import shuffle, sample
from scipy import stats
import math
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
from statsmodels.stats.proportion import multinomial_proportions_confint
import scipy
from google.cloud import bigquery        
from google.cloud import bigquery_storage         
from google.oauth2 import service_account


raw_data_dir = 'D:/sampled_synapse_points_random_sample_of_500_axons_from_each_gp_stregnth_and_type_agg20200916c3_unconstrained_incomplete'
multi_synapse_axons_path = 'c:/work/final/random_sample_of_500_axons_from_each_gp_stregnth_and_type_agg20200916c3_dict.json'
organised_axon_data_path = 'c:/work/final/random_sample_of_1000_axons_from_each_layer_and_type_agg20200916c3_dict.json'
balanced_strength_data = 'c:/work/final/random_sample_of_500_axons_from_each_gp_stregnth_and_type_agg20200916c3_dict.json'
balanced_strength_sample_dir = 'D:/sampled_synapse_points_random_sample_of_500_axons_from_each_gp_stregnth_and_type_agg20200916c3_unconstrained_incomplete'
results_dir = 'c:/work/final/connection_strength_analysis_unconstrained_v2'
synapse_db = 'goog14r0s5c3.synaptic_connections_with_skeleton_classes'
region_type_db = 'goog14r0seg1.agg20200916c3_regions_types'
seg_info_db = 'goog14r0seg1.agg20200916c3_regions_types'
syn_vx_size = {'x': 8, 'y': 8, 'z': 33}
cred_path = 'c:/work/alexshapsoncoe.json'
simulation_number = 1000

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


credentials = service_account.Credentials.from_service_account_file(cred_path)
client = bigquery.Client(project=credentials.project_id, credentials=credentials)
bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)


# FIRST GET TOTAL COUNTS FOR NON-AIS CONTACTING PURE AXONS:
axon_types = ('excitatory', 'inhibitory')


if not os.path.exists(f'{results_dir}/all_counts_total_{synapse_db}.json'):

    all_counts_total = {dtype: {x: 0 for x in range(1000)} for dtype in axon_types}
    all_counts_total_gp = {dtype: {x: 0 for x in range(1000)} for dtype in axon_types}

    for syn_type in axon_types:
        print(syn_type)

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





# NEW ANALYSIS:
with open(balanced_strength_data, 'r') as fp:
    balanced_axons_data = json.load(fp)

range_to_plot = range(1,19)

for dtype in  ['inhibitory', 'excitatory', 'all']:

    if dtype == 'all':
        real_strength_counts_to_plot = [all_counts_total_gp['excitatory'][x]+all_counts_total_gp['inhibitory'][x] for x in range_to_plot]
    else:
        real_strength_counts_to_plot = [all_counts_total_gp[dtype][x] for x in range_to_plot]

    total_num_real_axons_in_range = sum(real_strength_counts_to_plot)

    expected_strength_counts = {x:0 for x in range_to_plot}

    for strength in range_to_plot:

        n_real_axons_this_strength = all_counts_total_gp[dtype][strength]

        sim_strength_counts = {x:0 for x in range_to_plot}

        for axon_id in balanced_axons_data[str(strength)][dtype]:

            if not os.path.exists(f'{balanced_strength_sample_dir}/neurite_{axon_id}_simulations.json'):
                print(axon_id, stregnth)
                continue

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

        #assert num_sim_axons_this_strength == len(balanced_axons_data[str(strength)][dtype])

        for sim_strength in sim_strength_counts:
            
            prop_axons_with_this_strength = sim_strength_counts[sim_strength]/num_sim_axons_this_strength

            expected_n_axons_this_strength = prop_axons_with_this_strength*n_real_axons_this_strength

            expected_strength_counts[sim_strength] += expected_n_axons_this_strength

    expected_strength_counts_to_plot = [expected_strength_counts[x] for x in range_to_plot]

    plt.plot(range_to_plot, real_strength_counts_to_plot, '-o', color='red', label = 'Observed')
    plt.plot(range_to_plot, expected_strength_counts_to_plot, '-o', color='slateblue', label = 'Expected under null model')
    plt.xticks(range_to_plot)
    plt.xlabel('Strongest connection of axon')
    plt.ylabel('Number of axons')
    plt.yscale('log')
    plt.show()









# Combining E and I proportionally:


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

            if not os.path.exists(f'{balanced_strength_sample_dir}/neurite_{axon_id}_simulations.json'):
                print(axon_id, stregnth)
                continue

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

        #assert num_sim_axons_this_strength == len(balanced_axons_data[str(strength)][dtype])

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



import numpy as np
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

stats = importr('stats')
m = np.array(list(zip([round(x) for x in real_strength_counts_to_plot], [round(x) for x in expected_strength_counts_to_plot])))
res = stats.fisher_test(m, workspace = 900000000, simulate.p.value='TRUE')
p_val = res[0][0]








#p_val = scipy.stats.chisquare(real_strength_counts_to_plot, f_exp=expected_strength_counts_to_plot)[1]




        
        plt.legend(loc='lower center', bbox_to_anchor=(0, 1))
        plt.xticks(range_to_plot)
        plt.xlabel('Number of Synapses in a Connection')
        plt.ylabel('Proportion of Connections')
        pv_str = str(p_val)
        plt.title(f'Expected and Observed (in {real_data_to_plot}) Proportions of Connection strengths for {axon_type} axons. P = {pv_str}')

        plt.savefig(f'{results_dir}/Expected and Observed (in {real_data_to_plot}) Proportions of Connection strengths for {axon_type} axons.png')
        plt.clf()










if __name__ == '__main__':



    with open(organised_axon_data_path, 'r') as fp:
        all_sampled_axons_data = json.load(fp)

    # Check there is a file available for each of the multisubset neurites:
    with open(multi_synapse_axons_path, 'r') as fp:
        multisyn_neurite_subset = json.load(fp)

    available_neurites = [x.split('_')[1] for x in os.listdir(raw_data_dir) if 'simulations.json' in x]

    assert set(multisyn_neurite_subset).issubset(set(available_neurites))

    
    dtypes = ('simulated', 'real')
    

    all_counts = {layer: {axon_type: {dtype: {x: 0 for x in range(1, 51)} for dtype in dtypes} for axon_type in axon_types} for layer in (all_sampled_axons_data)}
    all_counts_gp = {layer: {axon_type: {dtype: {x: 0 for x in range(1, 51)} for dtype in dtypes} for axon_type in axon_types} for layer in (all_sampled_axons_data)}

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








    range_to_plot = range(1, 11)

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




for real_data_to_plot in ('sample', 'total'):

    for axon_type in ('excitatory', 'inhibitory', 'all'):

        if real_data_to_plot == 'sample':
            real_counts = all_counts_real[axon_type]

        if real_data_to_plot == 'total':
            real_counts = all_counts_total[axon_type]

        real_counts_to_plot = [real_counts[r] for r in range_to_plot]
        sim_counts_to_plot = [all_counts_sim[axon_type][r] for r in range_to_plot]
            
        # Plot proportion connection strengths:
        plt.figure(figsize=(20,10))

        all_props_real = [x/sum(real_counts_to_plot) for x in real_counts_to_plot]
        all_props_sim = [x/sum(sim_counts_to_plot) for x in sim_counts_to_plot]

        f_obs = [real_counts[x] for x in range_to_plot]
        f_exp = [p*sum(f_obs) for p in all_props_sim]

        # Calculate overall p-value:
        p_val = scipy.stats.chisquare(f_obs, f_exp=f_exp)[1]

        
        plt.plot(range_to_plot, all_props_real, '-o', color='red', label = 'Observed')
        plt.plot(range_to_plot, all_props_sim, '-o', color='slateblue', label = 'Expected under null model')
        plt.yscale('log')
        plt.legend(loc='lower center', bbox_to_anchor=(0, 1))
        plt.xticks(range_to_plot)
        plt.xlabel('Number of Synapses in a Connection')
        plt.ylabel('Proportion of Connections')
        pv_str = str(p_val)
        plt.title(f'Expected and Observed (in {real_data_to_plot}) Proportions of Connection strengths for {axon_type} axons. P = {pv_str}')

        plt.savefig(f'{results_dir}/Expected and Observed (in {real_data_to_plot}) Proportions of Connection strengths for {axon_type} axons.png')
        plt.clf()


        # Plot counts  of strongest connection strength:
        if real_data_to_plot == 'total': continue

        plt.figure(figsize=(20,10))
        p_val = scipy.stats.chisquare(real_counts_to_plot, f_exp=sim_counts_to_plot)[1]

        
        plt.plot(range_to_plot, real_counts_to_plot, '-o', color='red', label = 'Observed')
        plt.plot(range_to_plot, sim_counts_to_plot, '-o', color='slateblue', label = 'Expected under null model')
        plt.yscale('log')
        plt.legend(loc='lower center', bbox_to_anchor=(0, 1))
        plt.xticks(range_to_plot)
        plt.xlabel('Strongest connection of axon')
        plt.ylabel('Number of axons')
        pv_str = str(p_val)
        plt.title(f'Expected and Observed (in {real_data_to_plot}) Counts of Strongest Connection strengths for {axon_type} axons. P = {pv_str}')

        plt.savefig(f'{results_dir}/Expected and Observed (in {real_data_to_plot}) Counts of Strongest Connection strengths for {axon_type} axons.png')
        plt.clf()



'''
def normal_interp(x, y, a, xi, yi):
    rbf = scipy.interpolate.Rbf(x, y, a)
    ai = rbf(xi, yi)
    return ai

def rescaled_interp(x, y, a, xi, yi):
    a_rescaled = (a - a.min()) / a.ptp()
    ai = normal_interp(x, y, a_rescaled, xi, yi)
    ai = a.ptp() * ai + a.min()
    return ai

def plot(x, y, a, ai, title):
    fig, ax = plt.subplots()

    im = ax.imshow(ai.T, origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()])
    #ax.scatter(x, y, c=a)

    ax.set(xlabel='X', ylabel='Y', title=title)
    fig.colorbar(im)




for seg_id in multisyn_neurite_subset:
    print(seg_id)
    with open(f'{raw_data_dir}/neurite_{seg_id}_simulations.json', 'r') as fp:
        raw_data = json.load(fp)

    # First get partners
    simulated_partners = []
    real_partners = []

    for dtype in ('shaft', 'stalk'):

        for datum in raw_data[dtype]['real_partners']:
            real_partners.append(datum['partner_id'])

        for syn_id in raw_data[dtype]['simulated_partners'].keys():
            simulated_partners.append(datum['partner_id'])

        # Then count:
        partner_counts = Counter(real_partners)
        real_counts = Counter(partner_counts.values())

        for n_syn in real_counts.keys():
            if n_syn <= 1000:
                final_data['real'][n_syn]['count'] += real_counts[n_syn]

        shuffle(simulated_partners)

        batch_size = len(real_partners)
        simulated_batches = [simulated_partners[batch*batch_size:(batch+1)*batch_size] for batch in range(simulation_number)]

        # Put simluated_partners into batches:
        assert len(real_partners)*simulation_number == len(simulated_partners)

        strongest_connection_count = {x: 0 for x in range(1, 1001)}

        for batch in simulated_batches:

            this_batch_counts = Counter(Counter([x[0] for x in batch]).values())

            for n_syn in this_batch_counts.keys():

                if n_syn <= 1000:
                    final_data['simulated'][n_syn]['count'] += this_batch_counts[n_syn]

            strongest_count = max(this_batch_counts.keys())
            strongest_connection_count[strongest_count] += 1

        assert simulation_number == sum(strongest_connection_count.values())

        for partner in partner_counts.keys():
            partner_count = partner_counts[partner]
            this_or_stronger_count_freq = sum([strongest_connection_count[x] for x in range(partner_count, 1001)])
            p_val = this_or_stronger_count_freq/simulation_number

            ind_connection_result = {
                    'pre_seg_id':  seg_id,
                    'post_seg_id': partner,
                    'p_val': p_val,
                    'num_syn': partner_count,
                }

            individual_connection_results.append(ind_connection_result)

    # Add non-simulated single synapse counts:    
    with open(single_synapse_axons_path, 'r') as fp:
        single_syn = json.load(fp)   

    # And add to individual connection data:
    req_info = ['pre_synaptic_site.neuron_id AS pre_id', 'post_synaptic_partner.neuron_id AS post_id']
    res = cf.get_info_from_bigquery(req_info, 'pre_synaptic_site.neuron_id', single_syn, synapse_db, client)
    res2 = set([tuple([x['pre_id'], x['post_id']]) for x in res])

    for pre_id, post_id in res2:
        ind_connection_result = {
            'pre_seg_id':  str(pre_id),
            'post_seg_id': str(post_id),
            'p_val': 1.0,
            'num_syn': 1,
            }

        individual_connection_results.append(ind_connection_result)

    # And add to pooled data:
    final_data['simulated'][1]['count'] += len(single_syn)*simulation_number
    final_data['real'][1]['count'] += len(single_syn)

    # Add locations to individual connection stats:
    all_pres = list(set([x['pre_seg_id'] for x in individual_connection_results]))
    req_info = ['pre_synaptic_site.neuron_id AS pre_id', 'post_synaptic_partner.neuron_id AS post_id', 'location']
    res = cf.get_info_from_bigquery(req_info, 'pre_synaptic_site.neuron_id', all_pres, synapse_db, client)
    loc_d = {}

    for x in res:
        combo_id = str(x['pre_id']) + '_' + str(x['post_id'])
        if combo_id not in loc_d:
            loc_d[combo_id] = []

        loc = tuple([x['location'][a]*syn_vx_size[a] for a in ['x', 'y','z']])
        loc_d[combo_id].append(loc)

    for datum in individual_connection_results:
        combo_id = str(datum['pre_seg_id']) + '_' + str(datum['post_seg_id'])
        datum['syn_locs'] = tuple(loc_d[combo_id])

    # Save individual connection results:
    with open(f'{results_dir}/individual_connection_results.json', 'w') as fp:
        json.dump(individual_connection_results, fp)

    # Get all real connections:
    with open(real_count_path, 'r') as fp:
        all_counts = json.load(fp) 

    for n_syn in all_counts.keys():
        final_data['total'][int(n_syn)]['count'] = all_counts[n_syn]

    
    # Then get proportions and CIs (exclude count < 5 in either selected type):
    accepted_n_syns = [
        n_syn for n_syn in range(1,1001)
        if final_data[pair_to_compare[0]][n_syn]['count'] >= 5 
        and final_data[pair_to_compare[1]][n_syn]['count'] >= 5
    ]


    for analysis_type in pair_to_compare:
        accepted_counts = [final_data[analysis_type][n_syn]['count'] for n_syn in accepted_n_syns]
        confints = [[float(x[0]), float(x[1])] for x in multinomial_proportions_confint(accepted_counts)]
        for n_syn, ci in zip(accepted_n_syns, confints):
            prop = final_data[analysis_type][n_syn]['count']/sum(accepted_counts)
            final_data[analysis_type][n_syn]['proportion'] = prop
            final_data[analysis_type][n_syn]['confint'] = ci


    plt.figure(figsize=(20,10))

    # Calculate overall p-value:

    f_obs = [final_data['total'][k]['count'] for k in accepted_n_syns]
    f_exp = [final_data['simulated'][k]['proportion']*sum(f_obs) for k in accepted_n_syns]

    p_val = scipy.stats.chisquare(f_obs, f_exp=f_exp)[1]

    fig_key = list(zip(['slateblue', 'red'], pair_to_compare))

    for colour, d in fig_key:

        if d == 'total':
            k = 'Observed'
        if d == 'simulated':
            k = 'Expected under null model'
        if d == 'real':
            k = 'Observed in subset of axons'

        eb =   plt.errorbar(
            accepted_n_syns,
            [np.mean(final_data[d][x]['confint']) for x in accepted_n_syns],
            yerr=[final_data[d][x]['confint'][1] - final_data[d][x]['confint'][0] for x in accepted_n_syns],
            capsize=5,
            elinewidth=1,
            markeredgewidth=2,
            color = colour,
            #ls='none',
            label=k
        )

        eb[-1][0].set_linestyle('-.')

        # plt.scatter(
        #     accepted_n_syns,  
        #     [final_data[d][x]['proportion'] for x in accepted_n_syns],
        #     15, 
        #     color=colour
        #     )

    plt.yscale('log')
    plt.legend(loc='lower center', bbox_to_anchor=(0, 1))
    plt.xticks(accepted_n_syns)
    plt.xlabel('Number of Synapses in a Connection')
    plt.ylabel('Proportion of Connections')
    pv_str = str(p_val)
    plt.title(f'Expected and Observed Proportions of Connection strengths. P = {pv_str}')

    plt.savefig(f'{results_dir}/Expected and Observed Proportions of Connection strengths {pair_to_compare[0]} vs {pair_to_compare[1]}.png')
    plt.clf()

p_threshold = 1.0
sample_size = 5000# or all

to_plot = [x for x in individual_connection_results if x['p_val']<p_threshold]

if sample_size != 'all':
    to_plot = sample(to_plot, sample_size)

x_coords = [np.mean(x['syn_locs'], axis=0)[0] for x in to_plot]
y_coords = [np.mean(x['syn_locs'], axis=0)[1] for x in to_plot]
p_vals = [x['p_val'] for x in to_plot]

x = np.array(x_coords)
y = np.array(y_coords)
a = np.array(p_vals)

xi, yi = np.mgrid[x.min():x.max():10j, y.min():y.max():2j]

a_orig = normal_interp(x, y, a, xi, yi)
a_rescale = rescaled_interp(x, y, a, xi, yi)
plot(x, y, a, a_orig, 'Not Rescaled')
plot(x, y, a, a_rescale, 'Rescaled')
plt.show()



fig, ax = plt.subplots()
ax.scatter(x_coords, y_coords, c=p_vals)
cb = fig.colorbar(p_vals, ax=ax)
cb.set_label('Color Scale')
#ax.legend()
plt.show()
plt.clf()


'''

''' Use these websites to finish off:
https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/scatter_with_legend.html#sphx-glr-gallery-lines-bars-and-markers-scatter-with-legend-py
https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_with_legend.html
https://stackoverflow.com/questions/61084381/create-gradient-legend-matplotlib
https://stackoverflow.com/questions/59365942/smooth-2d-interpolation-map-using-z-values-1-column-at-known-x-and-y-coordinat
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
https://stackoverflow.com/questions/17577587/matplotlib-2d-graph-with-interpolation
'''



