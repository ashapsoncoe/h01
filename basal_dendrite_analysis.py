import json
from random import choices, choice, shuffle
from collections import Counter
import numpy as np
from statsmodels.stats.proportion import multinomial_proportions_confint
import matplotlib.pyplot as plt
import os
import scipy          
import common_functions as cf
from google.cloud import bigquery              
from google.oauth2 import service_account

data_file = 'c:/work/final/Layer_6_basal_cell_partners_agglo_20200916c3_v3_pure_axons_only'
raw_data_dir = 'c:/work/final/basal_dendrites_plots_and_data/sampled_synapse_points_Layer_6_basal_cell_partners_agglo_20200916c3_multi_bipolar_basal_d_targets'
just_seg_lists_save_dir =  'c:/work/final/basal_dendrites_plots_and_data/bd_seg_lists_v3_pure_axons_only'
results_dir = 'C:/work/FINAL/basal_dendrites_plots_and_data/'
corrected_fwd_rev_classification_path = 'c:/work/final/bipolar_c3_ids.json'
cred_path = 'c:/work/alexshapsoncoe.json'
synapse_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction1'


if __name__ == "__main__":

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    # Loading data:

    with open(corrected_fwd_rev_classification_path, 'r') as fp:
        corrected_fwd_rev_classification = json.load(fp)


    all_basal_d_data = {}

    for f in os.listdir(data_file):
        print(f)
        agglo_id = f.split('_')[0]

        with open(f'{data_file}/{f}', 'r') as fp:
            all_basal_d_data[agglo_id] = json.load(fp)

        c = None

        for x in ('fwd', 'rev', 'neither'):

            if os.path.exists(f'{just_seg_lists_save_dir}/{x}_bd_{agglo_id}_seg_lists.json'):
                with open(f'{just_seg_lists_save_dir}/{x}_bd_{agglo_id}_seg_lists.json', 'r') as fp:
                    c = json.load(fp)

        # Get AIS and other synapses:

        ais_partners = set(c['ais_partners'])
        other_dendrite_partners = set(c['other_dendrite_partners'])
        basal_partners = set(c['basal_dendrite_partners'])

        assert basal_partners & other_dendrite_partners == set()
        assert ais_partners & other_dendrite_partners == set()
        assert ais_partners & basal_partners == set()

        to_select = [
            'pre_synaptic_site.neuron_id AS pre_seg_id',
            'post_synaptic_partner.neuron_id AS post_seg_id',
            'location',
            'type',
            'post_synaptic_partner.id AS post_syn_id',
            'pre_synaptic_site.id AS pre_syn_id',

            ]

        results = cf.get_info_from_bigquery(to_select, 'post_synaptic_partner.neuron_id', [agglo_id], synapse_db, client)

        all_basal_d_data[agglo_id]['other_dendrite_synapses'] = []
        all_basal_d_data[agglo_id]['ais_synapses'] = []

        for r in results:

            final_datum = {
                'pre_seg_id': str(r['pre_seg_id']), 
                'syn_location': [r['location'][a] for a in ('x', 'y', 'z')], 
                'syn_type': r['type'], 
                'pre_syn_id': str(r['post_syn_id']), 
                'post_syn_id': str(r['pre_syn_id']),
            }

            if final_datum['pre_seg_id'] in other_dendrite_partners:
                all_basal_d_data[agglo_id]['other_dendrite_synapses'].append(final_datum)
            
            if final_datum['pre_seg_id'] in ais_partners:
                all_basal_d_data[agglo_id]['ais_synapses'].append(final_datum)



    # Organising data:

for post_struc_syn_type in ('basal_synapses', 'ais_synapses'):

    basal_d_2_ei_count = {x: {'e': 0 , 'i': 0} for x in all_basal_d_data}
    axon_ei_count = {}
    axon_fwd_rev_syn_count = {}
    axon2partners = {}

    for agglo_id in all_basal_d_data.keys():

        if post_struc_syn_type not in all_basal_d_data[agglo_id]: continue

        for syn in all_basal_d_data[agglo_id][post_struc_syn_type]:

            pre_seg_id = syn['pre_seg_id']

            if pre_seg_id not in axon_fwd_rev_syn_count:
                axon_fwd_rev_syn_count[pre_seg_id] = {'forward': 0, 'reverse': 0}

            if pre_seg_id not in axon_ei_count:
                axon_ei_count[pre_seg_id] = {'e': 0 , 'i': 0}

            if syn['syn_type'] == 1:
                axon_ei_count[pre_seg_id]['i'] += 1

            if syn['syn_type'] == 2:
                axon_ei_count[pre_seg_id]['e'] += 1
            
            if int(agglo_id) in corrected_fwd_rev_classification['positiveids']:
                axon_fwd_rev_syn_count[pre_seg_id]['forward'] += 1

            if int(agglo_id) in corrected_fwd_rev_classification['negativeids']:
                axon_fwd_rev_syn_count[pre_seg_id]['reverse'] += 1

            if pre_seg_id not in axon2partners:
                axon2partners[pre_seg_id] = {}

            if agglo_id not in axon2partners[pre_seg_id]:
                axon2partners[pre_seg_id][agglo_id] = 0

            axon2partners[pre_seg_id][agglo_id] += 1

        ei_counts = Counter([x['syn_type'] for x in all_basal_d_data[agglo_id][post_struc_syn_type]])

        for syn_code in ei_counts:

            if syn_code == 1:
                basal_d_2_ei_count[agglo_id]['i'] += ei_counts[syn_code]

            if syn_code == 2:
                basal_d_2_ei_count[agglo_id]['e'] += ei_counts[syn_code]




    all_e_axons = [x for x in axon_ei_count if axon_ei_count[x]['e']>axon_ei_count[x]['i']]
    all_i_axons = [x for x in axon_ei_count if axon_ei_count[x]['e']<axon_ei_count[x]['i']]


    all_bipolar_bd_by_syn = []

    for bd in all_basal_d_data.keys():

        if int(bd) in corrected_fwd_rev_classification['positiveids']+corrected_fwd_rev_classification['negativeids']:
            
            if post_struc_syn_type in all_basal_d_data[bd]:

                for syn in all_basal_d_data[bd][post_struc_syn_type]:

                    all_bipolar_bd_by_syn.append(bd)


    axons_targeting_multi_bipolar_basal_d = []

    for axon in axon2partners:

        this_axon_bipolar_partners = [p for p in axon2partners[axon] if int(p) in corrected_fwd_rev_classification['positiveids']+corrected_fwd_rev_classification['negativeids']]

        if len(set(this_axon_bipolar_partners)) > 1:
            axons_targeting_multi_bipolar_basal_d.append(axon)




    # Plot 2 syn 2 partner plots:
    simulation_type = 'synapse_numbers'

    pair_types = ('forward,forward', 'forward,reverse','reverse,reverse')

    for dtype, dataset in (('all', all_e_axons+all_i_axons), ('excitatory', all_e_axons), ('inhibitory', all_i_axons)):

        if post_struc_syn_type == 'ais_synapses' and dtype != 'inhibitory': continue

        counts = {a: {x: [] for x in pair_types} for a in ('real', 'simulated')}

        if post_struc_syn_type == 'ais_synapses':
            axons = [x for x in dataset if x in axons_targeting_multi_bipolar_basal_d]
        else:    
            axons = [x for x in dataset if sum(axon_ei_count[x].values())==2 and x in axons_targeting_multi_bipolar_basal_d]

        for ctype in ('real', 'simulated'):

            for axon in axons:
                
                if ctype == 'real':
                    this_axon_partners = list(axon2partners[axon].keys())
                    this_axon_partners = [x for x in this_axon_partners if int(x) in corrected_fwd_rev_classification['positiveids']+corrected_fwd_rev_classification['negativeids']]
                    
                    if len(this_axon_partners) > 2: continue

                    shuffle(this_axon_partners)
                    partner1, partner2 = this_axon_partners[:2]

                if ctype == 'simulated':

                    if simulation_type == 'synapse_numbers':
                        partner1 = choice(all_bipolar_bd_by_syn)
                        partner2 = choice([x for x in all_bipolar_bd_by_syn if x != partner1])

                    if simulation_type == 'empirical':

                        with open(f'{raw_data_dir}/neurite_{axon}_simulations.json', 'r') as fp:
                            sim_data = json.load(fp)

                        all_sim_partners = sim_data['simulation_all_sampled_partners']
                        all_sim_partners_fwd = [x for x in all_sim_partners if int(x) in corrected_fwd_rev_classification['positiveids']]
                        all_sim_partners_rev = [x for x in all_sim_partners if int(x) in corrected_fwd_rev_classification['negativeids']]

                        all_bipolar_partners = all_sim_partners_fwd+all_sim_partners_rev

                        c = Counter(all_bipolar_partners)

                        bipolar_ids_ordered_by_count = [x[0] for x in sorted([(seg_id, c[seg_id]) for seg_id in c], key = lambda x: x[1], reverse=True)]
                        
                        partner1, partner2 = bipolar_ids_ordered_by_count[:2]


                dirs = []

                for p in (partner1, partner2):

                    if int(p) in corrected_fwd_rev_classification['positiveids']:
                        p_dir = 'forward'
                    else:
                        assert int(p) in corrected_fwd_rev_classification['negativeids']
                        p_dir = 'reverse'
                    
                    dirs.append(p_dir)
                
                dirs.sort()

                dirs = ','.join(dirs)
                
                counts[ctype][dirs].append(axon)

        # Then make main plots:
        confints_dict = {}

        for stype in ('real', 'simulated'):
            data = [len(counts[stype][k]) for k in pair_types]
            confints = [[float(x[0]), float(x[1])] for x in multinomial_proportions_confint(data)]
            confints_dict[stype] = {pt: ci for pt, ci in zip(pair_types, confints)}


        real_counts = [len(counts['real'][k]) for k in pair_types]
        sim_counts = [len(counts['simulated'][k]) for k in pair_types]
        p_val = scipy.stats.chisquare(real_counts, f_exp=sim_counts)[1]
        pv_str = str(p_val)


        # Line plot:
        print(dtype)
        y_err = [confints_dict['real'][x][1] - confints_dict['real'][x][0] for x in pair_types]
        mean_vals = [np.mean(confints_dict['real'][x]) for x in pair_types]
        plt.errorbar(pair_types, mean_vals, fmt='.', yerr=y_err, capsize=5, elinewidth=0, markeredgewidth=2, color = 'slateblue', label='Observed')
        plt.plot(pair_types, [x/sum(real_counts) for x in sim_counts], 'or', label='Expected under null model')
        plt.rcParams["figure.figsize"] = (20,3)
        #plt.legend(loc='lower center', bbox_to_anchor=(1, 1))
        plt.xticks(range(3), pair_types)
        #plt.yticks(range(0,1.1, 0.1))
        plt.xlabel('Type of Basal Dendrite Pair Targeted by Axon')
        plt.ylabel('Proportion of Axons')
        pv_str = str(p_val)
        print(pv_str)
        plt.subplots_adjust(left=0.4, bottom=0.3)
        plt.savefig(f'{results_dir}/Proportion of {post_struc_syn_type} {dtype} axons contacting each type of basal dendrite pair, dot and CIs, p_{pv_str}.png')
        plt.clf()



