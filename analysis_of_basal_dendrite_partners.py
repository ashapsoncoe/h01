import json
from random import choices, choice
from collections import Counter
import numpy as np
from statsmodels.stats.proportion import multinomial_proportions_confint
import matplotlib.pyplot as plt
import os
import scipy          
import common_functions as cf

data_file = 'c:/work/final/Layer_6_basal_cell_partners_agglo_20200916c3_v3_pure_axons_only'
raw_data_dir = 'c:/work/final/basal_dendrites_plots_and_data/sampled_synapse_points_Layer_6_basal_cell_partners_agglo_20200916c3_multi_bipolar_basal_d_targets'
just_seg_lists_save_dir =  'c:/work/final/basal_dendrites_plots_and_data/bd_seg_lists_v3_pure_axons_only'
results_dir = 'C:/work/FINAL/basal_dendrites_plots_and_data/'
corrected_fwd_rev_classification_path = 'c:/work/final/bipolar_c3_ids.json'


if __name__ == "__main__":


    # Loading data:

    with open(corrected_fwd_rev_classification_path, 'r') as fp:
        corrected_fwd_rev_classification = json.load(fp)


    all_basal_d_data = {}

    for f in os.listdir(data_file):

        agglo_id = f.split('_')[0]

        with open(f'{data_file}/{f}', 'r') as fp:
            all_basal_d_data[agglo_id] = json.load(fp)

        c = None

        for x in ('fwd', 'rev', 'neither'):

            if os.path.exists(f'{just_seg_lists_save_dir}/{x}_bd_{agglo_id}_seg_lists.json'):
                with open(f'{just_seg_lists_save_dir}/{x}_bd_{agglo_id}_seg_lists.json', 'r') as fp:
                    c = json.load(fp)

        all_basal_d_data[agglo_id]['ais_inputs'] = c['ais_partners']
        all_basal_d_data[agglo_id]['basal_inputs'] = c["basal_dendrite_partners"]
        all_basal_d_data[agglo_id]['non_basal_inputs'] = c["other_dendrite_partners"]



    # Organising data:


    basal_d_2_ei_count = {x: {'e': 0 , 'i': 0} for x in all_basal_d_data}
    axon_ei_count = {}
    axon_fwd_rev_syn_count = {}
    axon2partners = {}

    for agglo_id in all_basal_d_data.keys():

        if 'basal_synapses' not in all_basal_d_data[agglo_id]: continue

        for syn in all_basal_d_data[agglo_id]['basal_synapses']:

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

        ei_counts = Counter([x['syn_type'] for x in all_basal_d_data[agglo_id]['basal_synapses']])

        for syn_code in ei_counts:

            if syn_code == 1:
                basal_d_2_ei_count[agglo_id]['i'] += ei_counts[syn_code]

            if syn_code == 2:
                basal_d_2_ei_count[agglo_id]['e'] += ei_counts[syn_code]




    all_e_axons = [x for x in axon_ei_count if axon_ei_count[x]['e']>axon_ei_count[x]['i']]
    all_i_axons = [x for x in axon_ei_count if axon_ei_count[x]['e']<axon_ei_count[x]['i']]


    all_bipolar_bd_by_syn = []

    for bd in all_basal_d_data.keys():

        angle = all_basal_d_data[bd]['elevation_angle']

        if int(bd) in corrected_fwd_rev_classification['positiveids']+corrected_fwd_rev_classification['negativeids']:
            
            if 'basal_synapses' in all_basal_d_data[bd]:

                for syn in all_basal_d_data[bd]['basal_synapses']:

                    all_bipolar_bd_by_syn.append(bd)


    axons_targeting_multi_bipolar_basal_d = []
    all_bipolar_basal_dendrites = set()

    for axon in axon2partners:

        this_axon_bipolar_partners = [p for p in axon2partners[axon] if int(p) in corrected_fwd_rev_classification['positiveids']+corrected_fwd_rev_classification['negativeids']]
        all_bipolar_basal_dendrites.update(this_axon_bipolar_partners)

        if len(set(this_axon_bipolar_partners)) > 1:
            axons_targeting_multi_bipolar_basal_d.append(axon)




    # Plot 2 syn 2 partner plots:
    simulation_type = 'synapse_numbers'

    pair_types = ('forward,forward', 'forward,reverse','reverse,reverse')

    for dtype, dataset in (('all', all_e_axons+all_i_axons), ('excitatory', all_e_axons), ('inhibitory', all_i_axons)):

        counts = {a: {x: [] for x in pair_types} for a in ('real', 'simulated')}

        axons = [x for x in dataset if sum(axon_ei_count[x].values())==2 and x in axons_targeting_multi_bipolar_basal_d]

        for ctype in ('real', 'simulated'):

            for axon in axons:
                
                if ctype == 'real':
                    partner1, partner2 = list(axon2partners[axon].keys())

                if ctype == 'simulated':

                    if simulation_type == 'synapse_numbers':
                        partner1 = choice(all_bipolar_bd_by_syn)
                        partner2 = choice([x for x in all_bipolar_bd_by_syn if x != partner1])

                    if simulation_type == 'empirical':

                        with open(f'{raw_data_dir}/neurite_{axon}_simulations.json', 'r') as fp:
                            sim_data = json.load(fp)

                        all_sim_partners = sim_data['simulation_all_sampled_partners']
                        all_sim_partners_fwd = [x for x in all_sim_partners if int(x) in int(p) in corrected_fwd_rev_classification['positiveids']]
                        all_sim_partners_rev = [x for x in all_sim_partners if int(x) in int(p) in corrected_fwd_rev_classification['negativeids']]

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

        y_err = [confints_dict['real'][x][1] - confints_dict['real'][x][0] for x in pair_types]
        mean_vals = [np.mean(confints_dict['real'][x]) for x in pair_types]
        plt.errorbar(pair_types, mean_vals, fmt='.', yerr=y_err, capsize=5, elinewidth=0, markeredgewidth=2, color = 'slateblue', label='Observed')
        plt.plot(pair_types, [x/sum(real_counts) for x in sim_counts], 'or', label='Expected under null model')
        plt.rcParams["figure.figsize"] = (20,3)
        #plt.legend(loc='lower center', bbox_to_anchor=(1, 1))
        plt.xticks(['FF', 'FR', 'RR'])
        plt.xlabel('Type of Basal Dendrite Pair Targeted by Axon')
        plt.ylabel('Proportion of Axons')
        pv_str = str(p_val)
        plt.subplots_adjust(left=0.4, bottom=0.3)
        plt.savefig(f'{results_dir}/Proportion of {dtype} axons contacting each type of basal dendrite pair, dot and CIs, p_{pv_str}.png')
        plt.clf()



