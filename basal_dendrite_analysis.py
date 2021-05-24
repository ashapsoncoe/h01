import json
from random import choices, choice
from scipy import stats
from collections import Counter
import numpy as np
from statsmodels.stats.proportion import multinomial_proportions_confint
import matplotlib.pyplot as plt
import os
import scipy
from google.cloud import bigquery             
from google.oauth2 import service_account
import common_functions as cf

data_file = 'D:/Layer_6_basal_cell_partners_agglo_20200916c3.json'
raw_data_dir = 'D:/sampled_synapse_points_Layer_6_basal_cell_partners_agglo_20200916c3_multi_bipolar_basal_d_targets'
multibasal_axons_path = 'D:/Layer_6_basal_cell_partners_agglo_20200916c3_multi_basal_d_targets.json'
results_dir = 'D:/basal_dendrites_plots_and_data/'
cred_path = 'D:/alexshapsoncoe.json'
synapse_db = 'goog14r0s5c3.synaptic_connections_with_skeleton_classes'
syn_lookup_key = 'basal_synapses' # or 'all_synapses'
make_ais_plot = True
upper_degree_bound = 45
lower_degree_bound = -45
phase2_start = 0000
syn_vx_size = [8,8,33]


def get_sorted_bd_pair_type(partner1, partner2, all_basal_d_data):

    dirs = []

    for p in (partner1, partner2):

        if all_basal_d_data[p]['elevation_angle'] > upper_degree_bound:
            p_dir = 'forward'
        else:
            assert all_basal_d_data[p]['elevation_angle'] < lower_degree_bound
            p_dir = 'reverse'
        
        dirs.append(p_dir)

    dirs.sort()

    dirs = ','.join(dirs)

    return dirs


if __name__ == '__main__':

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    # Loading data:
    with open(data_file, 'r') as fp:
        all_basal_d_data = json.load(fp)


    # If want to consider all input synpases, not just those on basal dendrites
    if syn_lookup_key == 'all_synapses':

        info_to_get = [
            'post_synaptic_partner.neuron_id AS post_seg_id',
            'pre_synaptic_site.neuron_id AS pre_seg_id',
            'type',
            'post_synaptic_partner.skel_type AS post_type',
            'pre_synaptic_site.skel_type AS pre_type',
            f'''location.x*{syn_vx_size[0]} AS x''',
            f'''location.y*{syn_vx_size[1]} AS y''',
            f'''location.z*{syn_vx_size[2]} AS z''',
            ]

        results = cf.get_info_from_bigquery(info_to_get, 'post_synaptic_partner.neuron_id', list(all_basal_d_data.keys()), synapse_db, client)

        results = [x for x in results if x['pre_type']=='axon' and x['post_type'] in ('soma', 'dendrite')]

        for syn in results:
            
            syn_loc = [syn['x'], syn['y'], syn['z']]
            pre_seg_id = str(syn['pre_seg_id'])
            post_seg_id = str(syn['post_seg_id'])

            this_syn_dict = {
                'pre_seg_id': pre_seg_id,
                'syn_location': syn_loc,
                'syn_type': syn['type'],
            }

            if 'all_synapses' not in all_basal_d_data[post_seg_id]:
                all_basal_d_data[post_seg_id]['all_synapses'] = []

            all_basal_d_data[post_seg_id]['all_synapses'].append(this_syn_dict)


    # Remove cells and synapses above specified z layer, and any axo-axonal axons:
    all_basal_d_data = {k: all_basal_d_data[k] for k in all_basal_d_data if all_basal_d_data[k]['cb_loc'][2]/33>phase2_start}

    for k in all_basal_d_data.keys():

        if syn_lookup_key not in all_basal_d_data[k]: continue

        accepted_syn = []
        accepted_basal_nodes = []

        for syn in all_basal_d_data[k][syn_lookup_key]:
            if syn['syn_location'][2]/33>phase2_start:
                accepted_syn.append(syn)

        if 'basal_node_locations' in all_basal_d_data[k]:
            for basal_node in all_basal_d_data[k]['basal_node_locations']:
                if basal_node[2]/33>phase2_start:
                    accepted_basal_nodes.append(basal_node)

        all_basal_d_data[k][syn_lookup_key] = accepted_syn
        all_basal_d_data[k]['basal_node_locations'] = accepted_basal_nodes

    all_bipolar_basal_d = [bd for bd in all_basal_d_data if all_basal_d_data[bd]['elevation_angle']> upper_degree_bound or all_basal_d_data[bd]['elevation_angle']< lower_degree_bound]


    ### GET ALL COUNTS (SIM AND REAL) FOR AIS, ALL DENDRITES, BASAL DENDRITES:

    if make_ais_plot == True:

        info_to_get = [
            'pre_synaptic_site.neuron_id AS pre_seg_id',
            'post_synaptic_partner.neuron_id AS post_seg_id',
            'post_synaptic_partner.skel_type',
            ]

        results = cf.get_info_from_bigquery(info_to_get, 'post_synaptic_partner.neuron_id', all_bipolar_basal_d, synapse_db, client)

        axon2syntype = {}

        for syn in results:

            axon_id = str(syn['pre_seg_id'])
            basal_d_id = str(syn['post_seg_id'])
            skel_type = syn['skel_type']

            if axon_id not in axon2syntype:
                axon2syntype[axon_id] = {}
            
            if basal_d_id not in axon2syntype[axon_id]:
                axon2syntype[axon_id][basal_d_id] = []

            axon2syntype[axon_id][basal_d_id].append(skel_type)

        # Discard connections where a majority of the connection was not made up of AIS synapses:
        axon2syntype_ais_only = {}

        for axon_id in axon2syntype:

            for basal_d_id in axon2syntype[axon_id]:

                syn_types = axon2syntype[axon_id][basal_d_id]

                num_ais = syn_types.count('axon initial segment')

                prop_ais = num_ais/len(syn_types)

                if prop_ais >= 0.5:

                    if axon_id not in axon2syntype_ais_only:
                        axon2syntype_ais_only[axon_id] = {}
                    
                    axon2syntype_ais_only[axon_id][basal_d_id] = num_ais


        # For each axon, find out how many pairs of basal d cells there are where it makes a majority of it's syn on the AIS:
        pair_types = ('forward,forward', 'forward,reverse','reverse,reverse')
        ais_syn_counts = {a: {x: [] for x in pair_types} for a in ('real', 'simulated')}



        # Get real data:

        for axon_id in axon2syntype_ais_only.keys():
                
            all_partners = list(axon2syntype_ais_only[axon_id].keys())

            for pos, partner1 in enumerate(all_partners):

                for partner2 in all_partners[pos+1:]:

                    assert partner1 != partner2

                    dirs = get_sorted_bd_pair_type(partner1, partner2, all_basal_d_data)
                
                    ais_syn_counts['real'][dirs].append(axon_id)


        # Get simulated result:
        all_ax_results = [a for b in [ais_syn_counts['real'][x] for x in pair_types] for a in b]

        all_ais_syn_partners = []  

        for axon_id in axon2syntype_ais_only:
            for partner in axon2syntype_ais_only[axon_id]:
                for syn in range(axon2syntype_ais_only[axon_id][partner]):
                    all_ais_syn_partners.append(partner)

        for axon_id in all_ax_results:

            partner1 = choice(all_ais_syn_partners)
            partner2 = choice([x for x in all_ais_syn_partners if x != partner1])

            dirs = get_sorted_bd_pair_type(partner1, partner2, all_basal_d_data)
                
            ais_syn_counts['simulated'][dirs].append(axon_id)


        confints_dict = {}

        for stype in ('real', 'simulated'):
            data = [len(ais_syn_counts[stype][k]) for k in pair_types]
            confints = [[float(x[0]), float(x[1])] for x in multinomial_proportions_confint(data)]
            confints_dict[stype] = {pt: ci for pt, ci in zip(pair_types, confints)}

        real_counts = [len(ais_syn_counts['real'][k]) for k in pair_types]
        sim_counts = [len(ais_syn_counts['simulated'][k]) for k in pair_types]
        p_val = scipy.stats.chisquare(real_counts, f_exp=sim_counts)[1]
        pv_str = str(p_val)


        # Line plot:

        y_err = [confints_dict['real'][x][1] - confints_dict['real'][x][0] for x in pair_types]
        mean_vals = [np.mean(confints_dict['real'][x]) for x in pair_types]
        plt.errorbar(pair_types, mean_vals, fmt='.', yerr=y_err, capsize=5, elinewidth=0, markeredgewidth=2, color = 'slateblue', label='Observed')
        plt.plot(pair_types, [x/sum(real_counts) for x in sim_counts], 'or', label='Expected under null model')
        plt.rcParams["figure.figsize"] = (20,3)
        #plt.legend(loc='lower center', bbox_to_anchor=(0, 1))
        plt.xticks(pair_types)
        plt.xlabel('Type of Basal Dendrite Pair Targeted by Axon')
        plt.ylabel('Proportion of Axons')
        pv_str = str(p_val)
        plt.title(f'Proportion of AIS-targeting axons contacting each type of basal dendrite pair. P-Value: {pv_str}')
        plt.savefig(f'{results_dir}/Proportion of AIS-targeting axons contacting each type of basal dendrite pair, dot and CIs, upper bound {upper_degree_bound}, lower bound {lower_degree_bound}_input_types_{syn_lookup_key}.png')
        plt.clf()



    ### NON-AIS ANALYSIS:


    # Identify AIS or axon-contacting axons' synapses:
    basal_d_with_syn = [b for b in all_basal_d_data if syn_lookup_key in all_basal_d_data[b]]
    all_axons = [[s['pre_seg_id'] for s in all_basal_d_data[b][syn_lookup_key]] for b in basal_d_with_syn]
    all_axons = set([a for b in all_axons for a in b])

    info_to_get = ['pre_synaptic_site.neuron_id', 'post_synaptic_partner.skel_type']

    res = cf.get_info_from_bigquery(info_to_get, 'pre_synaptic_site.neuron_id', list(all_axons), synapse_db, client)

    axons_making_axonal_syn = set([str(x['neuron_id']) for x in res if 'axon' in x['skel_type']])

    # Then remove any further manually identified AIS-contacting axons:
    manually_removed_ais_axons = ('59559173605')
    axons_making_axonal_syn.update(manually_removed_ais_axons)

    for k in all_basal_d_data.keys():

        if syn_lookup_key not in all_basal_d_data[k]: continue
        accepted_syn = [syn for syn in all_basal_d_data[k][syn_lookup_key] if syn['pre_seg_id'] not in axons_making_axonal_syn]
        all_basal_d_data[k][syn_lookup_key] = accepted_syn


    # Organising data:
    basal_d_2_ei_count = {x: {'e': 0 , 'i': 0} for x in all_basal_d_data}
    axon_ei_count = {}
    axon_fwd_rev_syn_count = {}
    axon2partners = {}

    for agglo_id in all_basal_d_data.keys():

        if syn_lookup_key not in all_basal_d_data[agglo_id]: continue

        for syn in all_basal_d_data[agglo_id][syn_lookup_key]:
            pre_seg_id = syn['pre_seg_id']

            if pre_seg_id not in axon_fwd_rev_syn_count:
                axon_fwd_rev_syn_count[pre_seg_id] = {'forward': 0, 'reverse': 0}

            if pre_seg_id not in axon_ei_count:
                axon_ei_count[pre_seg_id] = {'e': 0 , 'i': 0}

            if syn['syn_type'] == 1:
                axon_ei_count[pre_seg_id]['i'] += 1
            if syn['syn_type'] == 2:
                axon_ei_count[pre_seg_id]['e'] += 1
            
            if all_basal_d_data[agglo_id]['elevation_angle'] > upper_degree_bound:
                axon_fwd_rev_syn_count[pre_seg_id]['forward'] += 1

            if all_basal_d_data[agglo_id]['elevation_angle'] < lower_degree_bound:
                axon_fwd_rev_syn_count[pre_seg_id]['reverse'] += 1

            if pre_seg_id not in axon2partners:
                axon2partners[pre_seg_id] = {}

            if agglo_id not in axon2partners[pre_seg_id]:
                axon2partners[pre_seg_id][agglo_id] = 0

            axon2partners[pre_seg_id][agglo_id] += 1

        ei_counts = Counter([x['syn_type'] for x in all_basal_d_data[agglo_id][syn_lookup_key]])

        for syn_code in ei_counts:

            if syn_code == 1:
                basal_d_2_ei_count[agglo_id]['i'] += ei_counts[syn_code]

            if syn_code == 2:
                basal_d_2_ei_count[agglo_id]['e'] += ei_counts[syn_code]


    all_e_axons = [x for x in axon_ei_count if axon_ei_count[x]['e']>axon_ei_count[x]['i']]
    all_i_axons = [x for x in axon_ei_count if axon_ei_count[x]['e']<axon_ei_count[x]['i']]


    axons_targeting_multi_bipolar_basal_d = []
    all_bipolar_basal_dendrites = set()

    for axon in axon2partners:

        this_axon_bipolar_partners = [p for p in axon2partners[axon] if all_basal_d_data[p]['elevation_angle'] > upper_degree_bound or all_basal_d_data[p]['elevation_angle'] < lower_degree_bound]
        all_bipolar_basal_dendrites.update(this_axon_bipolar_partners)

        if len(set(this_axon_bipolar_partners)) > 1:
            axons_targeting_multi_bipolar_basal_d.append(axon)

    with open(f'{results_dir}/axons_targeting_multi_bipolar_basal_d_noais_upper_bound_{upper_degree_bound}_lower_bound_{lower_degree_bound}_phase2_bound_{phase2_start}_input_types_{syn_lookup_key}.json', 'w') as fp:
        json.dump(axons_targeting_multi_bipolar_basal_d, fp)

    with open(f'{results_dir}/all_bipolar_basal_dendrites_upper_bound_{upper_degree_bound}_lower_bound_{lower_degree_bound}_phase2_bound_{phase2_start}_input_types_{syn_lookup_key}.json', 'w') as fp:
        json.dump(list(all_bipolar_basal_dendrites), fp)



    # Check Z-position of axons vs number of outgoing synapses
    req_info = ['location', 'pre_synaptic_site.neuron_id AS pre_id', 'post_synaptic_partner.neuron_id AS post_id']
    axon_data = cf.get_info_from_bigquery(req_info, 'pre_synaptic_site.neuron_id', list(axon2partners.keys()), synapse_db, client)


    axons2_zlocations = {x: [] for x in axon2partners.keys()}

    for result in axon_data:
        axons2_zlocations[str(result['pre_id'])].append(result['location']['z'])


    all_z_coords = []
    n_outgoing_syn = []

    for axon_id in axons2_zlocations:
        all_z_coords.append(np.mean(axons2_zlocations[axon_id], axis=0))
        n_outgoing_syn.append(len(axons2_zlocations[axon_id]))

    plt.figure(figsize=(20,10))
    plt.scatter(all_z_coords, n_outgoing_syn, s=1)
    plt.xlabel('Average Z-axis position of synapses (in 30nm voxels)')
    plt.ylabel('Number of Synapses')
    plt.savefig(f'{results_dir}/Z-axis position of synapses vs Number of Synapses for Basal dendrite input axons_phase2_bound_{phase2_start}_input_types_{syn_lookup_key}.png')


    # Check basal dendrite angle vs z location of its inputs:
    all_angles = []
    ave_input_locs = []

    for basal_d in all_basal_d_data.keys():
        print(basal_d)
        if syn_lookup_key in all_basal_d_data[basal_d]:

            all_angles.append(all_basal_d_data[basal_d]['elevation_angle'])

            all_inputs = set([x['pre_seg_id'] for x in all_basal_d_data[basal_d][syn_lookup_key]])

            mean_z_per_axon = [np.mean(axons2_zlocations[axon_id], axis=0) for axon_id in all_inputs]

            mean_z = np.mean(mean_z_per_axon, axis=0)

            ave_input_locs.append(mean_z)


    plt.figure(figsize=(20,10))
    plt.scatter(all_angles, ave_input_locs, s=2)
    plt.ylabel('Average Z-axis position of axons (in 30nm voxels)')
    plt.xlabel('Elevation Angle of Basal Dendrite')
    plt.savefig(f'{results_dir}/Average Z-axis position of axons vs Elevation Angle of Basal Dendrite for Basal dendrites_phase2_bound_{phase2_start}_input_types_{syn_lookup_key}.png')

    # Check the cell body location distributions:

    all_fwd_cb_locs = []
    all_rev_cb_locs = []
    all_fwd_syn_counts = []
    all_rev_syn_counts = []
    all_fwd_basal_d_node_counts = []
    all_rev_basal_d_node_counts = []

    for basal_d in all_basal_d_data:

        if syn_lookup_key not in all_basal_d_data[basal_d]: continue

        cb_z = all_basal_d_data[basal_d]['cb_loc'][2]/33

        #if cb_z > 3000 and cb_z < 4000:
        n_syn = len(all_basal_d_data[basal_d][syn_lookup_key])
        n_basal_nodes = len(all_basal_d_data[basal_d]['basal_node_locations'])

        if all_basal_d_data[basal_d]['elevation_angle'] > upper_degree_bound:
            all_fwd_cb_locs.append(cb_z)
            all_fwd_syn_counts.append(n_syn)
            all_fwd_basal_d_node_counts.append(n_basal_nodes)

        if all_basal_d_data[basal_d]['elevation_angle'] < lower_degree_bound:
            all_rev_cb_locs.append(cb_z)
            all_rev_syn_counts.append(n_syn)
            all_rev_basal_d_node_counts.append(n_basal_nodes)




    plt.hist(all_fwd_basal_d_node_counts, bins=20)
    plt.savefig(f'{results_dir}/Forward basal dendrite node mass_phase2_bound_{phase2_start}_input_types_{syn_lookup_key}.png')
    plt.clf()

    plt.hist(all_rev_cb_locs, bins=20)
    plt.savefig(f'{results_dir}/Reverse basal dendrite node mass_phase2_bound_{phase2_start}_input_types_{syn_lookup_key}.png')
    plt.clf()

    print('Mean fwd node count:', np.mean(all_fwd_basal_d_node_counts, axis=0))
    print('Mean rev node count:', np.mean(all_rev_basal_d_node_counts, axis=0))



    all_bipolar_bd_by_syn = []

    for bd in all_basal_d_data.keys():

        angle = all_basal_d_data[bd]['elevation_angle']

        if angle > upper_degree_bound or angle < lower_degree_bound:

            if syn_lookup_key in all_basal_d_data[bd]:

                for syn in all_basal_d_data[bd][syn_lookup_key]:

                    all_bipolar_bd_by_syn.append(bd)


    simulation_type = 'synapse_numbers'

    # Plot 2 syn 2 partner plots:
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
                    all_sim_partners_fwd = [x for x in all_sim_partners if all_basal_d_data[x]['elevation_angle'] > upper_degree_bound]
                    all_sim_partners_rev = [x for x in all_sim_partners if all_basal_d_data[x]['elevation_angle'] < lower_degree_bound]

                    all_bipolar_partners = all_sim_partners_fwd+all_sim_partners_rev

                    c = Counter(all_bipolar_partners)

                    bipolar_ids_ordered_by_count = [x[0] for x in sorted([(seg_id, c[seg_id]) for seg_id in c], key = lambda x: x[1], reverse=True)]
                    
                    partner1, partner2 = bipolar_ids_ordered_by_count[:2]


            dirs = []

            for p in (partner1, partner2):

                if all_basal_d_data[p]['elevation_angle'] > upper_degree_bound:
                    p_dir = 'forward'
                else:
                    assert all_basal_d_data[p]['elevation_angle'] < lower_degree_bound
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
        plt.xticks(pair_types)
        plt.xlabel('Type of Basal Dendrite Pair Targeted by Axon')
        plt.ylabel('Proportion of Axons')
        pv_str = str(p_val)
        plt.subplots_adjust(left=0.4, bottom=0.3)
        #plt.title(f'Proportion of {dtype} axons contacting each type of basal dendrite pair. P-Value: {pv_str}')
        plt.savefig(f'{results_dir}/Proportion of {dtype} axons contacting each type of basal dendrite pair, dot and CIs, upper bound {upper_degree_bound}, lower bound {lower_degree_bound}_input_types_{syn_lookup_key}_p_{pv_str}.png')
        plt.clf()


        fig = plt.figure() figsize=(3,6))
        pl = fig.add_subplot(1, 1, 1)
        x_coord = 0
        tick_labs = []

        for pair_type in pair_types:

            for stype, col in (('real', 'red'), ('simulated', 'blue')):

                mean_val = np.mean(confints_dict[stype][pair_type], axis=0)
                lower_ci = abs(mean_val-confints_dict[stype][pair_type][0])
                upper_ci = abs(mean_val-confints_dict[stype][pair_type][1])

                pl.errorbar(x_coord, mean_val, yerr=[[lower_ci], [upper_ci]], fmt='o', color=col, capsize = 10)
                x_coord += 1

                tick_labs.append(f'{pair_type} {stype}')


        pl.set_xticklabels(['']+tick_labs,ha='center')
        plt.ylabel('Proportion of Axons')
        #plt.title(f'Proportion of {dtype} axons contacting each type of basal dendrite pair. P-Value: {pv_str}')
        plt.savefig(f'{results_dir}/Proportion of {dtype} axons contacting each type of basal dendrite pair, upper bound {upper_degree_bound}, lower bound {lower_degree_bound}_input_types_{syn_lookup_key}_p_{pv_str}.png')
        plt.clf()

    

