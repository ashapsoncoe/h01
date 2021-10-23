import json
from random import choices, choice, shuffle
from collections import Counter
import numpy as np
from numpy.core.fromnumeric import mean
from statsmodels.stats.proportion import multinomial_proportions_confint
import matplotlib.pyplot as plt
import os
import scipy          
import common_functions as cf
from google.cloud import bigquery              
from google.oauth2 import service_account
import pandas as pd
import neuroglancer


em = 'brainmaps://964355253395:h01:goog14r0_8nm'
agglo_seg = 'brainmaps://964355253395:h01:goog14r0seg1_agg20200916c3_flat'
data_dir = 'c:/work/final/Layer_6_basal_cell_partners_agglo_20200916c3_Oct_2021_pure_axons_only'
all_partners_save_dir = 'c:/work/final/basal_dendrites_plots_and_data/bd_all_partners_ng_states_Oct_2021_pure_axons_only'
just_bd_nodes_save_dir = 'c:/work/final/basal_dendrites_plots_and_data/bd_nodes_ng_states_Oct_2021_pure_axons_only'
just_seg_lists_save_dir =  'c:/work/final/basal_dendrites_plots_and_data/bd_seg_lists_Oct_2021_pure_axons_only'
plots_results_dir = 'C:/work/FINAL/basal_dendrites_plots_and_data/'
basal_id_classification_table_path = 'c:/work/final/goog14_L6basal_matrix_c3_404.csv'
cred_path = 'c:/work/alexshapsoncoe.json'
synapse_db = 'goog14r0s5c3.synaptic_connections_ei_conserv_reorient_fix_ei_spinecorrected_merge_correction2'
vx_size = [8,8,33]
neuron_color = 'red'
basal_inputs_color = 'green'
non_basal_inputs_color = 'blue'
ais_inputs_color = 'orange'


if __name__ == "__main__":

    for d in [all_partners_save_dir, just_bd_nodes_save_dir, just_seg_lists_save_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:

        dimensions = neuroglancer.CoordinateSpace(
            scales=vx_size,
            units='nm',
            names=['x', 'y', 'z']   )

        s.showSlices = False
        s.dimensions = dimensions
        s.crossSectionScale = 0.22398
        s.projectionScale = 4000
        s.position = np.array([257946, 178200, 2643])

        s.layers['EM'] = neuroglancer.ImageLayer(source=em)
        s.layers['L56 pyramidal neuron'] = neuroglancer.SegmentationLayer(source=agglo_seg)
        s.layers['basal_dendrite_inputs'] = neuroglancer.SegmentationLayer(source=agglo_seg)
        s.layers['non_basal_dendrite_inputs'] = neuroglancer.SegmentationLayer(source=agglo_seg)
        s.layers['ais_inputs'] = neuroglancer.SegmentationLayer(source=agglo_seg)
        s.layers['basal_node_locations'] = neuroglancer.AnnotationLayer()
        s.layers['manually_marked_basal_node'] = neuroglancer.AnnotationLayer()


    for f in os.listdir(data_dir):

        basal_id =  f.split('_')[0]

        with open(f'{data_dir}/{f}', 'r') as fp:
            basal_data = json.load(fp)


        if os.path.exists(f'{just_bd_nodes_save_dir}/bd_{basal_id}_basal_points.json'):
            continue

        if 'basal_synapses' in basal_data:
            basal_d_input_axons = set([x['pre_seg_id'] for x in basal_data['basal_synapses'] if str(x['pre_seg_id'])!='None'])
        else:
            basal_d_input_axons = set()


        info_to_get = [
                'pre_synaptic_site.neuron_id AS pre_seg_id',
                'LOWER(post_synaptic_partner.class_label) AS post_type',
                'LOWER(pre_synaptic_site.class_label) AS pre_type',
                ]

        results = cf.get_info_from_bigquery(info_to_get, 'post_synaptic_partner.neuron_id', [basal_id], synapse_db, client)

        pre_seg2type = {}

        for x in results:
            pre_seg2type[str(x['pre_seg_id'])] = x['pre_type']

        basal_d_input_axons = set([x for x in basal_d_input_axons if pre_seg2type[x] == 'axon'])
        non_basal_d_inputs = set([str(x['pre_seg_id']) for x in results if x['pre_type']=='axon' and str(x['pre_seg_id'])!='None'])
        ais_inputs = set([str(x['pre_seg_id']) for x in results if x['pre_type']=='axon' and x['post_type'] == 'ais' and str(x['pre_seg_id'])!='None'])

        basal_d_input_axons -= ais_inputs
        non_basal_d_inputs -= ais_inputs
        non_basal_d_inputs -= basal_d_input_axons


        with viewer.txn() as s:

            s.layers['basal_dendrite_inputs'].visible = True
            s.layers['non_basal_dendrite_inputs'].visible = True
            s.layers['ais_inputs'].visible = True

            s.layers['L56 pyramidal neuron'].segments = set([basal_id])
            s.layers['basal_dendrite_inputs'].segments = basal_d_input_axons
            s.layers['non_basal_dendrite_inputs'].segments = non_basal_d_inputs
            s.layers['ais_inputs'].segments = ais_inputs

            
            s.layers['L56 pyramidal neuron'].segment_colors[int(basal_id)] = neuron_color

            for seg in basal_d_input_axons:
                s.layers['basal_dendrite_inputs'].segment_colors[int(seg)] = basal_inputs_color 

            for seg in non_basal_d_inputs:
                s.layers['non_basal_dendrite_inputs'].segment_colors[int(seg)] = non_basal_inputs_color 

            for seg in ais_inputs:
                s.layers['ais_inputs'].segment_colors[int(seg)] = ais_inputs_color 

            point_annotations = []

            if 'basal_node_locations' in basal_data:
                for pos, point_raw in enumerate(basal_data['basal_node_locations']):

                    if pos%10 == 0:

                        point = np.array([point_raw[a]/vx_size[a] for a in range(3)])

                        pa = neuroglancer.PointAnnotation(id=pos, description=pos, point=point)

                        point_annotations.append(pa)

            s.layers['basal_node_locations'].annotations = point_annotations

            try:
                s.position = point
            except NameError:
                pass

            basal_point = np.array([basal_data['basal_d_com'][a]/vx_size[a] for a in range(3)])
            print(basal_point)
            basal_pa = neuroglancer.PointAnnotation(id='b', description='b', point=point)
            
            s.layers['manually_marked_basal_node'].annotations = [basal_pa]

            s.layers['manually_marked_basal_node'].annotationColor = 'white'



        json_state_for_seeing_partners = viewer.state.to_json()

        with open(f'{all_partners_save_dir}/bd_{basal_id}_all_partners.json', 'w') as fp:
            json.dump(json_state_for_seeing_partners, fp)


        with viewer.txn() as s:
            
            s.layers['basal_dendrite_inputs'].segments = set()
            s.layers['non_basal_dendrite_inputs'].segments = set()
            s.layers['ais_inputs'].segments = set()
            s.layers['basal_dendrite_inputs'].segment_colors = {}
            s.layers['non_basal_dendrite_inputs'].segment_colors = {}
            s.layers['ais_inputs'].segment_colors = {}

            s.layers['basal_dendrite_inputs'].visible = False
            s.layers['non_basal_dendrite_inputs'].visible = False
            s.layers['ais_inputs'].visible = False


            s.layers['L56 pyramidal neuron'].objectAlpha = 0.50


        json_state_for_seeing_points = viewer.state.to_json()

        with open(f'{just_bd_nodes_save_dir}/bd_{basal_id}_basal_points.json', 'w') as fp:
            json.dump(json_state_for_seeing_points, fp)
        
        seg_lists = {
            'basal_dendrite_partners': list(basal_d_input_axons),
            'other_dendrite_partners': list(non_basal_d_inputs),
            'ais_partners': list(ais_inputs),
        }

        with open(f'{just_seg_lists_save_dir}/bd_{basal_id}_seg_lists.json', 'w') as fp:
            json.dump(seg_lists, fp)


    # Loading data:

    basal_d_table = pd.read_csv(basal_id_classification_table_path)
    forward_ids = [int(basal_d_table.at[x, 'google_agglo_id']) for x in basal_d_table.index if basal_d_table.at[x, ' group(0:outlier'] == 1]
    reverse_ids = [int(basal_d_table.at[x, 'google_agglo_id']) for x in basal_d_table.index if basal_d_table.at[x, ' group(0:outlier'] == 2]


    all_basal_d_data = {}

    for f in os.listdir(data_dir):

        agglo_id = f.split('_')[0]

        with open(f'{data_dir}/{f}', 'r') as fp:
            all_basal_d_data[agglo_id] = json.load(fp)

        with open(f'{just_seg_lists_save_dir}/bd_{agglo_id}_seg_lists.json', 'r') as fp:
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

                if str(syn['syn_type']) == '1':
                    axon_ei_count[pre_seg_id]['i'] += 1

                if str(syn['syn_type']) == '2':
                    axon_ei_count[pre_seg_id]['e'] += 1
                    
                if int(agglo_id) in forward_ids:
                    axon_fwd_rev_syn_count[pre_seg_id]['forward'] += 1

                if int(agglo_id) in reverse_ids:
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




        all_e_axons = [x for x in axon_ei_count if axon_ei_count[x]['e'] > axon_ei_count[x]['i']]
        all_i_axons = [x for x in axon_ei_count if axon_ei_count[x]['e'] < axon_ei_count[x]['i']]


        all_bipolar_bd_by_syn = []

        for bd in all_basal_d_data.keys():

            if int(bd) in forward_ids+reverse_ids:
                
                if post_struc_syn_type in all_basal_d_data[bd]:

                    for syn in all_basal_d_data[bd][post_struc_syn_type]:

                        all_bipolar_bd_by_syn.append(bd)


        axons_targeting_multi_bipolar_basal_d = []

        for axon in axon2partners:

            this_axon_bipolar_partners = [p for p in axon2partners[axon] if int(p) in forward_ids+reverse_ids]

            if len(set(this_axon_bipolar_partners)) > 1:
                axons_targeting_multi_bipolar_basal_d.append(axon)




        # Plot 2 syn 2 partner plots:

        pair_types = ('forward,forward', 'forward,reverse','reverse,reverse')

        for dtype, dataset in (('all', all_e_axons+all_i_axons), ('excitatory', all_e_axons), ('inhibitory', all_i_axons)):

            if post_struc_syn_type == 'ais_synapses' and dtype != 'inhibitory': continue

            print(f'For {dtype} {post_struc_syn_type}')

            counts = {a: {x: [] for x in pair_types} for a in ('real', 'simulated')}

            if post_struc_syn_type == 'ais_synapses':
                axons = [x for x in dataset if sum(axon_ei_count[x].values())>=2 and x in axons_targeting_multi_bipolar_basal_d]
            else:
                axons = [x for x in dataset if sum(axon_ei_count[x].values())==2 and x in axons_targeting_multi_bipolar_basal_d]

            print('Number of axons', len(axons))

            for ctype in ('real', 'simulated'):

                for axon in axons:
                    
                    if ctype == 'real':
                        this_axon_partners = list(axon2partners[axon].keys())
                        this_axon_partners = [x for x in this_axon_partners if int(x) in forward_ids+reverse_ids]
                        shuffle(this_axon_partners)
                        partner1, partner2 = this_axon_partners[:2]

                    if ctype == 'simulated':

                        partner1 = choice(all_bipolar_bd_by_syn)
                        partner2 = choice([x for x in all_bipolar_bd_by_syn if x != partner1])

                    dirs = []

                    for p in (partner1, partner2):

                        if int(p) in forward_ids:
                            p_dir = 'forward'
                        else:
                            assert int(p) in reverse_ids
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
            observed_proportions = [x/sum(real_counts) for x in real_counts]
            expected_proportions = [x/sum(sim_counts) for x in sim_counts]
            p_val = scipy.stats.chisquare(real_counts, f_exp=sim_counts)[1]
            pv_str = str(p_val)
            print('Observed proportions:', observed_proportions)
            print('Expected proportions:', expected_proportions)
            print('P-value:', pv_str)


            # Line plot:
            y_err = [confints_dict['real'][x][1] - confints_dict['real'][x][0] for x in pair_types]
            plt.errorbar(pair_types, observed_proportions, fmt='.', yerr=y_err, capsize=5, elinewidth=0, markeredgewidth=2, color = 'slateblue', label='Observed')
            plt.plot(pair_types, expected_proportions, 'or', label='Expected under null model')
            plt.rcParams["figure.figsize"] = (20,3)
            plt.xticks(range(3), pair_types)
            plt.xlabel('Type of Basal Dendrite Pair Targeted by Axon')
            plt.ylabel('Proportion of Axons')
            plt.subplots_adjust(left=0.4, bottom=0.3)
            plt.savefig(f'{plots_results_dir}/Proportion of {post_struc_syn_type} {dtype} axons contacting each type of basal dendrite pair, dot and CIs, pv {pv_str}, n {len(axons)}.png')
            plt.clf()



