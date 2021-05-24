import json
import os
import numpy as np
from itertools import combinations
from networkx.readwrite.gml import literal_destringizer as destringizer
from networkx import read_gml
import common_functions as cf
from copy import deepcopy
from collections import Counter
import neuroglancer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
import pickle
from google.cloud import bigquery             
from google.oauth2 import service_account


# Settings:
credentials_file = 'C:/work/FINAL/alexshapsoncoe.json'
input_file = 'C:/work/FINAL/001_axons_pure_20_outputs.json'
skel_dir = 'D:/phase1_001_20percent_skeletons_with_predictions/'
syn_merge_file = 'C:/work/FINAL/001_synapse_merge_predictions_20200719.json'
syn_shard_dir = 'D:/synapse_shards/'
pr_data_dir = 'C:/work/FINAL/001_pr_axons/'
save_dir = 'C:/work/FINAL/'
skel_voxel_size = [32,32,30]
skel_divisor = 1e7
ann_vs = [4,4,30]
synapse_voxel_size = [8,8,30]
syn_db_name = 'goog8_001.synaptic_connections'
neurite_type = 'axon'
max_stalk_len_range  = range(16,31,4) #[6000] #range(4000,10000,1000)  # In nm
lower_min_range = range(5,9,1) #[2400] #range(1000,5000, 1000)  # In nm
higher_min_range = range(16,31,4) #[4800] #range(4000,10000,1000)  # In nm
min_branch_len_range = range(0,10,2) #[1800] #range(0,2100,200) # In nm
nodes_or_lengths = 'nodes'

# Functions:

def get_ground_truth(pr_data_dir, segs):

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    files_todo = [x for x in os.listdir(pr_data_dir) if x.split('_')[2] in segs]
    files_todo = [x for x in files_todo if 'm' in x.split('_')[8] or 'm' in x.split('_')[10]]
    files_todo = [x for x in files_todo if 't' in x.split('_')[8] or 't' in x.split('_')[10]]
    files_todo = [x for x in files_todo if 'e' in x.split('_')[8] or 'e' in x.split('_')[10]]

    ground_truth = {}

    for f in files_todo:

        seg_id = f.split('_')[2]

        ground_truth[seg_id] = {}

        g = read_gml(f'{pr_data_dir}/{f}')
        temp = destringizer(g.graph['info'])

        # Get shaft ends:

        ground_truth[seg_id]['shaft_ends'] = []

        for e_type in temp['end_points'].keys():
            if e_type in ['uncorrected_split', 'natural', 'exit_volume']:
                for p in temp['end_points'][e_type]:
                    ground_truth[seg_id]['shaft_ends'].append(
                        [int(p[0]*ann_vs[0]), int(p[1]*ann_vs[1]), int(p[2]*ann_vs[2])]
                    )

        # Get synapse info:

        ground_truth[seg_id]['synapses'] = {}

        if neurite_type == 'axon':
            own_type = 'Pre-synaptic structure'
            bq_key = 'pre_synaptic_site'
            bq_partner_key = 'post_synaptic_partner'

        if neurite_type == 'dendrite':
            own_type = 'Post-synaptic structure'
            bq_key = 'post_synaptic_partner'
            bq_partner_key = 'pre_synaptic_site'

        required_info = [
            f'pre_synaptic_site.centroid.x*{synapse_voxel_size[0]} AS pre_x',
            f'pre_synaptic_site.centroid.y*{synapse_voxel_size[1]} AS pre_y',
            f'pre_synaptic_site.centroid.z*{synapse_voxel_size[2]} AS pre_z',
            f'post_synaptic_partner.centroid.x*{synapse_voxel_size[0]} AS post_x',
            f'post_synaptic_partner.centroid.y*{synapse_voxel_size[1]} AS post_y',
            f'post_synaptic_partner.centroid.z*{synapse_voxel_size[2]} AS post_z',
            'post_synaptic_partner.id AS post_syn_id',
            f'{bq_partner_key}.neuron_id AS partner_seg_id',
            'pre_synaptic_site.id AS pre_syn_id',
            f'{bq_key}.neuron_id AS own_seg_id',
        ]

        results = cf.get_info_from_bigquery(required_info, f'{bq_key}.neuron_id', [seg_id], syn_db_name, client)

        seg_syn_data = {}

        for r in results:
            pre_centroid = (r['pre_x'], r['pre_y'], r['pre_z'])
            post_centroid = (r['post_x'], r['post_y'], r['post_z'])
            
            assert str(r['own_seg_id']) == seg_id

            partner_id = str(r['partner_seg_id'])
            pre_syn_id = str(r['pre_syn_id'])
            post_syn_id = str(r['post_syn_id'])

            if partner_id not in seg_syn_data:
                seg_syn_data[partner_id] = []

            common_info = [pre_centroid, post_centroid, pre_syn_id, post_syn_id]

            seg_syn_data[partner_id].append(common_info)


        combined_syn_data = cf.get_combined_synapses_for_a_seg(seg_id, seg_syn_data, syn_merge_d)

        for partner in combined_syn_data.keys():
            for syn_id in combined_syn_data[partner].keys():

                ground_truth[seg_id]['synapses'][syn_id] = {}
                ground_truth[seg_id]['synapses'][syn_id]['location'] = combined_syn_data[partner][syn_id]
                ground_truth[seg_id]['synapses'][syn_id]['partner_id'] = partner

                consituent_ids = syn_id.split('-')

                classification_results = [
                    temp['verified_synapses']['outgoing']['tp_synapses'][syn_id]['classification_results']
                    for syn_id in consituent_ids
                ]

                all_class_decisions = [x[own_type] for x in classification_results]

                synapse_type = Counter(all_class_decisions).most_common(1)[0][0]

                if synapse_type in ['Terminal bouton', 'Spine']:
                    ground_truth[seg_id]['synapses'][syn_id]['syn_type'] = 'stalk'
                else:
                    ground_truth[seg_id]['synapses'][syn_id]['syn_type'] = 'shaft'

                if synapse_type in ['Terminal bouton', 'Spine']:
                    root_locations = [x[f'{synapse_type} root'] for x in classification_results if f'{synapse_type} root' in x]
                    mean_rl = np.mean(root_locations, axis=0)
                    mean_rl = [int(mean_rl[0]*ann_vs[0]), int(mean_rl[1]*ann_vs[1]),int(mean_rl[2]*ann_vs[2])]
                else:
                    mean_rl = None

                ground_truth[seg_id]['synapses'][syn_id]['root_location'] = mean_rl

    return ground_truth

def test_root_and_bb_identification(seg_ids, max_stalk_len, lower_min, higher_min, min_branch_len, s_model=None):

    correct_node = 0
    incorrect_node = 0
    individual_bb_performances = []

    skel_graphs_temp = deepcopy(skel_graphs)

    for seg_id in seg_ids:

        g_temp = skel_graphs_temp[seg_id]

        g_temp = cf.assign_root_and_bb_nodes(g_temp, max_stalk_len, lower_min, higher_min, min_branch_len, nodes_or_lengths)

        root_pred_d = {n['synid']: n['rootnode'] for n in g_temp.vs if 's' in n['ntype']}

        for syn_id in root_pred_d.keys():

            if 'root_node' in ground_truth[seg_id]['synapses'][syn_id].keys():

                true_node = ground_truth[seg_id]['synapses'][syn_id]['root_node']

                selected_root_node = root_pred_d[syn_id]
                
                if selected_root_node == true_node:
                    correct_node += 1
                else:
                    incorrect_node+=1

        bb_nodes_pred = set([n.index for n in g_temp.vs if 'b' in n['ntype'] or 'r' in n['ntype']])
        bb_nodes_actual = ground_truth[seg_id]['bb_nodes']

        non_bb_nodes_pred = set([n.index for n in skel_graphs[seg_id].vs]) - bb_nodes_pred
        non_bb_nodes_actual = set([n.index for n in skel_graphs[seg_id].vs]) - bb_nodes_actual

        bb_nodes_tpr = len(bb_nodes_actual&bb_nodes_pred)/len(bb_nodes_actual)
        non_bb_nodes_tpr = len(non_bb_nodes_actual&non_bb_nodes_pred)/len(non_bb_nodes_actual)

        individual_bb_performances.append(np.mean([bb_nodes_tpr, non_bb_nodes_tpr]))

    node_performance = correct_node / (correct_node+incorrect_node)
    ave_bb_performance = np.mean(individual_bb_performances)

    X, Y = get_XY_for_synapses(seg_ids, skel_graphs_temp, ground_truth)

    if s_model == None:

        s_model = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000).fit(X, Y)

    syn_score = s_model.score(X, Y)

    #train_performance = np.mean([node_performance, ave_bb_performance, syn_score])

    return node_performance, ave_bb_performance, syn_score, s_model

def get_XY_for_synapses(seg_ids, skel_graphs, ground_truth):

    Y = []
    X = []

    for seg_id in seg_ids:

        g = skel_graphs[seg_id]

        node_d = {n['synid'] : n.index for n in g.vs if 's' in n['ntype']}

        for syn_id in ground_truth[seg_id]['synapses'].keys():

            if syn_id not in node_d:
                continue

            stype = ground_truth[seg_id]['synapses'][syn_id]['syn_type']

            if stype not in ['shaft', 'stalk']:
                continue

            node = node_d[syn_id]

            X.append([g.vs[node]['stalkdist'], g.vs[node]['nhr']])

            if stype == 'stalk':
                Y.append(1)
            else:
                Y.append(0)

    return X, Y
    


if __name__ == '__main__':

    if syn_merge_file != None:
        with open(syn_merge_file, 'r') as fp:
            syn_merge_d = json.load(fp)
    else:
        syn_merge_d = {}

    with open(input_file, 'r') as fp:
        neurites_todo = json.load(fp)

    ground_truth = get_ground_truth(pr_data_dir, neurites_todo)

    neurites_todo = list(ground_truth.keys())
    
    # Split GT into test and train:
    test_size = int(len(neurites_todo)*0.3)
    neurites_to_test = neurites_todo[-test_size:]
    neurites_to_train = neurites_todo[:-test_size]

    # Make skeleton graphs for all training and test segs:
    gt_syn_only = {x: ground_truth[x]['synapses'] for x in ground_truth.keys()}

    organized_neurites = [[] for i in range(10000)]

    for neurite in neurites_todo:
        organized_neurites[int(int(neurite)/skel_divisor)].append(neurite)

    skel_graphs = {}

    for i in range(10000):
        skel_graphs_batch = cf.make_skel_graphs_batch(i, organized_neurites[i], skel_dir, skel_voxel_size, credentials_file, neurite_type, syn_db_name, syn_location_d=gt_syn_only)
        skel_graphs.update(skel_graphs_batch)

    # Assign root nodes to ground-truth:
    for seg_id in neurites_todo:
        for syn_id in ground_truth[seg_id]['synapses'].keys():
            if ground_truth[seg_id]['synapses'][syn_id]['syn_type'] == 'stalk':
                loc = ground_truth[seg_id]['synapses'][syn_id]['root_location']
                selected_node = cf.get_skel_nodes_closest_to_synapses([loc], skel_graphs[seg_id], [n.index for n in skel_graphs[seg_id].vs])[0]
                ground_truth[seg_id]['synapses'][syn_id]['root_node'] = selected_node

    # Then get backbone ground-truth:
    for seg_id in ground_truth.keys():

        end_points = ground_truth[seg_id]['shaft_ends']
        end_nodes = cf.get_skel_nodes_closest_to_synapses(end_points, skel_graphs[seg_id], [n.index for n in skel_graphs[seg_id].vs])
        all_bb_nodes = set()

        for source in end_nodes:
            targets = end_nodes[end_nodes.index(source)+1:]
            bb_paths = skel_graphs[seg_id].get_all_shortest_paths(source, to=targets, mode='OUT')
            for path in bb_paths:
                all_bb_nodes.update(set(path))
        
        ground_truth[seg_id]['bb_nodes'] = all_bb_nodes


    # First get accurate identification of the root and bb nodes, on a per-synapse basis, using cross-validation:

    performance_data = []

    #all_combos = list(combinations(neurites_to_train, len(neurites_to_train)-test_size))

    kf = KFold(n_splits=5)
    for mini_train_ix, mini_test_ix in kf.split(neurites_to_train):

        print('Starting new test/train split')

        mini_train = [neurites_to_train[x] for x in mini_train_ix]
        mini_test = [neurites_to_train[x] for x in mini_test_ix]

        summary = {'train_node_performance': 0}

        for max_stalk_len in max_stalk_len_range:
            for lower_min in lower_min_range:
                for higher_min in higher_min_range:
                    for min_branch_len in min_branch_len_range:
                        #print(max_stalk_len, lower_min, higher_min, min_branch_len)
                        scores_and_model = test_root_and_bb_identification(mini_train, max_stalk_len, lower_min, higher_min, min_branch_len)
                        node_performance, ave_bb_performance, syn_performance, s_model = scores_and_model
                        #print('perf:', node_performance, ave_bb_performance, syn_performance)
                        
                        new_best = False

                        if node_performance > summary['train_node_performance']:
                            new_best = True

                        if node_performance == summary['train_node_performance']:

                            if syn_performance > summary['train_syn_performance']:
                                new_best = True

                            if ave_bb_performance > summary['train_bb_performance']:
                                new_best = True

                        if new_best == True:
                            summary['max_stalk_len'] = max_stalk_len
                            summary['lower_min'] = lower_min
                            summary['higher_min'] = higher_min
                            summary['min_branch_len'] = min_branch_len
                            summary['train_node_performance'] = node_performance
                            summary['train_bb_performance'] = ave_bb_performance
                            summary['train_syn_performance'] = syn_performance
                            summary['s_model'] = s_model


        # Performance on test fraction:
        vals = test_root_and_bb_identification(mini_test, summary['max_stalk_len'], summary['lower_min'], summary['higher_min'], summary['min_branch_len'], s_model=summary['s_model'])
        node_performance, ave_bb_performance, syn_performance, s_model = vals
        summary['test_node_performance'] = node_performance
        summary['test_bb_performance'] = ave_bb_performance
        summary['test_syn_performance'] = syn_performance

        performance_data.append(summary)

    max_stalk_len = Counter([x['max_stalk_len'] for x in performance_data]).most_common(1)[0][0]
    lower_min = Counter([x['lower_min'] for x in performance_data]).most_common(1)[0][0]
    higher_min = Counter([x['higher_min'] for x in performance_data]).most_common(1)[0][0]
    min_branch_len = Counter([x['min_branch_len'] for x in performance_data]).most_common(1)[0][0]

    skel_graphs_temp = deepcopy(skel_graphs)
    for seg_id in skel_graphs_temp.keys():
        skel_graphs_temp[seg_id] = cf.assign_root_and_bb_nodes(skel_graphs_temp[seg_id], max_stalk_len, lower_min, higher_min, min_branch_len, nodes_or_lengths)

    train_X, train_Y = get_XY_for_synapses(neurites_to_train, skel_graphs_temp, ground_truth)
    test_X, test_Y = get_XY_for_synapses(neurites_to_test, skel_graphs_temp, ground_truth)

    clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000).fit(train_X, train_Y)

    print(f'Chosen parameters: max stalk legnth: {max_stalk_len}, lower path threshold: {lower_min}, higher path threshold: {higher_min}, min branch len: {min_branch_len}')


    # Then on held-out test fraction:
    node_performance, ave_bb_performance, syn_performance, s_model = test_root_and_bb_identification(
                            neurites_to_test, max_stalk_len, lower_min, higher_min, min_branch_len, s_model=clf)

    print(f'Average performance on held-out test fraction was: node performance: {node_performance}, and bb performance: {ave_bb_performance}, and syn performance: {syn_performance}')

    chosen_parameters = {
        'max_stalk_len': max_stalk_len,
        'lower_min': lower_min,
        'higher_min': higher_min,
        'min_branch_len':  min_branch_len
    }

    performance_summary = {
        'root_node_prediction': float(node_performance),
        'shaft_node_prediction': float(ave_bb_performance),
        'synapse_type_prediction': float(syn_performance)
    }

    with open(f'{save_dir}/{neurite_type}_skeleton_pruning_parameters.json', 'w') as fp:
        json.dump(chosen_parameters, fp)

    with open(f'{save_dir}/{neurite_type}_skeleton_pruning_test_performance.json', 'w') as fp:
        json.dump(performance_summary, fp)

    with open(f'{save_dir}/{neurite_type}_synapse_classification_model.pkl', 'wb') as fp:
        pickle.dump(clf, fp)

