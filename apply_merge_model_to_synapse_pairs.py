from networkx import Graph, connected_components
import pickle
import json
import numpy as np


model_location = 'C:/work/synapse_merging/synapse_merge_model_20200718.pkl'
path_segs_file = 'C:/work/synapse_merging/same_agglo_pairs_with_dists_and_paths_cuttoff_5000nm.json'
skel_pre_file = 'C:/work/synapse_merging/skel_distance_outputs/pairs_pre_skel_results_001_v2.json'
skel_post_file = 'C:/work/synapse_merging/skel_distance_outputs/pairs_post_skel_results_001_v2.json'
results_file = 'C:/work/synapse_merging/001_synapse_merge_predictions_20200722.json'

if __name__ == '__main__':
    
    # Organize pre and post skels:
    skel_d = {}

    for dtype, input_file in [['pre', skel_pre_file], ['post', skel_post_file]]:

        with open(input_file, 'r') as fp:
            skels = json.load(fp)

        for x in skels:
            if x[dtype + '_path_len_nm'] != None and x['dist_nm'] > 0:
                syn1_id = x['synapse_1'][2] + '_' + x['synapse_1'][3]
                syn2_id = x['synapse_2'][2] + '_' + x['synapse_2'][3]
                pair_id = [syn1_id, syn2_id]
                pair_id.sort()
                pair_id = '-'.join(pair_id)

                if pair_id not in skel_d:
                    skel_d[pair_id] = {}

                skel_d[pair_id][dtype] = x[dtype + '_path_len_nm'] / x['dist_nm']

    ave_pre_skel_dist = np.mean([skel_d[x]['pre'] for x in skel_d.keys() if 'pre' in skel_d[x]])
    ave_post_skel_dist = np.mean([skel_d[x]['post'] for x in skel_d.keys() if 'post' in skel_d[x]])

    # Use model to make final decison:


    with open(model_location, 'rb') as fp:
        merge_model = pickle.load(fp)

    with open(path_segs_file, 'r') as fp:
        agglo_pairs = json.load(fp)

    final_decisions = []

    no_pre_skel_data_count = 0
    no_post_skel_data_count = 0

    for pair in agglo_pairs:

        dist = pair['dist_nm']

        assert dist < 5000

        syn1_id = pair['synapse_1'][2] + '_' + pair['synapse_1'][3]
        syn2_id = pair['synapse_2'][2] + '_' + pair['synapse_2'][3]
        pair_id = [syn1_id, syn2_id]
        pair_id.sort()
        pair_id = '-'.join(pair_id)

        permitted_segs = set([pair['pre_agglo_id'], pair['post_agglo_id'], '0'])
        post_line = set(pair['post_line'])
        pre_line = set(pair['pre_line'])

        if not post_line.issubset(permitted_segs):
            postline_other = 1
        else:
            postline_other = 0

        if not pre_line.issubset(permitted_segs):
            preline_other = 1
        else:
            preline_other = 0

        if pair['post_agglo_id'] in pre_line:

            preline_post = 1
            try:
                pre_skel_dist = skel_d[pair_id]['pre']
            except:
                no_pre_skel_data_count +=1
                pre_skel_dist = ave_pre_skel_dist

        else:
            preline_post = 0
            pre_skel_dist = ave_pre_skel_dist


        if pair['pre_agglo_id'] in post_line:

            postline_pre = 1
            try:
                post_skel_dist = skel_d[pair_id]['post']
            except:
                no_post_skel_data_count +=1
                post_skel_dist = ave_post_skel_dist         

        else:
            postline_pre = 0
            post_skel_dist = ave_post_skel_dist

        predictors = [[  dist, 
                        postline_other, 
                        preline_other, 
                        pre_skel_dist, 
                        post_skel_dist, 
                        preline_post, 
                        postline_pre
                        ]]


        prob_of_join = merge_model.predict_proba(predictors)[0][1]

        if prob_of_join > merge_model.balanced_threshold:
            final_decisions.append(pair_id.split('-'))

        else:
            final_decisions.append([pair_id.split('-')[0]])
            final_decisions.append([pair_id.split('-')[1]])


    # Save all pair final decisions:

    G = Graph()
    G.add_nodes_from([a for b in final_decisions for a in b])
    G.add_edges_from([l for l in final_decisions if len(l) ==2])
    final_to_save = list([tuple(x) for x in connected_components(G)])

    syn_merge_d = {}
    for x in final_to_save:
        for y in x:
            syn_merge_d[y] = x

    with open(results_file, 'w') as fp:
        json.dump(syn_merge_d, fp)



