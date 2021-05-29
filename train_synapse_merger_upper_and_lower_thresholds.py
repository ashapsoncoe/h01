import json
import os
from copy import deepcopy
from networkx.readwrite.gml import literal_destringizer as destringizer
from networkx import read_gml
import pandas as pd
import math
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import pickle

home_dir = 'c:/work/FINAL'
list_of_segs_file = 'C:/work/FINAL/001_axons_pure_20_outputs.json'
pr_data_dir = 'C:/work/FINAL/001_pr_axons'
seg_path_file = 'C:/work/FINAL/gt_pair_data_5000nm_cutoff_with_skel_dists.json'
save_path_of_model = 'C:/work/FINAL/synapse_merge_model_skel_only_20210412.pkl'
lower_threshold_range = range(750,3000,50)
upper_threshold_range = range(1000,5000,50)

def get_accuracies(true_vals, binary_predictions):
    
    c = list(zip(true_vals, binary_predictions))
   
    joined_correctly = len([x for x in c if x[0]==1 and x[1]==1])
    joined_incorrectly = len([x for x in c if x[0]==1 and x[1]==0])
    separated_incorrectly = len([x for x in c if x[0]==0 and x[1]==1])
    separated_correctly = len([x for x in c if x[0]==0 and x[1]==0])

    if joined_correctly+separated_incorrectly >0:
        join_accuracy = joined_correctly / (joined_correctly+separated_incorrectly)
    else:
        join_accuracy = 'none truly joined'

    if separated_correctly+joined_incorrectly >0:
        sep_accuracy = separated_correctly / (separated_correctly+joined_incorrectly)
    else:
        sep_accuracy = 'none truly separate'
    
    print(f'Num false mergers: {joined_incorrectly} out of {joined_correctly+joined_incorrectly} merge decisions')
    print(f'Num false splits: {separated_incorrectly} out of {separated_incorrectly+separated_correctly} split decisions')
    print('Seperation accuracy:', sep_accuracy)
    print('Join accuracy:', join_accuracy)

    return sep_accuracy, join_accuracy, joined_incorrectly,separated_incorrectly

def get_data_from_gml(list_of_segs_file, pr_data_dir, seg_path_file):

    path_data = json.load(open(seg_path_file, 'r'))
    segs = json.load(open(list_of_segs_file, 'r'))

    files_todo = [x for x in os.listdir(pr_data_dir) if x.split('_')[2] in segs]
    files_todo = [x for x in files_todo if 'm' in x.split('_')[8] or 'm' in x.split('_')[10]]

    syn_data = {}
    merge_data = []

    for f in files_todo:
        g = read_gml(f'{pr_data_dir}/{f}')
        temp = destringizer(g.graph['info'])
        for k in temp['verified_synapses'].keys():
            if 'tp_synapses' in temp['verified_synapses'][k]:
                for k2 in temp['verified_synapses'][k]['tp_synapses'].keys():
                    syn_data[k2] = deepcopy(temp['verified_synapses'][k]['tp_synapses'][k2])
        
        merge_data.extend(temp['synapse_merge_decisions'])

    # Summarize data: 

    df = pd.DataFrame(columns=['true_condition', 'dist', 'pre_skel_dist', 'pre_skel_dist_n', 'post_skel_dist','post_skel_dist_n'])

    for x in merge_data:

        x['synapse_ids'].sort()
        combined_id = '-'.join(x['synapse_ids'])

        if x['decision'] == 'join':
            tc=1
        if x['decision'] == 'separate':
            tc=0

        assert x['decision'] in ['join', 'separate']

        df.loc[combined_id] = [tc, x['distance_nm'], None, None, None, None]

    for x in path_data:
        syn1_id = x['synapse_1'][2] + '_' + x['synapse_1'][3]
        syn2_id = x['synapse_2'][2] + '_' + x['synapse_2'][3]

        both_ids = [syn1_id, syn2_id]
        both_ids.sort()
        combined_id = '-'.join(both_ids)

        pre_dist = euclidean(x['synapse_1'][0], x['synapse_2'][0])
        post_dist = euclidean(x['synapse_1'][1], x['synapse_2'][1])
        
        if combined_id in df.index:

            df.at[combined_id, 'pre_skel_dist'] = x['pre_path_len_nm']

            if pre_dist == 0:
                df.at[combined_id, 'pre_skel_dist_n'] = 0
            else:
                df.at[combined_id, 'pre_skel_dist_n'] = x['pre_path_len_nm']/pre_dist

            df.at[combined_id, 'post_skel_dist'] = x['post_path_len_nm']

            if post_dist == 0:
                df.at[combined_id, 'post_skel_dist_n'] = 0
            else:
                df.at[combined_id, 'post_skel_dist_n'] = x['post_path_len_nm']/post_dist

            

    training_df = df.sample(int(len(df)*.8))
    test_df = df.drop(list(training_df.index))

    training_df.to_csv(f'{home_dir}/synapse_merge_train.csv')
    test_df.to_csv(f'{home_dir}/synapse_merge_test.csv')

    return training_df, test_df

def df_to_arrays(df, upper_threshold, lower_threshold):

    X = []
    Y = []

    simple_pred = []
    simple_true = []

    for x in df.index:

        dist = df.at[x, 'dist']

        if dist >= upper_threshold:
            simple_pred.append(0)
            simple_true.append(df.at[x, 'true_condition'])
            continue

        if dist <= lower_threshold:
            simple_pred.append(1)
            simple_true.append(df.at[x, 'true_condition'])
            continue

        pre_skel_dist = df.at[x, 'pre_skel_dist_n']
        post_skel_dist = df.at[x, 'post_skel_dist_n']

        X.append([max(pre_skel_dist, post_skel_dist)])
                        
        Y.append(df.at[x, 'true_condition'])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, simple_pred, simple_true


if __name__ == '__main__':

    if 'synapse_merge_train.csv' in os.listdir(home_dir) and 'synapse_merge_test.csv' in os.listdir(home_dir):
        training_df = pd.read_csv(f'{home_dir}/synapse_merge_train.csv', index_col=0)
        test_df = pd.read_csv(f'{home_dir}/synapse_merge_test.csv', index_col=0)
    else:
        training_df, test_df = get_data_from_gml(list_of_segs_file, pr_data_dir, seg_path_file)


    # Train model
    X, Y, simple_pred, simple_true = df_to_arrays(training_df, 5000, 0)

    clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000).fit(X, Y)

    current_best_thresholds = None
    current_best_auc_roc = 0

    for upper_threshold in upper_threshold_range:

        for lower_threshold in lower_threshold_range:

            if lower_threshold > upper_threshold: continue

            X, Y, simple_pred, simple_true = df_to_arrays(training_df, upper_threshold, lower_threshold)

            if len(X) == 0:
                pred_x = []
            else:
                pred_x = list(clf.predict(X))

            binary_predictions = [int(a) for a in simple_pred + pred_x]
            true_vals = [int(a) for a in simple_true + list(Y)]

            auc_roc = roc_auc_score(true_vals, binary_predictions)

            if auc_roc > current_best_auc_roc:
                current_best_auc_roc = auc_roc
                current_best_thresholds = (lower_threshold, upper_threshold)


    clf.lower_threshold, clf.upper_threshold = current_best_thresholds
    clf.train_auc_roc = current_best_auc_roc
    
    # Then try complete algorithm on test dataset:

    X, Y, simple_pred, simple_true = df_to_arrays(test_df, clf.upper_threshold, clf.lower_threshold)

    if len(X) == 0:
        pred_x = []
    else:
        pred_x = list(clf.predict(X))

    binary_predictions = [int(a) for a in simple_pred + pred_x]
    true_vals = [int(a) for a in simple_true + list(Y)]

    sep_accuracy, join_accuracy, joined_incorrectly,separated_incorrectly = get_accuracies(true_vals, binary_predictions)

    clf.test_separation_accuracy = sep_accuracy
    clf.test_join_accuracy = sep_accuracy
    clf.test_auc_roc = roc_auc_score(true_vals, binary_predictions)

    # Save the model:
    with open(save_path_of_model, 'wb') as fp:
        pickle.dump(clf, fp)

