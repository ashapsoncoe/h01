from numpy import mean
from scipy.spatial.distance import euclidean
from google.cloud import bigquery             
from google.oauth2 import service_account
from google.cloud import bigquery_storage            
import common_functions as cf
import time
import networkx as nx
from zipfile import ZipFile
import os
import pickle
import json
import numpy as np
import igraph as ig
import pandas as pd

model_location = 'c:/work/FINAL/synapse_merge_model_skel_only_20210402.pkl' #'/home/alexshapsoncoe/drive/synapse_merge_model_skel_only_20210218.pkl'
credentials_file = 'C:/work/alexshapsoncoe.json' #'/home/alexshapsoncoe/drive/alexshapsoncoe.json'
output_dir = 'D:/temp_syn_merge/' #'/home/alexshapsoncoe/drive/synapse_merging_goog14r0s4_parallel/'
results_file = 'D:/temp_syn_merge/synapse_merge_predictions_20220224.json' #'/home/alexshapsoncoe/drive/synapse_merging_goog14r0s4_parallel/synapse_merge_predictions_20220224.json'
synapse_voxel_size = [8,8,33]
syn_db_name = 'goog14r0s4.synaptic_connections'
skel_dir = 'D:/goog14r0_20201123b_skel_class/' #'/home/alexshapsoncoe/drive/goog14r0_20201123b_skel_class/'
skel_voxel_size = [32,32,33]
skel_divisor = 42356404



def get_same_agglo_pairs(upper_threshold):

    print('Retrieving synapse pairs')
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    query = f"""WITH 
            all_edges AS (
                SELECT 
                    pre_synaptic_site.neuron_id AS pre_seg_id, 
                    post_synaptic_partner.neuron_id AS post_seg_id, 
                    COUNT(*) AS pair_count
                FROM {syn_db_name}
                GROUP BY pre_synaptic_site.neuron_id, post_synaptic_partner.neuron_id
                HAVING count(*) >= 2
                )

            SELECT 
                CAST(pre_synaptic_site.neuron_id as STRING) AS pre_agglo_id, 
                CAST(post_synaptic_partner.neuron_id as STRING) AS post_agglo_id, 
                CAST(pre_synaptic_site.id as STRING) AS pre_syn_id, 
                CAST(post_synaptic_partner.id as STRING) AS post_syn_id,
                CAST(post_synaptic_partner.centroid.x as STRING) AS post_centroid_x,
                CAST(post_synaptic_partner.centroid.y as STRING) AS post_centroid_y,
                CAST(post_synaptic_partner.centroid.z as STRING) AS post_centroid_z,
                CAST(pre_synaptic_site.centroid.x as STRING) AS pre_centroid_x,
                CAST(pre_synaptic_site.centroid.y as STRING) AS pre_centroid_y,
                CAST(pre_synaptic_site.centroid.z as STRING) AS pre_centroid_z,
                FROM {syn_db_name} AS all_syn
                INNER JOIN all_edges AS edge_info 
                    ON all_syn.pre_synaptic_site.neuron_id = edge_info.pre_seg_id AND all_syn.post_synaptic_partner.neuron_id = edge_info.post_seg_id
                WHERE pre_synaptic_site.neuron_id IS NOT NULL AND post_synaptic_partner.neuron_id IS NOT NULL
                """
    c = bigquery.job.QueryJobConfig(allow_large_results = True)
    df = client.query(query, job_config=c).result().to_dataframe(bqstorage_client=bqstorageclient) #, progress_bar_type='tqdm_gui')
    print('Retrieved synapse pairs')
        
    all_syn_seg_data = {}

    for x in df.index:
                
        pre_agglo_id = df.at[x, 'pre_agglo_id']
        post_agglo_id = df.at[x, 'post_agglo_id']

        pre_syn_id = df.at[x, 'pre_syn_id']
        post_syn_id = df.at[x, 'post_syn_id']
        
        post_centroid = tuple([
            int(df.at[x, 'post_centroid_x'])*synapse_voxel_size[0],
            int(df.at[x, 'post_centroid_y'])*synapse_voxel_size[1],
            int(df.at[x, 'post_centroid_z'])*synapse_voxel_size[2],
        ])

        pre_centroid = tuple([
            int(df.at[x, 'pre_centroid_x'])*synapse_voxel_size[0],
            int(df.at[x, 'pre_centroid_y'])*synapse_voxel_size[1],
            int(df.at[x, 'pre_centroid_z'])*synapse_voxel_size[2],
        ])

        common_info = {
            'pre_centroid': pre_centroid, 
            'post_centroid': post_centroid, 
            'pre_syn_id': pre_syn_id, 
            'post_syn_id': post_syn_id,
        }
        
        if pre_agglo_id not in all_syn_seg_data:
            all_syn_seg_data[pre_agglo_id] = {}
        
        if post_agglo_id not in all_syn_seg_data[pre_agglo_id]:
            all_syn_seg_data[pre_agglo_id][post_agglo_id] = []

        all_syn_seg_data[pre_agglo_id][post_agglo_id].append(common_info) 


    # Get all pairs and their distances:
    skel_site_pairs = [{} for x in range(10000)]

    same_agglo_pairs = []

    for pre_agglo_id in all_syn_seg_data.keys():

        for post_agglo_id in all_syn_seg_data[pre_agglo_id].keys():

            pair_synapses = all_syn_seg_data[pre_agglo_id][post_agglo_id]

            assert len(pair_synapses) > 1

            # Identify skeletons to obtain distances for:

            for syn1_pos, syn1 in enumerate(pair_synapses):

                for syn2 in pair_synapses[syn1_pos+1:]:

                    syn1_centre = mean([syn1['pre_centroid'], syn1['post_centroid']], axis=0)
                    syn2_centre = mean([syn2['pre_centroid'], syn2['post_centroid']], axis=0)
                    dist_nm = float(euclidean(syn1_centre, syn2_centre))

                    this_p = {  
                        'pre_agglo_id': pre_agglo_id,
                        'post_agglo_id': post_agglo_id,
                        'synapse_1': syn1,
                        'synapse_2': syn2,
                        'dist_nm':  dist_nm,
                            }

                    same_agglo_pairs.append(this_p)

                    if dist_nm < upper_threshold and dist_nm > lower_threshold:

                        for dtype in ['pre', 'post']:

                            agglo_id = this_p[f'{dtype}_agglo_id']

                            idx = int(int(agglo_id)/skel_divisor)
                            centroid1 = syn1[f'{dtype}_centroid']
                            syn_id1 = syn1[f'{dtype}_syn_id']
                            centroid2 = syn2[f'{dtype}_centroid']
                            syn_id2 = syn2[f'{dtype}_syn_id']

                            if agglo_id not in skel_site_pairs[idx]:
                                skel_site_pairs[idx][agglo_id] = []

                            skel_site_pairs[idx][agglo_id].append((syn_id1, centroid1, syn_id2, centroid2))

    print('Found '+ str(len(same_agglo_pairs)) + 'same seg pairs')

    with open(f'{output_dir}/same_agglo_pairs.json', 'w') as fp:
        json.dump(same_agglo_pairs, fp)

    with open(f'{output_dir}/skel_site_pairs.json', 'w') as fp:
        json.dump(skel_site_pairs, fp)


if __name__ == '__main__':

    if 'final_join_decisions.json' not in os.listdir(output_dir):

        with open(model_location, 'rb') as fp:
            merge_model = pickle.load(fp)

        lower_threshold = merge_model.lower_threshold
        upper_threshold = merge_model.upper_threshold

        if not (('same_agglo_pairs.json' in os.listdir(output_dir)) and ('skel_site_pairs.json' in os.listdir(output_dir))):

            get_same_agglo_pairs(upper_threshold)

        print('Loading skel_site_pairs')
        with open(f'{output_dir}/skel_site_pairs.json', 'r') as fp:
            skel_site_pairs = json.load(fp)
        print('Loaded skel_site_pairs')
            

        if 'skel_dists_temp' not in os.listdir(output_dir):
            os.mkdir(f'{output_dir}/skel_dists_temp')

        # Obtain skeleton distances:
        for x in range(10000):
        
            if f'{x}.json' in os.listdir(f'{output_dir}/skel_dists_temp/'):
                continue

            site_pair_d = skel_site_pairs[x]

            start = time.time()
            print(f'Starting dir {x}')
                
            results = {}

            skel_path = skel_dir + '/' + str(x) + '.zip'

            if not os.path.exists(skel_path):

                for agglo_id in site_pair_d.keys():
                    for syn_id1, centroid1, syn_id2, centroid2 in site_pair_d[agglo_id]:
                        new_id = [syn_id1, syn_id2]
                        new_id.sort()
                        new_id = '_'.join(new_id)
                        results[new_id] = 'no_skeleton'
                
                with open(f'{output_dir}/skel_dists_temp/{x}.json', 'w') as fp:
                    json.dump(results, fp)

                continue

            else:
                shard_dir = ZipFile(skel_path, 'r')

            this_batch_all_ids = set(site_pair_d.keys()) 

            if len(this_batch_all_ids) == 0:

                with open(f'{output_dir}/skel_dists_temp/{x}.json', 'w') as fp:
                    json.dump(results, fp)

                continue

            skel_data = cf.get_skel_data_from_shard_dir(this_batch_all_ids, shard_dir)
            
            skel_not_found_count = 0
            skel_found_count = 0

            for neuron_id in site_pair_d.keys():

                if neuron_id not in skel_data:
                    skel_not_found_count += 1
                    for syn_id1, centroid1, syn_id2, centroid2 in site_pair_d[neuron_id]:
                        new_id = [syn_id1, syn_id2]
                        new_id.sort()
                        new_id = '_'.join(new_id)
                        results[new_id] = 'no_skeleton'
                    continue
            
                skel_found_count += 1

                # Then make a graph of the segment in question:
                skel_g = cf.make_one_skel_graph_nx(skel_data[neuron_id], skel_voxel_size, join_components = False)

                syn_locations = [] 
                syn_ids = []

                for syn_id1, centroid1, syn_id2, centroid2 in site_pair_d[neuron_id]:
                    syn_ids.append(syn_id1)
                    syn_locations.append(centroid1)
                    syn_ids.append(syn_id2)
                    syn_locations.append(centroid2)


                syn_node_lookup_batch_size = 1000

                chosen_nodes = []
                num_batches = int(len(syn_locations)/syn_node_lookup_batch_size)

                for batch in range(num_batches+1):

                    syn_locs = syn_locations[batch*syn_node_lookup_batch_size:(batch+1)*syn_node_lookup_batch_size]
                    if syn_locs == []: break
                    chosen_node_batch = cf.get_skel_nodes_closest_to_synapses(syn_locs, skel_g, list(skel_g.nodes()))
                    chosen_nodes.extend(chosen_node_batch)

                assert len(chosen_nodes) == len(syn_ids)

                synid2node = {k: v for (k, v) in zip(syn_ids, chosen_nodes)}

                for syn_id1, centroid1, syn_id2, centroid2 in site_pair_d[neuron_id]:
                    
                    new_id = [syn_id1, syn_id2]
                    new_id.sort()
                    new_id = '_'.join(new_id)

                    skel_node1 = synid2node[syn_id1]
                    skel_node2 = synid2node[syn_id2]

                    if skel_node1 == skel_node2: 
                        results[new_id] = 0
                        continue

                    try:
                        sp = nx.shortest_path(skel_g, source=skel_node1, target=skel_node2)

                    except nx.exception.NetworkXNoPath:

                        results[new_id] = 'no_skeleton'
                        continue
                    
                        end_nodes = set([key for (key, value) in skel_g.degree() if value ==1])
                        cf.add_cc_bridging_edges_pairwise(skel_g, joining_nodes=end_nodes)

                        try:
                            sp = nx.shortest_path(skel_g, source=skel_node1, target=skel_node2)
                            
                        except nx.exception.NetworkXNoPath:
                            results[new_id] = 'no_skeleton'
                            continue

                    results[new_id] = cf.get_nm_dist_along_skel_path(skel_g, sp)

            with open(f'{output_dir}/skel_dists_temp/{x}.json', 'w') as fp:
                json.dump(results, fp)

            print(f'Dir {x} took {time.time()-start}, found skeletons for {skel_found_count} / {skel_found_count+skel_not_found_count} agglo IDs')



        del skel_site_pairs

        synpair2dist = {}

        for x in range(10000):

            with open(f'{output_dir}/skel_dists_temp/{x}.json', 'r') as fp:
                shard_dir = json.load(fp)
            
            synpair2dist.update(shard_dir)

        all_syn_combined_ids = {'pre': [], 'post': []}

        with open(f'{output_dir}/same_agglo_pairs.json', 'r') as fp:
            same_agglo_pairs = json.load(fp)

        for p in same_agglo_pairs:

            for dtype in ['pre', 'post']:

                combo_id = [p['synapse_1'][f'{dtype}_syn_id'], p['synapse_2'][f'{dtype}_syn_id']]
                combo_id.sort()
                combo_id = '_'.join(combo_id)

                all_syn_combined_ids[dtype].append(combo_id)
        
        if len(synpair2dist.keys()) > 1:
            ave_pre_skel_dist = np.mean([synpair2dist[x] for x in all_syn_combined_ids['pre'] if synpair2dist[x]!= 'no_skeleton'])
            ave_post_skel_dist = np.mean([synpair2dist[x] for x in all_syn_combined_ids['post'] if synpair2dist[x]!= 'no_skeleton'])

        # Use model to make final decison:
        if 'final_decisions_not_organized.json' not in os.listdir(output_dir):
            print('making final decisons')
            final_decisions = []

            no_skel_data_count = {'pre': 0, 'post': 0}

            for pair in same_agglo_pairs:

                syn1_id = pair['synapse_1']['pre_syn_id'] + '_' + pair['synapse_1']['post_syn_id']
                syn2_id = pair['synapse_2']['pre_syn_id'] + '_' + pair['synapse_2']['post_syn_id']
                pair_id = [syn1_id, syn2_id]
                pair_id.sort()

                dist = pair['dist_nm']

                if dist >= upper_threshold:
                    final_decisions.append([pair_id[0]])
                    final_decisions.append([pair_id[1]])
                    continue

                if dist <= lower_threshold:
                    final_decisions.append(pair_id)
                    continue

                # If not decided by distance thresholds, use skeleton distance:
                skel_dists = {'pre': None, 'post': None}

                for dtype in ['pre', 'post']:

                    combo_id = [pair['synapse_1'][f'{dtype}_syn_id'], pair['synapse_2'][f'{dtype}_syn_id']]
                    combo_id.sort()
                    combo_id = '_'.join(combo_id)

                    skel_dist = synpair2dist[combo_id]

                    if skel_dist == 'no_skeleton':
                        no_skel_data_count[dtype] +=1
                        skel_dists[dtype] = ave_pre_skel_dist
                    else:
                        skel_dists[dtype] = skel_dist

                predictors = [[max(skel_dists['pre'], skel_dists['post'])]]
            
                decision = int(merge_model.predict(predictors))
        
                if decision == 1:
                    final_decisions.append(pair_id)

                if decision == 0:
                    final_decisions.append([pair_id[0]])
                    final_decisions.append([pair_id[1]])

            with open(f'{output_dir}/final_decisions_not_organized.json', 'w') as fp:
                json.dump(final_decisions, fp)

            to_merge = [x for x in final_decisions if len(x)==2]

            final_join = [{'pre1': x[0].split('_')[0], 'post1': x[0].split('_')[1], 'pre2': x[1].split('_')[0], 'post2': x[1].split('_')[1]} for x in to_merge]
        
            with open(f'{output_dir}/final_join_decisions.json', 'w') as fp:
                json.dump(final_join, fp)
            
            print('Made all decisions')
            del same_agglo_pairs

    with open(f'{output_dir}/final_join_decisions.json', 'r') as fp:
        final_decisions  = json.load(fp)

    # Get tables of synapses to discard:
    g = ig.Graph()

    all_edges = [(f"{x['pre1']}_{x['post1']}", f"{x['pre2']}_{x['post2']}") for x in final_decisions]
    all_vertices = [a for b in all_edges for a in b]

    print(len(all_vertices), 'synapses in pairs')
    
    g.add_vertices(all_vertices)
    g.add_edges(all_edges)

    discarded_synapses = []

    for synapse_collection in g.components(mode='WEAK'):

        for synapse in synapse_collection[1:]:
            discarded_synapses.append(g.vs[synapse]['name'].split('_'))
    
    print(len(discarded_synapses), 'synapses discarded')
    
    df = pd.DataFrame(discarded_synapses, columns=['pre', 'post'])
        
    df.to_csv(f'{output_dir}/synapses_to_discard.csv', index=0)


### Apply merge model, giving new name 'with classes', to add classes to:

''' BigQuery command to use:
create table if not exists `goog14r0s4.synaptic_connections_ei_merge_correction1`
AS
select 

struct(    	
CAST(pre_synaptic_site.neuron_id AS INT64) AS neuron_id,		
CAST(pre_synaptic_site.id AS INT64) AS id,		
CAST(pre_synaptic_site.num_voxels AS INT64) AS num_voxels,			
struct(   	
CAST(pre_synaptic_site.centroid.x AS INT64) AS x,		
CAST(pre_synaptic_site.centroid.y AS INT64) AS y,		
CAST(pre_synaptic_site.centroid.z AS INT64) AS z
) as centroid,
CAST(pre_synaptic_site.type AS INT64) AS type,	
CAST(pre_synaptic_site.subtype AS INT64) AS subtype,		
CAST(pre_synaptic_site.confidence AS FLOAT64) AS confidence,	
struct(   	
struct(   	
CAST(pre_synaptic_site.bounding_box.start.x AS INT64) AS x,		
CAST(pre_synaptic_site.bounding_box.start.y AS INT64) AS y,		
CAST(pre_synaptic_site.bounding_box.start.z AS INT64) AS z
) as start,	
struct( 
CAST(pre_synaptic_site.bounding_box.size.x AS INT64) AS x,
CAST(pre_synaptic_site.bounding_box.size.y AS INT64) AS y,	
CAST(pre_synaptic_site.bounding_box.size.z AS INT64) AS z		
) as size,
CAST(pre_synaptic_site.bounding_box.description AS STRING) AS description,		
CAST(pre_synaptic_site.bounding_box.object_label AS STRING) AS object_label	
) as bounding_box,
CAST(pre_synaptic_site.mask AS BYTES) AS mask,
CAST(pre_synaptic_site.base_neuron_ids[SAFE_ORDINAL(1)] AS STRING) AS base_neuron_ids

) as pre_synaptic_site,



struct(    	
CAST(post_synaptic_partner.neuron_id AS INT64) AS neuron_id,		
CAST(post_synaptic_partner.id AS INT64) AS id,		
CAST(post_synaptic_partner.num_voxels AS INT64) AS num_voxels,			
struct(   	
CAST(post_synaptic_partner.centroid.x AS INT64) AS x,		
CAST(post_synaptic_partner.centroid.y AS INT64) AS y,		
CAST(post_synaptic_partner.centroid.z AS INT64) AS z
) as centroid,
CAST(post_synaptic_partner.type AS INT64) AS type,	
CAST(post_synaptic_partner.subtype AS INT64) AS subtype,		
CAST(post_synaptic_partner.confidence AS FLOAT64) AS confidence,	
struct(   	
struct(   	
CAST(post_synaptic_partner.bounding_box.start.x AS INT64) AS x,		
CAST(post_synaptic_partner.bounding_box.start.y AS INT64) AS y,		
CAST(post_synaptic_partner.bounding_box.start.z AS INT64) AS z
) as start,	
struct( 
CAST(post_synaptic_partner.bounding_box.size.x AS INT64) AS x,
CAST(post_synaptic_partner.bounding_box.size.y AS INT64) AS y,	
CAST(post_synaptic_partner.bounding_box.size.z AS INT64) AS z		
) as size,
CAST(post_synaptic_partner.bounding_box.description AS STRING) AS description,		
CAST(post_synaptic_partner.bounding_box.object_label AS STRING) AS object_label	
) as bounding_box,
CAST(post_synaptic_partner.mask AS BYTES) AS mask,
CAST(post_synaptic_partner.base_neuron_ids[SAFE_ORDINAL(1)] AS STRING) AS base_neuron_ids	

) as post_synaptic_partner,


struct( 	
CAST(location.x AS INT64) AS x,
CAST(location.y AS INT64) AS y,
CAST(location.z AS INT64) AS z	

) as location,

CAST(type AS INT64) AS type,	
CAST(contact_area AS FLOAT64) AS contact_area,
CAST(confidence AS FLOAT64) AS confidence,
struct( 	
struct( 	
CAST(bounding_box.start.x AS INT64) AS x,
CAST(bounding_box.start.y AS INT64) AS y,	
CAST(bounding_box.start.z AS INT64) AS z
) as start,	
struct( 		
CAST(bounding_box.size.x AS INT64) AS x,	
CAST(bounding_box.size.y AS INT64) AS y,	
CAST(bounding_box.size.z AS INT64) AS z
) as size,		
CAST(bounding_box.description AS STRING) AS	description,	
CAST(bounding_box.object_label AS INT64) AS object_label	

) as bounding_box

from goog14r0s4.synaptic_connections_ei t1
where not exists (
    select 1
    from goog14r0s4.duplicate_synapses_to_discard1 t2 
    where CAST(t1.pre_synaptic_site.id AS STRING) = t2.pre and CAST(t1.post_synaptic_partner.id AS STRING) = t2.post
)

'''
    