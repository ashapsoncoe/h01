import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

import json
from google.cloud import bigquery             
from google.oauth2 import service_account


credentials_file = 'alexshapsoncoe.json' # or your credentials file
save_path_dict = 'random_sample_of_10000i_10000e_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_0ax_dict.json'
save_path_list = 'random_sample_of_10000i_10000e_axons_from_each_gp_strength_c3_eirepredict_clean_dedup_0ax_list.json'
syn_db = 'goog14r0s5c3.synapse_c3_eirepredict_clean_dedup'
seg_info_db = 'goog14r0seg1.agg20200916c3_regions_types_circ_bounds_20230521'
skel_sql_table = 'goog14r0seg1.agg20200916c3_subcompartment_skeleton_counts_v1'
min_num_axon_nodes = 0
max_stregnth = True
sep_e_and_i = False

axons_per_type_per_stregnth = 10000

range_of_stregnths = range(1,21)



if __name__ == "__main__":

    credentials = service_account.Credentials.from_service_account_file(credentials_file)

    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    if sep_e_and_i == True:

        ei_types = ['inhibitory', 'excitatory']
    
    if sep_e_and_i == False:

        ei_types = ['all']


    results = {x: {} for x in range_of_stregnths}
    results_flat = []

    for strength in range_of_stregnths:

        for synapse_type in ei_types:

            if synapse_type == 'inhibitory':
                where_clause = 'InhibCount > ExciteCount'

            if synapse_type == 'excitatory':
                where_clause = 'InhibCount < ExciteCount'

            if synapse_type == 'all':
                where_clause = '1=1'

            if max_stregnth == True:
                query = f"""with pure_axons as (
                            select CAST(agglo_id AS STRING) as agglo_id
                            from {seg_info_db}
                            where type = 'pure axon fragment'
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

                            all_edges AS (
                            SELECT 
                                CAST(pre_synaptic_site.neuron_id AS STRING) AS pre_seg_id, 
                                CAST(post_synaptic_partner.neuron_id AS STRING) AS post_seg_id, 
                                COUNT(*) AS pair_count
                            FROM {syn_db}
                            GROUP BY pre_synaptic_site.neuron_id, post_synaptic_partner.neuron_id
                            ),

                            gp_edges AS (
                                SELECT pre_seg_id, MAX(pair_count) as max_partner_count
                                FROM all_edges
                                GROUP BY pre_seg_id
                            ),

                            e_and_i_counts as (
                            SELECT CAST(pre_synaptic_site.neuron_id AS STRING) AS agglo_id,
                                count(*) AS total,
                                sum(case when type = 1 then 1 else 0 end) AS InhibCount,
                                sum(case when type = 2 then 1 else 0 end) AS ExciteCount
                            FROM {syn_db}
                            GROUP BY agglo_id
                            ),

                            this_type_pre_segs as (
                            SELECT agglo_id,
                            FROM e_and_i_counts
                            WHERE {where_clause} 
                            ),

                            pure_axons_making_synapses as (
                                select agglo_id from this_type_pre_segs
                                intersect distinct
                                select agglo_id from pure_axons_large_enough
                            ),

                            pre_partners_with_correct_gp as (
                                select distinct pre_seg_id as agglo_id,
                                from gp_edges
                                where max_partner_count = {strength}
                            ),

                            pure_axons_with_correct_gp as (
                                select agglo_id from pre_partners_with_correct_gp
                                intersect distinct
                                select agglo_id from pure_axons_making_synapses
                            ),

                            segments_synapsing_onto_ais as (
                                select distinct CAST(pre_synaptic_site.neuron_id AS STRING) as seg_id
                                from {syn_db}
                                where 
                                SUBSTRING(post_synaptic_partner.tags[OFFSET((select offset 
                                    from unnest(post_synaptic_partner.tags) m with offset 
                                    where m LIKE '%class_label%'
                                ))], 13) = 'AIS'
                                
                            ),

                            non_ais_axons as (
                            select distinct agglo_id from pure_axons_with_correct_gp A
                            left join segments_synapsing_onto_ais B
                            on A.agglo_id = B.seg_id 
                            where B.seg_id IS NULL
                            )

                            select distinct agglo_id from non_ais_axons
                            ORDER BY RAND()
                            LIMIT {axons_per_type_per_stregnth}
                            """


            if max_stregnth == False:
                query = f"""with pure_axons as (
                            select CAST(agglo_id AS STRING) as agglo_id
                            from {seg_info_db}
                            where type = 'pure axon fragment'
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


                            e_and_i_counts as (
                            SELECT CAST(pre_synaptic_site.neuron_id AS STRING) AS agglo_id,
                                count(*) AS total,
                                sum(case when type = 1 then 1 else 0 end) AS InhibCount,
                                sum(case when type = 2 then 1 else 0 end) AS ExciteCount
                            FROM {syn_db}
                            GROUP BY agglo_id
                            ),

                            this_stregnth_pre_segs as (
                            SELECT agglo_id,
                            FROM e_and_i_counts
                            WHERE {where_clause} AND total = {strength}
                            ),

                            pure_axons_making_synapses as (
                                select agglo_id from this_stregnth_pre_segs
                                intersect distinct
                                select agglo_id from pure_axons_large_enough
                            ),

                            segments_synapsing_onto_ais as (
                                select distinct CAST(pre_synaptic_site.neuron_id AS STRING) as seg_id
                                from {syn_db}
                                SUBSTRING(post_synaptic_partner.tags[OFFSET((select offset 
                                    from unnest(post_synaptic_partner.tags) m with offset 
                                    where m LIKE '%class_label%'
                                ))], 13) = 'AIS'
                            ),

                            non_ais_axons as (
                            select distinct agglo_id from pure_axons_making_synapses A
                            left join segments_synapsing_onto_ais B
                            on A.agglo_id = B.seg_id 
                            where B.seg_id IS NULL
                            )

                            select distinct agglo_id from non_ais_axons
                            ORDER BY RAND()
                            LIMIT {axons_per_type_per_stregnth}
                            """

            res = [dict(x) for x in client.query(query).result()]
            this_layer_agglo_ids = [x['agglo_id'] for x in res]
            results[strength][synapse_type] = this_layer_agglo_ids
            results_flat.extend(this_layer_agglo_ids)

            print(f'{len(this_layer_agglo_ids)} {strength} {synapse_type} axons found')



    with open(save_path_list, 'w') as fp:
        json.dump(results_flat,  fp)

    with open(save_path_dict, 'w') as fp:
        json.dump(results,  fp)







