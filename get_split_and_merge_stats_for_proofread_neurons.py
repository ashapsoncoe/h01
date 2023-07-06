import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

import json
from common_functions_h01 import get_base2agglo
from google.cloud import bigquery              
from google.oauth2 import service_account
import pandas as pd


proofread_cells_dir = 'proofread104_neurons_20210511'
cred_path = 'alexshapsoncoe.json' # or your credentials file 
output_file_name = 'proofread_neurons_split_and_merge_stats.csv'

base_agglo_maps = {
    'c2': 'goog14r0seg1.agg20201123c2_resolved',
    'c3': 'goog14r0seg1.agg20200916c3_resolved_fixed',
}

agglo_edge_lists = {
    'c2': 'goog14r0seg1.agg20201123c2_agglotoedges',
    'c3': 'goog14r0seg1.agg20200916c3_agglotoedges_fixed',
}





if __name__ == '__main__':

    credentials = service_account.Credentials.from_service_account_file(cred_path)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)

    results = {}

    for f in os.listdir(proofread_cells_dir):

        with open(f'{proofread_cells_dir}/{f}', 'r') as fp:
            cell_data = json.load(fp)

        seg_id = f.split('_')[2]

        results[seg_id] = {}

        all_base_segs = [str(a) for b in cell_data['base_segments'].values() for a in b]

        results[seg_id]['N base segments in cell'] = len(all_base_segs)
        results[seg_id]['N base segment merge errors'] = len(cell_data['base_seg_merge_points'])

    
        # Get all split and merge errors recorded during each agglomeration proofreading

        for agglo_to_use in ('c2', 'c3'):

            base2agglo = get_base2agglo(all_base_segs, base_agglo_maps[agglo_to_use], client)
            all_agglo_segs = set(base2agglo.values())

            q = ','.join([str(x) for x in all_agglo_segs])
            db = agglo_edge_lists[agglo_to_use]
            query = f"""SELECT agglo_id, label_a, label_b FROM {db} WHERE agglo_id IN ({q})"""
            query_job = client.query(query)  
            agglo_segs_graph_edges = [[str(row['label_a']), str(row['label_b'])] for row in query_job.result()]

            removed_segs = set([a for b in agglo_segs_graph_edges for a in b]) - set(all_base_segs)

            merge_edges = []
            directly_merged_segs = set()

            for base_seg_a, base_seg_b in agglo_segs_graph_edges:

                if (base_seg_a in all_base_segs) and (base_seg_b in removed_segs):
                    merge_edges.append((base_seg_a, base_seg_b))
                    directly_merged_segs.add(base_seg_b)

                if (base_seg_a in removed_segs) and (base_seg_b in all_base_segs):
                    merge_edges.append((base_seg_a, base_seg_b))
                    directly_merged_segs.add(base_seg_a)

            results[seg_id][f'{agglo_to_use} N removed base segments'] = len(removed_segs)
            results[seg_id][f'{agglo_to_use} merge error corrections'] = len(directly_merged_segs)
            results[seg_id][f'{agglo_to_use} split error corrections'] = len(all_agglo_segs)-1


    df = pd.DataFrame(results).transpose()
    df.to_csv(output_file_name)     

            
        






