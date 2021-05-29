from google.cloud import bigquery             
from google.oauth2 import service_account
from google.cloud import bigquery_storage  
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import sample
import pandas as pd
from math import isnan
import copy
import pickle
import time
from multiprocessing import Pool, set_start_method
from itertools import repeat, product
import common_functions as cf
import pandas as pd


'''
create table if not exists `goog14r0seg1.agg20200916c3_xy_only`
AS
SELECT 
            CAST(id as STRING) AS seg_id, 
            (bbox.start.x + (bbox.size.x/2))*8 AS x,
            (bbox.start.y + (bbox.size.y/2))*8 AS y
            FROM goog14r0seg1.agg20200916c3_objinfo
            WHERE id IS NOT NULL
'''

agglo_seg_dir = '/home/alexshapsoncoe/drive/agg20200916c3_xy_only' # In nm coordinates
layer_bounds = "/home/alexshapsoncoe/drive/cortical_bounds.pkl"
output_dir = "/home/alexshapsoncoe/drive/agg20200916c3_layer_classifications"
cpu_num = 15
fragment_class_info_dir = '/home/alexshapsoncoe/drive/axon_dendrite_astrocyte_cilia_pure_and_majority_agglo_20200916c3/all_classifications.json'
cell_data_dir = '/home/alexshapsoncoe/drive/agglo_20200916c3_cell_data.json'




# ------------------ #
# ---HELPER FUNCS--- #
# ------------------ #

def raw_data_to_checked_data(raw_data_df, cell_data):

    np_data = raw_data_df.to_numpy()

    agglo_id_to_xy = {i["agglo_seg"]: (i["true_x"], i["true_y"]) for i in cell_data}


    substituted_data = []
    for row in np_data:
        seg_id = row[0]
        x = row[1]
        y = row[2]

        if seg_id in agglo_id_to_xy:
            x = agglo_id_to_xy[seg_id][0]
            y = agglo_id_to_xy[seg_id][1]

        substituted_data.append([int(seg_id), int(x),int(y)])
    
    substituted_data = np.array(substituted_data)

    return substituted_data

def do_one_dir(f_name, cell_data, bounds):
    
    if f_name in os.listdir(f'{output_dir}/temp/'):
        return
    
    start = time.time()
    print(f_name)
    data = []
    with open(f'{agglo_seg_dir}/{f_name}') as f:
        for line in f:
            raw = json.loads(line)
            data.append((raw['seg_id'], raw['x'], raw['y']))
    
    df = pd.DataFrame(data)
    
    # -----CONVERT COORDS----- #
    # if id in aglo json, sub x and y values for true_x and true_y
    checked_data = raw_data_to_checked_data(df, cell_data)

    # -----CLASSIFY CELLS USING BOUND SOPS----- #
    upper_bounds = bounds["upper"]
    lower_bounds = bounds["lower"]

    shard_layers = cf.fix_layer_mem(upper_bounds, lower_bounds, checked_data)[0]
    
    with open(f'{output_dir}/temp/{f_name}', 'w') as fp:
        json.dump(shard_layers, fp)

    print(f'Took {time.time()-start}s')

if __name__ == "__main__":
        
    with open(cell_data_dir, 'r') as fp:
        cell_data = json.load(fp)

    with open(layer_bounds, "rb") as f:
        bounds = pickle.load(f)
        
    if 'temp' not in os.listdir(output_dir):
        os.mkdir(f'{output_dir}/temp')

        
    pool = Pool(cpu_num)
    pool.starmap(do_one_dir, zip(os.listdir(agglo_seg_dir), repeat(cell_data), repeat(bounds)))
    pool.close()
    pool.join()
    
    

    print('Loading class info')
    agglocell2type = {x['agglo_seg']: x['type'] for x in cell_data}

    with open(fragment_class_info_dir, 'r') as fp:
        fragment_class_info = json.load(fp)
        
    print('Loaded class info')
        
    for dtype in ['axon', 'dendrite', 'astrocyte', 'cilium']:
        for dtype2 in ['pure', 'majority']:
            fragment_class_info[dtype][dtype2] = set(fragment_class_info[dtype][dtype2])
            fragment_class_info[dtype][dtype2] -= agglocell2type.keys()


    for f_name in os.listdir(f'{output_dir}/temp'):
        
        print(f_name)

        if f'region_and_type_{f_name}.csv' in os.listdir(output_dir): 
            continue
        
        with open(f'{output_dir}/temp/{f_name}', 'r') as fp:
            shard_layers = json.load(fp)
            
        for k in shard_layers.keys():
            shard_layers[k] = set(shard_layers[k])
        
        all_info_for_df = [] 
        
        #frag_segs = {frag_type: fragment_class_info[frag_type]['pure'] | fragment_class_info[frag_type]['majority'] for frag_type in ['axon', 'dendrite', 'astrocyte']}
                
        for layer in shard_layers.keys():
            
            this_layer_classified_segs = []
            
            for frag_type in ['axon', 'dendrite', 'astrocyte', 'cilium']:
                
                for dtype in ['pure', 'majority']:
                    
                    name_to_use = f'{dtype} {frag_type} fragment'
                
                    final_segs = fragment_class_info[frag_type][dtype] & shard_layers[layer]

                    this_layer_classified_segs.extend([(x, layer, name_to_use) for x in final_segs])
            
            cell_segs = shard_layers[layer] & agglocell2type.keys()

            print(len(cell_segs), 'cell segs', layer)
            
            this_layer_classified_segs.extend([(x, layer, agglocell2type[x]) for x in cell_segs])
            
            all_info_for_df.extend(this_layer_classified_segs)
            
            unclassified_segs = shard_layers[layer] - set([x[0] for x in this_layer_classified_segs])
            
            all_info_for_df.extend([(x, layer, 'unclassified') for x in unclassified_segs])
        
        df = pd.DataFrame(all_info_for_df, columns=['agglo_id', 'region', 'type'])
        
        df.to_csv(f'{output_dir}/region_and_type_{f_name}.csv', index=0, header=False)





