import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)


import json
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
from itertools import repeat
from common_functions_h01 import fix_layer_mem
import pandas as pd

# coauthors - Alexander Shapson-Coe, Luke Bailey

agglo_seg_dir = 'agg20200916c3_xy_only' # In nm coordinates - available at gs://h01_paper_public_files/agg20200916c3_xy_only
layer_bounds = 'cortical_bounds_circles.json'
output_dir = 'agg20200916c3_layer_classification_all_segs_circular_bounds'
cpu_num = 15
fragment_class_info_dir = 'axon_dendrite_astrocyte_cilia_pure_and_majority_agglo_20200916c3_all_classifications.json' # available at gs://h01_paper_public_files/axon_dendrite_astrocyte_cilia_pure_and_majority_agglo_20200916c3_all_classifications.json
cell_data_dir = 'agglo_20200916c3_cell_data.json'


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
    
    if f_name in os.listdir(f'{working_dir}/{output_dir}/temp/'):
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

    shard_layers = fix_layer_mem(bounds, checked_data)[0]

    for k in shard_layers.keys():
        shard_layers[k] = [int(x) for x in shard_layers[k]]
    
    with open(f'{working_dir}/{output_dir}/temp/{f_name}', 'w') as fp:
        json.dump(shard_layers, fp)

    print(f'Took {time.time()-start}s')

if __name__ == "__main__":
    
    if not os.path.exists(f'{working_dir}/{output_dir}'):
        os.mkdir(f'{working_dir}/{output_dir}')

    with open(cell_data_dir, 'r') as fp:
        cell_data = json.load(fp)

    with open(layer_bounds, "r") as f:
        bounds = json.load(f)
        
    if 'temp' not in os.listdir(f'{working_dir}/{output_dir}'):
        os.mkdir(f'{working_dir}/{output_dir}/temp')

        
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


    for f_name in os.listdir(f'{working_dir}/{output_dir}/temp'):
        
        print(f_name)

        # if f'region_and_type_{f_name}.csv' in os.listdir(output_dir): 
        #     continue
        
        with open(f'{working_dir}/{output_dir}/temp/{f_name}', 'r') as fp:
            shard_layers = json.load(fp)
            
        for k in shard_layers.keys():
            shard_layers[k] = set([str(x) for x in shard_layers[k]])
        
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
        
        df.to_csv(f'{working_dir}/{output_dir}/region_and_type_{f_name}.csv', index=0, header=False)

