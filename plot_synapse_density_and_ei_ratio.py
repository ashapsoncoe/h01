import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

from google.cloud import bigquery             
from google.oauth2 import service_account
from google.cloud import bigquery_storage  
from common_functions_h01 import fix_layer_mem
import numpy as np
import matplotlib.pyplot as plt
import time
import json

credentials_file = 'alexshapsoncoe.json' # or your credentials file
syn_db_name = 'lcht-goog-connectomics.goog14r0s5c3.synapse_c3_eirepredict_clean_dedup' 
save_dir = 'ei_syn_density_plots'
layers_file = 'cortical_bounds_circles.json'
syn_vx_size = [8,8,33]
cube_size = 10000 # in nm
sampling_factor = 1
number_total_syn = 149871669

if __name__ == "__main__":

    with open(layers_file, 'r') as fp:
        layers = json.load(fp)

    if not os.path.exists(f'{working_dir}\\{save_dir}'):
        os.mkdir(f'{working_dir}\\{save_dir}')


    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    query = f"""SELECT
                    CAST(location.x*{syn_vx_size[0]} as INT64) AS x,
                    CAST(location.y*{syn_vx_size[1]} as INT64) AS y,
                    CAST(location.z*{syn_vx_size[2]} as INT64) AS z,
                    CAST(type as INT64) AS t
                FROM {syn_db_name}
                """

    if sampling_factor !=1:
        query = query + f""" ORDER BY RAND() LIMIT {int(number_total_syn/sampling_factor)}"""

    c = bigquery.job.QueryJobConfig(allow_large_results = True)
    df = client.query(query, job_config=c).result().to_dataframe(bqstorage_client=bqstorageclient)#, progress_bar_type='tqdm_gui')

    min_x, max_x = min(df['x']), max(df['x'])
    min_y, max_y = min(df['y']), max(df['y'])
    min_z, max_z = min(df['z']), max(df['z'])

    array_size_x = (max_x//cube_size)+1
    array_size_y = (max_y//cube_size)+1
    array_size_z = (max_z//cube_size)+1


    all_counts = {x: np.zeros([array_size_x, array_size_y, array_size_z], dtype='int') for x in range(1,3)}

    df['x_cube_coords'] = df['x']//cube_size
    df['y_cube_coords'] = df['y']//cube_size
    df['z_cube_coords'] = df['z']//cube_size


    start = time.time()

    for row in df.index:

        if row%100000 == 0:
            print(time.time()-start, row)
            start = time.time()

        x = df.at[row, 'x_cube_coords']
        y = df.at[row, 'y_cube_coords']
        z = df.at[row, 'z_cube_coords']
        dtype = df.at[row, 't']
        all_counts[dtype][x, y, z] += 1


    # Get average E-I and average synapse count for each (X,Y), but disregarding zero counts to avoid areas without data from skewing:

    e_count = all_counts[2]*sampling_factor
    i_count = all_counts[1]*sampling_factor
    total_count =  e_count + i_count

    total_xy_avg = np.zeros([array_size_y, array_size_x])
    percent_e_avg = np.zeros([array_size_y, array_size_x])
    e_xy_avg = np.zeros([array_size_y, array_size_x])
    i_xy_avg = np.zeros([array_size_y, array_size_x])

    layer_names = ['White matter', 'Layer 6', 'Layer 5', 'Layer 4', 'Layer 3', 'Layer 2', 'Layer 1']
    measures = ['e_xy_avg', 'i_xy_avg', 'percent_e_avg', 'total_xy_avg']
    all_aves = {x: {a: [] for a in measures} for x in layer_names}

    tempy = []
        
    for x in range((max_x//cube_size)+1):
        print(x, max_x//cube_size)
        for y in range((max_y//cube_size)+1):

            tmp = fix_layer_mem(layers, np.array([[0, x*cube_size, y*cube_size]]))[0]
            tmp = [k for k in tmp if len(tmp[k]) >0]
            assert len(tmp) == 1
            layer = tmp[0]

            tempy.append([x*cube_size, y*cube_size, layer])

            total_count_this_xy = total_count[x, y] 
            e_count_this_xy = e_count[x, y] 
            i_count_this_xy = i_count[x, y] 

            zipped = zip(e_count_this_xy, i_count_this_xy, total_count_this_xy)

            non_zero_vals = [x for x in zipped if x[2] != 0]

            cubic_microns = (cube_size/1000)**3

            if len(non_zero_vals) >= 5:

                e_xy_avg[y,x] = np.mean([e for e, i, total in non_zero_vals], axis=0) / cubic_microns
                i_xy_avg[y,x] = np.mean([i for e, i, total in non_zero_vals], axis=0) / cubic_microns
                percent_e_avg[y,x] = np.mean([e/total for e, i, total in non_zero_vals], axis=0) * 100
                total_xy_avg[y,x] = np.mean([total for e, i, total in non_zero_vals], axis=0) / cubic_microns

                all_aves[layer]['e_xy_avg'].append(np.mean([e for e, i, total in non_zero_vals], axis=0) / cubic_microns)
                all_aves[layer]['i_xy_avg'].append(np.mean([i for e, i, total in non_zero_vals], axis=0) / cubic_microns)
                all_aves[layer]['percent_e_avg'].append(np.mean([e/total for e, i, total in non_zero_vals], axis=0) * 100)
                all_aves[layer]['total_xy_avg'].append(np.mean([total for e, i, total in non_zero_vals], axis=0) / cubic_microns)


    for measure in measures:
        for layer in layer_names:
            res = all_aves[layer][measure]
            print(measure, layer, f'average: {np.mean(res)}, based on {len(res)} synapses')


    all_d = (
        (e_xy_avg, 'Density of excitatory synapses (per cubic micron)', e_count, 'Synapses per cubic micron'), 
        (i_xy_avg, 'Density of inhibitory synapses (per cubic micron)', i_count, 'Synapses per cubic micron'), 
        (total_xy_avg, 'Density of all synapses (per cubic micron)', total_count, 'Synapses per cubic micron'), 
        (percent_e_avg, 'Excitatory percentage of synapses', None, r'% Excitatory synapses'),
    )

    for dataset, title, original, label in all_d:

        fig,ax = plt.subplots(1, figsize=(20,10))

        im = ax.imshow(dataset)
        im.axes.tick_params(color='white', labelcolor='white')
        ax.patch.set_facecolor('black')
        fig.patch.set_facecolor('black')  
        cb = plt.colorbar(im)
        cb.set_label(label, color='white', size=20)
        cb.ax.yaxis.set_tick_params(color='white', labelsize=18)
        cb.outline.set_edgecolor('white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

        for border in layers:

            x = border['center'][0]/(cube_size/1000)
            y = border['center'][1]/(cube_size/1000)
            rad = border['radius']/(cube_size/1000)
            circ = plt.Circle((x,y),rad, edgecolor='white', facecolor='none')
            ax.add_patch(circ)

        plt.title(title, color='white')
        plt.savefig(f'{working_dir}\\{save_dir}\\{title}_black_background.png')
        plt.clf()

