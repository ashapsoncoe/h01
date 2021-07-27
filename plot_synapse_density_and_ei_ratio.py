from google.cloud import bigquery             
from google.oauth2 import service_account
from google.cloud import bigquery_storage  
from scipy.spatial.distance import cdist 
import numpy as np
import matplotlib.pyplot as plt
import time

credentials_file = 'C:/work//alexshapsoncoe.json'
syn_db_name = 'goog14r0s5c3.synaptic_connections_with_skeleton_classes'
skel_type_identifier = 'class_label' # 'skel_type' 
save_dir = 'c:/work/final/all_syn_density_plots'
syn_vx_size = [8,8,33]
cube_size = 10000 # in nm


if __name__ == "__main__":

    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    query = f"""SELECT
                    CAST(location.x*{syn_vx_size[0]} as INT64) AS x,
                    CAST(location.y*{syn_vx_size[1]} as INT64) AS y,
                    CAST(location.z*{syn_vx_size[2]} as INT64) AS z,
                    CAST(type as INT64) AS t,
                    LOWER(pre_synaptic_site.{skel_type_identifier}) as pre_skel_type,
                    LOWER(post_synaptic_partner.{skel_type_identifier}) as post_skel_type
                FROM {syn_db_name}
                ORDER BY RAND()
                LIMIT 16529234
                """

                #WHERE LOWER({pre_identifier}) = 'axon' 
                #AND LOWER({post_identier}) IN ('dendrite', 'soma', 'axon inital segment')

    c = bigquery.job.QueryJobConfig(allow_large_results = True)
    df = client.query(query, job_config=c).result().to_dataframe(bqstorage_client=bqstorageclient, progress_bar_type='tqdm_gui')



min_x, max_x = min(df['x']), max(df['x'])
min_y, max_y = min(df['y']), max(df['y'])
min_z, max_z = min(df['z']), max(df['z'])

array_size_x = (max_x//cube_size)+1
array_size_y = (max_y//cube_size)+1
array_size_z = (max_z//cube_size)+1


pre_strucs = set(df['pre_skel_type'])
post_strucs = set(df['post_skel_type'])

all_counts = {}

for pre_struc in pre_strucs:
    for post_struc in post_strucs:
        key = f'{pre_struc}-{post_struc}'
        all_counts[key] = {}
        all_counts[key][1] = np.zeros([array_size_x, array_size_y, array_size_z], dtype='int')
        all_counts[key][2] = np.zeros([array_size_x, array_size_y, array_size_z], dtype='int')


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
    key = f"{df.at[row, 'pre_skel_type']}-{df.at[row, 'post_skel_type']}"
    all_counts[key][dtype][x, y, z] += 1




# Get average E-I and average synapse count for each (X,Y), but disregarding zero counts to avoid areas without data from skewing:

for key in all_counts.keys():

    pre_part = key.split('-')[0]
    post_part = key.split('-')[1]

    if pre_part not in ('axon', 'unknown'): continue

    if post_part not in ('dendrite', 'unknown', 'ais', 'soma'): continue

    print(key)

    e_count = all_counts[key][2]*10
    i_count = all_counts[key][1]*10
    total_count =  e_count + i_count

    total_xy_avg = np.zeros([array_size_y, array_size_x])
    percent_e_avg = np.zeros([array_size_y, array_size_x])
    e_xy_avg = np.zeros([array_size_y, array_size_x])
    i_xy_avg = np.zeros([array_size_y, array_size_x])

    for x in range((max_x//cube_size)+1):

        for y in range((max_y//cube_size)+1):

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

    all_d = (
        (e_xy_avg, 'excitatory', e_count), 
        (i_xy_avg, 'inhibitory', i_count), 
        (total_xy_avg, 'all', total_count), 
        (percent_e_avg, 'percent_e', None),
    )

    for dataset, dtype, original in all_d:

        if dtype == 'percent_e':
            total_syn_n = total_count.sum()
            title = f'Excitatory percentage of {total_syn_n} {dtype} {key} synapses'
        else:
            total_syn_n = original.sum()
            title = f'Density of {total_syn_n} {dtype} {key} synapses (per cubic micron)'

        plt.figure(figsize=(20,10))
        plt.imshow(dataset)
        plt.colorbar()
        plt.title(title)
        plt.savefig(f'{save_dir}/{title}.png')
        plt.clf()







