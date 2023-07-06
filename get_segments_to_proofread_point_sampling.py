import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

from google.oauth2 import service_account
from google.auth.transport import requests as auth_request
from google.cloud import bigquery  
from random import shuffle
import numpy as np
import json
from common_functions_h01 import ParallelLocationRequester, get_vx_size_and_upper_bounds
import os
      


num_segs_to_sample = 500
save_dir = 'sampled_points_for_seg_proofreading_june_2022'
credentials_file = 'alexshapsoncoe.json' # or your credentials file 
volume_id = '964355253395:h01:goog14r0seg1_agg20200916c3_flat'
seg_info_db = 'goog14r0seg1.agg20200916c3_regions_types_circ_bounds_no_duplicates'
max_parallel_requests = 10
batch_size = 10



class MemoryCache():
    # Workaround for error, from: 'https://github.com/googleapis/google-api-python-client/issues/325':
    _CACHE = {}

    def get(self, url):
        return MemoryCache._CACHE.get(url)

    def set(self, url, content):
        MemoryCache._CACHE[url] = content





if __name__ == '__main__':

    # Get voxel sizes:
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/brainmaps'])
    scoped_credentials.refresh(auth_request.Request())
    requester = ParallelLocationRequester(max_parallel_requests, credentials_file)
    agglo_vx_s, agglo_ub = get_vx_size_and_upper_bounds(scoped_credentials, volume_id)

    url = f'https://brainmaps.googleapis.com/v1/volumes/{volume_id}/values'

    x_rand = np.random.randint(1, high=agglo_ub[0], size=num_segs_to_sample*100)
    y_rand = np.random.randint(1, high=agglo_ub[1], size=num_segs_to_sample*100)
    z_rand = np.random.randint(1, high=agglo_ub[2], size=num_segs_to_sample*100)

    points_agglo_vx = list(zip(x_rand, y_rand, z_rand))

    points_agglo_vx.sort(key=lambda x: (x[0], x[1], x[2]))

    chunk_size = 100

    all_results = []

    for i in range(int(len(points_agglo_vx)/chunk_size)):

        this_chunk = points_agglo_vx[i*chunk_size:(i*chunk_size)+chunk_size]

        try:
            retrieved_ids = requester.retrieve_locations(batch_size, url, this_chunk)
        except:
            scoped_credentials.refresh(auth_request.Request())
            requester = ParallelLocationRequester(max_parallel_requests, credentials_file)
            retrieved_ids = requester.retrieve_locations(batch_size, url, this_chunk)

        all_results.extend(list(zip(retrieved_ids, [(int(a[0]), int(a[1]), int(a[2])) for a in this_chunk])))


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    seg_saveable = volume_id.replace(':', '_')

    seg_ids = list(set([x[0] for x in all_results if x[0] != '0']))
    shuffle(seg_ids)
    seg_ids = seg_ids[:num_segs_to_sample]


    with open(f'{save_dir}/{num_segs_to_sample}_segs_sampled_using_random_points_from_{seg_saveable}.json', 'w') as fp:
        json.dump(seg_ids, fp)

    with open(f'{save_dir}/{num_segs_to_sample}_segs_sampled_using_random_points_from_{seg_saveable}_with_points.json', 'w') as fp:
        json.dump([x for x in all_results if x[0] in set(seg_ids)], fp)

    with open(f'{save_dir}/all_{len(all_results)}_sampled_random_points.json', 'w') as fp:
        json.dump(all_results, fp)




