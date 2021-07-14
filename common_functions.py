import networkx as nx
import igraph as ig
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from zipfile import ZipFile
import os
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
import snappy
import urllib
import zlib
from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.cloud import bigquery             
import json
from google.auth.transport import requests as auth_request
import math
import time
import threading
import json
import numpy as np
import threading
import time
import urllib
import zlib
from absl import logging
from google.oauth2 import service_account
from google.auth.transport import requests as auth_request
from itertools import combinations
from collections import Counter
from random import choice
import gc




class ParallelLocationRequester:

  def __init__(self,
               max_parallel_requests,
               credentials_file,
               scopes=['https://www.googleapis.com/auth/brainmaps'],
               simulate_delays=False):
    logging.debug(f'ParallelRequester with {max_parallel_requests} parallelism')
    self._max_parallel_requests = max_parallel_requests
    logging.debug(f'Generating credentials using {credentials_file}')
    self._credentials = service_account.Credentials.from_service_account_file(credentials_file)
    self._scoped_credentials = self._credentials.with_scopes(scopes)
    # Note: This does not automatically refresh the token. It should work for
    # ~30 minutes, but then the token needs to be refreshed, or the script
    # restarted.
    self._scoped_credentials.refresh(auth_request.Request())
    self._simulate_delays = simulate_delays
    self._responses = []

  def retrieve_locations(self, batch_size, url, locations):
    logging.debug('Generating splits')
    pending_requests = []
    for i in range(0, len(locations), batch_size):
      batched_locations = locations[i:i + batch_size]
      request = {'locations': []}
      for location in batched_locations:
        request['locations'].append(','.join([str(x) for x in location]))
      pending_requests.append(request)

    logging.debug('Generating header')
    headers = {
        'Authorization': 'Bearer ' + self._scoped_credentials.token,
        'Content-type': 'application/json',
        'Accept-encoding': 'gzip',
    }

    logging.debug('Beginning parallel requests')
    self._responses = len(pending_requests) * [None]
    for request_id, body in enumerate(pending_requests):
      while threading.active_count() > self._max_parallel_requests:
        # Wait around for another thread to complete
        logging.info(
            f'There are {threading.active_count()-1}/'
            f'{self._max_parallel_requests} threads running, sleeping...')
        time.sleep(0.25)
      logging.debug('Thread is free, launching request')
      new_thread = threading.Thread(
          target=self._single_request, args=(request_id, url, headers, body))
      new_thread.start()
    logging.debug('Parallel requests complete, flattening results')

    while threading.active_count() > 1:
      # Wait around for threads to finish...
      logging.info(
          f'There are {threading.active_count()-1}/'
          f'{self._max_parallel_requests} threads remaining, waiting...')
      time.sleep(0.25)

    responses = []
    for batched_response in self._responses:
      uint64StrList = batched_response.get('uint64StrList', None)
      if not uint64StrList:
        continue
      values = uint64StrList.get('values', None)
      if not values:
        continue
      responses += values
    return responses

  def _single_request(self, request_id, url, headers, body):
    logging.info(f'Making request for request id:{request_id}')
    data = json.dumps(body).encode('utf-8')
    logging.debug(f'Request {request_id} body: {data}')
    req = urllib.request.Request(url, data, headers)
    resp = urllib.request.urlopen(req)
    http_response = resp.read()
    response = zlib.decompress(http_response, 16 + zlib.MAX_WBITS)
    response = json.loads(response.decode('utf-8'))
    if self._simulate_delays:
      delay = np.random.uniform(0, 1)
      logging.debug(
          f'Simulating a delay in request id {request_id} for {delay}s')
      time.sleep(delay)
    self._responses[request_id] = response
    logging.info(f'Request {request_id} complete')


def check_thread_alive(thr):
    thr.join(timeout=0.0)
    return thr.is_alive()


class MemoryCache():
    # Workaround for error 'https://github.com/googleapis/google-api-python-client/issues/325':
    _CACHE = {}

    def get(self, url):
        return MemoryCache._CACHE.get(url)

    def set(self, url, content):
        MemoryCache._CACHE[url] = content

def get_vx_size_and_upper_bounds(scoped_credentials, seg_id):

    service = build(
        'brainmaps',
        discoveryServiceUrl='https://brainmaps.googleapis.com/$discovery/rest?key=AIzaSyBAaxW5lG3PhdRxsj6tQGa322PoJ2WBUz8', 
        version='v1',
        credentials=scoped_credentials,
        cache=MemoryCache()  
        )

    req_id = seg_id.split('brainmaps://')[-1]

    if req_id.count(':') == 3:
        req_id = req_id[:req_id.rindex(':')]

    vol_data = service.volumes().get(volumeId=req_id).execute()

    g = vol_data['geometry'][0]

    pixel_size = np.array([int(g['pixelSize'][x]) for x in ['x', 'y','z']])
    upper_bounds = np.array([int(g['volumeSize'][x]) for x in ['x', 'y','z']])
    
    return pixel_size, upper_bounds

def get_orthogonal_vec(input_vec, angle): # in radians

    # From https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector

    a = input_vec[0]
    b = input_vec[1]
    c = input_vec[2]

    new_a = (-b * math.cos(angle)) - (((a*c)/math.sqrt(a**2+b**2)) * math.sin(angle))

    new_b = (a * math.cos(angle)) - (((b*c)/math.sqrt(a**2+b**2)) * math.sin(angle))

    new_c = math.sqrt(a**2+b**2) * math.sin(angle)

    new_vec = np.array([new_a, new_b, new_c])

    assert np.dot(input_vec, new_vec) < 1*10**-10

    return new_vec

def get_orthogonal_vec_stack(input_stack, random_angles_array): 

    cos_of_angles = np.cos(random_angles_array)
    sin_of_angles = np.sin(random_angles_array)
    a_vals = input_stack[:,0]
    b_vals = input_stack[:,1]
    c_vals = input_stack[:,2]
    sqrt_ab = np.sqrt((a_vals**2)+(b_vals**2))
    assert 0 not in sqrt_ab
    new_as = (-b_vals * cos_of_angles) - (((a_vals*c_vals)/sqrt_ab) * sin_of_angles)
    new_bs = (a_vals * cos_of_angles) - (((b_vals*c_vals)/sqrt_ab) * sin_of_angles)
    new_cs = sqrt_ab * sin_of_angles
    output_array_stack = np.vstack((new_as, new_bs, new_cs)).T

    return output_array_stack

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return (x,y,z)

def points_and_surrounds_to_string(x,y,z):

    each_string = []

    for x_pos in [str(x-1),str(x),str(x+1)]:
        for y_pos in [str(y-1),str(y),str(y+1)]:
            for z_pos in [str(z-1),str(z),str(z+1)]:
                each_string.append(x_pos+','+y_pos+','+z_pos)

    return each_string

def get_size_ordered_list_of_files(d):

    pairs = [(os.path.getsize(os.path.join(d, f)), f) for f in os.listdir(d)]
    pairs.sort(key=lambda s: s[0], reverse=True)
    
    return [x[1] for x in pairs]

def plot_a_3D_hull(array):
    hull = ConvexHull(array)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.plot(array.T[0], array.T[1], array.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(array[s, 0], array[s, 1], array[s, 2], "r-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    plt.show()
    plt.clf()

def get_second_order_polynomial_fit_to_layers(layer2_points, layer4_points, show_fit=True):

    x2 = np.array([x[0] for x in layer2_points])
    y2 = np.array([x[1] for x in layer2_points])
    
    coefficients2 = np.polyfit(x2, y2, 2)
    poly2 = np.poly1d(coefficients2)
    new_x2 = np.linspace(min(x2), max(x2))
    new_y2 = poly2(new_x2)
    
    x4 = np.array([x[0] for x in layer4_points])
    y4 = np.array([x[1] for x in layer4_points])
    
    coefficients4 = np.polyfit(x4, y4, 2)
    poly4 = np.poly1d(coefficients4)
    new_x4 = np.linspace(min(x4), max(x4))
    new_y4 = poly4(new_x4)

    if show_fit == True:
        plt.plot(x2,y2,'o')
        plt.plot(x2, y2, "o", new_x2, new_y2)
        plt.plot(x4,y4,'o')
        plt.plot(x4, y4, "o", new_x4, new_y4)
        plt.show()
        plt.clf()
    
    return coefficients2, coefficients4

def plot_empirical_vs_theoretical_dist(model, retrieved_dists, results_dir, file_name, x_lab, y_lab, title):

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    x = np.linspace(
        model['distr'].ppf(0.01, *model['arg'], loc=model['loc'], scale=model['scale']),
        model['distr'].ppf(0.99, *model['arg'], loc=model['loc'], scale=model['scale']), 
        100
        )

    rv = model['distr'](
                *model['arg'],
                loc=model['loc'], 
                scale=model['scale']
        )

    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='pdf')

    ax.hist(retrieved_dists, density=True, bins=model['num_bins'])
    ax.legend(loc='best', frameon=False)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title)

    plt.savefig(f'{results_dir}/{file_name}')

    # To avoid error caused by https://stackoverflow.com/questions/53897248/matplotlib-fail-to-allocate-bitmap
    plt.close('all')
    gc.collect()

def get_bb_nodes_by_lp(g):
   
    end_nodes = [n.index for n in g.vs if g.degree(n.index)==1]

    node_types = [n['ntype'] for n in g.vs]

    assert node_types.count('r') + node_types.count('sr') < 2

    has_root_nodes = ('r' in node_types or 'sr' in node_types)
    
    candidate_paths = []

    for source in end_nodes:
        targets = end_nodes[end_nodes.index(source)+1:]
        bb_paths = g.get_all_shortest_paths(source, to=targets, mode='OUT')

        for path in bb_paths:
            types_in_path = set([g.vs[x]['ntype'] for x in path])
            if 't' not in types_in_path:
                if has_root_nodes == True:
                    if 'r' in types_in_path or 'sr' in types_in_path:
                        candidate_paths.append(path)
                else:
                    candidate_paths.append(path)

    # If misplaced root node results in zero paths:
    if len(candidate_paths) == 0:

        for n in g.vs:

            if n['ntype'] in ['r', 't']:
                n['ntype'] = 'c'

            if n['ntype'] == 'sr':
                n['ntype'] = 's'

            if n['ntype'] == 's':
                n['rootnode'] = None

        longest_path = get_bb_nodes_by_lp(g)
            
    else:
        longest_path = max(candidate_paths, key=lambda x: len(x))

    return longest_path

def get_skel_data_from_shard_dir(this_batch_all_ids, shard_dir):

    all_files_to_load = [x for x in shard_dir.namelist() if x.split('.')[0] in this_batch_all_ids]

    c = [shard_dir.read(x) for x in all_files_to_load]
    c = [x.decode('utf-8') for x in c]
    c = [x.split('\n') for x in c]
    c = [[x.split(' ') for x in a] for a in c]
    c = [tuple([tuple([str(x[0]), str(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), str(x[6])]) for x in a]) for a in c]

    neuron_ids_only = [x.split('.')[0] for x in all_files_to_load]

    skel_data = {}

    for neuron_id, data in list(zip(neuron_ids_only, c)):
        if neuron_id in skel_data:
            skel_data[neuron_id].append(data)
        else:
            skel_data[neuron_id] = [data]
    
    for neuron_id in neuron_ids_only:
        skel_data[neuron_id] = tuple(skel_data[neuron_id])

    return skel_data

def get_cb_nodes_from_nxg(g):

    temp_g = deepcopy(g)

    ax_and_d_and_as_nodes = [x for x in temp_g.nodes() if temp_g.nodes[x]['nodeclass'] in ['0', '1', '2']]
    temp_g.remove_nodes_from(ax_and_d_and_as_nodes)
    con_coms = list(nx.connected_components(temp_g))
    for cc in con_coms:
        if set([temp_g.nodes[x]['nodeclass'] for x in cc]) == {'-1'}:
            temp_g.remove_nodes_from(cc) 

    # Identify soma cc by greatest average edge weight:
    con_coms = list(nx.connected_components(temp_g))
    soma_cc = None
    greatest_mean_edge_rad = 0

    for cc in con_coms:

        for node in cc: # All marked as unclassified until proven otherwise
            g.nodes[node]['nodeclass'] = '-1'

        mean_edge_rad = np.mean([g.edges[x]['radius'] for x in g.edges() if set(x).issubset(cc) and 'radius' in g.edges[x]])

        if mean_edge_rad > greatest_mean_edge_rad:
            greatest_mean_edge_rad = mean_edge_rad
            soma_cc = cc
    
    return soma_cc

def get_cb_nodes_from_igg(g):

    temp_g = deepcopy(g)

    ax_and_d_and_as_nodes = [n.index for n in temp_g.vs if n['nodeclass'] in ['0', '1', '2']]
    temp_g.delete_vertices(ax_and_d_and_as_nodes)

    con_coms = [list(x) for x in list(temp_g.components(mode='WEAK'))]

    unclassfied_nodes_to_delete = []

    for cc in con_coms:
        if set([temp_g.vs[x]['nodeclass'] for x in cc]) == {'-1'}:
            unclassfied_nodes_to_delete.extend(cc)
    
    temp_g.delete_vertices(unclassfied_nodes_to_delete) 

    # Identify soma cc by greatest average edge weight:

    soma_cc = []
    greatest_mean_edge_rad = 0

    for cc in temp_g.components(mode='WEAK'):

        for node in cc: # All marked as unclassified until proven otherwise
            node_name = temp_g.vs[node]['name']
            original_node_idx = [x.index for x in g.vs if x['name']==node_name][0]
            g.vs[original_node_idx]['nodeclass'] = '-1'

        radii = [edge['radius'] for edge in temp_g.es if set([edge.source, edge.target]).issubset(cc)]
        mean_edge_rad = np.mean([x for x in radii if x!=None])

        if mean_edge_rad >= greatest_mean_edge_rad:
            greatest_mean_edge_rad = mean_edge_rad
            soma_cc = cc
    
    soma_cc_names = set([temp_g.vs[n]['name'] for n in soma_cc])

    soma_cc_original_node_idx = set([x.index for x in g.vs if x['name'] in soma_cc_names])

    return soma_cc_original_node_idx

def make_one_skel_graph_ig(list_of_lists, skel_voxel_size, join_separate_components=True, join_by_end_nodes = False):

    g = ig.Graph(directed=False)
    
    all_nodeclasses = [a for b in [[n[1] for n in sublist] for sublist in list_of_lists] for a in b]
    all_node_x = [a for b in [[n[2]*skel_voxel_size[0] for n in sublist] for sublist in list_of_lists] for a in b]
    all_node_y = [a for b in [[n[3]*skel_voxel_size[1] for n in sublist] for sublist in list_of_lists] for a in b]
    all_node_z = [a for b in [[n[4]*skel_voxel_size[2] for n in sublist] for sublist in list_of_lists] for a in b]
    all_ntype = ['c' for x in all_node_z]
    all_node_ids = [a for b in [[str(n[0]) for n in sublist] for sublist in list_of_lists] for a in b]
    
    node_attributes = {
        'x': all_node_x,
        'y': all_node_y,
        'z': all_node_z,
        'ntype': all_ntype,
        'nodeclass': all_nodeclasses,
    }
    
    g.add_vertices(all_node_ids, attributes=node_attributes)

    all_edge_ids = [a for b in [[(str(n[0]), str(n[6])) for n in sublist if str(n[6]) != '-1'] for sublist in list_of_lists] for a in b]
    all_edge_radii = [a for b in [[float(n[5]) for n in sublist if str(n[6]) != '-1'] for sublist in list_of_lists] for a in b]

    g.add_edges(all_edge_ids, attributes={'radius': all_edge_radii})

    if len(list_of_lists) > 1 and join_separate_components==True:

        if join_by_end_nodes == True:
            end_nodes = set([n.index for n in g.vs if g.degree(n.index)==1])
            add_cc_bridging_edges_pairwise(g, joining_nodes=end_nodes)
            #join_ccs_for_one_graph_ig(g)

        else:
            add_cc_bridging_edges_pairwise(g)
            #join_ccs_for_one_graph_ig(g)



    return g

def get_base2agglo(base_segs, base_agglo_map, client):

    final_lookup = {}

    r = get_info_from_bigquery(['agglo_id', 'base_id'], 'base_id', base_segs, base_agglo_map, client)

    for x in r:
        final_lookup[str(x['base_id'])] = str(x['agglo_id'])

    for base_id in base_segs:
        if base_id not in final_lookup.keys():
            final_lookup[base_id] = base_id
    
    return final_lookup

def make_one_skel_graph_nx(list_of_lists, skel_voxel_size, join_components = True, join_by_end_nodes = False):

    g = nx.Graph()

    nodes_to_add = []

    for sublist in list_of_lists:
        nodes_to_add.extend([   [   x[0], 
                                    {   'x': x[2]*skel_voxel_size[0], 
                                        'y': x[3]*skel_voxel_size[1], 
                                        'z': x[4]*skel_voxel_size[2], 
                                        'ntype': 'c',
                                        'nodeclass': x[1]
                                        }
                                        ] 
                                        for x in sublist
                                        ]
                                        )

    g.add_nodes_from(nodes_to_add)

    edges_to_add = []
    for sublist in list_of_lists:
        edges_to_add.extend([[x[0], x[6], {'radius': x[5]}] for x in sublist if x[6] != '-1'])

    g.add_edges_from(edges_to_add)

    if len(list_of_lists) > 1 and join_components == True:

        if join_by_end_nodes == True:
            end_nodes = set([key for (key, value) in g.degree() if value ==1])
            add_cc_bridging_edges_pairwise(g, joining_nodes=end_nodes)
            #join_ccs_for_one_graph_nx(g, joining_nodes=end_nodes)

        else:
            add_cc_bridging_edges_pairwise(g)
            #join_ccs_for_one_graph_nx(g)

    return g

def get_root_node_for_s_node(s_node, g, max_stalk_len, lower_min, higher_min, nodes_or_lengths):

    g.vs[s_node]['rootnode'] = None

    all_end_nodes = [n.index for n in g.vs if g.degree(n.index)==1]

    paths_to_end_nodes = g.get_all_shortest_paths(s_node, to=all_end_nodes, mode='OUT')

    longest_path = max(paths_to_end_nodes, key=lambda x: len(x))

    for pos, root_candidate in enumerate(longest_path):

        if nodes_or_lengths == 'lengths':
            dist_to_this_candidate = get_nm_dist_along_skel_path(g, longest_path[:pos+1])

        if nodes_or_lengths == 'nodes':
            dist_to_this_candidate = pos+1

        if dist_to_this_candidate > max_stalk_len: break

        if g.vs[root_candidate]['ntype'] == 't': # For shared stalks
            continue
        
        # If synapse on the path , exit without assigning a root node:
        if 's' in g.vs[root_candidate]['ntype'] and root_candidate != s_node: 
            return

        previously_encountered = set(longest_path[:longest_path.index(root_candidate)])

        root_neighbours = g.neighbors(root_candidate)

        if len(root_neighbours) < 3 and root_candidate != s_node:
            continue

        first_over_upper = None
        first_over_lower = None

        for r in root_neighbours:
            this_r_paths = g.get_all_shortest_paths(r, to=all_end_nodes, mode='OUT')
            this_r_paths = [p for p in this_r_paths if s_node not in p and root_candidate not in p]
            this_r_paths = [p for p in this_r_paths if len(set(p) & previously_encountered) == 0]

            if len(this_r_paths) == 0:
                continue

            max_path_from_this_neighbour_nodes = max(this_r_paths, key=lambda x: len(x))

            if nodes_or_lengths == 'lengths':
                max_from_this_neighbour = get_nm_dist_along_skel_path(g, max_path_from_this_neighbour_nodes)

            if nodes_or_lengths == 'nodes':
                max_from_this_neighbour = len(max_path_from_this_neighbour_nodes)

            if first_over_upper == None:
                if max_from_this_neighbour > higher_min:
                    first_over_upper = max_from_this_neighbour
                else:
                    if max_from_this_neighbour > lower_min:
                        first_over_lower = max_from_this_neighbour
            else:
                if max_from_this_neighbour > lower_min:
                        first_over_lower = max_from_this_neighbour

            if (first_over_upper != None) and (first_over_lower != None):

                g.vs[s_node]['rootnode'] = root_candidate

                if 's' in g.vs[root_candidate]['ntype']:
                    g.vs[root_candidate]['ntype'] = 'sr'
                else:
                    g.vs[root_candidate]['ntype'] = 'r'
                
                for node in g.get_all_shortest_paths(s_node, to=root_candidate, mode='OUT')[0]:
                    if g.vs[node]['ntype'] == 'c': 
                        g.vs[node]['ntype'] = 't'

                return 
    
        # No going beyond an already-identified root nodes:
        if 'r' in g.vs[root_candidate]['ntype']:
            return

    return 

def get_bb_nodes(min_branch_len, g, nodes_or_lengths):

    all_bb_nodes = set()
    root_nodes = [n.index for n in g.vs if 'r' in n['ntype']]
    stalk_nodes = set([n.index for n in g.vs if n['ntype'] == 't'])

    for source in root_nodes:
        targets = root_nodes[root_nodes.index(source)+1:]
        bb_paths = g.get_all_shortest_paths(source, to=targets, mode='OUT')
        for path in bb_paths:
            all_bb_nodes.update(set(path))

    # If no bb nodes found due to very small segment or one or less root nodes, use longest path as backbone:
    if all_bb_nodes == set():
        all_bb_nodes = set(get_bb_nodes_by_lp(g))

    all_end_nodes = set([n.index for n in g.vs if g.degree(n.index)==1])
    all_neighbours = set([a for b in [g.neighbors(x) for x in all_bb_nodes] for a in b])
    all_neighbours -= stalk_nodes
    rejected_bb_nodes = set()

    while len(all_neighbours) > 0:

        all_neighbours = set([a for b in [g.neighbors(x) for x in all_bb_nodes] for a in b])
        all_neighbours -= stalk_nodes
        all_neighbours -= all_bb_nodes
        all_neighbours -= all_end_nodes
        all_neighbours -= rejected_bb_nodes

        for n in all_neighbours:
 
            candidate_paths = g.get_all_shortest_paths(n, to=all_end_nodes, mode='OUT')
            candidate_paths = [x for x in candidate_paths if len(set(x) & all_bb_nodes)==0]

            if nodes_or_lengths == 'lengths':
                dists = [get_nm_dist_along_skel_path(g, x) for x in candidate_paths]
                
            if nodes_or_lengths == 'nodes':  
                dists = [len(x) for x in candidate_paths]

            dists = list(zip(candidate_paths, dists))
            chosen_path, longest_len = max(dists, key= lambda x: x[1])

            if longest_len > min_branch_len:
                all_bb_nodes.update(set(chosen_path))
            
            for path, dist in dists:
                if dist <= min_branch_len:
                    rejected_bb_nodes.update(set(path))

    return all_bb_nodes

def get_skel_nodes_closest_to_synapses(syn_locations, g, node_list):
    
    if type(g) == ig.Graph:
        node_locs = [[g.vs[x]['x'], g.vs[x]['y'], g.vs[x]['z']] for x in node_list]

    if type(g) == nx.Graph:
        node_locs = [[g.nodes[x]['x'], g.nodes[x]['y'], g.nodes[x]['z']] for x in node_list]

    f = cdist(syn_locations, node_locs, 'euclidean')

    chosen_nodes = [node_list[np.argmin(c)] for c in f]

    return chosen_nodes

def get_neighbourhood_node_type(g, chosen_node, class_lookup):

    axon_dendrite_cutoff = 15
    soma_cutoff = 4

    chosen_node_class = class_lookup[g.vs[chosen_node]['nodeclass']]

    if chosen_node_class == 'soma':

        selected_nodes_classes = get_neighbourhood_node_classes(g, soma_cutoff, chosen_node)

    if chosen_node_class in ('dendrite', 'axon', 'axon initial segment', 'cilium', 'myelinated axon', 'myelinated axon internal fragment'):

        selected_nodes_classes = get_neighbourhood_node_classes(g, axon_dendrite_cutoff, chosen_node)
        
    if chosen_node_class == 'astrocyte':
        return 'astrocyte'

    c = Counter([class_lookup[n] for n in selected_nodes_classes])

    final_type = c.most_common()[0][0]
 
    return final_type

def get_neighbourhood_node_classes(g, path_cutoff, chosen_node):

    if type(g) == ig.Graph:

        selected_nodes = g.neighborhood(vertices=chosen_node, order=path_cutoff)
        selected_nodes_classes = [g.vs[n]['nodeclass'] for n in selected_nodes if g.vs[n]['nodeclass'] != '-1']

    if type(g) == nx.Graph:
        g2 = nx.generators.ego_graph(g, chosen_node, radius=path_cutoff)
        selected_nodes = list(g2.nodes())
        selected_nodes_classes = [g.nodes[n]['nodeclass'] for n in selected_nodes if g.nodes[n]['nodeclass'] != '-1']

    return selected_nodes_classes

def make_skel_graphs_batch(i, segs_to_load, skel_dir, skel_voxel_size, credentials_file, neurite_type, syn_db_name, synapse_voxel_size, syn_location_d=None):

    print('Making skeleton graphs, batch', i)

    skel_graphs = {}

    if len(segs_to_load) == 0:
        return skel_graphs

    # Get synapse locations:
    if syn_location_d == None:
        credentials = service_account.Credentials.from_service_account_file(credentials_file)
        client = bigquery.Client(project=credentials.project_id, credentials=credentials)

        if neurite_type == 'axon':
            bq_key = 'pre_synaptic_site'
            bq_partner_key = 'post_synaptic_partner'

        if neurite_type == 'dendrite':
            bq_key = 'post_synaptic_partner'
            bq_partner_key = 'pre_synaptic_site'

        required_info = [
            f'pre_synaptic_site.centroid.x*{synapse_voxel_size[0]} AS pre_x',
            f'pre_synaptic_site.centroid.y*{synapse_voxel_size[1]} AS pre_y',
            f'pre_synaptic_site.centroid.z*{synapse_voxel_size[2]} AS pre_z',
            f'post_synaptic_partner.centroid.x*{synapse_voxel_size[0]} AS post_x',
            f'post_synaptic_partner.centroid.y*{synapse_voxel_size[1]} AS post_y',
            f'post_synaptic_partner.centroid.z*{synapse_voxel_size[2]} AS post_z',
            'post_synaptic_partner.id AS post_syn_id',
            f'{bq_partner_key}.neuron_id AS partner_seg_id',
            'pre_synaptic_site.id AS pre_syn_id',
            f'{bq_key}.neuron_id AS own_seg_id',
        ]

        results = get_info_from_bigquery(required_info, f'{bq_key}.neuron_id', segs_to_load, syn_db_name, client)

        syn_location_d = {}

        for r in results:

            pre_centroid = (r['pre_x'], r['pre_y'], r['pre_z'])
            post_centroid = (r['post_x'], r['post_y'], r['post_z'])

            own_id = str(r['own_seg_id'])

            if own_id not in syn_location_d:
                syn_location_d[own_id] = {}

            combined_id = f"{r['pre_syn_id']}_{r['post_syn_id']}"

            syn_location_d[own_id][combined_id] = {}

            mean_loc = np.mean([pre_centroid, post_centroid], axis=0)
            
            if np.inf in mean_loc:
                mean_loc = pre_centroid

            syn_location_d[own_id][combined_id]['location'] = mean_loc
            syn_location_d[own_id][combined_id]['partner_id'] = str(r['partner_seg_id']) 


    # Open skeleton shard and get final list of segs to do:

    shard_dir = ZipFile(skel_dir + '/' + str(i) + '.zip', 'r')

    segs_with_syns = set(segs_to_load) & syn_location_d.keys()

    seg_list = list(set([x.split('.')[0] for x in shard_dir.namelist() if x.split('.')[0] in segs_with_syns]))

    raw_skel_dict = get_skel_data_from_shard_dir(seg_list, shard_dir)


    for seg_id in seg_list:

        g = make_one_skel_graph_ig(raw_skel_dict[seg_id], skel_voxel_size)
        
        #print(list(g.vs), i)

        # Assign each synapse in partners to its closest node in the skeleton:
        syn_ids = [x for x in syn_location_d[seg_id].keys()]
        syn_locations = [syn_location_d[seg_id][x]['location'] for x in syn_location_d[seg_id].keys()]
        syn_partners = [syn_location_d[seg_id][x]['partner_id'] for x in syn_location_d[seg_id].keys()]
        
        chosen_nodes = get_skel_nodes_closest_to_synapses(syn_locations, g, [n.index for n in g.vs])

        for node, syn_id, loc, partner in list(zip(chosen_nodes, syn_ids, syn_locations, syn_partners)):

            if g.vs[node]['ntype'] == 's': # If synapse already assigned to this node:
                node_list = [n.index for n in g.vs if n['ntype'] != 's']
                node_locs = [[g.vs[n]['x'], g.vs[n]['y'], g.vs[n]['z']] for n in node_list]

                # In the rare case that a small segment has only no non-synapse-assigned nodes, assumed to be same synapse but split:
                if node_locs == []:
                    print(f'No non-syn nodes to assign syn to, seg node num is {len(list(g.vs))}, with {len(syn_ids)} synapses')
                    time.sleep(30)
                    continue

                chosen_node = node_list[cdist([loc], node_locs, 'euclidean').argmin()]
            
            else:
                chosen_node = node
            
            g.vs[chosen_node]['ntype'] = 's'
            g.vs[chosen_node]['synid'] = syn_id
            g.vs[chosen_node]['truex'] = loc[0]
            g.vs[chosen_node]['truey'] = loc[1]
            g.vs[chosen_node]['truez'] = loc[2]
            g.vs[chosen_node]['partner'] = str(partner)

        skel_graphs[seg_id] = g

    return skel_graphs

def fix_layer_mem(bounds, all_point_data):

    cortical_layers={}
    cortical_layers_coords={}

    # use broadcasting to scale x and y coord data
    seg_ids = all_point_data[:,0]
    xy = all_point_data[:,[1,2]] / np.array([1000,1000])

    # get white matter layer
    mask = xy[:,1] > compute_y(bounds[0], xy[:,0])
    ids = seg_ids[mask]
    coords = xy[mask]
    cluster_name = "White matter"
    cortical_layers[cluster_name] = list(ids)
    cortical_layers_coords[cluster_name] = coords

    # get central layers
    for i in range(len(bounds) - 1):
        mask = np.logical_and(np.logical_or(xy[:,1] < compute_y(bounds[i],xy[:,0]), right_of_circle(bounds[i],xy[:,0])), \
                              xy[:,1] > compute_y(bounds[i+1], xy[:,0]))

        ids = seg_ids[mask]
        coords = xy[mask]
        cluster_name = "Layer " + str(6 - i)
        cortical_layers[cluster_name] = list(ids)
        cortical_layers_coords[cluster_name] = coords

    # get layer 1
    mask = xy[:,1] < compute_y(bounds[-1], xy[:,0])
    ids = seg_ids[mask]
    coords = xy[mask]
    cluster_name = "Layer 1"
    cortical_layers[cluster_name] = list(ids)
    cortical_layers_coords[cluster_name] = coords

    return cortical_layers, cortical_layers_coords

def compute_y(circle, x):

    # (x - xc)^2 + (y - yc)^2 = r^2
    # y = yc + sqrt(r^2 - (x - xc)^2)

    radius = circle["radius"]
    xc, yc = circle["center"]

    return  -1 * np.sqrt(-1 * np.square((x - xc))  + radius**2) + yc

def right_of_circle(circle,x):

    radius = circle["radius"]
    xc, yc = circle["center"]

    return x > radius + xc


def get_info_from_bigquery(info_to_get, info_to_use, info_to_use_ids, db_name, client, batch_size=10000):
        
    results = []
    
    num_batches = int(len(info_to_use_ids)/batch_size)
    
    info_to_get = ','.join([str(x) for x in info_to_get])

    for batch in range(num_batches+1):
        
        if num_batches > 10:
            print('Batch', batch, 'of', num_batches)

        ids = [str(x) for x in info_to_use_ids[batch*batch_size:(batch+1)*batch_size]]

        if ids ==[]: break

        q = ','.join(ids)
        
        query = f"SELECT {info_to_get} FROM {db_name} WHERE SAFE_CAST({info_to_use} AS INT64) IN ({q})"

                
        query_job = client.query(query)  

        results.extend([dict(row) for row in query_job.result()])
        
        if num_batches > 10:
            print((batch+1)*batch_size, 'ids data retrieved')
        
    return results

def assign_root_and_bb_nodes(g, max_stalk_len, lower_min, higher_min, min_branch_len, nodes_or_lengths):

    # Get initial root nodes:
    synapse_nodes = [n.index for n in g.vs if n['ntype'] == 's']

    for s_node in synapse_nodes:
        get_root_node_for_s_node(s_node, g, max_stalk_len, lower_min, higher_min, nodes_or_lengths)

    # Get backbone nodes:

    bb_nodes = get_bb_nodes(min_branch_len, g, nodes_or_lengths)

    for node in bb_nodes:
        if g.vs[node]['ntype'] == 'c':
            g.vs[node]['ntype'] = 'b'

        if g.vs[node]['ntype'] == 's':
            g.vs[node]['ntype'] = 'sb'

    # Then add root nodes based on bb predictions if not previously predicted:
    add_roots_dists_nhr(g, synapse_nodes)

    return g

def get_cg_max_and_cg_avg_vals(random_sample, batch_size, c_min=0):

    ic_max = []
    ic_avg = []
    num_samples = []

    num_batches = len(random_sample)//batch_size


    for batch in range(num_batches+1):

        ids = random_sample[batch*batch_size:(batch+1)*batch_size]

        if ids ==[]: break

        sample_size = (batch+1)*batch_size

        num_samples.append(sample_size)

        if batch == 0:
            ic_max.append(1)
            ic_avg.append(1)
            continue

        this_round_all_cg_scores = []

        this_round_counts = Counter(random_sample[:sample_size])
        this_round_total = sum(this_round_counts.values())
        prev_round_counts = Counter(random_sample[:batch*batch_size])
        prev_round_total = sum(prev_round_counts.values())

        for seg_id in this_round_counts:

            this_round_prop = this_round_counts[seg_id]/this_round_total

            if this_round_prop >= c_min:

                if seg_id in prev_round_counts:
                    prev_round_prop = prev_round_counts[seg_id]/prev_round_total
                else:
                    prev_round_prop = 0

                instantaneous_cg = abs(this_round_prop-prev_round_prop) / ((this_round_prop+prev_round_prop)*0.5)

                this_round_all_cg_scores.append(instantaneous_cg)

        cg_max = max(this_round_all_cg_scores)
        cg_avg = np.mean(this_round_all_cg_scores, axis=0)

        ic_max.append(cg_max)
        ic_avg.append(cg_avg)

    return ic_max, ic_avg, num_samples


def plot_seg_sampling_convergence(random_sample, save_path, batch_size=1, v_line_vals = [0.01, 0.005, 0.001]):

    fig, ax1 = plt.subplots(figsize=(20,10))

    ax1.set_xlabel('Number of sampled points')
    ax1.set_ylabel('Proportion of total', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    for type_in_q in set(random_sample):

        num_samples = []
        proportion = []

        for pos, item in enumerate(random_sample):

            sample_size = pos+1

            all_items_so_far = random_sample[:sample_size]

            type_in_q_count = all_items_so_far.count(type_in_q)

            prop = type_in_q_count/sample_size

            proportion.append(prop)

            num_samples.append(sample_size)

        line_col = choice(['red', 'green', 'black'])

        ax1.plot(num_samples, proportion, color=line_col)

    # Then plot CG max:
    ic_max, ic_avg, num_samples = get_cg_max_and_cg_avg_vals(random_sample, batch_size)

    min_ic_up_to_that_point = [min(ic_max[:pos+1]) for pos, x in enumerate(ic_max)]

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Instantaneous convergence rate', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(num_samples, min_ic_up_to_that_point, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')


    vline_x_coords = []
    for v_line_val in v_line_vals:
        for s_num, min_ic in zip(num_samples, min_ic_up_to_that_point):
            if min_ic <= v_line_val:
                vline_x_coords.append(s_num)
                break

    ax1.vlines(vline_x_coords, 0, 1, colors=['blue'])

    temp = ', '.join([str(x) for x in v_line_vals])
    ax1.set_title(f'Sample number vs. Sampled segment proportions and CGmax with CGmax values of {temp} marked')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close('all')






def add_roots_dists_nhr(g, synapse_nodes):

    for s_node in synapse_nodes:

        root_node = g.vs[s_node]['rootnode']

        if root_node == None:

            all_bb_nodes = set([n.index for n in g.vs if n['ntype'] in ['b', 'r', 'sr', 'sb']])
            all_paths_to_bb = g.get_all_shortest_paths(s_node, to=all_bb_nodes, mode='OUT')
            all_paths_to_bb = [x for x in all_paths_to_bb if len(set(x) & all_bb_nodes)==1]
            stalk_distances = [get_nm_dist_along_skel_path(g, x) for x in all_paths_to_bb]
            chosen_path = all_paths_to_bb[np.argmin(stalk_distances)]
            root_node = list(set(chosen_path) & all_bb_nodes)[0]
            g.vs[s_node]['rootnode'] = root_node

            if 's' in g.vs[root_node]['ntype']:
                g.vs[root_node]['ntype'] = 'sr'
            else:
                g.vs[root_node]['ntype'] = 'r'

            for node in chosen_path:
                if g.vs[node]['ntype'] == 'c': 
                    g.vs[node]['ntype'] = 't'

        # Then get stalkdist:

        sp = g.get_all_shortest_paths(s_node, to=root_node, mode='OUT')[0]
        g.vs[s_node]['stalkdist'] = get_nm_dist_along_skel_path(g, sp)

        # Then NHR:

        if len(sp) == 1:
            g.vs[s_node]['nhr'] = 0.0
        else:
            radii = [g.es[g.get_eid(sp[n],sp[n+1])]['radius'] for n in range(len(sp)-1)]
            radii = [x for x in radii if type(x) == float]
            if len(radii) < 2:
                g.vs[s_node]['nhr'] = 0.0
            else: 
                g.vs[s_node]['nhr'] = float(np.mean(radii[:2])/np.mean(radii[-2:]))

        # Then euclidean distances:
        s_coords = [g.vs[s_node]['truex'], g.vs[s_node]['truey'], g.vs[s_node]['truez']]
        g.vs[s_node]['eucdistshaft'] = get_euc_dist_to_edges_ig(g, s_coords, root_node) 

        root_coords = [g.vs[root_node]['x'], g.vs[root_node]['y'], g.vs[root_node]['z']]
        g.vs[s_node]['eucdistroot'] = float(euclidean(s_coords, root_coords))

def classify_syn_remove_stalks(g, clf):

    syn_nodes = [n.index for n in g.vs if 's' in n['ntype']]

    for syn_node in syn_nodes:
        
        prediction = int(clf.predict([[g.vs[syn_node]['stalkdist'], g.vs[syn_node]['nhr']]])[0])

        if prediction == 1:

            g.vs[syn_node]['stype'] = 'stalk'

        if prediction == 0:

            if g.vs[syn_node]['rootnode'] == syn_node:
                assert g.vs[syn_node]['ntype'] == 'sr'
                g.vs[syn_node]['stype'] = 'shaft'

            else:
                # Reassign to closest backbone node within 30 of root that is not already associated with a synapse:
                #assert g.vs[syn_node]['ntype'] in ['s', 'sb']
                    
                root_node = g.vs[syn_node]['rootnode']
                bb_nodes_to_consider = set([a for b in g.get_all_simple_paths(root_node, mode='OUT', cutoff=30) for a in b])
                bb_node_candidates = [x for x in bb_nodes_to_consider if g.vs[x]['ntype'] in ['b', 'r']]

                # If no non-syn bb nodes within 30 nodes (e.g. small segs), assume same synapse and allow overwrite:
                if bb_node_candidates == []:
                    bb_node_candidates = [x for x in bb_nodes_to_consider if g.vs[x]['ntype']]

                syn_location = [g.vs[syn_node]['truex'], g.vs[syn_node]['truey'], g.vs[syn_node]['truez']]
                new_node = get_skel_nodes_closest_to_synapses([syn_location], g, bb_node_candidates)[0]

                for k in ['synid', 'truex', 'truey', 'truez', 'rootnode', 'eucdistshaft', 'eucdistroot', 'partner']:
                    g.vs[new_node][k] = deepcopy(g.vs[syn_node][k])
                
                if g.vs[new_node]['ntype'] == 'b':
                    g.vs[new_node]['ntype'] = 'sb'

                if g.vs[new_node]['ntype'] == 'r':
                    g.vs[new_node]['ntype'] = 'sr'

                for k in ['synid', 'truex', 'truey', 'truez', 'rootnode', 'eucdistshaft', 'eucdistroot', 'stalkdist', 'nhr', 'partner']:
                    g.vs[syn_node][k] = None

                g.vs[syn_node]['ntype'] = 'c'

                for node in g.get_all_shortest_paths(syn_node, to=root_node, mode='OUT')[0]:
                    if g.vs[node]['ntype'] == 't':
                        g.vs[node]['ntype'] = 'c'

                g.vs[new_node]['stalkdist'] = 0
                g.vs[new_node]['stype'] = 'shaft'

    return g

def get_euc_dist_to_edges_ig(g, syn_coords, root_node, cutoff=30):

    bb_nodes_to_consider = g.get_all_simple_paths(root_node, mode='OUT', cutoff=cutoff)

    edge_list = []

    for l in bb_nodes_to_consider:
        accepted_edges = [
                [l[n], l[n+1]] for n in range(len(l)-1) 
                if set([g.vs[l[n]]['ntype'], g.vs[l[n+1]]['ntype']]).issubset(set(['b', 'r', 'sr', 'sb']))
                ]

        edge_list.extend(accepted_edges)

    backbone_coord_pairs = [
        [   np.array([g.vs[x[0]]['x'],g.vs[x[0]]['y'],g.vs[x[0]]['z']]),
            np.array([g.vs[x[1]]['x'],g.vs[x[1]]['y'],g.vs[x[1]]['z']])       ]
        
            for x in edge_list      ]
         
    x0 = np.array(syn_coords)
                                        
    min_dist = min([pnt2line(x0, x1, x2) for x1, x2 in backbone_coord_pairs])

    return min_dist

def join_ccs_for_one_graph_ig(g):
    start = time.time()

    conn_coms = [tuple(x) for x in list(g.components(mode='WEAK'))]

    # end_nodes = set([n.index for n in g.vs if g.degree(n.index)==1])

    # conn_coms = [set(x) for x in list(g.components(mode='WEAK'))]
    # conn_coms = [tuple(x-end_nodes) for x in conn_coms]

    while len(conn_coms) > 1:

        closest_pairs_this_round = []

        start_cc_n = len(conn_coms)

        print(start_cc_n, 'connected components in skeleton graph, adding another edge')

        for cc1 in conn_coms:

            pos1 = conn_coms.index(cc1)

            cc1_locs = [[g.vs[n]['x'], g.vs[n]['y'], g.vs[n]['z']] for n in cc1]

            for cc2 in conn_coms[pos1+1:]:

                cc2_locs = [[g.vs[n]['x'], g.vs[n]['y'], g.vs[n]['z']] for n in cc2]

                temp_dists = cdist(cc1_locs, cc2_locs, 'euclidean')

                min_dist_this_pair = np.min(temp_dists)

                min_pos_coords = np.argwhere(temp_dists == np.min(temp_dists))

                cc1_node = cc1[min_pos_coords[0][0]]
                cc2_node = cc2[min_pos_coords[0][1]]

                closest_pairs_this_round.append([cc1_node, cc2_node, min_dist_this_pair])

        selected_pair = min(closest_pairs_this_round, key=lambda x: x[2])

        g.add_edge(selected_pair[0], selected_pair[1])

        print('... added an edge between nodes', selected_pair[0], 'and', selected_pair[1], selected_pair[2], 'nm apart')
        conn_coms = [tuple(x) for x in list(g.components(mode='WEAK'))]
        # conn_coms = [set(x) for x in list(g.components(mode='WEAK'))]
        # conn_coms = [tuple(x-end_nodes) for x in conn_coms]

        assert len(conn_coms) == start_cc_n-1

    print(f'... total time to join was {time.time()-start} seconds')

def join_ccs_for_one_graph_nx(skel_g, joining_nodes=None):

    count = 0

    while nx.number_connected_components(skel_g) > 1:
        count +=1

        conn_coms = [tuple(x) for x in list(nx.connected_components(skel_g))]

        closest_pairs_this_round = []

        start_cc_n = len(conn_coms)

        print(start_cc_n, 'connected components in skeleton graph, adding another edge')

        for cc1 in conn_coms:
            
            pos1 = conn_coms.index(cc1)

            cc1_selected = cc1

            if joining_nodes!=None:
                cc1_ends = set(cc1) & joining_nodes
                if cc1_ends != set():
                    cc1_selected = cc1_ends

            cc1_locs = [[skel_g.nodes[n]['x'], skel_g.nodes[n]['y'], skel_g.nodes[n]['z']] for n in cc1_selected]

            for cc2 in conn_coms[pos1+1:]:

                cc2_selected = cc2

                if joining_nodes!=None:
                    cc2_ends = set(cc2) & joining_nodes
                    if cc2_ends != set():
                        cc2_selected = cc2_ends

                cc2_locs = [[skel_g.nodes[n]['x'], skel_g.nodes[n]['y'], skel_g.nodes[n]['z']] for n in cc2_selected]

                temp_dists = cdist(cc1_locs, cc2_locs, 'euclidean')

                min_dist_this_pair = np.min(temp_dists)

                min_pos_coords = np.argwhere(temp_dists == np.min(temp_dists))

                cc1_node = cc1[min_pos_coords[0][0]]
                cc2_node = cc2[min_pos_coords[0][1]]

                closest_pairs_this_round.append([cc1_node, cc2_node, min_dist_this_pair])

        closest_pairs_this_round = sorted(closest_pairs_this_round, key=lambda x: x[2])

        selected_pair = closest_pairs_this_round[0]

        skel_g.add_edge(selected_pair[0], selected_pair[1])

        print('... added an edge between nodes', selected_pair[0], 'and', selected_pair[1], selected_pair[2], 'nm apart')

        assert nx.number_connected_components(skel_g) == start_cc_n-1

def add_cc_bridging_edges_pairwise(skel_g, joining_nodes=None):

    if type(skel_g) == nx.Graph:
        con_comms = list(nx.connected_components(skel_g))

    if type(skel_g) == ig.Graph:
        con_comms = list(skel_g.components(mode='WEAK'))
        
    if len(con_comms) == 1: return

    candidate_edges = []

    for cc1, cc2 in combinations(con_comms, 2):

        if joining_nodes==None:
            cc1_list = list(cc1)
            cc2_list = list(cc2)
        else:
            cc1_list = [x for x in cc1 if x in joining_nodes]
            cc2_list = [x for x in cc2 if x in joining_nodes]
    
        if cc1_list == [] or cc2_list == []:
            continue

        if type(skel_g) == nx.Graph:
            cc1_node_locs = [[skel_g.nodes[n]['x'], skel_g.nodes[n]['y'], skel_g.nodes[n]['z']] for n in cc1_list]
            cc2_node_locs = [[skel_g.nodes[n]['x'], skel_g.nodes[n]['y'], skel_g.nodes[n]['z']] for n in cc2_list]

        if type(skel_g) == ig.Graph:
            cc1_node_locs = [[skel_g.vs[n]['x'], skel_g.vs[n]['y'], skel_g.vs[n]['z']] for n in cc1_list]
            cc2_node_locs = [[skel_g.vs[n]['x'], skel_g.vs[n]['y'], skel_g.vs[n]['z']] for n in cc2_list]

        f = cdist(cc1_node_locs, cc2_node_locs, 'euclidean')

        min_indices = np.unravel_index(np.argmin(f, axis=None), f.shape)

        sel_cc1 = cc1_list[min_indices[0]]
        sel_cc2 = cc2_list[min_indices[1]]
        dist = int(f[min_indices])  

        candidate_edges.append([sel_cc1, sel_cc2, dist])

    if candidate_edges == []: 
        return
    
    candidate_edges = sorted(candidate_edges, key = lambda x: x[2])
    
    consecutive_bridging_failure = 0
    
    for pos, candidate_edge in enumerate(candidate_edges):
        
        origin, target, dist = candidate_edge #min(candidate_edges, key = lambda x: x[2])
        
        if type(skel_g) == nx.Graph:
            if not nx.has_path(skel_g, origin, target):
                skel_g.add_edge(origin, target)
                #print(f'Added an edge between nodes {origin} and {target}, {dist} nm apart')
                consecutive_bridging_failure = 0
            else:
                consecutive_bridging_failure += 1

        if type(skel_g) == ig.Graph:
            paths_from_origin_to_target = skel_g.get_all_shortest_paths(origin, to=target, mode='OUT')


            if paths_from_origin_to_target == []:
                skel_g.add_edge(origin, target)
                #print(f'Added an edge between nodes {origin} and {target}, {dist} nm apart')
                consecutive_bridging_failure = 0
            else:
                consecutive_bridging_failure += 1

        if consecutive_bridging_failure == 3:
            add_cc_bridging_edges_pairwise(skel_g, joining_nodes=None)
            
        if pos%20 == 0:
            if type(skel_g) == nx.Graph:
                n_cc = nx.number_connected_components(skel_g)

            if type(skel_g) == ig.Graph:
                n_cc = len(list(skel_g.components(mode='WEAK')))
    
            print(f'{n_cc} connected components remain')
              
            if n_cc == 1:
                return
    
    if type(skel_g) == nx.Graph:
        n_cc = nx.number_connected_components(skel_g)

    if type(skel_g) == ig.Graph:
        n_cc = len(list(skel_g.components(mode='WEAK')))

    if n_cc > 1:
        add_cc_bridging_edges_pairwise(skel_g, joining_nodes=None)
    else:  
        assert n_cc == 1
  
def get_partners_and_distances_to_soma_one_dir(
        seg_ids, shard_dir, dtype, soma_target_type,
        max_syn_to_node_dist_nm, synapse_voxel_size, 
        skel_voxel_size, syn_db_name, credentials_file, 
        save_dir, class_lookup ):

    if dtype == 'pre':
        self_structure = 'axon'
        partner_side = 'post_synaptic_partner'
        own_side = 'pre_synaptic_site'

    if dtype == 'post':
        self_structure = 'dendrite,soma,axon initial segment'
        partner_side = 'pre_synaptic_site'
        own_side = 'post_synaptic_partner'

    complete_agglo_ids = set([x.split('_')[0] for x in os.listdir(save_dir) if self_structure in x])
    seg_ids = [str(x) for x in seg_ids]

    if set(seg_ids).issubset(complete_agglo_ids): return

    if not os.path.exists(shard_dir): return

    shard = ZipFile(shard_dir, 'r')

    # Get all outgoing/incoming synapes of these cells:
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
        
    required_info = [
        f'{partner_side}.neuron_id AS partner_agglo_id',
        f'{partner_side}.id AS partner_syn_id',
        f'{own_side}.neuron_id AS pr_cell_agglo_id',
        f'{own_side}.id AS pr_cell_syn_id',
        f'{own_side}.centroid.x*{synapse_voxel_size[0]} AS x',
        f'{own_side}.centroid.y*{synapse_voxel_size[1]} AS y',
        f'{own_side}.centroid.z*{synapse_voxel_size[2]} AS z',
        'type',

    ]

    results = get_info_from_bigquery(required_info, f'{own_side}.neuron_id', list(seg_ids), syn_db_name, client)

    organised_synapses = {x: [a for a in results if str(a['pr_cell_agglo_id'])==x] for x in seg_ids}

    skel_data = get_skel_data_from_shard_dir(seg_ids, shard)

    for pos, agglo_id in enumerate(seg_ids):

        final_results = []

        if f'{agglo_id}_partner_distances_along_{self_structure}.json' in os.listdir(save_dir): continue

        print(agglo_id, pos+1, len(seg_ids), shard_dir.split('/')[-1].split('.')[0])

        if agglo_id not in skel_data:
            print(agglo_id, 'not in skel!')
            continue

        g = make_one_skel_graph_ig(skel_data[agglo_id], skel_voxel_size, join_by_end_nodes = True)

        soma_nodes = list(get_cb_nodes_from_igg(g))
        
        if soma_nodes == []: continue
        
        class_nodes = [n.index for n in g.vs if n['nodeclass']!='-1']

        syn_locations = [[x['x'], x['y'], x['z']] for x in organised_synapses[agglo_id]]
        syn_types = [x['type'] for x in organised_synapses[agglo_id]]

        if syn_locations == []: continue

        closest_syn_nodes = get_skel_nodes_closest_to_synapses(syn_locations, g, class_nodes)

        syn_node_to_syn_type = {syn_node: syn_type for syn_node, syn_type in zip(closest_syn_nodes, syn_types)}

        selected_types = [get_neighbourhood_node_type(g, 15, node, class_lookup) for node in closest_syn_nodes]
        
        if soma_target_type == 'centre':
            
            soma_com = [np.mean([g.vs[n][a] for n in soma_nodes]) for a in ['x', 'y', 'z']]
            
            chosen_soma_node = get_skel_nodes_closest_to_synapses([soma_com], g, soma_nodes)[0]
            
            chosen_soma_nodes = [chosen_soma_node for x in closest_syn_nodes]

            shortest_paths = g.get_all_shortest_paths(chosen_soma_node, to=closest_syn_nodes, mode='OUT')

            sp_dists = [get_nm_dist_along_skel_path(g, skel_path) for skel_path in shortest_paths]
        
        
        if soma_target_type == 'surface':
            
            shortest_paths = []
            sp_dists = []
            chosen_soma_nodes = []
            
            for syn_node in closest_syn_nodes:
                
                sps = g.get_all_shortest_paths(syn_node, to=soma_nodes, mode='OUT')

                min_dist, min_dist_path  = min([(get_nm_dist_along_skel_path(g, sp), sp) for sp in sps], key= lambda x: x[0])
                
                chosen_soma_node = min_dist_path[-1]
                
                shortest_paths.append(min_dist_path)
                sp_dists.append(min_dist)
                chosen_soma_nodes.append(chosen_soma_node)

        
        zipped = zip(chosen_soma_nodes, sp_dists, selected_types, closest_syn_nodes, syn_locations, organised_synapses[agglo_id])

        for chosen_soma_node, min_dist, sel_type, syn_node, syn_loc, r in zipped:

            if sel_type not in self_structure: continue
            
            soma_node_loc = (g.vs[chosen_soma_node]['x'], g.vs[chosen_soma_node]['y'], g.vs[chosen_soma_node]['z'])

            distance_syn_to_syn_node = euclidean(syn_loc, [g.vs[syn_node]['x'], g.vs[syn_node]['y'], g.vs[syn_node]['z']])

            if distance_syn_to_syn_node <= max_syn_to_node_dist_nm:
                r['pr_cell_syn_type'] = sel_type
            else:
                r['pr_cell_syn_type'] = 'unknown'
                continue

            dists_to_other_syn_nodes = [len(x) for x in g.get_all_shortest_paths(syn_node, to=closest_syn_nodes, mode='OUT')]

            syn_nodes_and_dists = list(zip(closest_syn_nodes, dists_to_other_syn_nodes))

            syn_nodes_and_dists = sorted(syn_nodes_and_dists, key = lambda x: x[1])

            print(syn_nodes_and_dists)

            k = 10

            closest_k_types = [syn_node_to_syn_type[x[0]] for x in syn_nodes_and_dists[:k]]

            i_neighbour_count = closest_k_types.count(1)
            e_neighbour_count = closest_k_types.count(2)
            
            pr_cell_syn_node_loc = (g.vs[syn_node]['x'], g.vs[syn_node]['y'], g.vs[syn_node]['z'])

            final_result_this_syn = {
                'partner_agglo_id': str(r['partner_agglo_id']),
                'partner_syn_id': str(r['partner_syn_id']),
                'pr_cell_agglo_id': str(r['pr_cell_agglo_id']),
                'pr_cell_syn_id': str(r['pr_cell_syn_id']),
                'pr_cell_syn_loc': tuple(syn_loc),
                'pr_cell_syn_node_loc': tuple(pr_cell_syn_node_loc), 
                'pr_cell_syn_to_syn_node_dist': int(distance_syn_to_syn_node),
                'pr_cell_syn_type': sel_type,
                'pr_cell_soma_dist': int(min_dist),
                'pr_cell_soma_node_loc': tuple(soma_node_loc),
                'synapse_type': int(r['type']),
                'i_neighbour_count': i_neighbour_count,
                'e_neighbour_count': e_neighbour_count,
                'num_neighbours_used_for_ei_count': k,

            }

            final_results.append(final_result_this_syn)

        with open(f'{save_dir}/{agglo_id}_partner_distances_along_{self_structure}_to_{soma_target_type}.json', 'w') as fp:
            json.dump(final_results, fp)

def Bresenham3D(x1, y1, z1, x2, y2, z2): 
    ListOfPoints = [] 
    ListOfPoints.append((x1, y1, z1)) 
    dx = abs(x2 - x1) 
    dy = abs(y2 - y1) 
    dz = abs(z2 - z1) 
    if (x2 > x1): 
        xs = 1
    else: 
        xs = -1
    if (y2 > y1): 
        ys = 1
    else: 
        ys = -1
    if (z2 > z1): 
        zs = 1
    else: 
        zs = -1
  
    # Driving axis is X-axis" 
    if (dx >= dy and dx >= dz):         
        p1 = 2 * dy - dx 
        p2 = 2 * dz - dx 
        while (x1 != x2): 
            x1 += xs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dx 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dx 
            p1 += 2 * dy 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Y-axis" 
    elif (dy >= dx and dy >= dz):        
        p1 = 2 * dx - dy 
        p2 = 2 * dz - dy 
        while (y1 != y2): 
            y1 += ys 
            if (p1 >= 0): 
                x1 += xs 
                p1 -= 2 * dy 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dy 
            p1 += 2 * dx 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Z-axis" 
    else:         
        p1 = 2 * dy - dz 
        p2 = 2 * dx - dz 
        while (z1 != z2): 
            z1 += zs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dz 
            if (p2 >= 0): 
                x1 += xs 
                p2 -= 2 * dz 
            p1 += 2 * dy 
            p2 += 2 * dx 
            ListOfPoints.append((x1, y1, z1)) 
    return ListOfPoints 
  
def get_nm_dist_along_skel_path(g, sp):

    if len(sp) == 1:
        return 0

    if type(g) == ig.Graph:
        all_locations = [[g.vs[n][a] for a in ['x', 'y', 'z']] for n in sp]

    if type(g) == nx.Graph:
        all_locations = [[g.nodes[n][a] for a in ['x', 'y', 'z']] for n in sp]

    f = cdist(all_locations, all_locations, 'euclidean')
    
    all_dists = [dist_array[pos+1] for pos, dist_array in enumerate(f) if pos != len(dist_array)-1]
    
    return sum(all_dists)
  
  

'''
def get_nm_dist_along_skel_path_old(skel_g, sp):

    dist = 0

    for n1 in sp:
        if n1 != sp[-1]:
            n2 = sp[sp.index(n1)+1]

            if type(skel_g) == nx.Graph:
                n1_coord = [skel_g.nodes[n1]['x'], skel_g.nodes[n1]['y'], skel_g.nodes[n1]['z']]
                n2_coord = [skel_g.nodes[n2]['x'], skel_g.nodes[n2]['y'], skel_g.nodes[n2]['z']]

            if type(skel_g) == ig.Graph:
                n1_coord = [skel_g.vs[n1]['x'], skel_g.vs[n1]['y'], skel_g.vs[n1]['z']]
                n2_coord = [skel_g.vs[n2]['x'], skel_g.vs[n2]['y'], skel_g.vs[n2]['z']]

            dist += euclidean(n1_coord, n2_coord)
    
    return dist
'''


def lineseg_dist(p, a, b):

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))

def pnt2line(pnt, start, end):
    line_vec = end-start
    pnt_vec = pnt-start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec/line_len
    pnt_vec_scaled = pnt_vec*(1.0/line_len)
    t = np.dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec*t
    dist = np.linalg.norm(pnt_vec-nearest)
    return dist


    # Given a line with coordinates 'start' and 'end' and the
    # coordinates of a point 'pnt' the proc returns the shortest 
    # distance from pnt to the line and the coordinates of the 
    # nearest point on the line.
    #
    # 1  Convert the line segment to a vector ('line_vec').
    # 2  Create a vector connecting start to pnt ('pnt_vec').
    # 3  Find the length of the line vector ('line_len').
    # 4  Convert line_vec to a unit vector ('line_unitvec').
    # 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
    # 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
    # 7  Ensure t is in the range 0 to 1.
    # 8  Use t to get the nearest location on the line to the end
    #    of vector pnt_vec_scaled ('nearest').
    # 9  Calculate the distance from nearest to pnt_vec_scaled.
    # 10 Translate nearest back to the start/end line. 
    # Malcolm Kesson 16 Dec 2012

def get_combined_synapses_for_a_seg(seg_id, seg_syn_data, syn_merge_d):

    combined_syn_data = {}

    for p in seg_syn_data.keys():

        combined_syn_data[p] = {}

        syn_ids = ['_'.join([x[2], x[3]]) for x in seg_syn_data[p]]

        final_set_of_synapses = set()

        for x in syn_ids:
            if x in syn_merge_d:
                final_set_of_synapses.add(tuple(syn_merge_d[x]))
            else:
                final_set_of_synapses.add(tuple([x]))

        for combined_id in final_set_of_synapses:

            all_centroids = []

            for s in seg_syn_data[p]:
                syn_id = '_'.join([s[2], s[3]])
                if syn_id in combined_id:
                    pre_post_centroid = np.mean([s[0], s[1]], axis=0)
                    all_centroids.append(pre_post_centroid)
            
            mean_centroid = np.mean(all_centroids, axis=0)
            mean_centroid = tuple([int(x) for x in mean_centroid])

            combined_id = list(combined_id)
            combined_id.sort()
            combined_id = '-'.join(combined_id)
            combined_syn_data[p][combined_id] = mean_centroid

    return combined_syn_data


# Functions to retreive complete saturated data in blocks:

def get_full_vol(seg_id, agglo_seg_info_db, bbox_voxel_size):

    cylinder_radius = None
    bb_coords = None
    increment = 100

    pruned_skel_g = get_pruned_skeleton(seg_id, 'legnth')[0]

    cylinder_radius = get_longest_bouton_stalk(pruned_skel_g)

    bbox = get_seg_bb(seg_id, agglo_seg_info_db, credentials_file, google_project_id)

    # Get start position in 1nm co-ordinates:
    bbox_start = [bbox[0]['bbox']['start']['x']*bbox_voxel_size[0], 
                bbox[0]['bbox']['start']['y']*bbox_voxel_size[1], 
                bbox[0]['bbox']['start']['z']*bbox_voxel_size[2]]

    bbox_start = [max(int(x-cylinder_radius), 0) for x in bbox_start] 

    # Get start position in 32*32*30nm co-ordinates:

    bbox_start = [  int(bbox_start[0]/agglo_seg_voxel_size[0]),
                    int(bbox_start[1]/agglo_seg_voxel_size[1]),
                    int(bbox_start[2]/agglo_seg_voxel_size[2])
    ]

    # Get box size in 1nm co-ordinates:
    bbox_size = [bbox[0]['bbox']['size']['x']*bbox_voxel_size[0], 
                bbox[0]['bbox']['size']['y']*bbox_voxel_size[1], 
                bbox[0]['bbox']['size']['z']*bbox_voxel_size[2]]

    bbox_size = [int(x+(2*cylinder_radius)) for x in bbox_size]

    # Get box_size in 32*32*30nm co-ordinates:
    bbox_size = [   int(bbox_size[0]/agglo_seg_voxel_size[0]),
                    int(bbox_size[1]/agglo_seg_voxel_size[1]),
                    int(bbox_size[2]/agglo_seg_voxel_size[2])
    ]

    # Round up to nearest 100:

    bbox_size = [int(math.ceil(x / float(increment))) * increment for x in bbox_size]

    pool = Pool(4)
    print(list(range(bbox_start[2], min(bbox_start[2]+bbox_size[2], agglo_seg_upper_bounds[2]), increment)))
    block_list_multi = pool.starmap(
        get_blocks_for_a_z, 
        zip(    range(bbox_start[2], 
                min(bbox_start[2]+bbox_size[2], agglo_seg_upper_bounds[2]), increment),
                repeat(bbox_start), 
                repeat(bbox_size), 
                repeat(increment), 
                repeat(interface_seg),
                repeat('one_step'),
                repeat(seg_id),
                repeat(cylinder_radius),
                repeat(bb_coords)
                )
                                    )

    multi_test = np.block(block_list_multi)

    return multi_test

def get_sv(volume_id, corner, size, volume_datatype, scoped_credentials, agglo_seg = None):


  headers = {
      'Authorization': 'Bearer ' + scoped_credentials.token,
      'Content-type': 'application/json',
      'Accept-Encoding': 'gzip',
  }
  
    #   print((f'Requesting data for volume {volume_id} beginning at corner {corner}, '
    #         f'size={size}'))
  
  if agglo_seg == None:
    request = {
        'geometry': {
            'scale': 0,
            'corner': ','.join(str(x) for x in corner),
            'size': ','.join(str(x) for x in size),
        },
        'subvolumeFormat': 'RAW_SNAPPY'
        }
  
  else:

    request = {
        'geometry': {
            'scale': 0,
            'corner': ','.join(str(x) for x in corner),
            'size': ','.join(str(x) for x in size),
        },
        'subvolumeFormat': 'RAW_SNAPPY',
        "changeSpec": {"changeStackId": agglo_seg
    }}


    #   print('Request body: ' + json.dumps(request))
    #   print('Making request...')

  url = f'https://brainmaps.googleapis.com/v1/volumes/{volume_id}/subvolume:binary'
  req = urllib.request.Request(url, json.dumps(request).encode('utf-8'), headers)
  resp = urllib.request.urlopen(req)
  http_response = resp.read()
  subvolume_snappy_compressed = zlib.decompress(http_response, 16 + zlib.MAX_WBITS)
  data = snappy.decompress(subvolume_snappy_compressed)

  # Note that the returned data is in CZYX format, so a voxel in a single channel 
  # uint8 volume at location X=23, Y=1, Z=10 would be array[10, 1, 23].
  array = np.frombuffer(data, dtype=volume_datatype).reshape(size)
  
  return array

def get_seg_bb(seg_id, db_name, credentials_file, google_project_id):

    credentials = service_account.Credentials.from_service_account_file(credentials_file)

    client = bigquery.Client(project=google_project_id, credentials=credentials)

    query = f"""SELECT bbox FROM {db_name} WHERE id = {seg_id}"""
                
    query_job = client.query(query)  

    results = [dict(row) for row in query_job.result()]

    return results
                    
def get_blocks_for_a_z(z, start_coords, request_size, i, base_seg, mode, seg_id, cylinder_radius, bb_coords):
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    scopes = ['https://www.googleapis.com/auth/brainmaps']
    scoped_credentials = credentials.with_scopes(scopes)
    scoped_credentials.refresh(auth_request.Request())
    
    this_z_list = []

    this_z_dict = {}

    for y in range(start_coords[1], start_coords[1]+request_size[1], i):

        this_y_list = []

        for x in range(start_coords[0], start_coords[0]+request_size[0], i):

            if mode == 'direct_count':

                corners = [     [x,y,z], [x,y,z+i], [x,y+i,z+i], 
                                [x+i,y+i,z], [x+i,y+i,z+i],
                                [x,y+i,z], [x+i,y,z+i], [x+i,y,z]
                            ]

                corners = [[    x[0]*agglo_seg_voxel_size[0],
                                x[1]*agglo_seg_voxel_size[1],
                                x[2]*agglo_seg_voxel_size[2]    ]
                            for x in corners
                                ]

                dists = euclidean_distances(corners, bb_coords)

                shortest_dist = min([min(x) for x in dists])

                if shortest_dist <= cylinder_radius:

                    block = get_sv(base_seg, [x, y, z], [i,i,i], np.uint64, scoped_credentials)

                    values, counts = np.unique(block, return_counts=True)
                    this_block_results = dict(zip(values, counts))
                    for k in this_block_results.keys():
                        if int(k) in this_z_dict:
                            this_z_dict[int(k)] += this_block_results[k]
                        else:
                            this_z_dict[int(k)] = this_block_results[k]

            else:
                block = get_sv(base_seg, [x, y, z], [i,i,i], np.uint64, scoped_credentials)

                if mode == 'two_step':
                    np.save(f'{results_dir}/{seg_id}_tmp/{x}_{y}_{z}.npy', block)

                if mode == 'one_step':
                    this_y_list.append(block)
        
        if mode == 'one_step':
            this_z_list.append(this_y_list)
    
    if mode == 'one_step':
        return this_z_list
    
    if mode == 'direct_count':

        for k in this_z_dict.keys():
            this_z_dict[k] = int(this_z_dict[k])
        
        return this_z_dict
        #json.dump(this_z_dict, open(f'{results_dir}/{seg_id}_tmp/z_{z}_results_dict.json', 'w'))


