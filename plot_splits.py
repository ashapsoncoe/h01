import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


proofread_cells_dir = 'proofread_axons_and_dendrites_for_split_plotting_z'
z_thickness = 33
plot_bandwidth = 100

if __name__ == '__main__':

    proofread_cells = {x: {} for x in ('axon','dendrite')}

    for f in os.listdir(proofread_cells_dir):

        with open(f'{proofread_cells_dir}/{f}', 'r') as fp:
            cell_data = json.load(fp)

        seg_id = f.split('_')[2]

        seg_type = max([(k, len(cell_data['base_segments'][k])) for k in cell_data['base_segments'] if k != 'unknown'], key=lambda x: x[1])[0]

        proofread_cells[seg_type][seg_id] = cell_data

    edges_with_locations = {}
    all_split_point_annos = {}

    for seg_type in ('axon', 'dendrite'):

        edges_with_locations[seg_type] = {}
        all_split_point_annos[seg_type] = {}

        for seg_id in proofread_cells[seg_type]:

            cell_data = proofread_cells[seg_type][seg_id]

            # Get all edges between base segs in the final proofread object:

            all_edges = set([tuple(sorted(x)) for x in cell_data['graph_edges']])
            added_edges = set([tuple(sorted(x[:2])) for x in cell_data['added_graph_edges']])

            assert added_edges.issubset(all_edges)

            pre_existing_edges = all_edges - added_edges

            edges_with_locations[seg_type][seg_id] = {}

            for dataset, dkey in ((pre_existing_edges, 'pre_existing_edges'), (added_edges, 'added_edges')):
                edges_with_locations[seg_type][seg_id][dkey] = [[x[0], x[1], tuple(np.mean([cell_data['base_locations'][a] for a in (x[0], x[1])], axis=0, dtype=int))] for x in dataset]

            # Get all the added point annotations where the cause of the annotation wasn't a natural end or exiting the volume:
            all_split_point_annos[seg_type][seg_id] = [a for b in [cell_data['end_points'][k] for k in cell_data['end_points'] if k not in ('exit volume', 'natural')] for a in b]


    # Plot split errors in Z:
    for dtype in ('axon', 'dendrite'):

        z_locs_splits = [int(a/z_thickness) for b in [[x[2][2] for x in edges_with_locations[dtype][seg_id]['added_edges']] for seg_id in proofread_cells[dtype]] for a in b]

        point_annotations_z_coords = [a for b in [[int(x[2]/z_thickness) for x in all_split_point_annos[dtype][seg_id]] for seg_id in all_split_point_annos[dtype]] for a in b]

        z_locs_splits.extend(point_annotations_z_coords)

        upper_lim_plot = int(max(z_locs_splits)*1.25)
        xs = [[x] for x in np.linspace(0, upper_lim_plot,int(upper_lim_plot/plot_bandwidth))]
        density = KernelDensity(kernel='gaussian', bandwidth=plot_bandwidth).fit([[x] for x in z_locs_splits])
        log_dens = density.score_samples(xs)
        plt.plot(xs,np.exp(log_dens))
        plt.scatter(z_locs_splits, [0 for x in z_locs_splits], s=1.5, c='red')
        plt.xlabel('Z layer')
        plt.ylabel('Probability density')
        plt.title(f'Probability density of {dtype} splits across Z-layers, bandwidth = {plot_bandwidth}, {len(z_locs_splits)} splits total')
        plt.savefig(f'{dtype}_splits_in_z.png')
