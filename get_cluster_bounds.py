import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import hdbscan
from collections import Counter
import math
from sklearn.svm import SVC
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
import pickle 
from scipy.spatial import ConvexHull, Delaunay


# -------------------- #
# ----GLOBAL VARS----- #
# -------------------- #


top_dir = 'C:/work/FINAL'
cell_data = 'C:/work/FINAL/agglo_20201123b_cell_data.json'


use_only_neurons = True
use_z = False # original code has this set to false
max_minpts = 50
upper_edge = (500, 2250), (3500, 1800)
lower_edge = (500, 1600), (2750, 500)


# -------------------- #
# --HELPER FUNCTIONS-- #
# -------------------- #

class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

def fit_central_sops_RANSAC(cluster, sigma_coef):
    
    
    x = np.array([i[0] for i in cluster])
    y = np.array([i[1] for i in cluster])
    
    ransac = RANSACRegressor(PolynomialRegression(degree=2),
             residual_threshold=2 * np.std(y) / sigma_coef,
             random_state=0)

    ransac.fit(np.expand_dims(x, axis=1), y)
    
    inlier_mask = ransac.inlier_mask_
    
    inlier_cluster = []
    for i in range(len(x)):
        if inlier_mask[i] == True:
            inlier_cluster.append((x[i], y[i]))
    
    num_inliers = inlier_mask.sum()
    print(f"num outliers = {len(x) - num_inliers}")
    print(f"percent outliers = {((len(x) - num_inliers) / len(x)) * 100}")

    estimator = ransac.estimator_
    
    sop = np.poly1d(estimator.get_params()["coeffs"])
    
    x_min = min(x)
    x_max = max(x)
    sop_x = np.linspace(x_min, x_max, int(x_max - x_min))
    
    """
    plt.scatter(x, y, s = 7) 
    plt.plot(sop_x, sop(sop_x), '-', color = 'r') 
    """
    
    
    return [sop, inlier_cluster]

def fit_alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
            
    return edges


def shift_central_sop(cluster, sop, alpha):

    def compute_cost(points, sop):

        cost = 0

        for coord in points:
            x = coord[0]
            y = coord[1]
            cost += (y - sop(x)) ** 2

        return cost

    def shift_sop(points, sop, direction):

        def func(x, a, b, c):
            return a + b * x + c * x ** 2

        popt_cons, _ = curve_fit(func, points[:,0], points[:,1],
                                 bounds=([sop[0], -np.inf, -np.inf], [sop[0]+0.001, np.inf, np.inf]))

        popt_cons = list(popt_cons)
        popt_cons.reverse()

        bound = np.poly1d(popt_cons)

        return bound

    concave_hull = fit_alpha_shape(cluster, alpha=alpha, only_outer=True)

    """
    # -----PLOTTING CONCAVE HULL----- #
    plt.plot(cluster[:, 0], cluster[:, 1], '.')
    for i, j in concave_hull:
        plt.plot(cluster[[i, j], 0], cluster[[i, j], 1])
    plt.show()
    """

    # -----EXTRACTING UPPER AND LOWER BOUND POINTS----- #
    points_idxs = set()
    for i, j in concave_hull:
        points_idxs.add(i)
        points_idxs.add(j)

    hull = cluster[list(points_idxs)]
    """
    plt.scatter(hull[:,0], hull[:,1])
    plt.show()
    """

    upper_hull = []
    lower_hull = []

    for coord in hull:
        if coord[1] > sop(coord[0]):
            upper_hull.append(list(coord))
        else:
            lower_hull.append(list(coord))

    upper_hull = np.array(upper_hull)
    lower_hull = np.array(lower_hull)

    upper = shift_sop(upper_hull, sop, "up")
    lower = shift_sop(lower_hull, sop, "down")

    x_vals = np.linspace(np.min(cluster[:,0]), np.max(cluster[:,0]), 2000)
    
    """
    plt.scatter(cluster[:,0], cluster[:,1], s = 7)
    plt.plot(x_vals, lower(x_vals), color = 'r')
    plt.plot(x_vals, upper(x_vals), color = 'b')
    plt.show()
    """

    return upper, lower


# -------------------- #
# -----MAIN FUNC------ #
# -------------------- #

def main():


    # -----GET NEURONS----- #
    with open(cell_data, 'r') as fp:
        raw_data = json.load(fp)

    if use_only_neurons:
        raw_data = [x for x in raw_data if 'neuron' in x['type']]

    raw_data = [x for x in raw_data if x['soma_cubic_um'] != 'None']
    raw_data = [x for x in raw_data if x['soma_cubic_um'] > 0]

    cb_x = [x['true_x']/1000 for x in raw_data] # all x y z volume data here
    cb_y = [x['true_y']/1000 for x in raw_data]
    cb_z = [x['true_z']/1000 for x in raw_data]
    cb_som = [x['soma_cubic_um'] for x in raw_data]

    cm = plt.cm.get_cmap('RdYlBu')
    plt.plot([lower_edge[0][0], lower_edge[1][0]], [lower_edge[0][1], lower_edge[1][1]],  'k-')
    plt.plot([upper_edge[0][0], upper_edge[1][0]], [upper_edge[0][1], upper_edge[1][1]],  'k-')
    sc = plt.scatter(cb_x, cb_y, c=[math.log(x) for x in cb_som], s=7, cmap=cm)
    plt.colorbar(sc)
    plt.savefig(f'{top_dir}/cell_bodies_with_soma_size.png')
    plt.clf()

    if use_z:
        cell_input_data = list(zip(cb_x, cb_y, cb_z, cb_som))
    else:
        cell_input_data = list(zip(cb_x, cb_y, cb_som))


    ## DENSITY ESIMTAION

    # Do k-fold = 5, testing distances between 1 and 200:
    grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=0), {'bandwidth': range(1,201,1)}, cv=KFold(n_splits=5), n_jobs=-1)
    grid.fit(cell_input_data)
    pd.DataFrame(grid.cv_results_).to_csv(f'{top_dir}/cortical_layer_id_5_fold_gaussian_cv_results.csv')

    # Create the estimator using the best bandwidth:
    chosen_bandwidth = grid.best_params_['bandwidth']
    #chosen_bandwidth = 81
    estimator = KernelDensity(bandwidth=chosen_bandwidth, kernel='gaussian')
    estimator.fit(cell_input_data)


    # -----GET MIN_PTS (k) THAT GIVES AVE MUTUAL REACHABILITY DISTANCE (TO K NNs) THAT BEST CORRELATES WITH DENSITY----- #
    probs = np.exp(estimator.score_samples(cell_input_data))

    mr_dists = {k: {} for k in range(1,max_minpts+1)}

    for k in range(2,max_minpts+1):
        print(k)
        all_cell_mean_mr_dists = []

        nbrs = NearestNeighbors(n_neighbors=k).fit(cell_input_data)
        distances, indices = nbrs.kneighbors(cell_input_data)

        for curr_distances, curr_partners in zip(distances, indices):

            mutual_reachability_distances = []

            for i, p in enumerate(list(curr_partners)):

                dist_to_this_p = curr_distances[i]
                dk_of_this_p = distances[p][-1]
                dk_of_object_in_question = curr_distances[-1]
                mrd = np.max([dist_to_this_p, dk_of_this_p, dk_of_object_in_question])
                mutual_reachability_distances.append(mrd)

            mean_mrd = np.mean(mutual_reachability_distances)
            all_cell_mean_mr_dists.append(mean_mrd)

        mr_dists[k]['values'] = all_cell_mean_mr_dists
        mr_dists[k]['cc'] = abs(np.corrcoef([mr_dists[k]['values'], probs])[1,0])

    del mr_dists[1]
    min_pts = max([(k, mr_dists[k]['cc']) for k in mr_dists], key=lambda x: x[1])[0]
 

    # -----FIND CLUSTERS----- #
    #Min_cluster_size should be increased until >=3 clusters spanning cortex were obtained:
    cluster_labels_and_persistence = {}

    for mcs in range(50,100, 1):

        print(f"testing mcs: {mcs}")

        cluster_labels_and_persistence[mcs] = {}

        clusterer = hdbscan.HDBSCAN(min_samples = min_pts, min_cluster_size = mcs, gen_min_span_tree=True)
        clusterer.fit(cell_input_data)
        labels = [int(x) for x in clusterer.labels_] # poke here
        cluster_persistence = [float(x) for x in clusterer.cluster_persistence_]

        cluster_labels_and_persistence[mcs]['cluster_labels'] = labels
        cluster_labels_and_persistence[mcs]['cluster_persistence'] = cluster_persistence

        c = [x for x in zip(labels, cb_x, cb_y) if x[0] != -1]

        accepted_labels = []

        # cells in no cluster have label -1
        # cells in other clusters
        # cells in clusters that did not meet requirements - leaving volume for example

        for label in set(labels):
            if label != -1:

                all_xy = [np.array(x[1:]) for x in c if x[0]==label]

                a = np.array(upper_edge[0])
                b = np.array(upper_edge[1])
                above_upper_points = [p for p in all_xy if np.cross(p-a, b-a)<0]

                a = np.array(lower_edge[0])
                b = np.array(lower_edge[1])
                below_lower_points = [p for p in all_xy if np.cross(p-a, b-a)>0]

                if len(above_upper_points)>0 and len(below_lower_points)>0:
                    accepted_labels.append(label)

        labels_nz = [x[0] for x in c if x[0] in accepted_labels]

        cb_x_class = [x[1] for x in c if x[0] in accepted_labels]
        cb_y_class = [x[2] for x in c if x[0] in accepted_labels]

        """
        plt.plot([lower_edge[0][0], lower_edge[1][0]], [lower_edge[0][1], lower_edge[1][1]],  'k-')
        plt.plot([upper_edge[0][0], upper_edge[1][0]], [upper_edge[0][1], upper_edge[1][1]],  'k-')
        plt.scatter(cb_x_class, cb_y_class, c=labels_nz, s=7)

        plt.xlabel('x co-ordinate (microns)')
        plt.ylabel('y co-ordinate (microns)')
        plt.title(f'Min Cluster Size: {mcs}, Number of clusters:{len(accepted_labels)}')
        #plt.savefig(f'{top_dir}/cell_clustering_k{min_pts}_mcs{mcs}.png')
        #plt.clf()
        plt.show()
        """

        if len(accepted_labels) >= 2:
            print(f"Final mcs = {mcs}")
            break

        """
        clusterer.condensed_tree_.plot()
        plt.savefig(f'{top_dir}/cell_clustering_k11_mcs{mcs}_tree.png')
        plt.clf()
        """

    # with open(f'{top_dir}/cluster_labels_and_persistence.json', 'w') as fp:
    #     json.dump(cluster_labels_and_persistence, fp)

    cluster_1 = []
    cluster_2 = []
    cluster_3 = []

    for i in range(len(cb_x_class)):
        if labels_nz[i] == accepted_labels[0]:
            cluster_1.append([cb_x_class[i], cb_y_class[i]])
        elif labels_nz[i] == accepted_labels[1]:
            cluster_2.append([cb_x_class[i], cb_y_class[i]])
        elif labels_nz[i] == accepted_labels[2]:
            cluster_3.append([cb_x_class[i], cb_y_class[i]])
        else:
            raise ValueError("Error cluster key not recognised")


    # -----FIT BOUNDS----- #
    upper_bounds = []
    lower_bounds = []
    inlier_clusters = [] 

    clusters = [cluster_1, cluster_2, cluster_3]

    # cluster 1 min coeff to form convex shape is 2.5115
    [sop, inlier_cluster] = fit_central_sops_RANSAC(cluster_1, 2.5115) 
    np_cluster = np.array(inlier_cluster) 
    upper, lower = shift_central_sop(np_cluster, sop, 80)

    upper_bounds.append(upper)
    lower_bounds.append(lower)
    
    # for cluster 2 and 3, no points need to be removed by RANSAC to get
    # convex shape so low value of coef chosen (0.01) that results in 
    # no outliers beoing thrown out
    [sop, inlier_cluster] = fit_central_sops_RANSAC(cluster_2, 0.001)
    np_cluster = np.array(inlier_cluster) 
    upper, lower = shift_central_sop(np_cluster, sop, 30)

    upper_bounds.append(upper)
    lower_bounds.append(lower)

    clustors = [cluster_1, cluster_2, cluster_3]
    [sop, inlier_cluster] = fit_central_sops_RANSAC(cluster_3, 0.01)

    np_cluster = np.array(inlier_cluster)
    upper, lower = shift_central_sop(np_cluster, sop, 30)

    upper_bounds.append(upper)
    lower_bounds.append(lower)

    bounds = {"upper":upper_bounds, "lower":lower_bounds}

    with open(f"{top_dir}/cortical_bounds.pkl", "wb") as f:
        pickle.dump(bounds, f)

    # plot resulting graph
    for cluster, upper, lower in zip(clusters, upper_bounds, lower_bounds):

        cluster = np.array(cluster)
        min_x = np.min(cluster[:,0])
        max_x = np.max(cluster[:,0])

        x_vals = np.linspace(min_x, max_x, 2000)

        plt.scatter(cluster[:,0], cluster[:,1], s = 7)
        plt.plot(x_vals, upper(x_vals))
        plt.plot(x_vals, lower(x_vals))
        
    plt.show()


if __name__ == "__main__":
    main()
