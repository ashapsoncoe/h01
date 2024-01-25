import os
import sys

# stops warning messaged appearing
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sqrt, linspace, cos, sin
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import NearestNeighbors
import hdbscan
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial import Delaunay
from scipy import optimize
from math import pi

# Authors - Luke James Bailey, Alex Shapson-Coe

# -------------------- #
# ----GLOBAL VARS----- #
# -------------------- #

citation_print = """
The following sources were used in the making of this code. To view where each source was used,
citations are included in comments above the relavant code.

[1] - Stack Overflow. 2019. Iteratively fitting polynomial curve.
      [online] Available at: <https://stackoverflow.com/questions/55682156/iteratively-fitting-polynomial-curve>
      [Accessed 8 January 2021].
      answer author - Jirka B.

[2] - Stack Overflow. 2018. Calculate bounding polygon of alpha shape from the Delaunay triangulation.
      [online] Available at: <https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation>
      [Accessed 20 January 2021].
      answer author - Iddo Hanniel

[3] - SciPy Cookbook. 2011. Least squares circle.
      [online] Available at: <https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html>
      [Accessed 16 May 2021].

"""

cell_data = "agglo_20200916c3_cell_data.json"

use_only_neurons = True
use_z = False  # original code has this set to false
max_minpts = 50
upper_edge = (500, 2250), (3500, 1800)
lower_edge = (500, 1600), (2750, 500)


# -------------------- #
# --HELPER FUNCTIONS-- #
# -------------------- #

# Class used for RANSACRegressor
#
# citation - Stack Overflow. 2019. Iteratively fitting polynomial curve.
#            [online] Available at: <https://stackoverflow.com/questions/55682156/iteratively-fitting-polynomial-curve>
#            [Accessed 8 January 2021].
#            answer author - Jirka B.
class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {"coeffs": self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


# helper function to fit single second order polynomial with RANSAC
# param: x - x coords of points
# param: y - y coords of points
# param: sigma_coef - coefficent for residual_threshold of RANSAC
#
# citation - Stack Overflow. 2019. Iteratively fitting polynomial curve.
#            [online] Available at: <https://stackoverflow.com/questions/55682156/iteratively-fitting-polynomial-curve>
#            [Accessed 8 January 2021].
#            answer author - Jirka B.
def fit_single_RANSAC(x, y, sigma_coef):

    # check sigma_coef = 0
    ransac = RANSACRegressor(
        PolynomialRegression(degree=2),
        residual_threshold=2 * np.std(y) / sigma_coef,
        random_state=0, min_samples=10
    )

    ransac.fit(np.expand_dims(x, axis=1), y)
    estimator = ransac.estimator_
    sop = np.poly1d(estimator.get_params()["coeffs"])

    return ransac, sop


# finds the RANSAC regression curve with the minimum
# redidual threshold to result in a convex curve
# param: x - x coords of points
# param: y - y coords of points
# param: max_iters - maximum number of iterations search
#                    for optimal sigma_coef can take
def find_valid_RANSAC(x, y, max_iters=1000):

    # check sigma_coef = 0
    ransac, sop = fit_single_RANSAC(x, y, 0.000001)

    if sop[2] > 0:
        print(f"final sigma_coef ~ 0")
        return ransac

    lower_sb = 0
    upper_sb = 5
    iters = 0
    while True:
        sigma_coef = (lower_sb + upper_sb) / 2
        print(f"testing sigma_coef {sigma_coef}")

        ransac_1, sop_1 = fit_single_RANSAC(x, y, sigma_coef)
        ransac_2, sop_2 = fit_single_RANSAC(x, y, sigma_coef - 0.0001)

        if sop_1[2] > 0 and sop_2[2] <= 0:
            print(f"final sigma_coef = {sigma_coef}")
            return ransac_1
        elif sop_1[2] > 0:
            upper_sb = sigma_coef
        elif sop_1[2] < 0:
            lower_sb = sigma_coef
        else:
            print("ERROR - x^2 coef = 0")

        iters += 1
        if iters > max_iters:
            raise RuntimeError("Finding valid ransac exceeded max iters")


# Fits a second order polynnomial through the center
# of a cluster
# param: sigma_coef - coefficient for the redisual threshold of
#                     RANSAC regression
def fit_central_sops_RANSAC(cluster):

    x = np.array([i[0] for i in cluster])
    y = np.array([i[1] for i in cluster])

    ransac = find_valid_RANSAC(x, y)

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

    return [sop, inlier_cluster]


# Compute the alpha shape (concave hull) of a set of points.
# param: points - np.array of shape (n,2) points.
# param: alpha - alpha hyperparameter.
# param: only_outer - boolean value to specify if we keep only the outer border
# or also inner edges.
#
# citation - Stack Overflow. 2018. Calculate bounding polygon of alpha shape from the Delaunay triangulation.
#            [online] Available at: <https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation>
#            [Accessed 20 January 2021].
#            answer author - Iddo Hanniel
def fit_alpha_shape(points, alpha, only_outer=True):
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


# Use a concave hull and central second order polynomial
# to find the set of points that define the top and
# bottom of each cluster
# param: cluster - cluster of points
# param: sop - poly1d polynomial
# param: alpha - concave hull alpha hyperparameter
def get_upper_and_lower_bounds(cluster, sop, alpha):

    concave_hull = fit_alpha_shape(cluster, alpha=alpha, only_outer=True)

    # -----EXTRACTING UPPER AND LOWER BOUND POINTS----- #
    points_idxs = set()
    for i, j in concave_hull:
        points_idxs.add(i)
        points_idxs.add(j)

    hull = cluster[list(points_idxs)]

    upper_hull = []
    lower_hull = []

    for coord in hull:
        if coord[1] > sop(coord[0]):
            upper_hull.append(list(coord))
        else:
            lower_hull.append(list(coord))

    upper_hull = np.array(upper_hull)
    lower_hull = np.array(lower_hull)

    return upper_hull, lower_hull


# fit a circle to a set of points
# points - set of points to fit a circle to
# fix_center - if not none, circle will be fit with a center point
#              equal to this argument
#
# source - https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
# citation - SciPy Cookbook. 2011. Least squares circle.
#            [online] Available at: <https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html>
#            [Accessed 16 May 2021].
def circle_fit(points, fix_center=None):

    x_m = points[:, 0].mean()
    y_m = points[:, 1].mean()

    x = points[:, 0]
    y = points[:, 1]

    def calc_R(xc, yc):
        return sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    if fix_center is None:
        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(f, center_estimate)

        xc, yc = center
        Ri = calc_R(*center)
        R = Ri.mean()
    else:
        center = fix_center
        xc, yc = fix_center
        Ri = calc_R(*fix_center)
        R = Ri.mean()

    return {"center": center, "radius": R}


# helper function to plot all circles (bounds)
# param: upper_circles - list of the upper bounding circles
# param: lower_circles - list of the lower bounding circles
# param: inlier_custers - list of inlier cluster points
def plot_all_data_circles(upper_circles, lower_circles, inlier_custers, title):
    def plot_circle(center, radius, theta_1, theta_2):
        theta_vals = linspace(theta_1, theta_2, 500)
        x = radius * cos(theta_vals) + center[0]
        y = radius * sin(theta_vals) + center[1]
        plt.plot(x, y, "b-", lw=2)

    plt.rcParams["figure.figsize"] = (15, 10)

    for inliers, upper, lower in zip(inlier_custers, upper_circles, lower_circles):

        theta_1, theta_2 = -pi / 2, 0

        # plot upper
        plot_circle(upper["center"], upper["radius"], theta_1, theta_2)

        # plot lower
        plot_circle(lower["center"], lower["radius"], theta_1, theta_2)

        # plot cluster
        plt.scatter(inliers[:, 0], inliers[:, 1], s=1)

    plt.xlabel(r"x position ($\mu m$)")
    plt.ylabel(r"y position ($\mu m$)")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(title)


# -------------------- #
# -----MAIN FUNC------ #
# -------------------- #


def main():

    working_dir = os.path.dirname(__file__)
    sys.path.insert(0, working_dir)
    os.chdir(working_dir)

    # -----GET NEURONS----- #
    with open(cell_data, "r") as fp:
        raw_data = json.load(fp)

    working_dir = f'{working_dir}/cortical_layers_output'

    os.mkdir(working_dir)
    os.chdir(working_dir)

    if use_only_neurons:
        raw_data = [x for x in raw_data if "neuron" in x["type"]]

    raw_data = [x for x in raw_data if x["soma_cubic_um"] != "None"]
    raw_data = [x for x in raw_data if x["soma_cubic_um"] > 0]

    cb_x = [x["true_x"] / 1000 for x in raw_data]  # all x y z volume data here
    cb_y = [x["true_y"] / 1000 for x in raw_data]
    cb_z = [x["true_z"] / 1000 for x in raw_data]
    cb_som = [x["soma_cubic_um"] for x in raw_data]

    if use_z:
        cell_input_data = list(zip(cb_x, cb_y, cb_z, cb_som))
    else:
        cell_input_data = list(zip(cb_x, cb_y, cb_som))

    ## DENSITY ESIMTAION

    # Do k-fold = 5, testing distances between 1 and 200:
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian", rtol=0),
        {"bandwidth": range(1, 201, 1)},
        cv=KFold(n_splits=5),
        n_jobs=-1,
    )
    grid.fit(cell_input_data)
    pd.DataFrame(grid.cv_results_).to_csv(
        f"{working_dir}/cortical_layer_id_5_fold_gaussian_cv_results.csv"
    )

    # Create the estimator using the best bandwidth:
    chosen_bandwidth = grid.best_params_["bandwidth"]
    # chosen_bandwidth = 81
    estimator = KernelDensity(bandwidth=chosen_bandwidth, kernel="gaussian")
    estimator.fit(cell_input_data)

    # -----GET MIN_PTS (k) THAT GIVES AVE MUTUAL REACHABILITY DISTANCE (TO K NNs) THAT BEST CORRELATES WITH DENSITY----- #
    probs = np.exp(estimator.score_samples(cell_input_data))

    mr_dists = {k: {} for k in range(1, max_minpts + 1)}

    for k in range(2, max_minpts + 1):

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

        mr_dists[k]["values"] = all_cell_mean_mr_dists
        mr_dists[k]["cc"] = abs(np.corrcoef([mr_dists[k]["values"], probs])[1, 0])

    del mr_dists[1]
    min_pts = max([(k, mr_dists[k]["cc"]) for k in mr_dists], key=lambda x: x[1])[0]

    # -----FIND CLUSTERS----- #
    # Min_cluster_size should be increased until >=3 clusters spanning cortex were obtained:

    dist_metric = 'manhattan'
    cluster_labels_and_persistence = {}

    for mcs in range(50, 60, 1):

        print(f"testing mcs: {mcs}")

        cluster_labels_and_persistence[mcs] = {}

        clusterer = hdbscan.HDBSCAN(
            min_samples=min_pts, min_cluster_size=mcs, gen_min_span_tree=True, metric=dist_metric
        )
        clusterer.fit(cell_input_data)

        outlier_scores = [float(x) for x in clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)]]

        labels = [int(x) for x in clusterer.labels_]  # poke here
        cluster_persistence = [float(x) for x in clusterer.cluster_persistence_]

        cluster_labels_and_persistence[mcs]["cluster_persistence"] = cluster_persistence
        cluster_labels_and_persistence[mcs]["outlier_scores"] = outlier_scores

        cluster_labels_and_persistence[mcs]["cluster_labels"] = labels

        c = [x for x in zip(labels, cb_x, cb_y) if x[0] != -1]

        accepted_labels = []

        # cells in no cluster have label -1
        # cells in other clusters
        # cells in clusters that did not meet requirements - leaving volume for example

        for label in set(labels):
            if label != -1:

                all_xy = [np.array(x[1:]) for x in c if x[0] == label]

                a = np.array(upper_edge[0])
                b = np.array(upper_edge[1])
                above_upper_points = [p for p in all_xy if np.cross(p - a, b - a) < 0]

                a = np.array(lower_edge[0])
                b = np.array(lower_edge[1])
                below_lower_points = [p for p in all_xy if np.cross(p - a, b - a) > 0]

                if len(above_upper_points) > 0 and len(below_lower_points) > 0:
                    accepted_labels.append(label)

        labels_nz = [x[0] for x in c if x[0] in accepted_labels]

        cb_x_class = [x[1] for x in c if x[0] in accepted_labels]
        cb_y_class = [x[2] for x in c if x[0] in accepted_labels]

        if len(accepted_labels) >= 3:
            print(f"Final mcs = {mcs}, {len(accepted_labels)} clusters meeting criteria")
            break

    with open(f"{working_dir}/cluster_labels_and_persistence.json", "w") as fp:
        json.dump(cluster_labels_and_persistence, fp)

    clusters = []
    accepted_labels_dict = {}
    for i in range(len(accepted_labels)):
        clusters.append([])
        accepted_labels_dict[accepted_labels[i]] = i

    for i in range(len(cb_x_class)):
        if labels_nz[i] not in accepted_labels_dict:
            raise ValueError("Error cluster key not recognised")
        else:
            clusters[accepted_labels_dict[labels_nz[i]]].append(
                [cb_x_class[i], cb_y_class[i]]
            )

    # -----FIT BOUNDS----- #
    
    upper_bounds = []
    lower_bounds = []
    inlier_clusters = []

    clusters = sorted(clusters, key = lambda x: np.mean([a[0] for a in x]))

    for i, cluster in enumerate(clusters):
        
        alpha = 120

        [sop, inlier_cluster] = fit_central_sops_RANSAC(cluster)
        np_cluster = np.array(inlier_cluster)
        inlier_clusters.append(np_cluster)
        upper, lower = get_upper_and_lower_bounds(np_cluster, sop, alpha)
        center_circle = circle_fit(np_cluster)
        upper_bounds.append(
            circle_fit(np.array(upper), fix_center=center_circle["center"])
        )
        lower_bounds.append(
            circle_fit(np.array(lower), fix_center=center_circle["center"])
        )


    bounds = []
    for upper, lower in zip(upper_bounds, lower_bounds):
        upper["center"] = list(upper["center"])
        lower["center"] = list(lower["center"])
        bounds.append(upper)
        bounds.append(lower)

    with open(f"{working_dir}/cortical_bounds_circles.json", "w") as f:
        json.dump(bounds, f)

    plot_all_data_circles(upper_bounds, lower_bounds, inlier_clusters, f"cortical_bounds_circles_plot.png")



if __name__ == "__main__":
    print(citation_print)
    main()
