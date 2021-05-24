import json
import matplotlib.pyplot as plt
from math import log
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
from distfit import distfit
import pickle
import common_functions as cf

raw_dist_file = 'C:/work/FINAL/syn_dist_distributions/distance_measurements_multisyn_pure_001_axons_with_two_euc_dists.json'
save_location = 'C:/work/FINAL/syn_dist_distributions/'

r_skew_and_common = [
    'alpha', 'gamma', 'burr', 
    'beta', 'lognorm', 'pareto', 'dweibull', 'weibull_min', 'chi', 'chi2', 
    'weibull_max', 'lognorm', 'frechet_r', 'frechet_l', 'cauchy', 
    'logistic', 'norm', 'uniform', 'lognorm', 'pareto', 'expon',
    ]


if __name__ == '__main__':

    with open(raw_dist_file, 'r') as fp:
        raw_data = json.load(fp)


    all_euc_dists = {
        'stalk': {'to shaft': [], 'to root': []}, 
        'shaft': {'to shaft': [], 'to root': []}
    }

    for x in raw_data:
        for y in x[1]:
            all_euc_dists[y[2]]['to shaft'].append(y[0])
            all_euc_dists[y[2]]['to root'].append(y[1])


    # Get appropriate bin sizes first, by cross-validation:

    for dist_type in ['to shaft', 'to root']:

        for dtype in ['shaft', 'stalk']:

            if dist_type == 'to root' and dtype == 'shaft': continue

            grid = GridSearchCV(KernelDensity(kernel='tophat', rtol=0), {'bandwidth': range(0,400, 10)}, cv=KFold(n_splits=5), n_jobs=-1)
            input_data = all_euc_dists[dtype][dist_type]
            input_array = np.array(input_data)
            input_array = input_array.reshape(-1, 1)
            grid.fit(input_array) 
            bandwidth = grid.best_estimator_.bandwidth
            num_bins = int(max(input_data)/bandwidth)
            dist = distfit(bins=num_bins, distr=r_skew_and_common)
            dist.fit_transform(np.array(input_data))

            dist.model['num_bins'] = num_bins

            dist_name = dist.model['distr'].name
            rss = str(dist.model['RSS'])

            filename = f'Best fitting distribution for distance from {dtype}-type synapses {dist_type}.png'
            x_axis = f'Distance {dist_type} (nm)'
            y_axis = 'Probability density'
            title = f'Best fitting distribution for distance from {dtype}-type synapses {dist_type}: {dist_name} (RSS: {rss})'

            cf.plot_empirical_vs_theoretical_dist(dist.model, input_data, save_location, filename, x_axis, y_axis, title) 

            with open(f'{save_location}/{dtype}_euclidean_distance_{dist_type}_distribution_model.pkl', 'wb') as fp:
                pickle.dump(dist.model, fp)





