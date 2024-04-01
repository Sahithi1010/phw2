import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, linkage_type='ward', n_clusters=2):
    X, _ = dataset
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit AgglomerativeClustering estimator
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    hierarchical_cluster.fit(X_scaled)

    return hierarchical_cluster.labels_

def fit_modified(dataset):
    X, _ = dataset
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate linkage matrix
    Z = linkage(X_scaled, method='ward')

    # Find the maximum rate of change of the distance between successive cluster merges
    diff = np.diff(Z[:, 2], 2)
    max_diff_idx = np.argmax(diff)

    # Use the maximum rate of change as the cut-off distance
    cut_off_distance = Z[max_diff_idx, 2]

    # Perform hierarchical clustering with cut-off distance
    hierarchical_cluster = AgglomerativeClustering(distance_threshold=cut_off_distance, linkage='ward')
    hierarchical_cluster.fit(X_scaled)

    return hierarchical_cluster.labels_

def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    noisy_circles = make_circles(n_samples=100, factor=0.5, noise=0.05)
    noisy_moons = make_moons(n_samples=100, noise=0.05)
    blobs_varied = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    aniso = (np.dot(blobs_varied[0], [[0.6, -0.6], [-0.4, 0.8]]), blobs_varied[1])
    blobs = make_blobs(n_samples=100, random_state=8)

    datasets = {'nc': noisy_circles, 'nm': noisy_moons, 'bvv': blobs_varied, 'add': aniso, 'b': blobs}

    dct = answers["1A: datasets"] = {'nc': [noisy_circles[0],noisy_circles[1]],
                                     'nm': [noisy_moons[0],noisy_moons[1]],
                                     'bvv': [blobs_varied[0],blobs_varied[1]],
                                     'add': [aniso[0],aniso[1]],
                                     'b': [blobs[0],blobs[1]]}
    

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = [""]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
