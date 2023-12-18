#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find clusters of pointcloud

Author: FILL IN
MatrNr: FILL IN
"""

from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import distance
from scipy.stats import anderson
import matplotlib.pyplot as plt
from helper_functions import plot_clustering_results, silhouette_score

def kmeans(points: np.ndarray,
           n_clusters: int,
           n_iterations: int,
           max_singlerun_iterations: int,
           centers_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the k-means algorithm.

    :param points: The (down-sampled) points of the pointcloud to be clustered.
    :type points: np.ndarray with shape=(n_points, 3)

    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :type n_clusters: int

    :param n_iterations: Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :param centers_in: Start centers of the k-means algorithm.  If centers_in = None, the centers are randomly sampled
        from input data for each iteration.
    :type centers_in: np.ndarray with shape = (n_clusters, 3) or None

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ###################################################### 
    # Initialize cluster centers
    if centers_in is None:
        random_indices = np.random.choice(len(points), n_clusters, replace=False)
        centers = np.mean(points[random_indices], axis=0, keepdims=True)
    else:
        centers = centers_in

    # Assign points to the nearest cluster center
    labels = None
    best_inertia = np.inf


    for _ in range(n_iterations):
        labels_last = np.zeros(len(points))
        iterations = 0

        while iterations < max_singlerun_iterations:
            iterations += 1

            # Compute distances from data point to center
            # print(points)
          #  print(centers)
            distances = np.linalg.norm(points - centers[:, np.newaxis], axis=2)

            # Assign points to the nearest cluster center
            labels = np.argmin(distances, axis=0)

            # Update cluster centers
            centers = np.array([np.mean(points[labels == i], axis=0) for i in range(n_clusters)])

            # Check convergence
            if np.array_equal(labels, labels_last):
                break

            labels_last = labels

        # Compute inertia for this iteration
        inertia = np.sum(np.min(distances, axis=0))

        # Update best results if this iteration has lower inertia
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels_last

    return best_centers, best_labels

def iterative_kmeans(points: np.ndarray,
                     max_n_clusters: int,
                     n_iterations: int,
                     max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Applies the k-means algorithm multiple times and returns the best result in terms of silhouette score.

    This algorithm runs the k-means algorithm for all number of clusters until max_n_clusters. The silhouette score is
    calculated for each solution. The clusters with the highest silhouette score are returned.

    :param points: The (down-sampled) points of the pointcloud that should be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_n_clusters: The maximum number of clusters that is tested.
    :type max_n_clusters: int

    :param n_iterations: Number of time each k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of each k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    best_score = -1
    best_centers = []
    best_labels = []
    for k in range(1, max_n_clusters): 
        centers_i, labels_i = kmeans(points, k, n_iterations, max_singlerun_iterations)
        score = silhouette_score(points, centers_i, labels_i)

        if score >= best_score:
            best_score = score
            best_centers = centers_i
            best_labels = labels_i

    return best_centers, best_labels

def gmeans(points: np.ndarray,
           tolerance: float,
           max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the g-means algorithm.

    The algorithm was proposed by Hamerly, Greg, and Charles Elkan. "Learning the k in k-means." Advances in neural
    information processing systems 16 (2003).

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param tolerance: Tolerance for Anderson-Darling normality test
    :type tolerance: float

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Step 1: Start with k=1 cluster
    centers = np.mean(points, axis=0, keepdims=True)  # First cluster is mean of all points
   
    while True:
        # Step 2: Run k-means
        print(centers)
        new_centers, labels = kmeans(points, len(centers), n_iterations=1, max_singlerun_iterations=max_singlerun_iterations,
                                     centers_in=centers)  # Call kmeans function

        # Initialize the list to store the final centers
        final_centers = []

        # Step 3: Split clusters if necessary
        for cluster_idx in range(len(centers)):
            cluster_points = points[labels == cluster_idx]

            # Check if the cluster has enough points for covariance matrix calculation
            if len(cluster_points) < 2:
                final_centers.append(np.array(centers[cluster_idx]))  # Convert to NumPy array
                continue

            # Calculate covariance matrix and eigenvalues
            cov_matrix = np.cov(cluster_points, rowvar=False)
            _, eigenvalues = np.linalg.eigh(cov_matrix)

            # Calculate two new centers
            for i in range(2):
                lambda_max = np.max(eigenvalues)
                eigen_vector = np.linalg.eig(cov_matrix)[1][:, -1]
                offset = eigen_vector * np.sqrt(2 * lambda_max / np.pi)
                newer_centers = np.array([centers[cluster_idx] + offset, centers[cluster_idx] - offset])

                # Step 4: Run k-means on new centers and subset of points
                final_center, new_labels = kmeans(cluster_points, len(newer_centers), n_iterations=1,
                                                max_singlerun_iterations=max_singlerun_iterations, centers_in=newer_centers)
                final_centers.append(final_center)

                # Step 5: Project points onto cluster
                v = final_center[0] - final_center[1]

                # Check if the cluster has enough points for projection
                if len(cluster_points) >= 1:
                    norm_v = np.linalg.norm(v)

                    # Check if the norm of v is not zero before performing the division
                    if norm_v != 0:
                        x_projected = np.dot(cluster_points - final_center[1], v) / norm_v
                    else:
                        # Handle the case when the norm is zero (avoid division by zero)
                        x_projected = np.zeros(len(cluster_points))

                    # Step 6: Perform Anderson-Darling normality test
                    estimation, critical, _ = anderson(x_projected)
                    if estimation <= critical[-1] * tolerance:
                        # Check if there are enough clusters before accessing indices
                        if len(final_centers) > 2 * cluster_idx + 1:
                            final_centers[2 * cluster_idx] = final_center[0]
                            final_centers[2 * cluster_idx + 1] = final_center[1]

        # Step 8: Check convergence
        if np.array_equal(final_centers, centers):
            break
        else:
            centers = final_centers

    return centers, new_labels

    ############## Sources ###################
    # https://abhishek005.medium.com/g-means-for-non-spherical-clusters-from-scratch-44c1b1d2e3ea


def dbscan(points: np.ndarray,
           eps: float = 0.05,
           min_samples: int = 10) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    The algorithm was proposed in Ester, Martin, et al. "A density-based algorithm for discovering clusters in large
    spatial databases with noise." kdd. Vol. 96. No. 34. 1996.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """
    # 1. Initially set label for each point to 0
    labels = np.zeros(len(points), dtype=int)

    # C for labeling clusters
    C = 0

    ######################################################

    for data_point in range(len(points)):
        if labels[data_point] != 0:
            continue

        # 2. Retrieve points in an epsilon neighborhood (Region query)
        neighbors = [region_point for region_point in range(len(points)) if np.linalg.norm(points[region_point] - points[data_point]) < eps]

        # 3. If the number of points is smaller than min_samples, points are marked as noise
        if len(neighbors) < min_samples:
            labels[data_point] = -1

        # 4. If the number of points is bigger than min_samples, xi is a core point
        else:
            C += 1  # Cluster name
            labels[data_point] = C  # Data label with cluster name
            i = 0
            while i < len(neighbors):  # Loop over neighboring points
                Pn = neighbors[i]
                if labels[Pn] == -1 or labels[Pn] == 0:  # if point is noise or unlabeled, give it a label
                    labels[Pn] = C

                    # Region query (within the loop)
                    pt_neighbors = [region_point for region_point in range(len(points)) if np.linalg.norm(points[region_point] - points[Pn]) < eps]

                    if len(pt_neighbors) >= min_samples:
                        neighbors.extend(pt_neighbors)

                i += 1

    ######################################################

    return labels

###### Sources #######
# https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c
