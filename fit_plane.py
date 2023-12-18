#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: Tessa Pulli 
MatrNr: 12307536
"""

from typing import Tuple

import copy

import numpy as np
import open3d as o3d


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :return: (best_plane, best_inliers)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray)
    """
    ######################################################
       # Transform point cloud
    points = np.asarray(pcd.points)

    # Parameters for closing condition
    max_iterations = 1000
    sample_size = 3
    num_points = len(points)

    best_inliers = np.full(num_points, False, dtype=bool)
    best_num_inliers = 0

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Randomly sample 3 points
        np.random.seed()
        inliers = np.random.choice(num_points, size=sample_size, replace=False)

        # Extract coordinates of the sampled points
        x1, y1, z1 = points[inliers[0]]
        x2, y2, z2 = points[inliers[1]]
        x3, y3, z3 = points[inliers[2]]

        # Plane equation coefficients
        a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d = -(a * x1 + b * y1 + c * z1)
        plane_length = max(0.1, np.sqrt(a * a + b * b + c * c))  # 0.1 as the minimum value

        # Initialize current inliers
        current_inliers = []

        # Calculate distances and identify inliers
        for index in range(num_points):
            if index in inliers:
                continue

            x, y, z = points[index]
            distance = np.abs((a * x + b * y + c * z + d) / plane_length)

            if distance <= inlier_threshold:
                current_inliers.append(index)

        # Update best inliers
        if len(current_inliers) > best_num_inliers:
            best_num_inliers = len(current_inliers)
            best_inliers[:] = False
            best_inliers[current_inliers] = True

        # Termination condition
        e = best_num_inliers / num_points
        k_numerator = np.log(1 - confidence)
        k_denominator = np.log(1 - (1 - e) ** sample_size)
        k = k_numerator / k_denominator

        if (1 - (1 - e ** sample_size) ** k > (1 - confidence)):
            break

    # Refine the plane parameters using least squares
    A = np.column_stack((points[best_inliers], np.ones(len(points[best_inliers]))))
    b = np.zeros(len(points[best_inliers]))
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Extract refined plane parameters
    a, b, c, d = coefficients

    best_plane = np.array([a, b, c, d])

    ######################################################
    return best_plane, best_inliers

    ######## Sources ########
    # https://medium.com/@ajithraj_gangadharan/3d-ransac-algorithm-for-lidar-pcd-segmentation-315d2a51351
