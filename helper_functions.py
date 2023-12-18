#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions to plot the results and calculate the silhouette score

Author: FILL IN
MatrNr: FILL IN
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import distance


def silhouette_score(points: np.ndarray,
                     centers: np.ndarray,
                     labels: np.ndarray) -> float:
    """ Calculate the silhouette score for clusters.

    Calculate the silhouette score for clusters. It specifies how similar a sample is to its own cluster compared to
    other clusters. It was defined in Rousseeuw, Peter J. "Silhouettes: a graphical aid to the interpretation and
    validation of cluster analysis." J. Comput. Appl. Math. 20 (1987): 53-65.

    :param points: The (down-sampled) points of the pointcloud
    :type points: np.ndarray with shape = (n_points, 3)

    :param centers: The centers of the clusters. Each row is the center of a cluster.
    :type centers: np.ndarray with shape = (n_clusters, 3)

    :param labels: Array with a different label for each cluster for each point
            The label i corresponds with the center in centers[i]
    :type labels: np.ndarray with shape = (n_points,)

    :return: The calculated silhouette score is the mean silhouette coefficient of all samples in the range [-1, 1]
    :rtype: float
    """
    ######################################################
    # Initialize variables
    n = len(points)
    clusters = np.unique(labels)
    silhouette_scores = np.zeros(n)

    for i in range(n):
        cluster_idx = labels[i]
        cluster_points = points[labels == cluster_idx]

        # Calculate mean intra-cluster distance (a)
        a_i = np.mean(distance.cdist(points[i].reshape(1, -1), cluster_points, metric='euclidean'))

        # Calculate mean nearest-cluster distance (b)
        b_i = np.inf
        for j in clusters:
            if j != cluster_idx:
                other_cluster_points = points[labels == j]
                b_ij = np.mean(distance.cdist(points[i].reshape(1, -1), other_cluster_points, metric='euclidean'))
                b_i = min(b_i, b_ij)

        # Calculate silhouette score for the i-th data point
        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0  # Avoid division by zero

    # Mean of silhouette scores
    silhouette_score = np.mean(silhouette_scores)

    return silhouette_score

    ######################################################
    # Sources
    # https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
    # 


def plot_dominant_plane(pcd: o3d.geometry.PointCloud,
                        inliers: np.ndarray,
                        plane_eq: np.ndarray) -> None:
    """ Plot the inlier points in red and the rest of the pointcloud as is. A coordinate frame is drawn on the plane

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud

    :param inliers: Boolean array with the same size as pcd.points. Is True if the point at the index is an inlier
    :type inliers: np.array

    :param plane_eq: An array with the coefficients of the plane equation ax+by+cz+d=0
    :type plane_eq: np.array [a,b,c,d]

    :return: None
    """

    # Filter the inlier points and color them red
    inlier_indices = np.nonzero(inliers)[0]
    inlier_cloud = pcd.select_by_index(inlier_indices)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)

    # Create a rotation matrix according to the plane equation.
    # Detailed explanation of the approach can be found here: https://math.stackexchange.com/q/1957132
    normal_vector = -plane_eq[0:3] / np.linalg.norm(plane_eq[0:3])
    u1 = np.cross(normal_vector, [0, 0, 1])
    u2 = np.cross(normal_vector, u1)
    rot_mat = np.c_[u1, u2, normal_vector]

    # Create a coordinate frame and transform it to a point on the plane and with its z-axis in the same direction as
    # the normal vector of the plane
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.rotate(rot_mat, center=(0, 0, 0))
    if any(inlier_indices):
        coordinate_frame.translate(np.asarray(inlier_cloud.points)[-1])
        coordinate_frame.scale(0.3, np.asarray(inlier_cloud.points)[-1])

    geometries = [inlier_cloud, outlier_cloud, coordinate_frame]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in geometries:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.3, 0.32, -0.9])
    vc.set_lookat([-0.13, -0.15, 0.92])
    vc.set_up([0.22, -0.89, -0.39])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def plot_clustering_results(pcd: o3d.geometry.PointCloud,
                            labels: np.ndarray,
                            method_name: str,
                            cmap: str = "tab20"):
    labels = labels - labels.min()
    print(method_name + f": Point cloud has {int(labels.max()) + 1} clusters")
    colors = plt.get_cmap(cmap)(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

# From https://github.com/isl-org/Open3D/issues/2
def text_3d(text, pos, direction=None, degree=0.0, font='RobotoMono-Medium.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd
