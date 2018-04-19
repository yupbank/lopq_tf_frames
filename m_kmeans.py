from operator import itemgetter

import numpy as np

import tensorflow as tf
import tensorframes as tfs


def cosine_similarity(points, centers, centers_normalized=True, points_normalized=False):
    if not centers_normalized:
        centers = tf.nn.l2_normalize(centers, dim=1)
    if not points_normalized:
        points = tf.nn.l2_normalize(points, dim=1)

    return 1 - tf.matmul(points, centers, transpose_b=True)


def calculate_new_centers(points, old_centers, num_centroids, centers_normalized=True, points_normalized=False):
    return calculate_new_centers_for_m_slice([slice(0, None)],
                                             points,
                                             old_centers,
                                             num_centroids,
                                             centers_normalized,
                                             points_normalized)


def calculate_new_centers_for_m_slice(m_slice, points, old_centers, num_centroids, centers_normalized=True,
                                      points_normalized=False):
    vector_sums = []
    counts = []
    for feature_slice in m_slice:
        sub_points = points[:, feature_slice]
        sub_old_centers = old_centers[:, feature_slice]
        sub_similarities = cosine_similarity(sub_points, sub_old_centers, centers_normalized, points_normalized)
        sub_indexes = tf.argmin(sub_similarities, axis=1)
        sub_vector_sums = tf.unsorted_segment_sum(sub_points, sub_indexes, num_centroids)
        sub_counts = tf.unsorted_segment_sum(tf.ones_like(sub_indexes), sub_indexes, num_centroids)
        vector_sums.append(sub_vector_sums)
        counts.append(sub_counts)
    counts = tf.expand_dims(tf.concat(counts, axis=0), axis=0)
    vector_sums = tf.expand_dims(tf.concat(vector_sums, axis=0), axis=0)
    return counts, vector_sums


def kmeans(df, feature_column, num_centroids, num_features, max_iter=10):
    return m_kmeans(df, feature_column, num_centroids, num_features, 1, max_iter)


def m_kmeans(df, feature_column, num_centroids_each, num_features, m_groups, max_iter=10):
    """
    M K-means algorithm applied on a dataframe of points

    :param df: dataframe, contains all points
    :param feature_column: string, the points column within df
    :param num_centroids_each: int, k clusters
    :param num_features: int, dimension of a point vector
    :param m_groups: int, number of groups a point is spitted into
    :param max_iter: int, maximum number of iterations

    :return: numpy.array: [num_centroids, num_features], the k cluster centers with m groups concatenated
    """
    initial_centers = df.select(feature_column).take(num_centroids_each)
    centers = np.array(initial_centers).reshape(num_centroids_each, num_features)
    m_slice = map(lambda r: slice(min(r), max(r) + 1), np.array_split(xrange(num_features), m_groups))
    slices = np.array_split(xrange(m_groups * num_centroids_each), m_groups)
    df = tfs.analyze(df)

    while max_iter > 0:
        max_iter -= 1

        with tf.Graph().as_default():
            points = tf.placeholder(tf.double, shape=[None, num_features], name=feature_column)
            counts, vector_sums = calculate_new_centers_for_m_slice(m_slice, points, tf.nn.l2_normalize(centers, dim=1),
                                                                    num_centroids_each)
            counts = tf.identity(counts, name='counts')
            vector_sums = tf.identity(vector_sums, name='vector_sums')
            df2 = tfs.map_blocks([counts, vector_sums], df, trim=True)

        with tf.Graph().as_default():
            counts = tf.placeholder(tf.int64, shape=[None, num_centroids_each * m_groups], name='counts_input')
            vector_sums = tf.placeholder(tf.double,
                                         shape=[None, num_centroids_each * m_groups, num_features / m_groups],
                                         name='vector_sums_input')
            count = tf.reduce_sum(counts, axis=0, name='counts')
            vector_sum = tf.reduce_sum(vector_sums, axis=0, name='vector_sums')
            d_count, d_vector_sum = tfs.reduce_blocks([count, vector_sum], df2)
            new_centers = d_vector_sum / (d_count[:, np.newaxis] + 1e-7)
            new_centers = np.concatenate([new_centers[i] for i in slices], axis=1)
        if np.allclose(centers, new_centers):
            break
        else:
            centers = new_centers

    return new_centers


def _assign_center(points, centers, assigned_name='concat', m=1):
    num_features = centers.shape[1]
    m_slice = map(lambda r: slice(min(r), max(r) + 1), np.array_split(xrange(num_features), m))
    assigned = []
    for feature_slice in m_slice:
        sub_points = points[:, feature_slice]
        sub_centers = centers[:, feature_slice]
        similarities = cosine_similarity(sub_points, sub_centers)
        sub_assigned = tf.expand_dims(tf.argmin(similarities, axis=1), axis=1)
        assigned.append(sub_assigned)
    return tf.concat(assigned, axis=1, name=assigned_name)


def _residual_of_assigned(points, assigned, centers, residual_name=None):
    closest_center = tf.squeeze(tf.gather(centers, assigned), 1)
    return tf.subtract(points, closest_center, name=residual_name)


def assign_center(df, feature_column, residual_column, assigned_coarse_column, assigned_pq_column, coarse_centers, pq_centers, m):
    """
    Assign the points into corresponding indexes

    :param df: dataframe, contains all points
    :param feature_column: string, the points column within df
    :param residual_column: string, the points residual column to be saved
    :param assigned_coarse_column: string, the output column name for coarse index.
    :param assigned_pq_column: string, the output column name for pq indexes.
    :param coarse_centers: numpy.array, [num_centroids, num_features] the coarse cluster centers
    :param pq_centers: numpy.array, [num_centroids, num_features] the pq cluster centers
    :param m: int, number of groups a point is spitted into for pq
    :return: dataframe, contains two extra columns, `assigned_coarse_column` and `assigned_pq_column`
    """
    df = residual_of_closest(df, feature_column, residual_column, coarse_centers)
    num_features = coarse_centers.shape[1]
    with tf.Graph().as_default():
        points = tf.placeholder(tf.double, shape=[None, num_features], name=feature_column)
        residuals = tf.placeholder(tf.double, shape=[None, num_features], name=residual_column)
        assigned_coarse = _assign_center(points, coarse_centers, assigned_coarse_column)
        assigned_pq = _assign_center(residuals, pq_centers, assigned_pq_column, m)
        return tfs.map_blocks([assigned_coarse, assigned_pq], df)


def residual_of_closest(df, feature_column, residual_column, centers, assigned_column='assigned'):
    """
    Residual between points and their closest center

    :param df: dataframe, contains all points
    :param feature_column: string, the points column within df
    :param residual_column: string, the output column name for residual error between closest center.
    :param centers: numpy.array, [num_centroids, num_features] the k cluster centers
    :param assigned_column: string, the output column name for index of closest center.
    :return: dataframe, contains two extra columns, `residual_column`, `assigned_column`
    """
    df = tfs.analyze(df)
    num_features = centers.shape[1]
    with tf.Graph().as_default():
        points = tf.placeholder(tf.double, shape=[None, num_features], name=feature_column)
        assigned = _assign_center(points, centers)
        residual = _residual_of_assigned(points, assigned, centers, residual_column)
        return tfs.map_blocks([assigned, residual], df)
