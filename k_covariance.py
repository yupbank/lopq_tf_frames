import numpy as np

import starscream.tensorflow as tf
import tensorframes as tfs

def covariance(df, feature_column, num_features, coarse_center):
    with tf.Graph().as_default():
        features = tf.placeholder(tf.double, [None, num_features], name=feature_column)
        count = tf.identity(tf.ones_like(features)[:, 0], name='count')
        out = tf.identity(tf.map_fn(lambda x: tf.einsum('i,j->ij', x, x), features, dtype=tf.double), name='out')
        df1 = tfs.map_blocks(out, df)
    with tf.Graph().as_default():
        features = tf.placeholder(tf.double, [None, num_features], name=feature_column+'_input')
        out = tf.placeholder(tf.double, [None, num_features, num_features], name='out_input')
        count = tf.placeholder(tf.double, [None], name='count_input')
        expected_mean = tf.identity(tf.reduce_sum(features, axis=0), name=feature_column)
        expected_out = tf.identity(tf.reduce_sum(out, axis=0), name='out')
        expected_count = tf.identity(tf.reduce_sum(count, axis=0), name='count')
        df2 = tfs.aggregate([expected_mean, expected_out, expected_count], df1.groupby(coarse_center))
    with tf.Graph().as_default():
        features = tf.placeholder(tf.double, [None, num_features], name=feature_column)
        out = tf.placeholder(tf.double, [None, num_features, num_features], name='out')
        count = tf.placeholder(tf.double, [None], name='count')
        covariance = tf.identity(tf.map_fn(lambda (f, o, c): (o + tf.transpose(o)) / (2 * c - 2) - tf.einsum('i,j->ij', f, f), (features, out, count), dtype=tf.double), name='covariance')
        df3 = tfs.map_blocks(covariance, df2)
    return df3
