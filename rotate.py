def allocation_plan(m_group, num_values):
    assert num_values%m_group == 0
    order = range(num_values)
    per_group = num_values/m_group
    res = []
    cache = []
    for i in order:
        if i%per_group == 0:
            if i/per_group%2 == 0:
                cache = reversed(cache)
            res.extend(cache) 
            cache = []
        cache.append(i)

    if i/per_group%2 == 1:
        cache = reversed(cache)
    res.extend(cache) 

    return res

def eigenvalue_allocation(values, plan):
    return tf.gather(tf.argsort(values), plan)

def local_rotation(df, covariance_column, num_feature):
    with tf.Graph().as_default():
        covariance = tf.placeholder(tf.double, [None, num_feature, num_feature], name=covariance_column)
        eigen_values, eigen_vectors = tf.self_adjoint_eig(covariance)
        eigen_values = tf.identity(eigen_values, name='eigen_values')
        eigen_vectors = tf.identity(eigen_vectors, name='eigen_vectors')
        orders = tf.map_fn(lambda r: eigenvalue_allocation), eigen_values, dtype=tf.int32)
        optimal_rotations = tf.map_fn(lambda (vector, order): tf.gather(vector, order, axis=0), (eigen_vectors, orders), dtype=tf.double)
        return tfs.map_blocks(optimal_rotations, df)

