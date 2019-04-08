import sys
from utils import *
from ops import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Utils'))


def input_transform_net(point_cloud, batch_size, num_pts, k=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
            Return:
                Transformation matrix of size 3xK """
    input_image = tf.expand_dims(point_cloud, -1)
    net = conv2d(input_=input_image, k_w=3, output_dim=64, scope='tconv1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_1')
    net = group_norm(net, scope='t_gn_1')

    net = conv2d(input_=net, output_dim=64, scope='tconv2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_2')
    net = group_norm(net, scope='t_gn_2')

    net = conv2d(input_=net, output_dim=1024, scope='tconv3')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_3')
    net = group_norm(net, scope='t_gn_3')

    net = tf.nn.max_pool(net, ksize=[1, num_pts, 1, 1], strides=[1, 2, 2, 1], padding='VALID', name='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])

    net = dense(input_=net, output_size=512, scope='tfc1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_1')
    net = group_norm(net, scope='t_gn_fc_1')

    net = dense(input_=net, output_size=256, scope='tfc2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_2')
    net = group_norm(net, scope='t_gn_fc_2')

    with tf.variable_scope('transform_XYZ'):
        assert k == 3
        weights = tf.get_variable('weights', [256, k * k],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [k * k],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)

        biases += tf.constant(np.eye(k).flatten(), dtype=tf.float32)

        transform = tf.matmul(net, weights)

        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, k, k])

    return transform


def input_transform_net2(point_cloud, batch_size, num_pts, k=6):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
            Return:
                Transformation matrix of size 3xK """
    input_image = tf.expand_dims(point_cloud, -1)
    net = conv2d(input_=input_image, k_w=6, output_dim=64, sn=True, scope='tconv1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_1')
    # net = group_norm(net, scope='t_gn_1')

    net = conv2d(input_=net, output_dim=64, sn=True, scope='tconv2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_2')
    # net = group_norm(net, scope='t_gn_2')

    net = conv2d(input_=net, output_dim=1024, sn=True, scope='tconv3')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_3')
    # net = group_norm(net, scope='t_gn_3')

    net = tf.nn.max_pool(net, ksize=[1, num_pts, 1, 1], strides=[1, 2, 2, 1], padding='VALID', name='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])

    net = dense(input_=net, output_size=512, sn=True, scope='tfc1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_1')
    # net = group_norm(net, scope='t_gn_fc_1')

    net = dense(input_=net, output_size=256, sn=True, scope='tfc2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_2')
    # net = group_norm(net, scope='t_gn_fc_2')

    with tf.variable_scope('transform_XYZ'):
        assert k == 6
        weights = tf.get_variable('weights', [256, k * k],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [k * k],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)

        biases += tf.constant(np.eye(k).flatten(), dtype=tf.float32)

        transform = tf.matmul(net, weights)

        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, k, k])

    return transform


def feature_transform_net(inputs, batch_size, num_pts, k=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """

    net = conv2d(input_=inputs, output_dim=64, scope='tconv1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_1')
    net = group_norm(net, scope='t_gn_1')

    net = conv2d(input_=net, output_dim=128, scope='tconv2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_2')
    net = group_norm(net, scope='t_gn_2')

    net = conv2d(input_=net, output_dim=1024, scope='tconv3')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_3')
    net = group_norm(net, scope='t_gn_3')

    net = tf.nn.max_pool(net, ksize=[1, num_pts, 1, 1], strides=[1, 2, 2, 1], padding='VALID', name='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])

    net = dense(input_=net, output_size=512, scope='tfc1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_1')
    net = group_norm(net, scope='t_gn_fc_1')

    net = dense(input_=net, output_size=256, scope='tfc2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_2')
    net = group_norm(net, scope='t_gn_fc_2')

    with tf.variable_scope('transform_feat'):
        weights = tf.get_variable('weights', [256, k*k],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [k*k],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)

        biases += tf.constant(np.eye(k).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, k, k])
    return transform


def feature_transform_net2(inputs, batch_size,  num_pts, k=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """

    net = conv2d(input_=inputs, output_dim=64, sn=True, scope='tconv1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_1')
    # net = group_norm(net, scope='t_gn_1')

    net = conv2d(input_=net, output_dim=128, sn=True, scope='tconv2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_2')
    # net = group_norm(net, scope='t_gn_2')

    net = conv2d(input_=net, output_dim=1024, sn=True, scope='tconv3')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_3')
    # net = group_norm(net, scope='t_gn_3')

    net = tf.nn.max_pool(net, ksize=[1, num_pts, 1, 1], strides=[1, 2, 2, 1], padding='VALID', name='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])

    net = dense(input_=net, output_size=512, sn=True, scope='tfc1')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_1')
    # net = group_norm(net, scope='t_gn_fc_1')

    net = dense(input_=net, output_size=256, sn=True, scope='tfc2')
    # net = batch_norm(net, is_training=is_training, scope='t_bn_fc_2')
    # net = group_norm(net, scope='t_gn_fc_2')

    with tf.variable_scope('transform_feat'):
        weights = tf.get_variable('weights', [256, k*k],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [k*k],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)

        biases += tf.constant(np.eye(k).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, k, k])
    return transform
