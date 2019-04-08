# -*- coding: UTF-8 -*-
import sys
from utils import *
import time
from ops import *
import transform_nets
# os.path.dirname:Get a path other than the file name
# os.path.abspath(__file__):Get the absolute path of the current module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Utils'))


class Point2SR:
    def __init__(self, sess_, flog, batch_size=32, num_pts=2048, l1_lambda=0, epoch=200, lr_d=0.001, lr_g=0.0001):
        self.isSingleClass = True
        self.batch_size = batch_size
        self.num_pts = num_pts
        self.l1_lambda = l1_lambda
        self.epoch = epoch
        self.sess = sess_
        self.flog = flog
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.build_model()

    def generator(self, point_cloud):

        with tf.variable_scope('transform_net1'):
            transform = transform_nets.input_transform_net(point_cloud=point_cloud,
                                                           num_pts=self.num_pts,
                                                           batch_size=self.batch_size,
                                                           k=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)

        with tf.variable_scope("generator_1"):
            input_image = tf.expand_dims(point_cloud_transformed, -1)

            net1 = conv2d(input_=input_image, k_w=3, output_dim=64, scope='g_conv1')
            # net1 = batch_norm(x=net1, is_training=is_training, scope='g_bn_1')
            net1 = group_norm(net1, scope='g_gn_1')
            net1 = tf.nn.relu(net1)
            self.deadReLU_net1 = tf.summary.histogram("g_net1", net1)

            net2 = conv2d(input_=net1, output_dim=64, scope='g_conv2')
            # net2 = batch_norm(x=net2, is_training=is_training, scope='g_bn_2')
            net2 = group_norm(net2, scope='g_gn_2')
            net2 = tf.nn.relu(net2)
            self.deadReLU_net2 = tf.summary.histogram("g_net2", net2)

        with tf.variable_scope('transform_net2'):
            transform = transform_nets.feature_transform_net(inputs=net2,
                                                             num_pts=self.num_pts,
                                                             batch_size=self.batch_size,
                                                             k=64)
        net_transformed = tf.matmul(tf.squeeze(net2, axis=[2]), transform)
        point_feat = tf.expand_dims(net_transformed, [2])

        with tf.variable_scope("generator_2"):

            net_transformed = tf.expand_dims(net_transformed, [2])

            net3 = conv2d(input_=net_transformed, output_dim=128, scope='g_conv3')
            # net3 = batch_norm(x=net3, is_training=is_training, scope='g_bn_3')
            net3 = group_norm(net3, scope='g_gn_3')
            net3 = tf.nn.relu(net3)
            self.deadReLU_net3 = tf.summary.histogram("g_net3", net3)

            net4 = conv2d(input_=net3, output_dim=512, scope='g_conv4')
            # net4 = batch_norm(x=net4, is_training=is_training, scope='g_bn_4')
            net4 = group_norm(net4, scope='g_gn_4')
            net4 = tf.nn.relu(net4)
            self.deadReLU_net4 = tf.summary.histogram("g_net4", net4)

            net5 = conv2d(input_=net4, output_dim=1024, scope='g_conv5')
            # net5 = batch_norm(x=net5, is_training=is_training, scope='g_bn_5')
            net5 = group_norm(net5, scope='g_gn_5')
            net5 = tf.nn.relu(net5)
            self.deadReLU_net5 = tf.summary.histogram("g_net5", net5)

            net_max = tf.nn.max_pool(net5, ksize=[1, self.num_pts, 1, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="VALID",
                                     name="g_maxpool")

            expand = tf.tile(net_max, [1, self.num_pts, 1, 1])
            concat = tf.concat(axis=3, values=[point_feat, expand])

            out1 = conv2d(input_=concat, output_dim=512, scope='seg/g_conv1')
            # out1 = batch_norm(x=out1, is_training=is_training, scope='g_bn_seg_1')
            out1 = group_norm(out1, scope='g_gn_seg_1')
            out1 = tf.nn.relu(out1)
            self.deadReLU_out1 = tf.summary.histogram("g_out1", out1)
            # out1 = tf.nn.dropout(out1, keep_prob=keep_prob, name="seg/g_dp1")

            out2 = conv2d(input_=out1, output_dim=256, scope='seg/g_conv2')
            # out2 = batch_norm(x=out2, is_training=is_training, scope='g_bn_seg_2')
            out2 = group_norm(out2, scope='g_gn_seg_2')
            out2 = tf.nn.relu(out2)
            self.deadReLU_out2 = tf.summary.histogram("g_out2", out2)
            # out2 = tf.nn.dropout(out2, keep_prob=keep_prob, name="seg/g_dp2")

            out3 = conv2d(input_=out2, output_dim=128, scope='seg/g_conv3')
            # out3 = batch_norm(x=out3, is_training=is_training, scope='g_bn_seg_3')
            out3 = group_norm(out3, scope='g_gn_seg_3')
            out3 = tf.nn.relu(out3)
            self.deadReLU_out3 = tf.summary.histogram("g_out3", out3)

            out4 = conv2d(input_=out3, output_dim=128, scope='seg/g_conv4')
            # out4 = batch_norm(x=out4, is_training=is_training, scope='g_bn_seg_4')
            out4 = group_norm(out4, scope='g_gn_seg_4')
            out4 = tf.nn.relu(out4)
            self.deadReLU_out4 = tf.summary.histogram("g_out4", out4)

            out5 = conv2d(input_=out4, output_dim=3, scope='seg/g_conv5')
            out5 = tf.nn.tanh(out5)

            output = tf.reshape(out5, [self.batch_size, self.num_pts, 3])

            return output

    def discriminator(self, new_point_cloud, reuse):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            with tf.variable_scope('transform_net3'):
                transform = transform_nets.input_transform_net2(point_cloud=new_point_cloud,
                                                                num_pts=self.num_pts,
                                                                batch_size=self.batch_size,
                                                                k=6)
            point_cloud_transformed = tf.matmul(new_point_cloud, transform)
            input_image = tf.expand_dims(point_cloud_transformed, -1)

            net1 = conv2d(input_=input_image, output_dim=64, k_w=6, sn=True, scope='d_conv1')
            # net1 = batch_norm(net1, is_training=is_training, scope='d_bn_1')
            # net1 = group_norm(net1, scope='d_gn_1')
            net1 = tf.nn.relu(net1)

            net2 = conv2d(input_=net1, output_dim=64, sn=True, scope='d_conv2')
            # net2 = batch_norm(net2, is_training=is_training, scope='d_bn_2')
            # net2 = group_norm(net2, scope='d_gn_2')
            net2 = tf.nn.relu(net2)

            with tf.variable_scope('transform_net4'):
                transform = transform_nets.feature_transform_net2(inputs=net2,
                                                                  num_pts=self.num_pts,
                                                                  batch_size=self.batch_size,
                                                                  k=64)
            net_transformed = tf.matmul(tf.squeeze(net2, axis=[2]), transform)
            net_transformed = tf.expand_dims(net_transformed, [2])

            net3 = conv2d(input_=net_transformed, output_dim=128, sn=True, scope='d_conv3')
            # net3 = batch_norm(net3, is_training=is_training, scope='d_bn_3')
            # net3 = group_norm(net3, scope='d_gn_3')
            net3 = tf.nn.relu(net3)

            net4 = conv2d(input_=net3, output_dim=512, sn=True, scope='d_conv4')
            # net4 = batch_norm(net4, is_training=is_training, scope='d_bn_4')
            # net4 = group_norm(net4, scope='d_gn_4')
            net4 = tf.nn.relu(net4)

            net5 = conv2d(input_=net4, output_dim=1024, sn=True, scope='d_conv5')
            # net5 = batch_norm(net5, is_training=is_training, scope='d_bn_5')
            # net5 = group_norm(net5, scope='d_gn_5')
            net5 = tf.nn.relu(net5)

            net_max = tf.nn.max_pool(net5, ksize=[1, self.num_pts, 1, 1], strides=[1, 2, 2, 1],
                                     padding="VALID", name="d_maxpool")

            with tf.variable_scope("cls"):

                net6 = tf.reshape(net_max, [self.batch_size, -1])

                net7 = dense(net6, 512, sn=True, scope="d_fc1")
                # net7 = batch_norm(net7, is_training=is_training, scope='d_bn__cls_1')
                # net7 = group_norm(net7, scope='d_gn_7')
                net7 = tf.nn.relu(net7)
                net7 = tf.nn.dropout(net7, keep_prob=0.7, name="cls/d_dp1")

                net8 = dense(net7, 256, sn=True, scope="d_fc2")
                # net8 = batch_norm(net8, is_training=is_training, scope='d_bn__cls_2')
                # net8 = group_norm(net8, scope='d_gn_8')
                net8 = tf.nn.relu(net8)
                net8 = tf.nn.dropout(net8, keep_prob=0.7, name="cls/d_dp2")

                net_logit = dense(net8, 1, sn=True, scope="cls/d_fc3")

                net10 = tf.nn.sigmoid(net_logit)

                return net_logit, net10

    def build_model(self):
        pass

        self.all_point_cloud = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_pts, 6),
                                              name='all_point_cloud')
        self.bn_is_train = tf.placeholder(dtype=tf.bool, shape=(), name='bn_is_train')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

        self.ori_point_cloud = self.all_point_cloud[:, :, :3]
        self.copy_point_cloud = self.all_point_cloud[:, :, 3:]

        self.new_point_cloud = self.generator(self.copy_point_cloud)

        self.d_real_pointcloud_hist = tf.summary.histogram("ori_point_cloud", self.ori_point_cloud)
        self.d_fake_pointcloud_hist = tf.summary.histogram("new_point_cloud", self.new_point_cloud)
        self.real_point_cloud = tf.concat([self.ori_point_cloud, self.copy_point_cloud], axis=-1)
        self.fake_point_cloud = tf.concat([self.new_point_cloud, self.copy_point_cloud], axis=-1)

        self.D_real_logit, self.D_real = self.discriminator(self.real_point_cloud, reuse=False)
        self.D_fake_logit, self.D_fake = self.discriminator(self.fake_point_cloud, reuse=True)

        self.d_real = tf.reduce_mean(self.D_real)
        self.d_fake = tf.reduce_mean(self.D_fake)

        self.d_loss_real = tf.reduce_mean(tf.nn.softplus(-self.D_real_logit))
        self.d_loss_fake = tf.reduce_mean(tf.nn.softplus(self.D_fake_logit))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_gan = tf.reduce_mean(tf.nn.softplus(-self.D_fake_logit))
        self.g_loss_l1 = tf.reduce_mean(tf.abs(self.ori_point_cloud - self.new_point_cloud))
        self.g_loss = self.g_loss_gan + self.l1_lambda * self.g_loss_l1

        # d_loss_real d_loss_fake
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_gan_sum = tf.summary.scalar("g_loss_gan", self.g_loss_gan)
        self.g_loss_l1_sum = tf.summary.scalar("g_loss_l1", self.g_loss_l1)
        # g_loss d_loss(d_loss_real and d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        # d_real_probably d_fake_probably
        self.d_real_proba_sum = tf.summary.scalar("d_real_probably", self.d_real)
        self.d_fake_proba_sum = tf.summary.scalar("d_fake_probably", self.d_fake)

        self.d_sum = tf.summary.merge([self.d_loss_sum, self.d_loss_fake_sum, self.d_loss_real_sum,
                                       self.d_real_pointcloud_hist, self.d_fake_pointcloud_hist,
                                       self.d_real_proba_sum, self.d_fake_proba_sum,
                                       self.deadReLU_out1, self.deadReLU_out2, self.deadReLU_out3,
                                       self.deadReLU_net1, self.deadReLU_net2, self.deadReLU_net3,
                                       self.deadReLU_net4, self.deadReLU_net5])

        self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_loss_gan_sum, self.g_loss_l1_sum])

        # tf.trainable_variables:List of variables that need to be trained
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr_d,
                                                  name="d_optim").minimize(self.d_loss,
                                                                           var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr_g,
                                                  name="g_optim").minimize(self.g_loss,
                                                                           var_list=self.g_vars)

    def train(self, train_dir, coordinates, ncoordinates, ocoordinates,
              test_coordinates, test_ncoordinates, test_ocoordinates):

        double_coordinates = np.concatenate((coordinates, ncoordinates), axis=-1)
        test_double_coordinates = np.concatenate((test_coordinates, test_ncoordinates), axis=-1)

        assert double_coordinates.shape == (coordinates.shape[0], coordinates.shape[1], 6)
        show_all_variables()

        self.saver = tf.train.Saver(max_to_keep=200)
        model_dir = os.path.join(train_dir, "model")
        os.mkdir(os.path.join(train_dir, "logs"))
        train_sum_dir = os.path.join(train_dir, "logs", "train")
        test_sum_dir = os.path.join(train_dir, "logs", "test")
        test_sum_dir_only = os.path.join(train_dir, "logs", "test_only")
        test_sum_dir_csv = os.path.join(train_dir, "logs", "test_csv")
        train_sum_dir_only = os.path.join(train_dir, "logs", "train_only")
        # train_sum_dir_only2 = os.path.join(train_dir, "logs", "train_only2")

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(train_sum_dir):
            os.mkdir(train_sum_dir)
        if not os.path.exists(test_sum_dir):
            os.mkdir(test_sum_dir)
        if not os.path.exists(test_sum_dir_only):
            os.mkdir(test_sum_dir_only)
        if not os.path.exists(train_sum_dir_only):
            os.mkdir(train_sum_dir_only)
        if not os.path.exists(test_sum_dir_csv):
            os.mkdir(test_sum_dir_csv)
        self.train_writer = tf.summary.FileWriter(train_sum_dir, self.sess.graph)

        tf.global_variables_initializer().run()
        self.num_batches = coordinates.shape[0] // self.batch_size

        test_masks = np.random.choice(test_coordinates.shape[0], self.batch_size, replace=False)
        batch_test_double_coordinates = test_double_coordinates[test_masks]
        # batch_test_nnormalized_coordinates = test_nnormalized_coordinates[test_masks]
        # batch_test_coordinates = test_coordinates[test_masks]
        batch_test_ocoordinates = test_ocoordinates[test_masks]

        train_masks = np.random.choice(coordinates.shape[0], self.batch_size, replace=False)

        batch_train_double_coordinates = double_coordinates[train_masks]
        # batch_train_coordinates = coordinates[train_masks]
        batch_train_ocoordinates = ocoordinates[train_masks]
        # batch_train_ncoordinates = ncoordinates[train_masks]
        # batch_train_nnormalized_coordinates = nnormalized_coordinates[train_masks]

        start_time = time.time()
        for epoch in range(self.epoch):
            for idx in range(self.num_batches):
                global_step = epoch * self.num_batches + idx + 1
                masks = range(idx * self.batch_size, (idx + 1) * self.batch_size)
                batch_double_coordinates = double_coordinates[masks]
                d_real, d_fake, d_sum_save, d_loss_print, = self.sess.run(
                    [self.d_real, self.d_fake, self.d_sum, self.d_loss], feed_dict={
                        self.all_point_cloud: batch_double_coordinates,
                        self.bn_is_train: True,
                        self.keep_prob: 0.8})

                if d_real < 0.7:
                    self.sess.run([self.d_optim],
                                  feed_dict={self.all_point_cloud: batch_double_coordinates,
                                             self.bn_is_train: True,
                                             self.keep_prob: 0.8
                                             })
                self.train_writer.add_summary(d_sum_save, global_step)
                g_sum_save, g_loss_print, _ = self.sess.run([self.g_sum, self.g_loss, self.g_optim], feed_dict={
                    self.all_point_cloud: batch_double_coordinates,
                    self.bn_is_train: True,
                    self.keep_prob: 0.8})
                self.train_writer.add_summary(g_sum_save, global_step)
                g_sum_save, g_loss_print, _ = self.sess.run([self.g_sum, self.g_loss, self.g_optim], feed_dict={
                    self.all_point_cloud: batch_double_coordinates,
                    self.bn_is_train: True,
                    self.keep_prob: 0.8})
                self.train_writer.add_summary(g_sum_save, global_step)

                period = time.time() - start_time
                printout(self.flog, "Training! epoch %3d/%3d "
                                    "batch%3d/%3d time: %2dh%2dm%2ds  "
                                    "d_loss: %.4f g_loss: %.4f d_real: %.4f d_fake: %.4f" %
                         (epoch + 1, self.epoch, idx + 1, self.num_batches, period // 3600,
                          (period - 3600 * (period // 3600)) // 60, period % 60,
                          d_loss_print, g_loss_print, d_real, d_fake))

            train_new_coordinates = self.sess.run(self.new_point_cloud,
                                                  feed_dict={self.all_point_cloud: batch_train_double_coordinates,
                                                             self.bn_is_train: False,
                                                             self.keep_prob: 0.8})

            for idx, data in enumerate(batch_train_ocoordinates):
                name_ = os.path.join(train_sum_dir, "epoch{0}_{1}.png".format(epoch, idx))
                name_only = os.path.join(train_sum_dir_only, "epoch_only{0}_{1}.png".format(epoch, idx))
                # name_only2 = os.path.join(train_sum_dir_only2, "epoch_only{0}_{1}.png".format(epoch, idx))

                print("train shape:", train_new_coordinates[idx].shape)
                display(data, train_new_coordinates[idx], name_=name_)
                display_only(train_new_coordinates[idx], name_=name_only)
                # display_only(batch_train_ncoordinates[idx], name_=name_only2)
                printout(self.flog, "Saved! {}".format(name_))

            if epoch % 3 == 0:
                self.saver.save(self.sess, model_dir + "/model", epoch)
                test_new_coordinates = self.sess.run(self.new_point_cloud,
                                                     feed_dict={self.all_point_cloud: batch_test_double_coordinates,
                                                                self.bn_is_train: False, self.keep_prob: 0.8})

                for idx, data in enumerate(batch_test_ocoordinates):
                    name_ = os.path.join(test_sum_dir, "epoch{0}_{1}.png".format(epoch, idx))
                    name_only = os.path.join(test_sum_dir_only, "epoch_only{0}_{1}.png".format(epoch, idx))
                    name_csv = os.path.join(test_sum_dir_csv, "epoch_csv{0}_{1}.csv".format(epoch, idx))
                    display(data, test_new_coordinates[idx], name_=name_)
                    display_only(test_new_coordinates[idx], name_=name_only)
                    save_to_csv(data=test_new_coordinates[idx], nums=batch_test_ocoordinates, name_=name_csv)
                    printout(self.flog, "Saved! {}".format(name_))
                    printout(self.flog, "Saved! {}".format(name_csv))
