from __future__ import print_function

import time
import csv
import os
from datetime import datetime

import numpy as np
import scipy.ndimage.interpolation
from evaluationScript.noduleCADEvaluationLUNA16 import *
import tensorflow as tf
from tensorflow.python.training import moving_averages


class model_def:
    def __init__(self):
        self.patch_size = 33
        self.mod_cnt = 4
        self.lbl_cnt = 5
        self.kernel_size = 3
        self.filters = [64, 128, 256]
        self.alpha = 0.1

        self.batch_size = 128
        self.do_rate = 0.5

    def init_weight_bias(self, name, shape, filtercnt, trainable):
        weights = tf.get_variable(name=name + "w", shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  dtype=tf.float32, trainable=trainable)
        biases = tf.Variable(initial_value=tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b",
                             trainable=trainable)
        return weights, biases

    def conv_layer(self, data, weight, bias, padding, layer_cnt, is_leaky, is_cat, is_bn, is_res, res_data):
        if is_cat:
            if layer_cnt < 8:
                data = tf.concat([data, res_data], 3)
                conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
                if is_leaky:
                    return self.leaky_relu_layer(tf.nn.bias_add(conv, bias), self.alpha)
                else:
                    return tf.nn.relu(tf.nn.bias_add(conv, bias))
            else:
                # shape = data.get_shape().as_list()
                data = tf.concat([data, res_data], 3)
                # data = tf.reshape(data, [shape[0], 1, 1, shape[1] * shape[2] * shape[3]*2])
                conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
                # conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
                # return tf.reshape(conv, shape)
                if is_leaky:
                    return self.leaky_relu_layer(tf.nn.bias_add(conv, bias), self.alpha)
                else:
                    return tf.nn.relu(tf.nn.bias_add(conv, bias))

        elif is_bn:
            conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
            bn_conv = self.batch_norm_nn(conv, layer_cnt)
            if is_leaky:
                return self.leaky_relu_layer(tf.nn.bias_add(conv, bias), self.alpha)
            else:
                return tf.nn.relu(tf.nn.bias_add(bn_conv, bias))
        elif is_res:
            data = tf.add(data, res_data)
            conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
            if is_leaky:
                conv = tf.nn.bias_add(conv, bias)
                return self.leaky_relu_layer(conv, self.alpha)
            else:
                return tf.nn.relu(tf.nn.bias_add(conv, bias))
        else:
            conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
            if is_leaky:
                return self.leaky_relu_layer(tf.nn.bias_add(conv, bias), self.alpha)
            else:
                return tf.nn.relu(tf.nn.bias_add(conv, bias))

    def conv3d_layer(self, data, weight, bias, padding, is_res, res_data):
        if is_res:
            data = tf.concat([data, res_data], 3)
        conv = tf.nn.conv3d(input=data, filter=weight, strides=[1, 1, 1, 1, 1], padding=padding)
        return tf.nn.bias_add(conv, bias)

    def batch_norm_layer(self, data, train=True):
        return tf.contrib.layers.batch_norm(inputs=data, is_training=train)

    def relu_layer(self, conv):
        return tf.nn.relu(conv)

    def leaky_relu_layer(self, conv, alpha):
        return tf.nn.relu(conv) - alpha * tf.nn.relu(-conv)

    def depth_wise_conv_layer(self, data, weight, bias, padding, is_inception):
        conv = tf.nn.depthwise_conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
        if is_inception:
            return tf.nn.bias_add(conv, bias)
        return tf.nn.relu(tf.nn.bias_add(conv, bias))

    def pool_layer(self, data):
        return tf.nn.max_pool(value=data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    def fc_layer(self, data, weight, bias, dropout, batch_norm=False):
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]

        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
        if batch_norm:
            hidden = tf.contrib.layers.batch_norm(inputs=hidden, is_training=True)
            # self.batch_norm_layer(hidden)
        hidden = tf.nn.relu(hidden)
        if dropout < 1.:
            hidden = tf.nn.dropout(hidden, dropout)
        return hidden

    def fc_concat_layer(self, data1, data2, weight, bias, dropout, batch_norm=False):
        data = tf.concat([data1, data2], axis=3)
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]

        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
        if batch_norm:
            hidden = tf.contrib.layers.batch_norm(inputs=hidden, is_training=True)
            # self.batch_norm_layer(hidden)
        hidden = tf.nn.relu(hidden)
        if dropout < 1.:
            hidden = tf.nn.dropout(hidden, dropout)
        return hidden

    def output_layer(self, data, weight, bias, label):
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]
        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)

        if label is None:
            return None, tf.nn.softmax(hidden)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hidden, labels=label)), tf.nn.softmax(hidden)

    def Gradual_Multi_scale_CNN(self, train=True, is_inception=False, is_output_layer=False, is_leaky=False):
        conv_layer_shape = [[self.kernel_size, self.kernel_size, 6, 64],        # 0 input & 1 conv
                            [self.kernel_size, self.kernel_size, 64, 64],       # 2 conv
                            [self.kernel_size, self.kernel_size, 64 + 6, 128],  # 3 concat & conv
                            [self.kernel_size, self.kernel_size, 128, 128],     # 4 conv
                            [self.kernel_size, self.kernel_size, 128 + 6, 192], # 5 concat & conv
                            [self.kernel_size, self.kernel_size, 192, 192],     # 6 conv
                            # ___________________pool___________________ #
                            [self.kernel_size, self.kernel_size, 192, 256],     # 7 conv
                            [self.kernel_size, self.kernel_size, 256, 256],     # 8 conv
                            # _________________________________________ #
                            # [1, 1, 256 * 2, 256],  # 1*1 conv
                            # [self.kernel_size, self.kernel_size, 256, 256],     # 9 Add conv
                            # ___________________pool__________________ #
                            [self.kernel_size, self.kernel_size, 256, 512],     # 10 conv
                            [self.kernel_size, self.kernel_size, 512, 512]]     # 11 conv


        fc_layer_shape = [[4 * 4 * 512, 1024], [1024, 1024], [1024, 2]]

        if train:
            batch_size = self.batch_size
            do_rate = self.do_rate
            train_data_node1 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))
            train_data_node2 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))
            train_labels_node = tf.placeholder(tf.int64, shape=batch_size)

        else:
            batch_size = 1
            do_rate = 1.
            train_data_node1 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))
            train_data_node2 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))
            train_labels_node = None

        top_node1 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))
        top_node2 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))
        bottom_node1 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))
        bottom_node2 = tf.placeholder(tf.float32, shape=(batch_size, 20, 20, 6))

        cw1 = []
        cb1 = []
        cw2 = []
        cb2 = []

        fw2 = []
        fb2 = []

        layers1 = [train_data_node1]
        layers2 = [train_data_node2]

        cross_entropy, softmax = None, None
        for kernel, layer_cnt in zip(conv_layer_shape, range(len(conv_layer_shape))):
            if layer_cnt < len(conv_layer_shape)-2:
                w1, b1 = self.init_weight_bias(name="c1%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1],
                                               trainable=train)
                cw1.append(w1)
                cb1.append(b1)

            w2, b2 = self.init_weight_bias(name="c2%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1],
                                           trainable=train)
            cw2.append(w2)
            cb2.append(b2)
        for kernel, layer_cnt in zip(fc_layer_shape, range(len(fc_layer_shape))):
            w2, b2 = self.init_weight_bias(name="f2%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1],
                                           trainable=train)
            fw2.append(w2)
            fb2.append(b2)

        for w, b, layer_cnt in zip(cw1, cb1, range(len(cw1))):
            print(layer_cnt)
            res_node = None
            if layer_cnt == 2:
                res = False
                cat = True
                bn = False
                res_node = bottom_node1
            elif layer_cnt == 4:
                res = False
                cat = True
                bn = False
                res_node = bottom_node2
            else:
                res = False
                cat = False
                bn = False

            output = self.conv_layer(data=layers2[-1], weight=w, bias=b, padding="SAME", layer_cnt=layer_cnt,
                                     is_leaky=is_leaky, is_cat=cat, is_bn=bn, is_res=res, res_data=res_node)
            layers2.append(output)
            # output = self.relu_layer(tf.nn.bias_add(output, b))
            # layers2.append(output)

            if layer_cnt == 5:
                output = self.pool_layer(data=layers2[-1])
                layers2.append(output)

        for w, b, layer_cnt in zip(cw2, cb2, range(len(cw2))):
            res_node = None
            # org_node = None
            if layer_cnt == 2:
                res = False
                cat = True
                bn = False
                res_node = top_node1
                org_node = layers1[-1]
            elif layer_cnt == 4:
                res = False
                cat = True
                bn = False
                res_node = top_node2
                org_node = layers1[-1]
            elif layer_cnt == 7:
                res = True
                cat = False
                bn = False
                res_node = layers2[-1]
                org_node = layers1[-1]
            else:
                res = False
                cat = False
                bn = False
                org_node = layers1[-1]

            output = self.conv_layer(data=org_node, weight=w, bias=b, padding="SAME", layer_cnt=layer_cnt,
                                     is_leaky=is_leaky, is_cat=cat, is_bn=bn, is_res=res, res_data=res_node)
            layers1.append(output)
            # output = self.relu_layer(tf.nn.bias_add(output, b))
            # layers1.append(output)

            if layer_cnt == 5 or layer_cnt == 7:
                output = self.pool_layer(data=layers1[-1])
                layers1.append(output)

        for w2, b2, layer_cnt in zip(fw2, fb2, range(len(fw2))):
            if layer_cnt == 2:
                cross_entropy, softmax = self.output_layer(data=layers1[-1], weight=w2, bias=b2, label=train_labels_node)
            else:
                output1 = self.fc_layer(data=layers1[-1], weight=w2, bias=b2, dropout=do_rate, batch_norm=False)
                layers1.append(output1)

        return cross_entropy, softmax, layers1, layers2, train_data_node1, train_data_node2, \
               top_node1, top_node2, bottom_node1, bottom_node2, train_labels_node

class model_execute:
    def __init__(self, dataset, irs_dataset):
        self.irs_dataset = irs_dataset
        self.dataset = dataset
        self.input_shape = (-1, 240, 240, 155, 4)
        self.input_pm_shape = (-1, 240, 240, 155, 9)
        self.epochs = 40
        self.eval_freq = 128
        self.init_lr = 0.003
        self.final_lr = 0.0000001

        self.patch_size = 20
        self.kernel_size = 3
        self.filters = [64, 128, 256]
        self.mod_cnt = 4
        self.lbl_cnt = 2
        self.batch_size = 128

        self.hdd_output_path = "./output_test/"
        self.ssd_output_path = "./output_test/"

    def extract_feature_Map(self, feature, path, name):
        if not os.path.exists(path):
            os.mkdir(path)
        for f in range(feature.shape[-1]):
            plt.matshow(feature[0, :, :, f], fignum=1)
            plt.axis('off')
            fname = name + '%d.png' % f
            # path = './FeatureMap/ResModule/F1/'
            plt.savefig(path + fname)

    def train_Gradual_Multi_scale_CNN(self, cand_num, cross_entropy, softmax, data_node1, data_node2, top1, top2, bottom1, bottom2, label_node):
        logthis("Gradual Multi scale CNN training started!")
        if cand_num == 1:
            path = "./output_test/LUNA_V1/%s"
            lbl = np.memmap(filename=path % "/btm_train.lbl", dtype=np.uint8, mode="r")
            train_size = lbl.shape[0]
            data_shape = (6, 20, 20, train_size)
            data_reshape = (train_size, 20, 20, 6)
            data_btm = np.memmap(filename=path % "/btm_train.dat", dtype=np.float32, mode="r", shape=data_shape)
            data_btm = np.swapaxes(data_btm, 0, -1)
            data_mid = np.memmap(filename=path % "/mid_train_reshape.dat", dtype=np.float32, mode="r",
                                 shape=data_reshape)
            data_top = np.memmap(filename=path % "/top_train_reshape.dat", dtype=np.float32, mode="r",
                                 shape=data_reshape)

        elif cand_num == 2:
            path = "./output_test/LUNA_V2/%s"
            lbl = np.memmap(filename=path % "/btm_train.lbl", dtype=np.uint8, mode="r")
            train_size = lbl.shape[0]
            data_shape = (6, 20, 20, train_size)
            data_reshape = (train_size, 20, 20, 6)
            data_btm = np.memmap(filename=path % "/btm_train.dat", dtype=np.float32, mode="r", shape=data_shape)
            data_btm = np.swapaxes(data_btm, 0, -1)
            data_mid = np.memmap(filename=path % "/mid_train_reshape.dat", dtype=np.float32, mode="r",
                                 shape=data_reshape)
            data_top = np.memmap(filename=path % "/top_train_reshape.dat", dtype=np.float32, mode="r",
                                 shape=data_reshape)

        batch = tf.Variable(0, dtype=tf.float32)  # LR*D^EPOCH=FLR --> LR/FLR
        learning_rate = tf.train.exponential_decay(learning_rate=self.init_lr, global_step=batch * self.batch_size,
                                                       decay_steps=train_size, staircase=True,
                                                       decay_rate=np.power(self.final_lr / self.init_lr,
                                                                           np.float(1) / self.epochs))

        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=batch)
        predict = tf.to_double(100) * (
            tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(softmax, label_node, 1))))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            summary_path = self.hdd_output_path + "summary_%s/%d" % (int(time.time()))
            md_path = 'Multi_scale_Model'
            model_path = self.hdd_output_path + "model_%s/" % (md_path)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            tf.global_variables_initializer().run()
            print("Variable Initialized")
            tf.summary.scalar("error", predict)

            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=8, max_to_keep=100)
            start_time = time.time()
            cur_epoch = 0
            rand_idx = np.random.permutation(train_size)
            for step in range(int(self.epochs * train_size) // self.batch_size):
                offset = (step * self.batch_size) % (train_size - self.batch_size)
                # input node 'TOP_DOWN':
                input1_batch_data = data_top[rand_idx[offset:offset + self.batch_size]]
                top1_batch_data = data_btm[rand_idx[offset:offset + self.batch_size]]
                top2_batch_data = data_mid[rand_idx[offset:offset + self.batch_size]]
                # input node 'BOTTOM_UP':
                input2_batch_data = data_btm[rand_idx[offset:offset + self.batch_size]]
                bottom1_batch_data = data_mid[rand_idx[offset:offset + self.batch_size]]
                bottom2_batch_data = data_top[rand_idx[offset:offset + self.batch_size]]
                batch_labels = lbl[rand_idx[offset:offset + self.batch_size]]

                feed_dict = {data_node1: input1_batch_data, data_node2: input2_batch_data,
                             top1: top1_batch_data, top2: top2_batch_data,
                             bottom1: bottom1_batch_data, bottom2: bottom2_batch_data,
                             label_node: batch_labels}

                _, summary_out = sess.run([optimizer, summary_op], feed_dict=feed_dict)

                summary_writer.add_summary(summary_out, global_step=step * self.batch_size)
                if step % self.eval_freq == 0:
                    l, lr, predictions = sess.run(
                        [cross_entropy, learning_rate, predict],
                        feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.2f s' % (
                        step, float(step) * self.batch_size / train_size, elapsed_time))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % predictions)
                if cur_epoch != int((step * self.batch_size) / train_size):
                    print("Saved in path", saver.save(sess, model_path + "%d.ckpt" % (cur_epoch)))
                    rand_idx = np.random.permutation(train_size)
                cur_epoch = int((step * self.batch_size) / train_size)
                # if cur_epoch == 5:
                #     break
            print("Saved in path", saver.save(sess, model_path + "savedmodel_final.ckpt"))

        tf.reset_default_graph()

    def test_bck_Complex_RM_CNN(self, cand_num, softmax, data_node1, data_node2, top1, top2, bottom1, bottom2, md_num):
        logthis("Complex ConCat Module CNN testing started!")
        if cand_num == 1:
            data_path = "./output_test/btm_pm/%s"

        elif cand_num == 2:
            data_path = "./output_test/btm_pm_V2/%s"

        model_path = self.hdd_output_path + "model_" + "CPX_V2_ELEMENTWISESUM/%s.ckpt" % (md_num)
        # model_path = self.hdd_output_path + "model_" + "ComplexMod_V2_CONV/savedmodel_final.ckpt"
        pm_path = "./output_test/" + "CPX_ZI132_ZO321_Model.dat"
        lbl = np.memmap(filename=data_path % "btm_totalTotal.lbl", dtype=np.uint8, mode="r")
        test_shape = lbl.shape[0]
        data_shape = (6, 20, 20, test_shape)
        data_reshape = (lbl.shape[0], 1, 20, 20, 6)
        data_btm = np.memmap(filename=data_path % "btm_totalTotal.dat", dtype=np.float32, mode="r", shape=data_shape)
        data_btm = np.swapaxes(data_btm, 0, -1)[:, None]
        data_mid = np.memmap(filename=data_path % "mid_totalTotal_reshape.dat", dtype=np.float32, mode="r", shape=data_reshape)
        data_top = np.memmap(filename=data_path % "top_totalTotal_reshape.dat", dtype=np.float32, mode="r", shape=data_reshape)

        all_result = np.memmap(filename=pm_path, dtype=np.float32, mode="w+", shape=(test_shape, 2))
        with tf.Session() as sess: #config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            # tf.global_variables_initializer().run()
            # input1_tensor = sess.graph.get_tensor_by_name('Relu_19:0')

            # input2_tensor = sess.graph.get_tensor_by_name('Relu_15:0')
            # input3_tensor = sess.graph.get_tensor_by_name('Relu_16:0')
            #
            # input4_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
            # input5_tensor = sess.graph.get_tensor_by_name('Relu_3:0')
            # input6_tensor = sess.graph.get_tensor_by_name('Relu_5:0')
            # integral_tensor1 = sess.graph.get_tensor_by_name('Relu_16:0')
            # integral_tensor2 = sess.graph.get_tensor_by_name('Relu_17:0')
            # input1_tensor = sess.graph.get_tensor_by_name('Add:0')
            # input2_tensor = sess.graph.get_tensor_by_name('sub_11:0')
            # input3_tensor = sess.graph.get_tensor_by_name('sub_13:0')
            #
            # input4_tensor = sess.graph.get_tensor_by_name('sub_1:0')
            # input5_tensor = sess.graph.get_tensor_by_name('sub_3:0')
            # input6_tensor = sess.graph.get_tensor_by_name('sub_5:0')


            # # num = 27226
            # for num in range(test_shape):
            #
            #     all_result[num] = input1_tensor.eval(
            #             feed_dict={data_node1: data_top[num, :, :, :, :], data_node2: data_btm[num, :, :, :, :],
            #                         top1: data_mid[num, :, :, :, :], bottom1: data_mid[num, :, :, :, :],
            #                         top2: data_btm[num, :, :, :, :], bottom2: data_top[num, :, :, :, :]})
            # self.extract_feature_Map(feature1, path='./FeatureMap/Integral/ND_SUM/GT_%d_TN/' % num, name='nd_feature_out_')
            #
            # feature2 = input2_tensor.eval(
            #             feed_dict={data_node1: data_top[num, :, :, :, :], data_node2: data_btm[num, :, :, :, :],
            #                              top1: data_mid[num, :, :, :, :], bottom1: data_mid[num, :, :, :, :],
            #                              top2: data_btm[num, :, :, :, :], bottom2: data_top[num, :, :, :, :]})
            # self.extract_feature_Map(feature2, path='./FeatureMap/Integral/ND_in/GT_%d_TN/' % num, name='nd_feature_in_')
            #
            # feature3 = input3_tensor.eval(
            #     feed_dict={data_node1: data_top[num, :, :, :, :], data_node2: data_btm[num, :, :, :, :],
            #                              top1: data_mid[num, :, :, :, :], bottom1: data_mid[num, :, :, :, :],
            #                              top2: data_btm[num, :, :, :, :], bottom2: data_top[num, :, :, :, :]})
            # self.extract_feature_Map(feature3, path='./FeatureMap/Integral/ND_int/GT_%d_TN/' % num, name='nd_feature_int_')
            #
            # feature4 = input4_tensor.eval(feed_dict={data_node2: data_btm[num, :, :, :, :]})
            # self.extract_feature_Map(feature4, path='./FeatureMap/Bottomup/ND_F1/GT_%d_TN/' % num, name='nd_feature4_')
            #
            # feature5 = input5_tensor.eval(
            #     feed_dict={data_node2: data_btm[num, :, :, :, :], bottom1: data_mid[num, :, :, :, :]})
            # self.extract_feature_Map(feature5, path='./FeatureMap/Bottomup/ND_F2/GT_%d_TN/' % num, name='nd_feature5_')
            #
            # feature6 = input6_tensor.eval(
            #     feed_dict={data_node2: data_btm[num, :, :, :, :], bottom1: data_mid[num, :, :, :, :],
            #                bottom2: data_top[num, :, :, :, :]})
            # self.extract_feature_Map(feature6, path='./FeatureMap/Bottomup/ND_F3/GT_%d_TN/' % num, name='nd_feature_')

            #feature7 = integral_tensor1.eval(
            #   feed_dict={data_node1: data_top[num, :, :, :, :], data_node2: data_btm[num, :, :, :, :],
            #               top1: data_mid[num, :, :, :, :], bottom1: data_mid[num, :, :, :, :],
            #               top2: data_btm[num, :, :, :, :], bottom2: data_top[num, :, :, :, :]})
            #self.extract_feature_Map(feature7, path='./FeatureMap/Integral/ND_F4/GT_%d_TN/' % num, name='nd_feature_')

            #feature8 = integral_tensor2.eval(
            #    feed_dict={data_node1: data_top[num, :, :, :, :], data_node2: data_btm[num, :, :, :, :],
            #               top1: data_mid[num, :, :, :, :], bottom1: data_mid[num, :, :, :, :],
            #               top2: data_btm[num, :, :, :, :], bottom2: data_top[num, :, :, :, :]})
            #self.extract_feature_Map(feature8, path='./FeatureMap/Integral/ND_F5/GT_%d_TN/' % num, name='nd_feature_')
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            for i in range(test_shape):
                all_result[i] = sess.run(softmax, feed_dict={data_node1: data_top[i,:,:,:,:],
                                                             top1: data_mid[i, :, :, :, :],
                                                             top2: data_btm[i, :, :, :, :],
                                                             data_node2: data_btm[i,:,:,:,:],
                                                             bottom1: data_mid[i,:,:,:,:],
                                                             bottom2: data_top[i,:,:,:,:]})
                # cax1 = plt.matshow(data_top[13, :, :, 2053], fignum=1, cmap='gray')
                # plt.colorbar(cax1, ticks=[0, 1, .2, .4, .6, .8])
                # plt.clim(0, 1)
                if i % 1000 == 0:
                    print(i, data_btm.shape[0])
        return pm_path

    def code_test(self, model_num, model_name):

        if model_num == 0:
            pm_path = model_name
            data_path = "./output_test/btm/btm_totalTotal.%s"
        elif model_num == 1:
            pm_path = model_name
            data_path = "./output_test/mid/mid_totalTotal.%s"
        elif model_num == 2:
            pm_path = model_name
            data_path = "./output_test/top/top_totalTotal.%s"
        elif model_num == 3:
            pm_path = model_name
            data_path = "./output_test/btm_pm/btm_totalTotal.%s"
        elif model_num == 4:
            pm_path = model_name
            data_path = "./output_test/btm_pm_V2/btm_totalTotal.%s"
        else:
            pm_path = model_name
            data_path = "./output_test/btm/btm_totalTotal.%s"

        lbl = np.memmap(filename=data_path % "lbl", dtype=np.uint8, mode="r")
        all_result = np.memmap(filename=pm_path, dtype=np.float32, mode="r", shape=(lbl.shape[0], 2))
        all_result = np.argmax(all_result, axis=-1).astype(np.uint8)
        print(np.bincount(lbl))
        from sklearn.metrics import confusion_matrix
        aa = confusion_matrix(lbl, all_result)
        print("TP", aa[1, 1], aa[1, 1] * 100. / (aa[1,1] + aa[1,0]))
        print("FN", aa[1, 0], aa[1, 0] * 100. / (aa[1,1] + aa[1,0]))
        print("TN", aa[0, 0], aa[0, 0] * 100. / (aa[0,1] + aa[0,0]))
        print("FP", aa[0, 1], aa[0, 1] * 100. / (aa[0,1] + aa[0,0]))

        print("ACC", (aa[1, 1] + aa[0,0]) * 100. / np.sum(aa))
        print("SEN", aa[1, 1]  * 100. / (aa[1,1] + aa[1,0]))
        print("SPC", aa[0, 0] * 100. / (aa[0,0] + aa[0,1]))

def logthis(a):
    print("\n" + str(datetime.now()) + ": " + str(a))

def resize3D(data, resize, model, pm_flag):
    resizing = np.memmap(filename=model, dtype=np.float32, mode="w+", shape=resize)

    if pm_flag != 2:
        if len(data.shape) == 4:
            zSize = 6 / data.shape[3]
            ySize = 20 / data.shape[2]
            xSize = 20 / data.shape[1]

            for i, d in enumerate(data):
                resizing[i, :, :, :] = scipy.ndimage.interpolation.zoom(d, zoom=(ySize, xSize, zSize))
        else:
            zSize = 6 / data.shape[4]
            ySize = 20 / data.shape[3]
            xSize = 20 / data.shape[2]
            for i, d in enumerate(data):
                resizing[i, 0, :, :, :] = scipy.ndimage.interpolation.zoom(d, zoom=(1, ySize, xSize, zSize))
    else:
        if len(data.shape) == 4:
            zSize = 26 / data.shape[3]
            ySize = 40 / data.shape[2]
            xSize = 40 / data.shape[1]

            for i, d in enumerate(data):
                resizing[i, :, :, :] = scipy.ndimage.interpolation.zoom(d, zoom=(ySize, xSize, zSize))
        else:
            zSize = 26 / data.shape[4]
            ySize = 40 / data.shape[3]
            xSize = 40 / data.shape[2]
            for i, d in enumerate(data):
                resizing[i, 0, :, :, :] = scipy.ndimage.interpolation.zoom(d, zoom=(1, ySize, xSize, zSize))
    return resizing

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines[1:]

def mk_CSVwriter(lists, model_num, model_name):
    # model = ['btm', 'mid', 'top']

    if model_num == 0:
        pm_path = model_name
    elif model_num == 1:
        pm_path = model_name
    elif model_num == 2:
        pm_path = model_name
    elif model_num == 3:
        pm_path = model_name
    elif model_num == 4:
        pm_path = model_name

    data = np.memmap(filename=pm_path, dtype=np.float32, mode="r", shape=(len(lists), 2))

    output_dir = "evaluationScript/Submission/fold_Multiscale_In"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = 'Submission_' + pm_path[14:-4]
    CSV_filename = output_dir + '/%s.csv' % (filename)

    with open(CSV_filename, 'w', newline='') as csvfile:
        CSV_writer = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        CSV_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
        for list, d in zip(lists, data[:,1]):
            CSV_writer.writerow([list[0], float(list[1]), float(list[2]), float(list[3]), float(d)])
            print(list[0], float(list[1]), float(list[2]), float(list[3]), float(d))
    return output_dir, filename

def mk_FROC_results(output_dir, filename):
    annotations_filename = "evaluationScript/annotations/annotations.csv"
    annotations_excluded_filename = "evaluationScript/annotations/annotations_excluded.csv"
    seriesuids_filename = "evaluationScript/annotations/seriesuids.csv"
    results_filename = output_dir + '/%s.csv' % (filename)

    noduleCADEvaluation(annotations_filename, annotations_excluded_filename,
                        seriesuids_filename, results_filename, output_dir)

def CPM_results(output_dir, filename):
    CSV_results_filename = output_dir + '/froc_%s_bootstrapping.csv' % (filename)
    bootstrap = readCSV(CSV_results_filename)
    cpm0 = float(bootstrap[1][1])
    cpm1 = float(bootstrap[160][1])
    cpm2 = float(bootstrap[478][1])
    cpm3 = float(bootstrap[1112][1])
    cpm4 = float(bootstrap[2382][1])
    cpm5 = float(bootstrap[4922][1])
    cpm6 = float(bootstrap[9999][1])

    cpm = (cpm0 + cpm1 + cpm2 + cpm3 + cpm4 + cpm5 + cpm6) / 7.0

    print('0.125: %f, 0.25: %f, 0.5: %f, 1: %f, 2: %f, 4: %f, 8: %f, CPM: %f \n'
          % (cpm0, cpm1, cpm2, cpm3, cpm4, cpm5, cpm6, cpm))