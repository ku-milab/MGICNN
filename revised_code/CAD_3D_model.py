from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

from datetime import datetime
import matplotlib.pyplot as plt


def init_weight_bias(name, shape, filtercnt, trainable):

    if name[0] == 'c':
        weights = tf.get_variable(name=name + "w", shape=shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    else:
        weights = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32, trainable=trainable)
    biases = tf.Variable(initial_value=tf.constant(0, shape=[filtercnt], dtype=tf.float32), name=name + "b",
                         trainable=trainable)
    return weights, biases


def conv3d_layer(data, weight, bias, padding, is_inception):
    conv = tf.nn.conv3d(input=data, filter=weight, strides=[1, 1, 1, 1, 1], padding=padding)
    # conv = tf.layers.conv3d(inputs=data, filters=bias, kernel_size=weight, strides=(1, 1, 1),
    #                         padding=padding, data_format='channels_last',
    #                         activation= ,
    #                         kernel_initializer=tf.truncated_normal(shape=weight, mean=0.0, stddev=0.01,
    #                                                                dtype=tf.float32, seed=None, name=name + "w")
    #                         )
    # data_mean, data_var = tf.nn.moments(x=)

    # conv_norm = tf.nn.batch_normalization(x=conv, mean=data_mean, variance=)
    if is_inception:
        return tf.nn.bias_add(conv, bias)
    return tf.nn.bias_add(conv, bias)
    # return conv


def relu_layer(conv):
    return tf.nn.relu(conv)


def pool3d_layer(data, kernel, stride):
    return tf.nn.max_pool3d(input=data, ksize=kernel, strides=stride, padding="VALID")


def fc_layer(data, weight, bias, dropout, batch_norm):
    shape = data.get_shape().as_list()
    shape = [shape[0], np.prod(shape[1:])]
    hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
    if batch_norm:
        hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = tf.nn.relu(hidden)
    if dropout < 1.:
        hidden = tf.nn.dropout(hidden, dropout)
    return hidden


def output_layer(data, weight, bias, label):
    shape = data.get_shape().as_list()
    shape = [shape[0], np.prod(shape[1:])]
    hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
    if label is None:
        return None, tf.nn.softmax(hidden, dim=-1)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=hidden)),\
                                                                                    tf.nn.softmax(hidden, dim=-1)


class model_def:
    def __init__(self):
        # self.model_num = [1, 2, 3]
        self.patch_size = [20, 30, 40]
        self.z_size = [6, 10, 26]
        self.mod_cnt = 1
        self.lbl_cnt = 2
        self.CV_kernel_size = [[5, 5, 1], [5, 5, 3]]
        self.MP_kernel_size = [[1, 1, 1, 1, 1], [1, 1, 2, 2, 1], [1, 2, 2, 2, 1]]
        self.MP_stride_size = [[1, 1, 1, 1, 1], [1, 1, 2, 2, 1], [1, 2, 2, 2, 1]]
        self.filters = [64, 150, 250]

        self.batch_size = 200
        self.do_rate = 0.2


    def btm_CNN(self, train=True):
        """
        Archi-1 3D CNN
          layer: [5,5,3@64] [1,1,1@64] [5,5,3@64] [5,5,1@64] fc
          layer: [3,5,5@64] [1,1,1@64] [3,5,5,3@64] [1,5,5@64] fc

        """
        if train:
            batch_size = self.batch_size
            do_rate = self.do_rate
            train_data_node = tf.placeholder(tf.float32, shape=(
                batch_size, self.z_size[0], self.patch_size[0], self.patch_size[0], self.mod_cnt))
            train_labels_node = tf.placeholder(tf.int64, shape=batch_size)

        else:
            batch_size = 1
            do_rate = 1.
            train_data_node = tf.placeholder(tf.float32, shape=(
                batch_size, self.z_size[0], self.patch_size[0], self.patch_size[0], self.mod_cnt))
            train_labels_node = None



        layers = [train_data_node]

        cross_entropy, softmax = None, None
        w1, b1 = init_weight_bias(name="c%d" % (0), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.mod_cnt, self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_1 = conv3d_layer(data=layers[-1], weight=w1, bias=b1, padding="VALID", is_inception=False)
        # conv3d_1 = conv3d_layer(data=layers[-1], weight=conv3d_1_shape, bias=conv3d_1_shape[-1],
        #                         padding="VALID", name="c%d" % (0))
        layers.append(conv3d_1)
        relu3d_1 = relu_layer(conv3d_1)
        layers.append(relu3d_1)
        pool3d = pool3d_layer(data=layers[-1], kernel=self.MP_kernel_size[0], stride=self.MP_stride_size[0])
        layers.append(pool3d)

        w2, b2 = init_weight_bias(name="c%d" % (1), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.filters[0], self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_2 = conv3d_layer(data=layers[-1], weight=w2, bias=b2, padding="VALID", is_inception=False)
        layers.append(conv3d_2)
        relu3d_2 = relu_layer(conv3d_2)
        layers.append(relu3d_2)

        w3, b3 = init_weight_bias(name="c%d" % (2), shape=[self.CV_kernel_size[0][2], self.CV_kernel_size[0][0],
                                                           self.CV_kernel_size[0][1], self.filters[0], self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_3 = conv3d_layer(data=layers[-1], weight=w3, bias=b3, padding="VALID", is_inception=False)
        layers.append(conv3d_3)
        relu3d_3 = relu_layer(conv3d_3)
        layers.append(relu3d_3)

        fw1, fb1 = init_weight_bias(name="f%d" % (0), shape=[8 * 8 * 2 * 64, self.filters[1]],
                                    filtercnt=self.filters[1], trainable=train)
        fc_1 = fc_layer(data=layers[-1], weight=fw1, bias=fb1, dropout=do_rate, batch_norm=False)
        layers.append(fc_1)
        fw2, fb2 = init_weight_bias(name="f%d" % (1), shape=[self.filters[1], self.lbl_cnt],
                                    filtercnt=self.lbl_cnt, trainable=train)
        cross_entropy, softmax = output_layer(data=layers[-1], weight=fw2, bias=fb2, label=train_labels_node)

        # cw = [w1, w2, w3]
        # cb = [b1, b2, b3]
        # fw = [fw1, fw2]
        # fb = [fb1, fb2]

        # for kernel, layer_cnt in zip(conv3d_layer_shape, range(len(conv3d_layer_shape))):
        #     w, b = init_weight_bias(name="c%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1], trainable=train)
        #     cw.append(w)
        #     cb.append(b)
        # for kernel, layer_cnt in zip(fc_layer_shape, range(len(fc_layer_shape))):
        #     w, b = init_weight_bias(name="f%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1], trainable=train)
        #     fw.append(w)
        #     fb.append(b)

        # for w, b, layer_cnt in zip(cw, cb, range(len(cw))):
        #     output = conv3d_layer(data=layers[-1], weight=w, bias=b, padding="VALID", is_inception=False)
        #     layers.append(output)
        #     if layer_cnt == 0:
        #         output = pool3d_layer(data=layers[-1], kernel=self.MP_kernel_size[0], stride=self.MP_stride_size[0])
        #         layers.append(output)
        # for w, b, layer_cnt in zip(fw, fb, range(len(fw))):
        #     if layer_cnt == 1:
        #         cross_entropy, softmax = output_layer(data=layers[-1], weight=w, bias=b, label=train_labels_node)
        #     else:
        #         output = fc_layer(data=layers[-1], weight=w, bias=b, dropout=do_rate, batch_norm=False)
        #         layers.append(output)
        return cross_entropy, softmax, layers, train_data_node, train_labels_node


    def mid_CNN(self, fine_Flag, train=True):
        """
         Archi-2 3D CNN
          layer: [5,5,3@64] [2,2,1@64] [5,5,3@64] [5,5,3@64] fc
          layer: [3,5,5@64] [1,2,2@64] [3,5,5@64] [3,5,5@64] fc

        """
        if train:
            do_rate = self.do_rate
            batch_size = self.batch_size
            train_labels_node = tf.placeholder(tf.int64, shape=batch_size)
        else:
            do_rate = 1.
            batch_size = 1
            train_labels_node = None

        train_data_node = tf.placeholder(tf.float32, shape=(batch_size, self.patch_size[1], self.patch_size[1], self.z_size[1], self.mod_cnt))

        layers = [train_data_node]

        cross_entropy, softmax = None, None
        w1, b1 = init_weight_bias(name="c%d" % (0), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.mod_cnt, self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_1 = conv3d_layer(data=layers[-1], weight=w1, bias=b1, padding="VALID", is_inception=False)
        layers.append(conv3d_1)
        relu3d_1 = relu_layer(conv3d_1)
        layers.append(relu3d_1)
        pool3d = pool3d_layer(data=layers[-1], kernel=self.MP_kernel_size[0], stride=self.MP_stride_size[0])
        layers.append(pool3d)

        w2, b2 = init_weight_bias(name="c%d" % (1), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.filters[0], self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_2 = conv3d_layer(data=layers[-1], weight=w2, bias=b2, padding="VALID", is_inception=False)
        layers.append(conv3d_2)
        relu3d_2 = relu_layer(conv3d_2)
        layers.append(relu3d_2)

        w3, b3 = init_weight_bias(name="c%d" % (2), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.filters[0], self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_3 = conv3d_layer(data=layers[-1], weight=w3, bias=b3, padding="VALID", is_inception=False)
        layers.append(conv3d_3)
        relu3d_3 = relu_layer(conv3d_3)
        layers.append(relu3d_3)

        fw1, fb1 = init_weight_bias(name="f%d" % (0), shape=[5 * 5 * 4 * 64, self.filters[2]],
                                    filtercnt=self.filters[1], trainable=train)
        fc_1 = fc_layer(data=layers[-1], weight=fw1, bias=fb1, dropout=do_rate, batch_norm=False)
        layers.append(fc_1)
        fw2, fb2 = init_weight_bias(name="f%d" % (1), shape=[self.filters[2], self.lbl_cnt],
                                    filtercnt=self.lbl_cnt, trainable=train)
        cross_entropy, softmax = output_layer(data=layers[-1], weight=fw2, bias=fb2, label=train_labels_node)

        return cross_entropy, softmax, layers, train_data_node, train_labels_node


    def top_CNN(self, train=True):
        """
         Archi-3 3D CNN
         layer: [5,5,3@64] [2,2,2@64] [5,5,3@64] [5,5,3@64] fc
         layer: [3,5,5@64] [2,2,2@64] [3,5,5@64] [3,5,5@64] fc

        """
        if train:
            do_rate = self.do_rate
            batch_size = self.batch_size
            train_labels_node = tf.placeholder(tf.int64, shape=batch_size)
        else:
            do_rate = 1.
            batch_size = 1
            train_labels_node = None

        train_data_node = tf.placeholder(tf.float32, shape=(batch_size, self.z_size[2], self.patch_size[2], self.patch_size[2], self.mod_cnt))

        layers = [train_data_node]

        cross_entropy, softmax = None, None
        w1, b1 = init_weight_bias(name="c%d" % (0), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.mod_cnt, self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_1 = conv3d_layer(data=layers[-1], weight=w1, bias=b1, padding="VALID", is_inception=False)
        layers.append(conv3d_1)
        relu3d_1 = relu_layer(conv3d_1)
        layers.append(relu3d_1)
        pool3d = pool3d_layer(data=layers[-1], kernel=self.MP_kernel_size[2], stride=self.MP_stride_size[2])
        layers.append(pool3d)

        w2, b2 = init_weight_bias(name="c%d" % (1), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.filters[0], self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_2 = conv3d_layer(data=layers[-1], weight=w2, bias=b2, padding="VALID", is_inception=False)
        layers.append(conv3d_2)
        relu3d_2 = relu_layer(conv3d_2)
        layers.append(relu3d_2)

        w3, b3 = init_weight_bias(name="c%d" % (2), shape=[self.CV_kernel_size[1][2], self.CV_kernel_size[1][0],
                                                           self.CV_kernel_size[1][1], self.filters[0], self.filters[0]],
                                  filtercnt=self.filters[0], trainable=train)
        conv3d_3 = conv3d_layer(data=layers[-1], weight=w3, bias=b3, padding="VALID", is_inception=False)
        layers.append(conv3d_3)
        relu3d_3 = relu_layer(conv3d_3)
        layers.append(relu3d_3)

        fw1, fb1 = init_weight_bias(name="f%d" % (0), shape=[10 * 10 * 8 * 64, self.filters[2]],
                                    filtercnt=self.filters[2], trainable=train)
        fc_1 = fc_layer(data=layers[-1], weight=fw1, bias=fb1, dropout=do_rate, batch_norm=False)
        layers.append(fc_1)
        fw2, fb2 = init_weight_bias(name="f%d" % (1), shape=[self.filters[2], self.lbl_cnt],
                                    filtercnt=self.lbl_cnt, trainable=train)
        cross_entropy, softmax = output_layer(data=layers[-1], weight=fw2, bias=fb2, label=train_labels_node)

        return cross_entropy, softmax, layers, train_data_node, train_labels_node


class model_execute:
    def __init__(self, train_dataset, test_dataset, dataset, originset):
        self.train_dataset = train_dataset
        # self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.origin_dataset = originset
        self.lists_data = dataset

        self.data_name = ['btm', 'mid', 'top']

        self.epochs = 5
        self.eval_freq = 30
        self.init_lr = 0.000001
        self.pre_epochs = 30
        self.pre_init_lr = 0.3
        self.final_lr = 0.000001

        self.patch_size = [20, 30, 40]
        self.z_size = [6, 10, 26]
        self.mod_cnt = 1
        self.lbl_cnt = 2
        self.batch_size = 200

        self.hdd_output_path = "output_test/"
        self.data_path = "pre_process_data"
        self.origin_path = "origin_ext_voxel"


    def mk_patch_origial_CNN(self, dataset, lists_data, dataset_name, data_path, model_num):
        def find(target, obj):
            for i, lst in enumerate(obj):
                for j, name in enumerate(lst):
                    if name == target:
                        return i
            return (None, None)

        def count_data_size(target, lists):
            size = 0
            for i, name in enumerate(target):
                for list in lists:
                    if list[0] == name[5:-4]:
                        size += int(list[1])
                        print(list[1], size)
            return size

        if model_num == 0:
            output_path = self.hdd_output_path + 'btm/'
        elif model_num == 1:
            output_path = self.hdd_output_path + 'mid/'
        elif model_num == 2:
            output_path = self.hdd_output_path + 'top/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dataset_size = count_data_size(dataset, lists_data)
        # print(dataset_size)

        if dataset_name == 'total':
            if model_num == 0:
                data_name = 'btm'
                fn = output_path + data_name + '_' + dataset_name
            elif model_num == 1:
                data_name = 'mid'
                fn = output_path + data_name + '_' + dataset_name
            elif model_num == 2:
                data_name = 'top'
                fn = output_path + data_name + '_' + dataset_name

            pData = np.memmap(filename=fn + "Total.dat", dtype='float32', mode='w+',
                              shape=(self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num],
                                     dataset_size))
            plabel = np.memmap(filename=fn + "Total.lbl", dtype='uint8', mode='w+', shape=(1, 1, dataset_size))

            zEnd = 0
            for c, CTNum in enumerate(dataset):
                save_path = "%s/%s/%s.dat" % (self.origin_path, data_name, CTNum[5:-4])
                lbl_path = "%s/%s/%s.lbl" % (self.origin_path, data_name, CTNum[5:-4])
                data_idx = find(CTNum[5:-4], lists_data)

                print(lists_data[data_idx][1])
                label = np.memmap(filename=lbl_path, dtype='uint8', mode='r',
                                  shape=(1, 1, int(lists_data[data_idx][1])))

                data = np.memmap(filename=save_path, dtype='float32', mode='r',
                                 shape=(self.patch_size[model_num], self.patch_size[model_num], self.z_size[model_num],
                                        int(lists_data[data_idx][1])))

                print(label.shape, sum(label))
                if c == 0:
                    zStart = 0
                    zEnd = (zStart + int(lists_data[data_idx][1]))
                else:
                    zStart = zEnd
                    zEnd = (zStart + int(lists_data[data_idx][1]))

                print(zStart, int(lists_data[data_idx][1]), zEnd, data.shape)
                pData[:, :, :, zStart:zEnd] = data.copy()
                plabel[:, :, zStart:zEnd] = label.copy()

                del data, label
                print(CTNum)

            del pData, plabel

        else:
            if model_num == 0:
                fn = output_path + 'btm' + '_' + dataset_name
            elif model_num == 1:
                fn = output_path + 'mid' + '_' + dataset_name
            elif model_num == 2:
                fn = output_path + 'top' + '_' + dataset_name

            pData = np.memmap(filename=fn + "Total.dat", dtype='float32', mode='w+',
                    shape=(self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num], dataset_size))
            plabel = np.memmap(filename=fn +"Total.lbl", dtype='uint8', mode='w+', shape=(1, 1, dataset_size))

            zEnd = 0
            for c, CTNum in enumerate(dataset):
                save_path = "%s/%s.dat" % (data_path, CTNum[5:-4])
                lbl_path = "%s/%s.lbl" % (data_path, CTNum[5:-4])
                data_idx = find(CTNum[5:-4], lists_data)
                print(int(lists_data[data_idx][1]))
                data = np.memmap(filename=save_path, dtype='float32', mode='r',
                                 shape=(self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num],
                                        int(lists_data[data_idx][1])))
                label = np.memmap(filename=lbl_path, dtype='uint8', mode='r')
                print(label.shape, sum(label))

                if c == 0:
                    zStart = 0
                    zEnd = (zStart + int(lists_data[data_idx][1]))
                else:
                    zStart = zEnd
                    zEnd = (zStart + int(lists_data[data_idx][1]))

                print(zStart, int(lists_data[data_idx][1]), zEnd, data.shape)
                pData[:, :, :, zStart:zEnd] = data.copy()
                plabel[:, :, zStart:zEnd] = label.copy()

                del data, label
                print(CTNum)

            del pData, plabel

    # def pre_train_CNN(self, cross_entropy, softmax, data_node, label_node, model_num):
    #     logthis("Pre CNN training started!")
    #
    #     if model_num == 0:
    #         data_path = self.hdd_output_path + "/btm" + "/btm_trainTotal"
    #         val_path = self.hdd_output_path + "/btm" + "/btm_valTotal"
    #     elif model_num == 1:
    #         data_path = self.hdd_output_path + "/mid" + "/mid_trainTotal"
    #         val_path = self.hdd_output_path + "/mid"+ "/mid_valTotal"
    #     elif model_num == 2:
    #         data_path = self.hdd_output_path + "/top" + "/top_trainTotal"
    #         val_path = self.hdd_output_path + "/top" + "/top_valTotal"
    #
    #     if not os.path.exists("%s_reshape.dat" % (data_path)):
    #         lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
    #         train_size = lbl.shape[0]
    #         print(lbl.shape, sum(lbl))
    #         rand_idx = np.random.permutation(train_size)
    #         lbl = lbl[rand_idx]
    #         data_shape = (self.patch_size[model_num], self.patch_size[model_num],
    #                       self.z_size[model_num], train_size)
    #         data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
    #         # data = data[:, :, :, rand_idx]
    #         data_reshape = (train_size, self.patch_size[model_num], self.patch_size[model_num],
    #                       self.z_size[model_num])
    #         reshape_data = np.memmap(filename=data_path + '_reshape' + ".dat", dtype=np.float32,
    #                                  mode="w+", shape=data_reshape)
    #         lbl_reshape = (train_size)
    #         reshape_lbl = np.memmap(filename=data_path + '_reshape' + ".lbl", dtype=np.uint8,
    #                                  mode="w+", shape=lbl_reshape)
    #         data = data[:, :, :, rand_idx]
    #         data = np.transpose(data, axes=(3, 0, 1, 2))
    #         # data = data.append([])
    #         reshape_data[:, :, :, :] = data.copy()
    #         reshape_lbl[:] = lbl.copy()
    #         # plt.matshow(data[:,:,3,1], fignum=1, cmap=plt.cm.gray)
    #         # plt.show()
    #         del data, reshape_data, lbl, reshape_lbl
    #
    #     lbl = np.memmap(filename=data_path + '_reshape' + ".lbl", dtype=np.uint8, mode="r")
    #     val_lbl = np.memmap(filename=val_path + ".lbl", dtype=np.uint8, mode="r")
    #     train_size = lbl.shape[0]
    #     val_size = val_lbl.shape[0]
    #     print(lbl.shape, val_size, sum(lbl))
    #     data_reshape = (train_size, self.patch_size[model_num], self.patch_size[model_num],
    #                     self.z_size[model_num], self.mod_cnt)
    #     val_shape = (val_size, self.patch_size[model_num], self.patch_size[model_num],
    #                     self.z_size[model_num], self.mod_cnt)
    #     data = np.memmap(filename=data_path + '_reshape' + ".dat", dtype=np.float32,
    #                                  mode="r", shape=data_reshape)
    #     val = np.memmap(filename=val_path + ".dat", dtype=np.float32, mode="r", shape=val_shape)
    #
    #     batch = tf.Variable(0, dtype=tf.float32)  # LR*D^EPOCH=FLR --> LR/FLR
    #     learning_rate = tf.train.exponential_decay(learning_rate=self.init_lr, global_step=batch,
    #                                                decay_steps=25,
    #                                                decay_rate=0.95, staircase=True)
    #     # print(learning_rate)
    #     optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=batch)
    #     predict = tf.to_double(100) * (
    #         tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(softmax, label_node, 1))))
    #
    #     with tf.Session() as sess:
    #         summary_path = self.hdd_output_path + "summary_%s_pre_cnn/%d" % (self.data_name[model_num], int(time.time()))
    #         # summary_val_path = self.hdd_output_path + \
    #         #                    "summary_%s_pre_cnn/%d" % (self.data_name[model_num], int(time.time()))
    #         model_path = self.hdd_output_path + "model_%s_pre_cnn/" % (self.data_name[model_num])
    #         if not os.path.exists(model_path):
    #             os.makedirs(model_path)
    #         tf.global_variables_initializer().run()
    #         print("Variable Initialized")
    #         tf.summary.scalar("error", predict)
    #         summary_op = tf.summary.merge_all()
    #         summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    #         # summary_val_writer = tf.summary.FileWriter(summary_val_path, sess.graph)
    #         saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=30)
    #         start_time = time.time()
    #
    #         # batch size
    #         cur_epoch = 0
    #         for step in range(int(self.pre_epochs * train_size) // self.batch_size):
    #             offset = (step * self.batch_size) % (train_size - self.batch_size)
    #             val_offset = (step * self.batch_size) % (val_size - self.batch_size)
    #             # offset = step % self.batch_size
    #             batch_data = data[offset:offset + self.batch_size]
    #             batch_labels = lbl[offset:offset + self.batch_size]
    #             # batch_val = val[val_offset:val_offset + self.batch_size]
    #             # batch_val_labels = val_lbl[val_offset:val_offset + self.batch_size]
    #             feed_dict = {data_node: batch_data, label_node: batch_labels}
    #
    #             _, l, lr, predictions, summary_out = sess.run(
    #                 [optimizer, cross_entropy, learning_rate, predict, summary_op],
    #                 feed_dict=feed_dict)
    #             #
    #             # feed_dict_val = {data_node: batch_val, label_node:batch_val_labels}
    #             # predictions_val = sess.run([predict], feed_dict=feed_dict_val)
    #
    #             summary_writer.add_summary(summary_out, global_step=step * self.batch_size)
    #             if step % self.eval_freq == 0:
    #                 elapsed_time = time.time() - start_time
    #                 start_time = time.time()
    #                 print('Step %d (pre_epoch %.2f), %d s' % (step,
    #                                                           float(step) * self.batch_size / train_size, elapsed_time))
    #                 print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
    #                 # print('Minibatch error: %.3f Val error: %.3f' % (predictions, predictions_val[0]))
    #                 # print('Minibatch error: %.1f Var error: %.1f' % (predictions, predictions_val))
    #
    #             if np.floor(cur_epoch) != np.floor((step * self.batch_size) / train_size):
    #                 print(cur_epoch)
    #                 print((step * self.batch_size) / train_size)
    #                 # print(cur_epoch==(step * self.batch_size) / train_size)
    #                 print("Saved in path", saver.save(sess, model_path + "%d.ckpt" % (cur_epoch)))
    #
    #                 # randnum = np.random.randint(0, cut_size)
    #                 # curdata = data[randnum:train_size + randnum - cut_size]
    #                 # curlbl = lbl[randnum:train_size + randnum - cut_size]
    #             cur_epoch = (step * self.batch_size) / train_size
    #
    #         print("Saved in path", saver.save(sess, model_path + "savedmodel_final.ckpt"))
    #     tf.reset_default_graph()


    # def fine_tune_CNN(self, cross_entropy, softmax, data_node, label_node, model_num):
    #     def find(target, obj):
    #         lists = []
    #         for i, lst in enumerate(obj):
    #             # for j, name in enumerate(lst):
    #             if lst == target:
    #                 lists.append(i)
    #
    #         return lists
    #
    #     logthis("Fine Tune CNN training started!")
    #
    #     if model_num == 0:
    #         data_path = self.hdd_output_path + "/btm/" + "btm_fineTotal"
    #         val_path = self.hdd_output_path + "/btm/" + "btm_valTotal"
    #
    #     elif model_num == 1:
    #         data_path = self.hdd_output_path + "/mid/" + "mid_fineTotal"
    #         val_path = self.hdd_output_path + "/mid/" + "mid_valTotal"
    #
    #     elif model_num == 2:
    #         data_path = self.hdd_output_path + "/top/" + "top_fineTotal"
    #         val_path = self.hdd_output_path + "/top/" + "top_valTotal"
    #
    #     if not os.path.exists("%s_reshape.dat" % (data_path)):
    #         lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
    #         train_size = lbl.shape[0]
    #
    #         true_idx = find(1, lbl)
    #         false_idx = find(0, lbl)
    #
    #         tune_false_idx = false_idx[:len(true_idx)]
    #
    #         true_idx.extend(tune_false_idx)
    #
    #         rand_idx = np.random.permutation(len(true_idx))
    #         data_shape = (self.patch_size[model_num], self.patch_size[model_num],
    #                       self.z_size[model_num], train_size, self.mod_cnt)
    #
    #         data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
    #
    #         tune_lbl = lbl[true_idx]
    #         tune_data = data[:, :, :, true_idx, :]
    #         del data, lbl
    #
    #         lbl = tune_lbl[rand_idx]
    #         data = tune_data[:, :, :, rand_idx, :]
    #         data = np.transpose(data, axes=(3, 0, 1, 2, 4))
    #
    #         # data_reshape = (self.patch_size[model_num], self.patch_size[model_num],
    #         #               self.z_size[model_num], len(true_idx), self.mod_cnt)
    #         # reshape_data = np.memmap(filename=data_path + '_reshape' + ".dat", dtype=np.float32,
    #         #                          mode="w+", shape=data_reshape)
    #         # reshape_lbl = np.memmap(filename=data_path + '_reshape' + ".dat", dtype=np.float32,
    #         #                          mode="w+", shape=len(true_idx))
    #         #
    #         # reshape_data[:, :, :, :, :] = tune_data.copy()
    #         # reshape_lbl[:] = tune_lbl.copy()
    #
    #         del tune_data, tune_lbl
    #     else:
    #         lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
    #
    #         true_idx = find(1, lbl)
    #         false_idx = find(0, lbl)
    #         tune_false_idx = false_idx[:len(true_idx)]
    #
    #         true_idx.extend(tune_false_idx)
    #         data_reshape = (self.patch_size[model_num], self.patch_size[model_num],
    #                         self.z_size[model_num], len(true_idx), self.mod_cnt)
    #         data = np.memmap(filename=data_path + "_reshape.dat", dtype=np.float32, mode="r", shape=data_reshape)
    #         lbl = np.memmap(filename=data_path + "_reshape.lbl", dtype=np.uint8, mode="r")
    #
    #     val_lbl = np.memmap(filename=val_path + ".lbl", dtype=np.uint8, mode="r")
    #
    #     val_size = val_lbl.shape[0]
    #     val_shape = (self.patch_size[model_num], self.patch_size[model_num],
    #                       self.z_size[model_num], val_size, self.mod_cnt)
    #
    #     val = np.memmap(filename=val_path + ".dat", dtype=np.float32, mode="r", shape=val_shape)
    #     train_size = data.shape[0]
    #     batch = tf.Variable(0, dtype=tf.float32)  # LR*D^EPOCH=FLR --> LR/FLR
    #     learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=batch * self.batch_size,
    #                                                decay_steps=25, staircase=True,
    #                                                decay_rate=0.95)
    #     val = np.transpose(val, axes=(3, 0, 1, 2, 4))
    #     # print(learning_rate.value, batch.value)
    #     optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)
    #     predict = tf.to_double(100) * (
    #         tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(softmax, label_node, 1))))
    #
    #     with tf.Session() as sess:
    #         summary_path = self.hdd_output_path + "summary_%s_cnn/%d" % (self.data_name[model_num], int(time.time()))
    #
    #         fine_model_path = self.hdd_output_path + "model_%s_fine_cnn/" % (self.data_name[model_num])
    #         model_path = self.hdd_output_path + "model_%s_pre_cnn/savedmodel_final.ckpt" % (self.data_name[model_num])
    #         if not os.path.exists(fine_model_path):
    #             os.makedirs(fine_model_path)
    #             print("Model is not exist !!!")
    #
    #         tf.global_variables_initializer().run()
    #         print("Variable Initialized")
    #
    #         tf.summary.scalar("error", predict)
    #         summary_op = tf.summary.merge_all()
    #         summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    #         # conv2_w = tf.get_tensor_by_name('c2w')
    #         saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=30)
    #         saver.restore(sess, model_path)
    #
    #         start_time = time.time()
    #
    #         # batch size
    #         cur_epoch = 0
    #         for step in range(int(self.epochs * train_size) // self.batch_size):
    #             offset = (step * self.batch_size) % (train_size - self.batch_size)
    #             val_offset = (step * self.batch_size) % (val_size - self.batch_size)
    #             print(offset, val_offset)
    #             batch_data = data[offset:offset + self.batch_size]
    #             batch_labels = lbl[offset:offset + self.batch_size]
    #             batch_val = val[val_offset:val_offset + self.batch_size]
    #             batch_val_labels = val_lbl[val_offset:val_offset + self.batch_size]
    #
    #             feed_dict = {data_node: batch_data, label_node: batch_labels}
    #
    #             _, l, lr, predictions, summary_out = sess.run(
    #                 [optimizer, cross_entropy, learning_rate, predict, summary_op],
    #                 feed_dict=feed_dict)
    #             feed_dict_val = {data_node: batch_val, label_node: batch_val_labels}
    #             predictions_val = sess.run([predict], feed_dict=feed_dict_val)
    #
    #             summary_writer.add_summary(summary_out, global_step=step * self.batch_size)
    #             if step % self.eval_freq == 0:
    #                 elapsed_time = time.time() - start_time
    #                 start_time = time.time()
    #                 print(
    #                     'Step %d (epoch %.2f), %d s' % (step, float(step) * self.batch_size / train_size, elapsed_time))
    #                 print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
    #                 print('Minibatch error: %.2f Val error: %.2f' % (predictions, predictions_val[0]))
    #
    #             if np.floor(cur_epoch) != np.floor((step * self.batch_size) / train_size):
    #                 print(cur_epoch)
    #                 print((step * self.batch_size) / train_size)
    #                 # print(cur_epoch==(step * self.batch_size) / train_size)
    #                 print("Saved in path", saver.save(sess, fine_model_path + "%d.ckpt" % (cur_epoch)))
    #
    #                 # randnum = np.random.randint(0, cut_size)
    #                 # curdata = data[randnum:train_size + randnum - cut_size]
    #                 # curlbl = lbl[randnum:train_size + randnum - cut_size]
    #             cur_epoch = (step * self.batch_size) / train_size
    #
    #         print("Saved in path", saver.save(sess, fine_model_path + "savedmodel_final.ckpt"))
    #     tf.reset_default_graph()

    def train_original_CNN(self, cross_entropy, softmax, data_node, label_node, model_num):
        logthis("Original CNN training started!")

        if model_num == 0:
            data_path = self.hdd_output_path + "btm/" + "btm_trainTotal"
        elif model_num == 1:
            data_path = self.hdd_output_path + "mid/" + "mid_trainTotal"
        elif model_num == 2:
            data_path = self.hdd_output_path + "top/" + "top_trainTotal"

        if not os.path.exists("%s_reshape.dat" % (data_path)):
            lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
            train_size = lbl.shape[0]
            rand_idx = np.random.permutation(train_size)
            data_shape = (self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num], train_size)
            data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
            data_reshape = (train_size, self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num])
            reshape_data = np.memmap(filename=data_path + '_reshape' + ".dat", dtype=np.float32,
                                     mode="w+", shape=data_reshape)
            reshape_lbl = np.memmap(filename=data_path + '_reshape' + ".lbl", dtype=np.uint8,
                                    mode="w+", shape=(train_size))
            idx_size = len(rand_idx)
            print(idx_size, int(idx_size / 3), int(idx_size // 3))
            if self.data_name[model_num] == 'top':

                idx1 = rand_idx[:int(idx_size // 3)]
                reshape = (len(idx1), self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num])
                data1 = np.memmap(filename=data_path + '_reshape_temp' + ".dat", dtype=np.float32,
                                         mode="w+", shape=reshape)
                data1 = data[:, :, :, idx1].copy()
                data1 = np.transpose(data1, axes=(3, 0, 1, 2))
                # del data1

                idx2 = rand_idx[int(idx_size // 3):-int(idx_size // 3)]
                reshape = (len(idx2), self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num])
                data2 = np.memmap(filename=data_path + '_reshape_temp' + ".dat", dtype=np.float32,
                                         mode="w+", shape=reshape)
                data2 = data[:, :, :, idx2].copy()
                data2 = np.transpose(data2, axes=(3, 0, 1, 2))
                # del data2

                idx3 = rand_idx[int((idx_size*2) // 3):]
                reshape = (len(idx3), self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num])
                data3 = np.memmap(filename=data_path + '_reshape_temp' + ".dat", dtype=np.float32,
                                         mode="w+", shape=reshape)
                data3 = data[:, :, :, idx3].copy()
                data3 = np.transpose(data3, axes=(3, 0, 1, 2))
                # del data3

                reshape_data[:int(idx_size // 3), :, :, :] = data1.copy()
                reshape_data[int(idx_size // 3):-int(idx_size // 3), :, :, :] = data2.copy()
                reshape_data[int((idx_size*2) // 3):, :, :, :] = data3.copy()
                reshape_lbl = lbl.copy()
            else:
                lbl = lbl[rand_idx]
                reshape_lbl[:] = lbl.copy()
                data = data[:, :, :, rand_idx]
                data = np.transpose(data, axes=(3, 0, 1, 2))
                reshape_data[:, :, :, :] = data.copy()
            del data, reshape_data, lbl, reshape_lbl
            
        if self.data_name[model_num] == 'top':
            lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
            train_size = lbl.shape[0]
            # rand_idx = np.random.permutation(train_size)
            # lbl = lbl[rand_idx]
            data_shape = (self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num],
                        train_size, self.mod_cnt)
            data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
            data = np.transpose(data, axes=(3, 0, 1, 2, 4))
            print(data.shape)
        else:
            lbl = np.memmap(filename=data_path + '_reshape' + ".lbl", dtype=np.uint8, mode="r")
            train_size = lbl.shape[0]
            data_reshape = (train_size, self.z_size[model_num],self.patch_size[model_num], self.patch_size[model_num],
                             self.mod_cnt)
            data = np.memmap(filename=data_path + '_reshape' + ".dat", dtype=np.float32,
                                         mode="r", shape=data_reshape)

        # lbl = one_hot(lbl)

        batch = tf.Variable(0, dtype=tf.float32)  # LR*D^EPOCH=FLR --> LR/FLR
        learning_rate = tf.train.exponential_decay(learning_rate=self.init_lr, global_step=batch,
                                                   decay_steps=25,
                                                   decay_rate=0.95, staircase=True)
        # print(learning_rate.value, batch.value)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=batch)
        predict = tf.to_double(100) * (
            tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(softmax, label_node, 1))))

        with tf.Session() as sess:
            summary_path = self.hdd_output_path + "summary_%s_cnn/%d" % (self.data_name[model_num], int(time.time()))
            model_path = self.hdd_output_path  + "model_%s_cnn/" % (self.data_name[model_num])
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            tf.global_variables_initializer().run()
            print("Variable Initialized")
            tf.summary.scalar("error", predict)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=8, max_to_keep=30)
            start_time = time.time()

            # batch size
            cur_epoch = 0

            for step in range(int(self.epochs * train_size) // self.batch_size):
                offset = (step * self.batch_size) % (train_size - self.batch_size)
                print(int(offset / self.batch_size))
                batch_data = data[offset:offset + self.batch_size]
                batch_labels = lbl[offset:offset + self.batch_size]
                feed_dict = {data_node: batch_data, label_node: batch_labels}

                _, l, lr, predictions, summary_out = sess.run(
                    [optimizer, cross_entropy, learning_rate, predict, summary_op],
                    feed_dict=feed_dict)
                summary_writer.add_summary(summary_out, global_step=step * self.batch_size)
                if step % self.eval_freq == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print(
                        'Step %d (epoch %.2f), %d s' % (step, float(step) * self.batch_size / train_size, elapsed_time))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % predictions)

                if np.floor(cur_epoch) != np.floor((step * self.batch_size) / train_size):
                    print(cur_epoch)
                    print((step * self.batch_size) / train_size)
                    # print(cur_epoch==(step * self.batch_size) / train_size)
                    print("Saved in path", saver.save(sess, model_path + "%d.ckpt" % (cur_epoch)))
                cur_epoch = (step * self.batch_size) / train_size

            print("Saved in path", saver.save(sess, model_path + "savedmodel_final.ckpt"))
        tf.reset_default_graph()


    def test_original_CNN(self, softmax, data_node, model_epoch, model_num):
        logthis("Original CNN testing started!")
        if model_num == 0:
            data_path = self.hdd_output_path + "btm/" + "btm_testTotal"
            # output_path = self.hdd_output_path + "/btm"
        elif model_num == 1:
            data_path = self.hdd_output_path + "mid_totalTotal"
        elif model_num == 2:
            data_path = self.hdd_output_path + "top_totalTotal"
        # model_path = self.hdd_output_path + "model_%s_cnn/%s.ckpt" % (self.data_name[model_num], model_epoch)
        # if not os.path.exists(self.hdd_output_path + "model_%s_fine_cnn/%s.ckpt" % (self.data_name[model_num], model_epoch)):

        if not os.path.exists(self.hdd_output_path + "model_%s_cnn/%s.ckpt" % (self.data_name[model_num], model_epoch)):
            # model_path = self.hdd_output_path + "model_%s_fine_cnn/savedmodel_final.ckpt" % (self.data_name[model_num])
            model_path = self.hdd_output_path + "model_%s_cnn/savedmodel_final.ckpt" % (self.data_name[model_num])

        else:
            model_path = self.hdd_output_path + "model_%s_cnn/%s.ckpt" % (self.data_name[model_num], model_epoch)
        pm_path = self.hdd_output_path + "%s_pm_cnn.dat" % (self.data_name[model_num])
        img_path = self.hdd_output_path + "%s" % (self.data_name[model_num])

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        # load total data and make fusion probability
        lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
        print(lbl.shape, sum(lbl))
        data_shape = (lbl.shape[0], self.z_size[model_num], self.patch_size[model_num], self.patch_size[model_num],
                      self.mod_cnt)
        # size = lbl.shape[0]
        data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)

        result_shape = (lbl.shape[0], self.lbl_cnt)
        all_result = np.memmap(filename=pm_path, dtype=np.float32, mode="w+", shape=result_shape)
        # lbl = one_hot(lbl)

        # plt.interactive(False)
        # plt.figure()
        # plt.matshow(data[1, :, :, 5, 0], fignum=1)
        # plt.show()

        # make total dataset and test of each test dataset.
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

            for curdata, curdata_cnt in zip(data, range(data.shape[0])):
                result = np.zeros(shape=(data.shape[0], self.lbl_cnt), dtype=np.float32)
                # local_time = time.time()
                curdata = np.array(curdata)
                curdata.resize(self.mod_cnt, self.z_size[model_num], self.patch_size[model_num],
                                self.patch_size[model_num], self.mod_cnt)
                feed_dict = {data_node: curdata}
                result[curdata_cnt, :] = sess.run(softmax, feed_dict=feed_dict)
                print("Original CNN testing", curdata_cnt + 1, "/", result[curdata_cnt, :], end="\n")
                all_result[curdata_cnt, :] = result[curdata_cnt, :]

        tf.reset_default_graph()


def code_test(self):
        # data_path = self.hdd_output_path + "%s_orig.dat" % ("t")
        # data = np.memmap(filename=data_path, dtype=np.float32, mode="r").reshape(self.input_shape)

        #

        data_path = self.hdd_output_path + "%s_%s_irs.dat" % (self.dataset, self.irs_dataset)
        lbl_path = self.hdd_output_path + "%s_orig.lbl" % (self.dataset)
        data = np.memmap(filename=data_path, dtype=np.float32, mode="r").reshape(self.input_shape)
        label = np.memmap(filename=lbl_path, dtype=np.uint8, mode="r", shape=data.shape[:-1])

        # data_path = self.hdd_output_path + "%s_%s_patch_orig_cnn2" % (self.dataset, self.irs_dataset)
        # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
        # data_shape = (lbl.shape[0], self.patch_size, self.patch_size, self.mod_cnt)
        # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
        lbl_idx_cls = np.load(self.hdd_output_path + "%s_lblidx.npz.npy" % (self.dataset))
        lbl_idx = np.empty(shape=(0, 4), dtype=np.uint16)
        for cur_cnt, curlbl in enumerate(lbl_idx_cls):
            templbl = np.sort(curlbl.view("u2,u2,u2,u2"), order=["f0", "f3", "f1", "f2"], axis=0).view(np.uint16)
            print(cur_cnt)
            if cur_cnt == 0:
                continue
            else:
                lbl_idx = np.append(lbl_idx, templbl[::2], axis=0)
        lbl_idx = np.sort(lbl_idx.view("u2,u2,u2,u2"), order=["f0", "f1", "f2", "f3"], axis=0).view(np.uint16)

        ims = []
        fig = plt.figure(figsize=(24, 12))
        for a in range(10):
            templbl = lbl_idx[lbl_idx[:, 0] == a]
            for b in range(155):
                templbl2 = templbl[templbl[:, -1] == b]
                ax = plt.subplot(1, 1, 1)
                c = ax.imshow(data[a, :, :, b, 0], cmap="gray", animated=True)
                d = ax.scatter(templbl2[:, 2], templbl2[:, 1], facecolor="red", s=33 ** 2, marker="s",
                               c=label[templbl2[:, 0], templbl2[:, 1], templbl2[:, 2], templbl2[:, 3]])
                ims.append([c, d])

                #ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
                #plt.show()
                #plt.close()
                #
                # import matplotlib.pyplot as plt
                # print(lbl[:128])
                # for a in range(128):
                #     ax = plt.subplot(8,16,a+1)
                #     ax.imshow(data[a,...,0], cmap="gray")
                #     ax.axis("off")
                # plt.show()

                # pm_path = self.hdd_output_path + "%s_%s_pm_orig_cnn.dat" % ("t", self.irs_dataset)
                # result_shape = (self.input_shape[0],self.input_shape[1],self.input_shape[2],self.input_shape[3],self.mod_cnt+self.lbl_cnt)
                # all_result = np.memmap(filename=pm_path, dtype=np.float32, mode="r").reshape(result_shape)
                # import matplotlib.pyplot as plt
                # for a in range(9):
                #     if a<4:
                #         ax = plt.subplot(2,4,a+1)
                #         ax.imshow(all_result[0,:,:,70,a], cmap="gray")
                #     else:
                #         ax = plt.subplot(2, 5, a + 2)
                #         ax.imshow(all_result[0,:,:,70,a], vmin=0, vmax=1)
                # plt.show()

                # data_path = self.hdd_output_path + "%s_%s_patch_orig_cnn" % (self.dataset, self.irs_dataset)
                # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
                # data_shape = (lbl.shape[0], self.patch_size, self.patch_size, self.mod_cnt)
                # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
                #
                # import matplotlib.pyplot as plt
                # import matplotlib.animation as animation
                #
                # for i in range(0, data_shape[0],128):
                #     ims = []
                #     fig = plt.figure(figsize=(24,12))
                #     for j in range(4):
                #         im = []
                #         for curidx, curdata in enumerate(data[i:i+128]):
                #             ax = plt.subplot(8,16,curidx+1)
                #             a = ax.imshow(curdata[...,j], cmap="gray")
                #             ax.axis("off")
                #             im.append(a)
                #         ims.append(im)
                #
                #     ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
                #     plt.show()
                #     plt.close()




                # data_path = self.hdd_output_path + "%s_orig" % (self.dataset)
                # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r").reshape(self.input_shape)
                # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r", shape=data.shape[:-1])
                #
                # lbl_idx_cls = np.load(self.hdd_output_path + "%s_lblidx.npz.npy" % (self.dataset))
                # lbl_idx = np.empty(shape=(0,4), dtype=np.uint16)
                # for curlbl in lbl_idx_cls:
                #     templbl = np.sort(curlbl.view("u2,u2,u2,u2"), order=["f3", "f2", "f1", "f0"], axis=0).view(np.uint16)
                #     lbl_idx = np.append(lbl_idx, templbl[::2], axis=0)
                # # lbl_idx = np.sort(lbl_idx.view("u2,u2,u2,u2"), order=["f3", "f2", "f1", "f0"], axis=0).view(np.uint16)
                # lbl_idx = np.sort(lbl_idx.view("u2,u2,u2,u2"), order=["f0", "f1", "f2", "f3"], axis=0).view(np.uint16)
                #
                #
                # import matplotlib.pyplot as plt
                # import matplotlib.animation as animation
                #
                # ims = []
                # fig = plt.figure(figsize=(24, 12))
                # for curcnt in range(0,274,5):
                #     cur_idx = lbl_idx[lbl_idx[:,0]==curcnt]
                #     for curheight in range(155):
                #         cur_idx_height = cur_idx[cur_idx[:,-1]==curheight][:,1:3]
                #         ax1 = plt.subplot(2, 3, 1)
                #         ax2 = plt.subplot(2, 3, 2)
                #         ax3 = plt.subplot(2, 3, 3)
                #         ax4 = plt.subplot(2, 3, 4)
                #         ax5 = plt.subplot(2, 3, 5)
                #
                #         a=ax1.imshow(data[curcnt,...,curheight,0], cmap="gray", animated=True)
                #         b=ax2.imshow(data[curcnt,...,curheight,1], cmap="gray", animated=True)
                #         c=ax3.imshow(data[curcnt,...,curheight,2], cmap="gray", animated=True)
                #         d=ax4.imshow(data[curcnt,...,curheight,3], cmap="gray", animated=True)
                #         e = ax5.imshow(lbl[curcnt,...,curheight], vmin=0, vmax=4)
                #         a1 = ax1.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
                #         a2 = ax2.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
                #         a3 = ax3.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
                #         a4 = ax4.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
                #         a5 = ax5.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
                #         ims.append([a,b,c,d,e,a1,a2,a3,a4,a5])
                #     print(curcnt)
                #
                # ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
                # plt.show()
                # plt.close()


                # data_path2 = self.hdd_output_path + "%s_orig" % (self.dataset)
                # data2 = np.memmap(filename=data_path2 + ".dat", dtype=np.float32, mode="r").reshape(self.input_shape)
                #
                # allmean, allvar = np.load(self.hdd_output_path + "%s_%s_mv.npy" % (self.dataset, self.irs_dataset))
                # print(allmean, allvar)
                # for dat in data2:
                #     print(np.mean((dat-allmean)/allvar,axis=(0,1,2)),np.var((dat-allmean)/allvar,axis=(0,1,2)))

                # data_path = self.hdd_output_path + "%s_%s_patch_orig_cnn" % (self.dataset, self.irs_dataset)
                # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
                # data_shape = (lbl.shape[0], self.patch_size, self.patch_size, self.mod_cnt)
                # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
                #
                # allmean, allvar = np.load(self.hdd_output_path + "%s_%s_mv.npy" % (self.dataset, self.irs_dataset))
                # cnt = 0
                # print(np.mean(data,axis=(0,1,2)))
                # print(np.var(data[:data.shape[0]//2], axis=(0,1,2)))
                # for dat, curcnt in zip(data, range(data.shape[0])):
                #     for mod in range(self.mod_cnt):
                #         print(curcnt, mod, np.mean(dat[...,mod]), np.var(dat[...,mod]))
                # import matplotlib.pyplot as plt
                #
                # import matplotlib.animation as animation
                # # ims = []
                # # fig = plt.figure(figsize=(24,12))
                # print(data.shape)
                # for dat in data:
                #     fig = plt.figure(figsize=(24,12))
                #     ims=[]
                #     for curdepth in range(155):
                #         ax1 = plt.subplot(1,4,1)
                #         ax2 = plt.subplot(1,4,2)
                #         ax3 = plt.subplot(1,4,3)
                #         ax4 = plt.subplot(1,4,4)
                #         a =ax1.imshow(dat[...,curdepth, 0], cmap="gray", animated=True)
                #         b= ax2.imshow(dat[...,curdepth, 1], cmap="gray", animated=True)
                #         c= ax3.imshow(dat[...,curdepth, 2], cmap="gray", animated=True)
                #         d= ax4.imshow(dat[...,curdepth, 3], cmap="gray", animated=True)
                #         ims.append([a,b,c,d])
                #
                #     ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
                #     plt.show()
                #     plt.close()

                # currently 13240 patches are blank. WHY????
                # 274 images, max idx is 255
                # lbl_idx = np.load(self.hdd_output_path + "%s_lblidx2.npy" % (self.dataset))
                # lbl_idx = np.sort(lbl_idx.view("u1,u1,u1,u1"), order=["f3", "f2", "f1", "f0"], axis=0).view(np.uint8)
                # for a in np.argmax(lbl_idx, axis=0):
                #     print(lbl_idx[a])


def rolling_window_lastaxis(a, window):
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a, window):
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a


def get_filename(dataset):
    # h.86.VSD.Brain.XX.O.MR_T1.35779.nii
    name_list = os.listdir("../../../../data/")
    result = {}
    for name in name_list:
        temp = name.replace(".mhd", "").split(".")
        if temp[0] != dataset or temp[-2] != "MR_Flair":
            continue
        result[int(temp[1]) - 1] = int(temp[-1])
    return result


def logthis(a):
    print("\n" + str(datetime.now()) + ": " + str(a))


def one_hot(lists):
    mk_lbl = np.zeros(shape=(lists.shape[0], 2),dtype=int)

    for l, list in enumerate(lists):
        if l == 0:
            mk_lbl[l, :] = [0, 1]
        else:
            mk_lbl[l, :] = [1, 0]

    return mk_lbl